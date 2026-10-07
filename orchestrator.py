import os
import time
import uuid
import cv2
import numpy as np
from PIL import Image

from wrappers.trufor_wrapper import TruForPredictor
from wrappers.aasist_wrapper import AASISTPredictor
from wrappers.syncnet_wrapper import SyncNetPredictor
from fusion.sync_calibrator import SyncEvidenceCalibrator
from fusion.evidence_fusion import EvidenceFusionEngine
from fusion.reasoner import ForensicEvidenceReasoner

class CalibratorAdapter:
    def __init__(self, real_calibrator):
        self.real_calibrator = real_calibrator
        self.confidence_threshold = real_calibrator.reliable_confidence_threshold
        
    def calibrate(self, offset, conf):
        return self.real_calibrator.calibrate(offset, conf).get("sync_desync_score")

class InferenceOrchestrator:
    """
    End-to-end inference orchestrator for Multimodal Deepfake & Digital Forensics.
    Coordinates modality routing, artifact generation, timing, and evidence fusion
    without reloading model checkpoints.
    """
    def __init__(self):
        # 1. Initialize models ONCE.
        self.trufor = TruForPredictor(device="cpu")
        self.aasist = AASISTPredictor(device="cpu")
        self.syncnet = SyncNetPredictor(device="cpu")
        
        # 2. Initialize fusion logic
        self.sync_calibrator = SyncEvidenceCalibrator()
        self.fusion_engine = EvidenceFusionEngine(sync_calibrator=CalibratorAdapter(self.sync_calibrator))
        self.reasoner = ForensicEvidenceReasoner()

    def _determine_media_type(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            return 'image'
        elif ext in ['.wav', '.mp3', '.flac', '.m4a']:
            return 'audio'
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return 'video'
        else:
            raise ValueError(f"Unsupported media file extension: {ext}")

    def _analyze_video_temporal(self, video_path: str, base_artifact_dir: str) -> dict:
        """
        Hierarchical temporal analysis for videos using coarse and fine sampling.
        Returns aggregated video metrics and timeline data.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 25.0
        duration = total_frames / fps if fps > 0 else 0
        
        # Determine coarse timestamps
        coarse_timestamps = []
        if duration <= 1.0 or total_frames < int(fps):
            # Very short video, sample 3 frames or middle frame
            if total_frames > 2:
                coarse_timestamps = [0.0, duration / 2.0, duration * 0.9]
            elif total_frames > 0:
                coarse_timestamps = [duration / 2.0]
        else:
            # 1 FPS
            coarse_timestamps = [float(i) for i in range(int(np.ceil(duration)))]
        
        # Suspicious threshold heuristic
        SUSPICIOUS_THRESHOLD = 0.5
        
        analyzed_frames = {} # timestamp -> evidence dict
        
        def process_frame_at_time(t_sec):
            if t_sec in analyzed_frames:
                return analyzed_frames[t_sec]
            
            frame_idx = min(int(t_sec * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
                
            temp_frame_path = f"temp_frame_{uuid.uuid4().hex}.png"
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                vis_res = self.trufor.predict(temp_frame_path)
            except Exception:
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
                return None
            
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
            
            v_score = float(vis_res["visual_score"])
            regions = self.reasoner.extract_regions(vis_res["anomaly_map"], vis_res["reliability_map"])
            
            evidence = {
                "timestamp_sec": float(t_sec),
                "frame_index": int(frame_idx),
                "visual_score": v_score,
                "suspicious_regions": regions,
                "artifact_paths": {},
                "_anomaly_map": vis_res["anomaly_map"],
                "_reliability_map": vis_res["reliability_map"],
                "_frame": frame
            }
            analyzed_frames[t_sec] = evidence
            return evidence
            
        # Pass 1: Coarse
        suspicious_candidates = []
        for t in coarse_timestamps:
            ev = process_frame_at_time(t)
            if ev and ev["visual_score"] >= SUSPICIOUS_THRESHOLD:
                suspicious_candidates.append(t)
                
        # Pass 2: Fine sampling
        for t in suspicious_candidates:
            fine_offsets = [-0.5, -0.25, 0.25, 0.5]
            for off in fine_offsets:
                ft = t + off
                if 0 <= ft <= duration:
                    process_frame_at_time(ft)
                    
        cap.release()
        
        if not analyzed_frames:
            raise RuntimeError("No frames could be extracted.")
            
        # Group into intervals
        sorted_times = sorted(analyzed_frames.keys())
        intervals = []
        current_interval = None
        GAP_TOLERANCE = 1.0
        
        for t in sorted_times:
            ev = analyzed_frames[t]
            if ev["visual_score"] >= SUSPICIOUS_THRESHOLD:
                if current_interval is None:
                    current_interval = [ev]
                else:
                    if t - current_interval[-1]["timestamp_sec"] <= GAP_TOLERANCE:
                        current_interval.append(ev)
                    else:
                        intervals.append(current_interval)
                        current_interval = [ev]
            else:
                if current_interval is not None:
                    intervals.append(current_interval)
                    current_interval = None
        if current_interval is not None:
            intervals.append(current_interval)
            
        suspicious_timeline = []
        for interval in intervals:
            peak_ev = max(interval, key=lambda x: x["visual_score"])
            pt = peak_ev["timestamp_sec"]
            if not peak_ev["artifact_paths"]:
                ts_str = f"{pt:.2f}".replace('.', '_')
                fpath = os.path.join(base_artifact_dir, f"evidence_frame_{ts_str}.png")
                apath = os.path.join(base_artifact_dir, f"anomaly_map_{ts_str}.png")
                rpath = os.path.join(base_artifact_dir, f"reliability_map_{ts_str}.png")
                cv2.imwrite(fpath, peak_ev["_frame"])
                self._save_heatmap(peak_ev["_anomaly_map"], apath)
                self._save_heatmap(peak_ev["_reliability_map"], rpath)
                peak_ev["artifact_paths"] = {"evidence_frame": fpath, "anomaly_map": apath, "reliability_map": rpath}
            
            clean_frames = []
            for ev in interval:
                clean_ev = {k: v for k, v in ev.items() if not k.startswith('_')}
                clean_frames.append(clean_ev)
                
            suspicious_timeline.append({
                "start_time": interval[0]["timestamp_sec"],
                "end_time": interval[-1]["timestamp_sec"],
                "peak_time": pt,
                "peak_score": peak_ev["visual_score"],
                "evidence_frames": clean_frames
            })
            
        all_ev = list(analyzed_frames.values())
        global_peak_ev = max(all_ev, key=lambda x: x["visual_score"])
        
        if not global_peak_ev["artifact_paths"]:
            pt = global_peak_ev["timestamp_sec"]
            ts_str = f"{pt:.2f}".replace('.', '_')
            fpath = os.path.join(base_artifact_dir, f"evidence_frame_{ts_str}.png")
            apath = os.path.join(base_artifact_dir, f"anomaly_map_{ts_str}.png")
            rpath = os.path.join(base_artifact_dir, f"reliability_map_{ts_str}.png")
            cv2.imwrite(fpath, global_peak_ev["_frame"])
            self._save_heatmap(global_peak_ev["_anomaly_map"], apath)
            self._save_heatmap(global_peak_ev["_reliability_map"], rpath)
            global_peak_ev["artifact_paths"] = {"evidence_frame": fpath, "anomaly_map": apath, "reliability_map": rpath}
            
        mean_score = sum(x["visual_score"] for x in all_ev) / len(all_ev)
        
        for ev in all_ev:
            for k in list(ev.keys()):
                if k.startswith('_'):
                    del ev[k]
                    
        return {
            "mean_score": mean_score,
            "peak_score": global_peak_ev["visual_score"],
            "peak_timestamp_sec": global_peak_ev["timestamp_sec"],
            "frames_analyzed": len(all_ev),
            "evidence_frames": all_ev,
            "suspicious_timeline": suspicious_timeline,
            "global_peak_frame": global_peak_ev
        }

    def _save_heatmap(self, map_arr: np.ndarray, out_path: str):
        """Converts raw NumPy 2D array [0,1] to a grayscale PNG image."""
        img_arr = (np.clip(map_arr, 0.0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode='L')
        img.save(out_path)

    def analyze(self, media_path: str) -> dict:
        """
        Main entry point for end-to-end multimodal analysis.
        """
        t_total_start = time.perf_counter()
        
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found at {media_path}")

        media_type = self._determine_media_type(media_path)
        
        # State
        v_score = None
        a_score = None
        s_offset = None
        s_conf = None
        s_fps = 25.0
        
        anomaly_map = None
        reliability_map = None
        
        evidence_log = []
        visual_evidence = {
            "anomaly_map_path": None,
            "reliability_map_path": None
        }
        
        timing = {
            "visual_time_ms": None,
            "audio_time_ms": None,
            "sync_time_ms": None,
            "total_time_ms": None
        }

        # 1. VISUAL ROUTING (Image/Video)
        video_visual = None
        suspicious_timeline = []
        suspicious_regions_override = None
        
        if media_type in ['image', 'video']:
            t_vis = time.perf_counter()
            try:
                artifact_dir = os.path.join(os.getcwd(), f"artifacts_{uuid.uuid4().hex}")
                os.makedirs(artifact_dir, exist_ok=True)
                
                if media_type == 'image':
                    vis_res = self.trufor.predict(media_path)
                    v_score = float(vis_res["visual_score"])
                    
                    anomaly_path = os.path.join(artifact_dir, "anomaly_map.png")
                    reliability_path = os.path.join(artifact_dir, "reliability_map.png")
                    
                    self._save_heatmap(vis_res["anomaly_map"], anomaly_path)
                    self._save_heatmap(vis_res["reliability_map"], reliability_path)
                    
                    visual_evidence["anomaly_map_path"] = anomaly_path
                    visual_evidence["reliability_map_path"] = reliability_path
                    
                    anomaly_map = vis_res["anomaly_map"]
                    reliability_map = vis_res["reliability_map"]
                else:
                    video_visual = self._analyze_video_temporal(media_path, artifact_dir)
                    v_score = video_visual["peak_score"]
                    global_peak = video_visual["global_peak_frame"]
                    
                    visual_evidence["anomaly_map_path"] = global_peak["artifact_paths"].get("anomaly_map")
                    visual_evidence["reliability_map_path"] = global_peak["artifact_paths"].get("reliability_map")
                    suspicious_timeline = video_visual["suspicious_timeline"]
                    suspicious_regions_override = global_peak["suspicious_regions"]
                    # Reasoner doesn't need to rebuild regions for video because we've done it per frame
                    anomaly_map = None
                    reliability_map = None
                
            except Exception as e:
                evidence_log.append("Visual analysis unavailable")
            
            timing["visual_time_ms"] = int((time.perf_counter() - t_vis) * 1000)

        # 2. AUDIO ROUTING (Audio/Video)
        if media_type in ['audio', 'video']:
            t_aud = time.perf_counter()
            try:
                aud_res = self.aasist.predict(media_path)
                a_score = float(aud_res["audio_score"])
            except Exception as e:
                evidence_log.append("Audio analysis unavailable")
            
            timing["audio_time_ms"] = int((time.perf_counter() - t_aud) * 1000)

        # 3. SYNC ROUTING (Video only)
        if media_type == 'video':
            t_sync = time.perf_counter()
            try:
                # SyncNetPredictor execution
                sync_res = self.syncnet.predict(media_path)
                s_offset = int(sync_res["offset_frames"])
                s_conf = float(sync_res["confidence"])
                s_fps = float(sync_res["fps"])
                
                # Calibration explicitly executed for verification/metrics if needed, 
                # but we will pass raw values directly to fusion to avoid logic duplication.
                # The prompt explicitly asked to pass raw results to calibrator:
                _ = self.sync_calibrator.calibrate(s_offset, s_conf, fps=s_fps)
                
            except Exception as e:
                evidence_log.append("Synchronization analysis unavailable")
                
            timing["sync_time_ms"] = int((time.perf_counter() - t_sync) * 1000)

        # 4. FUSION
        try:
            # EvidenceFusionEngine intrinsically utilizes our sync_calibrator during fuse.
            fusion_payload = self.fusion_engine.fuse(
                visual_score=v_score,
                audio_score=a_score,
                sync_offset_frames=s_offset,
                sync_confidence=s_conf,
                fps=s_fps
            )
        except Exception as e:
            # If all modalities fail, fuse throws a ValueError
            fusion_payload = {
                "manipulation_score": 0.0,
                "decision": "UNCERTAIN",
                "modalities": {
                    "visual_score": None,
                    "audio_score": None,
                    "sync_desync_score": None
                },
                "sync_evidence": {
                    "offset_frames": None,
                    "offset_ms": None,
                    "confidence": None,
                    "reliable": None
                }
            }
            evidence_log.append("All modalities failed. Cannot produce fusion.")

        # 5. ASSEMBLE OUTPUT CONTRACT
        timing["total_time_ms"] = int((time.perf_counter() - t_total_start) * 1000)
        
        final_payload = {
            "manipulation_score": fusion_payload["manipulation_score"],
            "decision": fusion_payload["decision"],
            "modalities": fusion_payload["modalities"],
            "sync_evidence": fusion_payload["sync_evidence"] or {
                "offset_frames": None,
                "offset_ms": None,
                "confidence": None,
                "reliable": None
            },
            "visual_evidence": visual_evidence,
            "evidence": evidence_log,
            "suspicious_regions": suspicious_regions_override if suspicious_regions_override is not None else [],
            "suspicious_timeline": suspicious_timeline,
            "processing": timing
        }

        if video_visual:
            del video_visual["global_peak_frame"]
            final_payload["video_visual"] = video_visual

        final_payload = self.reasoner.generate_evidence(
            payload=final_payload,
            media_type=media_type,
            anomaly_map=anomaly_map,
            reliability_map=reliability_map
        )

        return final_payload
