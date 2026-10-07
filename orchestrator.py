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

    def _extract_middle_frame(self, video_path: str) -> str:
        """
        Extracts a single representative frame (the middle frame) for visual analysis.
        NOTE: This is representative-frame analysis, NOT full-video visual scanning.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video to extract frame: {video_path}")
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_idx = max(0, frame_count // 2)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            raise RuntimeError("Failed to extract middle frame from video.")
            
        temp_frame_path = f"temp_frame_{uuid.uuid4().hex}.png"
        cv2.imwrite(temp_frame_path, frame)
        return temp_frame_path

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
        if media_type in ['image', 'video']:
            t_vis = time.perf_counter()
            temp_frame = None
            try:
                target_img = media_path
                if media_type == 'video':
                    temp_frame = self._extract_middle_frame(media_path)
                    target_img = temp_frame
                    
                vis_res = self.trufor.predict(target_img)
                v_score = float(vis_res["visual_score"])
                
                # Save artifacts
                artifact_dir = os.path.join(os.getcwd(), f"artifacts_{uuid.uuid4().hex}")
                os.makedirs(artifact_dir, exist_ok=True)
                
                anomaly_path = os.path.join(artifact_dir, "anomaly_map.png")
                reliability_path = os.path.join(artifact_dir, "reliability_map.png")
                
                self._save_heatmap(vis_res["anomaly_map"], anomaly_path)
                self._save_heatmap(vis_res["reliability_map"], reliability_path)
                
                visual_evidence["anomaly_map_path"] = anomaly_path
                visual_evidence["reliability_map_path"] = reliability_path
                
                anomaly_map = vis_res["anomaly_map"]
                reliability_map = vis_res["reliability_map"]
                
            except Exception as e:
                evidence_log.append("Visual analysis unavailable")
            finally:
                if temp_frame and os.path.exists(temp_frame):
                    os.remove(temp_frame)
            
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
            "suspicious_regions": [],
            "suspicious_timeline": [],
            "processing": timing
        }

        final_payload = self.reasoner.generate_evidence(
            payload=final_payload,
            media_type=media_type,
            anomaly_map=anomaly_map,
            reliability_map=reliability_map
        )

        return final_payload
