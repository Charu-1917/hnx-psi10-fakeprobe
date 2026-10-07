import os
import json
import time
import cv2
import numpy as np
from orchestrator import InferenceOrchestrator

def make_json_serializable(obj):
    if isinstance(obj, bool):
        return obj
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    return obj

def serialize_result(res: dict, video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps <= 0: fps = 25.0
    duration_sec = frame_count / fps if fps > 0 else 0.0

    filename = os.path.basename(video_path)
    vv = res.get("video_visual", {})
    frames = vv.get("evidence_frames", [])
    
    coarse_frames = len([f for f in frames if float(f.get("timestamp_sec", 0)).is_integer()])
    total_frames = vv.get("frames_analyzed", 0)
    fine_frames = total_frames - coarse_frames

    peak_timestamp = vv.get("peak_timestamp_sec")
    
    out_frames = []
    for f in frames:
        out_frames.append({
            "timestamp_sec": f.get("timestamp_sec"),
            "frame_index": f.get("frame_index"),
            "visual_score": f.get("visual_score"),
            "is_peak": (f.get("timestamp_sec") == peak_timestamp),
            "suspicious_regions": f.get("suspicious_regions", [])
        })

    suspicious_timeline = []
    for interval in vv.get("suspicious_timeline", []):
        observations = [x.get("timestamp_sec") for x in interval.get("evidence_frames", [])]
        start_t = interval.get("start_time")
        end_t = interval.get("end_time")
        duration = max(0.0, end_t - start_t) if start_t is not None and end_t is not None else 0.0
        suspicious_timeline.append({
            "start_sec": start_t,
            "end_sec": end_t,
            "duration_sec": duration,
            "observations": observations
        })

    artifacts = []
    artifact_dir = None
    
    if res.get("visual_evidence", {}).get("anomaly_map_path"):
        apath = res["visual_evidence"]["anomaly_map_path"]
        if apath:
            artifact_dir = os.path.dirname(apath)
            artifacts.append({
                "path": os.path.abspath(apath),
                "timestamp_sec": peak_timestamp,
                "reason": "Global peak anomaly map"
            })
            
    if res.get("visual_evidence", {}).get("reliability_map_path"):
        rpath = res["visual_evidence"]["reliability_map_path"]
        if rpath:
            artifacts.append({
                "path": os.path.abspath(rpath),
                "timestamp_sec": peak_timestamp,
                "reason": "Global peak reliability map"
            })
            
    # Interval peaks
    for ev in frames:
        if ev.get("artifact_paths"):
            for k, v in ev["artifact_paths"].items():
                if v:
                    if not artifact_dir:
                        artifact_dir = os.path.dirname(v)
                    # Don't duplicate global peak artifacts if already added
                    if not any(a["path"] == os.path.abspath(v) for a in artifacts):
                        artifacts.append({
                            "path": os.path.abspath(v),
                            "timestamp_sec": ev.get("timestamp_sec"),
                            "reason": f"Interval peak {k}"
                        })

    sync_ev = res.get("sync_evidence") or {}
    modalities = res.get("modalities", {})
    processing = res.get("processing", {})

    def _ms_to_sec(ms):
        return ms / 1000.0 if ms is not None else None

    json_payload = {
        "input": {
            "filename": filename,
            "duration_sec": duration_sec,
            "fps": fps,
            "frame_count": frame_count
        },
        "temporal": {
            "coarse_frame_count": coarse_frames,
            "fine_frame_count": fine_frames,
            "total_frames_analyzed": total_frames,
            "frames": out_frames,
            "video_visual_score": vv.get("mean_score"),
            "peak_score": vv.get("peak_score"),
            "peak_timestamp_sec": peak_timestamp,
            "suspicious_timeline": suspicious_timeline
        },
        "audio": {
            "spoof_score": modalities.get("audio_score")
        },
        "sync": {
            "offset": sync_ev.get("offset_frames"),
            "offset_ms": sync_ev.get("offset_ms"),
            "confidence": sync_ev.get("confidence"),
            "sync_desync_score": modalities.get("sync_desync_score")
        },
        "fusion": {
            "manipulation_score": res.get("manipulation_score"),
            "decision": res.get("decision")
        },
        "artifacts": artifacts,
        "timings": {
            "visual_sec": _ms_to_sec(processing.get("visual_time_ms")),
            "audio_sec": _ms_to_sec(processing.get("audio_time_ms")),
            "sync_sec": _ms_to_sec(processing.get("sync_time_ms")),
            "total_sec": _ms_to_sec(processing.get("total_time_ms"))
        }
    }

    json_payload = make_json_serializable(json_payload)
    
    if not artifact_dir:
        import uuid
        artifact_dir = f"artifacts_missing_{uuid.uuid4().hex}"
        
    os.makedirs(artifact_dir, exist_ok=True)
    result_path = os.path.join(artifact_dir, "result.json")
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
        
    print(f"RESULT_JSON: {os.path.abspath(result_path)}")
    return json_payload

def main():
    video_path = r"C:\charu hack\tests\test_video.mp4"
    orchestrator = InferenceOrchestrator()
    
    t0 = time.time()
    res = orchestrator.analyze(video_path)
    t1 = time.time()
    
    print(f"Inference complete in {t1 - t0:.2f}s")
    serialize_result(res, video_path)
    
if __name__ == '__main__':
    main()
