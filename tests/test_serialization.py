import os
import json
import numpy as np
import pytest
from run_real_video import serialize_result

@pytest.fixture
def dummy_video(tmp_path):
    video_path = str(tmp_path / "test.mp4")
    # create empty file, cv2 will just read 0 fps/frames which our code handles gracefully
    with open(video_path, "wb") as f:
        f.write(b"")
    return video_path

def test_serialize_result(dummy_video, tmp_path):
    artifact_dir = str(tmp_path / "artifacts_test")
    os.makedirs(artifact_dir, exist_ok=True)
    
    dummy_res = {
        "manipulation_score": 0.85,
        "decision": "LIKELY_MANIPULATED",
        "modalities": {
            "visual_score": 0.9,
            "audio_score": np.float32(0.1),
            "sync_desync_score": None
        },
        "sync_evidence": {
            "offset_frames": 2,
            "offset_ms": np.float64(80.0),
            "confidence": 4.5,
            "reliable": True
        },
        "visual_evidence": {
            "anomaly_map_path": os.path.join(artifact_dir, "anomaly_map_0_00.png"),
            "reliability_map_path": os.path.join(artifact_dir, "reliability_map_0_00.png")
        },
        "video_visual": {
            "mean_score": 0.5,
            "peak_score": 0.9,
            "peak_timestamp_sec": 0.0,
            "frames_analyzed": 5,
            "evidence_frames": [
                {
                    "timestamp_sec": 0.0,
                    "frame_index": 0,
                    "visual_score": np.float32(0.9),
                    "suspicious_regions": [
                        {
                            "x": 10, "y": 20, "width": 50, "height": 50, "area": 2500,
                            "mean_anomaly": 0.8, "max_anomaly": 0.95, "mean_reliability": 0.9
                        }
                    ],
                    "artifact_paths": {
                        "evidence_frame": os.path.join(artifact_dir, "evidence_frame_0_00.png")
                    }
                },
                {
                    "timestamp_sec": 1.0,
                    "frame_index": 25,
                    "visual_score": np.float32(0.4),
                    "suspicious_regions": [],
                    "artifact_paths": {}
                }
            ],
            "suspicious_timeline": [
                {
                    "start_time": 0.0,
                    "end_time": 0.5,
                    "peak_time": 0.0,
                    "peak_score": 0.9,
                    "evidence_frames": [
                        {"timestamp_sec": 0.0},
                        {"timestamp_sec": 0.5}
                    ]
                }
            ]
        },
        "processing": {
            "visual_time_ms": 12000,
            "audio_time_ms": 1500,
            "sync_time_ms": None,
            "total_time_ms": 14000
        }
    }
    
    json_payload = serialize_result(dummy_res, dummy_video)
    
    result_path = os.path.join(artifact_dir, "result.json")
    assert os.path.exists(result_path)
    
    # Reload and verify
    with open(result_path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
        
    assert reloaded["input"]["filename"] == "test.mp4"
    assert reloaded["temporal"]["frames"][0]["is_peak"] is True
    assert reloaded["temporal"]["frames"][1]["is_peak"] is False
    assert reloaded["temporal"]["frames"][0]["suspicious_regions"][0]["area"] == 2500
    assert reloaded["temporal"]["suspicious_timeline"][0]["observations"] == [0.0, 0.5]
    
    # Check artifacts
    artifacts = reloaded["artifacts"]
    assert any("anomaly map" in a["reason"].lower() for a in artifacts)
    assert any("evidence_frame" in a["reason"].lower() for a in artifacts)
    
    # Numpy float serialization
    assert isinstance(reloaded["audio"]["spoof_score"], float)
    assert isinstance(reloaded["temporal"]["frames"][0]["visual_score"], float)
    
    # Timings
    assert reloaded["timings"]["visual_sec"] == 12.0
    assert reloaded["timings"]["sync_sec"] is None
