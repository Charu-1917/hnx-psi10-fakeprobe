"""
End-to-End Validation Script for FakeProbe-X Webpage and Pipeline.
Tests all modalities, reports, errors, and checks localhost status.
"""

import sys
import json
import urllib.request
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepfake_detector_core import (
    ImageDeepfakeDetector,
    AudioDeepfakeDetector,
    VideoDeepfakeDetector,
    AdaptiveEvidenceFusionEngine,
    ForensicReportGenerator,
    DecisionVerdict,
    QualityLevel,
)

eval_dir = PROJECT_ROOT / "evaluation"

def run_tests():
    print("=" * 60)
    print("  FAKEPROBE-X END-TO-END VALIDATION SUITE")
    print("=" * 60)

    # 1. Initialize models
    print("\n--- 1. INITIALIZING FORENSIC PIPELINE ---")
    img_det = ImageDeepfakeDetector()
    aud_det = AudioDeepfakeDetector()
    vid_det = VideoDeepfakeDetector()
    fusion = AdaptiveEvidenceFusionEngine()
    print("All models loaded successfully.")

    # Test 1: Real Image
    print("\n--- [TEST 1] REAL IMAGE: real_face_01.png ---")
    res = img_det.predict_structured(str(eval_dir / "test_images" / "real_face_01.png"))
    rep1 = fusion.fuse_image_evidence(res, sample_name="real_face_01.png")
    print(f"Verdict: {rep1.final_verdict.value}")
    print(f"Confidence: {rep1.confidence*100:.1f}%")
    print(f"Reliability: {rep1.reliability*100:.1f}% ({rep1.reliability_level.value})")
    print(f"Fake Probability: {rep1.final_fake_probability*100:.2f}%")
    print(f"Quality: {rep1.quality.quality_level.value} (Score: {rep1.quality.quality_score:.2f})")

    # Test 2: Fake Image
    print("\n--- [TEST 2] FAKE IMAGE: fake_face_01.png ---")
    res = img_det.predict_structured(str(eval_dir / "test_images" / "fake_face_01.png"))
    rep2 = fusion.fuse_image_evidence(res, sample_name="fake_face_01.png")
    print(f"Verdict: {rep2.final_verdict.value}")
    print(f"Confidence: {rep2.confidence*100:.1f}%")
    print(f"Reliability: {rep2.reliability*100:.1f}% ({rep2.reliability_level.value})")
    print(f"Fake Probability: {rep2.final_fake_probability*100:.2f}%")
    print(f"Quality: {rep2.quality.quality_level.value} (Score: {rep2.quality.quality_score:.2f})")

    # Test 3: Low Quality Image
    print("\n--- [TEST 3] DEGRADED IMAGE: degraded_low_quality.png ---")
    res = img_det.predict_structured(str(eval_dir / "test_images" / "degraded_low_quality.png"))
    rep3 = fusion.fuse_image_evidence(res, sample_name="degraded_low_quality.png")
    print(f"Verdict: {rep3.final_verdict.value}")
    print(f"Confidence: {rep3.confidence*100:.1f}%")
    print(f"Reliability: {rep3.reliability*100:.1f}%")
    print(f"Quality: {rep3.quality.quality_level.value} (Score: {rep3.quality.quality_score:.2f})")
    print(f"Issues: {rep3.quality.issues}")

    # Test 4: Real Video
    print("\n--- [TEST 4] REAL VIDEO: real_video_01.mp4 ---")
    res = vid_det.predict_structured(str(eval_dir / "test_videos" / "real_video_01.mp4"), img_det, aud_det, num_frames=5)
    rep4 = fusion.fuse_video_evidence(res, sample_name="real_video_01.mp4")
    print(f"Verdict: {rep4.final_verdict.value}")
    print(f"Confidence: {rep4.confidence*100:.1f}%")
    print(f"Reliability: {rep4.reliability*100:.1f}%")
    print(f"Fake Probability: {rep4.final_fake_probability*100:.2f}%")
    print(f"Agreement Score: {rep4.agreement_score*100:.1f}%")

    # Test 5: Fake Video
    print("\n--- [TEST 5] FAKE VIDEO: fake_video_01.mp4 ---")
    res = vid_det.predict_structured(str(eval_dir / "test_videos" / "fake_video_01.mp4"), img_det, aud_det, num_frames=5)
    rep5 = fusion.fuse_video_evidence(res, sample_name="fake_video_01.mp4")
    print(f"Verdict: {rep5.final_verdict.value}")
    print(f"Confidence: {rep5.confidence*100:.1f}%")
    print(f"Reliability: {rep5.reliability*100:.1f}%")
    print(f"Fake Probability: {rep5.final_fake_probability*100:.2f}%")
    suspicious_count = len(rep5.visual_artifacts.get("suspicious_frames", []))
    print(f"Suspicious Frames Count: {suspicious_count}")

    # Test 6: Real Audio
    print("\n--- [TEST 6] REAL AUDIO: real_voice_01.wav ---")
    res = aud_det.predict_structured(str(eval_dir / "test_audio" / "real_voice_01.wav"))
    rep6 = fusion.fuse_audio_evidence(res, sample_name="real_voice_01.wav")
    print(f"Verdict: {rep6.final_verdict.value}")
    print(f"Confidence: {rep6.confidence*100:.1f}%")
    print(f"Reliability: {rep6.reliability*100:.1f}%")
    print(f"Fake Probability: {rep6.final_fake_probability*100:.2f}%")

    # Test 7: Fake Audio
    print("\n--- [TEST 7] FAKE AUDIO: fake_voice_01.wav ---")
    res = aud_det.predict_structured(str(eval_dir / "test_audio" / "fake_voice_01.wav"))
    rep7 = fusion.fuse_audio_evidence(res, sample_name="fake_voice_01.wav")
    print(f"Verdict: {rep7.final_verdict.value}")
    print(f"Confidence: {rep7.confidence*100:.1f}%")
    print(f"Reliability: {rep7.reliability*100:.1f}%")
    print(f"Fake Probability: {rep7.final_fake_probability*100:.2f}%")

    # Test 8: Invalid File Rejection
    print("\n--- [TEST 8] INVALID FILE HANDLING ---")
    try:
        img_det.predict_structured(b"CORRUPTED_NON_IMAGE_DATA_BYTES")
        print("ERROR: Corrupted data was not rejected!")
    except ValueError as ve:
        print(f"PASSED: Successfully rejected invalid data with ValueError: '{ve}'")

    # Test 9: Missing Checkpoint Handling
    print("\n--- [TEST 9] MISSING CHECKPOINT HANDLING ---")
    try:
        ImageDeepfakeDetector(weights_path="non_existent_weights.pth")
        print("ERROR: Missing checkpoint did not raise FileNotFoundError!")
    except FileNotFoundError as fnf:
        print(f"PASSED: Handled missing checkpoint with FileNotFoundError: '{fnf}'")

    # Test 10: Report Generation (Markdown & JSON)
    print("\n--- [TEST 10] FORENSIC REPORT GENERATION ---")
    md_report = ForensicReportGenerator.generate_markdown_report(rep1)
    json_report = ForensicReportGenerator.generate_json_report(rep1)
    print(f"Markdown Certificate Generated: {len(md_report)} chars")
    parsed_json = json.loads(json_report)
    print(f"JSON Data Validated: sample_name='{parsed_json['sample_name']}', verdict='{parsed_json['final_verdict']}'")

    # Test 11: Localhost HTTP Check
    print("\n--- [TEST 11] LOCALHOST WEBPAGE HTTP STATUS ---")
    try:
        req = urllib.request.Request("http://localhost:8501", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"HTTP Status: {resp.status} (OK)")
    except Exception as e:
        print(f"HTTP Status Exception: {e}")

    print("\n" + "=" * 60)
    print("  ALL END-TO-END VALIDATION TESTS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
