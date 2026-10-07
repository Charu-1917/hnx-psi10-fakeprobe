"""
Unit Tests for Input Quality Analyzer in FakeProbe-X.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepfake_detector_core import InputQualityAnalyzer, QualityLevel


def test_image_quality_analyzer_sharp_vs_blurry():
    qa = InputQualityAnalyzer()
    
    # Sharp image with edges
    sharp_img = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(0, 300, 20):
        cv2.line(sharp_img, (i, 0), (i, 300), (255, 255, 255), 2)
    res_sharp = qa.analyze_image(sharp_img)
    assert res_sharp.quality_score >= 0.50
    assert res_sharp.is_acceptable is True

    # Blurry image
    blurry_img = cv2.GaussianBlur(sharp_img, (45, 45), 20)
    res_blurry = qa.analyze_image(blurry_img)
    assert res_blurry.quality_score < res_sharp.quality_score
    assert any("blur" in issue.lower() for issue in res_blurry.issues)


def test_audio_quality_analyzer_silence_and_clipping():
    qa = InputQualityAnalyzer()

    # Silent audio
    silent = np.zeros(16000 * 2, dtype=np.float32)
    res_silent = qa.analyze_audio(silent, sample_rate=16000)
    assert res_silent.quality_level == QualityLevel.LOW
    assert res_silent.is_acceptable is False

    # Clipped audio
    clipped = np.ones(16000 * 2, dtype=np.float32)
    res_clipped = qa.analyze_audio(clipped, sample_rate=16000)
    assert any("clip" in issue.lower() for issue in res_clipped.issues)


if __name__ == "__main__":
    test_image_quality_analyzer_sharp_vs_blurry()
    test_audio_quality_analyzer_silence_and_clipping()
    print("[PASS] All quality analyzer tests passed.")
