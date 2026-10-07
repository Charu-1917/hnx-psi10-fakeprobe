"""
Unit Tests for Frequency-Domain Forensic Analyzer.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepfake_detector_core import FrequencyForensicAnalyzer


def test_frequency_domain_spectrum_and_slope():
    fa = FrequencyForensicAnalyzer()

    # Uniform smooth image (steep spectral falloff)
    smooth = np.tile(np.linspace(50, 200, 256, dtype=np.uint8), (256, 1))
    smooth_rgb = cv2.cvtColor(smooth, cv2.COLOR_GRAY2RGB)
    res_smooth = fa.analyze_frequency_domain(smooth_rgb)

    assert "frequency_anomaly_score" in res_smooth
    assert "spectral_slope" in res_smooth
    assert "high_frequency_ratio" in res_smooth
    assert 0.0 <= res_smooth["frequency_anomaly_score"] <= 1.0


def test_frequency_high_freq_noise_detection():
    fa = FrequencyForensicAnalyzer()
    # High frequency checkerboard / noise pattern
    noise = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    res_noise = fa.analyze_frequency_domain(noise)

    # High frequency ratio should be significantly higher for white noise (> 0.05) vs smooth (< 0.001)
    assert res_noise["high_frequency_ratio"] > 0.05


if __name__ == "__main__":
    test_frequency_domain_spectrum_and_slope()
    test_frequency_high_freq_noise_detection()
    print("[PASS] All frequency analyzer tests passed.")
