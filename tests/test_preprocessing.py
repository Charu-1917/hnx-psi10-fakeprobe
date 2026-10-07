"""
Unit Tests for Image, Audio, and Video Preprocessing in FakeProbe-X.
"""

import cv2
import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepfake_detector_core import (
    compute_ela,
    compute_fft_magnitude,
    prepare_5ch_tensor,
    FaceDetector,
)


def test_compute_ela_shape_and_range():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    ela = compute_ela(img, quality=90)
    assert ela.shape == (100, 100)
    assert ela.dtype == np.float32
    assert 0.0 <= ela.min() and ela.max() <= 1.0


def test_compute_fft_magnitude_shape_and_range():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    fft_mag = compute_fft_magnitude(img)
    assert fft_mag.shape == (100, 100)
    assert fft_mag.dtype == np.float32
    assert 0.0 <= fft_mag.min() and fft_mag.max() <= 1.0


def test_prepare_5ch_tensor_dimensions_and_normalization():
    img = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
    tensor = prepare_5ch_tensor(img, size=224)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (5, 224, 224)
    assert tensor.dtype == torch.float32


def test_face_detector_fallback_on_noise():
    fd = FaceDetector(device="cpu")
    noise = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    crop, box, conf = fd.crop_primary_face(noise, target_size=224)
    assert crop.shape == (224, 224, 3)
    assert box is None
    assert conf == 0.0


if __name__ == "__main__":
    test_compute_ela_shape_and_range()
    test_compute_fft_magnitude_shape_and_range()
    test_prepare_5ch_tensor_dimensions_and_normalization()
    test_face_detector_fallback_on_noise()
    print("[PASS] All preprocessing tests passed.")
