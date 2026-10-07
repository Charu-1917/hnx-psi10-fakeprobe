"""
Benchmark Test Sample Generator for FakeProbe-X Evaluation Harness.
Creates standardized, reproducible real and synthetic test fixtures across Image, Video, and Audio modalities.
"""

import os
import cv2
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
IMG_DIR = EVAL_DIR / "test_images"
VID_DIR = EVAL_DIR / "test_videos"
AUD_DIR = EVAL_DIR / "test_audio"


def generate_face_image(is_fake: bool, seed: int = 42) -> np.ndarray:
    """Generate deterministic test face portrait with realistic sensor noise or synthetic blend artifacts."""
    np.random.seed(seed)
    h, w = 320, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Background texture
    for c in range(3):
        base = 130 + (c * 15)
        img[:, :, c] = np.clip(np.random.normal(base, 10, (h, w)), 0, 255).astype(np.uint8)

    # Face Oval (skin tone RGB)
    skin_color = (210, 180, 160) if not is_fake else (225, 170, 150)
    cv2.ellipse(img, (160, 160), (75, 100), 0, 0, 360, skin_color, -1)

    # Eyes
    cv2.ellipse(img, (130, 140), (12, 7), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (190, 140), (12, 7), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (130, 140), 5, (40, 30, 20), -1)
    cv2.circle(img, (190, 140), 5, (40, 30, 20), -1)

    # Nose
    cv2.ellipse(img, (160, 175), (6, 12), 0, 0, 360, (180, 150, 130), -1)

    # Mouth
    cv2.ellipse(img, (160, 215), (28, 10), 0, 0, 360, (170, 80, 80), -1)

    if is_fake:
        # Inject boundary blending discontinuity and high frequency checkerboard artifacts
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (160, 160), 65, 1.0, -1)
        grid_noise = np.sin(np.linspace(0, 50 * np.pi, w)) * np.sin(np.linspace(0, 50 * np.pi, h))[:, None]
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c] + (grid_noise * mask * 25.0), 0, 255).astype(np.uint8)
    else:
        # Natural sensor Poisson/Gaussian noise
        noise = np.random.normal(0, 2.5, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def generate_audio_signal(is_fake: bool, duration_sec: float = 2.0, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """Generate synthetic voice-like harmonic tone (Real) vs frequency-modulated/vocoder artifact signal (Fake)."""
    np.random.seed(seed)
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    if not is_fake:
        # Natural harmonic series with human vocal formants (150Hz, 300Hz, 450Hz) + breathing noise
        f0 = 160.0
        signal = (
            0.5 * np.sin(2 * np.pi * f0 * t) +
            0.3 * np.sin(2 * np.pi * (2 * f0) * t) +
            0.15 * np.sin(2 * np.pi * (3 * f0) * t) +
            0.05 * np.random.normal(0, 0.05, len(t))
        )
    else:
        # Deepfake vocoder artifact: metallic phase distortion + buzz chirp modulation
        f0 = 160.0
        mod = 30.0 * np.sin(2 * np.pi * 15 * t)
        signal = (
            0.6 * np.sin(2 * np.pi * (f0 + mod) * t) +
            0.4 * np.sign(np.sin(2 * np.pi * (3 * f0) * t)) * 0.2 +
            0.1 * np.sin(2 * np.pi * 3800 * t)
        )

    # Normalize to [-0.8, 0.8]
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = (signal / max_val) * 0.75
    return signal.astype(np.float32)


def generate_test_video(output_path: str, is_fake: bool, num_frames: int = 25, fps: int = 10, seed: int = 42):
    """Generate short MP4 test video clip."""
    h, w = 240, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for i in range(num_frames):
        frame = generate_face_image(is_fake=is_fake, seed=seed + i)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(frame_bgr, (w, h))
        out.write(resized)
    out.release()


def prepare_all_fixtures():
    """Create all test folders and samples."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    VID_DIR.mkdir(parents=True, exist_ok=True)
    AUD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Images: 5 Real, 5 Fake, 2 Degraded
    for i in range(1, 6):
        real_img = generate_face_image(is_fake=False, seed=100 + i)
        cv2.imwrite(str(IMG_DIR / f"real_face_{i:02d}.png"), cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR))

        fake_img = generate_face_image(is_fake=True, seed=200 + i)
        cv2.imwrite(str(IMG_DIR / f"fake_face_{i:02d}.png"), cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR))

    # Low quality / blurred test image
    low_q = cv2.GaussianBlur(generate_face_image(is_fake=False, seed=999), (31, 31), 10)
    cv2.imwrite(str(IMG_DIR / "degraded_low_quality.png"), cv2.cvtColor(low_q, cv2.COLOR_RGB2BGR))

    # 2. Audio: 5 Real, 5 Fake
    for i in range(1, 6):
        real_audio = generate_audio_signal(is_fake=False, duration_sec=2.5, seed=300 + i)
        wavfile.write(str(AUD_DIR / f"real_voice_{i:02d}.wav"), 16000, (real_audio * 32767).astype(np.int16))

        fake_audio = generate_audio_signal(is_fake=True, duration_sec=2.5, seed=400 + i)
        wavfile.write(str(AUD_DIR / f"fake_voice_{i:02d}.wav"), 16000, (fake_audio * 32767).astype(np.int16))

    # 3. Video: 3 Real, 3 Fake
    for i in range(1, 4):
        generate_test_video(str(VID_DIR / f"real_video_{i:02d}.mp4"), is_fake=False, num_frames=20, seed=500 + i)
        generate_test_video(str(VID_DIR / f"fake_video_{i:02d}.mp4"), is_fake=True, num_frames=20, seed=600 + i)

    print(f"[SUCCESS] Created evaluation test fixtures in {EVAL_DIR}")


if __name__ == "__main__":
    prepare_all_fixtures()
