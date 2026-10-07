"""
Input Quality Analyzer for FakeProbe-X.
Evaluates the physical and forensic quality of Images, Videos, and Audio signals.
Provides objective quality scores, quality levels (GOOD/MEDIUM/LOW), and specific degradation flags.
"""

import cv2
import numpy as np
from typing import Optional
from .types import QualityResult, QualityLevel
from configs.forensic_config import DEFAULT_CONFIG


class InputQualityAnalyzer:
    """Multi-modal input quality assessment suite."""

    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config

    # ==========================================
    # 1. IMAGE QUALITY ANALYSIS
    # ==========================================
    def analyze_image(
        self,
        image_rgb: np.ndarray,
        face_boxes: Optional[list] = None
    ) -> QualityResult:
        """
        Evaluate image quality based on resolution, blur, luminance, contrast, and face size.
        """
        issues = []
        metrics = {}

        h, w, c = image_rgb.shape
        metrics["height"] = h
        metrics["width"] = w
        metrics["megapixels"] = (h * w) / 1e6

        # A. Resolution Check
        min_w, min_h = self.config.min_image_resolution
        if w < min_w or h < min_h:
            issues.append(f"Low resolution ({w}x{h} < {min_w}x{min_h})")
            res_score = max(0.1, (w * h) / (min_w * min_h))
        else:
            res_score = min(1.0, (w * h) / (512 * 512))
        metrics["resolution_score"] = float(res_score)

        # B. Blur / Sharpness via Laplacian Variance
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics["laplacian_variance"] = float(laplacian_var)

        if laplacian_var < self.config.blur_laplacian_threshold_low:
            issues.append(f"Severe blur detected (sharpness: {laplacian_var:.1f})")
            blur_score = max(0.0, laplacian_var / self.config.blur_laplacian_threshold_low * 0.5)
        elif laplacian_var < self.config.blur_laplacian_threshold_good:
            issues.append(f"Moderate blur detected (sharpness: {laplacian_var:.1f})")
            blur_score = 0.5 + 0.5 * ((laplacian_var - self.config.blur_laplacian_threshold_low) /
                                      (self.config.blur_laplacian_threshold_good - self.config.blur_laplacian_threshold_low))
        else:
            blur_score = 1.0
        metrics["blur_score"] = float(blur_score)

        # C. Luminance & Exposure
        mean_lum = float(gray.mean())
        metrics["mean_luminance"] = mean_lum
        if mean_lum < 30.0:
            issues.append(f"Under-exposed / dark image (mean luminance: {mean_lum:.1f})")
            lum_score = max(0.2, mean_lum / 30.0)
        elif mean_lum > 225.0:
            issues.append(f"Over-exposed / washed out image (mean luminance: {mean_lum:.1f})")
            lum_score = max(0.2, (255.0 - mean_lum) / 30.0)
        else:
            lum_score = 1.0
        metrics["luminance_score"] = float(lum_score)

        # D. Contrast
        std_contrast = float(gray.std())
        metrics["contrast_std"] = std_contrast
        if std_contrast < 20.0:
            issues.append(f"Very low contrast (std: {std_contrast:.1f})")
            contrast_score = max(0.2, std_contrast / 20.0)
        else:
            contrast_score = min(1.0, std_contrast / 50.0)
        metrics["contrast_score"] = float(contrast_score)

        # E. Face Presence & Size
        num_faces = len(face_boxes) if face_boxes is not None else 0
        metrics["num_faces"] = num_faces

        if face_boxes is not None and len(face_boxes) > 0:
            # Check largest face coverage
            box = face_boxes[0]
            x1, y1, x2, y2 = box[:4]
            face_area = max(0, x2 - x1) * max(0, y2 - y1)
            face_ratio = face_area / (w * h)
            metrics["face_area_ratio"] = float(face_ratio)
            if face_ratio < self.config.min_face_size_ratio:
                issues.append(f"Face is very small relative to frame ({face_ratio*100:.1f}% area)")
                face_score = 0.6
            else:
                face_score = 1.0
        else:
            face_score = 0.7  # Non-face images get moderate penalty for face-specialized models
            metrics["face_area_ratio"] = 0.0

        # Weighted aggregate quality score
        quality_score = (
            0.25 * res_score +
            0.30 * blur_score +
            0.15 * lum_score +
            0.15 * contrast_score +
            0.15 * face_score
        )
        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        if quality_score >= 0.75 and len(issues) == 0:
            quality_level = QualityLevel.GOOD
        elif quality_score >= 0.45:
            quality_level = QualityLevel.MEDIUM
        else:
            quality_level = QualityLevel.LOW

        is_acceptable = quality_score >= 0.35

        return QualityResult(
            modality="image",
            quality_score=quality_score,
            quality_level=quality_level,
            is_acceptable=is_acceptable,
            issues=issues,
            metrics=metrics,
        )

    # ==========================================
    # 2. VIDEO QUALITY ANALYSIS
    # ==========================================
    def analyze_video(
        self,
        video_metadata: dict,
        frame_qualities: list[QualityResult],
        face_detection_rate: float,
        has_audio: bool
    ) -> QualityResult:
        """
        Evaluate video stream quality based on resolution, duration, face tracking rate, and frame sharpness.
        """
        issues = []
        metrics = {
            "fps": video_metadata.get("fps", 0.0),
            "duration_sec": video_metadata.get("duration_sec", 0.0),
            "frame_count": video_metadata.get("frame_count", 0),
            "width": video_metadata.get("width", 0),
            "height": video_metadata.get("height", 0),
            "face_detection_rate": face_detection_rate,
            "has_audio": has_audio,
        }

        # Resolution score
        w = video_metadata.get("width", 0)
        h = video_metadata.get("height", 0)
        if w < 128 or h < 128:
            issues.append(f"Low video resolution ({w}x{h})")
            res_score = 0.3
        else:
            res_score = min(1.0, (w * h) / (480 * 480))

        # Frame quality average
        if len(frame_qualities) > 0:
            avg_frame_score = float(np.mean([fq.quality_score for fq in frame_qualities]))
            for fq in frame_qualities:
                for issue in fq.issues:
                    if issue not in issues and len(issues) < 4:
                        issues.append(f"Frame warning: {issue}")
        else:
            avg_frame_score = 0.2
            issues.append("No valid frames could be decoded")

        metrics["avg_frame_quality"] = avg_frame_score

        # Face tracking coverage
        if face_detection_rate < 0.20:
            issues.append(f"Low facial visibility across frames ({face_detection_rate*100:.1f}%)")
            face_cov_score = 0.4
        else:
            face_cov_score = min(1.0, face_detection_rate)

        # Audio check
        if not has_audio:
            issues.append("No audio track detected in video (visual-only analysis)")
            audio_mod_score = 0.8  # Slight penalty for missing cross-modal audio stream
        else:
            audio_mod_score = 1.0

        quality_score = (
            0.20 * res_score +
            0.40 * avg_frame_score +
            0.25 * face_cov_score +
            0.15 * audio_mod_score
        )
        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        if quality_score >= 0.70 and len(issues) <= 1:
            quality_level = QualityLevel.GOOD
        elif quality_score >= 0.40:
            quality_level = QualityLevel.MEDIUM
        else:
            quality_level = QualityLevel.LOW

        is_acceptable = quality_score >= 0.30

        return QualityResult(
            modality="video",
            quality_score=quality_score,
            quality_level=quality_level,
            is_acceptable=is_acceptable,
            issues=issues,
            metrics=metrics,
        )

    # ==========================================
    # 3. AUDIO QUALITY ANALYSIS
    # ==========================================
    def analyze_audio(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000
    ) -> QualityResult:
        """
        Evaluate acoustic signal quality based on duration, clipping, SNR/silence, and energy.
        """
        issues = []
        metrics = {}

        duration_sec = len(audio_array) / float(sample_rate) if sample_rate > 0 else 0.0
        metrics["duration_sec"] = duration_sec
        metrics["sample_rate"] = sample_rate

        # Duration check
        if duration_sec < self.config.min_audio_duration_sec:
            issues.append(f"Audio clip is too short ({duration_sec:.2f}s < {self.config.min_audio_duration_sec}s)")
            dur_score = max(0.1, duration_sec / self.config.min_audio_duration_sec)
        else:
            dur_score = min(1.0, duration_sec / 3.0)
        metrics["duration_score"] = float(dur_score)

        if len(audio_array) == 0 or np.all(audio_array == 0):
            return QualityResult(
                modality="audio",
                quality_score=0.0,
                quality_level=QualityLevel.LOW,
                is_acceptable=False,
                issues=["Audio is empty or silent"],
                metrics={"duration_sec": 0.0},
            )

        # Clipping Check (|x| >= 0.99)
        clipped_samples = np.sum(np.abs(audio_array) >= 0.99)
        clip_ratio = float(clipped_samples / len(audio_array))
        metrics["clipping_ratio"] = clip_ratio
        if clip_ratio > self.config.audio_clipping_threshold:
            issues.append(f"Audio clipping distortion detected ({clip_ratio*100:.1f}% samples)")
            clip_score = max(0.2, 1.0 - clip_ratio * 4.0)
        else:
            clip_score = 1.0
        metrics["clipping_score"] = float(clip_score)

        # RMS Energy & Silence Ratio
        frame_len = int(sample_rate * 0.03)  # 30ms frames
        if frame_len > 0 and len(audio_array) >= frame_len:
            frames = np.array([
                audio_array[i:i+frame_len]
                for i in range(0, len(audio_array) - frame_len + 1, frame_len)
            ])
            frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-10)
            db = 20 * np.log10(frame_rms + 1e-10)
            silence_ratio = float(np.mean(db < self.config.audio_silence_threshold_db))
            metrics["silence_ratio"] = silence_ratio
            metrics["mean_db"] = float(np.mean(db))

            if silence_ratio > 0.70:
                issues.append(f"High silence ratio ({silence_ratio*100:.1f}% inactive)")
                silence_score = max(0.2, 1.0 - silence_ratio)
            else:
                silence_score = 1.0
        else:
            silence_score = 0.5
            metrics["silence_ratio"] = 0.0

        metrics["silence_score"] = float(silence_score)

        # Weighted aggregate audio quality score
        quality_score = (
            0.35 * dur_score +
            0.35 * silence_score +
            0.30 * clip_score
        )
        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        if quality_score >= 0.75 and len(issues) == 0:
            quality_level = QualityLevel.GOOD
        elif quality_score >= 0.45:
            quality_level = QualityLevel.MEDIUM
        else:
            quality_level = QualityLevel.LOW

        is_acceptable = quality_score >= 0.35

        return QualityResult(
            modality="audio",
            quality_score=quality_score,
            quality_level=quality_level,
            is_acceptable=is_acceptable,
            issues=issues,
            metrics=metrics,
        )
