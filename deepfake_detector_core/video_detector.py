"""
Multimodal Video Deepfake Detector for FakeProbe-X.
Combines visual multi-frame temporal feature pooling, acoustic feature extraction,
face tracking consistency, and temporal anomaly segmentation into a standardized detector.
"""

import os
import uuid
import subprocess
import cv2
import numpy as np
import joblib
from typing import Optional, List, Tuple

from .path_utils import get_video_model_path
from .types import DetectorResult, DecisionVerdict, QualityResult, QualityLevel
from .quality_analyzer import InputQualityAnalyzer
from .face_utils import FaceDetector
from configs.forensic_config import DEFAULT_CONFIG


def get_ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def transcode_for_browser(input_path: str) -> str | None:
    """Re-encode video to H.264 / yuv420p for guaranteed browser playback."""
    output_path = input_path + "_browser_preview.mp4"
    ffmpeg_exe = get_ffmpeg_executable()

    cmd = [
        ffmpeg_exe, "-y", "-v", "error",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        subprocess.run(cmd, timeout=30, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception:
        return None


def extract_frames_with_metadata(
    video_path: str,
    num_frames: int = 5,
    max_scan_frames: int = 300
) -> Tuple[List[np.ndarray], List[float], dict]:
    """
    Sample uniformly spaced frames from video and return frame timestamps and metadata.
    """
    frames = []
    timestamps = []
    metadata = {}
    cap = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames, timestamps, metadata

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = total_frames / fps if fps > 0 else 0.0

        metadata = {
            "fps": float(fps),
            "frame_count": total_frames,
            "width": width,
            "height": height,
            "duration_sec": float(duration_sec),
        }

        frame_buffer = []
        time_buffer = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret or count >= max_scan_frames:
                break
            if frame is not None and frame.mean() > 5:
                frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                time_buffer.append(count / fps)
            count += 1

        if len(frame_buffer) > 0:
            indices = np.linspace(0, len(frame_buffer) - 1, min(num_frames, len(frame_buffer)), dtype=int)
            frames = [frame_buffer[i] for i in indices]
            timestamps = [time_buffer[i] for i in indices]

    except Exception:
        pass
    finally:
        if cap is not None:
            cap.release()

    # Padding if video is extremely short
    if 0 < len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
            timestamps.append(timestamps[-1] if timestamps else 0.0)

    return frames, timestamps, metadata


def extract_audio_from_video(video_path: str) -> str | None:
    """Demux audio track to temporary WAV file."""
    tmp_dir = os.environ.get("TEMP", "/tmp")
    tmp_wav = os.path.join(tmp_dir, f"vid_audio_{uuid.uuid4().hex[:8]}.wav")
    ffmpeg_exe = get_ffmpeg_executable()

    cmd = [
        ffmpeg_exe, "-y", "-v", "error",
        "-i", video_path,
        "-t", "30",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        tmp_wav,
    ]
    process = None
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if process:
            process.kill()
            process.wait()
        return None
    except Exception:
        if process:
            process.kill()
            process.wait()
        return None

    if os.path.exists(tmp_wav) and os.path.getsize(tmp_wav) > 1000:
        return tmp_wav
    return None


class VideoDeepfakeDetector:
    """Multimodal intermediate fusion video deepfake detector."""

    def __init__(self, xgb_path: str | None = None, config=DEFAULT_CONFIG):
        self.config = config

        if xgb_path is None:
            xgb_path = get_video_model_path()

        if xgb_path is None or not os.path.exists(xgb_path):
            raise FileNotFoundError(
                f"Video fusion model weights file not found! Expected 'xgboost_fusion_model.joblib' in models or Final-Vid directory."
            )

        self.xgb_path = xgb_path
        self.classifier = joblib.load(self.xgb_path)

        self.quality_analyzer = InputQualityAnalyzer(config=self.config)
        self.face_detector = FaceDetector(device=self.config.device)

    def predict_structured(
        self,
        video_path: str,
        image_model,
        audio_model,
        num_frames: int = 5
    ) -> DetectorResult:
        """
        Run complete multimodal forensic analysis on a video file.
        Returns standardized DetectorResult with temporal timelines and frame scores.
        """
        # 1. Video Decoding & Frame Sampling
        frames, timestamps, meta = extract_frames_with_metadata(
            video_path,
            num_frames=num_frames,
            max_scan_frames=self.config.max_scan_frames
        )

        if len(frames) == 0:
            raise ValueError("Could not decode video or extract valid frames.")

        # 2. Face Tracking & Frame Quality Assessment
        face_track_res = self.face_detector.track_faces_across_frames(frames)
        frame_qualities = [
            self.quality_analyzer.analyze_image(f, face_boxes=[box] if box is not None else None)
            for f, box in zip(frames, face_track_res["boxes_per_frame"])
        ]

        # 3. Audio Extraction & Demuxing
        tmp_wav = extract_audio_from_video(video_path)
        has_audio = (tmp_wav is not None)

        quality_res = self.quality_analyzer.analyze_video(
            video_metadata=meta,
            frame_qualities=frame_qualities,
            face_detection_rate=face_track_res["detection_rate"],
            has_audio=has_audio
        )

        # 4. Frame Feature Extraction & Frame-Level Predictions
        frame_embeddings = []
        frame_fake_probs = []
        processed_face_frames = []

        for frame_rgb in frames:
            # Crop face if available for frame analysis
            crop_rgb, _, _ = self.face_detector.crop_primary_face(
                frame_rgb,
                target_size=self.config.image_input_size,
                margin=self.config.face_crop_margin
            )
            processed_face_frames.append(crop_rgb)

            # Feature embedding for fusion
            emb = image_model.extract_features(crop_rgb)
            frame_embeddings.append(emb)

            # Frame level single-frame prediction
            frame_pred = image_model.predict_structured(crop_rgb, apply_face_crop=False)
            frame_fake_probs.append(frame_pred.probability_fake)

        frame_embeddings = np.array(frame_embeddings)  # (N, 768)

        # Statistical Pooling (mean, max, std) -> (2304,)
        feat_mean = frame_embeddings.mean(axis=0)
        feat_max = frame_embeddings.max(axis=0)
        feat_std = frame_embeddings.std(axis=0)
        video_vector = np.concatenate([feat_mean, feat_max, feat_std])

        # 5. Acoustic Feature Extraction
        audio_detector_result = None
        if has_audio and tmp_wav is not None:
            try:
                audio_vector = audio_model.extract_features(tmp_wav)
                audio_detector_result = audio_model.predict_structured(tmp_wav)
            except Exception:
                audio_vector = np.zeros(self.config.whisper_embedding_dim, dtype=np.float32)
            finally:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
        else:
            audio_vector = np.zeros(self.config.whisper_embedding_dim, dtype=np.float32)

        # 6. Intermediate Fusion Classification
        fused = np.concatenate([video_vector, audio_vector])
        fused_2d = fused.reshape(1, -1)

        spoof_prob = float(self.classifier.predict_proba(fused_2d)[0, 1])
        real_prob = float(1.0 - spoof_prob)

        # 7. Temporal Consistency Analysis
        frame_prob_mean = float(np.mean(frame_fake_probs))
        frame_prob_std = float(np.std(frame_fake_probs))
        frame_prob_max = float(np.max(frame_fake_probs))

        suspicious_frames = []
        for idx, (prob, ts) in enumerate(zip(frame_fake_probs, timestamps)):
            if prob >= self.config.uncertainty_fake_boundary:
                suspicious_frames.append({
                    "frame_index": idx,
                    "timestamp_sec": round(ts, 2),
                    "fake_probability": round(prob, 4),
                    "severity": "HIGH" if prob > 0.80 else "MEDIUM"
                })

        # When audio is missing, adjust fusion confidence with visual frame ensemble
        if not has_audio:
            # Weighted average between XGBoost and visual frame mean
            adjusted_spoof_prob = 0.50 * spoof_prob + 0.50 * frame_prob_mean
        else:
            adjusted_spoof_prob = spoof_prob

        # Decision & Reliability
        is_fake = adjusted_spoof_prob >= self.config.video_decision_threshold
        boundary_dist = abs(adjusted_spoof_prob - 0.50) * 2.0
        confidence = float(np.clip(boundary_dist, 0.0, 1.0))

        # Reliability considers video quality, face tracking consistency, and audio availability
        reliability = float(np.clip(
            0.50 * quality_res.quality_score +
            0.25 * face_track_res["face_consistency_score"] +
            0.25 * (1.0 if has_audio else 0.70),
            0.0, 1.0
        ))

        # Three-Way Decision Logic
        if quality_res.quality_level == QualityLevel.LOW and not quality_res.is_acceptable:
            prediction = DecisionVerdict.UNCERTAIN
            detail_label = "Uncertain (Severe Video Artifacts / Low Quality)"
        elif self.config.uncertainty_real_boundary <= adjusted_spoof_prob <= self.config.uncertainty_fake_boundary:
            prediction = DecisionVerdict.UNCERTAIN
            detail_label = "Uncertain (Inconclusive Temporal Signatures)"
        elif adjusted_spoof_prob >= self.config.video_decision_threshold:
            prediction = DecisionVerdict.FAKE
            detail_label = "Fake (Deepfake Manipulation Detected)"
        else:
            prediction = DecisionVerdict.REAL
            detail_label = "Real (Authentic Video Stream)"

        evidence = {
            "has_audio": has_audio,
            "raw_fused_score": spoof_prob,
            "adjusted_fake_prob": adjusted_spoof_prob,
            "frame_timestamps": timestamps,
            "frame_fake_probabilities": frame_fake_probs,
            "frame_variance_std": frame_prob_std,
            "suspicious_frames": suspicious_frames,
            "face_tracking": face_track_res,
            "sampled_frames_rgb": frames,
            "processed_face_frames_rgb": processed_face_frames,
            "audio_result": audio_detector_result.to_dict() if audio_detector_result else None,
        }

        return DetectorResult(
            model="Multimodal Intermediate Fusion (CoAtNet + Whisper + XGBoost)",
            modality="video",
            prediction=prediction,
            raw_score=spoof_prob,
            probability_fake=adjusted_spoof_prob,
            probability_real=1.0 - adjusted_spoof_prob,
            confidence=confidence,
            reliability=reliability,
            quality=quality_res,
            evidence=evidence,
            detail_label=detail_label,
        )

    def predict(self, video_path: str, image_model, audio_model) -> dict:
        """Legacy dictionary interface."""
        res = self.predict_structured(video_path, image_model, audio_model, num_frames=self.config.default_num_frames)
        return {
            "prediction": res.prediction.value,
            "detail_label": res.detail_label,
            "is_fake": (res.prediction == DecisionVerdict.FAKE),
            "confidence": res.confidence,
            "reliability": res.reliability,
            "fake_prob": res.probability_fake,
            "real_prob": res.probability_real,
            "sampled_frames": res.evidence.get("sampled_frames_rgb"),
            "has_audio": res.evidence.get("has_audio"),
            "suspicious_frames": res.evidence.get("suspicious_frames"),
            "quality": res.quality.to_dict(),
        }
