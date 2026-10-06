import os
import uuid
import subprocess
import cv2
import numpy as np
import joblib
import tempfile
from .path_utils import get_video_model_path


NUM_FRAMES = 5
IMG_SIZE = 224
COATNET_DIM = 768
WHISPER_DIM = 512
VIDEO_FEAT_DIM = 3 * COATNET_DIM   # statistical pooling (mean, max, std) -> 2304
AUDIO_FEAT_DIM = WHISPER_DIM       # mean pooling -> 512
FUSED_DIM = VIDEO_FEAT_DIM + AUDIO_FEAT_DIM  # 2816
OPTIMAL_THRESHOLD = 0.5780         # derived from balanced evaluation


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


def extract_frames_from_video(video_path: str, num_frames: int = NUM_FRAMES) -> list[np.ndarray]:
    """Uniformly sample RGB frames from video."""
    frames = []
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames

        frame_buffer = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret or count >= 300:
                break
            if frame is not None and frame.mean() > 5:
                frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame_buffer.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            count += 1

        if len(frame_buffer) > 0:
            indices = np.linspace(0, len(frame_buffer) - 1, num_frames, dtype=int)
            frames = [frame_buffer[i] for i in indices]
    except Exception:
        pass
    finally:
        if cap is not None:
            cap.release()

    if 0 < len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[0].copy())

    return frames


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

    if os.path.exists(tmp_wav) and os.path.getsize(tmp_wav) > 0:
        return tmp_wav
    return None


class VideoDeepfakeDetector:
    """Multimodal (visual + acoustic intermediate fusion) deepfake detector."""

    def __init__(self, xgb_path: str | None = None):
        if xgb_path is None:
            xgb_path = get_video_model_path()

        if xgb_path is None or not os.path.exists(xgb_path):
            raise FileNotFoundError(
                f"Video fusion model weights file not found! Expected 'xgboost_fusion_model.joblib' in models or Final-Vid directory."
            )

        self.xgb_path = xgb_path
        self.classifier = joblib.load(self.xgb_path)

    def predict(self, video_path: str, image_model, audio_model) -> dict:
        """Run multimodal inference on a video file."""
        # 1. Visual features
        frames = extract_frames_from_video(video_path, NUM_FRAMES)
        if len(frames) == 0:
            raise ValueError("Could not decode video or extract valid frames.")

        frame_embeddings = []
        for frame_rgb in frames:
            emb = image_model.extract_features(frame_rgb)
            frame_embeddings.append(emb)

        frame_embeddings = np.array(frame_embeddings)  # (5, 768)

        # Statistical pooling (mean, max, std)
        feat_mean = frame_embeddings.mean(axis=0)
        feat_max = frame_embeddings.max(axis=0)
        feat_std = frame_embeddings.std(axis=0)
        video_vector = np.concatenate([feat_mean, feat_max, feat_std])  # (2304,)

        # 2. Audio features
        tmp_wav = extract_audio_from_video(video_path)
        has_audio = False
        if tmp_wav is not None:
            try:
                audio_vector = audio_model.extract_features(tmp_wav)
                has_audio = True
            except Exception:
                audio_vector = np.zeros(AUDIO_FEAT_DIM, dtype=np.float32)
            finally:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
        else:
            audio_vector = np.zeros(AUDIO_FEAT_DIM, dtype=np.float32)

        # 3. Intermediate Feature Fusion
        fused = np.concatenate([video_vector, audio_vector])  # (2816,)
        fused_2d = fused.reshape(1, -1)

        # 4. XGBoost Classification
        spoof_prob = float(self.classifier.predict_proba(fused_2d)[0, 1])
        real_prob = float(1.0 - spoof_prob)

        is_fake = spoof_prob >= OPTIMAL_THRESHOLD
        confidence = spoof_prob if is_fake else real_prob
        label = "FAKE" if is_fake else "REAL"
        detail_label = "Fake (Deepfake Video Detected)" if is_fake else "Real (Authentic Video)"

        return {
            "prediction": label,
            "detail_label": detail_label,
            "is_fake": is_fake,
            "confidence": confidence,
            "fake_prob": spoof_prob,
            "real_prob": real_prob,
            "sampled_frames": frames,
            "has_audio": has_audio,
        }
