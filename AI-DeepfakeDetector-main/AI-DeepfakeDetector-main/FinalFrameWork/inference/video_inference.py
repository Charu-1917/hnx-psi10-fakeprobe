import os
import uuid
import subprocess
import cv2
import numpy as np
import joblib


# ── Configuration (mirrors inference_config.json) 
NUM_FRAMES     = 5
IMG_SIZE       = 224
COATNET_DIM    = 768          # CoAtNet-0 penultimate dim
WHISPER_DIM    = 512          # whisper-base encoder hidden dim
VIDEO_FEAT_DIM = 3 * COATNET_DIM   # statistical pooling → 2304
AUDIO_FEAT_DIM = WHISPER_DIM       # mean pooling → 512
FUSED_DIM      = VIDEO_FEAT_DIM + AUDIO_FEAT_DIM  # 2816
OPTIMAL_THRESHOLD = 0.5780   # from balanced evaluation


# ── Frame sampling 
def extract_frames_from_video(video_path: str, num_frames: int = NUM_FRAMES):
    """Uniformly sample *num_frames* RGB frames from a video file.

    Reads all non-black frames sequentially (capped at 300) then picks
    *num_frames* evenly spaced indices — identical to training.
    """
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

    # Pad by replicating the first frame if fewer than num_frames
    if 0 < len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[0].copy())

    return frames


# ── Audio extraction from video 
def extract_audio_from_video(video_path: str) -> str | None:
    """Demux audio track to a temporary WAV file using ffmpeg.

    Returns the path to the temporary WAV, or None on failure.
    """
    tmp_wav = os.path.join(
        os.environ.get("TEMP", "/tmp"),
        f"vid_audio_{uuid.uuid4().hex[:8]}.wav",
    )
    ffmpeg_exe = "ffmpeg"
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    cmd = [
        ffmpeg_exe, "-y", "-v", "error",
        "-i", video_path,
        "-t", "30",          # max 30 s (same as training)
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


# ── Main detector class 
class VideoDeepfakeDetector:
    """Multimodal (video + audio) deepfake detector via intermediate fusion.

    This class loads **only** the lightweight XGBoost fusion model.
    Heavy neural networks (CoAtNet, Whisper) are borrowed from the
    Image and Audio detectors that are already resident in memory.
    """

    def __init__(self, xgb_path: str = "Final-Vid/xgboost_fusion_model.joblib"):
        self.classifier = joblib.load(xgb_path)

    # ── public API 
    def predict(self, video_path: str, image_model, audio_model):
        """Run full multimodal inference on a single video.

        Args:
            video_path:   Path to a local video file.
            image_model:  An ``ImageDeepfakeDetector`` instance (provides
                          ``extract_features(rgb_frame)``).
            audio_model:  An ``AudioDeepfakeDetector`` instance (provides
                          ``extract_features(audio_path)``).

        Returns:
            (label: str, confidence: float)
        """
        # ── 1. Visual features 
        frames = extract_frames_from_video(video_path, NUM_FRAMES)
        if len(frames) == 0:
            return "Error: Could not extract frames", 0.0

        frame_embeddings = []
        for frame_rgb in frames:
            emb = image_model.extract_features(frame_rgb)      # (768,)
            frame_embeddings.append(emb)

        frame_embeddings = np.array(frame_embeddings)           # (5, 768)

        # Statistical pooling — identical to training
        feat_mean = frame_embeddings.mean(axis=0)               # (768,)
        feat_max  = frame_embeddings.max(axis=0)                # (768,)
        feat_std  = frame_embeddings.std(axis=0)                # (768,)
        video_vector = np.concatenate([feat_mean, feat_max, feat_std])  # (2304,)

        # ── 2. Audio features
        tmp_wav = extract_audio_from_video(video_path)
        if tmp_wav is None:
            # If audio extraction fails, zero-pad the audio vector
            audio_vector = np.zeros(AUDIO_FEAT_DIM, dtype=np.float32)
        else:
            try:
                audio_vector = audio_model.extract_features(tmp_wav)  # (512,)
            except Exception:
                audio_vector = np.zeros(AUDIO_FEAT_DIM, dtype=np.float32)
            finally:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)

        # ── 3. Feature fusion 
        fused = np.concatenate([video_vector, audio_vector])    # (2816,)
        fused_2d = fused.reshape(1, -1)

        # ── 4. Classification with optimal threshold
        spoof_prob = float(self.classifier.predict_proba(fused_2d)[0, 1])

        if spoof_prob >= OPTIMAL_THRESHOLD:
            return "Fake (Deepfake Detected)", spoof_prob
        else:
            return "Real (Authentic Video)", 1 - spoof_prob
