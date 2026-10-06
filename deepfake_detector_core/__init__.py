"""
Deepfake Detector Integrated Core Package
Combines Image (CoAtNet 5-Channel), Audio (Whisper + XGBoost), and Video (Multimodal Fusion) Detectors.
"""

from .image_detector import ImageDeepfakeDetector
from .audio_detector import AudioDeepfakeDetector
from .video_detector import VideoDeepfakeDetector, transcode_for_browser
from .path_utils import get_image_model_path, get_audio_model_path, get_video_model_path

__all__ = [
    "ImageDeepfakeDetector",
    "AudioDeepfakeDetector",
    "VideoDeepfakeDetector",
    "transcode_for_browser",
    "get_image_model_path",
    "get_audio_model_path",
    "get_video_model_path",
]
