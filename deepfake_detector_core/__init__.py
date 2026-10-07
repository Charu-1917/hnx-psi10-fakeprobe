"""
FakeProbe-X Deepfake Detector Core Library.
Reliability-Aware Multimodal Deepfake Forensics System.
"""

from .types import (
    DecisionVerdict,
    QualityLevel,
    ReliabilityLevel,
    QualityResult,
    ForensicEvidence,
    DetectorResult,
    UnifiedForensicReport,
)
from .image_detector import (
    ImageDeepfakeDetector,
    compute_ela,
    compute_fft_magnitude,
    prepare_5ch_tensor,
    build_coatnet_5ch,
)
from .audio_detector import AudioDeepfakeDetector
from .video_detector import (
    VideoDeepfakeDetector,
    extract_frames_with_metadata,
    extract_audio_from_video,
    transcode_for_browser,
)
from .quality_analyzer import InputQualityAnalyzer
from .frequency_analyzer import FrequencyForensicAnalyzer
from .face_utils import FaceDetector
from .reliability import ReliabilityEstimator
from .fusion_engine import AdaptiveEvidenceFusionEngine
from .report_generator import ForensicReportGenerator
from .path_utils import (
    get_image_model_path,
    get_audio_model_path,
    get_video_model_path,
)

__version__ = "2.0.0"

__all__ = [
    "DecisionVerdict",
    "QualityLevel",
    "ReliabilityLevel",
    "QualityResult",
    "ForensicEvidence",
    "DetectorResult",
    "UnifiedForensicReport",
    "ImageDeepfakeDetector",
    "AudioDeepfakeDetector",
    "VideoDeepfakeDetector",
    "InputQualityAnalyzer",
    "FrequencyForensicAnalyzer",
    "FaceDetector",
    "ReliabilityEstimator",
    "AdaptiveEvidenceFusionEngine",
    "ForensicReportGenerator",
    "compute_ela",
    "compute_fft_magnitude",
    "prepare_5ch_tensor",
    "build_coatnet_5ch",
    "extract_frames_with_metadata",
    "extract_audio_from_video",
    "transcode_for_browser",
    "get_image_model_path",
    "get_audio_model_path",
    "get_video_model_path",
]
