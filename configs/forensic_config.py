"""
Centralized Configuration for FakeProbe-X Forensics System.
All paths are relative to project root. No hardcoded absolute user paths.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

# Base workspace directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ForensicConfig:
    # --- Device & Execution ---
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1" else "cpu"
    num_workers: int = 2
    random_seed: int = 42

    # --- Model Weights Paths (Relative to PROJECT_ROOT) ---
    image_model_rel_path: str = "AI-DeepfakeDetector-main/AI-DeepfakeDetector-main/FinalFrameWork/models/best_coatnet_5ch.pth"
    audio_model_rel_path: str = "AI-DeepfakeDetector-main/AI-DeepfakeDetector-main/FinalFrameWork/models/xgboost_asvspoof_model.joblib"
    video_fusion_model_rel_path: str = "AI-DeepfakeDetector-main/AI-DeepfakeDetector-main/FinalFrameWork/Final-Vid/xgboost_fusion_model.joblib"
    video_config_rel_path: str = "AI-DeepfakeDetector-main/AI-DeepfakeDetector-main/FinalFrameWork/Final-Vid/inference_config.json"

    # --- Image Processing Parameters ---
    image_input_size: int = 224
    ela_quality: int = 90
    use_face_crop: bool = True
    face_crop_margin: float = 0.25  # 25% margin around detected face bbox

    # Normalization constants (ImageNet for RGB, 0.5 for ELA & FFT)
    rgb_mean: tuple = (0.485, 0.456, 0.406)
    rgb_std: tuple = (0.229, 0.224, 0.225)
    forensic_mean: tuple = (0.5, 0.5)
    forensic_std: tuple = (0.5, 0.5)

    # --- Audio Processing Parameters ---
    audio_sample_rate: int = 16000
    whisper_model_name: str = "openai/whisper-base"
    audio_max_duration_sec: float = 30.0

    # --- Video Processing Parameters ---
    default_num_frames: int = 5
    max_scan_frames: int = 300
    video_pooling: str = "statistical"  # mean + max + std -> 2304 dim
    coatnet_embedding_dim: int = 768
    whisper_embedding_dim: int = 512
    fused_embedding_dim: int = 2816

    # --- Baseline Model Thresholds ---
    image_decision_threshold: float = 0.5000
    audio_decision_threshold: float = 0.5000
    video_decision_threshold: float = 0.5780

    # --- Three-Way Decision & Uncertainty Boundaries ---
    # Score < real_boundary -> REAL
    # Score > fake_boundary -> FAKE
    # real_boundary <= Score <= fake_boundary -> UNCERTAIN
    uncertainty_real_boundary: float = 0.4200
    uncertainty_fake_boundary: float = 0.5800
    min_reliability_threshold: float = 0.3500
    high_uncertainty_threshold: float = 0.6000

    # --- Input Quality Thresholds ---
    min_image_resolution: tuple = (128, 128)
    blur_laplacian_threshold_low: float = 40.0   # below this is blurry
    blur_laplacian_threshold_good: float = 100.0 # above this is sharp
    min_face_size_ratio: float = 0.05            # face area relative to image area
    min_audio_duration_sec: float = 0.50
    audio_clipping_threshold: float = 0.05       # >5% clipped samples indicates poor quality
    audio_silence_threshold_db: float = -40.0

    # --- Reliability Weighting Constants ---
    quality_weight: float = 0.30
    confidence_weight: float = 0.30
    agreement_weight: float = 0.25
    coverage_weight: float = 0.15

    # --- Adaptive Fusion Prior Weights ---
    fusion_weight_image_detector: float = 0.40
    fusion_weight_audio_detector: float = 0.30
    fusion_weight_frequency_forensic: float = 0.15
    fusion_weight_temporal_consistency: float = 0.15

    # --- Security & Resource Limits ---
    max_image_upload_mb: int = 25
    max_video_upload_mb: int = 150
    max_audio_upload_mb: int = 50
    max_video_duration_seconds: int = 120

    def get_image_model_path(self) -> Path:
        return PROJECT_ROOT / self.image_model_rel_path

    def get_audio_model_path(self) -> Path:
        return PROJECT_ROOT / self.audio_model_rel_path

    def get_video_model_path(self) -> Path:
        return PROJECT_ROOT / self.video_fusion_model_rel_path


# Global default configuration instance
DEFAULT_CONFIG = ForensicConfig()
