"""
Model and Checkpoint Path Utilities for FakeProbe-X.
Resolves relative model paths dynamically across workspace layouts.
"""

import os
from pathlib import Path
from configs.forensic_config import PROJECT_ROOT, DEFAULT_CONFIG


def find_file(candidates: list[Path | str]) -> str | None:
    for candidate in candidates:
        p = Path(candidate)
        if p.is_file() and p.exists():
            return str(p.resolve())
    return None


def get_image_model_path() -> str | None:
    candidates = [
        DEFAULT_CONFIG.get_image_model_path(),
        PROJECT_ROOT / "AI-DeepfakeDetector-main" / "AI-DeepfakeDetector-main" / "FinalFrameWork" / "models" / "best_coatnet_5ch.pth",
        PROJECT_ROOT / "FinalFrameWork" / "models" / "best_coatnet_5ch.pth",
        PROJECT_ROOT / "models" / "best_coatnet_5ch.pth",
    ]
    return find_file(candidates)


def get_audio_model_path() -> str | None:
    candidates = [
        DEFAULT_CONFIG.get_audio_model_path(),
        PROJECT_ROOT / "AI-DeepfakeDetector-main" / "AI-DeepfakeDetector-main" / "FinalFrameWork" / "models" / "xgboost_asvspoof_model.joblib",
        PROJECT_ROOT / "FinalFrameWork" / "models" / "xgboost_asvspoof_model.joblib",
        PROJECT_ROOT / "models" / "xgboost_asvspoof_model.joblib",
    ]
    return find_file(candidates)


def get_video_model_path() -> str | None:
    candidates = [
        DEFAULT_CONFIG.get_video_model_path(),
        PROJECT_ROOT / "AI-DeepfakeDetector-main" / "AI-DeepfakeDetector-main" / "FinalFrameWork" / "Final-Vid" / "xgboost_fusion_model.joblib",
        PROJECT_ROOT / "FinalFrameWork" / "Final-Vid" / "xgboost_fusion_model.joblib",
        PROJECT_ROOT / "Final-Vid" / "xgboost_fusion_model.joblib",
        PROJECT_ROOT / "models" / "xgboost_fusion_model.joblib",
    ]
    return find_file(candidates)
