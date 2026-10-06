import os
from pathlib import Path

# Base workspace directory (where app.py is located)
BASE_DIR = Path(__file__).resolve().parent.parent

def find_file(candidates: list[Path | str]) -> str | None:
    for candidate in candidates:
        p = Path(candidate)
        if p.is_file() and p.exists():
            return str(p.resolve())
    return None

def get_image_model_path() -> str | None:
    candidates = [
        BASE_DIR / "AI-DeepfakeDetector-main" / "AI-DeepfakeDetector-main" / "FinalFrameWork" / "models" / "best_coatnet_5ch.pth",
        BASE_DIR / "FinalFrameWork" / "models" / "best_coatnet_5ch.pth",
        BASE_DIR / "models" / "best_coatnet_5ch.pth",
    ]
    return find_file(candidates)

def get_audio_model_path() -> str | None:
    candidates = [
        BASE_DIR / "AI-DeepfakeDetector-main" / "AI-DeepfakeDetector-main" / "FinalFrameWork" / "models" / "xgboost_asvspoof_model.joblib",
        BASE_DIR / "FinalFrameWork" / "models" / "xgboost_asvspoof_model.joblib",
        BASE_DIR / "models" / "xgboost_asvspoof_model.joblib",
    ]
    return find_file(candidates)

def get_video_model_path() -> str | None:
    candidates = [
        BASE_DIR / "AI-DeepfakeDetector-main" / "AI-DeepfakeDetector-main" / "FinalFrameWork" / "Final-Vid" / "xgboost_fusion_model.joblib",
        BASE_DIR / "FinalFrameWork" / "Final-Vid" / "xgboost_fusion_model.joblib",
        BASE_DIR / "Final-Vid" / "xgboost_fusion_model.joblib",
        BASE_DIR / "models" / "xgboost_fusion_model.joblib",
    ]
    return find_file(candidates)
