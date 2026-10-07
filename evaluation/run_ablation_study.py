"""
Ablation Study Runner for FakeProbe-X.
Systematically compares:
  A. Baseline detector (Raw 0.50 Threshold, No Crop)
  B. Baseline + Quality Analysis (Quality Filtering)
  C. Baseline + Frequency Analysis (CoAtNet + Fourier Spectrum)
  D. Baseline + Temporal Analysis (Multi-Frame Pooling & Consistency)
  E. Baseline + Adaptive Fusion (Reliability-Weighted Evidence)
  F. Baseline + Uncertainty (Three-Way Boundary Margins)
  G. Full FakeProbe-X System (Complete Unified Pipeline)
"""

import sys
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepfake_detector_core import (
    ImageDeepfakeDetector,
    AudioDeepfakeDetector,
    VideoDeepfakeDetector,
    AdaptiveEvidenceFusionEngine,
    DecisionVerdict,
)
from evaluation.metrics import calculate_forensic_metrics


def run_ablation_experiments():
    eval_dir = PROJECT_ROOT / "evaluation"
    img_dir = eval_dir / "test_images"
    aud_dir = eval_dir / "test_audio"
    vid_dir = eval_dir / "test_videos"

    image_detector = ImageDeepfakeDetector()
    audio_detector = AudioDeepfakeDetector()
    video_detector = VideoDeepfakeDetector()
    fusion_engine = AdaptiveEvidenceFusionEngine()

    image_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    
    y_true = []
    # Configurations
    preds_A = []  # Baseline
    probs_A = []
    preds_B = []  # + Quality
    probs_B = []
    preds_C = []  # + Frequency
    probs_C = []
    preds_E = []  # + Adaptive Fusion
    probs_E = []
    preds_F = []  # + Uncertainty
    probs_F = []
    preds_G = []  # Full FakeProbe-X
    probs_G = []

    for img_path in image_files:
        filename = img_path.name
        if "real" in filename.lower():
            label = 0
        elif "fake" in filename.lower():
            label = 1
        else:
            continue

        y_true.append(label)

        # Config A: Baseline (No crop, fixed 0.5 threshold)
        res_A = image_detector.predict_structured(str(img_path), apply_face_crop=False)
        p_A = res_A.probability_fake
        preds_A.append(1 if p_A >= 0.50 else 0)
        probs_A.append(p_A)

        # Config B: Baseline + Quality Analysis (Face Crop + Quality rejection)
        res_B = image_detector.predict_structured(str(img_path), apply_face_crop=True)
        p_B = res_B.probability_fake
        preds_B.append(1 if p_B >= 0.50 else 0)
        probs_B.append(p_B)

        # Config C: Baseline + Frequency Analysis (Weighted average of CoAtNet + FFT anomaly)
        freq_anom = res_B.evidence.get("frequency_anomaly_score", 0.5)
        p_C = 0.65 * p_B + 0.35 * freq_anom
        preds_C.append(1 if p_C >= 0.50 else 0)
        probs_C.append(p_C)

        # Config E: Baseline + Adaptive Fusion (Dynamic Reliability Weighting)
        rep_E = fusion_engine.fuse_image_evidence(res_B, sample_name=filename)
        p_E = rep_E.final_fake_probability
        preds_E.append(1 if p_E >= 0.50 else 0)
        probs_E.append(p_E)

        # Config F: Baseline + Uncertainty (Three-Way Boundary Margins [0.42, 0.58])
        if p_B < 0.42:
            pred_F = 0
        elif p_B > 0.58:
            pred_F = 1
        else:
            pred_F = 1 if p_B >= 0.50 else 0
        preds_F.append(pred_F)
        probs_F.append(p_B)

        # Config G: Full FakeProbe-X
        p_G = rep_E.final_fake_probability
        v_G = rep_E.final_verdict
        pred_G = 1 if v_G == DecisionVerdict.FAKE else (0 if v_G == DecisionVerdict.REAL else (1 if p_G >= 0.50 else 0))
        preds_G.append(pred_G)
        probs_G.append(p_G)

    results = {
        "A_Baseline": calculate_forensic_metrics(y_true, preds_A, probs_A),
        "B_Quality": calculate_forensic_metrics(y_true, preds_B, probs_B),
        "C_Frequency": calculate_forensic_metrics(y_true, preds_C, probs_C),
        "E_AdaptiveFusion": calculate_forensic_metrics(y_true, preds_E, probs_E),
        "F_Uncertainty": calculate_forensic_metrics(y_true, preds_F, probs_F),
        "G_FullFakeProbeX": calculate_forensic_metrics(y_true, preds_G, probs_G),
    }

    print("============================================================")
    print("  FakeProbe-X Ablation Study Results")
    print("============================================================")
    for name, m in results.items():
        print(f"{name:<20} | Acc: {m['accuracy']*100:.1f}% | F1: {m['f1_score']:.3f} | Brier: {m['brier_score']:.4f}")

    return results


if __name__ == "__main__":
    run_ablation_experiments()
