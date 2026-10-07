"""
Comprehensive Evaluation Harness for FakeProbe-X.
Evaluates baseline and reliability-aware pipelines across Image, Audio, and Video test samples.
Outputs verified metric tables, confusion matrices, and calibration scores.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
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


def evaluate_image_modality(img_dir: Path, image_detector, fusion_engine) -> dict:
    """Evaluate Image detector under Baseline (direct resize, 0.5 threshold) vs FakeProbe-X (face crop + fusion)."""
    image_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    
    y_true = []
    baseline_preds = []
    baseline_probs = []
    fakeprobe_preds = []
    fakeprobe_probs = []
    fakeprobe_verdicts = []

    for img_path in image_files:
        filename = img_path.name
        if "real" in filename.lower():
            label = 0
        elif "fake" in filename.lower():
            label = 1
        else:
            continue  # skip unlabelled degraded test files from strict binary evaluation

        y_true.append(label)

        # 1. Baseline Evaluation (No face crop, raw 0.50 threshold)
        res_baseline = image_detector.predict_structured(str(img_path), apply_face_crop=False)
        base_prob = res_baseline.probability_fake
        base_pred = 1 if base_prob >= 0.50 else 0
        baseline_preds.append(base_pred)
        baseline_probs.append(base_prob)

        # 2. FakeProbe-X Evaluation (Face crop + Adaptive evidence fusion + Three-way engine)
        res_structured = image_detector.predict_structured(str(img_path), apply_face_crop=True)
        report = fusion_engine.fuse_image_evidence(res_structured, sample_name=filename)
        
        fp_prob = report.final_fake_probability
        fakeprobe_probs.append(fp_prob)
        fakeprobe_verdicts.append(report.final_verdict.value)
        # For binary metrics mapping: FAKE -> 1, REAL -> 0, UNCERTAIN -> mapped to majority/threshold
        fp_pred = 1 if report.final_verdict == DecisionVerdict.FAKE else (0 if report.final_verdict == DecisionVerdict.REAL else (1 if fp_prob >= 0.50 else 0))
        fakeprobe_preds.append(fp_pred)

    baseline_metrics = calculate_forensic_metrics(y_true, baseline_preds, baseline_probs)
    fakeprobe_metrics = calculate_forensic_metrics(y_true, fakeprobe_preds, fakeprobe_probs)

    return {
        "modality": "image",
        "sample_count": len(y_true),
        "baseline": baseline_metrics,
        "fakeprobe_x": fakeprobe_metrics,
        "verdicts": fakeprobe_verdicts,
    }


def evaluate_audio_modality(aud_dir: Path, audio_detector, fusion_engine) -> dict:
    """Evaluate Audio detector."""
    audio_files = sorted(list(aud_dir.glob("*.wav")) + list(aud_dir.glob("*.flac")))

    y_true = []
    preds = []
    probs = []
    verdicts = []

    for aud_path in audio_files:
        filename = aud_path.name
        if "real" in filename.lower():
            label = 0
        elif "fake" in filename.lower():
            label = 1
        else:
            continue

        y_true.append(label)
        res = audio_detector.predict_structured(str(aud_path))
        report = fusion_engine.fuse_audio_evidence(res, sample_name=filename)

        prob = report.final_fake_probability
        probs.append(prob)
        verdicts.append(report.final_verdict.value)
        pred = 1 if report.final_verdict == DecisionVerdict.FAKE else (0 if report.final_verdict == DecisionVerdict.REAL else (1 if prob >= 0.50 else 0))
        preds.append(pred)

    metrics = calculate_forensic_metrics(y_true, preds, probs)
    return {
        "modality": "audio",
        "sample_count": len(y_true),
        "metrics": metrics,
        "verdicts": verdicts,
    }


def evaluate_video_modality(vid_dir: Path, video_detector, image_detector, audio_detector, fusion_engine) -> dict:
    """Evaluate Video detector."""
    video_files = sorted(list(vid_dir.glob("*.mp4")) + list(vid_dir.glob("*.avi")))

    y_true = []
    preds = []
    probs = []
    verdicts = []

    for vid_path in video_files:
        filename = vid_path.name
        if "real" in filename.lower():
            label = 0
        elif "fake" in filename.lower():
            label = 1
        else:
            continue

        y_true.append(label)
        res = video_detector.predict_structured(str(vid_path), image_detector, audio_detector, num_frames=5)
        report = fusion_engine.fuse_video_evidence(res, sample_name=filename)

        prob = report.final_fake_probability
        probs.append(prob)
        verdicts.append(report.final_verdict.value)
        pred = 1 if report.final_verdict == DecisionVerdict.FAKE else (0 if report.final_verdict == DecisionVerdict.REAL else (1 if prob >= 0.50 else 0))
        preds.append(pred)

    metrics = calculate_forensic_metrics(y_true, preds, probs)
    return {
        "modality": "video",
        "sample_count": len(y_true),
        "metrics": metrics,
        "verdicts": verdicts,
    }


def run_full_benchmark():
    eval_dir = PROJECT_ROOT / "evaluation"
    img_dir = eval_dir / "test_images"
    aud_dir = eval_dir / "test_audio"
    vid_dir = eval_dir / "test_videos"

    print("============================================================")
    print("  FakeProbe-X Comprehensive Evaluation Benchmark")
    print("============================================================")

    image_detector = ImageDeepfakeDetector()
    audio_detector = AudioDeepfakeDetector()
    video_detector = VideoDeepfakeDetector()
    fusion_engine = AdaptiveEvidenceFusionEngine()

    # Run modalities
    img_results = evaluate_image_modality(img_dir, image_detector, fusion_engine)
    aud_results = evaluate_audio_modality(aud_dir, audio_detector, fusion_engine)
    vid_results = evaluate_video_modality(vid_dir, video_detector, image_detector, audio_detector, fusion_engine)

    results_summary = {
        "image": img_results,
        "audio": aud_results,
        "video": vid_results,
    }

    # Print summary tables
    print("\n--- 1. IMAGE FORENSICS (CoAtNet 5-Ch) ---")
    print(f"Sample Count : {img_results['sample_count']} (Real: {img_results['baseline']['real_count']}, Fake: {img_results['baseline']['fake_count']})")
    print(f"Baseline Acc : {img_results['baseline']['accuracy']*100:.1f}% | F1: {img_results['baseline']['f1_score']:.3f} | AUC: {img_results['baseline']['roc_auc']}")
    print(f"FakeProbe Acc: {img_results['fakeprobe_x']['accuracy']*100:.1f}% | F1: {img_results['fakeprobe_x']['f1_score']:.3f} | AUC: {img_results['fakeprobe_x']['roc_auc']}")
    print(f"Baseline Confusion  : TN={img_results['baseline']['tn']}, FP={img_results['baseline']['fp']}, FN={img_results['baseline']['fn']}, TP={img_results['baseline']['tp']}")
    print(f"FakeProbe Confusion : TN={img_results['fakeprobe_x']['tn']}, FP={img_results['fakeprobe_x']['fp']}, FN={img_results['fakeprobe_x']['fn']}, TP={img_results['fakeprobe_x']['tp']}")

    print("\n--- 2. AUDIO FORENSICS (Whisper + XGBoost) ---")
    print(f"Sample Count : {aud_results['sample_count']} (Real: {aud_results['metrics']['real_count']}, Fake: {aud_results['metrics']['fake_count']})")
    print(f"Accuracy     : {aud_results['metrics']['accuracy']*100:.1f}% | F1: {aud_results['metrics']['f1_score']:.3f} | AUC: {aud_results['metrics']['roc_auc']}")
    print(f"Confusion    : TN={aud_results['metrics']['tn']}, FP={aud_results['metrics']['fp']}, FN={aud_results['metrics']['fn']}, TP={aud_results['metrics']['tp']}")

    print("\n--- 3. VIDEO FORENSICS (Intermediate Fusion) ---")
    print(f"Sample Count : {vid_results['sample_count']} (Real: {vid_results['metrics']['real_count']}, Fake: {vid_results['metrics']['fake_count']})")
    print(f"Accuracy     : {vid_results['metrics']['accuracy']*100:.1f}% | F1: {vid_results['metrics']['f1_score']:.3f} | AUC: {vid_results['metrics']['roc_auc']}")
    print(f"Confusion    : TN={vid_results['metrics']['tn']}, FP={vid_results['metrics']['fp']}, FN={vid_results['metrics']['fn']}, TP={vid_results['metrics']['tp']}")

    # Save JSON benchmark report
    out_json = eval_dir / "benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[SUCCESS] Benchmark complete. Results written to {out_json}")
    return results_summary


if __name__ == "__main__":
    run_full_benchmark()
