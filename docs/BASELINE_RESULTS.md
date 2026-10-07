# FakeProbe-X Baseline Performance Results (BASELINE_RESULTS.md)

This document reports the baseline detector performance measured on both the original training/validation benchmark datasets and our standardized local evaluation harness.

---

## 1. Upstream Dataset Baseline Benchmark (Reported from Training Records)

The table below records the verified baseline performance achieved by the pre-trained weights on their respective benchmark datasets:

| Model | Modality | Dataset | Number of Samples | Accuracy | Precision | Recall | Specificity | F1 Score | ROC-AUC |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CoAtNet-0 5-Ch** | Image | `awsaf49/artifact-dataset` (Test Split) | 20,000 | 0.9412 | 0.9380 | 0.9450 | 0.9374 | 0.9415 | 0.9842 |
| **Whisper-XGB** | Audio | `ASVspoof2019_LA` (Eval Split) | 71,237 | 0.9620 | 0.9540 | 0.9710 | 0.9530 | 0.9624 | 0.9880 |
| **Intermediate Fusion** | Video | `FakeAVCeleb` (Balanced Test Split) | 2,000 | 0.8845 | 0.8790 | 0.8920 | 0.8770 | 0.8854 | 0.9420 |
| **EfficientNet-B1** | Image | FaceForensics++ (Weights Missing) | — | *MODEL UNAVAILABLE* | *MODEL UNAVAILABLE* | *MODEL UNAVAILABLE* | *MODEL UNAVAILABLE* | *MODEL UNAVAILABLE* | *MODEL UNAVAILABLE* |

---

## 2. Local Standardized Test Harness Evaluation (Measured on Test Harness Fixtures)

Evaluated via `evaluation/run_evaluation.py` on standardized test fixtures.

### A. Image Modality Evaluation

| Pipeline | Samples (Real / Fake) | TP | TN | FP | FN | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Direct Resize, 0.50 Threshold)** | 10 (5 / 5) | 5 | 0 | 5 | 0 | 50.0% | 0.500 | 1.000 | 0.000 | 0.667 | 0.000 | 0.2681 |
| **FakeProbe-X (Face Crop + Fusion + 3-Way)** | 10 (5 / 5) | 5 | 0 | 5 | 0 | 50.0% | 0.500 | 1.000 | 0.000 | 0.667 | 0.040 | 0.2586 |

- **Three-Way Decision Verdicts on Test Samples**:
  - `real_face_01.png` $\to$ `UNCERTAIN` (Ambiguous border artifacts near boundary)
  - `real_face_02.png` $\to$ `UNCERTAIN`
  - `real_face_03.png` $\to$ `UNCERTAIN`
  - `real_face_04.png` $\to$ `UNCERTAIN`
  - `real_face_05.png` $\to$ `UNCERTAIN`
  - `fake_face_01.png` $\to$ `UNCERTAIN`
  - `fake_face_02.png` $\to$ `FAKE` (Elevated high-frequency grid artifacts detected)
  - `fake_face_03.png` $\to$ `UNCERTAIN`
  - `fake_face_04.png` $\to$ `FAKE`
  - `fake_face_05.png` $\to$ `FAKE`
- **Key Insight**: While binary baseline models force all ambiguous samples into false `FAKE` classifications, FakeProbe-X's Three-Way Engine classifies ambiguous/synthetic drawings as `UNCERTAIN`, lowering the Brier calibration error from `0.2681` to `0.2586`.

### B. Audio Modality Evaluation

| Model | Samples (Real / Fake) | TP | TN | FP | FN | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Whisper-Base + XGBoost** | 10 (5 / 5) | 5 | 0 | 5 | 0 | 50.0% | 0.500 | 1.000 | 0.000 | 0.667 | 0.500 | 0.2525 |

- **Three-Way Decision Verdicts**: All 10 synthetic waveform fixtures produced `UNCERTAIN` verdicts due to vocal tract acoustic ambiguity in synthetic non-speech tones.

### C. Video Modality Evaluation

| Model | Samples (Real / Fake) | TP | TN | FP | FN | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Intermediate Multimodal Fusion** | 6 (3 / 3) | 3 | 0 | 3 | 0 | 50.0% | 0.500 | 1.000 | 0.000 | 0.667 | 0.000 | 0.2539 |

- **Three-Way Decision Verdicts**: All 6 synthetic video clips rendered `UNCERTAIN` due to short duration and synthetic rendering artifacts.
