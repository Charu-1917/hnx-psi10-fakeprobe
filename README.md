# FakeProbe-X: Reliability-Aware Multimodal Deepfake Forensics System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Branch: deepfake-integration](https://img.shields.io/badge/Branch-deepfake--integration-green.svg)]()

FakeProbe-X is an explainable, reliability-aware multimodal deepfake forensics framework that combines spatial residual convolutional architectures, 2D Fourier frequency domain analysis, acoustic speech embeddings, and temporal multi-frame consistency tracking into an adaptive evidence fusion engine.

---

## 1. Problem Statement & Motivation

Existing deepfake detection systems often force a binary decision ($P > 0.50 \implies \text{FAKE}$) on neural network activations without quantifying the physical quality of the input signal or measuring epistemic uncertainty. Consequently, when presented with:
- Heavy JPEG compression or motion blur,
- Full-scene photographs with non-facial backgrounds,
- Low-volume or clipped audio signals,
- Videos missing audio tracks,

traditional detectors experience severe false positive alarms, flagging genuine real media as deepfakes.

**FakeProbe-X** addresses this limitation through a **Reliability-Aware Evidence Fusion Architecture** that evaluates signal quality, cross-modal consensus, and uncertainty margins before rendering a legally and technically defensible three-way verdict (**REAL**, **FAKE**, or **UNCERTAIN**).

---

## 2. System Architecture & Modalities

```
                                  USER INPUT
                        (Image / Video / Audio Media)
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    INPUT VALIDATION &     │
                        │    SECURITY SANITIZATION  │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   INPUT QUALITY ANALYZER  │
                        │  (Blur, Lum, SNR, Faces)  │
                        └─────────────┬─────────────┘
                                      │
               ┌──────────────────────┼──────────────────────┐
               ▼                      ▼                      ▼
      ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
      │  IMAGE PIPELINE │    │  VIDEO PIPELINE │    │  AUDIO PIPELINE │
      │  • MTCNN Face   │    │  • Multi-Frame  │    │  • 16kHz Resamp │
      │  • 5-Ch CoAtNet │    │  • Face Track   │    │  • Whisper Enc  │
      │  • ELA Residual │    │  • Temp Pooling │    │  • XGBoost ASV  │
      │  • 2D FFT Spec  │    │  • XGB Fusion   │    │  • SNR / Energy │
      └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
               │                      │                      │
               └──────────────────────┼──────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │  FORENSIC EVIDENCE POOL   │
                        │  (Scores, Quality, Maps)  │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    MODEL AGREEMENT &      │
                        │   RELIABILITY ESTIMATOR   │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ ADAPTIVE EVIDENCE FUSION  │
                        │   weighted by Reliability │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ THREE-WAY DECISION ENGINE │
                        │  (REAL / FAKE / UNCERTAIN)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   EXPLAINABLE FORENSIC    │
                        │      AUDIT REPORT         │
                        └───────────────────────────┘
```

---

## 3. Core Technical Novelties

1. **Reliability-Aware Evidence Weighting**:
   $$\text{fused\_score} = \frac{\sum_i w_i \cdot r_i \cdot s_i}{\sum_i w_i \cdot r_i}$$
   where each evidence stream $i$ is dynamically scaled by its real-time signal reliability $r_i \in [0, 1]$.
2. **Three-Way Decision Engine**:
   - `FAKE`: $\text{fused\_score} \ge 0.58$ and $\text{reliability} \ge 0.35$.
   - `REAL`: $\text{fused\_score} \le 0.42$ and $\text{reliability} \ge 0.35$.
   - `UNCERTAIN`: Triggered when uncertainty $\ge 0.60$, reliability $< 0.35$, or score falls in deadband $[0.42, 0.58]$.
3. **Input Quality Awareness**: Rejects or downweights blurry ($< 40$ Laplacian variance), clipped, or underexposed media.
4. **Frequency-Domain Spectral Analysis**: Orthogonal 2D Fast Fourier log-magnitude spectra and radial power decay slope.
5. **Temporal Timeline & Frame Localization**: Maps per-frame fake probabilities across video sequences with suspicious timestamp logging.
6. **Asymmetric Modality Adaptation**: Gracefully handles silent videos without zero-padding bias.
7. **Explainable Forensic Reports**: Produces downloadable audit certificates with itemized reasoning.

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10 - 3.13
- FFmpeg (bundled automatically via `imageio-ffmpeg`)

```bash
git checkout deepfake-integration
pip install -r requirements.txt
```

---

## 5. Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```
Access the application at: `http://localhost:8501`

---

## 6. Running Tests and Evaluation

### Run Unit and Critical Scenarios Suite (14 Tests)
```bash
python tests/test_critical_scenarios.py
python tests/test_preprocessing.py
python tests/test_quality_analyzer.py
python tests/test_frequency_analyzer.py
python tests/test_reliability_decision.py
python tests/test_security_performance.py
```

### Run Full Benchmark Evaluation
```bash
python evaluation/run_evaluation.py
```

### Run Ablation Study
```bash
python evaluation/run_ablation_study.py
```

---

## 7. Performance Benchmarks

| Pipeline | Modality | Inference Latency | Primary Backbone | Checkpoint |
|:---|:---|:---:|:---|:---|
| **Image Forensics** | Image | **225.8 ms** / sample | CoAtNet-0 5-Ch (RGB+ELA+FFT) | `best_coatnet_5ch.pth` |
| **Audio Forensics** | Audio | **679.6 ms** / sample | Whisper-Base + XGBoost | `xgboost_asvspoof_model.joblib` |
| **Video Forensics** | Video | **1723.4 ms** / sample | Multimodal XGBoost (2816-dim) | `xgboost_fusion_model.joblib` |

---

## 8. Repository Documentation Directory

- [`docs/BASELINE_AUDIT.md`](docs/BASELINE_AUDIT.md): Exhaustive audit of all repository projects and checkpoints.
- [`docs/CURRENT_PIPELINE.md`](docs/CURRENT_PIPELINE.md): Step-by-step trace of image, audio, and video pipelines.
- [`docs/ERRORS_FIXED.md`](docs/ERRORS_FIXED.md): Root cause analysis and fixes for baseline issues.
- [`docs/BASELINE_RESULTS.md`](docs/BASELINE_RESULTS.md): Baseline performance tables.
- [`docs/ABLATION_STUDY.md`](docs/ABLATION_STUDY.md): 7-stage ablation study with measured Brier calibration scores.
- [`docs/DATASETS.md`](docs/DATASETS.md): Documentation of upstream training/eval datasets.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): System architecture and dataflow specifications.
- [`docs/NOVELTY.md`](docs/NOVELTY.md): Summary of technical innovations.

---

## 9. Known Limitations & Future Work

- **Lip-Sync Audio-Visual Consistency**: Currently provides cross-modal score agreement; fine-grained neural lip-motion synchronization (e.g. SyncNet) requires additional dedicated weights.
- **Model Checkpoints for EfficientNet**: EfficientNet-B1 training code is included in `multimodal-deepfake-detector-master` but weights were not provided in repository.
- **Future Enhancements**: Integration of spatial diffusion-specific noise footprint extractors and transformer cross-attention temporal models.
