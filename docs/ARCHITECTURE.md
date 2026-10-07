# FakeProbe-X System Architecture (ARCHITECTURE.md)

This document diagrams and explains the complete end-to-end execution flow of the **FakeProbe-X** reliability-aware deepfake forensics framework.

---

## 1. High-Level System Architecture

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
      │  • Face Detect  │    │  • Multi-Frame  │    │  • 16kHz Resamp │
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
                        │      REPORT & AUDIT       │
                        └───────────────────────────┘
```

---

## 2. Component Specifications

### 1. Input Validation & Security Sanitization
- **Path Sanitization**: Resolves relative filepaths, validates file header magic bytes, and blocks path traversal attempts (`../../`).
- **Resource Limiting**: Limits memory allocation and file upload buffers (Image: 25MB, Audio: 50MB, Video: 150MB, max scan 300 frames).

### 2. Input Quality Analyzer (`quality_analyzer.py`)
- **Image**: Measures Laplacian variance ($\sigma^2_{\nabla^2}$), luminance mean/std, and face-to-image area ratio.
- **Audio**: Measures duration, clipping percentage ($|x| \ge 0.99$), and silence/SNR ratio below $-40$ dB.
- **Video**: Measures decoding FPS, resolution, face tracking coverage, and per-frame sharpness.

### 3. Spatial & Frequency Modality Detectors
- **CoAtNet-0 5-Channel**: Computes Error Level Analysis (JPEG quality 90) and 2D FFT magnitude spectrum, concatenated with normalized RGB into a 5-channel tensor `(1, 5, 224, 224)`.
- **Whisper-Base + XGBoost**: Computes 80-channel log-mel spectrograms, extracts 512-dim mean-pooled encoder embeddings, and applies an ASVspoof-trained XGBoost classifier.
- **Intermediate Fusion Video Detector**: Samples 5 uniform frames, extracts 768-dim CoAtNet embeddings with statistical pooling (`mean`, `max`, `std` = 2304-dim), extracts Whisper audio embedding (512-dim), and classifies via XGBoost.

### 4. Reliability Estimator (`reliability.py`)
- Evaluates composite reliability:
  $$\text{reliability} = w_q \cdot \text{quality} + w_c \cdot \text{confidence} + w_a \cdot \text{agreement} + w_{\text{cov}} \cdot \text{coverage}$$
- Quantifies heuristic uncertainty $u \in [0, 1]$ as the harmonic inverse of confidence, signal quality, and detector consensus.

### 5. Adaptive Evidence Fusion Engine (`fusion_engine.py`)
- Dynamically calculates the fused spoof probability:
  $$\text{fused\_score} = \frac{\sum_i w_i \cdot r_i \cdot s_i}{\sum_i w_i \cdot r_i}$$
- Adapts dynamically to missing modalities without zero-padding bias.

### 6. Three-Way Decision Engine
- **`FAKE`**: $\text{fused\_score} \ge 0.58$ and $\text{reliability} \ge 0.35$.
- **`REAL`**: $\text{fused\_score} \le 0.42$ and $\text{reliability} \ge 0.35$.
- **`UNCERTAIN`**: Triggered if uncertainty $\ge 0.60$, reliability $< 0.35$, score in deadband $[0.42, 0.58]$, or input quality is degraded.
