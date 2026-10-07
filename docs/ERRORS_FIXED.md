# FakeProbe-X Error and Weakness Diagnosis (ERRORS_FIXED.md)

This document tracks all diagnosed errors, weaknesses, root causes, and validation statuses across the repository.

---

## 1. Summary of Issues

| ID | Issue | Category | Affected File(s) | Status |
|---|---|---|---|---|
| **BUG-01** | Real images misclassified as FAKE (REAL $\to$ FAKE false positives) | CONFIRMED | `deepfake_detector_core/image_detector.py` | Diagnosed / Pending Phase 5 |
| **BUG-02** | Absence of face detection/cropping causing background artifact distortion | CONFIRMED | `deepfake_detector_core/image_detector.py`, `video_detector.py` | Diagnosed / Pending Phase 5 |
| **BUG-03** | Silent zero-padding of audio features in video detector when audio track is absent | CONFIRMED | `deepfake_detector_core/video_detector.py` | Diagnosed / Pending Phase 5 |
| **BUG-04** | Rigid binary thresholding with no uncertainty buffer for borderline predictions | CONFIRMED | `deepfake_detector_core/image_detector.py`, `audio_detector.py`, `video_detector.py` | Diagnosed / Pending Phase 5 |
| **BUG-05** | Lack of Input Quality Analysis (blur, resolution, contrast, compression) | CONFIRMED | `deepfake_detector_core/*` | Diagnosed / Pending Phase 5 |
| **BUG-06** | Non-standardized return signatures across detectors | CONFIRMED | `deepfake_detector_core/*` | Diagnosed / Pending Phase 5 |
| **BUG-07** | Missing pre-trained weights in `multimodal-deepfake-detector-master` | CONFIRMED | `multimodal-deepfake-detector-master/` | Documented (MODEL UNAVAILABLE) |
| **BUG-08** | OpenCV coordinate indexing & color space validation | CONFIRMED | `deepfake_detector_core/*` | Diagnosed / Pending Phase 5 |

---

## 2. Detailed Bug Reports & Root Cause Analyses

### BUG-01 & BUG-02: REAL $\to$ FAKE False Positive & Face Localization Mismatch

- **ID**: `BUG-01` & `BUG-02`
- **Classification**: CONFIRMED
- **File**: `deepfake_detector_core/image_detector.py`, `app.py`
- **Function/Class**: `ImageDeepfakeDetector.predict()`, `prepare_5ch_tensor()`
- **Current Behavior**:
  - Unedited natural images, flat test patterns, or full-scene photos produce $\sigma(z) \in [0.505, 0.650]$, triggering a `FAKE` verdict.
- **Expected Behavior**:
  - Authentic photos should be classified as `REAL`.
  - Poor-quality or ambiguous inputs without clear facial forensic features should produce `UNCERTAIN` rather than false `FAKE`.
- **Root Cause**:
  1. **Training Distribution Mismatch**: CoAtNet-5ch was trained on `awsaf49/artifact-dataset`, which consists exclusively of cropped facial squares. When an uncropped full-scene photo is squashed to 224x224, background textures and high-frequency scene noise produce elevated ELA residuals and abnormal FFT power spectra.
  2. **Zero-Margin Threshold**: The model threshold was hard-coded at exactly `0.5000`. Predictions near 0.50 (e.g. 0.51) reflect high uncertainty/out-of-distribution responses rather than high-confidence deepfake artifacts.
  3. **Lack of Face Localization**: Without a face detector, whole-scene images are evaluated directly against a face-specialized model.
- **Fix Required**:
  1. Add automatic face detection (with OpenCV Haar / MTCNN) and face-focused feature extraction with smooth aspect-ratio preservation.
  2. Implement an Input Quality Analyzer to assess blur, lighting, contrast, and face presence.
  3. Implement a three-way decision engine (`REAL`, `FAKE`, `UNCERTAIN`) with adaptive reliability bounds ($[0.42, 0.58]$ uncertainty deadband).
  4. Compute independent frequency-domain statistical metrics (spectral roll-off, high-frequency energy ratio) to validate model outputs.
- **Validation Method**: Evaluate on known real photographic portraits, full-scene photos, and synthetic manipulated samples.
- **Status**: CONFIRMED (Fix to be implemented in Phase 5 & 8).

---

### BUG-03: Silent Zero-Padding in Video Multimodal Classifier

- **ID**: `BUG-03`
- **Classification**: CONFIRMED
- **File**: `deepfake_detector_core/video_detector.py`
- **Function/Class**: `VideoDeepfakeDetector.predict()`
- **Current Behavior**:
  - When video audio is missing, a 512-dim vector of zeros is concatenated with the 2304-dim visual vector.
- **Expected Behavior**:
  - When audio is unavailable, the system should adaptively use visual and temporal frame evidence alone, explicitly flagging `audio_available = False` and scaling reliability accordingly.
- **Root Cause**:
  - Hard-coded concatenation `np.concatenate([video_vector, audio_vector])` feeding into an XGBoost model trained on non-zero speech embeddings.
- **Fix Required**:
  - Implement Reliability-Aware Adaptive Fusion that checks modality availability.
  - If audio is absent, perform frame-level temporal aggregation and adjust overall multimodal confidence and reliability scores.
- **Validation Method**: Test silent video vs video with audio.
- **Status**: CONFIRMED.

---

### BUG-04: Rigid Binary Decisions Without Uncertainty Quantification

- **ID**: `BUG-04`
- **Classification**: CONFIRMED
- **File**: `deepfake_detector_core/image_detector.py`, `audio_detector.py`, `video_detector.py`
- **Function/Class**: `predict()`
- **Current Behavior**:
  - Direct step function: $\text{prob} > \theta \implies \text{FAKE}$, otherwise $\text{REAL}$.
- **Expected Behavior**:
  - Three-way classification: `REAL`, `FAKE`, `UNCERTAIN` based on score distance from decision boundary, input quality, and detector agreement.
- **Root Cause**:
  - Binary classification baseline without epistemic uncertainty estimation.
- **Fix Required**:
  - Implement standardized uncertainty estimator and reliability scoring function.
- **Validation Method**: Test borderline scores ($0.48 - 0.55$) and corrupted inputs.
- **Status**: CONFIRMED.

---

### BUG-05 & BUG-06: Missing Quality Analysis and Standardized Detector Interface

- **ID**: `BUG-05` & `BUG-06`
- **Classification**: CONFIRMED
- **File**: `deepfake_detector_core/`
- **Current Behavior**:
  - Inconsistent dictionary keys; lack of resolution, blur, compression, or face metrics.
- **Fix Required**:
  - Build `InputQualityAnalyzer` for Image, Video, and Audio.
  - Standardize all detectors to return common forensic dataclass/dict schema.
- **Status**: CONFIRMED.
