# FakeProbe-X Baseline Audit Report

This document presents a comprehensive audit of all integrated projects, models, pipelines, and artifacts present in the repository.

---

## 1. Project: AI-DeepfakeDetector (CoAtNet + Whisper + XGBoost Intermediate Fusion)

- **A. Project Name**: AI-DeepfakeDetector (`AI-DeepfakeDetector-main / FinalFrameWork`)
- **B. Purpose**: Multimodal deepfake detection supporting static forensic image classification, speech/audio voice cloning spoof detection, and intermediate feature-level video deepfake classification.
- **C. Supported Modalities**:
  - **Image**: Yes (5-channel CoAtNet-0 with RGB + ELA + FFT)
  - **Video**: Yes (5-frame temporal sampling + CoAtNet statistical pooling + Whisper acoustic pooling + XGBoost classifier)
  - **Audio**: Yes (Whisper-Base 512-dim encoder embeddings + XGBoost classifier)
  - **Multimodal**: Yes (2816-dim concatenated visual + acoustic vector)
- **D. Main Entry Point**: `AI-DeepfakeDetector-main/AI-DeepfakeDetector-main/FinalFrameWork/app.py` & `deepfake_detector_core`
- **E. Framework**: PyTorch (`torch`, `torchvision`, `torchaudio`), `timm` (0.9.x), `transformers` (Hugging Face Whisper), `xgboost`, `scikit-learn`, `librosa`, `OpenCV` (`cv2`), `Pillow`.
- **F. Model Architecture**:
  - *Image*: `coatnet_0_rw_224` with custom first conv layer accepting 5 input channels (3 RGB + 1 ELA + 1 FFT magnitude) and single logit binary head.
  - *Audio*: `openai/whisper-base` encoder (frozen, 512-dim mean representation) + `XGBClassifier` binary model.
  - *Video*: Visual branch (`coatnet_0_rw_224` 768-dim embeddings across 5 frames with mean, max, std pooling = 2304-dim) + Audio branch (Whisper encoder mean pooling = 512-dim) -> Fused (2816-dim) -> `XGBClassifier`.
- **G. Exact Model/Checkpoint Filename**:
  - `best_coatnet_5ch.pth` (106,811,923 bytes)
  - `xgboost_asvspoof_model.joblib` (17,946 bytes)
  - `xgboost_fusion_model.joblib` (136,774 bytes) & `xgboost_fusion_model.json` (178,240 bytes)
- **H. Checkpoint Availability**: AVAILABLE (all 3 binary files verified present in repository).
- **I. Model Loading Mechanism**:
  - PyTorch: `torch.load(weights_path, map_location=device)` into `build_coatnet_5ch()` model.
  - XGBoost Audio: `joblib.load(xgb_path)`.
  - XGBoost Video: `joblib.load(xgb_path)`.
- **J. Input Dimensions**:
  - *Image*: `(1, 5, 224, 224)` tensor.
  - *Audio*: 16 kHz mono 1D waveform `(N,)` -> Whisper log-mel spectrogram `(1, 80, 3000)` -> 512-dim embedding.
  - *Video*: 5 frames of `(224, 224, 3)` RGB images -> `(1, 2816)` feature vector.
- **K. Preprocessing**:
  - *Image*: Resize to 224x224 RGB, compute Error Level Analysis (JPEG quality 90), compute 2D FFT magnitude spectrum (log1p shift), scale both forensic maps to [0, 255] uint8, stack with RGB to shape `(224, 224, 5)`.
  - *Audio*: Resample to 16,000 Hz, Whisper feature extractor (`WhisperProcessor`).
  - *Video*: Extract up to 300 frames via OpenCV, select 5 uniformly spaced non-black frames, resize to 224x224 RGB, extract audio via FFmpeg to 16 kHz 16-bit PCM WAV.
- **L. Resize**: 224x224 using bilinear interpolation (`cv2.resize`).
- **M. Color/Channel Handling**: Input decoded to BGR via OpenCV and converted to RGB via `cv2.COLOR_BGR2RGB`. Stacked along axis 2 with ELA and FFT to form 5 channels.
- **N. Normalization**:
  - Channels 0-2 (RGB): ImageNet Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`.
  - Channels 3-4 (Forensic): Mean `[0.5, 0.5]`, Std `[0.5, 0.5]`.
- **O. Face Detection/Cropping**: None in baseline `AI-DeepfakeDetector-main` (processes whole frame resized to 224x224 directly; MTCNN is referenced in notebooks but not applied in the inference scripts).
- **P. Frame Sampling**: Uniform temporal sampling over the first 300 valid frames ($N=5$).
- **Q. Audio Preprocessing**: Librosa load at 16 kHz, Whisper log-mel spectrogram extractor.
- **R. Label Mapping**:
  - Image: `0` = Real/Bonafide, `1` = Fake/Spoof.
  - Audio: `0` = Bonafide (Real), `1` = Spoof (Fake).
  - Video: `0` = Bonafide (Real), `1` = Spoof (Fake).
- **S. Raw Model Output**:
  - Image: Unbounded single float logit $z \in \mathbb{R}$.
  - Audio: XGBoost predicted class probabilities `[P(Real), P(Fake)]`.
  - Video: XGBoost predicted class probabilities `[P(Real), P(Fake)]`.
- **T. Probability Calculation**:
  - Image: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
  - Audio: `predict_proba(embedding)[0, 1]`.
  - Video: `predict_proba(fused_features)[0, 1]`.
- **U. Confidence Calculation**: `confidence = prob if is_fake else (1.0 - prob)` (i.e., $\max(P(\text{Fake}), P(\text{Real}))$).
- **V. Decision Threshold**:
  - Image: `0.50`
  - Audio: `0.50`
  - Video: `0.5780` (tuned on FakeAVCeleb test set)
- **W. REAL/FAKE Mapping**: Score $\ge \text{threshold} \implies \text{FAKE}$, Score $< \text{threshold} \implies \text{REAL}$.
- **X. Fusion Method**: Intermediate feature concatenation (Visual pooled 2304-dim + Acoustic 512-dim = 2816-dim) followed by XGBoost classifier.
- **Y. Dataset References**:
  - Image: `awsaf49/artifact-dataset` (Kaggle).
  - Audio: `ASVspoof2019_LA` (Logical Access).
  - Video: `FakeAVCeleb` (Multimodal Deepfake Dataset).
- **Z. Training Information**:
  - Image: Focal Loss ($\alpha=0.25, \gamma=2.0$), Cosine Annealing, Gradual unfreezing, Hard negative mining, AdamW ($lr=1e-3$).
  - Audio: Whisper-base frozen encoder + XGBoost classifier on ASVspoof 2019 LA train set.
  - Video: Intermediate fusion XGBoost classifier on FakeAVCeleb balanced train set.
- **AA. Dependencies**: `torch`, `torchvision`, `torchaudio`, `timm`, `transformers`, `xgboost`, `librosa`, `opencv-python`, `numpy`, `pandas`, `Pillow`, `joblib`, `imageio-ffmpeg`.
- **AB. Known Errors**:
  - Whole-image distortion when non-square images are resized directly without face cropping or aspect-ratio preservation.
  - Baseline image detector trained on cropped face artifacts; full-scene non-cropped real photos exhibit forensic distribution shifts leading to false positives (REAL $\to$ FAKE misclassification).
  - Video detector silently zero-pads audio vector (512 zeros) when video has no audio track, but XGBoost model was trained on video+audio pairs.
  - Thresholds are hard-coded binary cuts with no uncertainty or reliability quantification.
- **AC. Missing Components**: Face detector preprocessing (MTCNN/RetinaFace/Haar) in the inference loop; uncertainty scoring; out-of-distribution detection; adaptive weighting.
- **AD. Whether Currently Runnable**: RUNNABLE (All weights, checkpoints, and dependencies are installed and verified).

---

## 2. Project: Multimodal-Deepfake-Detector-Master (EfficientNet-B1)

- **A. Project Name**: Multimodal Deepfake Detector Master (`multimodal-deepfake-detector-master`)
- **B. Purpose**: PyTorch-based training, validation, testing, and face extraction framework for binary deepfake detection using EfficientNet backbones.
- **C. Supported Modalities**:
  - **Image**: Yes (EfficientNet-B1 binary classifier)
  - **Video**: Video frame extraction scripts provided (`scripts/extract_faces.py`)
  - **Audio**: NOT VERIFIED (No audio models or pipelines implemented in codebase)
  - **Multimodal**: NOT VERIFIED (Name indicates multimodal, but implementation is image-based EfficientNet)
- **D. Main Entry Point**: `multimodal-deepfake-detector-master/multimodal-deepfake-detector-master/app.py` & `scripts/train.py`
- **E. Framework**: PyTorch (`torch`, `torchvision`), `efficientnet-pytorch`, `opencv-python`, `Pillow`, `albumentations`.
- **F. Model Architecture**: `EfficientNet-B1` with custom FC head (`Linear(num_ftrs, 1000) -> ReLU -> Dropout(0.5) -> Linear(1000, 2)`).
- **G. Exact Model/Checkpoint Filename**: `best_model.pth` or `checkpoint_epoch_*.pth` (referenced in code).
- **H. Checkpoint Availability**: MODEL UNAVAILABLE (No pre-trained `.pth` checkpoint was provided in the repository commit).
- **I. Model Loading Mechanism**: `model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])`.
- **J. Input Dimensions**: `(B, 3, 240, 240)` for EfficientNet-B1.
- **K. Preprocessing**: Standard torchvision/albumentations transforms (`Resize`, `Normalize`, `RandomHorizontalFlip`).
- **L. Resize**: 240x240 for EfficientNet-B1.
- **M. Color/Channel Handling**: 3-channel RGB.
- **N. Normalization**: ImageNet Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`.
- **O. Face Detection/Cropping**: OpenCV Haar Cascade / MTCNN scripts in `scripts/extract_faces.py`.
- **P. Frame Sampling**: Frame interval extraction in `scripts/extract_faces.py`.
- **Q. Audio Preprocessing**: NOT VERIFIED (None present).
- **R. Label Mapping**: `0` = Real, `1` = Fake.
- **S. Raw Model Output**: Logits `(B, 2)`.
- **T. Probability Calculation**: `torch.softmax(logits, dim=1)[:, 1]`.
- **U. Confidence Calculation**: `torch.max(softmax_probs, dim=1)`.
- **V. Decision Threshold**: `0.50` (or EER grid search threshold).
- **W. REAL/FAKE Mapping**: `pred = torch.argmax(logits, dim=1)`.
- **X. Fusion Method**: None.
- **Y. Dataset References**: Generic directory structure (`real_dirs`, `fake_dirs` for FaceForensics++/Celeb-DF).
- **Z. Training Information**: CrossEntropyLoss, AdamW ($lr=8e-4$), Warmup + StepLR scheduler.
- **AA. Dependencies**: `torch`, `torchvision`, `efficientnet-pytorch`, `albumentations`, `scikit-learn`.
- **AB. Known Errors**: Model weights are not bundled in repository.
- **AC. Missing Components**: Checkpoint file; audio/video fusion implementation.
- **AD. Whether Currently Runnable**: INFERENCE UNAVAILABLE (Requires pre-trained weights to run inference; training framework is complete).

---

## 3. Project: Awesome-Comprehensive-Deepfake-Detection-Main (Survey & Taxonomy)

- **A. Project Name**: Awesome-Comprehensive-Deepfake-Detection (`Awesome-Comprehensive-Deepfake-Detection-main`)
- **B. Purpose**: Literature survey, taxonomy of deepfake generation and detection algorithms, benchmark datasets compilation, and evaluation metric references.
- **C. Supported Modalities**: Literature documentation for Image, Video, Audio, Multimodal.
- **D. Main Entry Point**: `README.md`
- **E. Framework**: Markdown documentation / bibliography.
- **F. Model Architecture**: Survey of Spatial, Frequency, Temporal, Biological, and Multimodal architectures.
- **G. Exact Model/Checkpoint Filename**: None (Documentation only).
- **H. Checkpoint Availability**: NOT APPLICABLE.
- **I. Model Loading Mechanism**: NOT APPLICABLE.
- **J. Input Dimensions**: NOT APPLICABLE.
- **K. Preprocessing**: NOT APPLICABLE.
- **L. Resize**: NOT APPLICABLE.
- **M. Color/Channel Handling**: NOT APPLICABLE.
- **N. Normalization**: NOT APPLICABLE.
- **O. Face Detection/Cropping**: NOT APPLICABLE.
- **P. Frame Sampling**: NOT APPLICABLE.
- **Q. Audio Preprocessing**: NOT APPLICABLE.
- **R. Label Mapping**: NOT APPLICABLE.
- **S. Raw Model Output**: NOT APPLICABLE.
- **T. Probability Calculation**: NOT APPLICABLE.
- **U. Confidence Calculation**: NOT APPLICABLE.
- **V. Decision Threshold**: NOT APPLICABLE.
- **W. REAL/FAKE Mapping**: NOT APPLICABLE.
- **X. Fusion Method**: Survey of Early, Intermediate, Late, and Cross-Attention Fusion methods.
- **Y. Dataset References**: Exhaustive catalog of datasets: FaceForensics++, Celeb-DF, DFDC, DeeperForensics-1.0, FakeAVCeleb, ASVspoof, In-the-Wild.
- **Z. Training Information**: NOT APPLICABLE.
- **AA. Dependencies**: None.
- **AB. Known Errors**: None.
- **AC. Missing Components**: Standalone codebase/models (serves as domain reference).
- **AD. Whether Currently Runnable**: DOCUMENTATION ONLY.

---

## 4. Summary Matrix of Available Checkpoints and Detectors

| Detector | Architecture | Modality | Checkpoint Path | Status |
| :--- | :--- | :--- | :--- | :--- |
| **CoAtNet-0 5Ch** | CoAtNet-0 (5ch: RGB+ELA+FFT) | Image | `.../FinalFrameWork/models/best_coatnet_5ch.pth` | ✅ Available & Loaded |
| **Whisper-XGB** | Whisper-Base + XGBoost | Audio | `.../FinalFrameWork/models/xgboost_asvspoof_model.joblib` | ✅ Available & Loaded |
| **Multimodal XGB**| CoAtNet + Whisper + XGBoost | Video | `.../FinalFrameWork/Final-Vid/xgboost_fusion_model.joblib` | ✅ Available & Loaded |
| **EfficientNet-B1**| EfficientNet-B1 (2-class) | Image | None (`checkpoints/best_model.pth` not provided) | ⚠️ Model Unavailable |
