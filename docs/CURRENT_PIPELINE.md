# FakeProbe-X Current Pipeline Trace

This document details the exact execution flow of the baseline pipelines currently implemented in `deepfake_detector_core` and `app.py`.

---

## 1. Image Pipeline Trace

```
UPLOAD → VALIDATION → PREPROCESSING → FACE DETECTION/CROP → MODEL → RAW OUTPUT → PROBABILITY → THRESHOLD → FUSION → FINAL RESULT
```

### Stage 1: Upload & Ingestion
- **File**: `app.py`
- **Function**: `st.file_uploader(..., type=['jpg', 'jpeg', 'png', 'webp'])`
- **Input**: User-uploaded image file buffer (`UploadedFile`).
- **Output**: Raw binary bytes `image_bytes = file.read()`.
- **Current Behavior**: File size check limit (200MB); read as raw bytes directly into memory.

### Stage 2: Validation
- **File**: `deepfake_detector_core/image_detector.py`
- **Function**: `ImageDeepfakeDetector.predict()`
- **Input**: `image_bytes_or_array: bytes | np.ndarray`.
- **Output**: `image_rgb: np.ndarray` (shape: `(H, W, 3)`, dtype: `uint8`).
- **Current Behavior**: Uses `cv2.imdecode(nparr, cv2.IMREAD_COLOR)`. Raises `ValueError` if `image_bgr is None`. Converted via `cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)`.

### Stage 3: Preprocessing & Forensic Feature Generation
- **File**: `deepfake_detector_core/image_detector.py`
- **Function**: `prepare_5ch_tensor()`, `compute_ela()`, `compute_fft_magnitude()`
- **Input**: `image_rgb: np.ndarray` (`(H, W, 3)`).
- **Output**: 5-channel PyTorch tensor `(1, 5, 224, 224)` (float32).
- **Current Behavior**:
  1. Image is resized unconditionally to `(224, 224)` with `cv2.resize`.
  2. `compute_ela(image_rgb, quality=90)`: Saves image to JPEG memory buffer at quality 90, reads back, computes absolute difference across RGB channels, normalizes by max value.
  3. `compute_fft_magnitude(image_rgb)`: Converts RGB to grayscale, applies 2D Fast Fourier Transform `np.fft.fft2`, shifts DC component to center `np.fft.fftshift`, computes `np.log1p(np.abs(fft_shift))`, scales to `[0, 1]`.
  4. ELA and FFT are scaled to `uint8` [0, 255] and concatenated along channel axis with RGB.
  5. Tensor normalized: RGB channels by ImageNet mean/std `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`, Forensic channels by `([0.5, 0.5], [0.5, 0.5])`.

### Stage 4: Face Detection / Crop
- **File**: `deepfake_detector_core/image_detector.py`
- **Function**: None (Currently bypassed).
- **Input**: Full scene image.
- **Output**: Full scene image resized to 224x224.
- **Current Behavior**: Currently does NOT crop the face; whole image is warped to 224x224. (Identified as a key contributor to Real $\to$ Fake false positives on full-scene photographs).

### Stage 5: Model Forward Pass
- **File**: `deepfake_detector_core/image_detector.py`
- **Function**: `ImageDeepfakeDetector.predict()`
- **Input**: `image_tensor: torch.Tensor` `(1, 5, 224, 224)`.
- **Output**: Single scalar logit $z \in \mathbb{R}$.
- **Current Behavior**: Forward pass through `CoAtNet-0` (`coatnet_0_rw_224` with 5-channel adapted stem conv).

### Stage 6: Probability & Thresholding
- **File**: `deepfake_detector_core/image_detector.py`
- **Function**: `ImageDeepfakeDetector.predict()`
- **Input**: Logit $z$.
- **Output**: `fake_prob`, `real_prob`, `prediction`, `confidence`.
- **Current Behavior**:
  - `fake_prob = float(torch.sigmoid(outputs).item())`
  - `is_fake = fake_prob > 0.5`
  - `confidence = fake_prob if is_fake else (1.0 - fake_prob)`
  - `label = "FAKE" if is_fake else "REAL"`

### Stage 7: Result Formatting
- **File**: `app.py` & `deepfake_detector_core/image_detector.py`
- **Function**: UI display block in `app.py`.
- **Input**: Prediction dictionary containing probs, maps, labels.
- **Output**: Streamlit metrics cards, progress bars, and ELA/FFT side-by-side visualization.

---

## 2. Video Pipeline Trace

```
UPLOAD → VALIDATION → VIDEO DECODING → FRAME SAMPLING → FACE DETECTION → PREPROCESSING → FRAME MODEL → FRAME SCORES → TEMPORAL AGGREGATION → AUDIO ANALYSIS IF AVAILABLE → FUSION → FINAL RESULT
```

### Stage 1: Upload & Ingestion
- **File**: `app.py`
- **Function**: `st.file_uploader(..., type=['mp4', 'avi', 'mov', 'mkv', 'webm'])`
- **Input**: Video file buffer.
- **Output**: Temporary file on disk `tfile.name`.
- **Current Behavior**: Writes uploaded bytes to a named temporary file for OpenCV / FFmpeg access.

### Stage 2: Video Decoding & Frame Sampling
- **File**: `deepfake_detector_core/video_detector.py`
- **Function**: `extract_frames_from_video(video_path, num_frames=5)`
- **Input**: Path to video file.
- **Output**: List of 5 RGB frames `list[np.ndarray]`, each `(224, 224, 3)`.
- **Current Behavior**: Reads frames sequentially using `cv2.VideoCapture`. Filters frames with mean brightness $> 5$. Buffers up to 300 frames, samples 5 uniformly spaced indices.

### Stage 3: Face Detection
- **File**: `deepfake_detector_core/video_detector.py`
- **Function**: None in baseline.
- **Input**: Full video frame.
- **Output**: Full frame resized to 224x224.
- **Current Behavior**: Whole frame is used without face bounding box localization.

### Stage 4: Visual Frame Embeddings
- **File**: `deepfake_detector_core/image_detector.py` & `video_detector.py`
- **Function**: `ImageDeepfakeDetector.extract_features(frame_rgb)`
- **Input**: `(224, 224, 3)` RGB numpy frame.
- **Output**: 768-dimensional CoAtNet feature vector `np.ndarray (768,)`.
- **Current Behavior**: Frame is converted to 5-channel tensor via `prepare_5ch_tensor`, passed through CoAtNet, captured via forward hook on `model.head.global_pool`.

### Stage 5: Visual Temporal Pooling
- **File**: `deepfake_detector_core/video_detector.py`
- **Function**: `VideoDeepfakeDetector.predict()`
- **Input**: 5 frame embeddings `(5, 768)`.
- **Output**: 2304-dimensional pooled visual vector `video_vector`.
- **Current Behavior**: Computes statistical pooling: `mean` (768-dim), `max` (768-dim), `std` (768-dim) concatenated into `(2304,)`.

### Stage 6: Audio Demuxing & Acoustic Feature Extraction
- **File**: `deepfake_detector_core/video_detector.py`
- **Function**: `extract_audio_from_video(video_path)` & `AudioDeepfakeDetector.extract_features()`
- **Input**: `video_path`.
- **Output**: 512-dimensional Whisper acoustic embedding `audio_vector (512,)` or zero vector `np.zeros(512)` if no audio.
- **Current Behavior**: Invokes FFmpeg subprocess to extract 16 kHz mono WAV audio. If audio exists, passes to Whisper encoder and mean-pools over time to obtain `(512,)`.

### Stage 7: Intermediate Fusion & Classification
- **File**: `deepfake_detector_core/video_detector.py`
- **Function**: `VideoDeepfakeDetector.predict()`
- **Input**: Concatenated vector `fused = [video_vector, audio_vector]` `(2816,)`.
- **Output**: `spoof_prob`, `real_prob`, `prediction`, `confidence`.
- **Current Behavior**:
  - `spoof_prob = float(self.classifier.predict_proba(fused_2d)[0, 1])`
  - `is_fake = spoof_prob >= 0.5780` (tuned threshold from FakeAVCeleb)
  - `confidence = spoof_prob if is_fake else (1.0 - spoof_prob)`
  - `label = "FAKE" if is_fake else "REAL"`

---

## 3. Audio Pipeline Trace

```
UPLOAD → VALIDATION → AUDIO PREPROCESSING → MODEL → RAW OUTPUT → PROBABILITY → THRESHOLD → FINAL RESULT
```

### Stage 1: Upload & Ingestion
- **File**: `app.py`
- **Function**: `st.file_uploader(..., type=['wav', 'mp3', 'flac', 'ogg', 'm4a'])`
- **Input**: Audio file buffer.
- **Output**: Audio bytes or temporary audio file.
- **Current Behavior**: Saves to temporary file with appropriate suffix.

### Stage 2: Audio Preprocessing & Feature Extraction
- **File**: `deepfake_detector_core/audio_detector.py`
- **Function**: `AudioDeepfakeDetector.extract_features(audio_path)`
- **Input**: Path to audio file.
- **Output**: 512-dimensional Whisper encoder embedding `(512,)`.
- **Current Behavior**:
  1. `librosa.load(audio_path, sr=16000)` loads 1D waveform at 16,000 Hz.
  2. `WhisperProcessor` computes 80-channel log-mel spectrogram features `(1, 80, 3000)`.
  3. `WhisperModel.encoder` produces `last_hidden_state (1, T, 512)`.
  4. Temporal mean pooling across dimension 1 yields `(512,)` feature representation.

### Stage 3: XGBoost Classifier Forward Pass
- **File**: `deepfake_detector_core/audio_detector.py`
- **Function**: `AudioDeepfakeDetector.predict()`
- **Input**: 2D feature array `(1, 512)`.
- **Output**: Class probabilities `[P(Bonafide), P(Spoof)]`.
- **Current Behavior**: `probabilities = self.classifier.predict_proba(embedding_2d)[0]`.

### Stage 4: Probability, Thresholding & Result
- **File**: `deepfake_detector_core/audio_detector.py`
- **Function**: `AudioDeepfakeDetector.predict()`
- **Input**: Class probabilities.
- **Output**: `spoof_prob`, `real_prob`, `prediction`, `confidence`.
- **Current Behavior**:
  - `spoof_prob = float(probabilities[1])`
  - `real_prob = float(probabilities[0])`
  - `is_fake = spoof_prob > 0.5`
  - `confidence = spoof_prob if is_fake else real_prob`
  - `label = "FAKE" if is_fake else "REAL"`
