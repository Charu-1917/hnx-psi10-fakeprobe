# Final Framework — Multimodal Deepfake Detection Application

## Overview

The Final Framework is a production-ready Streamlit web application that integrates all three detection branches (Image, Audio, Video) into a unified, user-facing forensic analysis tool. The application provides an intuitive interface for uploading media files and receiving AI-powered deepfake verdicts in real time, translating complex backend multimodal machine learning models into accessible forensic insights.

---

## Directory Structure

```text
FinalFrameWork/
|-- app.py                          # Main Streamlit application entry point
|-- requirements.txt                # Python dependencies
|-- inference/
|   |-- __init__.py
|   |-- image_inference.py          # ImageDeepfakeDetector class handling CoAtNet
|   |-- audio_inference.py          # AudioDeepfakeDetector class handling Whisper+XGBoost
|   |-- video_inference.py          # VideoDeepfakeDetector class handling fusion logic
|-- models/
|   |-- best_coatnet_5ch.pth        # CoAtNet weights (shared by Image + Video)
|   |-- xgboost_asvspoof_model.joblib  # XGBoost audio classifier
|   |-- processor_config.json       # Whisper processor config
|   |-- tokenizer.json              # Whisper tokenizer
|   |-- tokenizer_config.json       # Whisper tokenizer settings
|-- Final-Vid/
|   |-- xgboost_fusion_model.joblib # XGBoost video fusion model
|   |-- xgboost_fusion_model.json   # Fusion model (JSON format)
|   |-- inference_config.json       # Video pipeline configuration
```

---

## Installation and Execution

### 1. Prerequisites

**System Dependency: FFmpeg**
The application requires FFmpeg for crucial media processing tasks, such as demuxing audio tracks from video files and transcoding videos for browser compatibility.
- **Windows:** Download from [FFmpeg official website](https://ffmpeg.org/download.html) and add the `bin` folder to your system PATH.
- **Linux:** `sudo apt-get update && sudo apt-get install ffmpeg`
- **macOS:** `brew install ffmpeg`

**Python Environment:**
It is highly recommended to use a virtual environment (e.g., `venv` or `conda`) running Python 3.10+.

### 2. Dependency Installation

Navigate to the `FinalFrameWork` directory and install the required Python packages:

```bash
cd FinalFrameWork
pip install -r requirements.txt
```

### 3. Model Weights Setup

Ensure that all necessary model weights are downloaded and placed in their respective directories:
- `models/best_coatnet_5ch.pth`
- `models/xgboost_asvspoof_model.joblib`
- `Final-Vid/xgboost_fusion_model.joblib`

### 4. Launching the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The application will launch and be available locally at `http://localhost:8501`.

---

## Application Architecture

### Model Loading & Caching

```python
@st.cache_resource(show_spinner=False)
def load_models():
    img_model = ImageDeepfakeDetector()    # Loads CoAtNet (~102MB)
    aud_model = AudioDeepfakeDetector()    # Loads Whisper-Base (~290MB)
    vid_model = VideoDeepfakeDetector()    # Loads XGBoost fusion (~129KB)
    return img_model, aud_model, vid_model
```

**Key Design Principle:** The `@st.cache_resource` decorator ensures that heavy deep learning models (CoAtNet and Whisper) are loaded into GPU/CPU memory exactly once at application startup. They are then kept in memory across all user sessions. The `VideoDeepfakeDetector` is deliberately designed to be lightweight—it only loads the XGBoost classifier and shares the CoAtNet and Whisper instances from the Image and Audio models, completely preventing memory duplication and reducing VRAM footprint.

### Module Architecture

```text
+-------------------------------------------------+
|                 Streamlit App (app.py)          |
|                                                 |
|  +-------------+ +--------------+ +----------+  |
|  |    Image    | |    Audio     | |  Video   |  |
|  |   Module    | |   Module     | |  Module  |  |
|  +------+------+ +------+-------+ +----+-----+  |
|         |               |              |        |
|  +------v------+ +------v-------+      |        |
|  |  CoAtNet    | |  Whisper +   |      |        |
|  | 5-Channel   | |  XGBoost     |<-----+        |
|  | Detector    | |  Classifier  |      |        |
|  +-------------+ +--------------+      |        |
|         |               |              |        |
|         +---------------+--------------+        |
|                         |                       |
|              +----------v----------+            |
|              |  XGBoost Fusion     |            |
|              |  Classifier         |            |
|              |  (Video only)       |            |
|              +---------------------+            |
+-------------------------------------------------+
```

---

## Analysis Modules

### 1. Image Analysis Module

**Supported formats:** JPG, PNG, JPEG

**Processing Pipeline:**
1. Upload and display the source image.
2. Click "Run Visual Scan".
3. `st.status()` provides a terminal-like UX while:
   - Extracting the Error Level Analysis (ELA) map.
   - Computing the Fast Fourier Transform (FFT) spectrum.
   - Running the CoAtNet 5-channel inference.
4. Display verdict card (green for Real, red for Fake) alongside a confidence progress bar.

**Forensic Pipeline Steps (displayed to user):**
```text
- Extracting Error Level Analysis (ELA) map...
- Computing Fast Fourier Transform (FFT) spectrum...
- Running CoAtNet 5-Channel inference...
```

**Verdict Outputs:**
- **Real Image:** Displayed with the confidence score.
- **Fake (AI Generated):** Displayed with the confidence score and an interpretation of the detected manipulation.

---

### 2. Audio Analysis Module

**Supported formats:** WAV, FLAC, MP3

**Processing Pipeline:**
1. Upload and initialize in-browser audio playback.
2. Click "Run Acoustic Scan".
3. While scanning:
   - Extract raw audio waveform at 16kHz.
   - Generate temporal embeddings via the Whisper Encoder.
   - Evaluate the 512-dim features through the XGBoost classifier.

**Forensic Pipeline Steps (displayed to user):**
```text
- Extracting raw audio waveform...
- Generating temporal embeddings via Whisper Encoder...
- Evaluating features through XGBoost topology...
```

**Verdict Outputs:**
- **Bonafide (Real Human):** Indicates natural human vocal characteristics and continuity.
- **Spoof (AI Generated):** Indicates voice cloning or TTS artifacts detected in the acoustic feature space.

---

### 3. Video Analysis Module

**Supported formats:** MP4, AVI, MOV, MKV

**Browser Compatibility Fix:**
A common issue in web-based video analysis is the "black screen" error, which occurs when deepfake videos use unconventional pixel formats (e.g., yuv444p) that HTML5 players cannot render. The app mitigates this by automatically transcoding uploaded videos to a standard H.264/yuv420p format before previewing:

```python
def transcode_for_browser(input_path: str) -> str | None:
    cmd = ["ffmpeg", "-y", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "ultrafast", "-c:a", "aac", "-movflags", "+faststart", ...]
```

**Multimodal Forensic Pipeline Steps (displayed to user):**
```text
- Sampling 5 uniform frames from temporal axis...
- Extracting per-frame CoAtNet embeddings (ELA + FFT + RGB)...
- Applying statistical pooling (mean + max + std)...
- Demuxing audio track via FFmpeg...
- Generating Whisper encoder embeddings...
- Fusing visual and acoustic feature vectors (2816-dim)...
- Running XGBoost fusion classifier...
```

**Verdict Outputs:**
- **Real (Authentic Video):** Indicates consistent audiovisual coherence and lack of manipulation.
- **Fake (Deepfake Detected):** Visual or acoustic manipulation successfully detected by the multimodal fusion logic.

---

## UI/UX Design

### Custom CSS Styling

```css
/* Smooth fade-in animation for loading elements */
.main { animation: fadeIn 0.8s ease-in-out; }

/* Dynamic result cards for clear visual feedback */
.result-card {
    padding: 20px;
    border-radius: 8px;
    background-color: rgba(128, 128, 128, 0.1);
    border-left: 6px solid;
}
.result-fake { border-left-color: #ff4b4b; } /* Red for fakes */
.result-real { border-left-color: #09ab3b; } /* Green for real */
```

### Key UX Features

| Feature | Implementation |
|---|---|
| **Progressive Status Updates** | `st.status()` with `expanded=True` provides real-time feedback during heavy processing. |
| **Confidence Visualization** | `st.progress()` bar combined with a percentage label makes the model's certainty intuitive. |
| **Side-by-side Layout** | `st.columns([1, 1])` allocates screen space efficiently for media playback alongside analysis logs. |
| **Theme Adaptability** | RGBA backgrounds with CSS opacity ensure the interface looks premium in both Dark and Light modes. |
| **Temp File Cleanup** | Automatic cleanup logic housed in `finally` blocks prevents server storage saturation. |
| **Error Handling** | Graceful exception catching displays clear "Error:" labels instead of crashing the UI. |

---

## Configuration

### `Final-Vid/inference_config.json`

This configuration file defines the exact parameters required for the Video fusion pipeline to operate correctly, ensuring sync between training and inference phases:

```json
{
    "pipeline": "Intermediate Fusion (CoAtNet + Whisper + XGBoost)",
    "coatnet_model": "coatnet_0_rw_224",
    "coatnet_input_channels": 5,
    "coatnet_input_size": 224,
    "whisper_model": "openai/whisper-base",
    "num_frames": 5,
    "frame_sampling": "uniform",
    "video_pooling": "statistical (mean + max + std)",
    "audio_pooling": "mean over time",
    "video_feat_dim": 2304,
    "audio_feat_dim": 512,
    "fused_feat_dim": 2816,
    "classifier": "XGBoost",
    "label_mapping": {"0": "Bonafide (Real)", "1": "Spoof (Fake)"},
    "normalization": {
        "rgb_mean": [0.485, 0.456, 0.406],
        "rgb_std": [0.229, 0.224, 0.225],
        "forensic_mean": [0.5, 0.5],
        "forensic_std": [0.5, 0.5]
    }
}
```

---

## Technical Stack

| Component | Technology |
|---|---|
| Web Framework | Streamlit |
| Image Detector | CoAtNet-0 (timm) + Custom 5-channel |
| Audio Detector | Whisper-Base + XGBoost |
| Video Detector | Multimodal Fusion (XGBoost) |
| Video Transcoding | FFmpeg |
| Model Caching | `@st.cache_resource` |
| GPU Support | CUDA (auto-detected natively via PyTorch) |
