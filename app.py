import os
import sys
import time
import tempfile
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from deepfake_detector_core import (
    ImageDeepfakeDetector,
    AudioDeepfakeDetector,
    VideoDeepfakeDetector,
    transcode_for_browser,
    get_image_model_path,
    get_audio_model_path,
    get_video_model_path,
)

# 1. Page Configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Styling (Clean, responsive, modern)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    
    .subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        font-weight: 400;
    }
    
    .result-container {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
        border-left: 6px solid;
        background-color: rgba(128, 128, 128, 0.07);
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    
    .result-real {
        border-left-color: #10b981;
    }
    
    .result-fake {
        border-left-color: #ef4444;
    }
    
    .badge-real {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.45rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.35rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .badge-fake {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 0.45rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.35rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    .stat-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# 3. Model Loading & Cache
@st.cache_resource(show_spinner=False)
def load_all_detectors():
    """Load and cache trained models once at startup."""
    img_det = ImageDeepfakeDetector()
    aud_det = AudioDeepfakeDetector()
    vid_det = VideoDeepfakeDetector()
    return img_det, aud_det, vid_det


# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">
        <span>🛡️</span> Deepfake Detection System
    </div>
    <div class="subtitle">
        Integrated Multimodal Media Forensics — Powered by CoAtNet 5-Channel, Whisper-Base, and Intermediate Feature Fusion
    </div>
</div>
""", unsafe_allow_html=True)


# Initialize Models
with st.spinner("Initializing Deepfake Forensic Models... Please wait."):
    try:
        image_detector, audio_detector, video_detector = load_all_detectors()
        models_ready = True
    except Exception as e:
        models_ready = False
        st.error(f"Error loading models: {e}")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    img_p = get_image_model_path()
    aud_p = get_audio_model_path()
    vid_p = get_video_model_path()

    st.write(f"🖼️ **Image Model:** {'✅ Ready' if img_p else '❌ Missing'}")
    st.write(f"🎙️ **Audio Model:** {'✅ Ready' if aud_p else '❌ Missing'}")
    st.write(f"🎥 **Video Model:** {'✅ Ready' if vid_p else '❌ Missing'}")
    
    st.markdown("---")
    st.markdown("### 🔬 Detection Technology")
    st.caption("**Image Engine:** CoAtNet-0 (RGB + ELA + FFT 5-Channel)")
    st.caption("**Audio Engine:** Whisper-Base + XGBoost (ASVspoof)")
    st.caption("**Video Engine:** Intermediate Multimodal Fusion (2816-dim)")
    
    st.markdown("---")
    st.caption("Deepfake Detection Unified Platform v1.0")


if not models_ready:
    st.stop()


# Main Navigation Tabs
tab_image, tab_video, tab_audio, tab_info = st.tabs([
    "🖼️ Image Detection",
    "🎥 Video Detection",
    "🎙️ Audio Detection",
    "ℹ️ System & Models",
])


# ==============================================================================
# TAB 1: IMAGE DETECTION
# ==============================================================================
with tab_image:
    st.subheader("Image Deepfake Detection")
    st.write("Upload a portrait, photograph, or facial image to analyze pixel forensics and frequency artifacts.")
    
    uploaded_image = st.file_uploader(
        "Upload Image (JPG, PNG, JPEG, WEBP)",
        type=["jpg", "png", "jpeg", "webp"],
        key="uploader_image"
    )
    
    if uploaded_image is not None:
        col_img_left, col_img_right = st.columns([1, 1], gap="large")
        
        with col_img_left:
            st.markdown("#### Input Preview")
            pil_img = Image.open(uploaded_image).convert("RGB")
            st.image(pil_img, caption=f"{uploaded_image.name} ({pil_img.width}x{pil_img.height})", use_container_width=True)
            
        with col_img_right:
            st.markdown("#### Forensic Analysis")
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True, key="btn_img"):
                with st.status("Executing Image Forensics...", expanded=True) as status:
                    st.write("Extracting Error Level Analysis (ELA) compression map...")
                    time.sleep(0.3)
                    st.write("Computing Fast Fourier Transform (FFT) 2D frequency spectrum...")
                    time.sleep(0.3)
                    st.write("Executing CoAtNet 5-Channel neural network inference...")
                    
                    uploaded_image.seek(0)
                    img_bytes = uploaded_image.read()
                    res = image_detector.predict(img_bytes)
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                
                # Render Result
                card_class = "result-fake" if res["is_fake"] else "result-real"
                badge_class = "badge-fake" if res["is_fake"] else "badge-real"
                icon = "⚠️" if res["is_fake"] else "✅"
                
                st.markdown(f"""
                <div class="result-container {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span class="{badge_class}">{icon} {res['prediction']}</span>
                        <span style="font-size: 1.4rem; font-weight: 700;">Confidence: {res['confidence']*100:.2f}%</span>
                    </div>
                    <p style="margin: 0; font-weight: 600; opacity: 0.85;">{res['detail_label']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probabilities
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Authentic (Real) Score", f"{res['real_prob']*100:.2f}%")
                with col_p2:
                    st.metric("Manipulated (Fake) Score", f"{res['fake_prob']*100:.2f}%")
                    
                st.progress(res["real_prob"], text=f"Authenticity Probability: {res['real_prob']*100:.2f}%")
                
                # Forensic Artifact Inspection
                with st.expander("🔬 View Extracted Forensic Maps (ELA & FFT)"):
                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        st.markdown("**Error Level Analysis (ELA)**")
                        st.image(res["ela_map"], caption="JPEG compression anomaly map", clamp=True, use_container_width=True)
                    with f_col2:
                        st.markdown("**FFT 2D Power Spectrum**")
                        st.image(res["fft_map"], caption="High-frequency generative artifacts", clamp=True, use_container_width=True)


# ==============================================================================
# TAB 2: VIDEO DETECTION
# ==============================================================================
with tab_video:
    st.subheader("Video Deepfake Detection")
    st.write("Upload a video to perform multimodal intermediate fusion analysis across temporal frames and acoustic tracks.")
    
    uploaded_video = st.file_uploader(
        "Upload Video (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        key="uploader_video"
    )
    
    if uploaded_video is not None:
        # Save temp video for inference and preview
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as tmp_v:
            tmp_v.write(uploaded_video.getbuffer())
            temp_video_path = tmp_v.name

        preview_path = transcode_for_browser(temp_video_path)
        
        col_vid_left, col_vid_right = st.columns([1, 1], gap="large")
        
        with col_vid_left:
            st.markdown("#### Video Preview")
            if preview_path and os.path.exists(preview_path):
                st.video(preview_path)
            else:
                st.video(uploaded_video)
                
        with col_vid_right:
            st.markdown("#### Multimodal Forensic Analysis")
            if st.button("🎥 Analyze Video", type="primary", use_container_width=True, key="btn_vid"):
                try:
                    with st.status("Executing Multimodal Forensic Pipeline...", expanded=True) as status:
                        st.write("Sampling temporal frames across video duration...")
                        time.sleep(0.3)
                        st.write("Extracting 5-Channel CoAtNet visual embeddings (RGB + ELA + FFT)...")
                        time.sleep(0.3)
                        st.write("Applying statistical pooling (mean + max + std -> 2304-dim)...")
                        time.sleep(0.3)
                        st.write("Demuxing audio track & extracting Whisper acoustic embeddings...")
                        time.sleep(0.3)
                        st.write("Fusing visual & acoustic feature vectors (2816-dim)...")
                        time.sleep(0.3)
                        st.write("Running trained XGBoost fusion classifier...")
                        
                        res = video_detector.predict(
                            video_path=temp_video_path,
                            image_model=image_detector,
                            audio_model=audio_detector,
                        )
                        status.update(label="Analysis Complete", state="complete", expanded=False)
                    
                    # Render Result
                    card_class = "result-fake" if res["is_fake"] else "result-real"
                    badge_class = "badge-fake" if res["is_fake"] else "badge-real"
                    icon = "⚠️" if res["is_fake"] else "✅"
                    
                    st.markdown(f"""
                    <div class="result-container {card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                            <span class="{badge_class}">{icon} {res['prediction']}</span>
                            <span style="font-size: 1.4rem; font-weight: 700;">Confidence: {res['confidence']*100:.2f}%</span>
                        </div>
                        <p style="margin: 0; font-weight: 600; opacity: 0.85;">{res['detail_label']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_vp1, col_vp2 = st.columns(2)
                    with col_vp1:
                        st.metric("Authentic (Real) Score", f"{res['real_prob']*100:.2f}%")
                    with col_vp2:
                        st.metric("Deepfake (Fake) Score", f"{res['fake_prob']*100:.2f}%")
                        
                    st.progress(res["real_prob"], text=f"Authenticity Probability: {res['real_prob']*100:.2f}%")
                    
                    if res.get("has_audio"):
                        st.info("🔊 Audio track detected and fused into multimodal prediction.")
                    else:
                        st.warning("🔇 No audio track found in video; visual features evaluated with zero-padded acoustic vector.")
                    
                    if len(res.get("sampled_frames", [])) > 0:
                        st.markdown("#### Sampled Video Frames")
                        frame_cols = st.columns(len(res["sampled_frames"]))
                        for i, f_img in enumerate(res["sampled_frames"]):
                            with frame_cols[i]:
                                st.image(f_img, caption=f"Frame #{i+1}", use_container_width=True)
                
                finally:
                    if os.path.exists(temp_video_path):
                        try:
                            os.remove(temp_video_path)
                        except Exception:
                            pass
                    if preview_path and os.path.exists(preview_path):
                        try:
                            os.remove(preview_path)
                        except Exception:
                            pass


# ==============================================================================
# TAB 3: AUDIO DETECTION
# ==============================================================================
with tab_audio:
    st.subheader("Audio Deepfake & Voice Clone Detection")
    st.write("Upload an audio recording to detect AI voice cloning, Text-to-Speech (TTS), or speech synthesis spoofs.")
    
    uploaded_audio = st.file_uploader(
        "Upload Audio (WAV, MP3, FLAC, OGG)",
        type=["wav", "mp3", "flac", "ogg"],
        key="uploader_audio"
    )
    
    if uploaded_audio is not None:
        col_aud_left, col_aud_right = st.columns([1, 1], gap="large")
        
        with col_aud_left:
            st.markdown("#### Audio Playback")
            st.audio(uploaded_audio)
            
        with col_aud_right:
            st.markdown("#### Acoustic Forensic Analysis")
            if st.button("🎙️ Analyze Audio", type="primary", use_container_width=True, key="btn_aud"):
                with st.status("Executing Acoustic Pipeline...", expanded=True) as status:
                    st.write("Extracting 16kHz audio waveform...")
                    time.sleep(0.3)
                    st.write("Generating Whisper-Base encoder temporal embeddings (512-dim)...")
                    time.sleep(0.3)
                    st.write("Classifying acoustic features via ASVspoof XGBoost model...")
                    
                    uploaded_audio.seek(0)
                    aud_bytes = uploaded_audio.read()
                    res = audio_detector.predict(aud_bytes, suffix=os.path.splitext(uploaded_audio.name)[1])
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                    
                # Render Result
                card_class = "result-fake" if res["is_fake"] else "result-real"
                badge_class = "badge-fake" if res["is_fake"] else "badge-real"
                icon = "⚠️" if res["is_fake"] else "✅"
                
                st.markdown(f"""
                <div class="result-container {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span class="{badge_class}">{icon} {res['prediction']}</span>
                        <span style="font-size: 1.4rem; font-weight: 700;">Confidence: {res['confidence']*100:.2f}%</span>
                    </div>
                    <p style="margin: 0; font-weight: 600; opacity: 0.85;">{res['detail_label']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_ap1, col_ap2 = st.columns(2)
                with col_ap1:
                    st.metric("Bonafide (Human) Score", f"{res['real_prob']*100:.2f}%")
                with col_ap2:
                    st.metric("Spoof (AI Voice) Score", f"{res['fake_prob']*100:.2f}%")
                    
                st.progress(res["real_prob"], text=f"Human Voice Probability: {res['real_prob']*100:.2f}%")


# ==============================================================================
# TAB 4: SYSTEM & MODEL INFO
# ==============================================================================
with tab_info:
    st.subheader("Deepfake Detection Integration Architecture")
    st.markdown("""
    This unified platform integrates all three existing repositories into a coherent, high-performance forensic application without retraining or placeholder mocks.
    """)
    
    col_i1, col_i2, col_i3 = st.columns(3)
    
    with col_i1:
        st.markdown("""
        <div class="stat-card">
            <h4>🖼️ Image Detector</h4>
            <p><strong>Architecture:</strong> CoAtNet-0 5-Channel</p>
            <p><strong>Input:</strong> RGB (3ch) + ELA (1ch) + FFT (1ch)</p>
            <p><strong>Trained Weights:</strong> <code>best_coatnet_5ch.pth</code></p>
            <p><strong>Size:</strong> 106.8 MB</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_i2:
        st.markdown("""
        <div class="stat-card">
            <h4>🎙️ Audio Detector</h4>
            <p><strong>Architecture:</strong> Whisper-Base + XGBoost</p>
            <p><strong>Embedding:</strong> 512-dim mean-pooled</p>
            <p><strong>Trained Weights:</strong> <code>xgboost_asvspoof_model.joblib</code></p>
            <p><strong>Benchmark:</strong> ASVspoof dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_i3:
        st.markdown("""
        <div class="stat-card">
            <h4>🎥 Video Detector</h4>
            <p><strong>Architecture:</strong> Intermediate Multimodal Fusion</p>
            <p><strong>Vector:</strong> 2304d (Visual) + 512d (Audio) = 2816d</p>
            <p><strong>Trained Weights:</strong> <code>xgboost_fusion_model.joblib</code></p>
            <p><strong>Threshold:</strong> 0.5780 (Balanced Eval)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Integration & Missing Checkpoints Transparency Report")
    st.markdown("""
    - **Project 1 (`AI-DeepfakeDetector-main`):** Provided the fully trained weights for 5-channel CoAtNet-0, Whisper+XGBoost audio classifier, and the multimodal fusion XGBoost model. Reused in full.
    - **Project 2 (`multimodal-deepfake-detector-master`):** Provided EfficientNet/MTCNN architecture definitions and dataset evaluation pipeline. Notice: It was published without a bundled `.pth` model checkpoint file (used runtime weight inputs). The unified application prioritizes the pre-trained 5-channel CoAtNet model.
    - **Project 3 (`Awesome-Comprehensive-Deepfake-Detection-main`):** Comprehensive research taxonomy, literature collection, and benchmarks.
    """)
