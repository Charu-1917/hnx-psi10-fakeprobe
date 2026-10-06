import os
import time
import tempfile
import subprocess
import streamlit as st
from inference.image_inference import ImageDeepfakeDetector
from inference.audio_inference import AudioDeepfakeDetector
from inference.video_inference import VideoDeepfakeDetector


def transcode_for_browser(input_path: str) -> str | None:
    """Re-encode a video to H.264 / yuv420p so every browser can play it.

    Many deepfake-generated videos use pixel formats (e.g. yuv444p) or
    codecs that browsers refuse to decode, resulting in a black screen.
    This function creates a lightweight, browser-safe copy.
    """
    output_path = input_path + "_preview.mp4"
    ffmpeg_exe = "ffmpeg"
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    cmd = [
        ffmpeg_exe, "-y", "-v", "error",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",   # speed over compression
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        subprocess.run(cmd, timeout=30, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception:
        return None

# 1. Page Configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🛡️",
    layout="wide"
)

# 2. UI/UX CSS Enhancements (Light/Dark Mode Compatible)
def inject_custom_css():
    st.markdown("""
    <style>
    /* Smooth fade-in effect for the main container */
    .main {
        animation: fadeIn 0.8s ease-in-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Responsive Result Cards */
    .result-card {
        padding: 20px;
        border-radius: 8px;
        background-color: rgba(128, 128, 128, 0.1); /* Adapts to light/dark */
        border-left: 6px solid;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .result-fake { border-left-color: #ff4b4b; } /* Streamlit Error Red */
    .result-real { border-left-color: #09ab3b; } /* Streamlit Success Green */
    
    .result-card h2 { margin-top: 0; font-size: 1.8rem; }
    .result-card p { margin-bottom: 0; opacity: 0.8; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# 3. Model Loading & Caching
@st.cache_resource(show_spinner=False)
def load_models():
    """
    Loads model weights into memory once at startup.
    Prevents memory overflow and eliminates loading delays during inference.
    The VideoDeepfakeDetector only loads a lightweight XGBoost classifier;
    heavy neural networks (CoAtNet, Whisper) are shared from img/aud models.
    """
    img_model = ImageDeepfakeDetector()
    aud_model = AudioDeepfakeDetector()
    vid_model = VideoDeepfakeDetector()
    return img_model, aud_model, vid_model

with st.spinner("Initializing Neural Networks... Please wait."):
    image_model, audio_model, video_model = load_models()

# 4. Sidebar Navigation
st.sidebar.title("🛡️ Deepfake Guard")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Select Analysis Module:",
    ["🖼️ Image Analysis", "🎙️ Audio Analysis", "🎥 Video Analysis"]
)

# 5. Image Analysis Module
if app_mode == "🖼️ Image Analysis":
    st.title("Image Deepfake Detection")
    st.markdown("Upload an image to verify if it is a genuine camera capture or an AI-generated forgery.")
    
    uploaded_image = st.file_uploader("Supported formats: JPG, PNG, JPEG", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### Source Image")
            with st.container(border=True):
                st.image(uploaded_image, width="content")
            
        with col2:
            st.markdown("### Forensic Analysis")
            if st.button("🔍 Run Visual Scan", use_container_width=True, type="primary"):
                
                # Interactive Terminal-like Status
                with st.status("Executing forensic pipeline...", expanded=True) as status:
                    st.write("Extracting Error Level Analysis (ELA) map...")
                    time.sleep(0.4) # Brief delay for visual UX
                    st.write("Computing Fast Fourier Transform (FFT) spectrum...")
                    time.sleep(0.4)
                    st.write("Running CoAtNet 5-Channel inference...")
                    
                    image_bytes = uploaded_image.read()
                    label, conf = image_model.predict(image_bytes)
                    
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                
                # Dynamic Results Formatting
                is_fake = "Fake" in label
                css_class = "result-fake" if is_fake else "result-real"
                icon = "⚠️" if is_fake else "✅"
                
                st.markdown(f"""
                <div class="result-card {css_class}">
                    <h2>{icon} {label}</h2>
                    <p>Diagnostic Conclusion</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(float(conf), text=f"Network Confidence Score: {conf*100:.2f}%")
                
                if is_fake:
                    st.error("High probability of AI generation or deepfake manipulation detected in the pixel frequency domain.")
                else:
                    st.success("The image artifacts are consistent with authentic, naturally captured photography.")

# 6. Audio Analysis Module
elif app_mode == "🎙️ Audio Analysis":
    st.title("Audio Spoofing Detection")
    st.markdown("Upload an audio recording to detect AI voice cloning, Text-To-Speech (TTS) algorithms, or deepfake audio.")
    
    uploaded_audio = st.file_uploader("Supported formats: WAV, FLAC, MP3", type=["wav", "flac", "mp3"])
    
    if uploaded_audio is not None:
        st.markdown("### Audio Playback")
        st.audio(uploaded_audio)
        st.markdown("---")
        
        if st.button("🎙️ Run Acoustic Scan", use_container_width=True, type="primary"):
            
            with st.status("Executing acoustic pipeline...", expanded=True) as status:
                st.write("Extracting raw audio waveform...")
                time.sleep(0.4)
                st.write("Generating temporal embeddings via Whisper Encoder...")
                time.sleep(0.4)
                st.write("Evaluating features through XGBoost topology...")
                
                temp_path = "temp_audio.flac"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                
                label, conf = audio_model.predict(temp_path)
                
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                status.update(label="Analysis Complete", state="complete", expanded=False)
            
            # Dynamic Results Formatting
            is_spoof = "Spoof" in label
            css_class = "result-fake" if is_spoof else "result-real"
            icon = "⚠️" if is_spoof else "✅"
            
            st.markdown(f"""
            <div class="result-card {css_class}">
                <h2>{icon} {label}</h2>
                <p>Diagnostic Conclusion</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(float(conf), text=f"Network Confidence Score: {conf*100:.2f}%")
            
            if is_spoof:
                st.error("Acoustic artifacts suggest this voice track was generated or altered by an AI model.")
            else:
                st.success("The recording features natural human vocal characteristics and biological breathing patterns.")

# 7. Video Analysis Module
elif app_mode == "🎥 Video Analysis":
    st.title("Video Deepfake Detection")
    st.markdown(
        "Upload a video to perform **multimodal forensic analysis** combining "
        "frame-level visual inspection (CoAtNet) with audio spoofing detection (Whisper), "
        "fused through an XGBoost classifier."
    )
    
    uploaded_video = st.file_uploader(
        "Supported formats: MP4, AVI, MOV, MKV",
        type=["mp4", "avi", "mov", "mkv"],
    )
    
    if uploaded_video is not None:
        # Save the original upload to disk (needed for both preview & analysis)
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_video.name)[1],
        ) as tmp_file:
            tmp_file.write(uploaded_video.getbuffer())
            temp_video_path = tmp_file.name

        # Transcode to H.264/yuv420p so the browser can always render it
        preview_path = transcode_for_browser(temp_video_path)

        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### Source Video")
            with st.container(border=True):
                if preview_path and os.path.exists(preview_path):
                    st.video(preview_path)
                else:
                    st.video(uploaded_video)  # fallback to original
        
        with col2:
            st.markdown("### Multimodal Forensic Analysis")
            if st.button("🎥 Run Video Scan", use_container_width=True, type="primary"):
                
                try:
                    with st.status("Executing multimodal forensic pipeline...", expanded=True) as status:
                        st.write("📹 Sampling 5 uniform frames from temporal axis...")
                        time.sleep(0.3)
                        st.write("🖼️ Extracting per-frame CoAtNet embeddings (ELA + FFT + RGB)...")
                        time.sleep(0.3)
                        st.write("📊 Applying statistical pooling (mean + max + std)...")
                        time.sleep(0.3)
                        st.write("🔊 Demuxing audio track via FFmpeg...")
                        time.sleep(0.3)
                        st.write("🎙️ Generating Whisper encoder embeddings...")
                        time.sleep(0.3)
                        st.write("🔗 Fusing visual and acoustic feature vectors (2816-dim)...")
                        time.sleep(0.3)
                        st.write("🧠 Running XGBoost fusion classifier...")
                        
                        label, conf = video_model.predict(
                            temp_video_path, image_model, audio_model
                        )
                        
                        status.update(
                            label="Analysis Complete", state="complete", expanded=False
                        )
                finally:
                    # Cleanup temporary files
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                    if preview_path and os.path.exists(preview_path):
                        os.remove(preview_path)
                
                # Handle error case
                if "Error" in label:
                    st.error(label)
                else:
                    # Dynamic Results Formatting
                    is_fake = "Fake" in label
                    css_class = "result-fake" if is_fake else "result-real"
                    icon = "⚠️" if is_fake else "✅"
                    
                    st.markdown(f"""
                    <div class="result-card {css_class}">
                        <h2>{icon} {label}</h2>
                        <p>Diagnostic Conclusion</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(
                        float(conf),
                        text=f"Network Confidence Score: {conf*100:.2f}%",
                    )
                    
                    if is_fake:
                        st.error(
                            "Multimodal analysis indicates deepfake manipulation. "
                            "Visual frame artifacts and/or audio spoofing signatures "
                            "were detected across the temporal and frequency domains."
                        )
                    else:
                        st.success(
                            "The video exhibits consistent audiovisual coherence. "
                            "Frame-level forensics and acoustic analysis suggest "
                            "authentic, unmanipulated content."
                        )