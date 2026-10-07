"""
FakeProbe-X: Reliability-Aware Multimodal Deepfake Forensics System.
Streamlit Web Interface providing multi-modal forensic inspection, reliability estimation,
and explainable forensic audit reporting.
"""

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
    AdaptiveEvidenceFusionEngine,
    ForensicReportGenerator,
    DecisionVerdict,
    QualityLevel,
    ReliabilityLevel,
    transcode_for_browser,
)

# 1. Page Configuration
st.set_page_config(
    page_title="FakeProbe-X Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom Modern Styling
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
    
    .verdict-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.0rem;
        margin-bottom: 1.2rem;
        border-left: 6px solid;
        background-color: rgba(128, 128, 128, 0.07);
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    
    .verdict-real {
        border-left-color: #10b981;
    }
    
    .verdict-fake {
        border-left-color: #ef4444;
    }

    .verdict-uncertain {
        border-left-color: #f59e0b;
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

    .badge-uncertain {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        padding: 0.45rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.35rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# 3. Cached Resource Loading
@st.cache_resource(show_spinner="Initializing FakeProbe-X Neural Forensics...")
def load_forensic_suite():
    img_det = ImageDeepfakeDetector()
    aud_det = AudioDeepfakeDetector()
    vid_det = VideoDeepfakeDetector()
    fusion = AdaptiveEvidenceFusionEngine()
    return img_det, aud_det, vid_det, fusion


try:
    image_detector, audio_detector, video_detector, fusion_engine = load_forensic_suite()
    system_ready = True
except Exception as e:
    st.error(f"❌ Failed to load forensic models: {e}")
    system_ready = False


# 4. Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡️ FakeProbe-X</div>
    <div class="subtitle">Reliability-Aware Multimodal Deepfake Forensics System</div>
</div>
""", unsafe_allow_html=True)


# 5. Sidebar System Diagnostics
with st.sidebar:
    st.header("⚙️ Forensic Engine")
    st.caption("Active Target Branch: `deepfake-integration`")
    st.divider()

    st.subheader("Model Status")
    st.write("🖼️ **Image Model**: CoAtNet-0 (5-Ch 224px)")
    st.write("🎙️ **Audio Model**: Whisper-Base + XGBoost")
    st.write("🎥 **Video Model**: Multimodal XGBoost (2816-dim)")
    st.write("🧠 **Decision Engine**: Reliability-Aware 3-Way")

    st.divider()
    st.subheader("Threshold Calibration")
    st.write("• Real Bound: `< 0.420`")
    st.write("• Deadband (Uncertain): `0.420 - 0.580`")
    st.write("• Fake Bound: `> 0.580`")
    st.write("• Min Reliability: `0.350`")


# 6. Tabbed User Interface
tab_image, tab_video, tab_audio = st.tabs(["🖼️ Image Forensics", "🎥 Video Forensics", "🎙️ Audio Forensics"])


def display_verdict_card(report):
    """Render unified verdict banner with quality, reliability, and confidence."""
    verdict = report.final_verdict.value
    css_class = "verdict-fake" if verdict == "FAKE" else ("verdict-real" if verdict == "REAL" else "verdict-uncertain")
    badge_class = "badge-fake" if verdict == "FAKE" else ("badge-real" if verdict == "REAL" else "badge-uncertain")
    icon = "🚨" if verdict == "FAKE" else ("✅" if verdict == "REAL" else "⚠️")

    st.markdown(f"""
    <div class="verdict-box {css_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
            <span class="{badge_class}">{icon} {verdict}</span>
            <span style="font-size: 1.1rem; font-weight: 600; color: #64748b;">
                Confidence: <strong style="color: #0f172a;">{report.confidence*100:.1f}%</strong>
            </span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div>
                <small style="color: #64748b;">Forensic Reliability</small><br>
                <strong>{report.reliability_level.value} ({report.reliability:.2f})</strong>
            </div>
            <div>
                <small style="color: #64748b;">Input Signal Quality</small><br>
                <strong>{report.quality.quality_level.value} ({report.quality.quality_score:.2f})</strong>
            </div>
            <div>
                <small style="color: #64748b;">Uncertainty Metric</small><br>
                <strong>{report.uncertainty_score:.2f} / 1.00</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# TAB 1: IMAGE FORENSICS
# ==========================================
with tab_image:
    st.markdown("### Image Deepfake & Forensic Analysis")
    st.write("Upload a portrait photo to evaluate spatial artifacts, Error Level Analysis (ELA), and 2D Fourier spectra.")

    uploaded_img = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader"
    )

    if uploaded_img is not None and system_ready:
        col_in1, col_in2 = st.columns([1, 1])
        img_bytes = uploaded_img.read()
        pil_view = Image.open(uploaded_img)

        with col_in1:
            st.image(pil_view, caption=f"Uploaded: {uploaded_img.name}", use_container_width=True)

        with col_in2:
            if st.button("🚀 Analyze Image", key="btn_img_analyze", type="primary"):
                with st.spinner("Executing CoAtNet-5ch, ELA, and Fourier Analysis..."):
                    t0 = time.time()
                    detector_res = image_detector.predict_structured(img_bytes, apply_face_crop=True)
                    report = fusion_engine.fuse_image_evidence(detector_res, sample_name=uploaded_img.name)
                    elapsed = time.time() - t0

                # Display Results
                display_verdict_card(report)

                # Probabilities & Details
                col_p1, col_p2 = st.columns(2)
                col_p1.metric("Fake Probability", f"{report.final_fake_probability*100:.1f}%")
                col_p2.metric("Real Probability", f"{report.final_real_probability*100:.1f}%")

                st.progress(report.final_fake_probability)

                # Evidence & Explanations
                st.subheader("🔍 Forensic Evidence & Reasoning")
                for reason in report.reasons:
                    st.write(f"• {reason}")

                # Visual Artifacts (ELA & FFT)
                st.subheader("🔬 Forensic Visualizations")
                col_v1, col_v2 = st.columns(2)
                if report.visual_artifacts.get("ela_map") is not None:
                    col_v1.image(report.visual_artifacts["ela_map"], caption="Error Level Analysis (ELA)", clamp=True, use_container_width=True)
                if report.visual_artifacts.get("fft_map") is not None:
                    col_v2.image(report.visual_artifacts["fft_map"], caption="2D Fast Fourier Transform (FFT) Magnitude", clamp=True, use_container_width=True)

                # Download Forensic Report
                md_report = ForensicReportGenerator.generate_markdown_report(report)
                st.download_button(
                    "📄 Download Forensic Audit Certificate (.md)",
                    data=md_report,
                    file_name=f"FakeProbeX_Report_{uploaded_img.name}.md",
                    mime="text/markdown"
                )


# ==========================================
# TAB 2: VIDEO FORENSICS
# ==========================================
with tab_video:
    st.markdown("### Video Multimodal Deepfake Forensics")
    st.write("Inspect video sequences with multi-frame temporal consistency, face tracking, and acoustic speech spoofing.")

    uploaded_vid = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="video_uploader"
    )

    if uploaded_vid is not None and system_ready:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_vid.read())
            tmp_video_path = tmp.name

        try:
            col_v1, col_v2 = st.columns([1, 1])
            with col_v1:
                st.video(tmp_video_path)

            with col_v2:
                num_frames = st.slider("Sampled Frame Count", min_value=3, max_value=10, value=5)
                if st.button("🚀 Analyze Video Stream", key="btn_vid_analyze", type="primary"):
                    with st.spinner("Decoding video, sampling frames, and evaluating multimodal fusion..."):
                        t0 = time.time()
                        detector_res = video_detector.predict_structured(
                            tmp_video_path,
                            image_detector,
                            audio_detector,
                            num_frames=num_frames
                        )
                        report = fusion_engine.fuse_video_evidence(detector_res, sample_name=uploaded_vid.name)
                        elapsed = time.time() - t0

                    display_verdict_card(report)

                    col_vp1, col_vp2 = st.columns(2)
                    col_vp1.metric("Multimodal Fake Probability", f"{report.final_fake_probability*100:.1f}%")
                    col_vp2.metric("Authentic Real Probability", f"{report.final_real_probability*100:.1f}%")

                    st.progress(report.final_fake_probability)

                    st.subheader("🔍 Forensic Evidence & Findings")
                    for reason in report.reasons:
                        st.write(f"• {reason}")

                    # Suspicious Timestamps
                    suspicious_frames = report.visual_artifacts.get("suspicious_frames", [])
                    if len(suspicious_frames) > 0:
                        st.subheader("⏱️ Suspicious Timestamps")
                        for sf in suspicious_frames:
                            st.warning(f"⚠️ Timestamp **{sf['timestamp_sec']}s** (Frame {sf['frame_index']}): Fake Prob **{sf['fake_probability']*100:.1f}%** ({sf['severity']} anomaly)")

                    # Frame Sample Gallery
                    sampled_frames = report.visual_artifacts.get("sampled_frames_rgb", [])
                    if len(sampled_frames) > 0:
                        st.subheader("🖼️ Sampled Temporal Frames")
                        st.image(sampled_frames, width=120)

                    md_report = ForensicReportGenerator.generate_markdown_report(report)
                    st.download_button(
                        "📄 Download Forensic Audit Certificate (.md)",
                        data=md_report,
                        file_name=f"FakeProbeX_Report_{uploaded_vid.name}.md",
                        mime="text/markdown"
                    )

        finally:
            if os.path.exists(tmp_video_path):
                try:
                    os.remove(tmp_video_path)
                except Exception:
                    pass


# ==========================================
# TAB 3: AUDIO FORENSICS
# ==========================================
with tab_audio:
    st.markdown("### Audio & Voice Cloning Forensics")
    st.write("Detect AI-generated speech, voice conversions, and neural vocoder artifacts via Whisper-Base + XGBoost.")

    uploaded_aud = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        key="audio_uploader"
    )

    if uploaded_aud is not None and system_ready:
        col_a1, col_a2 = st.columns([1, 1])
        aud_bytes = uploaded_aud.read()

        with col_a1:
            st.audio(aud_bytes)

        with col_a2:
            if st.button("🚀 Analyze Audio Track", key="btn_aud_analyze", type="primary"):
                with st.spinner("Extracting Whisper-Base acoustic embeddings and evaluating ASVspoof classifier..."):
                    t0 = time.time()
                    detector_res = audio_detector.predict_structured(aud_bytes, suffix=".wav")
                    report = fusion_engine.fuse_audio_evidence(detector_res, sample_name=uploaded_aud.name)
                    elapsed = time.time() - t0

                display_verdict_card(report)

                col_ap1, col_ap2 = st.columns(2)
                col_ap1.metric("Voice Spoof Probability", f"{report.final_fake_probability*100:.1f}%")
                col_ap2.metric("Bonafide Real Probability", f"{report.final_real_probability*100:.1f}%")

                st.progress(report.final_fake_probability)

                st.subheader("🔍 Acoustic Evidence & Findings")
                for reason in report.reasons:
                    st.write(f"• {reason}")

                md_report = ForensicReportGenerator.generate_markdown_report(report)
                st.download_button(
                    "📄 Download Forensic Audit Certificate (.md)",
                    data=md_report,
                    file_name=f"FakeProbeX_Report_{uploaded_aud.name}.md",
                    mime="text/markdown"
                )
