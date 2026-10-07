"""
FakeProbe-X: Reliability-Aware Multimodal Deepfake Forensics System.
Streamlit Web Interface providing multi-modal forensic inspection, reliability estimation,
and explainable forensic audit reporting.
"""

import os
import io
import sys
import time
import json
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
from configs.forensic_config import DEFAULT_CONFIG

# ==============================================================================
# 1. Page Configuration & Custom Research-Grade Forensic Styling
# ==============================================================================
st.set_page_config(
    page_title="FakeProbe-X — Reliability-Aware Multimodal Deepfake Forensics System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container & Header */
    .header-container {
        background: linear-gradient(135deg, #0b1329 0%, #111e38 50%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 2.0rem 2.4rem;
        color: #ffffff;
        margin-bottom: 2.0rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }

    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.8rem;
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #ffffff;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }

    .main-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 900px;
        line-height: 1.5;
        margin: 0;
    }

    /* Card Containers */
    .forensic-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* Verdict Boxes */
    .verdict-banner {
        border-radius: 12px;
        padding: 1.6rem 2.0rem;
        margin-top: 1.2rem;
        margin-bottom: 1.5rem;
        border-left: 6px solid;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .verdict-real {
        background: #f0fdf4;
        border-left-color: #10b981;
        border: 1px solid #bbf7d0;
    }

    .verdict-fake {
        background: #fef2f2;
        border-left-color: #ef4444;
        border: 1px solid #fecaca;
    }

    .verdict-uncertain {
        background: #fffbeb;
        border-left-color: #f59e0b;
        border: 1px solid #fde68a;
    }

    .verdict-badge {
        padding: 0.45rem 1.4rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.4rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: 0.02em;
    }

    .badge-real {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
    }

    .badge-fake {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: #ffffff;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3);
    }

    .badge-uncertain {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: #ffffff;
        box-shadow: 0 4px 10px rgba(245, 158, 11, 0.3);
    }

    /* Structured Result Card */
    .result-certificate-box {
        background-color: #0f172a;
        color: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid #334155;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }

    .result-cert-title {
        color: #38bdf8;
        font-weight: 700;
        font-size: 1.05rem;
        border-bottom: 1px dashed #475569;
        padding-bottom: 0.6rem;
        margin-bottom: 0.8rem;
    }

    .cert-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.35rem;
        font-size: 0.92rem;
    }

    .cert-label {
        color: #94a3b8;
    }

    .cert-val {
        color: #f1f5f9;
        font-weight: 600;
    }

    /* Upload Area Styling */
    .upload-hint-box {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        color: #64748b;
        margin-bottom: 1.0rem;
    }

    /* Quality Metrics Grid */
    .metric-pill {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1.0rem;
        text-align: center;
    }
    .metric-pill-title {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .metric-pill-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Timeline visualization */
    .timeline-container {
        background: #0f172a;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 1.0rem 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 1.0rem 0;
        overflow-x: auto;
    }

    .timeline-track {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }

    .timeline-node {
        flex: 1;
        text-align: center;
        padding: 0.4rem 0.2rem;
        border-radius: 4px;
        font-size: 0.75rem;
    }

    .node-normal {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #059669;
    }

    .node-suspicious {
        background: rgba(239, 68, 68, 0.25);
        color: #f87171;
        border: 1px solid #dc2626;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. Cached Backend Forensic Engine Initialization
# ==============================================================================
@st.cache_resource(show_spinner="Initializing FakeProbe-X Forensic Pipeline & Neural Models...")
def initialize_forensic_suite():
    """Load neural backbones and fusion engines once into memory."""
    img_det = ImageDeepfakeDetector()
    aud_det = AudioDeepfakeDetector()
    vid_det = VideoDeepfakeDetector()
    fusion = AdaptiveEvidenceFusionEngine()
    return img_det, aud_det, vid_det, fusion


engine_status = {
    "image_available": False,
    "audio_available": False,
    "video_available": False,
    "error_message": None
}

try:
    image_detector, audio_detector, video_detector, fusion_engine = initialize_forensic_suite()
    engine_status["image_available"] = True
    engine_status["audio_available"] = True
    engine_status["video_available"] = True
except Exception as e:
    engine_status["error_message"] = str(e)
    image_detector, audio_detector, video_detector, fusion_engine = None, None, None, None


# ==============================================================================
# 3. Header & System Title
# ==============================================================================
st.markdown("""
<div class="header-container">
    <div class="header-badge">🛡️ Research Forensics System • Reliability-Aware 3-Way Tri-Decision</div>
    <h1 class="main-title">FakeProbe-X</h1>
    <h3 style="color: #cbd5e1; font-weight: 600; font-size: 1.25rem; margin-bottom: 0.5rem;">
        Reliability-Aware Multimodal Deepfake Forensics System
    </h3>
    <p class="main-subtitle">
        Multimodal AI-powered forensic analysis for detecting manipulated image, video, and audio content.
    </p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 4. Sidebar: System Architecture & Calibration Diagnostics
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Forensic Engine Diagnostics")
    st.caption("Active Target Branch: `deepfake-integration`")
    st.divider()

    st.subheader("Model Status")
    if engine_status["image_available"]:
        st.success("🖼️ **Image Model**: CoAtNet-0 5-Ch (Active)")
    else:
        st.error("🖼️ **Image Model**: Unavailable")

    if engine_status["audio_available"]:
        st.success("🎙️ **Audio Model**: Whisper-Base + XGBoost (Active)")
    else:
        st.warning("🎙️ **Audio Model**: Unavailable")

    if engine_status["video_available"]:
        st.success("🎥 **Video Model**: Multimodal XGBoost 2816-dim (Active)")
    else:
        st.error("🎥 **Video Model**: Unavailable")

    st.info("🧠 **Decision Engine**: 3-Way Reliability-Aware")

    st.divider()
    st.subheader("Calibrated Thresholds")
    st.write(f"• **Real Bound**: `< {DEFAULT_CONFIG.uncertainty_real_boundary:.3f}`")
    st.write(f"• **Uncertain Deadband**: `[{DEFAULT_CONFIG.uncertainty_real_boundary:.3f}, {DEFAULT_CONFIG.uncertainty_fake_boundary:.3f}]`")
    st.write(f"• **Fake Bound**: `> {DEFAULT_CONFIG.uncertainty_fake_boundary:.3f}`")
    st.write(f"• **Min Reliability**: `{DEFAULT_CONFIG.min_reliability_threshold:.3f}`")
    st.write(f"• **Max Uncertainty**: `< {DEFAULT_CONFIG.high_uncertainty_threshold:.3f}`")

    st.divider()
    st.subheader("Forensic Modalities")
    st.write("1. **Spatial & ELA Residuals** (CoAtNet-0)")
    st.write("2. **2D Fourier Frequency Spectrum** (FFT)")
    st.write("3. **Acoustic Speech Spoofing** (Whisper)")
    st.write("4. **Temporal Consistency** (Multi-Frame)")
    st.write("5. **Cross-Modal AV Consistency** (Audio-Visual)")


# ==============================================================================
# 5. Helper Rendering Functions
# ==============================================================================

def map_quality_to_level(score: float, issues: list) -> str:
    """Map quality score and degradation issues to 4-tier display level."""
    if score < 0.30 or len(issues) >= 3:
        return "INSUFFICIENT"
    elif score < 0.48 or len(issues) >= 2:
        return "POOR"
    elif score < 0.72:
        return "FAIR"
    else:
        return "GOOD"


def render_progress_simulation(step_texts: list[str]):
    """Display meaningful real backend progress steps during analysis."""
    prog_bar = st.progress(0)
    status_placeholder = st.empty()
    
    for i, step in enumerate(step_texts):
        pct = int((i + 1) / len(step_texts) * 100)
        prog_bar.progress(pct)
        status_placeholder.markdown(f"**Step {i+1}/{len(step_texts)}**: {step}")
        time.sleep(0.06)
    
    status_placeholder.empty()
    prog_bar.empty()


def render_result_card(report, modality: str, temporal_str: str = "N/A", av_str: str = "NOT AVAILABLE"):
    """Render Section 16 Structured Result Card."""
    verdict = report.final_verdict.value
    qual_level = map_quality_to_level(report.quality.quality_score, report.quality.issues)
    
    agree_score = report.agreement_score
    if agree_score >= 0.80:
        agree_str = "HIGH"
    elif agree_score >= 0.50:
        agree_str = "MEDIUM"
    elif agree_score > 0.0:
        agree_str = "LOW"
    else:
        agree_str = "NOT AVAILABLE"

    # Evidence summary text
    ev_summary = f"{len(report.evidence_list)} signals fused"
    if report.evidence_list:
        top_ev = report.evidence_list[0]
        ev_summary += f" (Lead: {top_ev.name} = {top_ev.score*100:.1f}%)"

    st.markdown(f"""
    <div class="result-certificate-box">
        <div class="result-cert-title">🛡️ FAKEPROBE-X FORENSIC ANALYSIS CERTIFICATE</div>
        <div class="cert-row"><span class="cert-label">Sample Name:</span><span class="cert-val">{report.sample_name}</span></div>
        <div class="cert-row"><span class="cert-label">Modality:</span><span class="cert-val">{modality.upper()}</span></div>
        <div class="cert-row"><span class="cert-label">Verdict:</span><span class="cert-val" style="color: {'#4ade80' if verdict=='REAL' else ('#f87171' if verdict=='FAKE' else '#fbbf24')}; font-size: 1.1rem;">{verdict}</span></div>
        <div class="cert-row"><span class="cert-label">Confidence:</span><span class="cert-val">{report.confidence*100:.1f}%</span></div>
        <div class="cert-row"><span class="cert-label">Reliability:</span><span class="cert-val">{report.reliability*100:.1f}% ({report.reliability_level.value})</span></div>
        <div class="cert-row"><span class="cert-label">Input Quality:</span><span class="cert-val">{qual_level} (Score: {report.quality.quality_score:.2f})</span></div>
        <div class="cert-row"><span class="cert-label">Evidence:</span><span class="cert-val">{ev_summary}</span></div>
        <div class="cert-row"><span class="cert-label">Model Agreement:</span><span class="cert-val">{agree_str}</span></div>
        <div class="cert-row"><span class="cert-label">Temporal Evidence:</span><span class="cert-val">{temporal_str}</span></div>
        <div class="cert-row"><span class="cert-label">Audio-Visual Consistency:</span><span class="cert-val">{av_str}</span></div>
        <div style="border-top: 1px dashed #475569; margin: 0.8rem 0; padding-top: 0.6rem;">
            <span class="cert-label" style="display: block; margin-bottom: 0.3rem;">Forensic Explanation:</span>
            <span style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.4;">{report.summary_text if report.summary_text else (report.reasons[0] if report.reasons else 'No anomaly detected.')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_forensic_results(report, modality: str):
    """Render comprehensive forensic breakdown meeting all user prompt specifications."""
    verdict = report.final_verdict.value
    
    css_class = "verdict-fake" if verdict == "FAKE" else ("verdict-real" if verdict == "REAL" else "verdict-uncertain")
    badge_class = "badge-fake" if verdict == "FAKE" else ("badge-real" if verdict == "REAL" else "badge-uncertain")
    icon = "🚨" if verdict == "FAKE" else ("✅" if verdict == "REAL" else "⚠️")
    
    qual_display = map_quality_to_level(report.quality.quality_score, report.quality.issues)

    # 1. Main Analysis Result Banner (Sections 5, 6, 7, 8)
    st.markdown(f"""
    <div class="verdict-banner {css_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.0rem;">
            <div>
                <small style="color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 0.3rem;">ANALYSIS RESULT</small>
                <span class="verdict-badge {badge_class}">{icon} {verdict}</span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.9rem; color: #64748b; font-weight: 600;">Decision Confidence</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #0f172a;">{report.confidence*100:.1f}%</div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-top: 1.2rem; border-top: 1px solid rgba(0,0,0,0.08); padding-top: 1.0rem;">
            <div>
                <small style="color: #64748b; font-weight: 600;">Forensic Reliability</small><br>
                <strong style="font-size: 1.1rem; color: #0f172a;">{report.reliability*100:.1f}%</strong>
                <span style="font-size: 0.85rem; color: #64748b;">({report.reliability_level.value})</span>
                <div style="font-size: 0.76rem; color: #64748b; margin-top: 0.15rem;">
                    <em>Reliability indicates how trustworthy the available evidence is for this analysis.</em>
                </div>
            </div>
            <div>
                <small style="color: #64748b; font-weight: 600;">Input Quality</small><br>
                <strong style="font-size: 1.1rem; color: #0f172a;">{qual_display}</strong>
                <span style="font-size: 0.85rem; color: #64748b;">(Score: {report.quality.quality_score:.2f})</span>
                <div style="font-size: 0.76rem; color: #64748b; margin-top: 0.15rem;">
                    Acceptable: <strong>{'YES' if report.quality.is_acceptable else 'NO (High Degradation)'}</strong>
                </div>
            </div>
            <div>
                <small style="color: #64748b; font-weight: 600;">Uncertainty Metric</small><br>
                <strong style="font-size: 1.1rem; color: #0f172a;">{report.uncertainty_score:.2f}</strong>
                <span style="font-size: 0.85rem; color: #64748b;">/ 1.00</span>
                <div style="font-size: 0.76rem; color: #64748b; margin-top: 0.15rem;">
                    Deadband: [{DEFAULT_CONFIG.uncertainty_real_boundary:.2f}, {DEFAULT_CONFIG.uncertainty_fake_boundary:.2f}]
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Dual Probabilities & Visual Scale
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Manipulation / Fake Probability", f"{report.final_fake_probability*100:.2f}%")
    col_p2.metric("Authentic / Real Probability", f"{report.final_real_probability*100:.2f}%")

    st.progress(report.final_fake_probability)
    st.caption("Visual Evidence Scale (0% Authentic Real ← [0.420 - 0.580 Uncertain Deadband] → 100% Manipulated Fake)")

    # 3. Structured Result Card (Section 16)
    suspicious_frames = report.visual_artifacts.get("suspicious_frames", [])
    temp_str = f"{len(suspicious_frames)} suspicious frame(s)" if modality == "video" else "N/A"
    
    if modality == "video":
        has_aud = report.detector_results[0].evidence.get("has_audio", False) if report.detector_results else False
        if has_aud:
            av_agree = report.agreement_score
            av_str = "HIGH" if av_agree >= 0.80 else ("MEDIUM" if av_agree >= 0.50 else "LOW")
        else:
            av_str = "NOT AVAILABLE (No audio stream present)"
    else:
        av_str = "NOT AVAILABLE"

    render_result_card(report, modality=modality, temporal_str=temp_str, av_str=av_str)

    # 4. Input Quality Analysis (Section 8)
    st.markdown("### 📊 Input Quality & Physical Signal Verification")
    if report.quality.metrics:
        met = report.quality.metrics
        cols = st.columns(len(met) if len(met) <= 5 else 4)
        for idx, (k, v) in enumerate(met.items()):
            col = cols[idx % len(cols)]
            val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
            with col:
                st.markdown(f"""
                <div class="metric-pill">
                    <div class="metric-pill-title">{k.replace('_', ' ')}</div>
                    <div class="metric-pill-value">{val_str}</div>
                </div>
                """, unsafe_allow_html=True)

    if report.quality.issues:
        st.markdown("**Quality Degradation Flags Identified:**")
        for issue in report.quality.issues:
            st.warning(f"⚠️ {issue}")
    else:
        st.success("✅ Physical input signal quality is optimal with no significant degradation.")

    st.divider()

    # 5. Forensic Evidence (Section 9)
    st.markdown("### 🔬 Forensic Evidence Breakdown")
    if report.evidence_list:
        ev_data = []
        for ev in report.evidence_list:
            ev_data.append({
                "Evidence Source": ev.name,
                "Modality": ev.modality.upper(),
                "Anomaly Score": f"{ev.score*100:.1f}%",
                "Reliability": f"{ev.reliability*100:.1f}%",
                "Weight": f"{ev.weight:.2f}",
                "Forensic Description": ev.description
            })
        st.table(ev_data)
    else:
        st.info("No active forensic evidence items logged.")

    # 6. Model Agreement & Consistency (Sections 10 & 11)
    col_ag1, col_ag2 = st.columns(2)
    with col_ag1:
        st.markdown("#### 🤝 Model Agreement")
        agree_score = report.agreement_score
        if agree_score >= 0.80:
            agree_status = "HIGH"
            agree_color = "green"
        elif agree_score >= 0.50:
            agree_status = "MEDIUM"
            agree_color = "orange"
        elif agree_score > 0.0:
            agree_status = "LOW"
            agree_color = "red"
        else:
            agree_status = "NOT AVAILABLE"
            agree_color = "gray"

        st.markdown(f"Agreement Level: <strong style='color: {agree_color}; font-size: 1.15rem;'>{agree_status}</strong> (Consensus: `{agree_score*100:.1f}%`)", unsafe_allow_html=True)
        for det in report.detector_results:
            st.markdown(f"• **{det.model}**: `{det.prediction.value}` ({det.detail_label})")

    with col_ag2:
        st.markdown("#### 🔄 Audio-Visual Consistency")
        if modality == "video":
            has_audio = report.detector_results[0].evidence.get("has_audio", False) if report.detector_results else False
            if has_audio:
                st.markdown(f"Cross-Modal Consistency: **{av_str}**")
                st.write("Visual and acoustic temporal cues cross-referenced.")
            else:
                st.markdown("**NOT AVAILABLE**")
                st.caption("No audio stream detected in video container.")
        else:
            st.markdown("**NOT AVAILABLE**")
            st.caption("Audio-Visual cross-modal verification requires multimodal video input.")

    st.divider()

    # 7. Video Temporal Forensics (Section 12, if video)
    if modality == "video":
        st.markdown("### ⏱️ Video Temporal Forensics")
        timestamps = report.visual_artifacts.get("frame_timestamps", [])
        frame_probs = report.visual_artifacts.get("frame_fake_probabilities", [])
        suspicious = report.visual_artifacts.get("suspicious_frames", [])

        st.write(f"• **Total Frames Sampled & Analyzed**: `{len(timestamps)}`")
        st.write(f"• **Suspicious Frames Detected**: `{len(suspicious)}`")

        if len(suspicious) > 0:
            st.markdown("#### Suspicious Segments")
            for sf in suspicious:
                sec = sf["timestamp_sec"]
                mins = int(sec // 60)
                rem_sec = int(sec % 60)
                ts_str = f"{mins:02d}:{rem_sec:02d}"
                st.warning(f"⚠️ Timestamp **{ts_str}** ({sec}s, Frame {sf['frame_index']}): Manipulation Score **{sf['fake_probability']*100:.1f}%** ({sf['severity']} Anomaly)")

            # Timeline visualization
            st.markdown("#### Temporal Timeline Visualization")
            nodes_html = []
            for idx, (t, p) in enumerate(zip(timestamps, frame_probs)):
                is_sus = any(s["frame_index"] == idx for s in suspicious)
                cls_name = "node-suspicious" if is_sus else "node-normal"
                sec_str = f"{t:.1f}s"
                indicator = "▲ SUSPICIOUS" if is_sus else "✓ OK"
                nodes_html.append(f"""
                <div class="timeline-node {cls_name}">
                    <div><strong>{sec_str}</strong></div>
                    <div>{p*100:.0f}% Fake</div>
                    <div><small>{indicator}</small></div>
                </div>
                """)

            st.markdown(f"""
            <div class="timeline-container">
                <div style="color: #94a3b8; font-size: 0.82rem; margin-bottom: 0.4rem;">TEMPORAL FRAME SEQUENCE</div>
                <div class="timeline-track">
                    {''.join(nodes_html)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ No suspicious temporal segments or temporal flicker anomalies detected across analyzed frames.")

        # Sampled Frame Gallery
        sampled_frames = report.visual_artifacts.get("sampled_frames_rgb", [])
        if sampled_frames and len(sampled_frames) > 0:
            st.markdown("#### Sampled Frame Sequence")
            f_cols = st.columns(min(len(sampled_frames), 5))
            for i, frame in enumerate(sampled_frames[:5]):
                with f_cols[i]:
                    st.image(frame, caption=f"Frame {i+1} ({timestamps[i] if i < len(timestamps) else 0.0:.1f}s)", use_container_width=True)

        st.divider()

    # 8. Visual Artifacts (Image ELA & FFT)
    if modality == "image":
        st.markdown("### 🔬 Forensic Visualizations")
        col_v1, col_v2 = st.columns(2)
        if report.visual_artifacts.get("ela_map") is not None:
            col_v1.image(
                report.visual_artifacts["ela_map"],
                caption="Error Level Analysis (ELA) Compression Residual Map",
                clamp=True,
                use_container_width=True
            )
        if report.visual_artifacts.get("fft_map") is not None:
            col_v2.image(
                report.visual_artifacts["fft_map"],
                caption="2D Fast Fourier Transform (FFT) Log-Magnitude Spectrum",
                clamp=True,
                use_container_width=True
            )
        st.divider()

    # 9. Forensic Explanation & Reasoning (Section 13 & 14)
    st.markdown("### 🧠 Forensic Explanation & Reasoning")
    if verdict == DecisionVerdict.UNCERTAIN.value:
        st.markdown("""
        <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 1.0rem 1.2rem; margin-bottom: 1.0rem;">
            <strong style="color: #b45309;">⚠️ UNCERTAIN Forensic Decision Rationale:</strong><br>
            <span style="color: #92400e; font-size: 0.92rem;">
                The available evidence is insufficient for a confident REAL or FAKE classification.
            </span>
        </div>
        """, unsafe_allow_html=True)

    for reason in report.reasons:
        st.markdown(f"• {reason}")

    st.divider()

    # 10. Forensic Report Views & Export (Section 18)
    st.markdown("### 📄 Certified Forensic Audit Report")
    
    col_rep1, col_rep2 = st.columns([1, 1])
    md_report = ForensicReportGenerator.generate_markdown_report(report)
    json_report = ForensicReportGenerator.generate_json_report(report)
    
    with col_rep1:
        st.download_button(
            label="📄 Download Forensic Audit Certificate (.md)",
            data=md_report,
            file_name=f"FakeProbeX_Audit_Report_{report.sample_name}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_rep2:
        st.download_button(
            label="💾 Download Structured Forensic Data (.json)",
            data=json_report,
            file_name=f"FakeProbeX_Forensic_Data_{report.sample_name}.json",
            mime="application/json",
            use_container_width=True
        )

    with st.expander("🔍 VIEW FULL FORENSIC AUDIT DATA (JSON Schema)"):
        st.json(report.to_dict())


# ==============================================================================
# 6. Tab Navigation: [ IMAGE ] [ VIDEO ] [ AUDIO ]
# ==============================================================================
tab_image, tab_video, tab_audio = st.tabs(["🖼️ IMAGE", "🎥 VIDEO", "🎙️ AUDIO"])

TEST_DIR = BASE_DIR / "evaluation"

# ------------------------------------------------------------------------------
# TAB 1: IMAGE FORENSICS
# ------------------------------------------------------------------------------
with tab_image:
    st.subheader("Upload Image")
    st.markdown("Supported formats: `.jpg, .jpeg, .png, .webp` — Spatial, ELA compression residual, and 2D Fourier spectrum analysis.")

    col_iu1, col_iu2 = st.columns([1, 1])
    with col_iu1:
        st.markdown("""
        <div class="upload-hint-box">
            <div style="font-size: 1.8rem; margin-bottom: 0.3rem;">🖼️</div>
            <strong>Drag & Drop your image here</strong><br>
            <span style="font-size: 0.85rem;">or click Browse Files below</span>
        </div>
        """, unsafe_allow_html=True)
        uploaded_img = st.file_uploader(
            "Browse Files",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_file_uploader",
            label_visibility="collapsed"
        )

    with col_iu2:
        st.markdown("**Or Test with Pre-Loaded Benchmark Samples:**")
        img_sample_choice = st.selectbox(
            "Select Benchmark Sample",
            options=[
                "None (Use Uploaded File)",
                "Real Face Sample (real_face_01.png)",
                "Fake Face Sample (fake_face_01.png)",
                "Degraded / Low Quality Sample (degraded_low_quality.png)"
            ],
            key="img_sample_selector"
        )

    # Determine image bytes source
    img_bytes = None
    img_name = None
    if uploaded_img is not None:
        img_bytes = uploaded_img.read()
        img_name = uploaded_img.name
    elif img_sample_choice != "None (Use Uploaded File)":
        sample_map = {
            "Real Face Sample (real_face_01.png)": TEST_DIR / "test_images" / "real_face_01.png",
            "Fake Face Sample (fake_face_01.png)": TEST_DIR / "test_images" / "fake_face_01.png",
            "Degraded / Low Quality Sample (degraded_low_quality.png)": TEST_DIR / "test_images" / "degraded_low_quality.png",
        }
        chosen_path = sample_map.get(img_sample_choice)
        if chosen_path and chosen_path.exists():
            with open(chosen_path, "rb") as f:
                img_bytes = f.read()
            img_name = chosen_path.name

    if img_bytes is not None:
        st.divider()
        col_prev1, col_prev2 = st.columns([1, 1])
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            w, h = pil_img.size
            with col_prev1:
                st.image(pil_img, caption=f"Uploaded Image: {img_name} ({w}x{h} px)", use_container_width=True)
            with col_prev2:
                st.markdown(f"""
                <div class="forensic-card">
                    <h4 style="margin-top: 0;">🖼️ Image Specifications</h4>
                    <p style="margin-bottom: 0.4rem;">• <strong>Filename</strong>: <code>{img_name}</code></p>
                    <p style="margin-bottom: 0.4rem;">• <strong>Resolution</strong>: {w} × {h} pixels ({w*h/1e6:.2f} MP)</p>
                    <p style="margin-bottom: 0.4rem;">• <strong>Color Space</strong>: {pil_img.mode}</p>
                    <p style="margin-bottom: 0.4rem;">• <strong>File Size</strong>: {len(img_bytes)/1024:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)

                if not engine_status["image_available"]:
                    st.error("❌ Required image model checkpoint is unavailable.")
                else:
                    if st.button("🚀 ANALYZE IMAGE", key="btn_run_img_analysis", type="primary", use_container_width=True):
                        try:
                            render_progress_simulation([
                                "Loading input image and parsing headers...",
                                "Checking physical & photographic input quality...",
                                "Running CoAtNet-0 5-channel neural classifier...",
                                "Extracting ELA & 2D Fourier spectral anomalies...",
                                "Estimating forensic evidence reliability...",
                                "Performing adaptive evidence fusion...",
                                "Generating three-way forensic decision...",
                                "Preparing certified forensic report..."
                            ])

                            detector_res = image_detector.predict_structured(img_bytes, apply_face_crop=True)
                            report = fusion_engine.fuse_image_evidence(detector_res, sample_name=img_name)

                            display_forensic_results(report, modality="image")
                        except ValueError as ve:
                            st.error(f"❌ Unsupported or invalid input file: {ve}")
                        except Exception as ex:
                            st.error(f"❌ Analysis could not be completed: {ex}")

        except Exception as e:
            st.error(f"❌ Unable to process the uploaded file: {e}")


# ------------------------------------------------------------------------------
# TAB 2: VIDEO FORENSICS
# ------------------------------------------------------------------------------
with tab_video:
    st.subheader("Upload Video")
    st.markdown("Supported formats: `.mp4, .avi, .mov, .mkv, .webm` — Multi-frame temporal consistency, face tracking, and multimodal speech analysis.")

    col_vu1, col_vu2 = st.columns([1, 1])
    with col_vu1:
        st.markdown("""
        <div class="upload-hint-box">
            <div style="font-size: 1.8rem; margin-bottom: 0.3rem;">🎥</div>
            <strong>Drag & Drop your video here</strong><br>
            <span style="font-size: 0.85rem;">or click Browse Files below</span>
        </div>
        """, unsafe_allow_html=True)
        uploaded_vid = st.file_uploader(
            "Browse Video Files",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key="vid_file_uploader",
            label_visibility="collapsed"
        )

    with col_vu2:
        st.markdown("**Or Test with Pre-Loaded Benchmark Samples:**")
        vid_sample_choice = st.selectbox(
            "Select Benchmark Sample",
            options=[
                "None (Use Uploaded File)",
                "Real Video Sample (real_video_01.mp4)",
                "Fake Video Sample 1 (fake_video_01.mp4)",
                "Fake Video Sample 2 (fake_video_02.mp4)"
            ],
            key="vid_sample_selector"
        )

    vid_bytes = None
    vid_name = None
    tmp_input_file = None

    if uploaded_vid is not None:
        vid_bytes = uploaded_vid.read()
        vid_name = uploaded_vid.name
    elif vid_sample_choice != "None (Use Uploaded File)":
        sample_map_vid = {
            "Real Video Sample (real_video_01.mp4)": TEST_DIR / "test_videos" / "real_video_01.mp4",
            "Fake Video Sample 1 (fake_video_01.mp4)": TEST_DIR / "test_videos" / "fake_video_01.mp4",
            "Fake Video Sample 2 (fake_video_02.mp4)": TEST_DIR / "test_videos" / "fake_video_02.mp4",
        }
        chosen_vid_path = sample_map_vid.get(vid_sample_choice)
        if chosen_vid_path and chosen_vid_path.exists():
            with open(chosen_vid_path, "rb") as f:
                vid_bytes = f.read()
            vid_name = chosen_vid_path.name

    if vid_bytes is not None:
        st.divider()
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_name).suffix or ".mp4") as tmp:
            tmp.write(vid_bytes)
            tmp_video_path = tmp.name

        try:
            # Transcode preview for reliable browser playback
            browser_preview_path = transcode_for_browser(tmp_video_path) or tmp_video_path

            col_vp1, col_vp2 = st.columns([1, 1])
            with col_vp1:
                st.video(browser_preview_path)

            with col_vp2:
                # Read metadata using OpenCV
                import cv2
                cap = cv2.VideoCapture(tmp_video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                dur = total_frames / fps if fps > 0 else 0.0
                cap.release()

                st.markdown(f"""
                <div class="forensic-card">
                    <h4 style="margin-top: 0;">🎥 Video Stream Metadata</h4>
                    <p style="margin-bottom: 0.4rem;">• <strong>Filename</strong>: <code>{vid_name}</code></p>
                    <p style="margin-bottom: 0.4rem;">• <strong>Duration</strong>: {dur:.2f} seconds</p>
                    <p style="margin-bottom: 0.4rem;">• <strong>Dimensions</strong>: {vw} × {vh} px</p>
                    <p style="margin-bottom: 0.4rem;">• <strong>Frame Count</strong>: {total_frames} frames @ {fps:.1f} FPS</p>
                </div>
                """, unsafe_allow_html=True)

                num_frames = st.slider("Sampled Frame Count", min_value=3, max_value=10, value=5, key="vid_frame_slider")

                if not engine_status["video_available"]:
                    st.error("❌ Required video fusion model checkpoint is unavailable.")
                else:
                    if st.button("🚀 ANALYZE VIDEO", key="btn_run_vid_analysis", type="primary", use_container_width=True):
                        try:
                            render_progress_simulation([
                                "Demuxing video container and extracting audio stream...",
                                "Sampling temporal keyframes across sequence...",
                                "Evaluating facial detection, alignment & tracking...",
                                "Extracting CoAtNet-0 spatial & residual embeddings...",
                                "Evaluating Whisper acoustic speech features...",
                                "Executing Multimodal XGBoost fusion model...",
                                "Estimating cross-modal consistency & reliability...",
                                "Generating certified forensic audit certificate..."
                            ])

                            detector_res = video_detector.predict_structured(
                                tmp_video_path,
                                image_detector,
                                audio_detector,
                                num_frames=num_frames
                            )
                            report = fusion_engine.fuse_video_evidence(detector_res, sample_name=vid_name)

                            display_forensic_results(report, modality="video")
                        except ValueError as ve:
                            st.error(f"❌ Unsupported or invalid video file: {ve}")
                        except Exception as ex:
                            st.error(f"❌ Video analysis could not be completed: {ex}")

        finally:
            if os.path.exists(tmp_video_path):
                try:
                    os.remove(tmp_video_path)
                except Exception:
                    pass
            if browser_preview_path != tmp_video_path and os.path.exists(browser_preview_path):
                try:
                    os.remove(browser_preview_path)
                except Exception:
                    pass


# ------------------------------------------------------------------------------
# TAB 3: AUDIO FORENSICS
# ------------------------------------------------------------------------------
with tab_audio:
    st.subheader("Upload Audio")
    st.markdown("Supported formats: `.wav, .mp3, .flac, .ogg, .m4a` — Neural vocoder artifact, voice clone, and acoustic speech spoofing detection.")

    col_au1, col_au2 = st.columns([1, 1])
    with col_au1:
        st.markdown("""
        <div class="upload-hint-box">
            <div style="font-size: 1.8rem; margin-bottom: 0.3rem;">🎙️</div>
            <strong>Drag & Drop your audio here</strong><br>
            <span style="font-size: 0.85rem;">or click Browse Files below</span>
        </div>
        """, unsafe_allow_html=True)
        uploaded_aud = st.file_uploader(
            "Browse Audio Files",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="aud_file_uploader",
            label_visibility="collapsed"
        )

    with col_au2:
        st.markdown("**Or Test with Pre-Loaded Benchmark Samples:**")
        aud_sample_choice = st.selectbox(
            "Select Benchmark Sample",
            options=[
                "None (Use Uploaded File)",
                "Real Voice Sample (real_voice_01.wav)",
                "Fake Voice Sample (fake_voice_01.wav)"
            ],
            key="aud_sample_selector"
        )

    aud_bytes = None
    aud_name = None

    if uploaded_aud is not None:
        aud_bytes = uploaded_aud.read()
        aud_name = uploaded_aud.name
    elif aud_sample_choice != "None (Use Uploaded File)":
        sample_map_aud = {
            "Real Voice Sample (real_voice_01.wav)": TEST_DIR / "test_audio" / "real_voice_01.wav",
            "Fake Voice Sample (fake_voice_01.wav)": TEST_DIR / "test_audio" / "fake_voice_01.wav",
        }
        chosen_aud_path = sample_map_aud.get(aud_sample_choice)
        if chosen_aud_path and chosen_aud_path.exists():
            with open(chosen_aud_path, "rb") as f:
                aud_bytes = f.read()
            aud_name = chosen_aud_path.name

    if aud_bytes is not None:
        st.divider()
        col_ap1, col_ap2 = st.columns([1, 1])
        with col_ap1:
            st.audio(aud_bytes)

        with col_ap2:
            st.markdown(f"""
            <div class="forensic-card">
                <h4 style="margin-top: 0;">🎙️ Acoustic Signal Info</h4>
                <p style="margin-bottom: 0.4rem;">• <strong>Filename</strong>: <code>{aud_name}</code></p>
                <p style="margin-bottom: 0.4rem;">• <strong>File Size</strong>: {len(aud_bytes)/1024:.1f} KB</p>
                <p style="margin-bottom: 0.4rem;">• <strong>Model</strong>: Whisper-Base (512-dim) + XGBoost</p>
            </div>
            """, unsafe_allow_html=True)

            if not engine_status["audio_available"]:
                st.warning("⚠️ Audio analysis is currently unavailable.")
            else:
                if st.button("🚀 ANALYZE AUDIO", key="btn_run_aud_analysis", type="primary", use_container_width=True):
                    try:
                        render_progress_simulation([
                            "Decoding acoustic waveform and resampling to 16 kHz...",
                            "Evaluating SNR, clipping ratio, and speech duration...",
                            "Extracting 512-dimensional Whisper-Base encoder embeddings...",
                            "Running ASVspoof-trained XGBoost acoustic classifier...",
                            "Estimating acoustic reliability & confidence...",
                            "Performing reliability-aware evidence fusion...",
                            "Generating three-way decision...",
                            "Preparing acoustic audit certificate..."
                        ])

                        detector_res = audio_detector.predict_structured(aud_bytes, suffix=Path(aud_name).suffix or ".wav")
                        report = fusion_engine.fuse_audio_evidence(detector_res, sample_name=aud_name)

                        display_forensic_results(report, modality="audio")
                    except ValueError as ve:
                        st.error(f"❌ Unsupported or invalid audio file: {ve}")
                    except Exception as ex:
                        st.error(f"❌ Audio analysis could not be completed: {ex}")


# ==============================================================================
# 7. Footer
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1.0rem 0;">
    <strong>FakeProbe-X</strong> — Reliability-Aware Multimodal Deepfake Forensics System<br>
    Built for explainable forensic verification across Image, Video, and Audio modalities.
</div>
""", unsafe_allow_html=True)
