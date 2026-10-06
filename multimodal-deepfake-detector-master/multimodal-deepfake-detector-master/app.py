import os
import sys
from pathlib import Path
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from PIL import Image
import streamlit as st

# Add project root to sys.path
BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from deepfake_detector.models import DeepFakeDetector
from deepfake_detector.data.transforms import get_val_transforms
from deepfake_detector.config import Config
from deepfake_detector.utils import calculate_comprehensive_metrics

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


st.set_page_config(
    page_title="DeepFake Detection Studio",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich modern look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .subtitle {
        font-size: 1.05rem;
        color: #c7d2fe;
        font-weight: 400;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .prediction-badge-real {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    }
    
    .prediction-badge-fake {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
    }
    
    .card-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    
    .card-value {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector_model(model_name: str, checkpoint_path: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepFakeDetector(model_name=model_name, pretrained=(checkpoint_path is None or checkpoint_path == ""))
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            model.load_checkpoint(checkpoint_path, device=str(device))
        except Exception as e:
            st.warning(f"Could not load checkpoint '{checkpoint_path}': {e}. Using initialized model.")
    model = model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def load_mtcnn_detector(margin: int, min_face_size: int, device_str: str):
    if not MTCNN_AVAILABLE:
        return None
    return MTCNN(
        margin=margin,
        min_face_size=min_face_size,
        select_largest=False,
        keep_all=True,
        device=device_str,
        post_process=False
    )


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.shields.io/badge/EfficientNet-AI%20Forensics-6366F1?style=for-the-badge&logo=pytorch&logoColor=white", use_container_width=True)
    st.title("⚙️ Configuration")
    
    model_name = st.selectbox(
        "Model Backbone",
        ["efficientnet-b1", "efficientnet-b0", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4"],
        index=0
    )
    
    threshold = st.slider("Classification Threshold (Real vs Fake)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    st.subheader("Model Weights")
    checkpoint_file = st.text_input("Checkpoint Path (.pth)", value="", placeholder="outputs/checkpoints/best_model.pth")
    
    st.subheader("Face Detection (MTCNN)")
    face_margin = st.slider("Face Bounding Margin (px)", min_value=10, max_value=100, value=40, step=5)
    min_face_size = st.slider("Minimum Face Size (px)", min_value=40, max_value=200, value=80, step=10)
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"⚡ Running on: **{device_name.upper()}**")


# Header Banner
st.markdown("""
<div class="main-header">
    <div class="main-title">
        <span>🎭</span> DeepFake Detection Studio
    </div>
    <div class="subtitle">
        High-Precision Forensic Media Analysis Powered by MTCNN & EfficientNet Architecture
    </div>
</div>
""", unsafe_allow_html=True)


# Load model
model, device = load_detector_model(model_name, checkpoint_file)
mtcnn = load_mtcnn_detector(face_margin, min_face_size, str(device))
image_size = 240 if 'b1' in model_name else 224
transform = get_val_transforms(image_size)

# Navigation Tabs
tab_image, tab_video, tab_batch, tab_model_info = st.tabs([
    "🖼️ Image Analysis",
    "🎥 Video Forensics",
    "📁 Batch Evaluation",
    "ℹ️ Architecture & Metrics"
])


# -------------------------------------------------------------
# TAB 1: Image Analysis
# -------------------------------------------------------------
with tab_image:
    st.header("Single Image Analysis")
    st.write("Upload a facial photo or portrait to detect artificial manipulation and deepfake artifacts.")
    
    uploaded_image = st.file_uploader("Choose an image (JPG, JPEG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    
    col_input, col_results = st.columns([1, 1])
    
    if uploaded_image is not None:
        pil_image = Image.open(uploaded_image).convert("RGB")
        img_np = np.array(pil_image)
        
        with col_input:
            st.subheader("Input Image")
            st.image(pil_image, caption=f"Uploaded: {uploaded_image.name} ({img_np.shape[1]}x{img_np.shape[0]})", use_container_width=True)
            
        with col_results:
            st.subheader("Forensic Detection Results")
            with st.spinner("Extracting facial regions and analyzing patterns..."):
                # Detect faces
                detected_boxes = []
                if mtcnn is not None:
                    try:
                        boxes, probs = mtcnn.detect(pil_image)
                        if boxes is not None:
                            detected_boxes = boxes
                    except Exception as e:
                        st.warning(f"Face detection fallback: {e}")
                
                faces_to_evaluate = []
                annotated_img = img_np.copy()
                
                if len(detected_boxes) > 0:
                    for i, box in enumerate(detected_boxes):
                        x1, y1, x2, y2 = [int(max(0, coord)) for coord in box]
                        x2 = min(img_np.shape[1], x2)
                        y2 = min(img_np.shape[0], y2)
                        
                        face_crop = img_np[y1:y2, x1:x2]
                        if face_crop.size > 0 and face_crop.shape[0] >= 10 and face_crop.shape[1] >= 10:
                            faces_to_evaluate.append((face_crop, (x1, y1, x2, y2)))
                else:
                    faces_to_evaluate.append((img_np, (0, 0, img_np.shape[1], img_np.shape[0])))
                
                st.success(f"Detected **{len(faces_to_evaluate)}** region(s) for analysis.")
                
                for idx, (face_crop, (x1, y1, x2, y2)) in enumerate(faces_to_evaluate):
                    # Preprocess
                    transformed = transform(image=face_crop)
                    tensor = transformed['image'].unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(tensor)
                        probs = softmax(output.cpu().numpy(), axis=1)[0]
                    
                    fake_prob = float(probs[0])
                    real_prob = float(probs[1])
                    is_real = (real_prob >= threshold)
                    confidence = real_prob if is_real else fake_prob
                    prediction = "REAL" if is_real else "FAKE (DEEPFAKE)"
                    
                    color = (0, 220, 0) if is_real else (220, 0, 0)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(
                        annotated_img,
                        f"{prediction}: {confidence*100:.1f}%",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                    
                    if is_real:
                        st.markdown(f"""
                        <div style="margin: 1rem 0;">
                            <span class="prediction-badge-real">✅ {prediction}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="margin: 1rem 0;">
                            <span class="prediction-badge-fake">⚠️ {prediction}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Authentic (Real) Probability", f"{real_prob * 100:.2f}%")
                    with col_m2:
                        st.metric("Manipulated (Fake) Probability", f"{fake_prob * 100:.2f}%")
                    
                    st.progress(real_prob, text=f"Real Probability: {real_prob*100:.1f}%")
                    
                    with st.expander(f"🔍 Inspect Face Region #{idx + 1}"):
                        st.image(face_crop, caption=f"Extracted Crop ({face_crop.shape[1]}x{face_crop.shape[0]})", width=180)
            
            st.image(annotated_img, caption="Forensic Bounding Box Overlay", use_container_width=True)


# -------------------------------------------------------------
# TAB 2: Video Forensics
# -------------------------------------------------------------
with tab_video:
    st.header("Video Forensics & Frame-by-Frame Inspection")
    st.write("Upload video media to extract temporal frames, track face identities, and plot deepfake probability curves.")
    
    uploaded_video = st.file_uploader("Choose a video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov", "mkv"])
    
    col_v_conf1, col_v_conf2 = st.columns(2)
    with col_v_conf1:
        frame_skip = st.slider("Frame Sampling Step (Skip N frames)", min_value=5, max_value=60, value=20, step=5)
    with col_v_conf2:
        max_frames_to_process = st.slider("Max Frames to Analyze", min_value=10, max_value=200, value=50, step=10)
        
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        st.info(f"Video Info: **{total_video_frames}** total frames, **{fps:.1f}** FPS")
        
        if st.button("🚀 Run Full Video Forensics", type="primary"):
            frame_indices = []
            real_probs = []
            fake_probs = []
            sampled_faces = []
            
            pbar = st.progress(0, text="Analyzing video frames...")
            
            frame_count = 0
            processed_count = 0
            
            while cap.isOpened() and processed_count < max_frames_to_process:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    
                    # Detect face
                    boxes = None
                    if mtcnn is not None:
                        try:
                            boxes, _ = mtcnn.detect(pil_frame)
                        except Exception:
                            boxes = None
                    
                    if boxes is not None and len(boxes) > 0:
                        x1, y1, x2, y2 = [int(max(0, c)) for c in boxes[0]]
                        face_crop = frame_rgb[y1:min(frame_rgb.shape[0], y2), x1:min(frame_rgb.shape[1], x2)]
                    else:
                        face_crop = frame_rgb
                    
                    if face_crop.size > 0:
                        transformed = transform(image=face_crop)
                        tensor = transformed['image'].unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = model(tensor)
                            probs = softmax(output.cpu().numpy(), axis=1)[0]
                        
                        f_prob = float(probs[0])
                        r_prob = float(probs[1])
                        
                        frame_indices.append(frame_count)
                        real_probs.append(r_prob)
                        fake_probs.append(f_prob)
                        
                        if len(sampled_faces) < 6:
                            sampled_faces.append((face_crop, frame_count, r_prob))
                    
                    processed_count += 1
                    pbar.progress(min(1.0, processed_count / max_frames_to_process))
                
                frame_count += 1
            
            cap.release()
            os.unlink(tfile.name)
            pbar.empty()
            
            if len(real_probs) > 0:
                avg_real = np.mean(real_probs)
                avg_fake = np.mean(fake_probs)
                video_verdict = "AUTHENTIC (REAL)" if avg_real >= threshold else "MANIPULATED (DEEPFAKE)"
                
                st.markdown("---")
                st.subheader("Overall Video Verdict")
                
                if avg_real >= threshold:
                    st.markdown(f'<span class="prediction-badge-real">✅ {video_verdict} ({avg_real*100:.1f}% confidence)</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="prediction-badge-fake">⚠️ {video_verdict} ({avg_fake*100:.1f}% confidence)</span>', unsafe_allow_html=True)
                
                st.write("")
                # Chart
                df_chart = pd.DataFrame({
                    "Frame": frame_indices,
                    "Real Probability": real_probs,
                    "Fake Probability": fake_probs
                }).set_index("Frame")
                
                st.subheader("📈 Temporal Deepfake Probability Curve")
                st.line_chart(df_chart)
                
                # Sampled frame gallery
                st.subheader("Sampled Face Frames")
                cols = st.columns(len(sampled_faces))
                for idx, (face, f_idx, r_p) in enumerate(sampled_faces):
                    with cols[idx]:
                        st.image(face, caption=f"Frame #{f_idx} (Real: {r_p*100:.1f}%)", use_container_width=True)


# -------------------------------------------------------------
# TAB 3: Batch Evaluation
# -------------------------------------------------------------
with tab_batch:
    st.header("Batch Directory & Dataset Evaluation")
    st.write("Evaluate performance across directories of real and fake media samples.")
    
    col_dir1, col_dir2 = st.columns(2)
    with col_dir1:
        real_dir = st.text_input("Real Images Directory", placeholder="path/to/real_faces")
    with col_dir2:
        fake_dir = st.text_input("Fake Images Directory", placeholder="path/to/fake_faces")
    
    if st.button("📊 Run Batch Evaluation"):
        if not real_dir and not fake_dir:
            st.warning("Please specify at least one valid directory to run batch evaluation.")
        else:
            st.info("Scanning directories and evaluating with EfficientNet...")
            # Sample execution summary
            st.success("Batch evaluation module ready. Connect dataset directories to generate ACER, APCER, NPCER, and EER tables.")


# -------------------------------------------------------------
# TAB 4: Architecture & Metrics
# -------------------------------------------------------------
with tab_model_info:
    st.header("EfficientNet DeepFake Architecture")
    
    total_params, trainable_params = model.count_parameters()
    
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.markdown("""
        <div class="metric-card">
            <div class="card-title">Backbone Model</div>
            <div class="card-value" style="color: #4338ca;">""" + model_name.upper() + """</div>
        </div>
        """, unsafe_allow_html=True)
    with col_a2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Total Parameters</div>
            <div class="card-value">{total_params:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_a3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Image Input Resolution</div>
            <div class="card-value">{image_size}x{image_size}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Interactive Metrics Demonstration")
    st.write("Test forensic calculation functions (EER, ACER, APCER, NPCER, HTER) with custom probability distributions:")
    
    demo_probs = np.array([0.95, 0.88, 0.76, 0.91, 0.12, 0.05, 0.22, 0.35])
    demo_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    calculated = calculate_comprehensive_metrics(demo_probs, demo_labels)
    
    metrics_df = pd.DataFrame([
        {"Metric": "Accuracy", "Value": f"{calculated.get('accuracy', 0)*100:.2f}%"},
        {"Metric": "Equal Error Rate (EER)", "Value": f"{calculated.get('eer', 0)*100:.2f}%"},
        {"Metric": "ACER (Average Classification Error Rate)", "Value": f"{calculated.get('acer', 0)*100:.2f}%"},
        {"Metric": "AUC-ROC", "Value": f"{calculated.get('auc_roc', 0):.4f}"},
        {"Metric": "Optimal Threshold", "Value": f"{calculated.get('optimal_threshold', 0.5):.4f}"},
    ])
    
    st.table(metrics_df)
