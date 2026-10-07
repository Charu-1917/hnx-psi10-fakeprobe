import os
import sys
import time
import torch
import torch.nn as nn
from PIL import Image
import clip
import cv2
import numpy as np

def extract_frames_at_times(video_path, target_times_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    frames_data = []
    
    for t_sec in target_times_sec:
        # If the requested time is beyond the video duration, clamp to the last frame
        if t_sec > duration:
            if frames_data:
                continue # Skip if we already got the last available frames
            t_sec = max(0, duration - 1.0/fps)
            
        frame_idx = int(t_sec * fps)
        # Ensure we don't go beyond total frames
        frame_idx = min(frame_idx, total_frames - 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            actual_time = frame_idx / fps if fps > 0 else t_sec
            frames_data.append({"image": pil_image, "timestamp": actual_time, "source": f"video (target {t_sec}s)"})
        else:
            print(f"Failed to read frame at {t_sec}s")
            
    cap.release()
    return frames_data

def get_stats(scores):
    if not scores:
        return {}
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores))
    }

def main():
    print("UNIVFD MULTI-SAMPLE VALIDATION")
    
    weights_path = r"C:\charu hack\weights\fc_weights.pth"
    clip_model_name = "ViT-L/14"
    clip_download_root = r"C:\charu hack\weights"
    
    video_path = r"C:\charu hack\tests\test_video.mp4"
    image_path = r"C:\charu hack\tests\real_test_image.jpg"
    
    start_total = time.time()
    
    print("\n--- LOADING CLIP ---")
    t0 = time.time()
    try:
        model, preprocess = clip.load(clip_model_name, device="cpu", download_root=clip_download_root)
        t_clip = time.time() - t0
        print(f"CLIP load time: {t_clip:.2f}s")
    except Exception as e:
        print(f"CLIP load failed: {e}")
        sys.exit(1)
        
    print("\n--- LOADING CLASSIFIER ---")
    state_dict = torch.load(weights_path, map_location='cpu')
    classifier = nn.Linear(768, 1)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    
    # 1. Prepare Gemini frames
    print(f"\nExtracting frames from {video_path}")
    gemini_frames = extract_frames_at_times(video_path, [0, 2, 4, 6, 8])
    
    # 2. Prepare Real inputs
    print(f"Loading {image_path}")
    pil_real = Image.open(image_path).convert("RGB")
    real_frames = [{"image": pil_real, "timestamp": "N/A", "source": "image"}]
    
    # We don't have a real video in tests/ to sample from based on list_dir output
    
    # Function to run inference
    def run_inference(samples):
        results = []
        inference_times = []
        
        for idx, sample in enumerate(samples):
            t_inf0 = time.time()
            with torch.no_grad():
                img_input = preprocess(sample["image"]).unsqueeze(0).to("cpu")
                features = model.encode_image(img_input)
                features = features / features.norm(dim=-1, keepdim=True)
                logit = classifier(features.float()).squeeze().item()
                score = torch.sigmoid(torch.tensor(logit)).item()
            t_inf = time.time() - t_inf0
            inference_times.append(t_inf)
            
            results.append({
                "source": sample["source"],
                "timestamp": sample["timestamp"],
                "raw_logit": logit,
                "synthetic_score": score,
                "inference_time": t_inf
            })
        return results, inference_times

    print("\nRunning Inference on Gemini...")
    gemini_results, gemini_times = run_inference(gemini_frames)
    
    print("\nRunning Inference on Real...")
    real_results, real_times = run_inference(real_frames)
    
    # Process results
    gemini_scores = [r["synthetic_score"] for r in gemini_results]
    real_scores = [r["synthetic_score"] for r in real_results]
    
    gemini_stats = get_stats(gemini_scores)
    real_stats = get_stats(real_scores)
    
    avg_inf_time = np.mean(gemini_times + real_times)
    total_time = time.time() - start_total
    
    print("\nOUTPUT FORMATTING\n" + "="*50)
    print("UNIVFD MULTI-SAMPLE VALIDATION\n")
    
    print("Gemini T2V:")
    print(f"{'Source':<20} | {'Timestamp':<10} | {'Raw Logit':<12} | {'Synthetic Score':<15} | {'Inf Time'}")
    print("-" * 75)
    for r in gemini_results:
        ts = f"{r['timestamp']:.2f}s" if isinstance(r['timestamp'], float) else str(r['timestamp'])
        print(f"{r['source']:<20} | {ts:<10} | {r['raw_logit']:<12.4f} | {r['synthetic_score']:<15.4f} | {r['inference_time']:.2f}s")
        
    print("\nReal media:")
    print(f"{'Source':<20} | {'Timestamp':<10} | {'Raw Logit':<12} | {'Synthetic Score':<15} | {'Inf Time'}")
    print("-" * 75)
    for r in real_results:
        print(f"{r['source']:<20} | {r['timestamp']:<10} | {r['raw_logit']:<12.4f} | {r['synthetic_score']:<15.4f} | {r['inference_time']:.2f}s")
        
    print("\nStatistics:")
    print("GEMINI T2V:")
    if gemini_stats:
        print(f"- Minimum: {gemini_stats['min']:.4f}")
        print(f"- Maximum: {gemini_stats['max']:.4f}")
        print(f"- Mean: {gemini_stats['mean']:.4f}")
        print(f"- Median: {gemini_stats['median']:.4f}")
        print(f"- Standard Deviation: {gemini_stats['std']:.4f}")
    
    print("\nREAL:")
    if real_stats:
        print(f"- Minimum: {real_stats['min']:.4f}")
        print(f"- Maximum: {real_stats['max']:.4f}")
        print(f"- Mean: {real_stats['mean']:.4f}")
        print(f"- Median: {real_stats['median']:.4f}")
        print(f"- Standard Deviation: {real_stats['std']:.4f}")
        
    print("\nSeparation:")
    mean_diff = gemini_stats['mean'] - real_stats['mean']
    median_diff = gemini_stats['median'] - real_stats['median']
    print(f"- Mean Score Difference (Gemini - Real): {mean_diff:.4f}")
    print(f"- Median Score Difference (Gemini - Real): {median_diff:.4f}")
    
    print("\nPerformance:")
    print(f"- CLIP load time: {t_clip:.2f}s")
    print(f"- Average inference time/frame: {avg_inf_time:.2f}s")
    print(f"- Total experiment time: {total_time:.2f}s")
    print("- OOM/crash status: NO CRASH")

if __name__ == '__main__':
    main()
