import sys
import os
import shutil
import subprocess
import time
import torch
import cv2
import pickle

# Set environment PATH so subprocesses can find ffmpeg if installed via winget
os.environ["PATH"] = os.environ.get("PATH", "") + ";" + os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin")

# Path to SyncNet repository in scratch
SYNCNET_ROOT = r"C:\Users\yaso0\.gemini\antigravity-ide\brain\fe3baf12-faf4-4c52-9aae-e0bcb76cb59c\scratch\syncnet_python"
if SYNCNET_ROOT not in sys.path:
    sys.path.insert(0, SYNCNET_ROOT)

from SyncNetModel import S
from SyncNetInstance import SyncNetInstance

def check_ffmpeg():
    try:
        res = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0, res.stdout.split("\n")[0] if res.returncode == 0 else ""
    except FileNotFoundError:
        return False, "ffmpeg not found in PATH"

def get_video_info(videofile):
    cap = cv2.VideoCapture(videofile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return duration, fps, width, height

def main():
    print("=== SYNCNET SMOKE TEST ===")
    
    # 1. Verify ffmpeg
    ffmpeg_ok, ffmpeg_info = check_ffmpeg()
    print(f"FFmpeg check: {'PASS' if ffmpeg_ok else 'FAIL'} ({ffmpeg_info})")
    
    if not ffmpeg_ok:
        print("\n[STOPPING SYNCNET VIDEO PIPELINE TESTING]")
        print("Reason: ffmpeg is missing from system PATH.")
        print("SYNCNET BLOCKED — FFmpeg required")
        print("=== SYNCNET SMOKE TEST HALTED ===")
        return

    # 2. Test Video
    test_video = r"c:\charu hack\tests\test_video.mp4"
    if not os.path.exists(test_video):
        print("\n[STOPPING SYNCNET VIDEO PIPELINE TESTING]")
        print("SYNCNET BLOCKED — TEST VIDEO REQUIRED")
        print("=== SYNCNET SMOKE TEST HALTED ===")
        return

    duration, fps, w, h = get_video_info(test_video)
    print(f"Video Info: duration={duration:.2f}s, fps={fps:.2f}, resolution={w}x{h}")
    
    # 3. Model Loading
    weights_path = r"c:\charu hack\weights\syncnet_v2.model"
    device = "cpu"
    print(f"Target device: {device}")
    
    if not os.path.exists(weights_path):
        print(f"Checkpoint check: FAIL ({weights_path} not found)")
        return
    else:
        print(f"Checkpoint check: PASS ({weights_path})")
    
    s = SyncNetInstance()
    s.loadParameters(weights_path)
    print("model loaded: TRUE (SyncNetInstance loaded checkpoint on CPU)")
    
    # 4. Run preprocessing pipeline
    t0 = time.time()
    data_dir = r"c:\charu hack\tests\syncnet_work"
    cmd = [
        sys.executable,
        os.path.join(SYNCNET_ROOT, "run_pipeline.py"),
        "--videofile", test_video,
        "--reference", "smoke_test",
        "--data_dir", data_dir
    ]
    print(f"Running preprocessing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Preprocessing FAILED:")
        print(res.stderr)
        return
    print("Preprocessing: PASS")
    
    # Check if a cropped face video was created
    crop_dir = os.path.join(data_dir, "pycrop", "smoke_test")
    import glob
    cropped_files = glob.glob(os.path.join(crop_dir, "*.avi"))
    if not cropped_files:
        print("Preprocessing FAILED: No cropped face video generated.")
        return
    
    # Evaluate the first track
    eval_file = cropped_files[0]
    print(f"Evaluating cropped track: {eval_file}")
    
    # The 'evaluate' method expects an 'opt' object with 'tmp_dir', 'reference', 'batch_size', 'vshift'
    class Opts:
        tmp_dir = os.path.join(data_dir, "pytmp")
        reference = "smoke_test"
        batch_size = 20
        vshift = 15
    
    # 5. Full inference
    try:
        offset, conf, dists_npy = s.evaluate(Opts(), videofile=eval_file)
        min_dist = dists_npy.min()
        runtime = time.time() - t0
        
        print(f"Full inference: PASS")
        print(f"Offset: {offset}")
        print(f"Confidence: {conf:.3f}")
        print(f"Minimum distance: {min_dist:.3f}")
        print(f"Runtime: {runtime:.3f}s")
        print("=== SYNCNET SMOKE TEST PASSED ===")
    except Exception as e:
        print(f"Full inference: FAIL ({e})")
        
if __name__ == "__main__":
    main()
