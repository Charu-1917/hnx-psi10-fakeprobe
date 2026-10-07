import os
import sys
import uuid
import shutil
import subprocess
import time
import cv2
import glob

SYNCNET_ROOT = r"C:\Users\yaso0\.gemini\antigravity-ide\brain\fe3baf12-faf4-4c52-9aae-e0bcb76cb59c\scratch\syncnet_python"
SYNCNET_WEIGHTS = r"c:\charu hack\weights\syncnet_v2.model"

class SyncNetPredictor:
    """
    Isolated, reusable wrapper for SyncNet inference.
    Handles the verified preprocessing pipeline (S3FD face detection + cropping)
    and offset evaluation without reloading the model.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._load_model()

    def _load_model(self):
        # Ensure FFmpeg is accessible for SyncNetInstance's internal subprocess calls
        ffmpeg_dir = r"C:\Users\yaso0\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin"
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = os.environ.get("PATH", "") + ";" + ffmpeg_dir
            
        if not os.path.exists(SYNCNET_WEIGHTS):
            raise FileNotFoundError(f"SyncNet checkpoint not found at {SYNCNET_WEIGHTS}")
            
        # Import isolation
        original_path = list(sys.path)
        if SYNCNET_ROOT not in sys.path:
            sys.path.insert(0, SYNCNET_ROOT)
            
        try:
            from SyncNetInstance import SyncNetInstance
            self.model = SyncNetInstance()
            self.model.loadParameters(SYNCNET_WEIGHTS)
        finally:
            sys.path[:] = original_path

    def predict(self, video_path: str) -> dict:
        """
        Runs the verified SyncNet pipeline on the provided video.
        Uses a temporary directory for intermediate files, cleaned up safely afterward.
        
        Returns:
            {
                "offset_frames": int,
                "confidence": float,
                "fps": float,
                "offset_ms": float
            }
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
            
        # 1. Obtain original FPS to calculate milliseconds accurately
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps <= 0:
            fps = 25.0  # Safe fallback for corrupted headers
            
        reference = f"syncnet_run_{uuid.uuid4().hex}"
        # Store temporary data in a dedicated workspace folder
        temp_dir = os.path.join(os.getcwd(), "temp_syncnet_work", reference)
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # 2. Verified Preprocessing Pipeline via subprocess
            # SyncNet run_pipeline handles audio extraction, S3FD face detection, and cropping.
            cmd = [
                sys.executable,
                os.path.join(SYNCNET_ROOT, "run_pipeline.py"),
                "--videofile", video_path,
                "--reference", reference,
                "--data_dir", temp_dir
            ]
            
            # Ensure FFmpeg is accessible
            env = os.environ.copy()
            ffmpeg_dir = r"C:\Users\yaso0\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin"
            if ffmpeg_dir not in env.get("PATH", ""):
                env["PATH"] = env.get("PATH", "") + ";" + ffmpeg_dir
                
            res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"SyncNet preprocessing failed:\n{res.stderr}")
                
            # Locate the generated cropped face track
            crop_dir = os.path.join(temp_dir, "pycrop", reference)
            cropped_files = glob.glob(os.path.join(crop_dir, "*.avi"))
            
            if not cropped_files:
                raise RuntimeError("SyncNet preprocessing failed: No face tracks detected in the video.")
                
            eval_file = cropped_files[0]
            
            # 3. Inference
            class Opts:
                pass
            opts = Opts()
            opts.tmp_dir = os.path.join(temp_dir, "pytmp")
            opts.reference = reference
            opts.batch_size = 20
            opts.vshift = 15
            
            # Re-insert path for evaluate just in case internal dynamic imports are needed
            original_path = list(sys.path)
            if SYNCNET_ROOT not in sys.path:
                sys.path.insert(0, SYNCNET_ROOT)
                
            try:
                offset, conf, dists_npy = self.model.evaluate(opts, videofile=eval_file)
            finally:
                sys.path[:] = original_path
                
            # 4. Conversion
            # Negative offset means audio leads video, positive means audio lags video.
            offset_ms = (offset / fps) * 1000.0
            
            return {
                "offset_frames": int(offset),
                "confidence": float(conf),
                "fps": float(fps),
                "offset_ms": float(offset_ms)
            }
            
        finally:
            # 5. Safe Cleanup
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except OSError:
                    pass  # Avoid crashing if a file lock is held
