import os
import sys
import json
import uuid
import subprocess
import torch
import numpy as np
import soundfile as sf

AASIST_ROOT = r"C:\Users\yaso0\.gemini\antigravity-ide\brain\fe3baf12-faf4-4c52-9aae-e0bcb76cb59c\scratch\aasist"
AASIST_WEIGHTS = r"c:\charu hack\weights\AASIST.pth"

class AASISTPredictor:
    """
    Isolated, reusable wrapper for AASIST inference.
    Handles dynamic extraction of 16kHz mono audio via FFmpeg,
    sample tiling/truncation, and isolated class loading.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.target_samples = 64600
        self.target_sr = 16000
        self._load_model()

    def _load_model(self):
        if not os.path.exists(AASIST_WEIGHTS):
            raise FileNotFoundError(f"AASIST checkpoint not found at {AASIST_WEIGHTS}")
            
        conf_path = os.path.join(AASIST_ROOT, "config", "AASIST.conf")
        if not os.path.exists(conf_path):
            raise FileNotFoundError(f"Config not found at {conf_path}")
            
        with open(conf_path, "r") as f:
            config = json.load(f)
        model_config = config["model_config"]
        
        # Import isolation
        original_path = list(sys.path)
        if AASIST_ROOT not in sys.path:
            sys.path.insert(0, AASIST_ROOT)
            
        try:
            from models.AASIST import Model
            self.model = Model(model_config).to(self.device)
            checkpoint = torch.load(AASIST_WEIGHTS, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
        finally:
            sys.path[:] = original_path

    def _extract_audio(self, input_path: str) -> str:
        """Extracts audio to a temporary 16kHz mono WAV file using FFmpeg."""
        temp_wav = f"temp_aasist_{uuid.uuid4().hex}.wav"
        
        ffmpeg_path = r"C:\Users\yaso0\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
        if not os.path.exists(ffmpeg_path):
            ffmpeg_path = "ffmpeg"
            
        # -y: overwrite, -i: input, -ar: sample rate, -ac: channels
        cmd = [
            ffmpeg_path,
            "-y",
            "-i", input_path,
            "-ar", str(self.target_sr),
            "-ac", "1",
            temp_wav
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                raise RuntimeError(f"FFmpeg failed to extract audio: {result.stderr}")
            if not os.path.exists(temp_wav):
                raise RuntimeError("FFmpeg succeeded but temporary file was not created.")
            return temp_wav
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Ensure ffmpeg is installed and in PATH.")

    def _pad_audio(self, x: np.ndarray) -> np.ndarray:
        x_len = x.shape[0]
        if x_len >= self.target_samples:
            return x[:self.target_samples]
        num_repeats = int(self.target_samples / x_len) + 1
        padded_x = np.tile(x, num_repeats)[:self.target_samples]
        return padded_x

    def predict(self, input_path: str) -> dict:
        """
        Runs AASIST inference on any audio or video file.
        Extracts 16kHz mono WAV, formats to 64600 samples, and predicts.
        
        Returns:
            {
                "audio_score": float,  # Spoof probability [0, 1]
                "sample_rate": int,
                "num_samples": int
            }
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found at {input_path}")
            
        temp_wav = None
        try:
            temp_wav = self._extract_audio(input_path)
            
            # Load audio using soundfile
            audio_data, file_sr = sf.read(temp_wav)
            
            if len(audio_data) == 0:
                raise RuntimeError("Audio file is empty or has no valid waveform.")
                
            # Fallback mono conversion just in case FFmpeg output is weird
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            audio_padded = self._pad_audio(audio_data)
            
            # Inference
            audio_tensor = torch.tensor(audio_padded, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, logits = self.model(audio_tensor, Freq_aug=False)
                
            probabilities = torch.softmax(logits, dim=1)
            audio_score = probabilities[0, 0].item()
            
            assert 0.0 <= audio_score <= 1.0, f"audio_score {audio_score} out of bounds"
            
            return {
                "audio_score": audio_score,
                "sample_rate": self.target_sr,
                "num_samples": self.target_samples
            }
            
        except Exception as e:
            raise RuntimeError(f"AASIST inference failed: {str(e)}") from e
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass  # File may be held, ignore to not crash response
