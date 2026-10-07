import sys
import os
import json
import time
import numpy as np
import soundfile as sf
import torch

# Path to AASIST repository in scratch
AASIST_ROOT = r"C:\Users\yaso0\.gemini\antigravity-ide\brain\fe3baf12-faf4-4c52-9aae-e0bcb76cb59c\scratch\aasist"
if AASIST_ROOT not in sys.path:
    sys.path.insert(0, AASIST_ROOT)

from models.AASIST import Model

def pad_audio(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

def main():
    print("=== AASIST SMOKE TEST ===")
    device = torch.device("cpu")
    print(f"Target device: {device}")
    
    # 1. Load model config
    conf_path = os.path.join(AASIST_ROOT, "config", "AASIST.conf")
    with open(conf_path, "r") as f:
        config = json.load(f)
    model_config = config["model_config"]
    
    # 2. Instantiate AASIST and load weights
    weights_path = r"c:\charu hack\weights\AASIST.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")
        
    model = Model(model_config).to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("model loaded: TRUE")
    print(f"device: {device}")
    
    # 3. Create a temporary 16 kHz WAV test signal
    sr = 16000
    duration = 4.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz test sine tone (standard speech-band frequency)
    audio_signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    temp_wav_path = r"c:\charu hack\tests\temp_test_audio.wav"
    sf.write(temp_wav_path, audio_signal, sr)
    
    try:
        # 4. Load audio and verify format (mono, 16 kHz)
        audio_data, file_sr = sf.read(temp_wav_path)
        
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        print(f"input sample rate: {file_sr}")
        
        # Format input to standard AASIST length (64600 samples)
        audio_padded = pad_audio(audio_data, max_len=model_config.get("nb_samp", 64600))
        input_samples = len(audio_padded)
        print(f"input samples: {input_samples}")
        
        audio_tensor = torch.tensor(audio_padded, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 5. Run inference & measure runtime
        start_time = time.time()
        with torch.no_grad():
            _, logits = model(audio_tensor, Freq_aug=False)
        runtime = time.time() - start_time
        
        # 6. Apply softmax
        probabilities = torch.softmax(logits, dim=1)
        spoof_prob = probabilities[0, 0].item()
        bonafide_prob = probabilities[0, 1].item()
        
        # 7. Print results
        print(f"logits shape: {list(logits.shape)}")
        print(f"spoof probability: {spoof_prob:.6f}")
        print(f"bonafide probability: {bonafide_prob:.6f}")
        print(f"runtime: {runtime:.3f}s")
        print("=== AASIST SMOKE TEST PASSED ===")
        
    finally:
        # Clean up temporary WAV
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

if __name__ == "__main__":
    main()
