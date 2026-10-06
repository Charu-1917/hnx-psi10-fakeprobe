import os
import io
import tempfile
import torch
import librosa
import numpy as np
import joblib
from transformers import WhisperProcessor, WhisperModel
from .path_utils import get_audio_model_path


class AudioDeepfakeDetector:
    def __init__(self, model_name: str = "openai/whisper-base", xgb_path: str | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if xgb_path is None:
            xgb_path = get_audio_model_path()
            
        if xgb_path is None or not os.path.exists(xgb_path):
            raise FileNotFoundError(
                f"Audio model weights file not found! Expected 'xgboost_asvspoof_model.joblib' in models directory."
            )
            
        self.xgb_path = xgb_path
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.whisper = WhisperModel.from_pretrained(model_name).to(self.device)
        self.whisper.eval()
        
        self.classifier = joblib.load(self.xgb_path)

    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract a 512-dim Whisper encoder embedding from an audio file."""
        audio_array, sr = librosa.load(audio_path, sr=16000)
        
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        
        with torch.no_grad():
            encoder_outputs = self.whisper.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state
            audio_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            
        return audio_embedding

    def predict(self, audio_input: str | bytes, suffix: str = ".wav") -> dict:
        """Run deepfake/audio spoofing inference on an audio file or bytes."""
        temp_file = None
        if isinstance(audio_input, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_input)
                temp_file = tmp.name
            target_path = temp_file
        else:
            target_path = audio_input

        try:
            audio_embedding = self.extract_features(target_path)
            embedding_2d = np.expand_dims(audio_embedding, axis=0)
            
            probabilities = self.classifier.predict_proba(embedding_2d)[0]
            spoof_prob = float(probabilities[1])
            real_prob = float(probabilities[0])
            
            is_fake = spoof_prob > 0.5
            confidence = spoof_prob if is_fake else real_prob
            label = "FAKE" if is_fake else "REAL"
            detail_label = "Spoof (AI Generated Voice)" if is_fake else "Bonafide (Real Human Voice)"

            return {
                "prediction": label,
                "detail_label": detail_label,
                "is_fake": is_fake,
                "confidence": confidence,
                "fake_prob": spoof_prob,
                "real_prob": real_prob,
            }
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
