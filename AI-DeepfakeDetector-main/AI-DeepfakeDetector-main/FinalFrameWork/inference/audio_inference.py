import torch
import librosa
import numpy as np
import joblib
from transformers import WhisperProcessor, WhisperModel

class AudioDeepfakeDetector:
    def __init__(self, model_name="openai/whisper-base", xgb_path="models/xgboost_asvspoof_model.joblib"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.whisper = WhisperModel.from_pretrained(model_name).to(self.device)
        self.whisper.eval()
        
        self.classifier = joblib.load(xgb_path)

    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract a 512-dim Whisper embedding from an audio file.

        This method is used by both the standalone Audio module (via predict())
        and the Video module, ensuring a single Whisper model instance serves
        both pipelines.

        Args:
            audio_path: Path to a WAV/FLAC/MP3 file on disk.

        Returns:
            A 1-D numpy array of shape (512,) — the mean-pooled encoder embedding.
        """
        audio_array, sr = librosa.load(audio_path, sr=16000)
        
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        
        with torch.no_grad():
            encoder_outputs = self.whisper.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state
            audio_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            
        return audio_embedding

    def predict(self, audio_path):
        audio_embedding = self.extract_features(audio_path)
        embedding_2d = np.expand_dims(audio_embedding, axis=0)
        
        probabilities = self.classifier.predict_proba(embedding_2d)[0]
        spoof_prob = float(probabilities[1])
        
        if spoof_prob > 0.5:
            return "Spoof (AI Generated)", spoof_prob
        else:
            return "Bonafide (Real Human)", 1 - spoof_prob