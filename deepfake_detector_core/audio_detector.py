"""
Whisper-Base + XGBoost Audio Deepfake Detector for FakeProbe-X.
Extracts 512-dim Whisper acoustic embeddings and classifies speech spoofing with quality analysis.
"""

import os
import io
import tempfile
import torch
import librosa
import numpy as np
import joblib
from transformers import WhisperProcessor, WhisperModel
from typing import Union

from .path_utils import get_audio_model_path
from .types import DetectorResult, DecisionVerdict, QualityResult, QualityLevel
from .quality_analyzer import InputQualityAnalyzer
from configs.forensic_config import DEFAULT_CONFIG


class AudioDeepfakeDetector:
    """Whisper-Base + XGBoost acoustic deepfake / voice clone detector."""

    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        xgb_path: str | None = None,
        config=DEFAULT_CONFIG
    ):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu"

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
        self.quality_analyzer = InputQualityAnalyzer(config=self.config)

    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract a 512-dim Whisper encoder embedding from an audio file."""
        audio_array, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate)

        inputs = self.processor(audio_array, sampling_rate=self.config.audio_sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            encoder_outputs = self.whisper.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state
            audio_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

        return audio_embedding

    def predict_structured(
        self,
        audio_input: Union[str, bytes, bytearray, np.ndarray],
        suffix: str = ".wav"
    ) -> DetectorResult:
        """
        Run deepfake / audio spoofing inference on audio with quality evaluation.
        Returns standardized DetectorResult.
        """
        temp_file = None
        target_path = None
        audio_array = None

        try:
            if isinstance(audio_input, (bytes, bytearray)):
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_input)
                    temp_file = tmp.name
                target_path = temp_file
                audio_array, sr = librosa.load(target_path, sr=self.config.audio_sample_rate)
            elif isinstance(audio_input, np.ndarray):
                audio_array = audio_input
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    temp_file = tmp.name
                import scipy.io.wavfile as wavfile
                wavfile.write(temp_file, self.config.audio_sample_rate, (audio_array * 32767).astype(np.int16))
                target_path = temp_file
            else:
                target_path = audio_input
                audio_array, sr = librosa.load(target_path, sr=self.config.audio_sample_rate)

            # 1. Quality Analysis
            quality_res = self.quality_analyzer.analyze_audio(audio_array, sample_rate=self.config.audio_sample_rate)

            # 2. Extract Whisper Embeddings
            inputs = self.processor(audio_array, sampling_rate=self.config.audio_sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)

            with torch.no_grad():
                encoder_outputs = self.whisper.encoder(input_features)
                hidden_states = encoder_outputs.last_hidden_state
                audio_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

            embedding_2d = np.expand_dims(audio_embedding, axis=0)

            # 3. XGBoost Classification
            probabilities = self.classifier.predict_proba(embedding_2d)[0]
            spoof_prob = float(probabilities[1])
            real_prob = float(probabilities[0])

            # Raw margin from XGBoost decision boundary
            raw_margin = float(spoof_prob - 0.50)

            # 4. Reliability & Confidence
            boundary_dist = abs(spoof_prob - 0.50) * 2.0
            confidence = float(np.clip(boundary_dist, 0.0, 1.0))
            reliability = float(np.clip(0.65 * quality_res.quality_score + 0.35 * confidence, 0.0, 1.0))

            # 5. Three-Way Decision Logic
            if quality_res.quality_level == QualityLevel.LOW and not quality_res.is_acceptable:
                prediction = DecisionVerdict.UNCERTAIN
                detail_label = "Uncertain (Poor Audio Quality / High Noise)"
            elif self.config.uncertainty_real_boundary <= spoof_prob <= self.config.uncertainty_fake_boundary:
                prediction = DecisionVerdict.UNCERTAIN
                detail_label = "Uncertain (Ambiguous Voice Signatures)"
            elif spoof_prob > self.config.uncertainty_fake_boundary:
                prediction = DecisionVerdict.FAKE
                detail_label = "Fake (Synthetic Voice / Audio Spoof Detected)"
            else:
                prediction = DecisionVerdict.REAL
                detail_label = "Real (Authentic Human Voice)"

            evidence = {
                "whisper_embedding_norm": float(np.linalg.norm(audio_embedding)),
                "audio_duration_sec": quality_res.metrics.get("duration_sec", 0.0),
                "silence_ratio": quality_res.metrics.get("silence_ratio", 0.0),
                "clipping_ratio": quality_res.metrics.get("clipping_ratio", 0.0),
            }

            return DetectorResult(
                model="Whisper-Base + XGBoost (ASVspoof)",
                modality="audio",
                prediction=prediction,
                raw_score=raw_margin,
                probability_fake=spoof_prob,
                probability_real=real_prob,
                confidence=confidence,
                reliability=reliability,
                quality=quality_res,
                evidence=evidence,
                detail_label=detail_label,
            )

        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    def predict(self, audio_input: str | bytes, suffix: str = ".wav") -> dict:
        """
        Legacy dictionary interface for backwards compatibility.
        """
        res = self.predict_structured(audio_input, suffix=suffix)
        return {
            "prediction": res.prediction.value,
            "detail_label": res.detail_label,
            "is_fake": (res.prediction == DecisionVerdict.FAKE),
            "confidence": res.confidence,
            "reliability": res.reliability,
            "fake_prob": res.probability_fake,
            "real_prob": res.probability_real,
            "quality": res.quality.to_dict(),
        }
