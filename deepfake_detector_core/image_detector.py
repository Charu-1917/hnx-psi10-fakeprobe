"""
CoAtNet-0 5-Channel Image Deepfake Detector with Input Quality and Forensic Face Cropping.
Accepts RGB images, computes Error Level Analysis (ELA) and Fast Fourier Transform (FFT) spectra,
and produces standardized forensic detector predictions.
"""

import io
import cv2
import torch
import torch.nn as nn
import numpy as np
import timm
from pathlib import Path
from PIL import Image
from typing import Optional, Union

from .path_utils import get_image_model_path
from .types import DetectorResult, DecisionVerdict, QualityResult, QualityLevel
from .quality_analyzer import InputQualityAnalyzer
from .frequency_analyzer import FrequencyForensicAnalyzer
from .face_utils import FaceDetector
from configs.forensic_config import DEFAULT_CONFIG


def compute_ela(image_rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    """Compute Error Level Analysis (ELA) map."""
    pil_img = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = np.array(Image.open(buffer).convert("RGB")).astype(np.float32)
    original = image_rgb.astype(np.float32)
    ela_map = np.abs(original - compressed).mean(axis=2)
    ela_max = ela_map.max()
    if ela_max > 0:
        ela_map /= ela_max
    return ela_map.astype(np.float32)


def compute_fft_magnitude(image_rgb: np.ndarray) -> np.ndarray:
    """Compute Fast Fourier Transform (FFT) 2D log-magnitude spectrum."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude /= mag_max
    return magnitude.astype(np.float32)


def prepare_5ch_tensor(image_rgb: np.ndarray, size: int = 224, config=DEFAULT_CONFIG) -> torch.Tensor:
    """Convert an RGB image (numpy) to a normalized 5-channel tensor for CoAtNet."""
    if image_rgb.shape[0] != size or image_rgb.shape[1] != size:
        image_rgb = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)

    ela = compute_ela(image_rgb, quality=config.ela_quality)
    fft = compute_fft_magnitude(image_rgb)
    ela_u8 = (ela * 255).astype(np.uint8)
    fft_u8 = (fft * 255).astype(np.uint8)

    ch5 = np.concatenate([
        image_rgb,
        ela_u8[..., np.newaxis],
        fft_u8[..., np.newaxis]
    ], axis=2)

    tensor = torch.from_numpy(ch5).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([*config.rgb_mean, *config.forensic_mean]).view(5, 1, 1)
    std = torch.tensor([*config.rgb_std, *config.forensic_std]).view(5, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def build_coatnet_5ch(num_classes: int = 1):
    """Build 5-channel CoAtNet-0 model backbone."""
    model = timm.create_model("coatnet_0_rw_224", pretrained=False, num_classes=num_classes)
    original_conv = model.stem.conv1
    new_conv = nn.Conv2d(
        in_channels=5,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )
    model.stem.conv1 = new_conv
    return model


class ImageDeepfakeDetector:
    """CoAtNet-0 5-channel spatial-forensic deepfake detector."""

    def __init__(self, weights_path: str | None = None, config=DEFAULT_CONFIG):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        if weights_path is None:
            weights_path = get_image_model_path()

        if weights_path is None or not torch.os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Image model weights file not found! Expected 'best_coatnet_5ch.pth' in models directory."
            )

        self.weights_path = weights_path
        self.model = build_coatnet_5ch()

        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Modular components
        self.quality_analyzer = InputQualityAnalyzer(config=self.config)
        self.frequency_analyzer = FrequencyForensicAnalyzer()
        self.face_detector = FaceDetector(device=str(self.device))

        # Feature extraction hook for video multimodal pipeline (768-dim embedding)
        self._hook_features = {}
        self._hook_handle = self.model.head.global_pool.register_forward_hook(
            self._capture_hook
        )

    def _capture_hook(self, module, input, output):
        self._hook_features["embedding"] = output.detach()

    def predict_structured(
        self,
        image_bytes_or_array: Union[bytes, bytearray, np.ndarray],
        apply_face_crop: bool = True
    ) -> DetectorResult:
        """
        Run deepfake inference on input image with face localization and quality awareness.
        Returns standardized DetectorResult.
        """
        # Decode input
        if isinstance(image_bytes_or_array, (str, Path)):
            image_bgr = cv2.imread(str(image_bytes_or_array))
            if image_bgr is None:
                raise ValueError(f"Could not load image from filepath: {image_bytes_or_array}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_bytes_or_array, (bytes, bytearray)):
            nparr = np.frombuffer(image_bytes_or_array, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("Could not decode image from provided bytes.")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_bytes_or_array

        # 1. Face Detection & Cropping
        face_boxes, face_probs = self.face_detector.detect_faces(image_rgb)
        if apply_face_crop and len(face_boxes) > 0:
            proc_image, crop_box, face_conf = self.face_detector.crop_primary_face(
                image_rgb, target_size=self.config.image_input_size, margin=self.config.face_crop_margin
            )
            face_detected = True
        else:
            # Aspect-ratio preserving square center-crop
            h, w = image_rgb.shape[:2]
            side = min(h, w)
            sy = (h - side) // 2
            sx = (w - side) // 2
            proc_image = cv2.resize(
                image_rgb[sy:sy+side, sx:sx+side],
                (self.config.image_input_size, self.config.image_input_size),
                interpolation=cv2.INTER_AREA
            )
            crop_box = None
            face_conf = 0.0
            face_detected = False

        # 2. Input Quality & Frequency Analysis
        quality_res = self.quality_analyzer.analyze_image(image_rgb, face_boxes=face_boxes)
        freq_res = self.frequency_analyzer.analyze_frequency_domain(proc_image)
        ela_map = compute_ela(proc_image, quality=self.config.ela_quality)

        # 3. Model Forward Pass
        image_tensor = prepare_5ch_tensor(proc_image, size=self.config.image_input_size, config=self.config)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            raw_logit = float(outputs.squeeze().item())
            fake_prob = float(torch.sigmoid(outputs).item())

        real_prob = float(1.0 - fake_prob)

        # 4. Reliability & Confidence Calculation
        # Distance from 0.5 decision boundary
        boundary_dist = abs(fake_prob - 0.50) * 2.0
        confidence = float(np.clip(boundary_dist, 0.0, 1.0))

        # Reliability incorporates input quality and whether face was located
        face_coverage = 1.0 if face_detected else 0.65
        reliability = float(np.clip(0.60 * quality_res.quality_score + 0.25 * confidence + 0.15 * face_coverage, 0.0, 1.0))

        # 5. Three-Way Decision Logic
        if quality_res.quality_level == QualityLevel.LOW and not quality_res.is_acceptable:
            prediction = DecisionVerdict.UNCERTAIN
            detail_label = "Uncertain (Severe Image Degradation)"
        elif self.config.uncertainty_real_boundary <= fake_prob <= self.config.uncertainty_fake_boundary:
            prediction = DecisionVerdict.UNCERTAIN
            detail_label = "Uncertain (Borderline Artifacts)"
        elif fake_prob > self.config.uncertainty_fake_boundary:
            prediction = DecisionVerdict.FAKE
            detail_label = "Fake (Generative Face Synthesis Detected)"
        else:
            prediction = DecisionVerdict.REAL
            detail_label = "Real (Authentic Photographic Image)"

        evidence = {
            "raw_logit": raw_logit,
            "ela_map": ela_map,
            "fft_map": freq_res["norm_magnitude_map"],
            "frequency_anomaly_score": freq_res["frequency_anomaly_score"],
            "spectral_slope": freq_res["spectral_slope"],
            "high_frequency_ratio": freq_res["high_frequency_ratio"],
            "face_detected": face_detected,
            "face_box": crop_box,
            "face_boxes_all": face_boxes,
            "face_confidence": face_conf,
            "processed_crop_rgb": proc_image,
        }

        return DetectorResult(
            model="CoAtNet-0 5-Channel (RGB+ELA+FFT)",
            modality="image",
            prediction=prediction,
            raw_score=raw_logit,
            probability_fake=fake_prob,
            probability_real=real_prob,
            confidence=confidence,
            reliability=reliability,
            quality=quality_res,
            evidence=evidence,
            detail_label=detail_label,
        )

    def predict(self, image_bytes_or_array: bytes | np.ndarray) -> dict:
        """
        Legacy dictionary interface for backwards compatibility.
        """
        res = self.predict_structured(image_bytes_or_array, apply_face_crop=self.config.use_face_crop)
        return {
            "prediction": res.prediction.value,
            "detail_label": res.detail_label,
            "is_fake": (res.prediction == DecisionVerdict.FAKE),
            "confidence": res.confidence,
            "reliability": res.reliability,
            "fake_prob": res.probability_fake,
            "real_prob": res.probability_real,
            "raw_score": res.raw_score,
            "ela_map": res.evidence.get("ela_map"),
            "fft_map": res.evidence.get("fft_map"),
            "quality": res.quality.to_dict(),
            "frequency_evidence": {
                "anomaly_score": res.evidence.get("frequency_anomaly_score"),
                "spectral_slope": res.evidence.get("spectral_slope"),
                "high_frequency_ratio": res.evidence.get("high_frequency_ratio"),
            },
            "face_detected": res.evidence.get("face_detected"),
            "face_box": res.evidence.get("face_box"),
        }

    def extract_features(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract 768-dim CoAtNet embedding for a single RGB frame (used by Video detector)."""
        tensor = prepare_5ch_tensor(image_rgb, size=self.config.image_input_size, config=self.config)
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(tensor)
        embedding = self._hook_features["embedding"].squeeze().cpu().numpy()
        self._hook_features.clear()
        return embedding
