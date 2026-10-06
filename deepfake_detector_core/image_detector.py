import io
import cv2
import torch
import torch.nn as nn
import numpy as np
import timm
from PIL import Image
from .path_utils import get_image_model_path


def compute_ela(image_rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    """Compute Error Level Analysis (ELA) map."""
    pil_img = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = np.array(Image.open(buffer).convert('RGB')).astype(np.float32)
    original = image_rgb.astype(np.float32)
    ela_map = np.abs(original - compressed).mean(axis=2)
    ela_max = ela_map.max()
    if ela_max > 0:
        ela_map /= ela_max
    return ela_map.astype(np.float32)


def compute_fft_magnitude(image_rgb: np.ndarray) -> np.ndarray:
    """Compute Fast Fourier Transform (FFT) 2D magnitude spectrum."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude /= mag_max
    return magnitude.astype(np.float32)


def prepare_5ch_tensor(image_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """Convert an RGB image (numpy) to a normalized 5-channel tensor for CoAtNet."""
    image_rgb = cv2.resize(image_rgb, (size, size))
    ela = compute_ela(image_rgb)
    fft = compute_fft_magnitude(image_rgb)
    ela_u8 = (ela * 255).astype(np.uint8)
    fft_u8 = (fft * 255).astype(np.uint8)
    ch5 = np.concatenate([
        image_rgb,
        ela_u8[..., np.newaxis],
        fft_u8[..., np.newaxis]
    ], axis=2)
    tensor = torch.from_numpy(ch5).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5]).view(5, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5]).view(5, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def build_coatnet_5ch(num_classes: int = 1):
    """Build 5-channel CoAtNet-0 model backbone."""
    model = timm.create_model('coatnet_0_rw_224', pretrained=False, num_classes=num_classes)
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
    def __init__(self, weights_path: str | None = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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

        # Feature extraction hook for video multimodal pipeline (768-dim embedding)
        self._hook_features = {}
        self._hook_handle = self.model.head.global_pool.register_forward_hook(
            self._capture_hook
        )

    def _capture_hook(self, module, input, output):
        self._hook_features['embedding'] = output.detach()

    def predict(self, image_bytes_or_array: bytes | np.ndarray) -> dict:
        """Run deepfake inference on input image bytes or RGB numpy array."""
        if isinstance(image_bytes_or_array, (bytes, bytearray)):
            nparr = np.frombuffer(image_bytes_or_array, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("Could not decode image from provided bytes.")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_bytes_or_array

        # Compute forensic maps for UI explanation
        ela_map = compute_ela(image_rgb)
        fft_map = compute_fft_magnitude(image_rgb)

        image_tensor = prepare_5ch_tensor(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            fake_prob = float(torch.sigmoid(outputs).item())

        is_fake = fake_prob > 0.5
        confidence = fake_prob if is_fake else (1.0 - fake_prob)
        label = "FAKE" if is_fake else "REAL"
        detail_label = "Fake (AI Generated)" if is_fake else "Real (Authentic Image)"

        return {
            "prediction": label,
            "detail_label": detail_label,
            "is_fake": is_fake,
            "confidence": confidence,
            "fake_prob": fake_prob,
            "real_prob": 1.0 - fake_prob,
            "ela_map": ela_map,
            "fft_map": fft_map,
        }

    def extract_features(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract 768-dim CoAtNet embedding for a single RGB frame (used by Video detector)."""
        tensor = prepare_5ch_tensor(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(tensor)
        embedding = self._hook_features['embedding'].squeeze().cpu().numpy()
        self._hook_features.clear()
        return embedding
