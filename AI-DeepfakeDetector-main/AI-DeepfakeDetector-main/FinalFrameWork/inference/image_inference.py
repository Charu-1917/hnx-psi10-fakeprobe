import cv2
import io
import torch
import torch.nn as nn
import numpy as np
import timm
from PIL import Image

def compute_ela(image_rgb: np.ndarray, quality: int = 90) -> np.ndarray:
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
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude /= mag_max
    return magnitude.astype(np.float32)

def prepare_5ch_tensor(image_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """Convert an RGB image (numpy) to a normalized 5-channel tensor for CoAtNet.
    This replicates the exact preprocessing used during training.
    """
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
    std  = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5]).view(5, 1, 1)
    tensor = (tensor - mean) / std
    return tensor

def build_coatnet_5ch(num_classes=1):
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
    def __init__(self, weights_path="models/best_coatnet_5ch.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_coatnet_5ch()
        
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # ── Feature-extraction hook (mirrors training pipeline) ──
        # Captures the 768-dim embedding from the global average pool
        # layer, which sits just before the classification head.
        self._hook_features = {}
        self._hook_handle = self.model.head.global_pool.register_forward_hook(
            self._capture_hook
        )

    def _capture_hook(self, module, input, output):
        """Forward hook callback — stores the pre-logits embedding."""
        self._hook_features['embedding'] = output.detach()

    def predict(self, image_bytes):
       
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        image_tensor = prepare_5ch_tensor(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            prob = torch.sigmoid(outputs).item()

        if prob > 0.5:
            return "Fake (AI Generated)", prob
        else:
            return "Real Image", 1 - prob

    def extract_features(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract the 768-dim CoAtNet embedding for a single RGB frame.

        This method is used by the Video module to obtain per-frame
        feature vectors without instantiating a second CoAtNet copy.

        Args:
            image_rgb: A numpy array in RGB format (any resolution).

        Returns:
            A 1-D numpy array of shape (768,) — the pre-logits embedding.
        """
        tensor = prepare_5ch_tensor(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.model(tensor)

        embedding = self._hook_features['embedding'].squeeze().cpu().numpy()
        self._hook_features.clear()
        return embedding