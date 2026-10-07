import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

TRUFOR_ROOT = r"C:\Users\yaso0\.gemini\antigravity-ide\brain\fe3baf12-faf4-4c52-9aae-e0bcb76cb59c\scratch\TruFor\TruFor_train_test"
TRUFOR_WEIGHTS = r"c:\charu hack\weights\trufor.pth.tar"

class TruForPredictor:
    """
    Isolated, reusable wrapper for TruFor inference.
    Handles configuration, single-time model loading, preprocessing,
    and dimension padding required by the MiT backbone.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._load_model()

    def _load_model(self):
        """
        Dynamically imports TruFor to prevent global sys.path pollution.
        """
        if not os.path.exists(TRUFOR_WEIGHTS):
            raise FileNotFoundError(f"TruFor checkpoint not found at {TRUFOR_WEIGHTS}")

        # Import isolation strategy:
        # We temporarily insert TRUFOR_ROOT into sys.path to import the model,
        # and restore the original path in a finally block to prevent 
        # permanently polluting the global path for other modules.
        original_path = list(sys.path)
        if TRUFOR_ROOT not in sys.path:
            sys.path.insert(0, TRUFOR_ROOT)
        
        try:
            from lib.config import config
            from lib.models.cmx.builder_np_conf import EncoderDecoder

            config_file = os.path.join(TRUFOR_ROOT, "lib", "config", "trufor_ph3.yaml")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config not found at {config_file}")

            # Re-initialize config (defrosting just in case it was frozen by another import)
            config.defrost()
            config.merge_from_file(config_file)
            config.freeze()

            self.model = EncoderDecoder(cfg=config)
            checkpoint = torch.load(TRUFOR_WEIGHTS, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()

        finally:
            sys.path[:] = original_path

    def predict(self, image_path: str) -> dict:
        """
        Runs TruFor inference on a single image.
        
        Returns:
            {
                "visual_score": float,            # [0, 1] manipulation probability
                "anomaly_map": numpy.ndarray,     # 2D map cropped to original image dimensions
                "reliability_map": numpy.ndarray  # 2D map cropped to original image dimensions
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        try:
            # 1. Load image (RGB)
            img = Image.open(image_path).convert("RGB")
            img_arr = np.array(img)
            H_orig, W_orig = img_arr.shape[:2]
            
            # 2. Preprocessing (division by 256.0 matches TruFor reference dataset_test.py)
            tensor = torch.tensor(img_arr.transpose(2, 0, 1), dtype=torch.float) / 256.0
            
            # 3. Padding to multiples of 32
            pad_h = (32 - H_orig % 32) % 32
            pad_w = (32 - W_orig % 32) % 32
            
            if pad_h > 0 or pad_w > 0:
                # F.pad format is (pad_left, pad_right, pad_top, pad_bottom)
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
                
            tensor = tensor.unsqueeze(0).to(self.device)
            
            # 4. Inference
            with torch.no_grad():
                pred, conf, det, npp = self.model(tensor, save_np=False)
            
            # 5. Output Extraction
            if det is None:
                raise RuntimeError("TruFor model returned None for detection logits.")
                
            visual_score = torch.sigmoid(det).item()
            
            pred = torch.squeeze(pred, 0)
            anomaly_map_padded = F.softmax(pred, dim=0)[1].cpu().numpy()
            
            if conf is None:
                raise RuntimeError("TruFor model returned None for confidence map.")
                
            conf = torch.squeeze(conf, 0)
            reliability_map_padded = torch.sigmoid(conf)[0].cpu().numpy()
            
            # 6. Crop back to original dimensions
            anomaly_map = anomaly_map_padded[:H_orig, :W_orig]
            reliability_map = reliability_map_padded[:H_orig, :W_orig]
            
            # Validation
            assert 0.0 <= visual_score <= 1.0, f"visual_score {visual_score} out of bounds"
            assert anomaly_map.shape == (H_orig, W_orig), "Anomaly map shape mismatch"
            assert reliability_map.shape == (H_orig, W_orig), "Reliability map shape mismatch"
            
            return {
                "visual_score": visual_score,
                "anomaly_map": anomaly_map,
                "reliability_map": reliability_map
            }
            
        except Exception as e:
            raise RuntimeError(f"TruFor inference failed: {str(e)}") from e
