import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np

# Path to TruFor repository in scratch
TRUFOR_ROOT = r"C:\Users\yaso0\.gemini\antigravity-ide\brain\fe3baf12-faf4-4c52-9aae-e0bcb76cb59c\scratch\TruFor\TruFor_train_test"
if TRUFOR_ROOT not in sys.path:
    sys.path.insert(0, TRUFOR_ROOT)

from lib.config import config
from lib.models.cmx.builder_np_conf import EncoderDecoder

def main():
    print("=== TRUFOR SMOKE TEST ===")
    device = torch.device("cpu")
    
    # 1. Load TruFor configuration
    config_file = os.path.join(TRUFOR_ROOT, "lib", "config", "trufor_ph3.yaml")
    config.defrost()
    config.merge_from_file(config_file)
    config.freeze()
    
    # 2. Instantiate model and load checkpoint
    weights_path = r"c:\charu hack\weights\trufor.pth.tar"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")
    
    model = EncoderDecoder(cfg=config)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    print("model loaded: TRUE")
    print(f"device: {device}")
    
    # 3. Create tiny test image (256x256 RGB, multiple of 32 for MiT backbone)
    H, W = 256, 256
    img_data = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    rgb_tensor = torch.tensor(img_data.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0) / 256.0
    rgb_tensor = rgb_tensor.to(device)
    print(f"input dimensions: {list(rgb_tensor.shape)}")
    
    # 4. Run inference & measure runtime
    start_time = time.time()
    with torch.no_grad():
        pred, conf, det, npp = model(rgb_tensor, save_np=False)
    runtime = time.time() - start_time
    
    # 5. Extract outputs
    score = None
    if det is not None:
        score = torch.sigmoid(det).item()
    
    pred = torch.squeeze(pred, 0)
    anomaly_map = F.softmax(pred, dim=0)[1].cpu().numpy()
    
    reliability_map = None
    if conf is not None:
        conf = torch.squeeze(conf, 0)
        conf = torch.sigmoid(conf)[0]
        reliability_map = conf.cpu().numpy()
        
    print(f"score: {score:.6f}" if score is not None else "score: None")
    print(f"anomaly map shape: {anomaly_map.shape}")
    print(f"reliability map shape: {reliability_map.shape if reliability_map is not None else 'None'}")
    print(f"runtime: {runtime:.3f}s")
    print("=== TRUFOR SMOKE TEST PASSED ===")

if __name__ == "__main__":
    main()
