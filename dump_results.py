import time
import json
import uuid
from unittest.mock import patch

from orchestrator import InferenceOrchestrator

def main():
    video_path = r"C:\charu hack\tests\test_video.mp4"
    
    # We will mock AASIST and SyncNet so this script completes quickly.
    # The visual temporal logic inside orchestrator will still run fully.
    with patch('orchestrator.AASISTPredictor') as MockAASIST, patch('orchestrator.SyncNetPredictor') as MockSyncNet:
        MockAASIST.return_value.predict.return_value = {"audio_score": 0.5, "sample_rate": 16000, "num_samples": 16000}
        MockSyncNet.return_value.predict.return_value = {"offset_frames": 1, "confidence": 3.5, "fps": 25.0}
        
        orchestrator = InferenceOrchestrator()
        
        # Override the predictors with our mocks
        orchestrator.aasist = MockAASIST.return_value
        orchestrator.syncnet = MockSyncNet.return_value
        
        t0 = time.time()
        res = orchestrator.analyze(video_path)
        t1 = time.time()
        
    with open('dump.json', 'w') as f:
        json.dump(res, f, indent=2)
        
    print(f"Dumped to dump.json. Runtime: {t1-t0:.2f}s")

if __name__ == '__main__':
    main()
