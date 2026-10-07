import unittest
import os
import time

from wrappers.syncnet_wrapper import SyncNetPredictor

class TestSyncNetWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = SyncNetPredictor(device="cpu")
        cls.test_video_path = r"c:\charu hack\tests\test_video.mp4"

    def test_missing_input(self):
        with self.assertRaises(FileNotFoundError):
            self.predictor.predict("nonexistent_video_123.mp4")

    def test_inference_on_real_video(self):
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Real test video not found at {self.test_video_path}")
            
        # First Inference
        start_time = time.time()
        result1 = self.predictor.predict(self.test_video_path)
        runtime1 = time.time() - start_time
        
        print(f"\nReal video SyncNet offset: {result1['offset_frames']} frames")
        print(f"Confidence: {result1['confidence']:.3f}")
        print(f"Runtime 1: {runtime1:.3f}s")
        
        self.assertIn("offset_frames", result1)
        self.assertIn("confidence", result1)
        self.assertIn("fps", result1)
        self.assertIn("offset_ms", result1)
        
        self.assertIsInstance(result1["offset_frames"], int)
        self.assertTrue(isinstance(result1["confidence"], float))
        self.assertTrue(isinstance(result1["offset_ms"], float))
        
        # Second Inference to test reuse
        start_time2 = time.time()
        result2 = self.predictor.predict(self.test_video_path)
        runtime2 = time.time() - start_time2
        
        print(f"Runtime 2: {runtime2:.3f}s")
        
        # Offset and confidence should be identical
        self.assertEqual(result1["offset_frames"], result2["offset_frames"])
        self.assertAlmostEqual(result1["confidence"], result2["confidence"], places=3)
        
if __name__ == "__main__":
    unittest.main()
