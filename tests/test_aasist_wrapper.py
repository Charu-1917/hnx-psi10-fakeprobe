import unittest
import os
import time

from wrappers.aasist_wrapper import AASISTPredictor

class TestAASISTWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = AASISTPredictor(device="cpu")
        cls.test_video_path = r"c:\charu hack\tests\test_video.mp4"

    def test_missing_input(self):
        with self.assertRaises(FileNotFoundError):
            self.predictor.predict("nonexistent_video_123.mp4")

    def test_inference_on_real_video(self):
        # Skip if test video doesn't exist
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Real test video not found at {self.test_video_path}")
            
        start_time = time.time()
        result1 = self.predictor.predict(self.test_video_path)
        runtime1 = time.time() - start_time
        print(f"\nReal video audio_score: {result1['audio_score']:.6f} (runtime: {runtime1:.3f}s)")
        
        self.assertIn("audio_score", result1)
        self.assertIn("sample_rate", result1)
        self.assertIn("num_samples", result1)
        
        self.assertTrue(0.0 <= result1["audio_score"] <= 1.0)
        self.assertEqual(result1["sample_rate"], 16000)
        self.assertEqual(result1["num_samples"], 64600)
        
        # Verify repeated prediction works without reloading and produces consistent result
        start_time2 = time.time()
        result2 = self.predictor.predict(self.test_video_path)
        runtime2 = time.time() - start_time2
        print(f"Repeated prediction audio_score: {result2['audio_score']:.6f} (runtime: {runtime2:.3f}s)")
        
        self.assertAlmostEqual(result1["audio_score"], result2["audio_score"], places=5)

if __name__ == "__main__":
    unittest.main()
