import unittest
import os
import numpy as np
from PIL import Image

from wrappers.trufor_wrapper import TruForPredictor

class TestTruForWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a tiny test image (e.g. 100x100) to test padding/cropping correctly
        cls.test_image_path = "temp_test_image.png"
        img = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
        img.save(cls.test_image_path)
        
        # Load the predictor once for the test suite
        cls.predictor = TruForPredictor(device="cpu")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)

    def test_missing_image(self):
        with self.assertRaises(FileNotFoundError):
            self.predictor.predict("nonexistent_image_123.jpg")

    def test_inference_shapes_and_bounds(self):
        result = self.predictor.predict(self.test_image_path)
        
        self.assertIn("visual_score", result)
        self.assertIn("anomaly_map", result)
        self.assertIn("reliability_map", result)
        
        visual_score = result["visual_score"]
        anomaly_map = result["anomaly_map"]
        reliability_map = result["reliability_map"]
        
        self.assertTrue(0.0 <= visual_score <= 1.0)
        self.assertEqual(anomaly_map.shape, (100, 100))
        self.assertEqual(reliability_map.shape, (100, 100))
        
        # Check map value ranges
        self.assertTrue(np.all((anomaly_map >= 0.0) & (anomaly_map <= 1.0)))
        self.assertTrue(np.all((reliability_map >= 0.0) & (reliability_map <= 1.0)))

if __name__ == "__main__":
    unittest.main()
