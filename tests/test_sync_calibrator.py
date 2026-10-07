import unittest
import math
from fusion.sync_calibrator import SyncEvidenceCalibrator

class TestSyncCalibrator(unittest.TestCase):
    def setUp(self):
        self.calibrator = SyncEvidenceCalibrator(
            offset_tolerance_frames=2, 
            reliable_confidence_threshold=3.0, 
            max_offset_penalty=15
        )

    def test_zero_offset(self):
        res = self.calibrator.calibrate(0, 5.0)
        self.assertEqual(res["sync_desync_score"], 0.0)
        self.assertTrue(res["reliable"])

    def test_small_positive_offset(self):
        res = self.calibrator.calibrate(1, 5.0)
        self.assertTrue(0.0 < res["sync_desync_score"] <= 0.2)

    def test_small_negative_offset(self):
        res = self.calibrator.calibrate(-2, 5.0)
        self.assertEqual(res["sync_desync_score"], 0.2)

    def test_larger_positive_offset(self):
        res = self.calibrator.calibrate(10, 5.0)
        self.assertTrue(0.2 < res["sync_desync_score"] < 1.0)

    def test_larger_negative_offset(self):
        res = self.calibrator.calibrate(-20, 5.0)
        self.assertEqual(res["sync_desync_score"], 1.0)

    def test_low_confidence(self):
        res = self.calibrator.calibrate(0, 1.0, fps=25.0)
        self.assertIsNone(res["sync_desync_score"])
        self.assertFalse(res["reliable"])
        self.assertEqual(res["offset_frames"], 0)
        self.assertEqual(res["confidence"], 1.0)
        self.assertEqual(res["offset_ms"], 0.0)

    def test_missing_fps(self):
        res = self.calibrator.calibrate(0, 5.0)
        self.assertIsNone(res["offset_ms"])

    def test_invalid_fps(self):
        with self.assertRaises(ValueError):
            self.calibrator.calibrate(0, 5.0, fps=-1)

    def test_nan_confidence(self):
        with self.assertRaises(ValueError):
            self.calibrator.calibrate(0, float("nan"))
            
    def test_inf_offset(self):
        with self.assertRaises(ValueError):
            self.calibrator.calibrate(float("inf"), 5.0)

    def test_score_bounds(self):
        # Extreme offset but high confidence
        res1 = self.calibrator.calibrate(999, 999.0)
        self.assertTrue(0.0 <= res1["sync_desync_score"] <= 1.0)
        
        # Low confidence (should be None)
        res2 = self.calibrator.calibrate(-999, -999.0)
        self.assertIsNone(res2["sync_desync_score"])
        self.assertFalse(res2["reliable"])

    def test_deterministic(self):
        res1 = self.calibrator.calibrate(5, 5.0)
        res2 = self.calibrator.calibrate(5, 5.0)
        self.assertEqual(res1, res2)

    def test_offset_ms_calculation(self):
        res = self.calibrator.calibrate(1, 5.0, fps=25.0)
        self.assertEqual(res["offset_ms"], 40.0)

    def test_regression_case(self):
        # Known real SyncNet result
        res = self.calibrator.calibrate(-1, 9.813, fps=25.0)
        self.assertEqual(res["offset_frames"], -1)
        self.assertEqual(res["offset_ms"], -40.0)
        self.assertEqual(res["confidence"], 9.813)
        self.assertTrue(res["reliable"])
        self.assertTrue(0.0 <= res["sync_desync_score"] <= 1.0)
        
        print("\nRegression Output:", res)

if __name__ == "__main__":
    unittest.main()
