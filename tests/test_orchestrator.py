import unittest
import os
from unittest.mock import patch, MagicMock

from orchestrator import InferenceOrchestrator

class TestOrchestrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We mock the predictors to avoid loading huge models or running slow inference (SyncNet) in CI repeatedly,
        # but the orchestrator initialization proves model loading doesn't crash.
        # However, to be safe during setUpClass, we mock them before init.
        pass

    def setUp(self):
        # Provide clean mocks for each predictor
        patcher_trufor = patch('orchestrator.TruForPredictor')
        patcher_aasist = patch('orchestrator.AASISTPredictor')
        patcher_syncnet = patch('orchestrator.SyncNetPredictor')
        
        self.mock_trufor_cls = patcher_trufor.start()
        self.mock_aasist_cls = patcher_aasist.start()
        self.mock_syncnet_cls = patcher_syncnet.start()
        
        self.addCleanup(patcher_trufor.stop)
        self.addCleanup(patcher_aasist.stop)
        self.addCleanup(patcher_syncnet.stop)
        
        # Configure mock return values for typical responses
        self.mock_trufor = self.mock_trufor_cls.return_value
        import numpy as np
        dummy_map = np.zeros((32, 32), dtype=np.float32)
        self.mock_trufor.predict.return_value = {
            "visual_score": 0.85,
            "anomaly_map": dummy_map,
            "reliability_map": dummy_map
        }
        
        self.mock_aasist = self.mock_aasist_cls.return_value
        self.mock_aasist.predict.return_value = {
            "audio_score": 0.15,
            "sample_rate": 16000,
            "num_samples": 64600
        }
        
        self.mock_syncnet = self.mock_syncnet_cls.return_value
        self.mock_syncnet.predict.return_value = {
            "offset_frames": 1,
            "confidence": 8.5,
            "fps": 25.0,
            "offset_ms": 40.0
        }
        
        # Initialize orchestrator
        self.orchestrator = InferenceOrchestrator()
        
        # Dummy test files
        self.dummy_img = "dummy.jpg"
        self.dummy_aud = "dummy.wav"
        
        with open(self.dummy_img, "wb") as f:
            f.write(b"")
        with open(self.dummy_aud, "wb") as f:
            f.write(b"")

    def tearDown(self):
        if os.path.exists(self.dummy_img): os.remove(self.dummy_img)
        if os.path.exists(self.dummy_aud): os.remove(self.dummy_aud)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.orchestrator.analyze("nonexistent_path_123.jpg")

    def test_image_routing(self):
        result = self.orchestrator.analyze(self.dummy_img)
        self.mock_trufor.predict.assert_called_once()
        self.mock_aasist.predict.assert_not_called()
        self.mock_syncnet.predict.assert_not_called()
        
        self.assertEqual(result["modalities"]["visual_score"], 0.85)
        self.assertIsNone(result["modalities"]["audio_score"])
        self.assertIsNone(result["modalities"]["sync_desync_score"])
        
        self.assertIsNotNone(result["visual_evidence"]["anomaly_map_path"])
        self.assertIsNotNone(result["visual_evidence"]["reliability_map_path"])
        self.assertTrue(os.path.exists(result["visual_evidence"]["anomaly_map_path"]))
        self.assertTrue(os.path.exists(result["visual_evidence"]["reliability_map_path"]))
        
        self.assertIsNotNone(result["processing"]["visual_time_ms"])
        self.assertIsNone(result["processing"]["audio_time_ms"])

    def test_audio_routing(self):
        result = self.orchestrator.analyze(self.dummy_aud)
        self.mock_aasist.predict.assert_called_once()
        self.mock_trufor.predict.assert_not_called()
        self.mock_syncnet.predict.assert_not_called()
        
        self.assertEqual(result["modalities"]["audio_score"], 0.15)
        self.assertIsNone(result["modalities"]["visual_score"])
        self.assertIsNone(result["modalities"]["sync_desync_score"])
        
        self.assertIsNotNone(result["processing"]["audio_time_ms"])
        self.assertIsNone(result["processing"]["visual_time_ms"])
        
    @patch('orchestrator.InferenceOrchestrator._extract_middle_frame')
    def test_video_routing(self, mock_extract):
        mock_extract.return_value = self.dummy_img
        
        dummy_vid = "dummy.mp4"
        with open(dummy_vid, "wb") as f:
            f.write(b"")
            
        try:
            result = self.orchestrator.analyze(dummy_vid)
            self.mock_trufor.predict.assert_called_once()
            self.mock_aasist.predict.assert_called_once()
            self.mock_syncnet.predict.assert_called_once()
            
            # Verify output structure
            self.assertIn("manipulation_score", result)
            self.assertIn("decision", result)
            self.assertEqual(result["modalities"]["visual_score"], 0.85)
            self.assertEqual(result["modalities"]["audio_score"], 0.15)
            self.assertIsNotNone(result["modalities"]["sync_desync_score"])
            
            self.assertIsNotNone(result["sync_evidence"]["offset_frames"])
            self.assertIsNotNone(result["sync_evidence"]["offset_ms"])
            self.assertIsNotNone(result["sync_evidence"]["confidence"])
            self.assertIsNotNone(result["sync_evidence"]["reliable"])
            
            self.assertIsNotNone(result["processing"]["visual_time_ms"])
            self.assertIsNotNone(result["processing"]["audio_time_ms"])
            self.assertIsNotNone(result["processing"]["sync_time_ms"])
            self.assertIsNotNone(result["processing"]["total_time_ms"])
        finally:
            if os.path.exists(dummy_vid): os.remove(dummy_vid)

    def test_missing_modality_behavior(self):
        # Force TruFor to fail
        self.mock_trufor.predict.side_effect = Exception("Mocked failure")
        try:
            result = self.orchestrator.analyze(self.dummy_img)
            self.assertIsNone(result["modalities"]["visual_score"])
            self.assertIn("Visual analysis unavailable", result["evidence"])
        finally:
            self.mock_trufor.predict.side_effect = None
        # Should gracefully return empty/null rather than crash
        
if __name__ == "__main__":
    unittest.main()
