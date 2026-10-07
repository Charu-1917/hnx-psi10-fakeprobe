import unittest
from fusion.evidence_fusion import EvidenceFusionEngine, SyncEvidenceCalibrator

class TestEvidenceFusionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EvidenceFusionEngine(
            w_visual=0.40, w_audio=0.30, w_sync=0.30,
            threshold_authentic=0.30, threshold_manipulated=0.70
        )

    def test_1_visual_only(self):
        res = self.engine.fuse(visual_score=0.8)
        self.assertAlmostEqual(res["manipulation_score"], 0.8)
        self.assertEqual(res["decision"], "LIKELY_MANIPULATED")
        self.assertIsNone(res["modalities"]["audio_score"])

    def test_2_audio_only(self):
        res = self.engine.fuse(audio_score=0.2)
        self.assertAlmostEqual(res["manipulation_score"], 0.2)
        self.assertEqual(res["decision"], "LIKELY_AUTHENTIC")

    def test_3_visual_and_audio(self):
        res = self.engine.fuse(visual_score=0.9, audio_score=0.1)
        # weight visual=0.4, audio=0.3 -> total weight = 0.7
        # score = (0.9*0.4 + 0.1*0.3)/0.7 = 0.39 / 0.7 = 0.5571...
        self.assertAlmostEqual(res["manipulation_score"], 0.39 / 0.7)
        self.assertEqual(res["decision"], "UNCERTAIN")

    def test_4_all_three_modalities(self):
        # Sync offset=0, conf=5.0 -> authentic -> sync_desync_score = 0.0
        res = self.engine.fuse(visual_score=0.5, audio_score=0.5, sync_offset_frames=0, sync_confidence=5.0)
        # score = (0.5*0.4 + 0.5*0.3 + 0.0*0.3)/1.0 = (0.2 + 0.15) = 0.35
        self.assertAlmostEqual(res["manipulation_score"], 0.35)

    def test_5_missing_sync(self):
        res = self.engine.fuse(visual_score=0.1, audio_score=0.1)
        self.assertAlmostEqual(res["manipulation_score"], 0.1)

    def test_6_missing_audio(self):
        # Sync offset=2, conf=4.0 -> manipulated -> sync_desync_score = 1.0 (since > offset_tolerance 1)
        res = self.engine.fuse(visual_score=0.1, sync_offset_frames=2, sync_confidence=4.0)
        # weight visual=0.4, sync=0.3 -> total 0.7
        # score = (0.1*0.4 + 1.0*0.3) / 0.7 = 0.34 / 0.7 = 0.4857...
        self.assertAlmostEqual(res["manipulation_score"], 0.34 / 0.7)

    def test_7_missing_visual(self):
        res = self.engine.fuse(audio_score=1.0, sync_offset_frames=0, sync_confidence=5.0)
        # weight audio=0.3, sync=0.3 -> total 0.6
        # score = (1.0*0.3 + 0.0*0.3) / 0.6 = 0.5
        self.assertAlmostEqual(res["manipulation_score"], 0.5)

    def test_8_no_modalities(self):
        with self.assertRaises(ValueError):
            self.engine.fuse()

    def test_9_score_boundary_0(self):
        res = self.engine.fuse(visual_score=0.0, audio_score=0.0)
        self.assertAlmostEqual(res["manipulation_score"], 0.0)
        self.assertEqual(res["decision"], "LIKELY_AUTHENTIC")

    def test_10_score_boundary_1(self):
        res = self.engine.fuse(visual_score=1.0, audio_score=1.0)
        self.assertAlmostEqual(res["manipulation_score"], 1.0)
        self.assertEqual(res["decision"], "LIKELY_MANIPULATED")

    def test_11_decision_threshold_behavior(self):
        # 0.2999
        res_2999 = self.engine.fuse(visual_score=0.2999)
        self.assertEqual(res_2999["decision"], "LIKELY_AUTHENTIC")
        
        # 0.30
        res_30 = self.engine.fuse(visual_score=0.30)
        self.assertEqual(res_30["decision"], "UNCERTAIN")
        
        # 0.6999
        res_6999 = self.engine.fuse(visual_score=0.6999)
        self.assertEqual(res_6999["decision"], "UNCERTAIN")
        
        # 0.70 (mathematically exact 0.70 can be 0.699999999999999 due to floats)
        # Using 0.7 * 0.4 / 0.4 -> may result in 0.699999999999999
        res_70 = self.engine.fuse(visual_score=0.70)
        self.assertEqual(res_70["decision"], "LIKELY_MANIPULATED")
        
        # 0.7001
        res_7001 = self.engine.fuse(visual_score=0.7001)
        self.assertEqual(res_7001["decision"], "LIKELY_MANIPULATED")

    def test_12_invalid_scores(self):
        with self.assertRaises(ValueError):
            self.engine.fuse(visual_score=1.5)
        with self.assertRaises(ValueError):
            self.engine.fuse(audio_score=-0.1)

    def test_13_syncnet_raw_evidence_preservation(self):
        res = self.engine.fuse(visual_score=0.5, sync_offset_frames=3, sync_confidence=2.5)
        self.assertIsNotNone(res["sync_evidence"])
        self.assertEqual(res["sync_evidence"]["offset_frames"], 3)
        self.assertAlmostEqual(res["sync_evidence"]["confidence"], 2.5)
        self.assertFalse(res["sync_evidence"]["reliable"])  # 2.5 < 3.0 threshold
        # Since < threshold, sync score is 1.0.

    def test_14_offset_conversion_to_ms(self):
        res = self.engine.fuse(sync_offset_frames=2, sync_confidence=5.0)
        # default fps is 25.0 -> 2 frames = 2/25 = 0.08s = 80ms
        self.assertAlmostEqual(res["sync_evidence"]["offset_ms"], 80.0)

if __name__ == "__main__":
    unittest.main()
