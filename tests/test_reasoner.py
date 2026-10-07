import unittest
import numpy as np
from fusion.reasoner import ForensicEvidenceReasoner

class TestForensicEvidenceReasoner(unittest.TestCase):
    def setUp(self):
        self.reasoner = ForensicEvidenceReasoner(anomaly_threshold=0.5, min_area_pixels=100)
        self.base_payload = {
            "modalities": {"visual_score": 0.8},
            "decision": "LIKELY_MANIPULATED",
            "evidence": []
        }

    def test_zero_map(self):
        # 1. Zero Map
        anomaly_map = np.zeros((100, 100), dtype=np.float32)
        res = self.reasoner.generate_evidence(self.base_payload.copy(), "image", anomaly_map)
        
        self.assertEqual(len(res["suspicious_regions"]), 0)
        self.assertIn("Visual analysis identified no significant anomalous regions.", res["evidence"])

    def test_single_region(self):
        # 2. Single region
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[50:100, 50:100] = 1.0  # 50x50 = 2500 area
        
        res = self.reasoner.generate_evidence(self.base_payload.copy(), "image", anomaly_map)
        
        regions = res["suspicious_regions"]
        self.assertEqual(len(regions), 1)
        r = regions[0]
        self.assertEqual(r["x"], 50)
        self.assertEqual(r["y"], 50)
        self.assertEqual(r["width"], 50)
        self.assertEqual(r["height"], 50)
        self.assertEqual(r["area"], 2500)
        self.assertEqual(r["mean_anomaly"], 1.0)
        self.assertEqual(r["max_anomaly"], 1.0)
        
        self.assertIn("Visual analysis identified 1 anomalous region(s).", res["evidence"])

    def test_multiple_regions(self):
        # 3. Multiple regions
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[10:30, 10:30] = 0.9  # 20x20 = 400 area
        anomaly_map[100:150, 100:150] = 0.8 # 50x50 = 2500 area
        
        res = self.reasoner.generate_evidence(self.base_payload.copy(), "image", anomaly_map)
        self.assertEqual(len(res["suspicious_regions"]), 2)

    def test_noise_filtering(self):
        # 4. Noise filtering (min_area_pixels = 100)
        anomaly_map = np.zeros((100, 100), dtype=np.float32)
        anomaly_map[10:15, 10:15] = 1.0  # 5x5 = 25 area, should be filtered
        
        res = self.reasoner.generate_evidence(self.base_payload.copy(), "image", anomaly_map)
        self.assertEqual(len(res["suspicious_regions"]), 0)

    def test_reliability(self):
        # 5/6. Region stats and Reliability
        anomaly_map = np.zeros((100, 100), dtype=np.float32)
        anomaly_map[20:40, 20:40] = 0.6  # 20x20 = 400 area
        
        rel_map = np.zeros((100, 100), dtype=np.float32)
        rel_map[20:40, 20:40] = 0.75
        
        res = self.reasoner.generate_evidence(self.base_payload.copy(), "image", anomaly_map, rel_map)
        r = res["suspicious_regions"][0]
        
        self.assertEqual(r["mean_anomaly"], 0.6)
        self.assertEqual(r["mean_reliability"], 0.75)

    def test_incompatible_reliability(self):
        # 7. Incompatible Reliability
        anomaly_map = np.zeros((100, 100), dtype=np.float32)
        anomaly_map[20:40, 20:40] = 1.0
        
        bad_rel_map = np.zeros((50, 50), dtype=np.float32)
        
        # Should not crash, just omits mean_reliability
        res = self.reasoner.generate_evidence(self.base_payload.copy(), "image", anomaly_map, bad_rel_map)
        self.assertEqual(len(res["suspicious_regions"]), 1)
        self.assertNotIn("mean_reliability", res["suspicious_regions"][0])

    def test_video_evidence(self):
        # 9. Video evidence prefix
        payload = {"modalities": {"visual_score": 0.5}}
        res = self.reasoner.generate_evidence(payload, "video", np.zeros((10,10), dtype=np.float32))
        self.assertIn("Representative-frame visual analysis identified no significant anomalous regions.", res["evidence"])

    def test_audio_evidence(self):
        # 10. Audio evidence
        payload = {"modalities": {"audio_score": 0.852}}
        res = self.reasoner.generate_evidence(payload, "audio")
        self.assertIn("Audio forensic analysis produced an AASIST spoof score of 0.85.", res["evidence"])

    def test_sync_evidence(self):
        # 11 & 12. Reliable and Unreliable Sync
        p_reliable = {
            "sync_evidence": {"reliable": True, "offset_ms": -40.0}
        }
        res1 = self.reasoner.generate_evidence(p_reliable, "video")
        self.assertIn("Audio-video synchronization analysis found an offset of -40.0 ms.", res1["evidence"])
        
        p_unreliable = {
            "sync_evidence": {"reliable": False}
        }
        res2 = self.reasoner.generate_evidence(p_unreliable, "video")
        self.assertIn("Audio-video synchronization evidence was unavailable because model confidence was below the configured reliability threshold.", res2["evidence"])

    def test_missing_modalities(self):
        # 13. Missing Modalities
        res = self.reasoner.generate_evidence({}, "image")
        self.assertIn("Visual analysis was not performed or was unavailable.", res["evidence"])
        self.assertEqual(res["suspicious_regions"], [])

    def test_conclusions(self):
        # 14/15/16. Authentic/Uncertain/Manipulated text
        for dec in ["LIKELY_AUTHENTIC", "UNCERTAIN", "LIKELY_MANIPULATED"]:
            p = {"decision": dec}
            res = self.reasoner.generate_evidence(p, "image")
            self.assertIn(f"The final fusion result is {dec}.", res["evidence"])

    def test_determinism(self):
        # 17. Determinism
        p1 = {"decision": "LIKELY_AUTHENTIC", "modalities": {"audio_score": 0.5}}
        res1 = self.reasoner.generate_evidence(p1.copy(), "audio")
        res2 = self.reasoner.generate_evidence(p1.copy(), "audio")
        self.assertEqual(res1, res2)

if __name__ == "__main__":
    unittest.main()
