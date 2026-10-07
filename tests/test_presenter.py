"""Presenter tests on hand-written fixtures (no model is run)."""
import json
import os
import unittest

from presenter import present

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAST_BASE = {
    "sample_name": "x", "final_fake_probability": 0.81, "final_real_probability": 0.19, "confidence": 0.62,
    "reliability": 0.7, "reliability_level": "HIGH", "uncertainty_score": 0.1, "agreement_score": 1.0,
    "quality": {"quality_level": "GOOD", "is_acceptable": True, "issues": []},
    "evidence_list": [], "reasons": ["The face texture shows generator artifacts."], "summary_text": "s",
    "detector_summaries": [],
}


def fast(modality, **kw):
    d = dict(FAST_BASE, modality=modality, final_verdict="FAKE")
    d.update(kw)
    return d


class PresenterTests(unittest.TestCase):
    def test_fast_image_face_box_is_approximate_and_originals_kept(self):
        src = fast("image", _extras={"frame_size": {"w": 800, "h": 600}, "face_box": [300, 100, 500, 360]})
        out = present(src, "image")
        self.assertEqual(out["verdict_label"], "AI_GENERATED")
        self.assertEqual(out["ai_probability_pct"], 81)
        self.assertEqual(out["certainty"], "high")  # reliability HIGH, confidence .62 > .5
        r = out["frame_regions"][0]["regions"][0]
        self.assertEqual((r["x"], r["y"], r["w"], r["h"], r["source"]), (300, 100, 200, 260, "face_box"))
        self.assertIsNone(r["mean_anomaly"])  # the engine gives none: nothing invented
        self.assertIn("approximate", out["all_detected_areas"][0]["what"])
        self.assertEqual(out["final_verdict"], "FAKE")  # original key preserved
        self.assertEqual(src.get("verdict_label"), None)  # input not mutated

    def test_fast_audio_has_no_segments_only_whole_clip_reason(self):
        out = present(fast("audio", final_verdict="UNCERTAIN", final_fake_probability=0.5, confidence=0.1), "audio")
        self.assertEqual(out["verdict_label"], "POSSIBLY_AI")
        self.assertEqual(out["audio_segments"], [])
        self.assertEqual(out["certainty"], "low")
        self.assertTrue(any("whole clip" in r["text"] for r in out["reasons"]))

    def test_fast_video_five_moments_and_low_quality_gives_not_enough_info(self):
        tm = {"frame_timestamps": [0.0, 1.0, 2.0, 3.0, 4.0],
              "frame_fake_probabilities": [0.2, 0.7, 0.8, 0.3, 0.4],
              "suspicious_frames": [{"frame_index": 1}, {"frame_index": 2}]}
        boxes = [[10, 10, 110, 130], None, [12, 10, 112, 130], None, [11, 11, 111, 131]]
        out = present(fast("video", temporal_metadata=tm, _extras={"frame_size": {"w": 640, "h": 360},
                                                                    "boxes_per_frame": boxes}), "video")
        self.assertEqual(len(out["timeline"]), 5)
        self.assertEqual([p["suspicious"] for p in out["timeline"]], [False, True, True, False, False])
        self.assertEqual(len(out["frame_regions"]), 3)  # frames without a box get no region
        self.assertTrue(any("5 moments" in r["text"] for r in out["reasons"]))
        low = present(fast("video", quality={"quality_level": "LOW", "is_acceptable": False,
                                              "issues": ["image is blurry"]}), "video")
        self.assertEqual(low["verdict_label"], "NOT_ENOUGH_INFO")
        self.assertIsNone(low["ai_probability_pct"])
        self.assertTrue(any(r["signal"] == "quality" and "blurry" in r["text"] for r in low["reasons"]))

    def test_deep_image(self):
        src = {"manipulation_score": 0.72, "decision": "LIKELY_MANIPULATED",
               "modalities": {"visual_score": 0.72, "audio_score": None, "sync_desync_score": None},
               "sync_evidence": {}, "evidence": [], "suspicious_timeline": [],
               "suspicious_regions": [{"x": 300, "y": 140, "width": 200, "height": 260, "area": 52000,
                                       "mean_anomaly": 0.8, "max_anomaly": 0.95, "mean_reliability": 0.7}]}
        out = present(src, "image", frame_size={"w": 800, "h": 600})
        self.assertEqual((out["verdict_label"], out["ai_probability_pct"], out["certainty"]),
                         ("AI_GENERATED", 72, "high"))
        self.assertEqual(out["all_detected_areas"][0]["where"], "middle-centre of the frame")
        self.assertEqual(out["all_detected_areas"][0]["percent_fake"], 80)
        self.assertEqual(out["audio_segments"], [])

    def test_deep_with_no_modalities_is_not_enough_info(self):
        src = {"manipulation_score": 0.0, "decision": "UNCERTAIN",
               "modalities": {"visual_score": None, "audio_score": None, "sync_desync_score": None},
               "sync_evidence": {}, "evidence": [], "suspicious_regions": [], "suspicious_timeline": []}
        self.assertEqual(present(src, "video")["verdict_label"], "NOT_ENOUGH_INFO")

    def test_real_dump_converts(self):
        d = json.load(open(os.path.join(ROOT, "dump.json")))
        out = present(d, "video")
        self.assertEqual(len(out["timeline"]), 20)
        self.assertEqual(max(p["fake_prob"] for p in out["timeline"]), round(d["modalities"]["visual_score"], 4))
        self.assertEqual(len(out["all_detected_areas"]), sum(len(f["regions"]) for f in out["frame_regions"]))


if __name__ == "__main__":
    unittest.main()
