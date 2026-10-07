"""
Builds demo_results/{video,image,audio}.json in the UI schema.

Nothing here is a live analysis. Every file is flagged "demo": true.
The video demo is a cached REAL run (dump.json) converted by presenter.present(): demo=true,
sample_data=false. Image and audio demos are synthetic: demo=true, sample_data=true.

Run:  python scripts/build_demo_results.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from presenter import present  # noqa: E402
OUT = os.path.join(ROOT, "demo_results")

W_VIS, W_AUD, W_SYNC = 0.40, 0.30, 0.30  # same weights as fusion/evidence_fusion.py


def where(x, y, w, h, fw, fh):
    cx, cy = (x + w / 2) / fw, (y + h / 2) / fh
    v = "upper" if cy < .35 else "lower" if cy > .65 else "middle"
    hz = "left" if cx < .35 else "right" if cx > .65 else "centre"
    return f"{v}-{hz} of the frame"


def what(w, h, x, fw, fh):
    big = w * h / (fw * fh) > .04
    centred = abs(x + w / 2 - fw / 2) < fw * .2
    return "large central area" if big and centred else "central area" if centred else "edge or background of the picture"


def label(p):
    return "Very likely fake" if p >= .6 else "Maybe fake" if p >= .45 else "Slightly suspicious"


def verdict(score):
    return "AI_GENERATED" if score > .5 else "POSSIBLY_AI" if score >= .35 else "LIKELY_REAL"


def certainty(score):
    d = abs(score - .5)
    return "high" if d > .25 else "medium" if d > .10 else "low"


def decision(score):  # thresholds of EvidenceFusionEngine
    return "LIKELY_AUTHENTIC" if score < .30 else "LIKELY_MANIPULATED" if score >= .70 else "UNCERTAIN"


def sync_score(offset, tol=2, mx=15):  # mirrors SyncEvidenceCalibrator (reliable case)
    a = abs(offset)
    if a <= tol:
        return a / tol * .2
    return .2 + (min(a, mx) - tol) / (mx - tol) * .8


def regions_out(regs, src="heatmap"):
    return [{"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"], "mean_anomaly": r["mean"],
             "max_anomaly": r["max"], "reliability": r["rel"], "source": src} for r in regs]


def write(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print("wrote", name)


def build_video():
    """Cached REAL run: dump.json (Ranjith's temporal TruFor run) converted by the presenter.

    In that run audio (AASIST) and lip-sync (SyncNet) were MOCKED placeholders (see dump_results.py), so
    they are removed here and the score is the visual-only fusion, exactly what EvidenceFusionEngine
    does when a modality is missing (weights are renormalised over the available ones)."""
    d = json.load(open(os.path.join(ROOT, "dump.json")))
    v = d["modalities"]["visual_score"]
    d["modalities"] = {"visual_score": v, "audio_score": None, "sync_desync_score": None}
    d["sync_evidence"] = {"offset_frames": None, "offset_ms": None, "confidence": None, "reliable": None}
    d["manipulation_score"] = v
    d["decision"] = decision(v)
    d["evidence"] = [f"Visual analysis checked {d['video_visual']['frames_analyzed']} moments of the video.",
                     "Audio and lip-sync were not measured in this cached run.",
                     f"The final fusion result is {d['decision']}."]
    d["visual_evidence"] = {"anomaly_map_path": None, "reliability_map_path": None}  # local paths removed
    for e in d["video_visual"]["evidence_frames"]:
        e["artifact_paths"] = {}
    for t in d["suspicious_timeline"]:
        for e in t["evidence_frames"]:
            e["artifact_paths"] = {}
    d["processing"] = {"visual_time_ms": d["processing"]["visual_time_ms"], "audio_time_ms": None,
                       "sync_time_ms": None, "total_time_ms": None,
                       "note": "visual time is from the real CPU run; audio/sync were not run"}
    out = present(d, "video", "demo")
    out["reasons"].append({"signal": "voice", "severity": "warn",
                           "text": "Voice and lip-sync: not measured in this cached run, so they are not part of the score."})
    out.update({"demo": True, "sample_data": False, "cached_real_run": True, "video_duration_sec": 10.0,
                "provenance": "Cached result of a real run of the deep engine (TruFor visual analysis on a local "
                              "test clip, 20 moments, CPU). Audio and lip-sync were not measured. "
                              "The frame size is estimated from the box positions."})
    return out


def build_image():
    fw, fh = 800, 600
    regs = [{"x": 300, "y": 140, "w": 200, "h": 260, "mean": .78, "max": .95, "rel": .84},
            {"x": 40, "y": 30, "w": 120, "h": 90, "mean": .52, "max": .70, "rel": .70}]
    score = .82
    areas = [{"n": i + 1, "kind": "region", "time": "-", "where": where(r["x"], r["y"], r["w"], r["h"], fw, fh),
              "what": what(r["w"], r["h"], r["x"], fw, fh), "percent_fake": round(r["mean"] * 100),
              "label": label(r["mean"])} for i, r in enumerate(regs)]
    return {
        "demo": True, "sample_data": True,
        "provenance": "Fully synthetic illustration. No image was analysed.",
        "media_type": "image", "engine": "demo",
        "verdict_label": verdict(score), "ai_probability_pct": round(score * 100), "certainty": certainty(score),
        "reasons": [{"signal": "visual", "severity": "bad",
                     "text": f"Picture: {len(regs)} areas look edited or generated. The main one is in the "
                             f"{areas[0]['where']} ({areas[0]['percent_fake']}% fake)."}],
        "timeline": [],
        "frame_regions": [{"timestamp_sec": None, "frame_index": None, "score": score, "regions": regions_out(regs)}],
        "audio_segments": [], "all_detected_areas": areas, "frame_size": {"w": fw, "h": fh},
        "manipulation_score": score, "decision": decision(score),
        "modalities": {"visual_score": score, "audio_score": None, "sync_desync_score": None},
        "sync_evidence": {"offset_frames": None, "offset_ms": None, "confidence": None, "reliable": None},
        "visual_evidence": {"anomaly_map_path": None, "reliability_map_path": None},
        "evidence": ["Sample result. Visual analysis identified 2 anomalous region(s).",
                     f"The final fusion result is {decision(score)}."],
        "suspicious_regions": [{"x": r["x"], "y": r["y"], "width": r["w"], "height": r["h"],
                                "area": r["w"] * r["h"], "mean_anomaly": r["mean"], "max_anomaly": r["max"],
                                "mean_reliability": r["rel"]} for r in regs],
        "suspicious_timeline": [],
        "processing": {"visual_time_ms": None, "audio_time_ms": None, "sync_time_ms": None, "total_time_ms": None},
    }


def build_audio():
    score, dur = .78, 18.0
    segs = [{"start": 3.2, "end": 4.6, "score": .81, "reason": "flat, robotic pitch and no natural breathing"},
            {"start": 9.0, "end": 10.4, "score": .72,
             "reason": "over-smooth sound texture typical of voice-cloning models"},
            {"start": 14.0, "end": 15.2, "score": .66, "reason": "the voice sounds unnaturally steady here"}]
    areas = [{"n": i + 1, "kind": "audio", "time": f"{s['start']}s – {s['end']}s", "where": "in the voice",
              "what": "voice", "percent_fake": round(s["score"] * 100), "label": label(s["score"])}
             for i, s in enumerate(segs)]
    return {
        "demo": True, "sample_data": True,
        "provenance": "Fully synthetic illustration. The segment times are made up: the real audio model "
                      "(AASIST) returns one score per clip, not time segments.",
        "media_type": "audio", "engine": "demo",
        "verdict_label": verdict(score), "ai_probability_pct": round(score * 100), "certainty": certainty(score),
        "reasons": [{"signal": "voice", "severity": "bad",
                     "text": f"Voice: synthetic-voice score {score}. Parts that sound fake (sample): "
                             + "; ".join(f"{s['start']}s–{s['end']}s: {s['reason']}" for s in segs) + "."}],
        "timeline": [], "frame_regions": [], "audio_segments": segs, "all_detected_areas": areas,
        "frame_size": None, "audio_duration_sec": dur,
        "manipulation_score": score, "decision": decision(score),
        "modalities": {"visual_score": None, "audio_score": score, "sync_desync_score": None},
        "sync_evidence": {"offset_frames": None, "offset_ms": None, "confidence": None, "reliable": None},
        "visual_evidence": {"anomaly_map_path": None, "reliability_map_path": None},
        "evidence": [f"Sample result. Audio forensic analysis produced a spoof score of {score:.2f}.",
                     f"The final fusion result is {decision(score)}."],
        "suspicious_regions": [], "suspicious_timeline": [],
        "processing": {"visual_time_ms": None, "audio_time_ms": None, "sync_time_ms": None, "total_time_ms": None},
    }


if __name__ == "__main__":
    write("video.json", build_video())
    write("image.json", build_image())
    write("audio.json", build_audio())
