"""
Builds demo_results/{video,image,audio}.json -- SAMPLE data in the UI schema.

Nothing here is a live analysis. Every file is flagged "demo": true and "sample_data": true.
The video sample reuses the per-frame TruFor scores/regions stored in dump.json (a real run on a
local test clip); its audio score, lip-sync offset and frame size are SAMPLE values (the dump used
mocked audio/sync). Image and audio samples are fully synthetic illustrations.

Run:  python scripts/build_demo_results.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    d = json.load(open(os.path.join(ROOT, "dump.json")))
    fw, fh = 1280, 720  # SAMPLE assumption; dump regions fit inside 1184x720
    frames = sorted(d["video_visual"]["evidence_frames"], key=lambda e: e["timestamp_sec"])
    thr = .5
    timeline = [{"t": e["timestamp_sec"], "fake_prob": round(e["visual_score"], 4),
                 "suspicious": e["visual_score"] >= thr} for e in frames]
    frame_regions, areas = [], []
    for e in frames:
        regs = [{"x": r["x"], "y": r["y"], "w": r["width"], "h": r["height"], "mean": r["mean_anomaly"],
                 "max": r["max_anomaly"], "rel": r.get("mean_reliability", 0.0)} for r in e["suspicious_regions"]]
        frame_regions.append({"timestamp_sec": e["timestamp_sec"], "frame_index": e["frame_index"],
                              "score": round(e["visual_score"], 4), "regions": regions_out(regs)})
        for r in regs:
            areas.append({"n": len(areas) + 1, "kind": "region", "time": f"{e['timestamp_sec']}s",
                          "where": where(r["x"], r["y"], r["w"], r["h"], fw, fh),
                          "what": what(r["w"], r["h"], r["x"], fw, fh),
                          "percent_fake": round(r["mean"] * 100), "label": label(r["mean"])})
    tl = d["suspicious_timeline"]
    ivs = ", ".join(f"{i['start_time']:g}s" if i["start_time"] == i["end_time"]
                    else f"{i['start_time']:g}–{i['end_time']:g}s" for i in tl)
    peak = max(frames, key=lambda e: e["visual_score"])
    hot = [e for e in frames if e["visual_score"] >= thr]
    v, a, off = peak["visual_score"], 0.62, 4  # a, off = SAMPLE values
    ss = sync_score(off)
    fused = (W_VIS * v + W_AUD * a + W_SYNC * ss) / (W_VIS + W_AUD + W_SYNC)
    pr = peak["suspicious_regions"][0]
    reasons = [
        {"signal": "visual", "severity": "bad" if len(hot) > len(frames) / 4 else "warn",
         "text": f"Picture: {len(hot)} of {len(frames)} checked moments show signs of editing. Strongest at "
                 f"{peak['timestamp_sec']}s (score {peak['visual_score']:.2f}), around the "
                 f"{where(pr['x'], pr['y'], pr['width'], pr['height'], fw, fh)} (approximate)."},
        {"signal": "over_time", "severity": "bad" if len(tl) > 1 else "warn",
         "text": f"Over time: suspicious in {len(tl)} separate moments ({ivs}). A repeating pattern is "
                 f"stronger evidence than one odd frame."},
        {"signal": "voice", "severity": "bad" if a > .5 else "ok",
         "text": f"Voice: the synthetic-voice score for the whole clip is {a} "
                 f"({'the voice shows signs of being generated' if a > .5 else 'the voice looks natural'}). "
                 f"The checker gives one score per clip, so it cannot say which seconds are fake."},
        {"signal": "lipsync", "severity": "bad" if off > 2 else "ok",
         "text": f"Voice: lips are {off} frames out of step with the audio "
                 f"{'(typical of dubbed / face-swapped video)' if off > 2 else '(normal)'}."},
    ]
    return {
        "demo": True, "sample_data": True,
        "provenance": "Visual scores, regions and timeline come from a real TruFor run on a local test clip "
                      "(dump.json). Audio score, lip-sync offset and frame size are SAMPLE values. "
                      "This is not a live analysis.",
        "media_type": "video", "engine": "demo",
        "verdict_label": verdict(fused), "ai_probability_pct": round(fused * 100), "certainty": certainty(fused),
        "reasons": reasons, "timeline": timeline, "frame_regions": frame_regions, "audio_segments": [],
        "all_detected_areas": areas, "frame_size": {"w": fw, "h": fh},
        "video_duration_sec": 10.0,
        "manipulation_score": round(fused, 4), "decision": decision(fused),
        "modalities": {"visual_score": round(v, 4), "audio_score": a, "sync_desync_score": round(ss, 4)},
        "sync_evidence": {"offset_frames": off, "offset_ms": round(off / 25.0 * 1000, 1),
                          "confidence": 3.4, "reliable": True},
        "visual_evidence": {"anomaly_map_path": None, "reliability_map_path": None},
        "evidence": [f"Sample result. Visual analysis checked {len(frames)} moments, {len(hot)} above {thr}.",
                     f"Audio spoof score {a} (sample value).", f"Sync offset {off} frames (sample value).",
                     f"The final fusion result is {decision(fused)}."],
        "suspicious_regions": peak["suspicious_regions"],
        "suspicious_timeline": [{k: v2 for k, v2 in i.items() if k != "evidence_frames"} for i in tl],
        "video_visual": {k: d["video_visual"][k] for k in ("mean_score", "peak_score", "peak_timestamp_sec",
                                                           "frames_analyzed")},
        "processing": {"visual_time_ms": d["processing"]["visual_time_ms"], "audio_time_ms": None,
                       "sync_time_ms": None, "total_time_ms": None,
                       "note": "visual time is from the real CPU run; audio/sync were not timed"},
    }


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
