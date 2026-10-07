"""
Presenter: turns an engine result into the plain-language UI schema.

Accepts two input shapes and returns the ORIGINAL keys plus new ones (original keys are never removed):
  * deep   - the InferenceOrchestrator result (manipulation_score, modalities, video_visual ...)
  * fast   - FakeProbe-X UnifiedForensicReport.to_dict() (final_verdict, final_fake_probability ...),
             optionally with the "_extras" block added by engines.run_fast (face boxes, frame size)

Rule: only values present in the result are used. Missing data is omitted, never invented.
"""

LABELS = ("AI_GENERATED", "POSSIBLY_AI", "LIKELY_REAL", "NOT_ENOUGH_INFO")
SUSPICIOUS_FRAME_SCORE = 0.5  # same threshold orchestrator.py uses for suspicious frames


# ------------------------------------------------------------------ small helpers
def _where(x, y, w, h, fw, fh):
    if not fw or not fh:
        return None
    cx, cy = (x + w / 2) / fw, (y + h / 2) / fh
    v = "upper" if cy < .35 else "lower" if cy > .65 else "middle"
    hz = "left" if cx < .35 else "right" if cx > .65 else "centre"
    return f"{v}-{hz} of the frame"


def _what(x, w, h, fw, fh):
    if not fw or not fh:
        return "an area of the picture"
    big = w * h / (fw * fh) > .04
    centred = abs(x + w / 2 - fw / 2) < fw * .2
    return "large central area" if big and centred else "central area" if centred else "edge or background of the picture"


def _sev_label(p):
    return "Very likely fake" if p >= .6 else "Maybe fake" if p >= .45 else "Slightly suspicious"


def _certainty_from_score(score):
    d = abs(score - .5)
    return "high" if d > .25 else "medium" if d > .10 else "low"


def _label_from_score(score):
    return "AI_GENERATED" if score > .5 else "POSSIBLY_AI" if score >= .35 else "LIKELY_REAL"


def _fmt_t(t):
    return f"{t:g}s"


def _frame_size(given, boxes):
    """Real size if known, otherwise a lower bound from the boxes (flagged as estimated)."""
    if given and given.get("w") and given.get("h"):
        return {"w": int(given["w"]), "h": int(given["h"])}
    if boxes:
        return {"w": int(max(b[0] + b[2] for b in boxes)), "h": int(max(b[1] + b[3] for b in boxes)),
                "estimated": True}
    return None


def _areas_from_frames(frame_regions, fs, media_type):
    areas = []
    for fr in frame_regions:
        for r in fr["regions"]:
            m = r.get("mean_anomaly")
            pct = round(m * 100) if m is not None else round(fr["score"] * 100)
            areas.append({
                "n": len(areas) + 1, "kind": "region",
                "time": _fmt_t(fr["timestamp_sec"]) if fr.get("timestamp_sec") is not None else "-",
                "where": _where(r["x"], r["y"], r["w"], r["h"], fs and fs["w"], fs and fs["h"]) or "position unknown",
                "what": "face area (approximate)" if r.get("source") == "face_box"
                else _what(r["x"], r["w"], r["h"], fs and fs["w"], fs and fs["h"]),
                "percent_fake": pct, "label": _sev_label(pct / 100),
                "frame_index": fr.get("frame_index"),
            })
    return areas


# ------------------------------------------------------------------ deep shape
def _present_deep(res, media_type, frame_size):
    mods = res.get("modalities") or {}
    score = res.get("manipulation_score")
    have = [k for k in ("visual_score", "audio_score", "sync_desync_score") if mods.get(k) is not None]
    reasons, timeline, frame_regions = [], [], []

    # ---- visual / over time
    vv = res.get("video_visual")
    if media_type == "video" and vv and vv.get("evidence_frames"):
        frames = sorted(vv["evidence_frames"], key=lambda e: e["timestamp_sec"])
        for e in frames:
            regs = [{"x": r["x"], "y": r["y"], "w": r["width"], "h": r["height"],
                     "mean_anomaly": r.get("mean_anomaly"), "max_anomaly": r.get("max_anomaly"),
                     "reliability": r.get("mean_reliability"), "source": "heatmap"}
                    for r in e.get("suspicious_regions", [])]
            frame_regions.append({"timestamp_sec": e["timestamp_sec"], "frame_index": e.get("frame_index"),
                                  "score": round(e["visual_score"], 4), "regions": regs})
            timeline.append({"t": e["timestamp_sec"], "fake_prob": round(e["visual_score"], 4),
                             "suspicious": e["visual_score"] >= SUSPICIOUS_FRAME_SCORE})
    elif media_type in ("image", "video") and res.get("suspicious_regions") is not None and mods.get("visual_score") is not None:
        regs = [{"x": r["x"], "y": r["y"], "w": r["width"], "h": r["height"], "mean_anomaly": r.get("mean_anomaly"),
                 "max_anomaly": r.get("max_anomaly"), "reliability": r.get("mean_reliability"), "source": "heatmap"}
                for r in res["suspicious_regions"]]
        frame_regions.append({"timestamp_sec": 0.0 if media_type == "video" else None,
                              "frame_index": 0 if media_type == "video" else None,
                              "score": round(mods["visual_score"], 4), "regions": regs})

    all_boxes = [(r["x"], r["y"], r["w"], r["h"]) for fr in frame_regions for r in fr["regions"]]
    fs = _frame_size(frame_size, all_boxes)

    if mods.get("visual_score") is not None and frame_regions:
        n = len(frame_regions)
        peak = max(frame_regions, key=lambda f: f["score"])
        hot = [f for f in frame_regions if f["score"] >= SUSPICIOUS_FRAME_SCORE]
        first = peak["regions"][0] if peak["regions"] else None
        pos = _where(first["x"], first["y"], first["w"], first["h"], fs and fs["w"], fs and fs["h"]) if first else None
        pos_txt = f", around the {pos} (approximate)" if pos else ""
        if media_type == "image":
            reasons.append({"signal": "visual", "severity": "bad" if peak["score"] > .5 else "warn",
                            "text": f"Picture: {len(peak['regions'])} area(s) look edited or generated"
                                    + (f", the main one in the {pos}" if pos else "") + "."})
        elif n <= 1:
            reasons.append({"signal": "visual", "severity": "bad" if peak["score"] > .5 else "warn",
                            "text": "Picture: only one moment of the video was checked, so this says little about "
                                    f"the rest. That moment scored {peak['score']:.2f}{pos_txt}."})
        else:
            reasons.append({"signal": "visual", "severity": "bad" if len(hot) > n / 4 else "warn",
                            "text": f"Picture: {len(hot)} of {n} checked moments show signs of editing. Strongest at "
                                    f"{_fmt_t(peak['timestamp_sec'])} (score {peak['score']:.2f}){pos_txt}."})
    if media_type == "video" and res.get("suspicious_timeline"):
        tl = res["suspicious_timeline"]
        ivs = ", ".join(_fmt_t(i["start_time"]) if i["start_time"] == i["end_time"]
                        else f"{_fmt_t(i['start_time'])}–{_fmt_t(i['end_time'])}" for i in tl)
        reasons.append({"signal": "over_time", "severity": "bad" if len(tl) > 1 else "warn",
                        "text": f"Over time: suspicious in {len(tl)} separate moment(s) ({ivs}). "
                                "A repeating pattern is stronger evidence than one odd frame."})

    # ---- voice (whole clip only: AASIST gives one score per file)
    audio_segments = []
    a = mods.get("audio_score")
    if a is not None and media_type in ("audio", "video"):
        reasons.append({"signal": "voice", "severity": "bad" if a > .5 else "ok",
                        "text": f"Voice: the synthetic-voice score for the whole clip is {a:.2f} "
                                f"({'the voice shows signs of being generated' if a > .5 else 'the voice looks natural'}). "
                                "The checker gives one score per clip, so it cannot say which seconds are fake."})

    # ---- lip-sync
    se = res.get("sync_evidence") or {}
    if media_type == "video" and se.get("reliable") is True and se.get("offset_frames") is not None:
        off = abs(se["offset_frames"])
        reasons.append({"signal": "lipsync", "severity": "bad" if off > 2 else "ok",
                        "text": f"Voice: lips are {off} frame(s) out of step with the audio "
                                + ("(typical of dubbed / face-swapped video)." if off > 2 else "(normal).")})
    elif media_type == "video" and se.get("reliable") is False:
        reasons.append({"signal": "lipsync", "severity": "warn",
                        "text": "Lip-sync: it could not be measured reliably (the checker was not confident), "
                                "so it is not used."})

    # ---- verdict
    if score is None or not have:
        label, pct, cert = "NOT_ENOUGH_INFO", None, "low"
        reasons.append({"signal": "quality", "severity": "warn",
                        "text": "No check could be completed, so there is not enough information for a verdict."})
    else:
        label, pct, cert = _label_from_score(score), round(score * 100), _certainty_from_score(score)

    return {"verdict_label": label, "ai_probability_pct": pct, "certainty": cert, "reasons": reasons,
            "timeline": timeline, "frame_regions": frame_regions, "audio_segments": audio_segments,
            "frame_size": fs}


# ------------------------------------------------------------------ fast (FakeProbe-X) shape
_CERT_ORDER = ["low", "medium", "high"]


def _cert_fast(res):
    lvl = str(res.get("reliability_level", "LOW")).lower()
    lvl = lvl if lvl in _CERT_ORDER else "low"
    conf = res.get("confidence")
    if conf is not None:  # the engine's own confidence caps the certainty
        cap = "high" if conf > .5 else "medium" if conf > .2 else "low"
        lvl = _CERT_ORDER[min(_CERT_ORDER.index(lvl), _CERT_ORDER.index(cap))]
    return lvl


def _present_fast(res, media_type, frame_size):
    extras = res.get("_extras") or {}
    fprob = res.get("final_fake_probability")
    q = res.get("quality") or {}
    verdict = {"FAKE": "AI_GENERATED", "REAL": "LIKELY_REAL", "UNCERTAIN": "POSSIBLY_AI"}.get(
        res.get("final_verdict"), "NOT_ENOUGH_INFO")
    low_quality = q.get("quality_level") == "LOW" or q.get("is_acceptable") is False
    if low_quality:
        verdict = "NOT_ENOUGH_INFO"
    sev = {"AI_GENERATED": "bad", "POSSIBLY_AI": "warn", "LIKELY_REAL": "ok", "NOT_ENOUGH_INFO": "warn"}[verdict]
    sig = {"image": "visual", "video": "visual", "audio": "voice"}.get(media_type, "visual")

    reasons = []
    for issue in q.get("issues") or []:
        txt = str(issue).strip().rstrip(".")
        reasons.append({"signal": "quality", "severity": "warn",
                        "text": f"Input quality: {txt[:1].upper() + txt[1:]}. This makes the result less reliable."})
    if low_quality:
        reasons.append({"signal": "quality", "severity": "warn",
                        "text": "The input quality is too low for a trustworthy verdict, so we do not give one."})
    for r in res.get("reasons") or []:
        reasons.append({"signal": sig, "severity": sev, "text": str(r)})

    timeline, frame_regions, audio_segments = [], [], []
    tm = res.get("temporal_metadata") or {}
    ts, fp = tm.get("frame_timestamps") or [], tm.get("frame_fake_probabilities") or []
    sus = {s.get("frame_index") for s in tm.get("suspicious_frames") or []}
    if media_type == "video" and ts and len(ts) == len(fp):
        for i, (t, p) in enumerate(zip(ts, fp)):
            timeline.append({"t": t, "fake_prob": p, "suspicious": i in sus})
        reasons.append({"signal": "over_time", "severity": "warn" if not sus else "bad",
                        "text": f"Over time: {len(ts)} moments of the video were checked (not every frame); "
                                f"{len(sus)} of them looked suspicious."})

    fs = _frame_size(frame_size or extras.get("frame_size"), None)

    def box(b):  # [x1,y1,x2,y2] -> x,y,w,h (clamped to >=0)
        x1, y1, x2, y2 = (int(v) for v in b)
        return {"x": max(0, x1), "y": max(0, y1), "w": max(0, x2 - x1), "h": max(0, y2 - y1),
                "mean_anomaly": None, "max_anomaly": None, "reliability": None, "source": "face_box"}

    if media_type == "image" and extras.get("face_box"):
        frame_regions.append({"timestamp_sec": None, "frame_index": None, "score": fprob,
                              "regions": [box(extras["face_box"])]})
    elif media_type == "video" and extras.get("boxes_per_frame") and ts:
        for i, b in enumerate(extras["boxes_per_frame"]):
            if b is not None and i < len(ts):
                frame_regions.append({"timestamp_sec": ts[i], "frame_index": i, "score": fp[i] if i < len(fp) else fprob,
                                      "regions": [box(b)]})
    if frame_regions:
        reasons.append({"signal": sig, "severity": "ok",
                        "text": "Position: the box shows the face area the checker looked at. It is approximate: "
                                "this engine scores the whole face, it does not mark exact fake pixels."})

    if media_type == "audio":
        reasons.append({"signal": "voice", "severity": sev,
                        "text": f"Voice: the synthetic-voice score for the whole clip is {fprob:.2f}. "
                                "The checker gives one score per clip, so it cannot say which seconds are fake."})
    if media_type == "video" and extras.get("has_audio") is False:
        reasons.append({"signal": "voice", "severity": "warn",
                        "text": "The video has no usable audio track, so the voice and lip-sync checks were skipped."})

    pct = None if verdict == "NOT_ENOUGH_INFO" or fprob is None else round(fprob * 100)
    return {"verdict_label": verdict, "ai_probability_pct": pct, "certainty": _cert_fast(res), "reasons": reasons,
            "timeline": timeline, "frame_regions": frame_regions, "audio_segments": audio_segments,
            "frame_size": fs}


# ------------------------------------------------------------------ public API
def present(result: dict, media_type: str, engine: str | None = None, frame_size: dict | None = None) -> dict:
    """Return `result` plus the UI keys. Does not mutate the input."""
    if media_type not in ("video", "image", "audio"):
        raise ValueError("media_type must be video, image or audio")
    if "verdict_label" in result:  # already presented (e.g. cached demo)
        return dict(result)
    if "final_verdict" in result:
        shape, ui = "fast", _present_fast(result, media_type, frame_size)
    elif "manipulation_score" in result:
        shape, ui = "deep", _present_deep(result, media_type, frame_size)
    else:
        raise ValueError("Unrecognised result shape (expected FakeProbe-X or orchestrator output)")
    ui["all_detected_areas"] = _areas_from_frames(ui["frame_regions"], ui["frame_size"], media_type)
    out = dict(result)
    out.update(ui)
    out["media_type"] = media_type
    out["engine"] = engine or shape
    return out
