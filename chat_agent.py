"""
Chat agent: answers questions ONLY from the presented result JSON.

Tools read the result; a keyword router picks tools; the answer is composed from tool output.
Optional: if OLLAMA_MODEL is set and localhost:11434 responds, the draft answer is reworded by the
local LLM (told to use only the evidence). Any error, or any number not in the evidence, falls back
to the rule-based answer.
"""
import json
import os
import re
import urllib.request

LABEL_TXT = {"AI_GENERATED": "AI-generated or edited", "POSSIBLY_AI": "possibly AI (needs review)",
             "LIKELY_REAL": "likely real", "NOT_ENOUGH_INFO": "not enough information for a verdict"}


# ------------------------------------------------------------------ tools
def get_summary(R):
    return {"label": R.get("verdict_label"), "pct": R.get("ai_probability_pct"), "certainty": R.get("certainty"),
            "media_type": R.get("media_type"), "engine": R.get("engine"),
            "sample": bool(R.get("sample_data")),
            "cached": bool(R.get("demo") or R.get("cached_real_run")),
            "n_areas": len(R.get("all_detected_areas") or [])}


def get_frame(R, t):
    for f in R.get("frame_regions") or []:
        if f.get("timestamp_sec") is not None and abs(f["timestamp_sec"] - t) < 1e-6:
            return f
    return None


def get_intervals(R):
    tl = R.get("timeline") or []
    return {"checked": len(tl), "suspicious": [p for p in tl if p.get("suspicious")],
            "deep_intervals": R.get("suspicious_timeline") or [], "audio_segments": R.get("audio_segments") or []}


def modalities(R):
    out = {}
    for r in R.get("reasons") or []:
        out.setdefault(r.get("signal"), []).append({"severity": r.get("severity"), "text": r.get("text")})
    mods = R.get("modalities")
    return {"signals": out, "raw": mods if isinstance(mods, dict) else None, "sync": R.get("sync_evidence")}


def _dups(R):
    seen, d = set(), []
    for p in R.get("timeline") or []:
        if p["t"] in seen:
            d.append(p["t"])
        seen.add(p["t"])
    return d


def explain_peak(R):
    tl = R.get("timeline") or []
    if not tl:
        return {"peak": None}
    p = max(tl, key=lambda x: x["fake_prob"])
    fr = get_frame(R, p["t"])
    note = ("The strongest moment is the very first frame. First frames can score high because of video "
            "start-up effects (fade-in, codec start, face-finder warm-up), so a first-frame peak alone is weak "
            "evidence." if p["t"] == 0 else "")
    return {"peak": p, "regions": len((fr or {}).get("regions") or []), "note": note, "duplicates": _dups(R)}


def audit(R):
    q = [r["text"] for r in (R.get("reasons") or []) if r.get("signal") == "quality"]
    return {"quality_notes": q, "certainty": R.get("certainty"), "reliability_level": R.get("reliability_level"),
            "processing": R.get("processing"), "frames_checked": len(R.get("timeline") or []),
            "duplicates": _dups(R), "frame_size_estimated": bool((R.get("frame_size") or {}).get("estimated"))}


TOOLS = {"get_summary": get_summary, "get_frame": get_frame, "get_intervals": get_intervals,
         "modalities": modalities, "explain_peak": explain_peak, "audit": audit}


# ------------------------------------------------------------------ router
def route(q):
    q = q.lower()
    calls = []
    if re.search(r"uncertain|reliab|trust|sure|confiden|certain|quality|blur|slow|runtime|time|duplicate|audit|correct", q):
        calls += ["audit", "get_summary"]
    if re.search(r"peak|first|frame 0|strongest|highest", q):
        calls.append("explain_peak")
    if re.search(r"interval|timeline|when|where|moment|position|segment", q):
        calls += ["get_intervals", "get_summary"]
    if re.search(r"audio|voice|sync|lip|modal|agree", q):
        calls.append("modalities")
    if not calls:
        calls = ["get_summary", "get_intervals", "modalities"]
    return list(dict.fromkeys(calls))


# ------------------------------------------------------------------ composer
def compose(res, R):
    o = []
    s = res.get("get_summary")
    if s:
        txt = f"Result: {LABEL_TXT.get(s['label'], s['label'])}"
        if s["pct"] is not None:
            txt += f", AI-probability estimate {s['pct']}%, certainty {s['certainty']}"
        txt += "."
        if s["sample"]:
            txt += " This is SAMPLE data, not a live analysis."
        elif s["cached"]:
            txt += " This is a cached result of an earlier run."
        o.append(txt)
    if "explain_peak" in res:
        e = res["explain_peak"]
        if e["peak"]:
            o.append(f"The strongest moment is at {e['peak']['t']:g}s (score {e['peak']['fake_prob']:.2f}), "
                     f"with {e['regions']} marked area(s). {e['note']}".strip())
            if e["duplicates"]:
                o.append("Timestamps sampled twice: " + ", ".join(f"{d:g}s" for d in e["duplicates"]) + ".")
        else:
            o.append("There is no per-moment data in this result, so there is no single peak to explain.")
    if "get_intervals" in res:
        g = res["get_intervals"]
        parts = []
        if g["checked"]:
            if g["suspicious"]:
                parts.append(f"{g['checked']} moments were checked; {len(g['suspicious'])} looked suspicious at "
                             + ", ".join(f"{p['t']:g}s" for p in g["suspicious"][:12]) + ".")
            else:
                parts.append(f"{g['checked']} moments were checked; none looked suspicious.")
        if g["audio_segments"]:
            parts.append("Voice segments: " + "; ".join(
                f"{a['start']:g}-{a['end']:g}s ({a.get('reason', '')})" for a in g["audio_segments"]))
        elif any(r.get("signal") == "voice" for r in R.get("reasons") or []):
            parts.append("This engine gives one voice score for the whole clip, so it cannot name fake seconds.")
        o.append(" ".join(parts) or "This result has no time information.")
        areas = R.get("all_detected_areas") or []
        if areas:
            top = max(areas, key=lambda a: a["percent_fake"])
            o.append(f"{len(areas)} area(s) are listed in step 3. The strongest is #{top['n']}: {top['where']} "
                     f"at {top['time']} ({top['percent_fake']}%, {top['label']}).")
        else:
            o.append("No position boxes exist in this result, so none are shown.")
    if "modalities" in res:
        m = res["modalities"]["signals"]
        lines = [x["text"] for k, xs in m.items() if k != "quality" for x in xs]
        o.append("Signals: " + (" ".join(lines) if lines else "no per-signal explanation is available.")
                 + " When signals disagree, certainty goes down; when they agree it goes up.")
    if "audit" in res:
        a = res["audit"]
        t = f"How reliable: certainty is {a['certainty']}."
        if a["reliability_level"]:
            t += f" Engine reliability: {a['reliability_level']}."
        t += (" Quality notes: " + " ".join(a["quality_notes"])) if a["quality_notes"] \
            else " No input-quality problems were reported."
        if a["frames_checked"]:
            t += f" {a['frames_checked']} moments were checked, not every frame."
        if a["frame_size_estimated"]:
            t += " The frame size was estimated, so box positions are approximate."
        if s and s["label"] == "POSSIBLY_AI":
            t += " It is 'possibly AI' because the score sits between the real and fake zones."
        o.append(t)
    o.append("This is an estimate from pretrained models, not legal proof.")
    return "\n\n".join(o)


# ------------------------------------------------------------------ optional LLM
def _ollama(prompt):
    model = os.environ.get("OLLAMA_MODEL")
    if not model:
        return None
    try:
        req = urllib.request.Request("http://localhost:11434/api/generate", method="POST",
                                     data=json.dumps({"model": model, "stream": False, "prompt": prompt}).encode())
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read()).get("response") or None
    except Exception:
        return None


_NUM = re.compile(r"\d+(?:\.\d+)?")


def _numbers_ok(text, source):
    allowed = set(_NUM.findall(source))
    return all(n in allowed for n in _NUM.findall(text))


def answer(question: str, result: dict) -> dict:
    calls = route(question)
    res = {c: TOOLS[c](result) for c in calls}
    draft = compose(res, result)
    evidence = json.dumps(res, default=str)[:3500]
    llm_used = False
    reworded = _ollama("You are a deepfake forensics assistant. Using ONLY this evidence, answer for a "
                       f"non-expert. Do not add numbers.\nEVIDENCE: {evidence}\nDRAFT: {draft}\nQUESTION: {question}")
    if reworded and _numbers_ok(reworded, draft + evidence):
        draft, llm_used = reworded, True
    return {"answer": draft, "tools_used": calls, "llm_used": llm_used}
