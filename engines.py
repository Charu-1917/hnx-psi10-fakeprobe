"""
Engine registry for the API: "demo" (cached sample JSON), "fast" (FakeProbe-X) and "deep"
(Ranjith's TruFor/AASIST/SyncNet orchestrator).

Heavy imports (torch, timm, cv2 ...) only happen when a real engine is first used, so the API starts
and demos fully offline with none of the model dependencies or weights installed.
"""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(ROOT, "demo_results")
MEDIA_TYPES = ("video", "image", "audio")

IMAGE_EXT = {".png", ".jpg", ".jpeg"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


class EngineUnavailable(Exception):
    """Raised when an engine cannot run (missing weights / packages). The API maps it to HTTP 503."""


def media_type_from_name(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    raise ValueError(f"Unsupported media file extension: {ext or '(none)'}")


# ---------------------------------------------------------------- demo
def load_demo(media_type: str, variant: str = "ai") -> dict:
    """variant: "ai" (default file) or "real" (image_real.json / audio_real.json). Video has one demo."""
    if media_type not in MEDIA_TYPES:
        raise ValueError(f"media_type must be one of {MEDIA_TYPES}")
    if variant not in ("ai", "real"):
        raise ValueError("variant must be ai or real")
    name = f"{media_type}_real" if variant == "real" and media_type != "video" else media_type
    path = os.path.join(DEMO_DIR, f"{name}.json")
    if not os.path.isfile(path):
        raise EngineUnavailable(f"Demo file missing: demo_results/{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def demo_ready() -> bool:
    names = [f"{m}.json" for m in MEDIA_TYPES] + ["image_real.json", "audio_real.json"]
    return all(os.path.isfile(os.path.join(DEMO_DIR, n)) for n in names)


# ---------------------------------------------------------------- deep (existing orchestrator)
def _deep_paths() -> dict:
    """Model locations. Env vars override; the defaults are the values hard-coded in wrappers/*."""
    wdir = os.environ.get("WEIGHTS_DIR")
    paths = {
        "TRUFOR_ROOT": os.environ.get("TRUFOR_DIR"),
        "AASIST_ROOT": os.environ.get("AASIST_DIR"),
        "SYNCNET_ROOT": os.environ.get("SYNCNET_DIR"),
    }
    if wdir:
        paths.update({"TRUFOR_WEIGHTS": os.path.join(wdir, "trufor.pth.tar"),
                      "AASIST_WEIGHTS": os.path.join(wdir, "AASIST.pth"),
                      "SYNCNET_WEIGHTS": os.path.join(wdir, "syncnet_v2.model")})
    return {k: v for k, v in paths.items() if v}


def _deep_effective_paths() -> dict:
    """Read the wrapper defaults from source text (no torch import) and apply env overrides."""
    import re
    defaults = {}
    for mod, names in (("trufor_wrapper", ("TRUFOR_ROOT", "TRUFOR_WEIGHTS")),
                       ("aasist_wrapper", ("AASIST_ROOT", "AASIST_WEIGHTS")),
                       ("syncnet_wrapper", ("SYNCNET_ROOT", "SYNCNET_WEIGHTS"))):
        try:
            src = open(os.path.join(ROOT, "wrappers", mod + ".py"), encoding="utf-8").read()
        except OSError:
            continue
        for n in names:
            m = re.search(rf'^{n}\s*=\s*r?["\'](.+?)["\']', src, re.M)
            if m:
                defaults[n] = m.group(1)
    defaults.update(_deep_paths())
    return defaults


def deep_status():
    """(ready, reason). Cheap: only checks that packages and model paths exist."""
    for pkg in ("torch", "cv2", "numpy", "PIL"):
        if importlib.util.find_spec(pkg) is None:
            return False, f"Python package '{pkg}' is not installed."
    missing = [f"{k}={v}" for k, v in _deep_effective_paths().items() if not os.path.exists(v)]
    if missing:
        return False, ("Model files/folders not found: " + "; ".join(missing)
                       + ". Set TRUFOR_DIR, AASIST_DIR, SYNCNET_DIR and WEIGHTS_DIR.")
    return True, "ok"


_deep = {"orch": None}


def run_deep(path: str) -> dict:
    ok, why = deep_status()
    if not ok:
        raise EngineUnavailable(f"Deep engine unavailable: {why}")
    if _deep["orch"] is None:
        # wrappers/* keep their module-level constants; override them here (wrappers stay untouched)
        import wrappers.trufor_wrapper as tw
        import wrappers.aasist_wrapper as aw
        import wrappers.syncnet_wrapper as sw
        for mod in (tw, aw, sw):
            for k, v in _deep_paths().items():
                if hasattr(mod, k):
                    setattr(mod, k, v)
        try:
            from orchestrator import InferenceOrchestrator
            _deep["orch"] = InferenceOrchestrator()
        except Exception as e:  # noqa: BLE001 - surface any load failure as "unavailable"
            raise EngineUnavailable(f"Deep engine failed to load: {e}") from e
    result = _deep["orch"].analyze(path)
    mods = result.get("modalities", {})
    if all(mods.get(k) is None for k in ("visual_score", "audio_score", "sync_desync_score")):
        raise EngineUnavailable("Deep engine produced no usable result: " + "; ".join(result.get("evidence", [])))
    return result


# ---------------------------------------------------------------- fast (FakeProbe-X)
FAST_WEIGHTS = {"image": "best_coatnet_5ch.pth", "audio": "xgboost_asvspoof_model.joblib",
                "video": "xgboost_fusion_model.joblib"}
FAST_PACKAGES = ("torch", "timm", "cv2", "numpy", "joblib", "xgboost", "librosa", "transformers",
                 "facenet_pytorch", "scipy", "PIL")


def _fast_models_dir() -> str:
    return os.environ.get("FAKEPROBEX_MODELS_DIR", os.path.join(ROOT, "models"))


def fast_status():
    """(ready, reason). Cheap: checks packages and weight files, never imports torch."""
    for pkg in FAST_PACKAGES:
        if importlib.util.find_spec(pkg) is None:
            return False, f"Python package '{pkg}' is not installed (see requirements_fakeprobex.txt)."
    mdir = _fast_models_dir()
    missing = [f for f in FAST_WEIGHTS.values() if not os.path.isfile(os.path.join(mdir, f))]
    if missing:
        return False, f"FakeProbe-X weights missing in {mdir}: {', '.join(missing)}."
    return True, "ok"


_fast = {"cfg": False}
_fast_models = {}


def _fast_get(name):
    """Load one FakeProbe-X component on first use (the audio detector downloads Whisper, so an image
    check must not wait for it)."""
    if not _fast["cfg"]:
        from configs.forensic_config import DEFAULT_CONFIG
        mdir = _fast_models_dir()
        # PROJECT_ROOT / <absolute path> resolves to the absolute path, so this redirects the loaders
        DEFAULT_CONFIG.image_model_rel_path = os.path.join(mdir, FAST_WEIGHTS["image"])
        DEFAULT_CONFIG.audio_model_rel_path = os.path.join(mdir, FAST_WEIGHTS["audio"])
        DEFAULT_CONFIG.video_fusion_model_rel_path = os.path.join(mdir, FAST_WEIGHTS["video"])
        _fast["cfg"] = True
    if name not in _fast_models:
        try:
            import deepfake_detector_core as core
            cls = {"image": core.ImageDeepfakeDetector, "audio": core.AudioDeepfakeDetector,
                   "video": core.VideoDeepfakeDetector, "fusion": core.AdaptiveEvidenceFusionEngine}[name]
            _fast_models[name] = cls()
        except Exception as e:  # noqa: BLE001
            raise EngineUnavailable(f"Fast engine (FakeProbe-X) failed to load its {name} model: {e}") from e
    return _fast_models[name]


def run_fast(path: str, media_type: str, sample_name: str) -> dict:
    """Run FakeProbe-X and return UnifiedForensicReport.to_dict() plus an "_extras" block with the
    data to_dict() drops (face boxes, frame size). Errors propagate; nothing is swallowed."""
    ok, why = fast_status()
    if not ok:
        raise EngineUnavailable(f"Fast engine unavailable: {why}")
    d = {"fusion": _fast_get("fusion")}
    if media_type in ("image", "video"):
        d["image"] = _fast_get("image")
    if media_type in ("audio", "video"):
        d["audio"] = _fast_get("audio")
    if media_type == "video":
        d["video"] = _fast_get("video")
    if media_type == "image":
        import cv2
        res = d["image"].predict_structured(path, apply_face_crop=True)
        report = d["fusion"].fuse_image_evidence(res, sample_name=sample_name)
        img = cv2.imread(path)
        extras = {"frame_size": {"w": int(img.shape[1]), "h": int(img.shape[0])} if img is not None else None,
                  "face_box": _plain(res.evidence.get("face_box")),
                  "face_boxes_all": _plain(res.evidence.get("face_boxes_all")),
                  "face_confidence": _plain(res.evidence.get("face_confidence"))}
    elif media_type == "audio":
        res = d["audio"].predict_structured(path)
        report = d["fusion"].fuse_audio_evidence(res, sample_name=sample_name)
        extras = {}
    else:
        res = d["video"].predict_structured(path, d["image"], d["audio"], num_frames=5)
        report = d["fusion"].fuse_video_evidence(res, sample_name=sample_name)
        frames = res.evidence.get("sampled_frames_rgb") or []
        ft = res.evidence.get("face_tracking", {})
        extras = {"frame_size": {"w": int(frames[0].shape[1]), "h": int(frames[0].shape[0])} if frames else None,
                  "boxes_per_frame": _plain(ft.get("boxes_per_frame")),
                  "has_audio": bool(res.evidence.get("has_audio"))}
    out = report.to_dict()
    out["_extras"] = extras
    return out


def _plain(o):
    """numpy -> json-safe python."""
    if o is None:
        return None
    if isinstance(o, (list, tuple)):
        return [_plain(x) for x in o]
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (int, float, str, bool)):
        return o
    return float(o)


def health() -> dict:
    f_ok, f_why = fast_status()
    d_ok, d_why = deep_status()
    return {"default_engine": pick_engine() or "demo", "demo_ready": demo_ready(),
            "fast_ready": f_ok, "fast_reason": None if f_ok else f_why,
            "deep_ready": d_ok, "deep_reason": None if d_ok else d_why}


def pick_engine():
    """First ready live engine (fast, then deep), or None."""
    if fast_status()[0]:
        return "fast"
    if deep_status()[0]:
        return "deep"
    return None
