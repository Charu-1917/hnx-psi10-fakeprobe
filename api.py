import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

import engines
from engines import EngineUnavailable
from presenter import present

# No model is loaded at startup: the API always starts, and demo mode works with no weights.
app = FastAPI(title="ORCA Forensics API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO_HINT = "Use engine=demo (or GET /api/v1/demo/{video|image|audio}) to run the cached sample result instead."


def _unavailable(e: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail={"message": str(e), "run_demo_instead": True, "hint": DEMO_HINT})


def _media_size(path: str, media_type: str):
    """Real width/height of an uploaded image/video (used to place boxes). None if unreadable."""
    if media_type == "audio":
        return None
    try:
        import cv2
        if media_type == "image":
            img = cv2.imread(path)
            return {"w": int(img.shape[1]), "h": int(img.shape[0])} if img is not None else None
        cap = cv2.VideoCapture(path)
        size = {"w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        cap.release()
        return size if size["w"] and size["h"] else None
    except Exception:
        return None


@app.get("/api/v1/health")
def health():
    return engines.health()


@app.get("/api/v1/demo/{media_type}")
def demo(media_type: str):
    if media_type not in engines.MEDIA_TYPES:
        raise HTTPException(status_code=404, detail=f"media_type must be one of {list(engines.MEDIA_TYPES)}")
    try:
        return JSONResponse(content=engines.load_demo(media_type))
    except EngineUnavailable as e:
        raise _unavailable(e)


@app.post("/api/v1/analyze")
def analyze_media(file: UploadFile = File(...), engine: str = Query("demo")):
    """engine = demo (default, cached SAMPLE result) | fast (FakeProbe-X) | deep (TruFor/AASIST/SyncNet)."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file supplied")
    if engine not in ("demo", "fast", "deep"):
        raise HTTPException(status_code=400, detail="engine must be demo, fast or deep")

    try:
        media_type = engines.media_type_from_name(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if engine == "demo":
        # Replays the cached sample for this media type. The upload is NOT analysed.
        try:
            result = engines.load_demo(media_type)
        except EngineUnavailable as e:
            raise _unavailable(e)
        result["uploaded_filename"] = file.filename
        return JSONResponse(content=result)

    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(os.getcwd(), f"temp_upload_{uuid.uuid4().hex}{ext}")

    try:
        # Save uploaded file to temp path
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if engine == "fast":
            result = engines.run_fast(temp_path, media_type, sample_name=file.filename)
            return JSONResponse(content=present(result, media_type, "fast"))

        # engine == "deep": existing orchestrator, created lazily on first use
        result = engines.run_deep(temp_path)
        frame_size = _media_size(temp_path, media_type)

        # Rewrite absolute artifact paths to API URLs
        vis_ev = result.get("visual_evidence", {})
        if vis_ev:
            for key in ["anomaly_map_path", "reliability_map_path"]:
                abs_path = vis_ev.get(key)
                if abs_path and isinstance(abs_path, str):
                    rel = os.path.relpath(abs_path, os.getcwd())
                    # Convert OS paths to URI paths
                    parts = rel.replace("\\", "/").split("/")
                    if len(parts) >= 2 and parts[0].startswith("artifacts_"):
                        uid = parts[0][len("artifacts_"):]
                        filename = parts[1]
                        vis_ev[key] = f"/api/v1/artifacts/{uid}/{filename}"

        return JSONResponse(content=present(result, media_type, "deep", frame_size=frame_size))

    except EngineUnavailable as e:
        raise _unavailable(e)
    except HTTPException:
        raise
    except Exception as e:
        # Return generic error wrapper
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/v1/artifacts/{uid}/{filename}")
async def get_artifact(uid: str, filename: str):
    # Prevent path traversal in parameters
    if ".." in uid or "/" in uid or "\\" in uid:
        raise HTTPException(status_code=400, detail="Invalid artifact identifier")
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    target_dir = os.path.abspath(os.path.join(os.getcwd(), f"artifacts_{uid}"))
    target_file = os.path.abspath(os.path.join(target_dir, filename))

    # Verify the target file resolves strictly inside the intended artifact directory
    if not target_file.startswith(target_dir):
        raise HTTPException(status_code=403, detail="Path traversal detected")

    if not os.path.exists(target_file) or not os.path.isfile(target_file):
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(target_file)
