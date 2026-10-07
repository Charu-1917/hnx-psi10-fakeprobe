import os
import uuid
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from orchestrator import InferenceOrchestrator

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the heavy ML pipeline once on startup
    app.state.orchestrator = InferenceOrchestrator()
    yield
    # Cleanup on shutdown if needed
    app.state.orchestrator = None

app = FastAPI(title="ORCA Forensics API", lifespan=lifespan)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/analyze")
async def analyze_media(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file supplied")
        
    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(os.getcwd(), f"temp_upload_{uuid.uuid4().hex}{ext}")
    
    try:
        # Save uploaded file to temp path
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run inference using global orchestrator
        result = app.state.orchestrator.analyze(temp_path)
        
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
                    
        return JSONResponse(content=result)
        
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
