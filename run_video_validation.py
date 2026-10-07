import os
import cv2
import json
import time
import subprocess
from orchestrator import InferenceOrchestrator

def get_video_properties(path):
    props = {}
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        props["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        props["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        props["fps"] = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if props["fps"] > 0:
            props["duration_sec"] = frame_count / props["fps"]
    cap.release()
    
    ffmpeg_path = r"C:\Users\yaso0\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = "ffmpeg"
        
    cmd = [ffmpeg_path, "-i", path]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = res.stderr
    props["audio_stream_exists"] = "Audio:" in out
    
    return props

if __name__ == "__main__":
    vid_path = "tests/test_video.mp4"
    props = get_video_properties(vid_path)
    
    orc = InferenceOrchestrator()
    try:
        t0 = time.perf_counter()
        res = orc.analyze(vid_path)
        t1 = time.perf_counter()
        
        print(json.dumps({
            "status": "SUCCESS",
            "runtime_sec": t1 - t0,
            "media_properties": props,
            "payload": res
        }, indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"error": str(e)}))
