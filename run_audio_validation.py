import os
import time
import json
import subprocess

def extract_audio():
    out_path = "tests/real_test_audio.wav"
    if os.path.exists(out_path):
        os.remove(out_path)
    
    ffmpeg_path = r"C:\Users\yaso0\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
    cmd = [ffmpeg_path, "-y", "-i", "tests/test_video.mp4", "-q:a", "0", "-map", "a", out_path]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode == 0 and os.path.exists(out_path):
        return out_path
    return None

if __name__ == "__main__":
    audio_path = extract_audio()
    if not audio_path:
        print(json.dumps({"error": "Failed to extract audio track from video"}))
        exit(1)
        
    from orchestrator import InferenceOrchestrator
    
    orc = InferenceOrchestrator()
    try:
        t0 = time.perf_counter()
        res = orc.analyze(audio_path)
        t1 = time.perf_counter()
        
        print(json.dumps({
            "status": "SUCCESS",
            "runtime_sec": t1 - t0,
            "payload": res
        }, indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"error": str(e)}))
