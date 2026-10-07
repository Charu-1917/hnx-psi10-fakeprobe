import time
from orchestrator import InferenceOrchestrator

def main():
    video_path = r"C:\charu hack\tests\test_video.mp4"
    orchestrator = InferenceOrchestrator()
    
    t0 = time.time()
    res = orchestrator.analyze(video_path)
    t1 = time.time()
    
    vv = res.get("video_visual", {})
    
    frames_analyzed = vv.get("frames_analyzed", 0)
    coarse_frames = len([f for f in vv.get("evidence_frames", []) if f["timestamp_sec"] == int(f["timestamp_sec"])]) # rough estimate based on integer timestamps
    fine_frames = frames_analyzed - coarse_frames
    
    print(f"duration: unknown")
    print(f"coarse frames: {coarse_frames}")
    print(f"fine frames: {fine_frames}")
    print(f"total frames: {frames_analyzed}")
    print(f"peak timestamp: {vv.get('peak_timestamp_sec')}")
    print(f"peak score: {vv.get('peak_score')}")
    print(f"suspicious intervals: {len(vv.get('suspicious_timeline', []))}")
    print(f"runtime: {t1 - t0:.2f}s")
    
if __name__ == '__main__':
    main()
