import cv2
import json
import time

def extract_frame():
    cap = cv2.VideoCapture("tests/test_video.mp4")
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite("tests/real_test_image.jpg", frame)
        return "tests/real_test_image.jpg"
    return None

if __name__ == "__main__":
    img_path = extract_frame()
    if not img_path:
        print(json.dumps({"error": "Failed to extract test image"}))
        exit(1)
        
    from orchestrator import InferenceOrchestrator
    
    orc = InferenceOrchestrator()
    try:
        t0 = time.perf_counter()
        res = orc.analyze(img_path)
        t1 = time.perf_counter()
        
        print(json.dumps({
            "status": "SUCCESS",
            "runtime_sec": t1 - t0,
            "payload": res
        }, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
