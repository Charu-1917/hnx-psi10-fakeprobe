import os
import glob
import time
import json
import httpx

BASE_URL = "http://127.0.0.1:8000"
client = httpx.Client(base_url=BASE_URL, timeout=60.0)

def test_demos():
    print("\n--- TEST: Demo Endpoints ---", flush=True)
    demos = [
        ("video", "ai"),
        ("image", "ai"),
        ("image", "real"),
        ("audio", "ai"),
        ("audio", "real"),
    ]
    required_keys = ["verdict_label", "ai_probability_pct", "certainty", "reasons", "all_detected_areas"]
    for media_type, variant in demos:
        url = f"/api/v1/demo/{media_type}?variant={variant}"
        resp = client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            missing = [k for k in required_keys if k not in data]
            if not missing:
                print(f"PASS: GET {url} -> 200, all keys present. Verdict: {data.get('verdict_label')}, {data.get('ai_probability_pct')}%, certainty={data.get('certainty')}", flush=True)
            else:
                print(f"FAIL: GET {url} -> missing keys: {missing}", flush=True)
        else:
            print(f"FAIL: GET {url} -> status {resp.status_code}", flush=True)

def test_fast_engine():
    print("\n--- TEST: Fast Engine Analyze ---", flush=True)
    files = []
    files.extend(glob.glob("evaluation/test_images/*.png")[:3])
    files.extend(glob.glob("evaluation/test_audio/*.wav")[:2])
    files.extend(glob.glob("evaluation/test_videos/*.mp4")[:2])
    
    print(f"{'File':<25} | {'Verdict':<15} | {'%':<5} | {'Certainty':<10} | {'Seconds':<8}", flush=True)
    print("-" * 75, flush=True)
    for fpath in files:
        fname = os.path.basename(fpath)
        t0 = time.time()
        try:
            with open(fpath, "rb") as f:
                resp = client.post("/api/v1/analyze?engine=fast", files={"file": (fname, f.read())})
            elapsed = time.time() - t0
            if resp.status_code == 200:
                data = resp.json()
                verdict = str(data.get("verdict_label", "N/A"))
                pct = str(data.get("ai_probability_pct", "N/A"))
                cert = str(data.get("certainty", "N/A"))
                print(f"{fname:<25} | {verdict:<15} | {pct:<5} | {cert:<10} | {elapsed:.2f}s", flush=True)
            else:
                print(f"{fname:<25} | HTTP {resp.status_code} ({elapsed:.2f}s)", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"{fname:<25} | ERROR: {e} ({elapsed:.2f}s)", flush=True)

def test_deep_engine():
    print("\n--- TEST: Deep Engine Analyze (Expect 503) ---", flush=True)
    img = glob.glob("evaluation/test_images/*.png")[0]
    fname = os.path.basename(img)
    with open(img, "rb") as f:
        resp = client.post("/api/v1/analyze?engine=deep", files={"file": (fname, f.read())})
    if resp.status_code == 503:
        detail = resp.json().get("detail", {})
        if detail.get("run_demo_instead"):
            print(f"PASS: POST deep returned 503 with run_demo_instead=True.", flush=True)
        else:
            print(f"FAIL: 503 but detail missing run_demo_instead: {detail}", flush=True)
    else:
        print(f"FAIL: Expected 503, got {resp.status_code}", flush=True)

def test_txt_upload():
    print("\n--- TEST: .txt Upload (Expect 400) ---", flush=True)
    resp = client.post("/api/v1/analyze?engine=fast", files={"file": ("test.txt", b"Hello world")})
    if resp.status_code == 400:
        print(f"PASS: Uploading .txt returned 400: {resp.json().get('detail')}", flush=True)
    else:
        print(f"FAIL: Expected 400, got {resp.status_code}", flush=True)

def test_chat():
    print("\n--- TEST: Chat Endpoint ---", flush=True)
    questions = [
        "Summarise the verdict",
        "Why is this AI?",
        "Where is it fake?",
        "Do audio and lip-sync agree?",
        "How reliable is this?"
    ]
    demos = ["video", "image", "audio"]
    for m in demos:
        resp = client.get(f"/api/v1/demo/{m}?variant=ai")
        res_data = resp.json()
        print(f"\nTesting Chat with demo: {m}", flush=True)
        for q in questions:
            chat_resp = client.post("/api/v1/chat", json={"question": q, "result": res_data})
            if chat_resp.status_code == 200:
                answer = chat_resp.json().get("answer", "")
                if answer and len(answer.strip()) > 0:
                    print(f"  PASS: Q: '{q}' -> A ({len(answer)} chars): {answer[:65]}...", flush=True)
                else:
                    print(f"  FAIL: Empty answer for '{q}'", flush=True)
            else:
                print(f"  FAIL: HTTP {chat_resp.status_code} for '{q}'", flush=True)

if __name__ == "__main__":
    test_demos()
    test_fast_engine()
    test_deep_engine()
    test_txt_upload()
    test_chat()
