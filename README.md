# HNX26PSI10 — DeepTrace: Multimodal Deepfake & Digital Forensics

Upload a video, image or audio file → the system says **AI / possibly AI / real**, gives an AI-probability and a
certainty level, explains **why** in plain words, shows **where** (boxes on the frame, time segments for audio),
and lets you **ask questions** about the result. Research papers behind the approach are in `docs/papers/`.

## Run it (3 commands)

```bash
pip install -r requirements_api.txt            # 1. API dependencies (no ML packages needed for demo mode)
uvicorn api:app --port 8000                    # 2. API  ->  http://127.0.0.1:8000/docs
cd frontend && npm install && npm run dev      # 3. UI   ->  http://localhost:5173
```

Open the UI and press **Demo: Video / Image / Audio**. This works fully offline with no model weights.

## Engines

| Engine | What it is | Needs | If missing |
|---|---|---|---|
| `demo` (default) | Replays a cached result from `demo_results/*.json` | nothing | – |
| `fast` | FakeProbe-X (`deepfake_detector_core/`): CoAtNet image, Whisper+XGBoost audio, XGBoost video fusion | `pip install -r requirements_fakeprobex.txt` and the weights in `models/` (`best_coatnet_5ch.pth`, `xgboost_asvspoof_model.joblib`, `xgboost_fusion_model.joblib`) | HTTP 503 + "Run demo instead" |
| `deep` | TruFor + AASIST + SyncNet orchestrator (`orchestrator.py`), slow on CPU | torch/opencv, the three model repos and their weights | HTTP 503 + "Run demo instead" |

Pick the engine in the UI, or call `POST /api/v1/analyze?engine=demo|fast|deep` (default `demo`).
No model is loaded at startup; heavy engines load lazily on first use. `GET /api/v1/health` reports what is ready.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `VITE_API_URL` | API address used by the UI (build/dev time) | `http://127.0.0.1:8000` |
| `TRUFOR_DIR`, `AASIST_DIR`, `SYNCNET_DIR` | Model code folders for the deep engine | values hard-coded in `wrappers/*` |
| `WEIGHTS_DIR` | Folder with `trufor.pth.tar`, `AASIST.pth`, `syncnet_v2.model` | values hard-coded in `wrappers/*` |
| `FAKEPROBEX_MODELS_DIR` | Folder with the FakeProbe-X weights | `./models` |
| `OLLAMA_MODEL` | If set and Ollama runs on `localhost:11434`, the chat assistant rewords its answers with that model | unset (rule-based answers) |

## Demo mode — what is real and what is not

Every demo result carries `"demo": true` and the UI shows a **"Sample result (cached demo)"** badge. Never read a demo as a live analysis.

* `demo_results/video.json` — a **cached real run** of the deep engine's visual analysis (TruFor, 20 moments of a local test clip,
  about 7 minutes on CPU). Audio and lip-sync were *mocked* in that run, so they are **not shown or scored**.
  The frame size is estimated from the box positions.
* `demo_results/image.json`, `audio.json` — **synthetic samples** (`"sample_data": true`). The audio segment times are invented
  for illustration: the real audio model (AASIST) gives one score per clip, not time segments.
* Rebuild them with `python scripts/build_demo_results.py`.

## API

| Route | Purpose |
|---|---|
| `GET /api/v1/health` | `demo_ready`, `fast_ready`, `deep_ready` (+ reasons) |
| `GET /api/v1/demo/{video\|image\|audio}` | cached result |
| `POST /api/v1/analyze?engine=…` | upload a file (`file` form field) |
| `POST /api/v1/chat` | `{question, result}` → `{answer, tools_used, llm_used}`; answers come only from the posted result |

The result keeps all original engine keys and adds `verdict_label`, `ai_probability_pct`, `certainty`, `reasons`,
`timeline`, `frame_regions`, `audio_segments`, `all_detected_areas`, `frame_size` (see `presenter.py`).

## Known limits (please read)

* **FakeProbe-X accuracy on real media is unmeasured.** The only benchmark in the repo (`evaluation/benchmark_results.json`)
  shows 50% on 10 images, 10 audio clips and 6 videos, and it was run on **synthetic fixtures** (drawn faces, generated tones —
  see `evaluation/prepare_test_harness.py`), not on real deepfakes. No real-world accuracy claim can be made from it.
* The live `fast` and `deep` engines were **not run in this build** (no weights/packages on the dev machine). Their adapter code
  is covered only by hand-written fixtures in `tests/test_presenter.py`.
* Image boxes from `fast` are the **face area only** (approximate); the engine does not localise fake pixels.
  Video analysis in `fast` looks at **5 moments**, not every frame. Audio gives **one score per clip, no time segments**.
* Scores are estimates from pretrained models, not legal proof.

## Tests

```bash
PYTHONPATH=. python tests/test_presenter.py
PYTHONPATH=. python tests/test_chat_agent.py
cd frontend && npm run build
```

## Repository layout

`api.py` API · `engines.py` engine registry · `presenter.py` plain-language adapter · `chat_agent.py` chat tools ·
`demo_results/` cached results · `frontend/` React UI · `deepfake_detector_core/` FakeProbe-X · `wrappers/`, `fusion/`,
`orchestrator.py` deep engine · `docs/papers/` research library · `docs/reference_ui.html` UI reference.
