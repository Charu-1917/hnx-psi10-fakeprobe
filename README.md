# HNX26PSI10 — DeepTrace: Multimodal Deepfake & Digital Forensics

HackNex 2026 (Problem Statement PSI10). Upload a video, image or audio file → the system says **AI-GENERATED / POSSIBLY AI /
LOOKS REAL**, gives an AI-probability and a certainty level, explains **why** in plain words, shows **where** (boxes on the
frame, a timeline for video, time segments for audio) and lets you **ask questions** about the result.
The research behind the approach is in the paper library below.

## Start it

**Windows, one click:** double-click `run_demo.bat` (starts the API and the UI in two windows and opens the browser).

**One command (after a one-time UI build):**

```bash
pip install -r requirements_api.txt                 # once
cd frontend && npm install && npm run build && cd ..   # once
python -m uvicorn api:app --port 8000               # then open http://localhost:8000
```

**Developer mode (hot reload):**

```bash
python -m uvicorn api:app --port 8000               # API  -> http://127.0.0.1:8000/docs
cd frontend && npm install && npm run dev           # UI   -> http://localhost:5173
```

Press **Demo: Video / Real photo / AI photo / Real voice / AI voice**. This works fully offline with no model weights.

## 60-second demo script

| Time | Click | Say |
|---|---|---|
| 0:00 | Open the page; point at the 4 steps and the "demo ✔ · fast ✖ · deep ✖" tag | "Upload → Check → See where → Ask why. It starts with no models installed, in demo mode." |
| 0:10 | **Demo: Video** | "This is a cached real run of our TruFor visual analysis on a 10-second clip — the blue badge says so. It is not live." |
| 0:20 | Read the label, the 61 % and "certainty: medium", then the three reasons | "It says AI-generated at 61 % with medium certainty, and explains: 3 of 20 checked moments look edited, and it repeats over time. Voice and lip-sync were not measured in this run, and it tells you so." |
| 0:30 | Click a red dot on the timeline, then **Show ALL on one image**, then item **10** in the list | "Each dot is one checked moment. The boxes are where the model sees edits; the numbered list matches the boxes." |
| 0:40 | **Demo: Real photo**, then **Demo: AI photo** | "Same screen for an image: a clean photo reads LOOKS REAL; an AI one gets boxes. These two are labelled sample data — illustrations, not analyses." |
| 0:48 | **Demo: AI voice**, click a red block on the waveform | "For audio we show where the voice sounds synthetic. The segment times here are sample data — the real audio model gives one score per clip." |
| 0:54 | Click **How reliable is this?** | "The assistant answers only from the result — no invented numbers; it can use a local Ollama model to reword." |
| 0:58 | Open **Technical details** | "Experts get raw scores, timings and the full JSON. It is an estimate, not legal proof." |

## Engines

| Engine | What it is | Needs | If missing |
|---|---|---|---|
| `demo` (default) | Replays a cached result from `demo_results/*.json` | nothing | – |
| `fast` | FakeProbe-X (`deepfake_detector_core/`): CoAtNet image, Whisper+XGBoost audio, XGBoost video fusion | `pip install -r requirements_fakeprobex.txt` and the weights in `models/` (`best_coatnet_5ch.pth`, `xgboost_asvspoof_model.joblib`, `xgboost_fusion_model.joblib`) | HTTP 503 + "Run demo instead" |
| `deep` | TruFor + AASIST + SyncNet orchestrator (`orchestrator.py`), slow on CPU (1–2 min and more) | torch/opencv, the three model repos and their weights | HTTP 503 + "Run demo instead" |

Pick the engine in the UI, or call `POST /api/v1/analyze?engine=demo|fast|deep` (default `demo`). In demo mode an uploaded file is
**not analysed**: the UI says so and offers Fast/Deep when they are ready. No model is loaded at startup; heavy engines load lazily
on first use. `GET /api/v1/health` reports what is ready.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `VITE_API_URL` | API address used by the UI (build/dev time). Same origin when served on port 8000 | `http://127.0.0.1:8000` |
| `TRUFOR_DIR`, `AASIST_DIR`, `SYNCNET_DIR` | Model code folders for the deep engine | values hard-coded in `wrappers/*` |
| `WEIGHTS_DIR` | Folder with `trufor.pth.tar`, `AASIST.pth`, `syncnet_v2.model` | values hard-coded in `wrappers/*` |
| `FAKEPROBEX_MODELS_DIR` | Folder with the FakeProbe-X weights | `./models` |
| `OLLAMA_MODEL` | If set and Ollama runs on `localhost:11434`, the chat assistant rewords its answers with that model (numbers not in the evidence are rejected) | unset (rule-based answers) |

## Demo data — what is real and what is not

Every demo result carries `"demo": true`, and the UI badges it. **Never read a demo as a live analysis.**

* `demo_results/video.json` — a **cached real run** of the deep engine's visual analysis (TruFor, 20 moments of a local test clip,
  about 7 minutes on CPU). Badge: "Cached real run". Audio and lip-sync were *mocked* in that run, so they are **not shown or scored**.
  The frame size is estimated from the box positions, and the original frames are not stored, so boxes appear on a blank frame.
* `demo_results/image.json`, `image_real.json`, `audio.json`, `audio_real.json` — **synthetic samples** (`"sample_data": true`,
  badge "Sample result (cached demo)"): one clearly-AI and one clearly-real example per type. The audio segment times are invented
  for illustration: the real audio model (AASIST) gives one score per clip, not time segments.
* Rebuild them with `python scripts/build_demo_results.py`. The UI's offline fallback copy (`frontend/src/demo/`) is generated from
  `demo_results/` by `npm run sync-demo` (runs automatically before `dev` and `build`) — never edit it by hand.

## API

| Route | Purpose |
|---|---|
| `GET /api/v1/health` | `demo_ready`, `fast_ready`, `deep_ready` (+ reasons) |
| `GET /api/v1/demo/{video\|image\|audio}?variant=ai\|real` | cached result (`real` exists for image and audio) |
| `POST /api/v1/analyze?engine=…` | upload a file (`file` form field) |
| `POST /api/v1/chat` | `{question, result}` → `{answer, tools_used, llm_used}`; answers come only from the posted result |

The result keeps all original engine keys and adds `verdict_label`, `ai_probability_pct`, `certainty`, `reasons`,
`timeline`, `frame_regions`, `audio_segments`, `all_detected_areas`, `frame_size` (see `presenter.py`).

## Measured on real media (live `fast` engine, this machine, CPU)

A first honest measurement with a handful of real and AI files. **It shows the `fast` engine does not separate real from AI:
nearly everything scores about 60 %.** Treat its verdicts as unreliable until it is properly evaluated.

| File | What it really is | Verdict | AI % | Seconds |
|---|---|---|---|---|
| obama.jpg | real photo | AI_GENERATED | 61 | 74.7 (first call loads the models) |
| lena.jpg | real photo | AI_GENERATED | 60 | 1.8 |
| real1.jpg | real portrait (128 px) | AI_GENERATED | 62 | 0.4 |
| ai_1.png | StyleGAN face | POSSIBLY_AI | 56 | 0.6 |
| ai_3.png | StyleGAN face | AI_GENERATED | 63 | 0.6 |
| ai_4.png | StyleGAN face | AI_GENERATED | 61 | 0.6 |
| real_libri1.wav | real speech | AI_GENERATED | 74 | 1.2 |
| tts_sapi16.wav | Windows TTS voice | AI_GENERATED | 66 | 7.8 |
| clip.mp4 | real video | AI_GENERATED | 59 (low certainty) | 14.2 |

Two public Hugging Face image detectors were also tried offline on the same images: `prithivMLmods/Deep-Fake-Detector-v2-Model`
called every real photo 64–92 % "Deepfake", and `dima806/deepfake_vs_real_image_detection` called every real photo real but
missed 4 of 5 StyleGAN faces. A "lite" engine built on them is **not implemented**.

**Uploads never get a demo result:** with `engine=demo` the API answers 400 ("Demo mode shows a cached example, not your file");
`engine=auto` (the default) uses the first ready live engine (`fast`, then `deep`) and otherwise returns 503.

## Known limits

* **FakeProbe-X accuracy on real media is unmeasured.** The only benchmark in the repo (`evaluation/benchmark_results.json`)
  shows 50% on 10 images, 10 audio clips and 6 videos, and it was run on **synthetic fixtures** (drawn faces, generated tones —
  see `evaluation/prepare_test_harness.py`), not on real deepfakes. No real-world accuracy claim can be made from it.
* The live `fast` and `deep` engines were **not run in this build** (no weights/packages on the dev machine). Their adapter code
  is covered only by hand-written fixtures in `tests/test_presenter.py`.
* Image boxes from `fast` are the **face area only** (approximate); that engine does not localise fake pixels.
  Video analysis in `fast` looks at **5 moments**, not every frame. Audio gives **one score per clip, no time segments**.
* The deep engine's lip-sync and voice scores come from placeholder-calibrated heuristics (`fusion/sync_calibrator.py`); the
  fusion weights and thresholds are not validated.
* Verdict labels use fixed score cut-offs (above 50 % = AI-GENERATED, 35–50 % = POSSIBLY AI, below 35 % = LOOKS REAL); a score just
  above 50 % is labelled AI-GENERATED with *low* certainty — read the certainty.
* The image/audio/video demos use sample or cached data; the demo UI never draws boxes on a file you uploaded in demo mode.
* Estimates from pretrained models — not legal proof.

## Tests

```bash
PYTHONPATH=. python tests/test_presenter.py
PYTHONPATH=. python tests/test_chat_agent.py
cd frontend && npm run build
```

## Paper library

Browse the full annotated bibliography in **[`docs/papers/README.md`](docs/papers/README.md)**.

```
docs/papers/
├── README.md                                  ← annotated index of all papers
├── 01-visual-generation-and-forensics/
├── 02-audio-voice-cloning/
├── 03-multimodal-fusion-explainability/
└── 04-detection-surveys/
```

| Group | Papers | Covers |
|---|---|---|
| [Visual generation & forensics](docs/papers/01-visual-generation-and-forensics/) | 10 | Face-swap, synthetic-face, and diffusion generation techniques and their forensic traces |
| [Audio & voice cloning](docs/papers/02-audio-voice-cloning/) | 14 | Voice cloning, neural vocoders, lip-sync and dubbing generation |
| [Multimodal fusion & explainability](docs/papers/03-multimodal-fusion-explainability/) | 9 | Fusing modalities, localizing manipulations, generating explanations |
| [Detection surveys](docs/papers/04-detection-surveys/) | 8 | Field taxonomy, standard datasets, documented open challenges |

## Repository layout

`api.py` API · `engines.py` engine registry · `presenter.py` plain-language adapter · `chat_agent.py` chat tools ·
`demo_results/` cached results · `frontend/` React UI · `deepfake_detector_core/` FakeProbe-X · `wrappers/`, `fusion/`,
`orchestrator.py` deep engine · `docs/papers/` research library · `docs/reference_ui.html` UI reference · `run_demo.bat` one-click start.

## Team

- [Charu-1917](https://github.com/Charu-1917)
- [ranjith-saravanan](https://github.com/ranjith-saravanan)
