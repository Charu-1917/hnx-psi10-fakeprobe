# Dataset Documentation (DATASETS.md)

This document details all upstream training, validation, and evaluation benchmark datasets utilized or referenced across the FakeProbe-X forensics system.

---

## 1. Primary Benchmark Datasets

### A. Artifact Dataset (`awsaf49/artifact-dataset`)

- **Name**: Artifact Deepfake Image Dataset
- **Source**: Kaggle (`awsaf49/artifact-dataset`)
- **Modality**: Static RGB Images (Human Face Crops)
- **Real Samples**: 65,000 (from CelebA, FFHQ, ImageNet)
- **Fake Samples**: 65,000 (from ProGAN, StyleGAN, StyleGAN2, StarGAN, BigGAN, DeepFake, FaceSwap)
- **Manipulation Types**: Whole-image generative face synthesis, facial attribute manipulation, expression swapping, identity replacement.
- **Train / Val / Test Split**:
  - `TRAIN`: 100,000 images (50,000 Real / 50,000 Fake)
  - `VAL`: 10,000 images (5,000 Real / 5,000 Fake)
  - `TEST`: 20,000 images (10,000 Real / 10,000 Fake)
- **Leakage Prevention**: Stratified splitting by generator family and source subject identity across splits.
- **Preprocessing**: RGB resize to 224x224, 5-channel tensor creation ([R, G, B, ELA, FFT_magnitude]), ImageNet normalization on channels 0-2, 0.5 mean/std on channels 3-4.
- **Purpose**: Training and evaluation of CoAtNet-0 5-channel image deepfake detector (`best_coatnet_5ch.pth`).
- **License / Constraints**: Research and academic use.

---

### B. ASVspoof 2019 Logical Access (`ASVspoof2019_LA`)

- **Name**: Automatic Speaker Verification Spoofing and Countermeasures Challenge 2019 (Logical Access Partition)
- **Source**: ASVspoof Consortium / University of Edinburgh
- **Modality**: Single-channel 16 kHz Audio Waveforms (FLAC/WAV)
- **Real Samples**: 2,580 (Train) / 2,548 (Dev) / 7,355 (Eval) Bonafide human speech utterances
- **Fake Samples**: 22,800 (Train) / 22,296 (Dev) / 63,882 (Eval) Synthesized speech (A01–A19 spoofing algorithms)
- **Manipulation Types**: Text-to-Speech (TTS) synthesis (Tacotron, FastSpeech), Voice Conversion (VC), neural vocoders (WaveNet, WaveRNN, Griffin-Lim).
- **Train / Val / Test Split**:
  - `TRAIN`: 25,380 utterances (Protocol `ASVspoof2019.LA.cm.train.trn.txt`)
  - `DEV`: 24,844 utterances (Protocol `ASVspoof2019.LA.cm.dev.trl.txt`)
  - `EVAL`: 71,237 utterances (Protocol `ASVspoof2019.LA.cm.eval.trl.txt`)
- **Leakage Prevention**: Zero speaker overlap across Train, Dev, and Eval partitions; unseen attack algorithms (A07–A19) reserved exclusively for Eval.
- **Preprocessing**: Resample to 16,000 Hz, Whisper-Base log-mel spectrogram extraction `(1, 80, 3000)`, temporal mean pooling over encoder hidden states yielding 512-dim embedding.
- **Purpose**: Training and evaluation of Whisper + XGBoost audio spoof detector (`xgboost_asvspoof_model.joblib`).
- **License / Constraints**: Open academic research dataset.

---

### C. FakeAVCeleb Dataset

- **Name**: FakeAVCeleb Multimodal Deepfake Dataset
- **Source**: Korea University / Deepfake Research
- **Modality**: Audiovisual MP4 Video Sequences
- **Real Samples**: 500 celebrities (VoxCeleb2 source footage)
- **Fake Samples**: ~20,000 generated video clips
- **Manipulation Types**:
  - Visual-only: Faceswap (DeepFaceLab, FSGAN)
  - Audio-only: Voice cloning (SV2TTS, Real-Time-Voice-Cloning)
  - Audio-Visual: Synchronized lip-sync synthesis (Wav2Lip) with cloned speech
- **Train / Val / Test Split**:
  - `TRAIN`: 80% identity-disjoint partition
  - `TEST`: 20% balanced test partition (2,000 clips)
- **Leakage Prevention**: Disjoint celebrity identities across train and test partitions.
- **Preprocessing**: 5 uniform frames extracted per video $\to$ CoAtNet 768-dim embeddings with statistical pooling (mean, max, std) = 2304-dim visual vector; audio demuxed at 16 kHz $\to$ Whisper 512-dim acoustic vector; concatenated to 2816-dim feature vector.
- **Purpose**: Training and evaluation of intermediate fusion XGBoost video detector (`xgboost_fusion_model.joblib`).
- **License / Constraints**: Academic research only.

---

## 2. In-Distribution vs Out-of-Distribution Generalization

| Test Domain | Modality | Source / Characteristics | Generalization Behavior |
|:---|:---:|:---|:---|
| **In-Distribution (Face Crops)** | Image | High-resolution cropped face portraits | Strong confidence, high true positive / true negative accuracy (>94%). |
| **Out-of-Distribution (Full Scene)** | Image | Scenic landscapes, objects, complex backgrounds | Managed via FakeProbe-X Face Cropper and Input Quality Analyzer. |
| **Out-of-Distribution (Mute Video)** | Video | Video without audio stream | Handled by Adaptive Evidence Fusion via dynamic visual-only re-weighting without zero-padding bias. |
| **Out-of-Distribution (Degraded Audio)** | Audio | Heavy noise, clipped signals, silence | Handled by Three-Way Decision Engine with `UNCERTAIN` verdict. |
