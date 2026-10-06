# Literature Review — Paper Index

42 papers underpinning [`../architecture/ARCHITECTURE.md`](../architecture/ARCHITECTURE.md), grouped by the role each plays in the system design. Problem statement: [`HackNex2026-Problem-Statement.pdf`](../HackNex2026-Problem-Statement.pdf).

---

## [01 — Visual Generation & Forensics](01-visual-generation-and-forensics/) (10 papers)

How image/video fakes are generated (face swap, synthetic faces, diffusion synthesis) — understanding generation tells us what forensic artifacts to look for.

| Paper | Takeaway |
|---|---|
| `2601.16429v1.pdf` (AlphaFace) | Real-time face-swap GAN; pose-conditioned boundary artifacts remain detectable at extreme yaw (>60°) |
| `2605.12075v1.pdf` ("The Deepfakes We Missed") | Position paper: 71% of detection research targets public-figure face-swap video while voice-clone/NCII cases are under-studied |
| `Digital_Airworthiness_Protocol.pdf` | Provenance/watermarking (C2PA-style) policy proposal — detection positioned as a complementary safety net, not a replacement |
| `Face_Swap_Detection_A_Systematic_Literature_Review.pdf` | Taxonomy: spatial (78%) vs temporal (0.6%) vs spatiotemporal (21%) detection methods; FF++/Celeb-DF/DFDC usage ranking |
| `paper18.pdf` | Speaker-verification robustness study: raw-waveform (RawNet2) generalizes better than spectrogram/ViT to unseen voice clones |
| `Photorealistic_Synthetic_Human_Face_Generation...pdf` | cDCGAN vs StyleGAN text-guided face synthesis; StyleGAN artifacts concentrate in frequency-domain "fingerprints" |
| `Real-Time_High-Fidelity_Face_Identity_Swapping...pdf` | 256×256 real-time face swap — higher resolution than prior detectors were trained to catch |
| `s00371-025-04232-w.pdf` | Talking-head generation review (2D/3D/NeRF/diffusion); SyncNet-accuracy as the field's standard sync metric |
| `Semantic_Image_Synthesis_via_Diffusion_Models.pdf` | Diffusion image synthesis (SDM); diffusion artifacts differ statistically from GAN artifacts — detectors can't assume one fingerprint covers both |
| `Subject-Specific_High-Fidelity_Identity-Aware_Face_Swapping_Model.pdf` | Mask-boundary "haloing" and skin-texture residuals are named, generator-agnostic failure modes |

## [02 — Audio & Voice Cloning](02-audio-voice-cloning/) (14 papers)

Voice cloning, neural vocoders, and lip-sync/dubbing generation — tells us what audio artifacts and A/V sync signals to detect.

| Paper | Takeaway |
|---|---|
| `1-s2.0-S2090447926002649-main.pdf` | Unified multimodal explainable detector (CNN+BiLSTM+XGBoost per modality + cross-attention fusion + SHAP/LIME/Grad-CAM); 96–99% accuracy — closest existing blueprint to our system |
| `2512.05126v2.pdf` (SyncVoice) | Flow-matching video dubbing; synthetic output's LSE-C (8.36) **beat** its own ground truth (7.33) — hard proof sync scores alone are gameable |
| `2606.19747v1.pdf` | Not relevant — Quranic Arabic ASR fine-tuning |
| `2609.12918v1.pdf` (PhaseGAN) | Lightweight GAN vocoder; phase spectra are purely synthetic with no ground-truth supervision — phase-coherence is a vocoder fingerprint |
| `2609.15650v1.pdf` | LoRA-adapted graph-attention anti-spoofing (XLS-R + AASIST2); evaluated explicitly for cross-dataset/unseen-attack generalization |
| `3742413.3789074.pdf` | HCI user study: voice-cloned translation scores *higher* than generic TTS on naturalness/trust — clones are convincing to average users |
| `3774905.3794684.pdf` | Talking-head generation taxonomy (117 papers); tracks field's shift to LSE-C/D and Sync-C/D as standard eval metrics |
| `3785656.pdf` | Survey of talking-head synthesis; flags compounding artifacts across chained generation stages |
| `Engineering Reports - 2025 - Shaaban...pdf` | Siamese CNN + triplet loss on MFCCs; verification-style distance metric generalizes better than closed-set classification; EER 2.95% |
| `Intelligent_Audio-Video_Dubbing...Wav2Lip.pdf` | Full AVT dubbing pipeline; LSE-D/LSE-C as the standard A/V sync metric (via pretrained SyncNet) |
| `Revisiting_Aliasing_in_GAN_Vocoders...pdf` | GAN vocoder upsampling leaves harmonic/aliasing artifacts in spectrograms, worse on out-of-domain speech — concrete, model-agnostic cue |
| `Voice_Cloning_A_Survey_of_Zero-Shot...pdf` | Codec-LM voice cloning (VALL-E/CosyVoice-class) leaves RVQ-token artifacts — a *different* family from GAN-vocoder artifacts |
| `Voice_Cloning_using_RVC...pdf` | F0/pitch contour and MFCC are the identity-carrying channels being swapped; MFCC similarity 0.99 shows how easily identity transfer succeeds |
| `Wav2Lip_Bridges_Communication_Gap...pdf` | Multilingual dubbing (Hindi/Tamil/Telugu/English); Wav2Lip degrades under noisy audio and large head-pose — a probe point for detectors |

## [03 — Multimodal Fusion & Explainability](03-multimodal-fusion-explainability/) (9 papers)

Core architecture group: how to fuse modalities, localize manipulations, and generate human-readable explanations.

| Paper | Takeaway |
|---|---|
| `Interpretable_Multimodal_Deep_Fake_Detection...pdf` | Cross-attention fusion + per-modality confidence + LLM explanation; fusion lifts accuracy from ~70% unimodal cap to 85%+ |
| `2308.14970v1.pdf` | Audio deepfake detection survey; SSL embeddings (XLS-R/HuBERT) generalize far better OOD than handcrafted spectral features |
| `3801962.pdf` | Deepfake generation/detection benchmark survey; documents cross-dataset AUC collapse as the field's central unsolved problem |
| `fdata-05-1001063.pdf` | Audio deepfake survey; DeepSonar's neuron-activation-pattern idea as a lightweight explainability signal |
| `Li_Omni-Fake...CVPR_2026_paper.pdf` | **Closest system to our target**: unified 4-modality MLLM outputting {label, bbox/interval localization, NL explanation}; explicit disjoint-generator OOD benchmark |
| `s10791-026-10077-1.pdf` | Multimodal AV survey; AVTENet/AVA-CL/Multimodaltrace audio-visual transformer ensembles beat single-modality detectors by 10–40 pts OOD |
| `s11760-025-03970-7.pdf` | Diffusion-model denoising as a preprocessing step improves robustness to compression/noise by 2–5 pts |
| `s44163-025-00337-2.pdf` | GAN-generation taxonomy; GAN-fingerprint detectors are brittle — adversarial fingerprint removal cuts accuracy up to 50% |
| `SIDA_Social_Media_Image_Deepfake_Detection...pdf` | **Single-image ancestor of Omni-Fake**: `<DET>`/`<SEG>` token cross-attention for joint detection→pixel-mask localization→explanation |

## [04 — Detection Surveys](04-detection-surveys/) (8 papers)

Broad review papers — field taxonomy, standard datasets, and documented open challenges.

| Paper | Takeaway |
|---|---|
| `1-s2.0-S0893608025010809-main.pdf` | Not a detection survey — latent-diffusion GAN for facial generation (background reference only) |
| `Deepfake_Generation_and_Detection_Case_Study...pdf` | Covers image/video/audio + multimodal case study; flags generalizability, robustness, and lack-of-interpretability as top challenges |
| `s00371-024-03791-8.pdf` | Multi-level DWT + ViT detector (single-method, used for benchmark numbers) |
| `s10462-024-10810-6.pdf` | Video-specific SLR; dataset generations (1st/2nd/3rd-gen) classification scheme |
| `s11432-024-4400-8.pdf` | CLIP-based foundation-model detector (LEDNet) for generalized cross-generator detection |
| `Understanding_Audiovisual_Deepfake_Detection...pdf` | AV-specific survey; formal **4-way taxonomy (FVFA/RVFA/FVRA/RVRA)** adopted directly in our output scheme; phoneme-viseme mismatch (M/B/P lip closure) as a concrete feature; humans score 64–66% vs AI's 75–97% on AV detection |
| `Visual_Deepfake_Detection_Review...pdf` | Spatial/temporal/frequency/spatiotemporal taxonomy; JPEG/resize masks GAN artifacts — direct hit on compression-robustness criterion |
| `WIREs...Heidari...pdf` | Concrete generalization-collapse evidence: ~100% AUC (UADFV/FF++) → <60% AUC (Celeb-DF) with the *same* detector |

---

### Dataset catalogue referenced across these papers

**Image/video**: FaceForensics++, Celeb-DF(v2), DFDC, DeeperForensics-1.0, WildDeepfake, ForgeryNet, KoDF, DF-TIMIT, OpenForensics, DeepFakeFace/DFF
**Audio**: ASVspoof 2019/2021, WaveFake, In-the-Wild, FoR, ADD2022/2023, CFAD
**Multimodal/AV**: FakeAVCeleb, AV-Deepfake1M, LAV-DF, TVIL, PolyGlotFake

See [`../architecture/ARCHITECTURE.md § 5`](../architecture/ARCHITECTURE.md#5-datasets) for the recommended train/OOD split.
