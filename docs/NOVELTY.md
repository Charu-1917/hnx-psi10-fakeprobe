# FakeProbe-X System Novelty (NOVELTY.md)

## Core Project-Level Novelty: Reliability-Aware Multimodal Forensic Evidence Fusion

Standard deepfake detectors typically apply rigid binary classification thresholds ($P > 0.50 \implies \text{FAKE}$) to neural network outputs. In real-world forensic applications, this naive approach fails catastrophically when inputs are degraded by blur, out-of-distribution compression, non-face geometries, or missing modalities (e.g. video without audio), causing frequent false positive alarms on authentic photographs and unedited footage.

**FakeProbe-X** introduces a **Reliability-Aware Multimodal Evidence Fusion Architecture** that evaluates data quality, physical frequency spectra, temporal dynamics, cross-modal consensus, and epistemic uncertainty before rendering a three-way verdict (**REAL**, **FAKE**, or **UNCERTAIN**).

---

## 1. Breakdown of Novel Capabilities

### 1. Reliability-Aware Dynamic Evidence Weighting
- **Existing Capability**: Fixed static weights or arbitrary neural concatenation across available features regardless of signal quality.
- **FakeProbe-X Capability**: Computes dynamic reliability $r_i \in [0, 1]$ for every evidence stream as a composite function of input resolution, sharpness, facial bounding-box stability, and detector confidence margin. Fused fake probability is calculated via:
  $$\text{fused\_score} = \frac{\sum_i w_i \cdot r_i \cdot s_i}{\sum_i w_i \cdot r_i}$$
- **Why It Helps**: Prevents degraded, low-quality, or uncorroborated detector responses from dominating the final forensic decision.

### 2. Three-Way Decision Engine (`REAL` / `FAKE` / `UNCERTAIN`)
- **Existing Capability**: Binary forced choice ($> 0.50$ is FAKE, $\le 0.50$ is REAL) resulting in overconfident false alarms on ambiguous inputs.
- **FakeProbe-X Capability**: Establishes an empirical uncertainty deadband ($[0.42, 0.58]$) coupled with epistemic uncertainty quantification ($u > 0.60 \implies \text{UNCERTAIN}$).
- **Why It Helps**: Eliminates forced misclassifications on out-of-distribution patterns, providing forensic examiners with an honest measure of epistemic confidence.

### 3. Input Quality & Spatial Face Awareness
- **Existing Capability**: Direct warping of full-scene images to 224x224, squashing aspect ratios and introducing background texture noise into face-specialized networks.
- **FakeProbe-X Capability**: Integrated `InputQualityAnalyzer` and `FaceDetector` with margin-padded facial boundary preservation ($25\%$ padding to preserve blend lines) and Laplacian sharpness variance scoring.
- **Why It Helps**: Directly resolves the **REAL $\to$ FAKE** false positive problem caused by non-facial scene artifacts.

### 4. Frequency-Domain Forensic Spectrum
- **Existing Capability**: Purely spatial convolutional / transformer activations.
- **FakeProbe-X Capability**: Orthogonal 2D Fast Fourier Transform log-magnitude spectra, azimuthally averaged radial power distribution, and high-frequency energy ratio calculations.
- **Why It Helps**: Detects periodic generator upsampling artifacts and spectral roll-off deviations characteristic of diffusion and GAN synthesis.

### 5. Multi-Frame Temporal Consistency & Anomaly Timeline
- **Existing Capability**: Random single-frame inference or unweighted global pooling.
- **FakeProbe-X Capability**: Configurable uniform temporal sampling with frame-by-frame anomaly scoring, spatial bounding-box jitter tracking, and suspicious timestamp localization.
- **Why It Helps**: Pinpoints temporal splice boundaries and transient facial artifacts in deepfake video sequences.

### 6. Graceful Modality Availability & Cross-Modal Consistency
- **Existing Capability**: Hardcoded feature concatenation where missing audio tracks are filled with zeros, causing out-of-distribution distortion in multimodal classifiers.
- **FakeProbe-X Capability**: Explicit modality tracking; adjusts weights dynamically when audio is missing and cross-examines visual vs acoustic evidence to detect voice-swap vs face-swap divergence.
- **Why It Helps**: Prevents zero-padding prediction corruption and reveals asymmetric modal manipulations.

### 7. Explainable Forensic Audit Reports
- **Existing Capability**: Unexplained binary scalar prediction.
- **FakeProbe-X Capability**: Machine-readable and human-auditable markdown forensic certificates detailing signal quality metrics, individual detector margins, evidence weights, and itemized reasoning.
- **Why It Helps**: Provides legal and forensic defensibility for downstream investigation.
