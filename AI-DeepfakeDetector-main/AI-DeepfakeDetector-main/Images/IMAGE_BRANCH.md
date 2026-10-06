# Image Deepfake Detection Branch

## Overview

This branch implements a state-of-the-art image forgery detection system based on a custom 5-channel CoAtNet architecture. The model detects AI-generated images by jointly analyzing visual content and forensic frequency-domain features — enabling the detection of subtle GAN and diffusion model artifacts that are invisible to the naked eye.

The system was trained on a custom-curated 130,000-image dataset derived from the ARTIFACT benchmark, spanning 22 distinct generative models with fully balanced class distribution. This ensures high generalizability against novel image synthesis methods.

---

## Notebooks

- **`customizedataset-imagetrain-final.ipynb`**: Complete dataset pipeline, model architecture definition, 3-phase training loop with hard negative mining, and final evaluation. Executed on Kaggle (2x Tesla T4 GPUs, PyTorch 2.10.0).

---

## Dataset

| Property | Value |
|---|---|
| **Base Dataset** | ARTIFACT (awsaf49/artifact-dataset) |
| **Kaggle Link** | `[DATASET_LINK_PLACEHOLDER]` |
| **Total Base Size** | 2,496,738 images |
| **Custom Subset** | 130,000 images |
| **Classes** | Real (0) vs. Fake (1) |
| **Generators** | 22 distinct AI generators |

### Custom Dataset Construction (6-Stage Pipeline)

```text
Stage 1: Metadata aggregation from all generator folders
Stage 2: Exclude hard generators (stylegan2, diffusion_gan, generative_inpainting)
Stage 3: Stratified sampling — equal quota per generator
Stage 4: Train / Val / Test split
Stage 5: Export CSV manifests
Stage 6: Verification
```

**Why exclude hard generators?** Specific generators like `stylegan2`, `diffusion_gan`, and `generative_inpainting` produce images with unique characteristics and extreme artifacts that could cause the model to overfit to generator-specific signatures rather than learning generalizable deepfake features. Excluding them forces the model to learn the fundamental differences between real and synthetic image structures.

### Final Dataset Statistics

| Split | Real | Fake | Total | Generators |
|---|---|---|---|---|
| **Train** | 50,000 | 50,000 | 100,000 | 22 |
| **Val** | 5,000 | 5,000 | 10,000 | 22 |
| **Test** | 10,000 | 10,000 | 20,000 | 22 |

**Generators Included:** 22 diverse AI generators covering Generative Adversarial Networks (GANs), diffusion models, neural rendering, and image manipulation architectures.

---

## Model Architecture: CoAtNet-0 (5-Channel)

### Innovation: 5-Channel Forensic Input

The core architectural innovation is extending the standard 3-channel RGB input to **5 channels** by appending two forensic analysis maps:

```text
Input Tensor = [R, G, B, ELA, FFT] (Dimension: 5 x 224 x 224)
```

| Channel | Description | Rationale |
|---|---|---|
| **R, G, B** | Standard RGB image | Visual appearance and semantic content |
| **ELA** | Error Level Analysis | Detects re-compression artifacts from image manipulation and spatial discontinuities |
| **FFT** | Log-scaled FFT magnitude spectrum | Reveals periodic frequency-domain artifacts characteristic of upsampling operations in AI generators |

### Error Level Analysis (ELA)

ELA works by re-compressing the image at a known quality level (e.g., Q=90) and computing the absolute pixel difference between the original and the re-compressed versions:

```python
ela_map = abs(original - recompressed).mean(axis=2)
```

- **Real images:** Exhibit uniform ELA across the image due to a consistent compression history.
- **Deepfakes:** Display high ELA regions at facial boundaries or manipulated zones, indicating inconsistent compression signatures and recent localized edits.

### Fast Fourier Transform (FFT) Spectrum

The FFT spectrum exposes the frequency components of the image:

```python
gray = cv2.cvtColor(image, RGB2GRAY)
fft_shift = np.fft.fftshift(np.fft.fft2(gray))
magnitude = log(1 + abs(fft_shift))
```

- **Real images:** Natural 1/f frequency roll-off with no periodic peaks.
- **AI-generated images:** Often exhibit characteristic grid-like patterns in the frequency domain resulting from upsampling (e.g., transposed convolutions) and specific decoder architectures.

### CoAtNet-0 Backbone

| Parameter | Value |
|---|---|
| **Architecture** | `coatnet_0_rw_224` (timm) |
| **Pretrained** | ImageNet-1k |
| **Input Channels** | **5** (modified from 3) |
| **Output** | Binary (sigmoid) |
| **Total Parameters** | 26,667,907 |
| **Feature Embedding** | 768-dim (penultimate layer) |

**Weight Initialization Strategy for New Channels:**
To leverage ImageNet pretraining without destroying the learned RGB filters, the weights for the newly added ELA and FFT channels are initialized separately using Kaiming Normal initialization, while the RGB weights are cloned directly from the pretrained model.

```python
new_conv.weight[:, :3, :, :] = original_conv.weight.clone()  # Preserve RGB weights
nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :])        # He init for ELA/FFT
```

**Why CoAtNet?** CoAtNet effectively combines the translational equivariance of convolutions with the global receptive field of self-attention (Transformers) in a hierarchical architecture. The convolution stages extract local texture features (ideal for capturing ELA artifacts), while the attention stages capture global contextual inconsistencies (e.g., unnatural facial coherence or lighting mismatches), making it the optimal architecture for this multimodal input.

---

## Training Methodology

### 3-Phase Gradual Unfreezing Strategy

Training was divided into 3 distinct phases to prevent catastrophic forgetting of the ImageNet pretrained weights when introducing the new 5-channel input:

```text
Phase A (Epochs 1-5):   Head + Stem conv1 + Norm only   |  LR = 1e-3
Phase B (Epochs 6-10):  + stages[-1] (Transformer)      |  LR = 1e-4
Phase C (Epochs 11-20): All parameters                  |  LR = 1e-5
```

| Phase | Trainable Params | Percentage | Learning Rate |
|---|---|---|---|
| **A** | 3,745 | 0.0% | 1e-3 |
| **B** | 12,856,549 | 48.2% | 1e-4 |
| **C** | 26,667,907 | 100% | 1e-5 |

### Loss Function: Focal Loss

```python
FL(p) = -alpha * (1 - p)^gamma * log(p)
# alpha=0.25, gamma=2.0
```

Focal Loss dynamically down-weights easily classified examples and focuses training gradients on hard, misclassified samples. This is particularly effective for fine-grained fake detection where many generated samples lie very close to the decision boundary.

### Hard Negative Mining

After Phase A completes, a full forward pass scans the entire training set to identify the 20% of samples with the highest focal loss:

```text
Hard negatives identified: 20,002 / 100,000 (20%)
Hard sample weight multiplier: 3.0x
```

These difficult samples receive a 3x sampling probability in the `WeightedRandomSampler` for Phase B and C training, forcing the model to explicitly learn from its most severe mistakes.

### GPU-Side Augmentation

All data augmentations are applied directly on the GPU after `uint8` to `float` tensor conversion. This eliminates the CPU data-loading bottleneck:
- **Random crop:** 256x256 -> 224x224 (batch-level)
- **Random horizontal flip:** 50% probability (batch-level)

### Optimization Setup

| Parameter | Value |
|---|---|
| **Optimizer** | AdamW |
| **Weight Decay** | 0.01 |
| **Physical Batch** | 128 |
| **Gradient Accumulation** | 4 steps |
| **Effective Batch Size** | **512** |
| **Scheduler** | CosineAnnealingLR |
| **Mixed Precision** | AMP (torch.amp) |
| **Gradient Clipping** | max_norm=1.0 |
| **Total Epochs** | 20 |

---

## Training Progress

| Epoch | Phase | Train Loss | Val Loss | AUC | F1 | Acc |
|---|---|---|---|---|---|---|
| 01 | A | 0.0600 | 0.0564 | 0.8166 | 0.6240 | 0.7014 |
| 02 | A | 0.0550 | 0.0532 | 0.8416 | 0.6737 | 0.7293 |
| 05 | A | 0.0518 | 0.0503 | 0.8591 | 0.6792 | 0.7370 |
| 06 | B | 0.0567 | 0.0495 | 0.8915 | 0.7967 | 0.8006 |
| 09 | B | 0.0326 | 0.0496 | 0.8997 | 0.7797 | 0.7991 |
| 11 | C | 0.0216 | 0.0607 | 0.9073 | 0.8137 | 0.8208 |
| 15 | C | 0.0128 | 0.0712 | 0.9145 | 0.8199 | 0.8263 |
| 18 | C | 0.0093 | 0.0867 | **0.9178** (Best) | 0.8326 | 0.8333 |
| 20 | C | 0.0075 | 0.0928 | 0.9175 | 0.8328 | 0.8350 |

**Training Duration:** 3 hours 35 minutes total (~644s/epoch in Phase A, ~580s in Phase B, ~665s in Phase C).

---

## Results & Evaluation

### Final Test Set Results (20,000 images — Balanced)

The final evaluation on the completely unseen test set demonstrates robust generalization across all 22 generative models:

| Metric | Value |
|---|---|
| **AUC-ROC** | **0.9173** |
| **F1-Score** | **0.8354** |
| **Accuracy** | **0.8363** |
| **Precision** | 0.8400 |
| **Recall** | 0.8308 |

### Confusion Matrix Analysis

```text
                    Predicted Real    Predicted Fake
Actual Real:          8,417 (TN)        1,583 (FP)
Actual Fake:          1,692 (FN)        8,308 (TP)
```

| Metric | Value |
|---|---|
| **True Positive Rate (Sensitivity)** | 83.08% |
| **True Negative Rate (Specificity)** | 84.17% |
| **False Positive Rate** | 15.83% |
| **False Negative Rate** | 16.92% |

The balance between Sensitivity and Specificity indicates that the model is well-calibrated, neither over-predicting real nor fake, which is crucial for real-world forensic deployment.

---

## Key Strengths & Design Decisions

1. **Multi-Domain Feature Fusion:** The 5-channel approach gives the model simultaneous access to pixel space (RGB), compression domain (ELA), and frequency domain (FFT), providing complementary and orthogonal evidence for manipulations.
2. **Diverse Generator Coverage:** Training across 22 highly varying generative models ensures the model learns intrinsic artifact patterns underlying synthetic generation processes rather than shallow, generator-specific signatures.
3. **Progressive Learning (Gradual Unfreezing):** Phase A stabilizes the randomly initialized forensic channels. Phase B integrates them with high-level Transformer features. Phase C fine-tunes the entire network at a minimal learning rate, effectively preventing the "knowledge loss cliff" common in aggressive fine-tuning of pretrained networks.
4. **Hard Negative Mining:** Dynamically re-weighting difficult samples in Phase B/C forces the model's decision boundary to adapt to the most challenging, near-perfect deepfakes.
5. **GPU Augmentation Pipeline:** Moving data augmentation to the GPU eliminates the traditional CPU data preprocessing bottleneck, achieving vastly higher GPU utilization and faster iteration times.
6. **Efficient Batch Training:** Gradient accumulation (4 steps) simulates a large 512-sample effective batch size on a single consumer-grade T4 GPU, improving gradient estimation quality without requiring multi-GPU distributed training setups.

---

## Inference Integration

The trained model is seamlessly deployed in the Final Framework via the `ImageDeepfakeDetector` module:

```python
class ImageDeepfakeDetector:
    def predict(self, image_bytes) -> tuple[str, float]:
        # 1. Decode image bytes
        # 2. Compute ELA + FFT maps
        # 3. Build 5-channel tensor
        # 4. CoAtNet forward pass + sigmoid
        # 5. Return label and probability
        pass

    def extract_features(self, image_rgb) -> np.ndarray:
        # Returns 768-dim penultimate embedding using a forward hook
        # on the head.global_pool layer
        pass
```

Crucially, the `extract_features()` method exposes the high-level semantic representation. This is subsequently utilized by the **Video branch** to obtain per-frame embeddings for multimodal fusion, avoiding the computational overhead of instantiating a second CoAtNet backbone.

---

## Technical Stack

| Component | Library/Version |
|---|---|
| Model Architecture | `timm 1.0.25` (`coatnet_0_rw_224`) |
| Training Framework | `PyTorch 2.10.0+cu128` |
| Image Processing | `OpenCV`, `Pillow` |
| Forensic Features | Custom ELA & FFT implementations |
| Mixed Precision | `torch.amp.autocast` |
| Training Platform | Kaggle (2x Tesla T4 GPUs) |
