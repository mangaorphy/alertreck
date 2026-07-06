# Alertreck — Model Design Overview

**Version:** 1.1  
**Last Updated:** 2026-06-30  
**Author:** Orpheus Manga

---

## 1. Four-Paradigm Framework

Alertreck trains five models across four learning paradigms on identical data to answer whether paradigm choice matters for acoustic threat detection on an edge device.

| # | Model | Paradigm | Input | Framework |
|---|---|---|---|---|
| 1 | CNN | Supervised | 128-bin log-mel (128 × 301) | PyTorch |
| 2 | ProtoNet | Metric learning | 128-bin log-mel (128 × 301) | PyTorch |
| 3 | W2V2-L2 | Frozen transfer | Raw waveform @ 16 kHz | HuggingFace + torchaudio |
| 4 | Conv-AE | Unsupervised anomaly | 128-bin log-mel (128 × 301) | PyTorch |
| 5 | OC-SVM | Classical one-class | 120-dim MFCC+Δ+ΔΔ | scikit-learn |

**Seven-class taxonomy:** `background_animals`, `background_wind_rain`, `threat_gunshot`, `threat_chainsaw`, `threat_vehicle`, `threat_human`, `threat_dog`

**Shared training data:** 8,907 clips, stratified 60/20/20 split, seed 42, test set locked.

---

## 2. Model 1 — CNN (Supervised Baseline)

### 2.1 Role

Primary production model deployed on Pi 4. Establishes the supervised upper bound for the comparison study.

### 2.2 Input

```
128-bin log-mel spectrogram
  window : 25 ms (Hann), hop : 10 ms
  shape  : (1, 128, 301)   — (channels, mel_bins, time_frames)
  origin : data/processed/mel/{split}/
```

### 2.3 Architecture

```
Input (1, 128, 301)
  │
  ├── Block 1: Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
  ├── Block 2: Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
  ├── Block 3: Conv2D(128, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
  ├── Block 4: Conv2D(256, 3×3) → BN → ReLU → GlobalAvgPool
  │
  ├── Dense(512) → ReLU → Dropout(0.5)
  └── Dense(7) → Softmax
```

**Parameters:** ~1.2 M  
**Receptive field:** captures ~300 ms spectro-temporal patterns per layer stack.

### 2.4 Training

| Setting | Value |
|---|---|
| Loss | Focal loss (γ = 2, α = inverse class frequency) |
| Optimiser | Adam (lr = 1e-3, weight decay = 1e-4) |
| Scheduler | CosineAnnealingLR (T_max = 50 epochs) |
| Batch size | 64 |
| Early stopping | patience = 10 on val F1 |
| Curriculum | Phase A (≥15 dB) → Phase B (10–15 dB) → Phase C (5–10 dB SNR) |

Focal loss suppresses the large background class gradient so the model learns the rarer threat classes properly.

### 2.5 Augmentation (training only)

SpecAugment (freq/time masking), FilterAugment (random EQ), Mixup (α = 0.4), compound augmentation pipeline. Applied to mel shards before batching.

### 2.6 Deployment

```
best_model.pt → export_model.py → alertreck_cnn.onnx (fp32, opset 17, dynamic batch + frames)
                                     ↓
                             ONNX Runtime on Pi 4 (CPU)
                               inference latency < 1.5 s / window
```

Grad-CAM is applied to the last conv block at eval time for qualitative explainability.

---

## 3. Model 2 — ProtoNet (Metric Learning)

### 3.1 Role

Few-shot-capable metric learning baseline. Reuses the CNN encoder; only the prototypical head changes. Answers whether metric learning improves generalisation to underrepresented classes (e.g. `threat_chainsaw` at 108 clips).

### 3.2 Input

Same log-mel shards as CNN: `(1, 128, 301)` from `data/processed/mel/`.

### 3.3 Architecture

```
Shared Encoder  ← identical to CNN blocks 1–4 + GlobalAvgPool → 256-dim embedding
      │
      ▼
  Embedding space (256-dim L2-normalised)
      │
  ┌───┴────────────────────────┐
  │  Episode construction       │
  │  support set → class        │
  │  prototypes (mean per class)│
  └───────────────┬────────────┘
                  │
  Query distances → nearest prototype → class label
```

The encoder is **initialised from the CNN's best checkpoint** and then fine-tuned with the episodic objective.

### 3.4 Training

| Setting | Value |
|---|---|
| Loss | Prototypical loss + SupCon (λ = 0.1) |
| Optimiser | Adam (lr = 1e-4 — lower than CNN to preserve pretrained weights) |
| Episode | N=7 way, K=5 shot, 15 queries per class |
| Curriculum | Same three-phase SNR curriculum as CNN |
| Scheduler | CosineAnnealingLR (T_max = 30 epochs post-pretraining) |

SupCon auxiliary term (λ = 0.1) tightens within-class cluster separation in embedding space, which matters for classes with very few support examples.

### 3.5 Critical dependency

**ProtoNet cannot start until the CNN encoder checkpoint is available.** Stage 3b follows Stage 3a on the critical path.

### 3.6 Deployment

ProtoNet can operate in two modes on Pi 4:
1. **Nearest-prototype**: compute query embedding, find closest stored prototype — same latency as CNN
2. **Linear probe fallback**: freeze encoder, fit a `LogisticRegression` head on stored prototypes — pure scikit-learn, no PyTorch on Pi

---

## 4. Model 3 — W2V2-L2 (Frozen Transfer)

### 4.1 Role

Out-of-species / out-of-domain frozen transfer baseline. Answers RQ5: does the Geldenhuys & Niesler (2026) finding (frozen layer-2 wav2vec approaching supervised CNN for elephant calls) extend to non-biological threats and survive INT8 quantisation?

### 4.2 Why wav2vec 2.0 layer 2 specifically?

wav2vec 2.0 was pretrained on 960 h of LibriSpeech English speech at 16 kHz. Its lower transformer layers (1–3) encode low-level acoustic features (pitch contours, transients, harmonic structure) that transfer broadly to non-speech audio. Layers 4+ encode speech-specific representations (phonemes, prosody) that actively harm non-speech classification. Layer 2 is the empirically optimal truncation point from the cited study.

### 4.3 Input

```
3-second audio clip
  resampled to 16 kHz mono (48,000 samples)     ← NOT the mel shards
  origin : data/processed/w2v2_l2/{split}/       ← new Step 4b shards
```

The existing mel `.npz` shards **cannot** be used — wav2vec 2.0 expects raw waveforms.

### 4.4 Architecture

```
Raw waveform (48,000 samples @ 16 kHz)
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  wav2vec 2.0 base — FULLY FROZEN (no gradients)       │
│                                                       │
│  Feature extractor: 7-layer CNN stack                 │
│    stride 320 → ~150 frames, 512-dim each             │
│                                                       │
│  Positional encoding                                  │
│                                                       │
│  Transformer Layer 1  ──────────────────────          │
│  Transformer Layer 2  ── tap hidden states ◄──────┐  │
│  Layers 3–12:  NOT LOADED  (memory saving)         │  │
└────────────────────────────────────────────────────┘  │
  │                                                      │
  ▼  shape: (batch, ~150, 768)                          │
Mean-pool over time axis                                 │
  │                                                      │
  ▼  shape: (batch, 768)                                │
┌──────────────────────┐                                │
│  Linear(768 → 7)      │  ← ONLY PART THAT TRAINS      │
│  ~5,376 parameters    │                                │
└──────────────────────┘
  │
  ▼
7-class logits
```

### 4.5 Pre-extraction strategy

Because the encoder is frozen, its output is deterministic. The efficient approach:

```
Stage 2b  → run all clips through frozen encoder once
           → mean-pool → save 768-dim vectors to data/processed/w2v2_l2/

Stage 4a  → load 768-dim .npz shards
          → train Linear(768, 7) only
          → completes in seconds, CPU-only, no GPU needed for head training
```

### 4.6 Training

| Setting | Value |
|---|---|
| Loss | Focal-weighted cross-entropy (γ = 2) |
| Optimiser | AdamW (lr = 1e-3, weight decay = 1e-4) |
| Encoder | Frozen — zero gradients throughout all phases |
| Batch size | 256 (small model, large batches fine) |
| Curriculum | Same three SNR phases (applied at waveform extraction time) |
| Head option A | `nn.Linear(768, 7)` — start here |
| Head option B | `nn.Linear(768, 256) → ReLU → nn.Linear(256, 7)` — only if A underfits |

No spectrogram augmentations (SpecAugment, Mixup) — those apply to mel representations. Waveform-level augmentation (noise, gain) can be applied during the Step 4b extraction pass.

### 4.7 Deployment

```
Frozen encoder stub (layers 1–2 only, ~40 MB ONNX) + INT8 linear head
Total footprint: ~42 MB  vs  ~360 MB for full wav2vec 2.0 base
Inference: raw mic audio → resample 16 kHz → encoder → mean-pool → head → class
```

The INT8 quantised head is negligible (~20 KB). Encoder forward pass on Pi 4 ARM: estimated < 2 s per 3-second clip (to be benchmarked in Stage 5).

---

## 5. Model 4 — Conv-AE (Unsupervised Anomaly Detection)

### 5.1 Role

Learns a compressed representation of background-only audio. At inference, high reconstruction error signals an anomalous (potentially threatening) acoustic event. Answers whether unsupervised detection is viable when labelled threat data is scarce.

### 5.2 Input

```
128-bin log-mel spectrogram (1, 128, 301)
Training: background_animals + background_wind_rain clips ONLY
Inference: all classes — anomaly score = MSE reconstruction error
```

### 5.3 Architecture

```
Encoder
  Input  (1, 128, 301)
  Conv2D(16, 3×3) → BN → ReLU → MaxPool(2×2)    →  (16,  64, 150)
  Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)    →  (32,  32,  75)
  Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2)    →  (64,  16,  37)
  Flatten → Dense(128)                            →  latent (128-dim)

Decoder
  Dense(128) → reshape (64, 16, 37)
  ConvTranspose2D(64, 3×3) → BN → ReLU → Upsample  →  (64, 32, 75)
  ConvTranspose2D(32, 3×3) → BN → ReLU → Upsample  →  (32, 64, 150)
  ConvTranspose2D(16, 3×3) → BN → ReLU → Upsample  →  (16, 128, 301)
  ConvTranspose2D(1,  1×1) → Sigmoid                →  (1,  128, 301)
```

### 5.4 Training

| Setting | Value |
|---|---|
| Loss | MSE (pixel-wise reconstruction error) |
| Optimiser | Adam (lr = 1e-3) |
| Training data | Background classes only (anomaly detection assumption) |
| Curriculum | Same three SNR phases (background clips only) |
| Batch size | 32 |
| Early stopping | patience = 10 on val reconstruction loss |

### 5.5 Anomaly threshold

After training, compute reconstruction error distribution on the **background validation set**. Set the anomaly threshold at the 95th percentile. Any test window with MSE above this threshold is flagged as anomalous.

At test time, report per-class AUC-ROC treating each threat class as the positive class against the background distribution.

### 5.6 Deployment

```
Encoder + Decoder → ONNX (opset 17)
Inference: compute mel → run AE → MSE vs background distribution → threshold → alert/no alert
(Note: when selected on validation detection AUC, Conv-AE reaches binary AUC 0.805 and gunshot AUC
0.839 — the best of the two anomaly detectors — but its 29 M params / ~110 MB and below-chance vehicle
AUC keep it as a complementary signal, not the primary classifier.)
```

The AE cannot report *which* threat is present — only that something unusual is detected. At deployment, the CNN is the primary classifier; Conv-AE is a complementary anomaly signal.

---

## 6. Model 5 — OC-SVM (Classical One-Class Baseline)

### 6.1 Role

Classical machine learning baseline requiring no deep learning. Trained on background-class MFCC features. Provides a lower-bound comparison and validates whether a GPU-free, library-only approach is viable on constrained hardware.

### 6.2 Input

```
MFCC + Δ + ΔΔ (40 coefficients × 3 → 120-dim vector)
Computed at 44.1 kHz, same 3-second windows
Origin: data/processed/mfcc/{split}/
Training: background classes only
```

MFCC is chosen over log-mel because it provides a compact, decorrelated feature vector — ideal for the kernel SVM's distance computations.

### 6.3 Architecture

```
120-dim MFCC+Δ+ΔΔ vector
  │
  ▼
StandardScaler (fit on background training set)
  │
  ▼
PCA (retain 95% variance, typically ~40–60 components)
  │
  ▼
OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
  │
  ▼
Decision function score → threshold → in-distribution / anomalous
```

### 6.4 Training

| Setting | Value |
|---|---|
| Loss | One-class hinge loss (OC-SVM objective) |
| Kernel | RBF |
| ν (nu) | 0.05 (expected anomaly fraction ≤ 5 %) |
| γ (gamma) | 'scale' (1 / (n_features × X.var())) |
| Training data | Background classes only |
| Curriculum | Same three SNR phases |
| Hyperparameter search | GridSearchCV on val set (ν ∈ {0.01, 0.05, 0.1}, γ ∈ {'scale','auto'}) |

### 6.5 Anomaly threshold

OC-SVM outputs a signed decision score. Threshold = 0 (the SVM boundary). Clips with score < 0 are anomalous.

### 6.6 Deployment

```
scaler + pca + svm → pickle (< 1 MB)
Inference: extract MFCC → transform → PCA → OC-SVM.predict → alert/no alert
CPU-only, < 50 ms per window on Pi 4
```

OC-SVM is the lightest model in the study. It can run alongside the CNN with negligible overhead as a confirmatory second opinion.

---

## 7. Comparative Summary

| Model | Trainable Params | Input Type | Training Data | Outputs | Pi 4 Latency (est.) |
|---|---|---|---|---|---|
| CNN | ~1.2 M | Log-mel (128×300) | All 7 classes | Class label + probabilities | < 1.5 s |
| ProtoNet | ~1.2 M (encoder) | Log-mel (128×300) | All 7 classes | Nearest prototype + distance | < 1.5 s |
| W2V2-L2 | 5,376 (head only) | Raw waveform 16 kHz | All 7 classes | Class label + probabilities | < 2 s (TBD) |
| Conv-AE | ~800 K | Log-mel (128×300) | Background only | Reconstruction error (anomaly score) | < 1.5 s |
| OC-SVM | < 1 MB (kernel) | MFCC+Δ+ΔΔ (120-dim) | Background only | In-distribution / anomalous | < 0.05 s |

### Evaluation targets (all models)

- AUC-ROC > 0.85 (per class)
- F1 > 0.80 (per class)
- False positive rate < 20%
- Inference latency < 30 s (alert delivery, end-to-end)
- Power ≤ 5 W sustained
- Grad-CAM qualitative validation (CNN and ProtoNet)

---

## 8. Data Flow Summary

```
dataset/ (raw audio, 7 classes)
    │
    ▼  scripts/audio_preprocessing.py
    │
    ├── Step 4a → data/processed/mel/{train,val,test}/     → CNN, ProtoNet, Conv-AE
    ├── Step 4b → data/processed/w2v2_l2/{train,val,test}/ → W2V2-L2 (768-dim embeddings)
    └── Step 4c → data/processed/mfcc/{train,val,test}/    → OC-SVM
```

Steps 4a and 4c are complete. **Step 4b (16 kHz waveform → frozen encoder → 768-dim) is the current Stage 2 blocker.**

---

## 9. Critical Path

```
Stage 2b: generate w2v2_l2/ shards (independent)
    │
    ▼
Stage 3a: train CNN          ──→  Stage 3b: train ProtoNet (needs CNN encoder)
    │                                    │
    └────────────────────────────────────┘
                                         │
    Stage 4a: train W2V2-L2 (independent of CNN)
    Stage 4b: train Conv-AE  (independent)
    Stage 4c: train OC-SVM   (independent)
                                         │
                                         ▼
    Stage 5: comparative evaluation (all five models, identical test set)
                                         │
                                         ▼
    Stage 6: deploy CNN (primary) + OC-SVM (confirmatory) on Pi 4
```

W2V2-L2 is independent of CNN — it only needs the Stage 2b waveform shards. All Stage 4 models can run in parallel once their respective inputs are ready.
