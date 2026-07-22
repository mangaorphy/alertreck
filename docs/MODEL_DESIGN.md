# Alertreck — Model Design Overview

**Version:** 2.0 (as-built)
**Last Updated:** 2026-07-09
**Author:** Orpheus Manga

> This document describes the models **as implemented and evaluated**, not as
> originally proposed. All architecture details are verified against the training
> notebooks (`notebooks/03a`–`04c`) and `models/*/results.json`; all latency and
> resource figures are measured on the deployment Raspberry Pi 4.

---

## 1. Four-Paradigm Framework

Alertreck trains **five models across four learning paradigms** on identical data
to answer whether paradigm choice matters for acoustic threat detection on an edge
device.

| # | Model | Paradigm | Input | Framework | Role |
|---|---|---|---|---|---|
| 1 | CNN | Supervised classification | 128-bin log-mel (128 × 301) | PyTorch | **Deployed** (primary) |
| 2 | ProtoNet | Metric learning (few-shot) | 128-bin log-mel (128 × 301) | PyTorch | Accuracy benchmark |
| 3 | W2V2-L2 | Frozen transfer | Raw waveform @ 16 kHz | HuggingFace + torchaudio | Benchmark (RQ5) |
| 4 | Conv-AE | Unsupervised anomaly | 128-bin log-mel (128 × 301) | PyTorch | Anomaly detector |
| 5 | OC-SVM | Classical one-class | 120-dim MFCC+Δ+ΔΔ | scikit-learn | Confirmatory (lightest) |

**Seven-class taxonomy:** `background_animals`, `background_wind_rain`,
`threat_chainsaw`, `threat_dog`, `threat_gunshot`, `threat_human`, `threat_vehicle`
(labels 0–6). Background classes never alert; the five threat classes trigger alerts.

**Shared training data:** 11,333 raw audio files (≈ 17.5 h) standardized into
**26,623 three-second windows**, split **group-aware 60/20/20** by parent recording
(seed 42) so segments of one recording never span train/test. Window counts:
train 14,854 · val 5,844 · test 5,925.

---

## 2. Model 1 — CNN (Supervised, Deployed)

### 2.1 Role

Primary production model deployed on the Pi 4. Establishes the supervised upper
bound for the comparison and is the smallest self-contained graph, which is why it
is the deployment pick over the statistically-tied ProtoNet.

### 2.2 Input

```
128-bin log-mel spectrogram
  sample rate : 44.1 kHz mono
  clip        : 3 s (132,300 samples)
  window      : 25 ms Hann (win_length 1102) · hop 10 ms (441) · n_fft 2048
  scaling     : power_to_db(ref=max)
  shape       : (1, 128, 301)   — (channels, mel_bins, time_frames)
  origin      : data/processed/mel/{split}/shard_NNN.npz
```

### 2.3 Architecture (`AudioCNN` — verified against `notebooks/03a-train-cnn.ipynb`)

Four VGG-style convolutional blocks (**two 3×3 convs each**), global average
pooling, then a two-layer classifier head.

```
Input (1 × 128 × 301)
   │
   ▼
ConvBlock 1:  Conv2d 1→32 (3×3, pad1, no bias) → BN → ReLU
              Conv2d 32→32 (3×3, pad1, no bias) → BN → ReLU
              MaxPool 2×2 → Dropout2d(0.2)                     → (32 × 64 × 150)
   ▼
ConvBlock 2:  Conv2d 32→64 ×2 (→BN→ReLU) → MaxPool → Dropout   → (64 × 32 × 75)
   ▼
ConvBlock 3:  Conv2d 64→128 ×2 (→BN→ReLU) → MaxPool → Dropout  → (128 × 16 × 37)
   ▼
ConvBlock 4:  Conv2d 128→256 ×2 (→BN→ReLU) → MaxPool → Dropout → (256 × 8 × 18)
   ▼
AdaptiveAvgPool2d(1×1)                                          → (256)
   ▼
Classifier:   Flatten → Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→7)
   ▼
7 logits → Softmax (at inference)
```

| Property | Value |
|---|---|
| Total parameters | **1,206,439** (~1.2 M) |
| Conv per block | 2 (3×3, padding 1, bias-free — BatchNorm supplies the shift) |
| Channel progression | 1 → 32 → 64 → 128 → 256 |
| Bridge | `AdaptiveAvgPool2d((1,1))` — makes the head input-size agnostic |
| Head | `Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→7)` |

The `encode()` method returns the 256-dim pre-classifier embedding; ProtoNet reuses
it (§3), so the two models share one backbone and the comparison is apples-to-apples.

### 2.4 Training (verified from `results.json` → `best_config`)

| Setting | Value |
|---|---|
| Loss | **Focal loss** (γ = 2.0, α = per-class weights) |
| Optimiser | **AdamW** (lr = 1e-3, weight_decay = 1e-4) |
| Scheduler | **CosineAnnealingLR** (T_max = epochs, η_min = 1e-6) |
| Batch size | 256 |
| Dropout | conv 0.2, fc 0.5 |
| Curriculum | Phase A (≥15 dB) → B (10–15 dB) → C (5–10 dB SNR) |

Focal loss (γ) down-weights easy, well-classified windows so the model focuses on
hard examples; the per-class α weights counter class imbalance. Together they target
the rarest, hardest threats (vehicle, dog) — see §7.

### 2.5 Augmentation (training only)

Three-phase compound curriculum applied at preprocessing: waveform effects
(noise, RIR, lowpass, MP3, gain, clip), then spectrogram-level SpecAugment
(time/freq masks) and FilterAugment (±6 dB random EQ); Mixup (α = 0.4) is applied in
the DataLoader. Full detail: `docs/AUDIO_PREPROCESSING.md`.

### 2.6 Deployment

```
best_model.pt → scripts/export_model.py → alertreck_cnn.onnx
  fp32, opset 17, dynamic batch axis, input "mel_spectrogram" [batch,1,128,301]
  size: 4.6 MB
       ↓
  ONNX Runtime on Pi 4 (CPU, no PyTorch)
```

**Measured inference latency (Pi 4, CPU, this ONNX graph):**
min 154 ms · median 158 ms · mean ≈ 190 ms · p95 351 ms per 3 s window.
This is ~15× faster than the 3 s window, so inference is comfortably real-time with
large headroom. (An earlier "≤ 80 ms" figure was an estimate, not a measurement, and
is superseded by these numbers.)

Grad-CAM is applied to the last conv block at eval time for explainability (RQ4).

---

## 3. Model 2 — ProtoNet (Metric Learning)

### 3.1 Role

Few-shot-capable metric-learning benchmark. Reuses the trained CNN encoder and
replaces the linear head with nearest-prototype classification. Answers RQ2: does
metric learning match the supervised CNN? (Result: statistically tied.)

### 3.2 Input

Same log-mel shards as CNN: `(1, 128, 301)` from `data/processed/mel/`.

### 3.3 Architecture (verified against `notebooks/03b-train-protonet.ipynb`)

```
Pretrained CNN encoder (blocks 1–4 + AdaptiveAvgPool) → 256-dim feature
   │   (loaded from the CNN checkpoint; encoder frozen in phase 1, unfrozen later)
   ▼
Projection head:  Linear(256→256, no bias) → BN1d → ReLU → Linear(256→256, no bias)
   ▼
L2-normalise → unit-length 256-dim embedding (on the hypersphere)
   ▼
forward(x) = embedding · prototypesᵀ   → 7 cosine-similarity scores
```

Prototypes are **class-mean embeddings**, stored as a registered buffer (computed,
not gradient-trained). Classification = nearest prototype by cosine similarity. Only
the projection head (and, in the fine-tune phase, the encoder) is trained.

### 3.4 Training

| Setting | Value |
|---|---|
| Loss | Prototypical + SupCon auxiliary (λ = 0.1) |
| Optimiser | Adam (low lr to preserve pretrained encoder) |
| Episodes | 7-way, 5-shot, 15 queries/class |
| Encoder | Frozen first, then unfrozen for end-to-end fine-tuning |
| Scheduler | CosineAnnealingLR |
| Curriculum | Same three-phase SNR curriculum as CNN |

### 3.5 Critical dependency

ProtoNet **requires the CNN encoder checkpoint** — Stage 3b follows Stage 3a.

### 3.6 Deployment

Nearest-prototype inference has the same cost as the CNN backbone. Its appeal is
few-shot extension: a new threat class can be added by computing its prototype from a
handful of examples, without retraining a classifier head.

---

## 4. Model 3 — W2V2-L2 (Frozen Transfer)

### 4.1 Role

Out-of-domain frozen-transfer benchmark. Answers RQ5: does the finding that a frozen
low wav2vec layer approaches a supervised CNN (for elephant calls) extend to
non-biological threats? **Result: not supported** — W2V2-L2 trails the CNN.

### 4.2 Why wav2vec 2.0 layer 2

wav2vec 2.0 was pretrained on 960 h of English speech at 16 kHz. Its lower
transformer layers encode low-level acoustic features (transients, harmonic
structure) that transfer to non-speech audio; higher layers encode speech-specific
representations that harm non-speech classification. Layer 2 is the empirically
chosen truncation point.

### 4.3 Input

```
3-second audio clip → 16 kHz mono (48,000 samples)   ← NOT the mel shards
origin : data/processed/w2v2_l2/{split}/  (768-dim pre-extracted embeddings)
```

### 4.4 Architecture

```
Raw waveform (48,000 @ 16 kHz)
   ▼
wav2vec 2.0 base — FULLY FROZEN
  7-layer CNN feature extractor (stride 320 → ~150 frames, 512-dim)
  + positional encoding
  Transformer layers 1–2 → tap layer-2 hidden states   (layers 3–12 not loaded)
   ▼  (batch, ~150, 768)
Mean-pool over time  →  (batch, 768)
   ▼
Trainable head (L2 variant): Linear(768→256) → ReLU → Linear(256→7)
   ▼
7-class logits
```

### 4.5 Pre-extraction strategy

The frozen encoder is deterministic, so all clips are passed through it **once**
(Stage 2b) and the 768-dim mean-pooled vectors are cached to
`data/processed/w2v2_l2/`. Head training then loads these vectors and completes
quickly on CPU.

### 4.6 Training

| Setting | Value |
|---|---|
| Loss | Focal-weighted cross-entropy (γ = 2) |
| Optimiser | AdamW (lr = 1e-3, weight_decay = 1e-4) |
| Encoder | Frozen throughout |
| Batch size | 256 |
| Curriculum | Same three SNR phases (applied at waveform extraction) |

### 4.7 Deployment

Not deployed (benchmark only). The frozen layer-1–2 encoder stub (~40 MB ONNX) plus
an INT8 head is feasible on the Pi but was not selected, since the CNN both scores
higher and is far smaller (4.6 MB).

---

## 5. Model 4 — Conv-AE (Unsupervised Anomaly Detection)

### 5.1 Role

Learns a compressed representation of **background-only** audio; high reconstruction
error at inference signals an anomalous (potentially threatening) event. Answers RQ3:
is label-free anomaly detection viable? (Partially — useful second opinion, not the
primary classifier.)

### 5.2 Input

```
128-bin log-mel (1, 128, 301)
Training : background_animals + background_wind_rain ONLY
Inference: all classes — anomaly score = MSE reconstruction error
```

### 5.3 Architecture

```
Encoder
  Input (1, 128, 301)
  Conv2D(16, 3×3) → BN → ReLU → MaxPool 2×2   → (16,  64, 150)
  Conv2D(32, 3×3) → BN → ReLU → MaxPool 2×2   → (32,  32,  75)
  Conv2D(64, 3×3) → BN → ReLU → MaxPool 2×2   → (64,  16,  37)
  Flatten → Dense → latent (latent_dim)

Decoder  (mirror)
  Dense → reshape (64, 16, 37)
  ConvTranspose 64→32→16 (→BN→ReLU→Upsample)
  ConvTranspose → (1, 128, 301) → Sigmoid
```

Latent dimension, learning rate, dropout and weight decay were tuned with Optuna
(`models/conv_ae/optuna_conv_ae.json`).

### 5.4 Training

| Setting | Value |
|---|---|
| Loss | MSE (pixel-wise reconstruction) |
| Optimiser | Adam |
| Training data | Background classes only |
| Selection metric | Validation detection AUC (not reconstruction loss) |
| Curriculum | Same three SNR phases (background clips only) |

### 5.5 Anomaly threshold

Reconstruction-error distribution is measured on the background validation set; the
threshold is set at a high percentile. Test windows above it are flagged anomalous.
Per-class AUC-ROC is reported treating each threat class as positive vs background.

### 5.6 Deployment

Not the primary classifier. When selected on validation detection AUC, Conv-AE
reaches **binary AUC 0.805** (gunshot AUC 0.839) — the better of the two anomaly
detectors — but it cannot name *which* threat is present, and its size keeps it as a
complementary signal only.

---

## 6. Model 5 — OC-SVM (Classical One-Class Baseline)

### 6.1 Role

GPU-free, library-only baseline trained on background MFCC features. Lower-bound
comparison and validation that a lightweight approach runs on constrained hardware.

### 6.2 Input

```
MFCC + Δ + ΔΔ (40 coefficients × 3 → 120-dim vector)
44.1 kHz, same 3-second windows
Origin  : data/processed/mfcc/{split}/
Training: background classes only
```

MFCC is chosen over log-mel for its compact, decorrelated vectors — ideal for the
kernel SVM's distance computations.

### 6.3 Architecture

```
120-dim MFCC+Δ+ΔΔ
   ▼ StandardScaler (fit on background train set)
   ▼ PCA (retain ~95% variance)
   ▼ OneClassSVM(kernel='rbf', nu, gamma)   ← grid-searched
   ▼ decision score → threshold → in-distribution / anomalous
```

### 6.4 Training

| Setting | Value |
|---|---|
| Objective | One-class SVM (RBF kernel) |
| Hyperparameters | ν and γ via grid search on val (`models/oc_svm/grid_search_oc_svm.json`) |
| Training data | Background classes only |
| Curriculum | Same three SNR phases |

### 6.5 Deployment

```
scaler + pca + svm → pickle (< 1 MB)
Inference: MFCC → transform → PCA → OC-SVM.predict → alert/no alert
CPU-only, lightest model in the study
```

OC-SVM reaches **binary AUC 0.719** — below Conv-AE but negligible in footprint, so
it can run alongside the CNN as a confirmatory second opinion.

---

## 7. Comparative Summary (measured results, leak-free group-aware split)

Headline metrics from `models/*/results.json`:

| Model | Test Acc | Macro F1 | Macro AUC | Trainable Params | Input | Deployed |
|---|---|---|---|---|---|---|
| **CNN** | **0.8263** | **0.8069** | **0.9757** | 1.21 M | Log-mel 128×301 | ✅ primary |
| ProtoNet | 0.8241 | 0.8036 | 0.9748 | ~1.2 M (shared encoder) | Log-mel 128×301 | benchmark |
| W2V2-L2 | 0.7806 | 0.7626 | 0.9605 | head only | Waveform 16 kHz | benchmark |
| Conv-AE | — | — | 0.8050 (binary) | — | Log-mel 128×301 | complementary |
| OC-SVM | — | — | 0.7192 (binary) | — | MFCC 120-dim | confirmatory |

### Per-class F1 (test set; window counts in parentheses)

| Class | Train / Test win. | CNN | ProtoNet | W2V2-L2 |
|---|---|---|---|---|
| background_animals | 3,053 / 1,373 | 0.836 | 0.820 | 0.762 |
| background_wind_rain | 4,195 / 1,855 | 0.830 | 0.823 | 0.756 |
| threat_chainsaw | 1,499 / 547 | 0.824 | 0.813 | 0.780 |
| threat_dog | 751 / 235 | 0.794 | 0.708 | 0.738 |
| threat_gunshot | 2,287 / 1,000 | 0.815 | 0.843 | 0.812 |
| threat_human | 2,431 / 696 | 0.868 | 0.895 | 0.858 |
| threat_vehicle | 638 / 219 | **0.681** | 0.723 | 0.632 |

The two smallest threat classes by **window count** — `threat_vehicle` (638) and
`threat_dog` (751) — are the two weakest per-class F1s across every model. Note that
`threat_chainsaw` has few raw files but long recordings, so overlapping windowing
yields 1,499 training windows and solid F1 (0.824). Vehicle's high AUC (≈ 0.97)
despite low F1 shows it is **mis-thresholded, not mis-represented** — a
threshold-calibration opportunity, not a modelling failure.

### Measured edge performance (Pi 4, CPU)

| Metric | Value |
|---|---|
| CNN inference latency | ≈ 160 ms / 3 s window (min 154, mean 190) — real-time |
| Power draw | ≈ 3.0–3.5 W (load-based estimate; no throttling at 47.7 °C) |
| Model size | 4.6 MB ONNX |

---

## 8. Data Flow Summary

```
dataset/ (raw audio, 7 classes, 11,333 files)
    │  scripts/audio_preprocessing.py
    ├── mel/{train,val,test}(+train_aug_A/B/C)  → CNN, ProtoNet, Conv-AE
    ├── mfcc/{train,val,test}                    → OC-SVM
    └── w2v2_l2/{train,val,test} (768-dim)       → W2V2-L2
```

All feature representations are computed up front so every model pulls whichever
input suits it without re-running audio processing.

---

## 9. Model Selection Rationale

- **CNN and ProtoNet are statistically tied** (macro-F1 0.8069 vs 0.8036, Δ 0.003).
  The tie means deployment can be decided on **footprint**: the CNN is a single
  self-contained 4.6 MB ONNX graph with no backbone dependency, so it is deployed.
- **W2V2-L2 trails** (0.7626) — the RQ5 hypothesis that frozen transfer beats
  task-trained supervised learning is **not supported** for this domain.
- **Conv-AE > OC-SVM** among anomaly detectors (AUC 0.805 vs 0.719); either can run
  as a label-free second opinion, with OC-SVM the lighter option.
- **On the edge, the CNN is primary; OC-SVM is an optional confirmatory signal.**
