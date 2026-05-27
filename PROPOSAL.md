# Alertreck — Acoustic Poaching Threat Detection System
## Capstone Project Proposal

**Student:** Orpheus Manga (o.manga@alustudent.com)  
**Facilitator:** Dirac  
**Programme:** BSc Software Engineering / Machine Learning — African Leadership University  
**Date:** April 2026

---

## 1. Problem Statement

Wildlife poaching remains a critical threat to biodiversity across Africa. Remote park rangers often lack the tools to detect poaching activity in real time, particularly at night or across large terrain. Existing commercial systems are expensive, require internet connectivity, and are not designed for resource-constrained field conditions.

**Alertreck** addresses this by deploying an acoustic threat detection system on a Raspberry Pi 4 (cost ≤ USD 80) that listens continuously for poaching-related sounds — chainsaws, gunshots, vehicle engines, and human presence — and triggers an alert without any internet connection.

---

## 2. Proposed Approach

### 2.1 System Architecture

```
Microphone (USB)
      ↓
  Audio Buffer (3-second rolling window @ 44.1 kHz)
      ↓
  Log-Mel Spectrogram (128 × 259)
      ↓
  CNN / ViT Inference (ONNX Runtime on Pi 4)
      ↓
  Threat Decision Engine (per-class threshold + cooldown)
      ↓
  Alert + Evidence Storage (WAV + JSON)
      ↓
  GSM / LoRaWAN Notification (stub — future work)
```

### 2.2 Research Question

**Which model architecture — supervised (CNN, tiny-AST) or unsupervised (Conv-AE, OC-SVM) — provides the best trade-off between detection accuracy and inference speed on a Raspberry Pi 4?**

The comparison is designed to inform a deployment decision: a high-accuracy supervised model may be preferable if training data is abundant, while an unsupervised model may generalise better to novel sounds not seen during training.

### 2.3 Models Under Comparison

| # | Model | Type | Input | Notebook |
|---|---|---|---|---|
| 1 | CNN from Scratch | Supervised | 128×259 log-mel | `03a_train_cnn.ipynb` |
| 2 | Tiny Audio Spectrogram Transformer | Supervised | 224×224 (resized) | `03b_train_tiny_ast.ipynb` |
| 3 | Convolutional Autoencoder | Unsupervised | 128×259 log-mel | `04a_train_conv_ae.ipynb` |
| 4 | One-Class SVM | Unsupervised | 120×259 MFCC+Δ+ΔΔ | `04b_train_oc_svm.ipynb` |

All four models are trained on identical data splits to ensure a fair comparison.

---

## 3. Dataset

### 3.1 Class Taxonomy

The dataset uses a 7-class fine-grained taxonomy. At inference time, classes 0–1 are silent (background); classes 2–6 trigger alerts.

| Class | Label | Kind | Alert Level |
|---|---|---|---|
| `background_animals` | 0 | Background | None |
| `background_wind_rain` | 1 | Background | None |
| `threat_chainsaw` | 2 | Threat | HIGH |
| `threat_dog` | 3 | Threat context | MEDIUM |
| `threat_gunshot` | 4 | Threat | HIGH |
| `threat_human` | 5 | Threat | HIGH |
| `threat_vehicle` | 6 | Threat | HIGH |

### 3.2 Data Sources

| Class | Source | Files | Method |
|---|---|---|---|
| `background_animals` | ff1010bird (freefield1010), DATASET02 (elephant, lion, bird), ESC-50 (animals) | 2,140 | Random sample 500 bird clips; 200 per DATASET02 species; ESC-50 animal categories |
| `background_wind_rain` | DATASET02 (rainfall, wind), ESC-50 (rain, wind) | 680 | 200 per source category |
| `threat_chainsaw` | AudioSet (YouTube) via `yt-dlp` | 529 | Segment download using label `/m/01j4z9` from `unbalanced_train_segments.csv` |
| `threat_dog` | UrbanSound8K, ESC-50 (dog bark) | 1,040 | Full category extraction |
| `threat_gunshot` | UrbanSound8K, ESC-50 (gunshot), AudioSet | 2,400 | Full category extraction + AudioSet top-up |
| `threat_human` | Mozilla Common Voice (English) | 1,040 | Random sample of validated clips |
| `threat_vehicle` | UrbanSound8K (engine idling, car horn), ESC-50 | 1,040 | Full category extraction |

### 3.3 Dataset Statistics

| Metric | Value |
|---|---|
| Total audio files | **8,863** |
| Total audio duration | **~10.67 hours** |
| Total 3-second windows | **17,054** |
| Training windows (after augmentation ×6) | **71,624** |
| Validation windows | **3,411** |
| Test windows | **3,411** |

### 3.4 Data Collection Scripts

All extraction scripts are version-controlled under `scripts/audio_extraction/`:

| Script | Purpose |
|---|---|
| `extract_ff1010_birds.py` | Samples 500 bird clips from freefield1010 dataset |
| `extract_esc50.py` | Maps ESC-50 categories to project taxonomy |
| `extract_dataset02.py` | Extracts 200 clips per species from DATASET02 |
| `download_chainsaws_500.py` | Downloads chainsaw clips from AudioSet via yt-dlp |
| `merge_extracted_audio.py` | Merges sources into unified dataset folders |
| `data_manifest.py` | Generates `manifest.csv` with per-file metadata |

---

## 4. Preprocessing Pipeline

**Script:** `scripts/audio_preprocessing.py`

All audio is processed through a fixed pipeline before model training:

| Step | Detail |
|---|---|
| Resample | 44,100 Hz mono |
| Normalise | RMS normalisation (target = 0.1) |
| Window | 3-second clips, 50% overlap (hop = 1.5s) |
| Split | 60 / 20 / 20 stratified by class, seed = 42 |
| Augmentation | ×6 per training window: time-shift ±200ms, gain ±6dB, pitch ±2 semitones, noise @ 5/10/20 dB SNR |
| Feature: Mel | 128-bin log-mel spectrogram (128 × 259) |
| Feature: MFCC | 40 MFCC + Δ + ΔΔ (120 × 259) |
| Storage | Compressed `.npz` shards (1,000 samples each) |

---

## 5. Current Progress

### 5.1 Completed

- [x] Dataset collection and taxonomy design (8,863 files across 7 classes)
- [x] Full preprocessing pipeline with augmentation
- [x] CNN from scratch — trained and evaluated on Kaggle T4 GPU
- [x] Tiny-AST (ViT-Tiny) — notebook built, training in progress
- [x] Model export pipeline (PyTorch → ONNX)
- [x] Edge deployment on Raspberry Pi 4 (ONNX Runtime, USB microphone, systemd service)
- [x] Real-time inference loop with per-class threat detection and evidence storage

### 5.2 CNN Results (Model 1)

| Metric | Value |
|---|---|
| Test accuracy | **98.09%** |
| Test macro-F1 | **97.76%** |
| gunshot F1 | **1.000** (perfect) |
| human F1 | 0.989 |
| chainsaw F1 | 0.980 |
| dog F1 | 0.936 (weakest — dog/animal overlap) |
| Parameters | 1,206,439 |
| Best epoch | 26 / 40 |

### 5.3 Remaining Work

- [ ] Tiny-AST — complete training and evaluation
- [ ] Convolutional Autoencoder (`04a_train_conv_ae.ipynb`)
- [ ] One-Class SVM (`04b_train_oc_svm.ipynb`)
- [ ] Cross-model comparison table and analysis
- [ ] Pi 4 inference latency benchmarks
- [ ] Final report

---

## 6. Deployment Architecture (Pi 4)

```
Raspberry Pi 4 (4 GB RAM)
  OS        : Raspberry Pi OS Lite 64-bit
  Runtime   : Python 3.11 + ONNX Runtime 1.17
  Model     : alertreck_cnn.onnx  (1.2M params, ~5 MB)
  Mic       : USB PnP Sound Device (44.1 kHz, mono)
  Service   : systemd alertrack.service (auto-start on boot)
  Storage   : Evidence WAV files + JSON alerts (local SD card)
  Network   : GSM stub ready for SIM800L integration
```

**Inference latency target:** ≤ 1.5 seconds per 3-second window  
**Hardware cost:** ≤ USD 80 (Pi 4 + USB mic + SD card + GSM module)

---

## 7. Key Design Decisions

| Decision | Rationale |
|---|---|
| 44.1 kHz sample rate | Captures full acoustic range of all threat sounds |
| 3-second windows | Long enough for a gunshot echo or chainsaw burst |
| 7 fine-grained classes | Enables per-threat alerting rather than binary threat/no-threat |
| ONNX Runtime (not TFLite) | Lighter dependency, no TensorFlow required on Pi |
| Supervised + unsupervised comparison | Capstone requirement; addresses the case where labelled data is scarce |
| AudioSet chainsaw download | No pre-existing chainsaw dataset; required YouTube scraping |

---

## 8. Repository Structure

```
alertreck/
  dataset/                  raw audio by class folder
  data/processed/           mel and mfcc NPZ shards
  models/custom_cnn/        trained ONNX model
  notebooks/                training notebooks (Kaggle)
  scripts/
    audio_extraction/       data collection scripts
    audio_preprocessing.py  Stage 2 pipeline
    export_model.py         PyTorch → ONNX export
  alertrack/                Raspberry Pi daemon
  ROADMAP.md                project milestones
  DEPLOYMENT.md             Pi setup guide
  AUDIO_PREPROCESSING.md    pipeline documentation
```

---

*This document will be updated as remaining models are trained and benchmarked.*
