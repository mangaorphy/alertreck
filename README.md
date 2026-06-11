# Alertreck

**Offline edge-AI system for anti-poaching audio detection on a Raspberry Pi 4.**
Alertreck listens continuously, detects gunshots, chainsaws, vehicles and human voices in real time,
and sends GPS-tagged alerts to rangers via GSM/SMS when a threat is confirmed — entirely offline, no
internet required.

It is also a comparative research study: **five models across four machine-learning paradigms** are
trained on the same data and benchmarked, then the best edge candidate is deployed with a live
**Grad-CAM explainability dashboard**.

---

## Data, Dataset & Models

The raw audio, processed feature shards, and trained model artifacts are **too large for git** and live
in Google Drive:

### → **[Alertreck Data, Dataset & Models (Google Drive)](https://drive.google.com/drive/folders/1U9BwIUNQ8Snl5RxR8LHthWfdOc_EdcTM?usp=sharing)**

| Folder | Place into | Contents |
|---|---|---|
| `dataset/` | `./dataset/` | Raw audio, 7 class folders (8,907 files) |
| `data/processed/` | `./data/processed/` | Mel / MFCC shards, W2V2 embeddings, `splits.json`, `manifest.json` |
| `models/` | `./models/` | Trained checkpoints, ONNX exports, `results.json` per model |

Download these and drop them into the matching paths at the repo root to run training, evaluation, the
report notebook, or the dashboard.

---

## Overview

| Paradigm | Model | Notebook | Role |
|---|---|---|---|
| Supervised classification | Custom CNN | `03a-train-cnn` | **Deployed** (edge) |
| Few-shot metric learning | Prototypical Network | `03b-train-protonet` | Accuracy benchmark |
| Transfer learning (frozen) | Wav2Vec2-L2 | `04a-train-w2v2-l2` | Benchmark |
| Unsupervised anomaly detection | Convolutional Autoencoder | `04b-train-conv-ae` | Negative result |
| Classical anomaly detection | One-Class SVM | `04c-train-oc-svm` | Confirmatory |

---

## Hardware

| Component | Spec |
|---|---|
| Edge device | Raspberry Pi 4 (4 GB) |
| Microphone | USB cardioid mic |
| Alert module | SIM808 GSM/GPS over UART |
| Deployment | Offline-first; SMS/GPRS alert with GPS coordinates |

---

## Dataset

7 audio classes, 44.1 kHz, sliced into 3-second windows. Background classes never alert; threat
classes (2–6) trigger alerts.

| Label | Class | Category | Raw files |
|---|---|---|---|
| 0 | `background_animals` | Background | 2,140 |
| 1 | `background_wind_rain` | Background | 680 |
| 2 | `threat_chainsaw` | Threat | 567 |
| 3 | `threat_dog` | Threat | 1,040 |
| 4 | `threat_gunshot` | Threat | 2,400 |
| 5 | `threat_human` | Threat | 1,040 |
| 6 | `threat_vehicle` | Threat | 1,040 |

Feature extraction, the file-level 60/20/20 split, and the three-phase augmentation curriculum are
documented in **[docs/AUDIO_PREPROCESSING.md](docs/AUDIO_PREPROCESSING.md)**.

---

## Project Structure

```
alertreck/
├── dataset/                       # Raw audio (7 class folders)            ── from Drive
├── data/processed/                # Mel/MFCC shards, W2V2 embeddings        ── from Drive
│   ├── mel/   {train, val, test, train_aug_A/B/C}/  shard_NNN.npz
│   ├── mfcc/  {same}/                                shard_NNN.npz
│   ├── w2v2_l2/                    # 768-dim L2-normalised embeddings
│   ├── splits.json                # Stable file-level 60/20/20 (seed 42)
│   └── manifest.json              # Full run parameters + script SHA256
├── notebooks/
│   ├── 00-model-report.ipynb      # Consolidated report: data viz + all metrics + Grad-CAM
│   ├── 02b-prepare-w2v2-embeddings.ipynb
│   ├── 03a-train-cnn.ipynb        ├ 03b-train-protonet.ipynb
│   ├── 04a-train-w2v2-l2.ipynb    ├ 04b-train-conv-ae.ipynb
│   └── 04c-train-oc-svm.ipynb
├── scripts/
│   ├── audio_preprocessing.py     # Mel/MFCC shards + augmentation curriculum
│   ├── prepare_w2v2_embeddings.py # W2V2 embedding extraction
│   ├── export_model.py            # PyTorch → ONNX (CNN, 301-frame, dynamic width)
│   ├── data_manifest.py  ├ data_loader.py
├── models/                        # Checkpoints, ONNX, results.json         ── from Drive
│   └── {custom_cnn, protonet, w2v2_l2, conv_ae, oc_svm}/
├── alertrack/                     # Raspberry Pi edge service
│   ├── main.py                    # Onset-triggered inference loop
│   ├── audio/{recorder,preprocess,onset}.py
│   ├── inference/{model,decision}.py
│   ├── sensors/gps.py  ├ alerts/notifier.py  ├ storage/{evidence,logger}.py
│   ├── config.py  └ alertrack.service        # systemd unit
├── dashboard/                     # Grad-CAM explainability dashboard (Flask, runs on a Mac)
│   ├── app.py  ├ gradcam.py  ├ templates/index.html
│   ├── sync_events.sh  └ README.md
├── docs/                          # Full documentation (see below)
└── README.md
```

---

## Pipeline

### 1. Preprocess (local)
```bash
python3 scripts/audio_preprocessing.py --aug-phase A B C
```
Writes mel + MFCC shards to `data/processed/`. Upload to Kaggle as `alertreck-mel2` / `alertreck-mfcc`.

### 2. W2V2 embeddings (Kaggle GPU, ~2 h)
Run `02b-prepare-w2v2-embeddings.ipynb`; upload the output as the `w2v2-embeddings` dataset.

### 3. Train (Kaggle GPU)
Run `03a → 03b → 04a → 04b → 04c`. Each saves `results.json` + an ONNX/joblib model.

### 4. Export for the edge
```bash
/opt/anaconda3/bin/python scripts/export_model.py \
  --model models/custom_cnn/best_model.pt --out models/custom_cnn/alertreck_cnn.onnx
```

### 5. Deploy to the Pi
Full runbook — flashing, mic calibration, onset tuning, GSM/GPS — in
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

---

## Results

| Model | Test Acc | Macro F1 | AUC | Gunshot | Edge-ready |
|---|---|---|---|---|---|
| ProtoNet | **0.9311** | 0.9205 | **0.9938** | F1 0.9969 | ✅ |
| Wav2Vec2-L2 | 0.9297 | **0.9210** | 0.9911 | F1 0.9979 | ❌ (94 M backbone) |
| **CNN (deployed)** | 0.9264 | 0.9166 | — | F1 **0.9990** | ✅ **best** |
| OC-SVM (binary) | 0.5100 | — | 0.7790 | AUC 0.805 | ✅ |
| Conv-AE (binary) | 0.5147 | — | 0.6033 | AUC 0.372 | |

The three discriminative models are statistically tied at the top, all near-perfect on **gunshot** (the
highest-stakes class). The **CNN** is deployed (smallest, self-contained, real-time on a Pi 4 CPU);
**ProtoNet** is the accuracy benchmark. Among anomaly detectors the classical **OC-SVM** clearly beats
the deep **Conv-AE**. Full analysis: **[docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md)**.

Reproduce every chart in **[notebooks/00-model-report.ipynb](notebooks/00-model-report.ipynb)** — class
distributions, all-model metrics, and Grad-CAM.

---

## Edge System (`alertrack/`)

The deployed daemon runs a fast ONNX CNN with field-hardened audio handling:

- **Onset-triggered inference** — classifies the 3 s window around a detected energy onset (adaptive
  noise floor), instead of a blind timer: fewer false positives, less CPU, event centred in the window.
- **Train/serve-consistent preprocessing** — EBU R128 loudness + `win_length=1102` + 301 frames,
  matching training exactly, plus a 50/60 Hz mains-hum high-pass for field mics.
- **Decision engine** — per-class thresholds, alert levels (HIGH/MEDIUM), independent cooldowns.
- **Alerting** — SIM808 GSM/SMS with GPS coordinates; audio + metadata evidence saved per detection.
- **Resilience** — `systemd` auto-restart, mic/GPS reconnect, storage limits.

---

## Explainability Dashboard (`dashboard/`)

A Flask LAN dashboard showing each detected sound with a **Grad-CAM** heatmap over its mel spectrogram —
*why* the CNN flagged it. Runs on a separate machine (Mac) where PyTorch lives, so the Pi stays
ONNX-only; it auto-syncs detections from the Pi and refreshes itself. See
**[dashboard/README.md](dashboard/README.md)**.

```bash
cd dashboard && /opt/anaconda3/bin/python app.py    # → http://<mac-ip>:8000
```

---

## Documentation

- [ROADMAP.md](docs/ROADMAP.md) — milestones and research questions
- [Alertreck_Proposal_Updated.pdf](Alertreck_Proposal_Updated.pdf) — original project proposal
- [DESIGN.md](docs/DESIGN.md) — system architecture
- [MODEL_DESIGN.md](docs/MODEL_DESIGN.md) — model specifications
- [AUDIO_PREPROCESSING.md](docs/AUDIO_PREPROCESSING.md) — feature extraction + augmentation curriculum
- [MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) — comparative results and model selection
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) — Raspberry Pi deployment runbook
- [dashboard/README.md](dashboard/README.md) — Grad-CAM dashboard

---

## Requirements

Training runs on Kaggle (Python 3.10, PyTorch 2.x, CUDA). Edge inference runs on a Raspberry Pi 4 with
ONNX Runtime (CPU) — no torch on the Pi. The Grad-CAM dashboard needs torch, on the Mac only.

```
# Training / dashboard (Mac/Kaggle):  torch transformers scikit-learn librosa soundfile
#                                      onnxruntime joblib tqdm matplotlib flask
# Edge (Raspberry Pi):                 numpy librosa sounddevice soundfile onnxruntime pyserial psutil
```

---

*Capstone project — African Leadership University, 2025–2026.*
*Large artifacts (dataset, processed data, models): [Google Drive](https://drive.google.com/drive/folders/1U9BwIUNQ8Snl5RxR8LHthWfdOc_EdcTM?usp=sharing).*
