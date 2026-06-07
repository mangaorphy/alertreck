# Alertreck

Offline edge-AI system for anti-poaching audio detection on a Raspberry Pi 4.
Detects gunshots, chainsaws, human voices, and other threat sounds in real time,
then sends GPS-tagged alerts via GSM when a threat is confirmed — no internet required.

---

## Overview

Alertreck is a capstone research project comparing five audio classification models
across four machine learning paradigms for wildlife protection:

| Paradigm | Model | Notebook |
|---|---|---|
| Supervised classification | Custom CNN | `03a-train-cnn` |
| Few-shot learning | Prototypical Network | `03b-train-protonet` |
| Transfer learning (frozen) | Wav2Vec2-L2 | `04a-train-w2v2-l2` |
| Unsupervised anomaly detection | Convolutional Autoencoder | `04b-train-conv-ae` |
| Classical anomaly detection | One-Class SVM | `04c-train-oc-svm` |

---

## Hardware

| Component | Spec |
|---|---|
| Edge device | Raspberry Pi 4 (4 GB) |
| Microphone | USB cardioid mic |
| Alert module | SIM808 GSM/GPS hat |
| Deployment | Offline-first; alert via SMS/GPRS |

---

## Dataset

7 audio classes sampled at 44.1 kHz, clipped to 3-second windows:

| Label | Class | Category |
|---|---|---|
| 0 | `background_animals` | Background |
| 1 | `background_wind_rain` | Background |
| 2 | `threat_chainsaw` | Threat |
| 3 | `threat_dog` | Threat |
| 4 | `threat_gunshot` | Threat |
| 5 | `threat_human` | Threat |
| 6 | `threat_vehicle` | Threat |

Raw audio is in `dataset/`. Processed shards (mel spectrograms, MFCCs, W2V2 embeddings)
are in `data/processed/` and mirrored as Kaggle datasets for GPU training.

---

## Project Structure

```
alertreck/
├── dataset/                    # Raw audio files (7 class folders)
├── data/
│   └── processed/
│       ├── mel/                # Log-mel spectrogram shards (128 × 301)
│       ├── mfcc/               # MFCC+Δ+ΔΔ shards (120 × T)
│       ├── w2v2_l2/            # Wav2Vec2 embeddings (768-dim, L2-normalised)
│       ├── splits.json         # Stable 60/20/20 file-level split (seed 42)
│       └── manifest.json       # Per-file metadata
├── notebooks/
│   ├── 02b-prepare-w2v2-embeddings.ipynb
│   ├── 03a-train-cnn.ipynb
│   ├── 03b-train-protonet.ipynb
│   ├── 04a-train-w2v2-l2.ipynb
│   ├── 04b-train-conv-ae.ipynb
│   └── 04c-train-oc-svm.ipynb
├── scripts/
│   ├── audio_preprocessing.py      # Mel / MFCC shard generation + augmentation
│   ├── prepare_w2v2_embeddings.py  # W2V2 embedding extraction
│   ├── data_manifest.py
│   ├── data_loader.py
│   └── export_model.py
├── models/                     # Saved checkpoints and ONNX exports
├── alertrack/                  # Raspberry Pi deployment service
├── docs/                       # Project documentation
│   ├── ROADMAP.md
│   ├── DESIGN.md
│   ├── MODEL_DESIGN.md
│   ├── AUDIO_PREPROCESSING.md
│   ├── DEPLOYMENT.md
│   └── PROPOSAL.md
└── README.md
```

---

## Training Pipeline

### 1. Preprocessing (local)

```bash
# Generate clean splits + all curriculum augmentation phases
python3 scripts/audio_preprocessing.py --aug-phase A B C
```

Outputs mel and MFCC shards to `data/processed/`. Upload to Kaggle as datasets
`alertreck-mel2` (mel) and `alertreck-mfcc` (MFCC) before running training notebooks.

### 2. W2V2 Embeddings (Kaggle)

Run `02b-prepare-w2v2-embeddings.ipynb` on Kaggle (GPU, ~2 hrs).
Upload the output as dataset `w2v2-embeddings`.

### 3. Model Training (Kaggle GPU)

Run notebooks `03a` → `03b` → `04a` → `04b` → `04c` in order.
Each notebook saves `results.json` and an ONNX or joblib model to `/kaggle/working/`.

---

## Results

| Model | Test Acc | Macro F1 | Macro AUC | Gunshot F1 |
|---|---|---|---|---|
| ProtoNet | **0.9311** | 0.9205 | **0.9938** | 0.9969 |
| Wav2Vec2-L2 | 0.9297 | **0.9210** | 0.9911 | 0.9979 |
| CNN | 0.9264 | 0.9166 | — | **0.9990** |
| Conv-AE (binary) | 0.5147 | — | 0.6033 | (AUC 0.37) |
| OC-SVM (binary) | — | — | — | — |

The three discriminative models are statistically tied at the top. **ProtoNet** is the accuracy
benchmark; the **CNN** is recommended for deployment (smallest, self-contained, real-time on Pi 4).
Full breakdown and the deployment rationale: [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md).

---

## Documentation

Full design and implementation notes are in [`docs/`](docs/):

- [ROADMAP.md](docs/ROADMAP.md) — project milestones and research questions
- [DESIGN.md](docs/DESIGN.md) — system architecture
- [MODEL_DESIGN.md](docs/MODEL_DESIGN.md) — model specifications
- [MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) — comparative results and model selection
- [AUDIO_PREPROCESSING.md](docs/AUDIO_PREPROCESSING.md) — feature extraction details
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) — Raspberry Pi deployment guide

---

## Requirements

Training runs on Kaggle (Python 3.10, PyTorch 2.x, CUDA).  
Edge inference runs on Raspberry Pi 4 with ONNX Runtime (CPU) or scikit-learn (OC-SVM).

```
torch torchvision torchaudio
transformers
scikit-learn
librosa soundfile
onnxruntime
joblib
tqdm matplotlib
```

---

*Capstone project — African Leadership University, 2025–2026*
