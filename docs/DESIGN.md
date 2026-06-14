# Alertreck — System Design Document

**Version:** 3.0
**Last Updated:** June 2026
**Author:** Orpheus Manga

---

## 1. Overview

Alertreck is an offline-first acoustic threat detection system designed to run continuously on a Raspberry Pi 4 deployed in remote conservation areas. It listens to a USB microphone, classifies 3-second audio windows into one of seven fine-grained classes, and triggers alerts for poaching-related sounds (chainsaws, gunshots, vehicles, human voices, dog barks).

The system trades model size for inference speed and resilience: a 1.2 M-parameter CNN runs in under 1.5 seconds per window on a Pi 4, with no internet dependency.

---

## 2. Goals & Non-Goals

### Goals

- **Real-time detection** of poaching threats from acoustic signals
- **Offline operation** — no internet required at inference time
- **Low cost** — total hardware ≤ USD 80
- **Per-class alerting** — every threat type detected independently with its own threshold
- **Evidence preservation** — every alert produces a WAV recording for forensic review
- **Fault tolerance** — auto-reconnect on microphone or GPS failures
- **Reproducibility** — fixed seed, hashed preprocessing script, manifest of all parameters

### Non-Goals

- Real-time spectrogram streaming to a cloud dashboard (out of scope; field-deployed)
- Multi-microphone localisation (single mic per device)
- Speaker identification or voice transcription
- Onboard model retraining (training runs on Kaggle T4)

---

## 3. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ALERTRECK (Raspberry Pi 4)                   │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐               │
│  │  USB Mic    │──▶│  Audio       │──▶│  Mel         │               │
│  │  44.1 kHz   │   │  Recorder    │   │  Preprocess  │               │
│  └─────────────┘   │  (3 s buf,   │   │  EBU + HPF   │               │
│                    │  onset-trig) │   └──────┬───────┘               │
│                    └──────────────┘          │ (1, 128, 301)         │
│                                              ▼                       │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐               │
│  │  Evidence   │◀──│  Decision    │◀──│  ONNX        │               │
│  │  WAV+JSON   │   │  Engine      │   │  CNN         │               │
│  └─────────────┘   │  per-class   │   │  Inference   │               │
│         ▲          │  threshold   │   └──────────────┘               │
│         │          │  + cooldown  │                                  │
│         │          └──────┬───────┘                                  │
│  ┌──────┴───────┐         │                                          │
│  │  Alert       │◀────────┘                                          │
│  │  Notifier    │                                                    │
│  │  (console +  │                                                    │
│  │   SIM808 SMS)│                                                    │
│  └──────────────┘                                                    │
│                                                                      │
│  ┌──────────────┐                                                    │
│  │  GPS Reader  │  (SIM808, /dev/ttyAMA0)                            │
│  │  NMEA→coords │                                                    │
│  └──────────────┘                                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

The system runs a single inference loop. A rolling 3-second audio buffer is monitored by an
**onset detector** (adaptive noise floor); when an energy onset is detected the window around it is
preprocessed, classified, and routed through the decision engine, which maintains per-class thresholds
and cooldowns to suppress alert spam. Each detection also saves a mel + metadata sidecar consumed by the
off-device **Grad-CAM dashboard** (`dashboard/`).

---

## 4. Repository Layout

```
alertreck/
├── dataset/                        Raw audio organised by class folder
│   ├── background_animals/
│   ├── background_wind_rain/
│   ├── threat_chainsaw/
│   ├── threat_dog/
│   ├── threat_gunshot/
│   ├── threat_human/
│   └── threat_vehicle/
│
├── data/processed/                 Output of preprocessing pipeline
│   ├── mel/{train,val,test}/       NPZ shards of log-mel features
│   ├── mfcc/{train,val,test}/      NPZ shards of MFCC+Δ+ΔΔ features
│   └── manifest.json               Run parameters + SHA256
│
├── models/
│   └── custom_cnn/
│       ├── alertreck_cnn.onnx      Exported model for Pi inference
│       ├── best_model.pt           PyTorch checkpoint (training output)
│       ├── results.json            Test metrics
│       ├── confusion_matrix.png
│       └── training_curves.png
│
├── notebooks/                      Kaggle T4 training notebooks
│   ├── 00-model-report.ipynb       Consolidated report (data viz + metrics + Grad-CAM)
│   ├── 02b-prepare-w2v2-embeddings.ipynb
│   ├── 03a-train-cnn.ipynb         CNN from scratch  (deployed)
│   ├── 03b-train-protonet.ipynb    Prototypical Network (few-shot)
│   ├── 04a-train-w2v2-l2.ipynb     Frozen wav2vec2 layer-2 + head
│   ├── 04b-train-conv-ae.ipynb     Convolutional Autoencoder
│   └── 04c-train-oc-svm.ipynb      One-Class SVM
│
├── scripts/
│   ├── audio_extraction/           Per-source data downloaders
│   ├── audio_preprocessing.py      Stage 2 — windowing + features
│   ├── data_manifest.py            Per-file metadata index
│   └── export_model.py             PyTorch → ONNX conversion
│
└── alertrack/                      Edge daemon (runs on Pi)
    ├── main.py                     Entry point + orchestration
    ├── config.py                   All thresholds, paths, constants
    ├── audio/
    │   ├── recorder.py             USB mic capture, rolling buffer
    │   ├── onset.py                Adaptive energy-onset detector (triggers inference)
    │   └── preprocess.py           Mel spectrogram + EBU R128 + hum high-pass
    ├── inference/
    │   ├── model.py                ONNX Runtime wrapper
    │   └── decision.py             Per-class threshold + cooldown logic
    ├── sensors/
    │   └── gps.py                  NMEA parser, optional /dev/ttyUSB0
    ├── alerts/
    │   └── notifier.py             Alert dispatcher (console + stubs)
    ├── storage/
    │   ├── logger.py               Rotating file log
    │   └── evidence.py             Per-threat WAV archive
    └── alertrack.service           systemd unit
```

---

## 5. Data Pipeline Design

### 5.1 Stage 1 — Data Collection

Audio is sourced from heterogeneous datasets, each contributing to specific classes:

| Source | Classes Contributed | Method |
|---|---|---|
| AudioSet (YouTube) | chainsaw | `yt-dlp` segment download by AudioSet label |
| ESC-50 | animals, wind/rain, gunshot, dog, vehicle | Category remapping |
| UrbanSound8K | dog, gunshot, vehicle | Full category extraction |
| Mozilla Common Voice | human | Random sample of validated clips |
| ff1010bird (freefield1010) | animals (birds) | Random 500 bird-positive clips |
| DATASET02 | animals, wind/rain | 200 clips per source species |

A **manifest CSV** (`data/processed/manifest.csv`) records every file's source, duration, sample rate, and class assignment. This supports auditability and removal of contaminated subsets (e.g. horse audio was removed when found to contain human background talking).

### 5.2 Stage 2 — Preprocessing

**Script:** `scripts/audio_preprocessing.py`

| Step | Detail | Reason |
|---|---|---|
| Resample | 44,100 Hz mono | Captures full acoustic range; consistent across sources |
| Loudness norm | EBU R128 → −23 dBFS, clipped to [−1, 1] | Removes recording-volume bias; volume-invariant features |
| Window | 3 s clips, 50% overlap (hop = 1.5 s) | Multiplies training samples; detects events at any phase |
| Augmentation (train only) | Three-phase curriculum A→B→C (1/2/3 copies, rising SNR/severity): noise, RIR, lowpass, mp3, gain, clip + SpecAugment + FilterAugment | Domain robustness for field deployment |
| Feature: log-mel | 128 bins, n_fft=2048, win=1102, hop=441 → **(128, 301)** | Used by CNN, ProtoNet, Conv-AE |
| Feature: MFCC+Δ+ΔΔ | 40 coeffs × 3 derivatives → **(120, 301)** | Used by OC-SVM |
| Storage | NPZ shards of 1,000 samples | Fast random access, compressed |

**Reproducibility:** seed = 42 throughout; `script_sha256` hash recorded in `manifest.json` so any change to the script is detectable.

**Split:** stratified 60 / 20 / 20 at the **file level** (all windows of one file stay in one split) — preserves class proportions *and* prevents leakage between overlapping windows. See [AUDIO_PREPROCESSING.md](AUDIO_PREPROCESSING.md) for the full pipeline and curriculum.

> W2V2-L2 does not use these shards — it consumes raw 16 kHz audio through the frozen `wav2vec2-base`
> backbone (notebook `02b`).

### 5.3 Stage 3 — Training (Kaggle T4)

Each model trains independently on the same NPZ shards (W2V2-L2 on raw-audio embeddings). The test set
is never augmented, to ensure honest evaluation. Five models span four ML paradigms.

| Model | Paradigm | Status | Test Acc | Macro F1 | AUC |
|---|---|---|---|---|---|
| CNN from scratch | Supervised | ✅ **Deployed** | 0.9264 | 0.9166 | — |
| ProtoNet | Few-shot metric | ✅ Done | **0.9311** | 0.9205 | **0.9938** |
| W2V2-L2 (frozen transfer) | Out-of-species transfer | ✅ Done | 0.9297 | **0.9210** | 0.9911 |
| Conv-AE | Unsupervised anomaly | ✅ Done | 0.5147† | — | 0.6033† |
| OC-SVM | Classical anomaly | ✅ Done | 0.5100† | — | 0.7790† |

† Conv-AE / OC-SVM are binary anomaly detectors (threat vs background); metrics are binary.

The three discriminative models are statistically tied at the top, all near-perfect on gunshot (F1 ≈
0.997). The **CNN** is deployed for its small, self-contained footprint. Full analysis:
[MODEL_COMPARISON.md](MODEL_COMPARISON.md). *(tiny-AST was dropped 2026-06-01 in favour of W2V2-L2 — see
[ROADMAP.md](ROADMAP.md).)*

### 5.4 Stage 4 — Export

**Script:** `scripts/export_model.py`

- Loads `best_model.pt` (PyTorch checkpoint)
- Exports to ONNX with opset 17, dynamic batch dimension
- Validates with `onnxruntime` before saving
- Output: `models/custom_cnn/alertreck_cnn.onnx` (~5 MB)

ONNX was chosen over TFLite because:
- No TensorFlow dependency on the Pi (lighter footprint)
- `onnxruntime` has pre-built ARM wheels
- Native PyTorch export is more reliable than the Torch → TF → TFLite chain

---

## 6. Edge Daemon Design (`alertrack/`)

### 6.1 Component Responsibilities

| Module | Responsibility | Key Class |
|---|---|---|
| `audio/recorder.py` | Captures mic audio in a background thread; maintains a rolling deque buffer | `AudioRecorder` |
| `audio/onset.py` | Tracks an adaptive noise floor; fires when energy rises above it | `OnsetDetector` |
| `audio/preprocess.py` | Converts a raw waveform to a (1, 128, 301) mel spectrogram (EBU R128 + hum HPF) | `AudioPreprocessor` |
| `inference/model.py` | Loads ONNX model and runs softmax on logits | `ONNXModel` |
| `inference/decision.py` | Applies per-class threshold + per-class cooldown | `ThreatDecisionEngine` |
| `sensors/gps.py` | Reads NMEA over UART, parses lat/lon | `GPSReader` |
| `alerts/notifier.py` | Builds and dispatches alert dictionaries | `AlertNotifier` |
| `storage/evidence.py` | Saves a 3 s WAV file per alert in date/threat-type folders | `EvidenceManager` |
| `storage/logger.py` | Rotating logger writing to file + console | `get_logger()` |
| `main.py` | Orchestrates the inference loop; handles SIGINT/SIGTERM | `ALERTRACKSystem` |

### 6.2 Inference Loop (main.py)

```
poll every ONSET_POLL_INTERVAL (0.25 s):
    1. wait for buffer ready (3 s of audio)
    2. onset detector: trigger if recent energy ≥ ONSET_TRIGGER_DB above the
       adaptive noise floor (and past the refractory window); else loop
    3. on trigger: settle ONSET_SETTLE_S so the event centres in the buffer,
       then snapshot the buffer → numpy array (132 300 samples)
    3b. preprocess:
         a. silence gate (skip if RMS < SILENCE_THRESHOLD)
         b. high-pass filter @ 90 Hz (remove 50/60 Hz mains hum)
         c. EBU R128 loudness normalisation (−23 dBFS)
         d. compute 128-bin log-mel spectrogram → (1, 128, 301)
    4. inference:
         a. ONNX session.run → logits
         b. softmax → probabilities
         c. argmax → predicted class
    5. decide:
         if class is a configured threat AND
            confidence ≥ class_threshold AND
            (now - last_alert_time[class]) ≥ class_cooldown:
              fire alert
    6. on alert:
         a. save WAV evidence (16-bit PCM, 44.1 kHz)
         b. build alert dict (id, class, conf, location, audio_path)
         c. notifier.send_alert(alert) → console + stubs
    7. periodic stats every STATS_INTERVAL (1 hr)
```

### 6.3 Per-Class Decision Logic

Every fine-grained class has its own configuration in `THREAT_CONFIG`:

```python
THREAT_CONFIG = {
    "threat_chainsaw":  (0.60, "HIGH",   300),
    "threat_dog":       (0.60, "MEDIUM", 300),
    "threat_gunshot":   (0.60, "HIGH",    60),  # short cooldown — instantaneous
    "threat_human":     (0.60, "HIGH",   300),
    "threat_vehicle":   (0.60, "HIGH",   300),
}
```

Format: `(threshold, level, cooldown_seconds)`.

Background classes (`background_animals`, `background_wind_rain`) are silent — they never trigger alerts regardless of confidence.

This design preserves the granularity of the model output. There is no class collapsing; a chainsaw is reported as a chainsaw, not as a generic "THREAT".

### 6.4 Silence Gate

Microphones on the Pi can pick up substantial electrical hum (≥ 0.6 RMS in some cases). To prevent the model from classifying amplified hum as a vehicle, the preprocessor applies a silence gate before inference:

```python
if rms < SILENCE_THRESHOLD:    # default 0.01
    return None                # skip this window
```

This is paired with an FFT high-pass filter at 90 Hz (`HPF_CUTOFF_HZ`, raised-cosine transition) that removes the 50/60 Hz mains-hum the model would otherwise misclassify as `threat_vehicle`. Because the training audio had no such hum, removing it brings the served signal *closer* to the training distribution rather than adding train/serve skew.

### 6.5 Evidence Layout

```
data/evidence/
  2026-04-21/
    threat_chainsaw/
      threat_chainsaw_20260421_143052_<alertID>.wav
    threat_gunshot/
      threat_gunshot_20260421_143124_<alertID>.wav
```

Each WAV is the exact 3-second buffer that triggered the alert. The alert ID is a SHA256 hash truncated to 16 chars and is also embedded in the alert JSON, allowing forensic correlation.

When total evidence exceeds `MAX_EVIDENCE_STORAGE_GB` (100 GB), oldest files are auto-deleted.

### 6.6 Alert Schema

```json
{
  "alert_id":          "fdc6029d045ef3d3",
  "timestamp":         "2026-04-21T15:26:59.338806Z",
  "device_id":         "ALERTRACK_001",
  "device_location":   "UNKNOWN_RESERVE",
  "threat_type":       "threat_vehicle",
  "threat_level":      "HIGH",
  "confidence":        0.930,
  "class_probabilities": {
    "background_animals":   0.001,
    "background_wind_rain": 0.001,
    "threat_chainsaw":      0.005,
    "threat_dog":           0.012,
    "threat_gunshot":       0.001,
    "threat_human":         0.050,
    "threat_vehicle":       0.930
  },
  "latitude":      "UNKNOWN",
  "longitude":     "UNKNOWN",
  "audio_evidence": "data/evidence/2026-04-21/threat_vehicle/..."
}
```

The full probability vector is included so post-hoc analysis can detect ambiguous predictions (e.g. when chainsaw and vehicle are both above 0.4).

---

## 7. Configuration

All tunable parameters are concentrated in `alertrack/config.py`. There are no hidden constants in other modules.

| Group | Parameters |
|---|---|
| Audio | `SAMPLE_RATE`, `CLIP_SECONDS`, `BUFFER_SIZE`, `MIC_DEVICE_INDEX` |
| Preprocessing | `N_MELS`, `N_FFT`, `HOP_STFT`, `FMIN`, `FMAX`, `SILENCE_THRESHOLD` |
| Detection | `THREAT_CONFIG` (per-class threshold/level/cooldown), `BACKGROUND_CLASSES` |
| Loop | `INFERENCE_INTERVAL`, `STATS_INTERVAL` |
| GPS | `GPS_ENABLED`, `GPS_PORT`, `GPS_BAUDRATE` |
| Storage | `EVIDENCE_DIR`, `MAX_EVIDENCE_STORAGE_GB`, `ALERT_RETENTION_DAYS` |
| Logging | `LOG_LEVEL`, `LOG_FILE`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT` |

`validate_config()` is called at startup and reports any issues (missing model file, unusual sample rate, invalid thresholds) without crashing.

Environment variable overrides are supported for deployment-time customisation:

```bash
ALERTRACK_DEVICE_ID=ALERTRACK_042 \
ALERTRACK_LOCATION="Kruger_North" \
ALERTRACK_DEBUG=true \
python3 -m alertrack.main
```

---

## 8. Operational Concerns

### 8.1 systemd Integration

`alertrack/alertrack.service` is installed to `/etc/systemd/system/`:

```ini
[Service]
User=alertreck
WorkingDirectory=/home/alertreck/alertreck
ExecStart=/home/alertreck/alertreck/venv/bin/python -m alertrack.main
Restart=always
RestartSec=10
```

The service auto-restarts on crash with a 10-second delay. Logs are written to both `data/logs/alertrack.log` (rotating, 10 MB × 5 backups) and `journalctl -u alertrack`.

### 8.2 Resource Usage (Pi 4, 4 GB)

| Metric | Observed |
|---|---|
| CPU (idle inference) | ~30–50 % of one core |
| Memory | ~300–500 MB |
| Inference latency | < 1 s per window |
| ONNX model size | ~5 MB |
| Evidence per day | ~50–500 MB (depends on activity) |

### 8.3 Failure Modes & Recovery

| Failure | Detection | Recovery |
|---|---|---|
| Microphone disconnects | sounddevice exception | Reconnect after `MICROPHONE_RECONNECT_DELAY` (5 s) |
| GPS port unavailable | serial exception | Continue without location; mark `UNKNOWN` |
| ONNX inference throws | try/except around session.run | Skip window; log error |
| Disk full | `MAX_EVIDENCE_STORAGE_GB` exceeded | Auto-delete oldest evidence files |
| Process crash | systemd watches | Auto-restart after 10 s |
| Buffer overflow | `sd.read` returns overflow flag | Counter incremented; logged in stats |

---

## 9. Trade-offs & Alternatives Considered

### 9.1 Coarse vs Fine-Grained Classification

**Considered:** Collapsing 7 classes into 3 (BACKGROUND, THREAT_CONTEXT, THREAT).
**Chose:** Fine-grained 7-class output with per-class alert configuration.
**Why:** Rangers benefit from knowing *which* threat is active. A chainsaw alert and a gunshot alert require very different responses.

### 9.2 ONNX vs TFLite

**Considered:** TensorFlow Lite (the original deployment design).
**Chose:** ONNX Runtime.
**Why:** Native PyTorch → ONNX export avoids the lossy PyTorch → TF → TFLite chain. ONNX Runtime has pre-built ARM wheels and no TF dependency.

### 9.3 Sample Rate

**Considered:** 16 kHz (standard for many speech models, smaller buffer).
**Chose:** 44.1 kHz.
**Why:** Captures full spectrum of mechanical sounds (chainsaw at 8 kHz, gunshot transients up to 15 kHz). Memory cost is acceptable on a 4 GB Pi.

### 9.4 Window Length

**Considered:** 1 s, 5 s, 10 s windows.
**Chose:** 3 s with 50 % overlap.
**Why:** Long enough to capture a gunshot echo or a chainsaw burst, short enough to keep inference latency ≤ 1.5 s.

### 9.5 Four-Paradigm Model Comparison

**Comparing:** supervised CNN, few-shot ProtoNet, frozen out-of-species transfer (W2V2-L2), and two
anomaly detectors (Conv-AE, OC-SVM).
**Why:** the capstone evaluates which learning paradigm best suits scarce, domain-shifted acoustic data.
*(tiny-AST — a fine-tuned audio transformer — was the original transfer arm but was dropped 2026-06-01:
fine-tuning a transformer on a small corpus overfits under domain shift. A frozen, truncated wav2vec 2.0
layer-2 embedding is a genuinely distinct paradigm and a better fit. See [ROADMAP.md](ROADMAP.md).)*

### 9.6 Offline-First Alerting

**Chose:** offline-first GSM/SMS via SIM808, with GPS coordinates — no internet dependency.
**Why:** cellular SMS reaches rangers where data coverage is intermittent. Explainability (Grad-CAM) is
served by an **off-device dashboard** that syncs detections over the LAN when one is available, keeping
the Pi itself lean (ONNX-only). Both are implemented (`alertrack/alerts/notifier.py`, `dashboard/`).

---

## 10. Future Work

| Item | Status / Priority |
|---|---|
| Five models across four paradigms | ✅ Done |
| SIM808 GSM/SMS + GPS alerting | ✅ Done |
| Grad-CAM explainability dashboard | ✅ Done |
| Onset-triggered inference | ✅ Done |
| Cross-model latency benchmark on Pi 4 | High |
| Quantise CNN to int8 with onnxruntime quantization | Medium |
| Add LoRaWAN module for low-power transmission | Low |
| Onboard model retraining loop (active learning) | Research |

---

## 11. Glossary

| Term | Definition |
|---|---|
| AudioSet | Google's large-scale dataset of YouTube-sourced audio events |
| Mel spectrogram | 2D time-frequency representation on the perceptual mel scale |
| MFCC | Mel-frequency cepstral coefficients — compact timbre features |
| ONNX | Open Neural Network Exchange — cross-framework model format |
| RMS | Root mean square — energy / loudness measure of an audio signal |
| SpecAugment | Spectrogram-domain data augmentation (frequency/time masking) |
| Stratified split | Train/val/test split that preserves class proportions |

---

*This document evolves with the codebase. When adding a major component, update §5–6 and §9 to reflect the change.*
