# Alertreck — System Design Document

**Version:** 2.0
**Last Updated:** April 2026
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
│  └─────────────┘   │  (3 s buf)   │   │  +HPF +RMS   │               │
│                    └──────────────┘   └──────┬───────┘               │
│                                              │ (1, 128, 259)         │
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
│  │  (console,   │                                                    │
│  │   GSM stub)  │                                                    │
│  └──────────────┘                                                    │
│                                                                      │
│  ┌──────────────┐                                                    │
│  │  GPS Reader  │  (optional, /dev/ttyUSB0)                          │
│  │  NMEA→coords │                                                    │
│  └──────────────┘                                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

The system runs a single inference loop. A rolling 3-second audio buffer is sampled every 1.5 seconds; each sample is preprocessed, classified, and routed through the decision engine, which maintains per-class thresholds and cooldowns to suppress alert spam.

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
│   ├── 03a_train_cnn.ipynb         CNN from scratch
│   ├── 03b_train_tiny_ast.ipynb    Vision Transformer (ViT-Tiny)
│   ├── 04a_train_conv_ae.ipynb     Convolutional Autoencoder (TODO)
│   └── 04b_train_oc_svm.ipynb      One-Class SVM (TODO)
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
    │   └── preprocess.py           Mel spectrogram + HPF + RMS norm
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
| RMS normalisation | target = 0.1 | Removes recording-volume bias from training signal |
| Window | 3 s clips, 50% overlap (hop = 1.5 s) | Multiplies training samples; detects events at any phase |
| Augmentation (train only) | ×6 per window: time-shift ±200 ms, gain ±6 dB, pitch ±2 st, noise @ 5/10/20 dB SNR | Domain robustness for field deployment |
| Feature: log-mel | 128 bins, n_fft=2048, hop=512 → (128, 259) | Used by CNN and ViT |
| Feature: MFCC+Δ+ΔΔ | 40 coeffs × 3 derivatives → (120, 259) | Used by Conv-AE and OC-SVM |
| Storage | NPZ shards of 1,000 samples | Fast random access, compressed |

**Reproducibility:** seed = 42 throughout; `script_sha256` hash recorded in `manifest.json` so any change to the script is detectable.

**Split:** stratified 60 / 20 / 20 by class on the fine-grained labels — every class keeps proportional representation in train/val/test.

### 5.3 Stage 3 — Training (Kaggle T4)

Each model trains independently on the same NPZ shards. Test set is never augmented to ensure honest evaluation.

| Model | Status | Test Acc | Test F1 |
|---|---|---|---|
| CNN from scratch | ✅ Done | 98.09% | 0.978 |
| Tiny-AST (ViT-Tiny) | 🟡 Training | — | — |
| Conv-AE | ⬜ Pending | — | — |
| OC-SVM | ⬜ Pending | — | — |

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
| `audio/preprocess.py` | Converts a raw waveform to a (1, 128, 259) mel spectrogram | `AudioPreprocessor` |
| `inference/model.py` | Loads ONNX model and runs softmax on logits | `ONNXModel` |
| `inference/decision.py` | Applies per-class threshold + per-class cooldown | `ThreatDecisionEngine` |
| `sensors/gps.py` | Reads NMEA over UART, parses lat/lon | `GPSReader` |
| `alerts/notifier.py` | Builds and dispatches alert dictionaries | `AlertNotifier` |
| `storage/evidence.py` | Saves a 3 s WAV file per alert in date/threat-type folders | `EvidenceManager` |
| `storage/logger.py` | Rotating logger writing to file + console | `get_logger()` |
| `main.py` | Orchestrates the inference loop; handles SIGINT/SIGTERM | `ALERTRACKSystem` |

### 6.2 Inference Loop (main.py)

```
every INFERENCE_INTERVAL (1.5 s):
    1. wait for buffer ready (3 s of audio)
    2. snapshot the buffer → numpy array (132 300 samples)
    3. preprocess:
         a. silence gate (skip if RMS < SILENCE_THRESHOLD)
         b. high-pass filter @ 120 Hz (remove electrical hum)
         c. RMS-normalise to 0.1
         d. compute 128-bin log-mel spectrogram
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

This is paired with a 4th-order Butterworth high-pass filter at 120 Hz to remove 50/60 Hz mains hum and harmonics that the model would otherwise misclassify.

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
WorkingDirectory=/home/pi/alertreck
ExecStart=/home/pi/alertreck/venv/bin/python3 -m alertrack.main
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

### 9.5 Pretrained vs From Scratch

**Comparing:** ViT-Tiny (pretrained ImageNet) vs CNN from scratch.
**Why both:** Capstone evaluation requires understanding whether transfer learning from a non-audio domain helps acoustic classification. Both train on the same data for fair comparison.

### 9.6 Local Storage Only (No Cloud)

**Considered:** Streaming alerts to a cloud dashboard via cellular.
**Chose:** Local-only for now; GSM stub left in `notifier.py`.
**Why:** Field deployment must work offline first. Cellular is intermittent in remote parks. Future work: integrate SIM800L + LoRaWAN.

---

## 10. Future Work

| Item | Priority |
|---|---|
| Complete Conv-AE and OC-SVM models | High (capstone) |
| Cross-model latency benchmark on Pi 4 | High |
| Quantise CNN to int8 with onnxruntime quantization | Medium |
| Integrate SIM800L for GSM SMS alerts | Medium |
| Add LoRaWAN module for low-power transmission | Low |
| Web dashboard for evidence review | Low |
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
