# Audio Preprocessing Pipeline — Stage 2

**Script:** [`scripts/audio_preprocessing.py`](scripts/audio_preprocessing.py)  
**Output:** `data/processed/`  
**Completed:** 2026-04-20

---

## Overview

This pipeline takes raw audio files from `dataset/` and converts them into machine-learning-ready feature arrays. It is the bridge between raw audio collection (Stage 1) and model training (Stage 3).

```
dataset/
  background_animals/    (2,140 files)
  background_wind_rain/  (680 files)
  threat_chainsaw/       (529 files)
  threat_dog/            (1,040 files)
  threat_gunshot/        (2,400 files)
  threat_human/          (1,040 files)
  threat_vehicle/        (1,040 files)
          ↓
  audio_preprocessing.py
          ↓
data/processed/
  mel/  {train, val, test}/  shard_NNN.npz
  mfcc/ {train, val, test}/  shard_NNN.npz
  manifest.json
```

---

## Fixed Parameters

| Parameter | Value | Reason |
|---|---|---|
| Sample rate | 44,100 Hz | CD-quality; consistent across all sources |
| Clip length | 3 seconds | Long enough to capture a gunshot echo or chainsaw burst |
| Overlap | 50 % (hop = 1.5 s) | More training windows per file; detects events at any position |
| Mel bins | 128 | Detailed frequency resolution for spectrograms |
| MFCC coefficients | 40 | Compact timbre representation |
| FFT window | 2,048 samples (~46 ms) | Good frequency resolution |
| STFT hop | 512 samples (~11 ms) | Good time resolution |
| Random seed | 42 | Fully reproducible splits and augmentation |

---

## Class Labels

Each dataset folder maps to its own integer label — 7-class fine-grained classification:

| Folder | Label | Kind |
|---|---|---|
| `background_animals` | 0 | Background |
| `background_wind_rain` | 1 | Background |
| `threat_chainsaw` | 2 | Threat |
| `threat_dog` | 3 | Threat context |
| `threat_gunshot` | 4 | Threat |
| `threat_human` | 5 | Threat |
| `threat_vehicle` | 6 | Threat |

> At inference time, labels 2–6 trigger an alert; labels 0–1 do not.

---

## Pipeline Steps

### Step 1 — Load Audio (`load_mono`)

Every audio file is loaded and standardised:
- Reads WAV/FLAC/OGG via `soundfile`
- Reads MP3 via `ffmpeg` subprocess (soundfile reads MP3 as 0 samples on Python 3.13)
- Stereo → mono by averaging left and right channels
- Resamples to 44,100 Hz if the file uses a different sample rate

### Step 2 — RMS Normalisation (`rms_normalize`)

Each file's volume is normalised to a fixed RMS level (0.1) before any processing. This ensures the model does not learn from recording volume differences — a quiet gunshot and a loud one should look the same after normalisation.

### Step 3 — Windowing (`slice_windows`)

Each file is sliced into 3-second windows with 50% overlap:

```
Example — 7 second file:
  [  Window 1  ]              0.0s → 3.0s
        [  Window 2  ]        1.5s → 4.5s
              [  Window 3  ]  3.0s → 6.0s
              (7s - 3s hop = partial, discarded)
```

Files shorter than 3 seconds are zero-padded to exactly 3 seconds (1 window).  
This multiplies the number of training samples — a 10 s file produces 6 windows.

**Total windows collected from 8,863 files: 17,054**

| Class | Files | Windows |
|---|---|---|
| `background_animals` | 2,140 | 5,770 |
| `background_wind_rain` | 680 | 1,360 |
| `threat_chainsaw` | 529 | 2,550 |
| `threat_dog` | 1,040 | 1,080 |
| `threat_gunshot` | 2,400 | 2,400 |
| `threat_human` | 1,040 | 3,814 |
| `threat_vehicle` | 1,040 | 1,080 |

### Step 4 — Train / Val / Test Split

Windows are split **60 / 20 / 20** stratified by class (each class keeps its proportion in every split) using seed 42:

| Split | Windows |
|---|---|
| Train | 10,232 |
| Val | 3,411 |
| Test | 3,411 |

Stratification ensures no class is over- or under-represented in any split.

### Step 5 — Augmentation (train split only)

For every original training window, **6 augmented copies** are generated:

| Augmentation | What it does | Purpose |
|---|---|---|
| Time-shift ±200 ms | Rolls the waveform forward or backward | Model is not position-dependent |
| Gain ±6 dB | Makes clip louder or quieter | Handles varying mic distances in the field |
| Pitch shift ±2 semitones | Raises or lowers pitch | Same chainsaw sounds different at different RPMs |
| Noise @ 5 dB SNR | Mixes with heavy wind/rain background | Hard field conditions |
| Noise @ 10 dB SNR | Mixes with moderate background | Typical field conditions |
| Noise @ 20 dB SNR | Mixes with light background | Good recording conditions |

The noise source is real `background_wind_rain` clips (100 clips randomly sampled at startup).

**After augmentation:**

| Split | Original | After augmentation |
|---|---|---|
| Train | 10,232 | **71,624** (×7) |
| Val | 3,411 | 3,411 (unchanged) |
| Test | 3,411 | 3,411 (unchanged) |

Val and test are never augmented to ensure honest evaluation.

### Step 6 — Feature Extraction

Two feature types are extracted from every window:

#### Log-Mel Spectrogram
A 2D representation of sound — like a photograph of the audio:
- Shape: **(128, 259)** — 128 frequency bins × 259 time frames for a 3 s clip
- Frequency axis uses the mel scale, which mimics human hearing (more detail at low frequencies)
- Values are in decibels (log scale)
- Used by: **CNN from scratch** and **tiny-AST**

```
Frequency (mel) ↑
128 bins        |  [darker = louder]
                |  ░░▓▓▓▓░░░░░▓▓▓▓▓░░░░
                └─────────────────────→ Time (259 frames)
```

#### MFCC + Δ + ΔΔ
A compact description of how the spectral shape changes over time:
- **MFCC (40 rows):** snapshot of timbre at each frame
- **Δ (40 rows):** rate of change (first derivative)
- **ΔΔ (40 rows):** acceleration of change (second derivative)
- Shape: **(120, 259)**
- Used by: **One-Class SVM** and **Convolutional Autoencoder**

### Step 7 — Shard Writing

Features are saved in compressed `.npz` batches of 1,000 samples each (shards). Each shard contains:

```python
shard_000.npz
  X     → shape (1000, 128, 259)  # feature arrays
  y     → shape (1000,)           # integer labels 0–6
  meta  → shape (1000,)           # "class|source_file" strings
```

---

## Output Structure

```
data/processed/
  mel/
    train/   shard_000.npz … shard_071.npz   (72 shards, 71,624 samples, 7.1 GB)
    val/     shard_000.npz … shard_003.npz   (4 shards,   3,411 samples)
    test/    shard_000.npz … shard_003.npz   (4 shards,   3,411 samples)
  mfcc/
    train/   shard_000.npz … shard_071.npz   (72 shards, 71,624 samples)
    val/     shard_000.npz … shard_003.npz   (4 shards,   3,411 samples)
    test/    shard_000.npz … shard_003.npz   (4 shards,   3,411 samples)
  manifest.json                               (run parameters + SHA256)
  manifest.csv                                (per-file metadata)
```

**Total disk usage: ~17 GB**

---

## manifest.json

Records every parameter used so results are fully reproducible:

```json
{
  "seed": 42,
  "sample_rate": 44100,
  "clip_seconds": 3.0,
  "hop_seconds": 1.5,
  "n_mels": 128,
  "n_mfcc": 40,
  "augmentation": true,
  "label_map": { "background_animals": 0, ... },
  "splits": {
    "train": { "original": 10232, "total": 71624 },
    "val":   { "original": 3411,  "total": 3411  },
    "test":  { "original": 3411,  "total": 3411  }
  },
  "script_sha256": "ef07cc..."
}
```

The `script_sha256` hash ensures that if the preprocessing script ever changes, you can detect it and know the processed data may differ.

---

## How to Re-run

```bash
# Full run with augmentation (recommended)
/opt/anaconda3/bin/python scripts/audio_preprocessing.py

# Without augmentation (faster, for debugging)
/opt/anaconda3/bin/python scripts/audio_preprocessing.py --no-aug

# Smaller shards (useful if RAM is limited)
/opt/anaconda3/bin/python scripts/audio_preprocessing.py --shard-size 500
```

> Re-running will overwrite existing shards in `data/processed/`.

---

## Known Issues Fixed

| Issue | Fix |
|---|---|
| `soundfile` reads MP3 files as 0 seconds on Python 3.13 | MP3 files now decoded via `ffmpeg` subprocess |
| `librosa.load` fails on Python 3.13 (`aifc` module removed) | Avoided for MP3; only used for pitch-shift augmentation |
