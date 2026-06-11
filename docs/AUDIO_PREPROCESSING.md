# Audio Preprocessing Pipeline — Stage 2

**Script:** [`scripts/audio_preprocessing.py`](scripts/audio_preprocessing.py)
**Output:** `data/processed/`
**Reproducibility:** seed 42 · `script_sha256` recorded in `manifest.json`

> 📦 **Raw audio (`dataset/`) and processed shards (`data/processed/`) are on Google Drive:**
> [Alertreck Data, Dataset & Models](https://drive.google.com/drive/folders/1U9BwIUNQ8Snl5RxR8LHthWfdOc_EdcTM?usp=sharing).
> Download and place them at the repo root to re-run or inspect the pipeline.

---

## Overview

This pipeline converts raw audio in `dataset/` into machine-learning-ready feature shards. It is the
bridge between raw audio collection (Stage 1) and model training (Stage 3+). The defining feature is a
**three-phase augmentation curriculum (A → B → C)** of increasing difficulty, each written to its own
output path so training can ramp difficulty across phases.

```
dataset/  (8,907 files)
        │
        ▼  audio_preprocessing.py
        │
   ┌────┴───────────────────────────────────────────────┐
   │ load_mono → EBU R128 normalise → 3 s windowing      │
   │ → FILE-LEVEL 60/20/20 split (splits.json, seed 42)  │
   └────┬───────────────────────────────────────────────┘
        │
   ┌────┴─────────────┬──────────────────────────────────┐
   ▼                  ▼                                   ▼
 clean              augmentation curriculum            val / test
 (train/val/test)   (train only, phases A/B/C)         (clean only — never augmented)
        │                  │
        ▼                  ▼
   extract mel + mfcc   per-phase aug → mel + mfcc
        │                  │
        ▼                  ▼
   data/processed/mel/{train,val,test}      data/processed/mel/train_aug_{A,B,C}
   data/processed/mfcc/{train,val,test}     data/processed/mfcc/train_aug_{A,B,C}
```

---

## Fixed Parameters

These match `manifest.json` exactly, and the edge preprocessor in
[`alertrack/audio/preprocess.py`](alertrack/audio/preprocess.py) reproduces them at inference.

| Parameter | Value | Reason |
|---|---|---|
| Sample rate | 44,100 Hz | CD-quality; consistent across all sources |
| Clip length | 3.0 s (132,300 samples) | Captures a gunshot echo or chainsaw burst |
| Window hop | 1.5 s (50 % overlap) | More windows per file; detects events at any position |
| Mel bins | 128 | Frequency resolution for spectrograms |
| MFCC coefficients | 40 (+Δ +ΔΔ → 120 rows) | Compact timbre + dynamics |
| FFT size (`n_fft`) | 2,048 (~46 ms) | Frequency resolution |
| STFT window (`win_length`) | 1,102 (25 ms @ 44.1 kHz), Hann | Matches training feature extraction |
| STFT hop (`hop_stft`) | 441 (10 ms @ 44.1 kHz) | → **301 time frames** per 3 s clip |
| Loudness norm | **EBU R128 → −23 dBFS**, clipped to [−1, 1] | Volume-invariant; a quiet and loud gunshot look the same |
| Mel scaling | `power_to_db(S, ref=np.max)` | dB relative to clip max |
| Random seed | 42 | Reproducible splits and augmentation |

> **Frame count:** `1 + floor(132300 / 441) = 301` (librosa `center=True`). All mel/MFCC shards are
> `(128, 301)` / `(120, 301)`. This 301 is the input width the trained models expect.

---

## Class Labels

| Folder | Label | Kind | Files |
|---|---|---|---|
| `background_animals` | 0 | Background | 2,140 |
| `background_wind_rain` | 1 | Background | 680 |
| `threat_chainsaw` | 2 | Threat | 567 |
| `threat_dog` | 3 | Threat context | 1,040 |
| `threat_gunshot` | 4 | Threat | 2,400 |
| `threat_human` | 5 | Threat | 1,040 |
| `threat_vehicle` | 6 | Threat | 1,040 |

> At inference, labels 2–6 trigger an alert; labels 0–1 do not.

---

## Pipeline Steps

### Step 1 — Load Audio (`load_mono`)
- Decodes any format (MP3/WAV/FLAC/OGG) via an `ffmpeg` subprocess to 16/44.1 kHz mono float32 — this
  avoids the `resampy`/`soundfile` failure modes that silently drop non-MP3 files.
- Stereo → mono by channel averaging; resampled to 44,100 Hz.

### Step 2 — EBU R128 Loudness Normalisation (`ebu_r128_normalize`)
- Scales each clip to **−23 dBFS** (target RMS ≈ 0.0708) and clips to [−1, 1].
- Makes the model invariant to recording volume / mic distance.

### Step 3 — Windowing
- Each file is sliced into 3 s windows with 50 % overlap (hop 1.5 s). Files < 3 s are zero-padded to one
  window. A 10 s file → 6 windows.

### Step 4 — File-Level 60/20/20 Split (`splits.json`)
- The split is **stratified by class at the file level**, not the window level. All windows from one file
  stay in the same split — this prevents leakage (a model seeing window 1 in train and window 2 in test).
- Stable across runs (seed 42), persisted to `data/processed/splits.json`.

### Step 4b — Feature Extraction (per window)
| Feature | Shape | Contents | Used by |
|---|---|---|---|
| Log-mel spectrogram | **(128, 301)** | mel power → dB (`ref=np.max`) | CNN (03a), ProtoNet (03b), Conv-AE (04b) |
| MFCC + Δ + ΔΔ | **(120, 301)** | 40 MFCC + 40 Δ + 40 ΔΔ | OC-SVM (04c) |

> The W2V2-L2 model (04a) does **not** use these shards — it consumes raw audio through the
> `wav2vec2-base` backbone (see [`notebooks/02b-prepare-w2v2-embeddings.ipynb`](notebooks/02b-prepare-w2v2-embeddings.ipynb)).

### Steps 5–6 — Augmentation (train split only)
Applied only when `--aug-phase` is passed. Val/test are **never** augmented.

| Step | What |
|---|---|
| 5 — DIR calibration (optional) | Convolve with a USB-mic sweep-tone impulse response (`--dir-ir`) to match deployment mic colour |
| 6a — SpecAugment (mel) | 2 time masks (≤40 frames) + 2 freq masks (≤20 bins) |
| 6b — Compound effect pool | `noise │ rir │ lowpass │ mp3 │ gain │ clip` (rir ⊕ clip are mutually exclusive) |
| 6c — FilterAugment | ±6 dB smooth random frequency curve over mel bins |
| 6d — mixup | Applied in the DataLoader at train time, not written to disk |

---

## The Three-Phase Curriculum

This is the core of the pipeline. Each phase applies progressively harder augmentation and emits **more
copies per clean window**, to its own output directory. Training ramps through them (Phase A → B → C).

| Phase | Effects/clip | Severity | Noise SNR | Copies / window | Output path | Windows | Shards |
|---|---|---|---|---|---|---|---|
| **A** | 1–2 | 0.3 | ≥ 15 dB | ×1 | `…/train_aug_A/` | 10,223 | 11 |
| **B** | 2–4 | 0.5 | 10–15 dB | ×2 | `…/train_aug_B/` | 20,446 | 21 |
| **C** | 2–5 | 1.0 | 5–10 dB | ×3 | `…/train_aug_C/` | 30,669 | 31 |

During training, each phase concatenates the **clean** train set with that phase's augmented set:

```
Phase A loader = train (clean)  +  train_aug_A     ( 10,223 + 10,223 )
Phase B loader = train (clean)  +  train_aug_B     ( 10,223 + 20,446 )
Phase C loader = train (clean)  +  train_aug_C     ( 10,223 + 30,669 )
```

So the model starts on mild conditions (Phase A: high SNR, few effects) and finishes on the hardest
(Phase C: 5–10 dB SNR, up to 5 stacked effects, 3× the data) — a difficulty curriculum.

---

## Split & Window Counts (`manifest.json`)

| Split | Clean windows | Shards |
|---|---|---|
| Train (clean) | 10,223 | 11 |
| Train aug A | 10,223 | 11 |
| Train aug B | 20,446 | 21 |
| Train aug C | 30,669 | 31 |
| Val | 3,392 | 4 |
| Test | 3,439 | 4 |

Total clean windows: **17,054** (10,223 train + 3,392 val + 3,439 test).
Shards are written in compressed `.npz` batches of 1,000 samples.

```python
shard_000.npz
  X     → (≤1000, 128, 301)   # mel  (or (≤1000, 120, 301) for mfcc)
  y     → (≤1000,)            # integer labels 0–6
  meta  → (≤1000,)            # per-window source metadata
```

---

## Output Structure

```
data/processed/
  mel/
    train/         11 shards   (10,223 windows)
    train_aug_A/   11 shards   (10,223)
    train_aug_B/   21 shards   (20,446)
    train_aug_C/   31 shards   (30,669)
    val/            4 shards   ( 3,392)
    test/           4 shards   ( 3,439)
  mfcc/
    train/ train_aug_A/ train_aug_B/ train_aug_C/ val/ test/   (same shape, (120, 301))
  splits.json        # stable file-level 60/20/20 assignment (seed 42)
  manifest.json      # all run parameters + script SHA256
```

---

## manifest.json (actual)

```json
{
  "seed": 42,
  "sample_rate": 44100,
  "clip_seconds": 3.0,
  "hop_seconds": 1.5,
  "n_mels": 128,
  "n_mfcc": 40,
  "n_fft": 2048,
  "win_length": 1102,
  "hop_stft": 441,
  "ebu_target_dbfs": -23,
  "spec_augment_train": true,
  "filter_augment_aug": true,
  "curriculum_phases": ["A", "B", "C"],
  "dir_calibration_ir": null,
  "splits": {
    "train": { "clean": 10223, "aug_A": 10223, "aug_B": 20446, "aug_C": 30669 },
    "val":   { "clean": 3392 },
    "test":  { "clean": 3439 }
  },
  "script_sha256": "9922489ab8ebef71948d6bd9bc9788f93740b6fcf552b3338cdf5182fcab18de"
}
```

The `script_sha256` lets you detect if the preprocessing script changed (and therefore that the
processed data may differ).

---

## How to Re-run

```bash
# Clean splits only (no augmentation) — fast, for debugging
/opt/anaconda3/bin/python scripts/audio_preprocessing.py

# Full run: clean + all three curriculum phases (what the models were trained on)
/opt/anaconda3/bin/python scripts/audio_preprocessing.py --aug-phase A B C

# A single phase (e.g. regenerate Phase B only)
/opt/anaconda3/bin/python scripts/audio_preprocessing.py --aug-phase B

# With deployment-mic impulse response (DIR calibration)
/opt/anaconda3/bin/python scripts/audio_preprocessing.py --aug-phase A B C --dir-ir usb_mic_ir.wav

# Smaller shards if RAM is limited
/opt/anaconda3/bin/python scripts/audio_preprocessing.py --aug-phase A B C --shard-size 500
```

> Re-running overwrites existing shards in `data/processed/`. After regenerating, re-upload the Kaggle
> datasets (`alertreck-mel2`, `alertreck-mfcc`) before retraining.

---

## Notes

- **File-level split** (not window-level) is deliberate — it is the difference between an honest test
  set and silent leakage between near-identical overlapping windows.
- **EBU R128, not RMS-to-0.1** — the edge preprocessor matches this exactly. Any deviation (window
  length, normalisation, or an extra filter) is train/serve skew and degrades live accuracy.
- The deployment path adds **one** deliberate, training-absent step: a 50/60 Hz mains-hum high-pass to
  counter field-mic hum (see [DEPLOYMENT.md](DEPLOYMENT.md) Part 6A) — it removes content absent from
  training, so it reduces skew rather than adding it.
