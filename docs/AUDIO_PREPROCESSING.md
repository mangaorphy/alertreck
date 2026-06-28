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
bridge between raw audio collection (Stage 1) and model training (Stage 3+). Two design choices define
it: a **group-aware split** that prevents recording leakage, and a **three-phase augmentation
curriculum (A → B → C)** of increasing difficulty, each written to its own output path so training can
ramp difficulty across phases.

```
dataset/  (11,333 files)
        │
        ▼  audio_preprocessing.py
        │
   ┌────┴────────────────────────────────────────────────────┐
   │ GROUP-AWARE 60/20/20 split (splits.json, seed 42)        │
   │ load_mono → EBU R128 normalise → class-dependent windows │
   └────┬────────────────────────────────────────────────────┘
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
| `background_animals` | 0 | Background | 2,139 |
| `background_wind_rain` | 1 | Background | 2,000 |
| `threat_chainsaw` | 2 | Threat | 568 |
| `threat_dog` | 3 | Threat context | 1,040 |
| `threat_gunshot` | 4 | Threat | 3,304 |
| `threat_human` | 5 | Threat | 1,242 |
| `threat_vehicle` | 6 | Threat | 1,040 |

Total: **11,333 files**. At inference, labels 2–6 trigger an alert; labels 0–1 do not.

---

## Pipeline Steps

### Step 1 — Load Audio (`load_mono`)
- Decodes any format (MP3/WAV/FLAC/OGG/M4A): MP3 via an `ffmpeg` subprocess, others via `soundfile`,
  then resampled with librosa `kaiser_best` — this avoids the `resampy` failure modes that silently
  drop files.
- Stereo → mono by channel averaging; resampled to **44,100 Hz** mono float32.

### Step 2 — EBU R128 Loudness Normalisation (`ebu_r128_normalize`)
- Scales each clip to **−23 dBFS** (target RMS ≈ 0.0708) and clips to [−1, 1].
- Makes the model invariant to recording volume / mic distance.

### Step 3 — Windowing (class-dependent)
The 3 s windowing strategy depends on the class, because impulsive and continuous sounds have different
weak-label risks:

- **Impulsive classes — `threat_gunshot`, `threat_dog`** → **event-based selection**
  (`select_event_windows`). A gunshot is a ~200 ms spike inside a long clip; blind slicing would label
  seconds of silence as "gunshot." Instead: high-pass at 180 Hz (ignore hum/rumble), measure energy in
  50 ms frames, take the clip's **background floor = 20th percentile**, mark frames **≥ 8 dB above
  floor** as events, merge onsets within 1 s, and emit one 3 s window **centred on each event**.
  Fallback: the single loudest window, so no clip is ever lost.
- **Continuous classes — animals, wind/rain, chainsaw, human, vehicle** → blind 3 s windows, 1.5 s hop
  (50 % overlap), zero-padded if < 3 s. The sound fills the clip, so every window is genuinely that
  class.

This mirrors the edge onset logic in [`alertrack/audio/onset.py`](alertrack/audio/onset.py) and directly
reduces false positives by removing mislabelled silent windows.

### Step 4 — Group-Aware 60/20/20 Split (`splits.json`)
- The split is **stratified by class** and assigned by **parent recording**, not by file
  (`scripts/grouping.py`). Many clips are segments of one recording (e.g. `gunshots122_00013_1` and
  `gunshots122_00009_1`; UrbanSound8K slices sharing a Freesound ID; `ds02 …_part_N` chunks).
  `group_key()` maps each file to its source recording, and `group_aware_split()` keeps **every group
  entirely within one split**.
- **Why it matters:** an earlier file-level split leaked recordings across train/test, inflating
  `threat_gunshot` F1 to 0.999. The group-aware split removed it — the honest test scores are lower but
  reflect true field generalisation (confirmed on-device).
- Stable across runs (seed 42), persisted to `data/processed/splits.json`. Regenerate independently with
  [`scripts/regenerate_splits.py`](scripts/regenerate_splits.py), which prints a before/after leakage
  report.

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
| 6a — SpecAugment (mel) | 2 time masks (≤40 frames) + 2 freq masks (≤20 bins) — also applied to clean train |
| 6b — Compound effect pool | `noise │ rir │ lowpass │ mp3 │ gain │ clip` (rir ⊕ clip are mutually exclusive) |
| 6c — FilterAugment | ±6 dB smooth random frequency curve over mel bins |
| 6d — mixup | Applied in the DataLoader at train time, not written to disk |

---

## The Three-Phase Curriculum

This is the core of the pipeline. Each phase applies progressively harder augmentation and emits **more
copies per clean window**, to its own output directory. Training ramps through them (Phase A → B → C).

| Phase | Effects/clip | Severity | Noise SNR | Copies / window | Output path | Windows | Shards |
|---|---|---|---|---|---|---|---|
| **A** | 1–2 | 0.3 | 15–30 dB | ×1 | `…/train_aug_A/` | 14,854 | 15 |
| **B** | 2–4 | 0.5 | 10–15 dB | ×2 | `…/train_aug_B/` | 29,708 | 30 |
| **C** | 2–5 | 1.0 | 5–10 dB | ×3 | `…/train_aug_C/` | 44,562 | 45 |

During training, each phase concatenates the **clean** train set with that phase's augmented set:

```
Phase A loader = train (clean)  +  train_aug_A     ( 14,854 + 14,854 )
Phase B loader = train (clean)  +  train_aug_B     ( 14,854 + 29,708 )
Phase C loader = train (clean)  +  train_aug_C     ( 14,854 + 44,562 )
```

So the model starts on mild conditions (Phase A: high SNR, few effects) and finishes on the hardest
(Phase C: 5–10 dB SNR, up to 5 stacked effects, 3× the data) — a difficulty curriculum.

---

## Split & Window Counts (`manifest.json`)

| Split | Clean windows | Shards |
|---|---|---|
| Train (clean) | 14,854 | 15 |
| Train aug A | 14,854 | 15 |
| Train aug B | 29,708 | 30 |
| Train aug C | 44,562 | 45 |
| Val | 5,844 | 6 |
| Test | 5,925 | 6 |

Total clean windows: **26,623** (14,854 train + 5,844 val + 5,925 test).
Shards are written in compressed `.npz` batches of 1,000 samples. The shard writer **purges stale shards
from any previous run** before writing, so a shorter run can never leave orphaned shards behind.

Per-class file split (60/20/20, group-aware):

| Class | Train | Val | Test |
|---|---|---|---|
| background_animals | 1,283 | 428 | 428 |
| background_wind_rain | 1,200 | 400 | 400 |
| threat_chainsaw | 341 | 114 | 113 |
| threat_dog | 624 | 208 | 208 |
| threat_gunshot | 1,982 | 661 | 661 |
| threat_human | 745 | 249 | 248 |
| threat_vehicle | 624 | 208 | 208 |

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
    train/         15 shards   (14,854 windows)
    train_aug_A/   15 shards   (14,854)
    train_aug_B/   30 shards   (29,708)
    train_aug_C/   45 shards   (44,562)
    val/            6 shards   ( 5,844)
    test/           6 shards   ( 5,925)
  mfcc/
    train/ train_aug_A/ train_aug_B/ train_aug_C/ val/ test/   (same shape, (120, 301))
  splits.json        # group-aware 60/20/20 assignment by parent recording (seed 42)
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
    "train": { "clean": 14854, "aug_A": 14854, "aug_B": 29708, "aug_C": 44562 },
    "val":   { "clean": 5844 },
    "test":  { "clean": 5925 }
  },
  "script_sha256": "9cd01d8a5bc32add02866989aa649f8429057b6e812bfa7349b1c61b038449d1"
}
```

The `script_sha256` lets you detect if the preprocessing script changed (and therefore that the
processed data may differ).

---

## How to Re-run

```bash
# (Optional) regenerate the group-aware split + leakage report, without re-sharding
/opt/anaconda3/bin/python scripts/regenerate_splits.py

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

> Re-running overwrites existing shards in `data/processed/` (the writer purges each directory first).
> After regenerating, re-upload the Kaggle datasets (mel + mfcc + splits.json) before retraining.

---

## Notes

- **Group-aware split** (not file- or window-level) is deliberate — segments of the same parent
  recording never span splits, which is the difference between an honest test set and silent leakage
  that inflated `threat_gunshot` F1 to 0.999.
- **Event-based windowing** for impulsive classes removes silent windows that blind slicing would
  mislabel as a threat — a direct false-positive reduction.
- **EBU R128, not RMS-to-0.1** — the edge preprocessor matches this exactly. Any deviation (window
  length, normalisation, or an extra filter) is train/serve skew and degrades live accuracy.
- The deployment path adds **one** deliberate, training-absent step: a 50/60 Hz mains-hum high-pass to
  counter field-mic hum (see [DEPLOYMENT.md](DEPLOYMENT.md) Part 6A) — it removes content absent from
  training, so it reduces skew rather than adding it.
