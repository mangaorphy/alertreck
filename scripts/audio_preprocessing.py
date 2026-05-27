"""
Stage 2 — Audio Preprocessing Pipeline (Compound Augmentation + Curriculum)
==========================================================================
Spec (ROADMAP §3.1, Proposal §2.3.7):
  Step 1  Resample → 44.1 kHz mono (Kaiser-best)
  Step 2  Normalise → −23 dBFS EBU R128
  Step 3  Segment 3 s clips, 1.5 s hop (50 % overlap)
  Step 4a 128-bin log-mel  (win=1102, hop=441, n_fft=2048, Hann)
  Step 4b MFCC + Δ + ΔΔ  (40 → 120 rows)
  Step 5  DIR calibration (optional --dir-ir WAV)  — sweep-tone IR of deployment USB mic
  Step 6a SpecAugment on mel — 2 time masks (max 40 fr) + 2 freq masks (max 20 bins)
  Step 6b Compound augmentation pool — noise | rir | lowpass | mp3 | gain | clip
           Physical constraint: rir ⊕ clip are mutually exclusive
  Step 6c FilterAugment — ±6 dB smooth random freq curve on mel bins
  Step 6d mixup_batch() — call from DataLoader, not here
  Step 6e Learnability filter — stub; active when --conv-ae supplied

Curriculum phases (ROADMAP §3.3):
  A  1–2 effects, severity 0.3, SNR ≥15 dB,   1 aug copy / window
  B  2–4 effects, severity 0.5, SNR 10–15 dB,  2 aug copies / window
  C  2–5 effects, severity 1.0, SNR 5–10 dB,   3 aug copies / window

Output:
  data/processed/mel/{train,val,test}/shard_NNN.npz
  data/processed/mel/train_aug_{A,B,C}/shard_NNN.npz   (generated per --aug-phase)
  data/processed/mfcc/{same}
  data/processed/splits.json    (stable file-level 60/20/20, seed 42)
  data/processed/manifest.json

Usage:
  python3 scripts/audio_preprocessing.py                      # clean splits only
  python3 scripts/audio_preprocessing.py --aug-phase A B C   # + all curriculum phases
  python3 scripts/audio_preprocessing.py --aug-phase B       # Phase B only
  python3 scripts/audio_preprocessing.py --dir-ir usb_mic_ir.wav  # record with: arecord -D plughw:1,0 -d 10 sweep.wav
  python3 scripts/audio_preprocessing.py --shard-size 500
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
SEED         = 42
SR           = 44_100
CLIP_LEN     = 3.0
HOP_LEN      = 1.5
CLIP_SAMPLES = int(SR * CLIP_LEN)   # 132 300
HOP_SAMPLES  = int(SR * HOP_LEN)    #  66 150

N_MELS      = 128
N_MFCC      = 40
N_FFT       = 2048
WIN_LENGTH  = 1102                  # 25 ms @ 44.1 kHz
HOP_STFT    = 441                   # 10 ms @ 44.1 kHz

EBU_TARGET  = 10 ** (-23.0 / 20.0) # −23 dBFS ≈ 0.07079

FOLDER_TO_LABEL: dict[str, int] = {
    "background_animals":   0,
    "background_wind_rain": 1,
    "threat_chainsaw":      2,
    "threat_dog":           3,
    "threat_gunshot":       4,
    "threat_human":         5,
    "threat_vehicle":       6,
}
LABEL_NAMES = {v: k for k, v in FOLDER_TO_LABEL.items()}
AUDIO_EXTS  = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
OUT_ROOT    = REPO_ROOT / "data" / "processed"
SPLITS_JSON = OUT_ROOT / "splits.json"

PHASE_CFG: dict[str, dict] = {
    "A": {"min_fx": 1, "max_fx": 2, "severity": 0.3, "snr_lo": 15, "snr_hi": 30, "copies": 1},
    "B": {"min_fx": 2, "max_fx": 4, "severity": 0.5, "snr_lo": 10, "snr_hi": 15, "copies": 2},
    "C": {"min_fx": 2, "max_fx": 5, "severity": 1.0, "snr_lo":  5, "snr_hi": 10, "copies": 3},
}

# ── worker-process globals (populated by _init_worker) ────────────────────────
_g_noise_pool: list[np.ndarray] = []
_g_ir: np.ndarray | None = None


def _init_worker(noise_paths: list[str], ir_path: str | None) -> None:
    global _g_noise_pool, _g_ir
    random.seed(SEED)
    np.random.seed(SEED)
    _g_noise_pool.clear()
    for p in noise_paths:
        a = load_mono(Path(p))
        if a is not None:
            _g_noise_pool.append(ebu_r128_normalize(a))
    _g_ir = load_mono(Path(ir_path)) if ir_path else None


# ── audio I/O ─────────────────────────────────────────────────────────────────

def load_mono(path: Path) -> np.ndarray | None:
    try:
        if path.suffix.lower() == ".mp3":
            cmd = ["ffmpeg", "-i", str(path), "-f", "f32le",
                   "-ar", str(SR), "-ac", "1", "pipe:1", "-loglevel", "quiet"]
            result = subprocess.run(cmd, capture_output=True)
            if not result.stdout:
                return None
            return np.frombuffer(result.stdout, dtype=np.float32).copy()
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR,
                                     res_type="kaiser_best")
        return audio.astype(np.float32)
    except Exception:
        return None


def ebu_r128_normalize(audio: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    return np.clip(audio * (EBU_TARGET / rms), -1.0, 1.0).astype(np.float32)


def slice_windows(audio: np.ndarray) -> list[np.ndarray]:
    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    wins, start = [], 0
    while start + CLIP_SAMPLES <= len(audio):
        wins.append(audio[start: start + CLIP_SAMPLES].copy())
        start += HOP_SAMPLES
    return wins


# ── DIR calibration ───────────────────────────────────────────────────────────

def dir_calibrate(audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
    out = np.convolve(audio, ir, mode="full")[: len(audio)]
    return out.astype(np.float32)


# ── effect implementations (numpy + librosa + ffmpeg; no scipy) ───────────────

def _sinc_lowpass(cutoff_hz: float, num_taps: int = 101) -> np.ndarray:
    if num_taps % 2 == 0:
        num_taps += 1
    fc = cutoff_hz / SR
    n  = np.arange(num_taps) - (num_taps - 1) / 2
    h  = np.sinc(2 * fc * n) * np.blackman(num_taps)
    h /= h.sum()
    return h.astype(np.float32)


def _apply_lowpass(audio: np.ndarray, severity: float) -> np.ndarray:
    # severity 0 → 22050 Hz (pass-through); severity 1 → 1000 Hz
    cutoff = max(500.0, (1.0 - severity) * (SR / 2) + severity * 1000.0)
    h = _sinc_lowpass(cutoff)
    return np.clip(np.convolve(audio, h, mode="same"), -1.0, 1.0).astype(np.float32)


def _synthetic_rir(rt60_s: float) -> np.ndarray:
    length = max(1, int(SR * rt60_s))
    t   = np.arange(length) / SR
    env = np.exp(-6.9 * t / rt60_s)
    rir = (np.random.randn(length) * env).astype(np.float32)
    peak = np.abs(rir).max()
    return rir / peak if peak > 1e-9 else rir


def _apply_rir(audio: np.ndarray, severity: float) -> np.ndarray:
    rt60 = 0.05 + severity * 0.45   # 50 ms (mild) → 500 ms (severe)
    rir  = _synthetic_rir(rt60)
    out  = np.convolve(audio, rir, mode="full")[: len(audio)]
    peak = np.abs(out).max()
    return (out / peak if peak > 1e-9 else out).astype(np.float32)


def _apply_mp3(audio: np.ndarray, severity: float) -> np.ndarray:
    bitrate  = int(128 - severity * 96)   # 128 kbps (mild) → 32 kbps (severe)
    in_path  = mp3_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            in_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
        sf.write(in_path, audio, SR, format="WAV", subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-b:a", f"{bitrate}k",
             mp3_path, "-loglevel", "quiet"],
            check=True, capture_output=True,
        )
        result = subprocess.run(
            ["ffmpeg", "-i", mp3_path, "-f", "f32le",
             "-ar", str(SR), "-ac", "1", "pipe:1", "-loglevel", "quiet"],
            capture_output=True,
        )
        dec = np.frombuffer(result.stdout, dtype=np.float32)
        if len(dec) >= len(audio):
            return dec[: len(audio)].copy()
        return np.pad(dec, (0, len(audio) - len(dec))).copy()
    except Exception:
        return audio.copy()
    finally:
        for p in [in_path, mp3_path]:
            if p:
                Path(p).unlink(missing_ok=True)


def _apply_noise(audio: np.ndarray, severity: float,
                 snr_lo: float, snr_hi: float) -> np.ndarray:
    snr_db = random.uniform(snr_lo, snr_hi)
    if _g_noise_pool and random.random() < 0.7:
        bg = random.choice(_g_noise_pool)
        if len(bg) < len(audio):
            bg = np.tile(bg, int(np.ceil(len(audio) / len(bg))))
        bg = ebu_r128_normalize(bg[: len(audio)])
    else:
        bg = np.random.randn(len(audio)).astype(np.float32)
    sig_p = np.mean(audio ** 2) + 1e-10
    noi_p = np.mean(bg    ** 2) + 1e-10
    scale = np.sqrt(sig_p / (noi_p * 10 ** (snr_db / 10)))
    return np.clip(audio + bg * scale, -1.0, 1.0).astype(np.float32)


def _apply_gain(audio: np.ndarray, severity: float) -> np.ndarray:
    db = random.uniform(-severity * 12.0, severity * 12.0)
    return np.clip(audio * 10 ** (db / 20), -1.0, 1.0).astype(np.float32)


def _apply_clip(audio: np.ndarray, severity: float) -> np.ndarray:
    threshold = max(0.1, 1.0 - severity * 0.6)
    return np.clip(audio, -threshold, threshold).astype(np.float32)


# ── compound augmentation ─────────────────────────────────────────────────────

_FX_POOL      = ["noise", "rir", "lowpass", "mp3", "gain", "clip"]
_INCOMPATIBLE = frozenset([("rir", "clip"), ("clip", "rir")])


def _sample_fx(min_fx: int, max_fx: int) -> list[str]:
    n    = random.randint(min_fx, min(max_fx, len(_FX_POOL)))
    pool = list(_FX_POOL)
    random.shuffle(pool)
    chosen: list[str] = []
    for fx in pool:
        if len(chosen) >= n:
            break
        if any((fx, ex) in _INCOMPATIBLE for ex in chosen):
            continue
        chosen.append(fx)
    return chosen


def compound_augment(audio: np.ndarray, phase: str) -> list[np.ndarray]:
    cfg  = PHASE_CFG[phase]
    sev  = cfg["severity"]
    outs = []
    for _ in range(cfg["copies"]):
        out = audio.copy()
        for fx in _sample_fx(cfg["min_fx"], cfg["max_fx"]):
            if   fx == "noise":   out = _apply_noise(out, sev, cfg["snr_lo"], cfg["snr_hi"])
            elif fx == "rir":     out = _apply_rir(out, sev)
            elif fx == "lowpass": out = _apply_lowpass(out, sev)
            elif fx == "mp3":     out = _apply_mp3(out, sev)
            elif fx == "gain":    out = _apply_gain(out, sev)
            elif fx == "clip":    out = _apply_clip(out, sev)
        outs.append(out)
    return outs


# ── feature extraction ────────────────────────────────────────────────────────

def extract_mel(audio: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_STFT,
        win_length=WIN_LENGTH, window="hann",
        fmax=SR // 2,
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def extract_mfcc_delta(audio: np.ndarray) -> np.ndarray:
    m  = librosa.feature.mfcc(
        y=audio, sr=SR, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_STFT,
        win_length=WIN_LENGTH,
    )
    d  = librosa.feature.delta(m)
    dd = librosa.feature.delta(m, order=2)
    return np.concatenate([m, d, dd], axis=0).astype(np.float32)  # (120, T)


# ── spectrogram augmentations ─────────────────────────────────────────────────

def spec_augment(spec: np.ndarray,
                 n_time: int = 2, t_max: int = 40,
                 n_freq: int = 2, f_max: int = 20) -> np.ndarray:
    out  = spec.copy()
    fill = float(out.mean())
    F, T = out.shape
    for _ in range(n_freq):
        f  = random.randint(0, f_max)
        f0 = random.randint(0, max(0, F - f))
        out[f0: f0 + f, :] = fill
    for _ in range(n_time):
        t  = random.randint(0, t_max)
        t0 = random.randint(0, max(0, T - t))
        out[:, t0: t0 + t] = fill
    return out


def filter_augment(mel: np.ndarray, n_ctrl: int = 5, max_db: float = 6.0) -> np.ndarray:
    F     = mel.shape[0]
    ctrl  = np.random.uniform(-max_db, max_db, n_ctrl)
    curve = np.interp(np.arange(F), np.linspace(0, F - 1, n_ctrl), ctrl)
    return (mel + curve[:, None]).astype(np.float32)


# ── mixup utility (call from DataLoader, not here) ────────────────────────────

def mixup_batch(X1: np.ndarray, y1: np.ndarray,
                X2: np.ndarray, y2: np.ndarray,
                alpha: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    lam = np.random.beta(alpha, alpha)
    return lam * X1 + (1 - lam) * X2, lam * y1 + (1 - lam) * y2


# ── learnability filter stub ──────────────────────────────────────────────────

def learnability_ok(audio: np.ndarray) -> bool:
    return True  # stub: pass all clips until Conv-AE (Stage 4a) is trained


# ── worker functions ──────────────────────────────────────────────────────────

def _process_file(args: tuple) -> list[dict]:
    cls, path_str, do_dir, do_specaug = args
    audio = load_mono(Path(path_str))
    if audio is None:
        return []
    audio = ebu_r128_normalize(audio)
    if do_dir and _g_ir is not None:
        audio = ebu_r128_normalize(dir_calibrate(audio, _g_ir))

    label = FOLDER_TO_LABEL[cls]
    meta  = f"{cls}|{Path(path_str).relative_to(REPO_ROOT)}"
    out   = []
    for win in slice_windows(audio):
        mel  = extract_mel(win)
        mfcc = extract_mfcc_delta(win)
        if do_specaug:
            mel = spec_augment(mel)
        out.append({"mel": mel, "mfcc": mfcc, "label": label, "meta": meta})
    return out


def _process_file_aug(args: tuple) -> list[dict]:
    cls, path_str, phase, do_dir = args
    audio = load_mono(Path(path_str))
    if audio is None:
        return []
    audio = ebu_r128_normalize(audio)
    if do_dir and _g_ir is not None:
        audio = ebu_r128_normalize(dir_calibrate(audio, _g_ir))

    label = FOLDER_TO_LABEL[cls]
    meta  = f"{cls}|{Path(path_str).relative_to(REPO_ROOT)}|aug_{phase}"
    out   = []
    for win in slice_windows(audio):
        for aug in compound_augment(win, phase):
            if not learnability_ok(aug):
                continue
            mel  = extract_mel(aug)
            mfcc = extract_mfcc_delta(aug)
            mel  = spec_augment(mel)
            mel  = filter_augment(mel)
            out.append({"mel": mel, "mfcc": mfcc, "label": label, "meta": meta})
    return out


# ── dataset collection ────────────────────────────────────────────────────────

def collect_files() -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = defaultdict(list)
    for folder in sorted(DATASET_DIR.iterdir()):
        if not folder.is_dir():
            continue
        cls = folder.name
        if cls not in FOLDER_TO_LABEL:
            print(f"  skipping unmapped folder: {cls}")
            continue
        found = sorted(f for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        files[cls].extend(found)
        print(f"  {cls:<26}  {len(found):>5} files")
    return files


def make_splits(files: dict[str, list[Path]]) -> dict[str, list[tuple[str, Path]]]:
    if SPLITS_JSON.exists():
        data      = json.loads(SPLITS_JSON.read_text())
        all_paths = [p for recs in data.values() for _, p in recs]
        if all(Path(p).exists() for p in all_paths):
            print("  Loaded stable splits from splits.json")
            return {s: [(c, Path(p)) for c, p in recs] for s, recs in data.items()}
        print("  splits.json stale (files moved?), rebuilding …")

    all_items = [(cls, p) for cls, paths in sorted(files.items()) for p in paths]
    strat     = [cls for cls, _ in all_items]
    idx       = list(range(len(all_items)))

    idx_tv, idx_te = train_test_split(idx, test_size=0.20,
                                      stratify=strat, random_state=SEED)
    idx_tr, idx_va = train_test_split(idx_tv, test_size=0.25,
                                      stratify=[strat[i] for i in idx_tv],
                                      random_state=SEED)

    splits = {
        "train": [all_items[i] for i in idx_tr],
        "val":   [all_items[i] for i in idx_va],
        "test":  [all_items[i] for i in idx_te],
    }
    SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON.write_text(json.dumps(
        {s: [(c, str(p)) for c, p in recs] for s, recs in splits.items()}, indent=2))
    print("  Saved stable splits to splits.json")
    return splits


# ── shard writer ──────────────────────────────────────────────────────────────

class ShardWriter:
    def __init__(self, mel_dir: Path, mfcc_dir: Path, shard_size: int):
        mel_dir.mkdir(parents=True, exist_ok=True)
        mfcc_dir.mkdir(parents=True, exist_ok=True)
        self._mel_dir  = mel_dir
        self._mfcc_dir = mfcc_dir
        self._size     = shard_size
        self._buf: list[dict] = []
        self._idx  = 0
        self.total = 0

    def add(self, rec: dict) -> None:
        self._buf.append(rec)
        self.total += 1
        if len(self._buf) >= self._size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        tag  = f"shard_{self._idx:03d}"
        mel  = np.stack([r["mel"]   for r in self._buf])
        mfcc = np.stack([r["mfcc"]  for r in self._buf])
        y    = np.array([r["label"] for r in self._buf], dtype=np.int8)
        meta = np.array([r["meta"]  for r in self._buf], dtype=object)
        np.savez_compressed(self._mel_dir  / f"{tag}.npz", X=mel,  y=y, meta=meta)
        np.savez_compressed(self._mfcc_dir / f"{tag}.npz", X=mfcc, y=y, meta=meta)
        self._buf.clear()
        self._idx += 1

    def close(self) -> None:
        self._flush()


# ── split processing ──────────────────────────────────────────────────────────

def process_split(name: str, items: list[tuple[str, Path]],
                  mel_base: Path, mfcc_base: Path,
                  shard_size: int, noise_paths: list[str],
                  ir_path: str | None, phases: list[str]) -> dict:
    do_dir     = ir_path is not None
    do_specaug = name == "train"
    n_workers  = min(4, os.cpu_count() or 4)

    # clean split
    writer     = ShardWriter(mel_base / name, mfcc_base / name, shard_size)
    clean_args = [(cls, str(p), do_dir, do_specaug) for cls, p in items]
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker,
                             initargs=(noise_paths, ir_path)) as ex:
        futs = {ex.submit(_process_file, a): a for a in clean_args}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"  {name:<12} clean"):
            for rec in fut.result():
                writer.add(rec)
    writer.close()
    counts = {"clean": writer.total}

    if name != "train" or not phases:
        return counts

    for phase in phases:
        aug_writer = ShardWriter(
            mel_base  / f"train_aug_{phase}",
            mfcc_base / f"train_aug_{phase}",
            shard_size,
        )
        aug_args = [(cls, str(p), phase, do_dir) for cls, p in items]
        cfg  = PHASE_CFG[phase]
        desc = (f"  train_aug_{phase}  "
                f"(fx {cfg['min_fx']}–{cfg['max_fx']}, "
                f"sev {cfg['severity']}, ×{cfg['copies']})")
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker,
                                 initargs=(noise_paths, ir_path)) as ex:
            futs = {ex.submit(_process_file_aug, a): a for a in aug_args}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
                for rec in fut.result():
                    aug_writer.add(rec)
        aug_writer.close()
        counts[f"aug_{phase}"] = aug_writer.total

    return counts


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Alertreck Stage 2 — compound augmentation + curriculum preprocessing")
    ap.add_argument("--aug-phase",  nargs="*", choices=["A", "B", "C"],
                    default=[], metavar="PHASE",
                    help="Curriculum phases to generate (A B C). Default: none (clean only).")
    ap.add_argument("--dir-ir",     type=Path, default=None,
                    help="Deployment USB microphone IR WAV for DIR calibration (step 5).")
    ap.add_argument("--conv-ae",    type=Path, default=None,
                    help="Trained Conv-AE ONNX for learnability filter stub (step 6e).")
    ap.add_argument("--shard-size", type=int,  default=1000)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    phases = args.aug_phase or []
    print("=== Alertreck Stage 2 — Compound Augmentation Pipeline ===")
    print(f"SR={SR} Hz  win={WIN_LENGTH} ({WIN_LENGTH/SR*1000:.1f} ms)  "
          f"hop={HOP_STFT} ({HOP_STFT/SR*1000:.1f} ms)  "
          f"n_fft={N_FFT}  EBU={20*np.log10(EBU_TARGET):.0f} dBFS")
    print(f"Curriculum phases : {phases or 'none (clean splits only)'}")
    if args.conv_ae:
        print("Learnability filter: stub (wire after Stage 4a Conv-AE training)")
    print()

    # impulse response
    ir_path: str | None = None
    if args.dir_ir:
        test_ir = load_mono(args.dir_ir)
        if test_ir is None:
            print(f"WARNING: could not load IR {args.dir_ir} — skipping DIR calibration")
        else:
            ir_path = str(args.dir_ir)
            print(f"DIR IR: {args.dir_ir.name}  ({len(test_ir)/SR*1000:.1f} ms)\n")

    # noise pool paths (loaded inside each worker via _init_worker)
    noise_paths: list[str] = []
    noise_dir = DATASET_DIR / "background_wind_rain"
    if noise_dir.exists():
        nfiles      = sorted(f for f in noise_dir.iterdir()
                             if f.suffix.lower() in AUDIO_EXTS)
        sampled     = random.sample(nfiles, min(100, len(nfiles)))
        noise_paths = [str(f) for f in sampled]
    print(f"Noise pool : {len(noise_paths)} wind-rain clips (loaded per worker)\n")

    # file inventory + stable splits
    print("Scanning dataset …")
    files = collect_files()
    print(f"Total files: {sum(len(v) for v in files.values())}\n")

    splits = make_splits(files)
    print()
    for s, recs in splits.items():
        cnt = Counter(cls for cls, _ in recs)
        print(f"  {s:<6}  {len(recs):>5} files  " +
              "  ".join(f"{c}:{n}" for c, n in sorted(cnt.items())))
    print()

    mel_base  = OUT_ROOT / "mel"
    mfcc_base = OUT_ROOT / "mfcc"

    manifest: dict = {
        "seed": SEED, "sample_rate": SR,
        "clip_seconds": CLIP_LEN, "hop_seconds": HOP_LEN,
        "n_mels": N_MELS, "n_mfcc": N_MFCC,
        "n_fft": N_FFT, "win_length": WIN_LENGTH, "hop_stft": HOP_STFT,
        "ebu_target_dbfs": -23,
        "spec_augment_train": True,
        "filter_augment_aug": True,
        "curriculum_phases": phases,
        "dir_calibration_ir": ir_path,
        "label_map":   FOLDER_TO_LABEL,
        "label_names": {str(k): v for k, v in LABEL_NAMES.items()},
        "splits": {},
    }

    for name, items in splits.items():
        print(f"[{name.upper()}] — {len(items)} files")
        counts = process_split(name, items, mel_base, mfcc_base,
                               args.shard_size, noise_paths, ir_path, phases)
        manifest["splits"][name] = counts
        for k, v in counts.items():
            print(f"  {k}: {v:,} windows")
        print()

    manifest["script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()

    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest → {manifest_path.relative_to(REPO_ROOT)}")
    print("=== Stage 2 complete ===")


if __name__ == "__main__":
    main()
