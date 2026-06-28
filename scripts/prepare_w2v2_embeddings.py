"""
Stage 2b — W2V2-L2 Embedding Extraction
========================================
Reads the stable file-level splits from data/processed/splits.json, loads each
audio clip, resamples to 16 kHz mono, optionally applies compound augmentation,
runs batches through the frozen, layer-2-TRUNCATED wav2vec 2.0 base encoder,
taps hidden states after transformer layer 2, mean-pools over the time axis to
produce a 768-dim embedding per window, and saves as .npz shards.

The encoder is physically truncated to its first 2 transformer layers
(Geldenhuys & Niesler, 2026): the upper 10 layers are deleted, so the forward
pass stops at the tap point. Embeddings are identical to reading layer 2 of the
full network, but at a fraction of the parameters / compute — the on-device
motivation for the frozen-transfer (W2V2-L2) arm. The truncated param count is
printed at load time.

The encoder is run ONCE per window and the result stored — training the linear
head in notebook 04a never touches the encoder again, so GPU is only needed
here, not during head training.

Output layout:
  data/processed/w2v2_l2/train/shard_NNN.npz
  data/processed/w2v2_l2/val/shard_NNN.npz
  data/processed/w2v2_l2/test/shard_NNN.npz
  data/processed/w2v2_l2/train_aug_A/shard_NNN.npz   (if --aug-phase A)
  data/processed/w2v2_l2/train_aug_B/shard_NNN.npz   (if --aug-phase B)
  data/processed/w2v2_l2/train_aug_C/shard_NNN.npz   (if --aug-phase C)
  data/processed/w2v2_l2/manifest.json

Each shard .npz contains:
  X  : float32  (N, 768)  — L2-normalised wav2vec 2.0 layer-2 embeddings
  y  : int64    (N,)      — class label  0–6

Augmentation phases match audio_preprocessing.py exactly (same PHASE_CFG,
same compound effects, same SNR ranges) but operate on 16 kHz waveforms.
SpecAugment and FilterAugment are NOT applied — those are spectrogram ops.

Usage:
  # Clean splits only (val + test + train, no augmentation)
  python3 scripts/prepare_w2v2_embeddings.py

  # Clean + all curriculum phases
  python3 scripts/prepare_w2v2_embeddings.py --aug-phase A B C

  # Single phase, explicit device, smaller batches for CPU
  python3 scripts/prepare_w2v2_embeddings.py --aug-phase B --device cpu --batch-size 8

  # Kaggle (GPU T4/P100)
  python3 scripts/prepare_w2v2_embeddings.py --aug-phase A B C --device cuda --batch-size 64

Requirements:
  pip install transformers torch torchaudio librosa soundfile tqdm
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
SEED         = 42
SR           = 16_000           # wav2vec 2.0 requires 16 kHz
CLIP_LEN     = 3.0              # seconds — matches mel/mfcc preprocessing
HOP_LEN      = 1.5              # 50 % overlap
CLIP_SAMPLES = int(SR * CLIP_LEN)     # 48 000
HOP_SAMPLES  = int(SR * HOP_LEN)      # 24 000
EMBED_DIM    = 768              # wav2vec 2.0 base hidden size
W2V2_LAYER   = 2               # tap hidden states after transformer layer 2
W2V2_HUB     = "facebook/wav2vec2-base"

EBU_TARGET = 10 ** (-23.0 / 20.0)   # −23 dBFS

FOLDER_TO_LABEL: dict[str, int] = {
    "background_animals":   0,
    "background_wind_rain": 1,
    "threat_chainsaw":      2,
    "threat_dog":           3,
    "threat_gunshot":       4,
    "threat_human":         5,
    "threat_vehicle":       6,
}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
OUT_ROOT    = REPO_ROOT / "data" / "processed" / "w2v2_l2"
SPLITS_JSON = REPO_ROOT / "data" / "processed" / "splits.json"

# Curriculum phases — identical to audio_preprocessing.py
PHASE_CFG: dict[str, dict] = {
    "A": {"min_fx": 1, "max_fx": 2, "severity": 0.3, "snr_lo": 15, "snr_hi": 30, "copies": 1},
    "B": {"min_fx": 2, "max_fx": 4, "severity": 0.5, "snr_lo": 10, "snr_hi": 15, "copies": 2},
    "C": {"min_fx": 2, "max_fx": 5, "severity": 1.0, "snr_lo":  5, "snr_hi": 10, "copies": 3},
}

# Module-level noise pool (loaded once at startup)
_noise_pool: list[np.ndarray] = []


# ── audio I/O ──────────────────────────────────────────────────────────────────

def load_mono(path: Path) -> np.ndarray | None:
    """Load any audio format via ffmpeg, resample to 16 kHz mono float32."""
    try:
        cmd = ["ffmpeg", "-i", str(path), "-f", "f32le",
               "-ar", str(SR), "-ac", "1", "pipe:1", "-loglevel", "quiet"]
        result = subprocess.run(cmd, capture_output=True)
        if not result.stdout:
            return None
        return np.frombuffer(result.stdout, dtype=np.float32).copy()
    except Exception:
        return None


def ebu_r128_normalize(audio: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    return np.clip(audio * (EBU_TARGET / rms), -1.0, 1.0).astype(np.float32)


def slice_windows(audio: np.ndarray) -> list[np.ndarray]:
    """Segment into 3-second windows with 50 % hop; zero-pad short clips."""
    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    wins, start = [], 0
    while start + CLIP_SAMPLES <= len(audio):
        wins.append(audio[start: start + CLIP_SAMPLES].copy())
        start += HOP_SAMPLES
    return wins


# ── waveform augmentation (adapted for 16 kHz) ────────────────────────────────

def _sinc_lowpass(cutoff_hz: float, num_taps: int = 101) -> np.ndarray:
    if num_taps % 2 == 0:
        num_taps += 1
    fc = cutoff_hz / SR
    n  = np.arange(num_taps) - (num_taps - 1) / 2
    h  = np.sinc(2 * fc * n) * np.blackman(num_taps)
    h /= h.sum()
    return h.astype(np.float32)


def _apply_lowpass(audio: np.ndarray, severity: float) -> np.ndarray:
    # severity 0 → 8 kHz (Nyquist); severity 1 → 500 Hz
    cutoff = max(500.0, (1.0 - severity) * (SR / 2) + severity * 500.0)
    return np.clip(np.convolve(audio, _sinc_lowpass(cutoff), mode="same"),
                   -1.0, 1.0).astype(np.float32)


def _synthetic_rir(rt60_s: float) -> np.ndarray:
    length = max(1, int(SR * rt60_s))
    t   = np.arange(length) / SR
    env = np.exp(-6.9 * t / rt60_s)
    rir = (np.random.randn(length) * env).astype(np.float32)
    peak = np.abs(rir).max()
    return rir / peak if peak > 1e-9 else rir


def _apply_rir(audio: np.ndarray, severity: float) -> np.ndarray:
    rt60 = 0.05 + severity * 0.45
    out  = np.convolve(audio, _synthetic_rir(rt60), mode="full")[: len(audio)]
    peak = np.abs(out).max()
    return (out / peak if peak > 1e-9 else out).astype(np.float32)


def _apply_mp3(audio: np.ndarray, severity: float) -> np.ndarray:
    bitrate = int(128 - severity * 96)
    in_path = mp3_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            in_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
        sf.write(in_path, audio, SR, format="WAV", subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-i", in_path, "-b:a", f"{bitrate}k",
                        mp3_path, "-loglevel", "quiet"],
                       check=True, capture_output=True)
        result = subprocess.run(
            ["ffmpeg", "-i", mp3_path, "-f", "f32le",
             "-ar", str(SR), "-ac", "1", "pipe:1", "-loglevel", "quiet"],
            capture_output=True)
        dec = np.frombuffer(result.stdout, dtype=np.float32)
        return (dec[: len(audio)].copy() if len(dec) >= len(audio)
                else np.pad(dec, (0, len(audio) - len(dec))).copy())
    except Exception:
        return audio.copy()
    finally:
        for p in [in_path, mp3_path]:
            if p:
                Path(p).unlink(missing_ok=True)


def _apply_noise(audio: np.ndarray, severity: float,
                 snr_lo: float, snr_hi: float) -> np.ndarray:
    snr_db = random.uniform(snr_lo, snr_hi)
    if _noise_pool and random.random() < 0.7:
        bg = random.choice(_noise_pool)
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
    """Apply compound waveform augmentation; return list of augmented copies."""
    cfg = PHASE_CFG[phase]
    sev = cfg["severity"]
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


# ── wav2vec 2.0 encoder ────────────────────────────────────────────────────────

def load_encoder(device: torch.device):
    """Load facebook/wav2vec2-base, freeze all parameters, return (extractor, model)."""
    try:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    except ImportError:
        raise ImportError(
            "transformers not installed — run: pip install transformers")

    print(f"Loading {W2V2_HUB} …")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(W2V2_HUB)
    model = Wav2Vec2Model.from_pretrained(W2V2_HUB)

    full_params = sum(p.numel() for p in model.parameters())

    # ── Truncation (Geldenhuys & Niesler, 2026) ──────────────────────────────
    # We only ever tap hidden_states[W2V2_LAYER], so the transformer layers above
    # it are dead weight. Physically drop them: the forward pass then stops at
    # layer W2V2_LAYER, giving identical embeddings at a fraction of the params /
    # compute — this is the "~10% of the network, well suited to on-device" claim
    # that motivates the frozen-transfer arm.
    model.encoder.layers = model.encoder.layers[:W2V2_LAYER]

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Full encoder : {full_params:,} params  (12 transformer layers)")
    print(f"  Truncated    : {n_params:,} params  (layers 1–{W2V2_LAYER} kept)  "
          f"= {100 * n_params / full_params:.1f}% of full network")
    print(f"  Device       : {device}")
    print(f"  Tapping      : hidden_states[{W2V2_LAYER}]  (after transformer layer {W2V2_LAYER})")
    print(f"  Output dim   : {EMBED_DIM}\n")
    return feature_extractor, model


@torch.no_grad()
def encode_batch(
    windows: list[np.ndarray],
    feature_extractor,
    model,
    device: torch.device,
) -> np.ndarray:
    """
    Encode a list of 16 kHz waveform arrays → (N, 768) float32 numpy array.

    Uses Wav2Vec2FeatureExtractor for input normalisation (zero-mean unit-var
    per sequence), extracts hidden states after transformer layer W2V2_LAYER,
    and mean-pools over the time axis.
    """
    inputs = feature_extractor(
        windows,
        sampling_rate=SR,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs.input_values.to(device)           # (B, 48000)
    outputs = model(input_values, output_hidden_states=True)
    # hidden_states: tuple len = num_layers + 1
    #   [0]  = after feature-extractor projection
    #   [k]  = after transformer layer k
    layer_out  = outputs.hidden_states[W2V2_LAYER]          # (B, ~150, 768)
    embeddings = layer_out.mean(dim=1)                      # (B, 768)
    # L2-normalise so cosine distance == Euclidean in unit sphere
    embeddings = F.normalize(embeddings, dim=1)
    return embeddings.cpu().float().numpy()


# ── shard writer ───────────────────────────────────────────────────────────────

class ShardWriter:
    """Accumulates (embedding, label) pairs and flushes to compressed .npz shards."""

    def __init__(self, out_dir: Path, shard_size: int) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._dir  = out_dir
        self._size = shard_size
        self._X:   list[np.ndarray] = []
        self._y:   list[int]        = []
        self._idx  = 0
        self.total = 0

    def add(self, embedding: np.ndarray, label: int) -> None:
        self._X.append(embedding)
        self._y.append(label)
        self.total += 1
        if len(self._X) >= self._size:
            self._flush()

    def _flush(self) -> None:
        if not self._X:
            return
        np.savez_compressed(
            self._dir / f"shard_{self._idx:03d}.npz",
            X=np.stack(self._X).astype(np.float32),
            y=np.array(self._y, dtype=np.int64),
        )
        self._X.clear()
        self._y.clear()
        self._idx += 1

    def close(self) -> None:
        self._flush()


# ── batch encoder helper ───────────────────────────────────────────────────────

def _flush_batch(
    batch_wins: list[np.ndarray],
    batch_lbls: list[int],
    writer: ShardWriter,
    feature_extractor,
    model,
    device: torch.device,
) -> None:
    """Encode pending batch and write to shard; clears both lists in-place."""
    if not batch_wins:
        return
    embs = encode_batch(batch_wins, feature_extractor, model, device)
    for emb, lbl in zip(embs, batch_lbls):
        writer.add(emb, lbl)
    batch_wins.clear()
    batch_lbls.clear()


# ── split processing ───────────────────────────────────────────────────────────

def process_split(
    split_name: str,
    items: list[tuple[str, str]],
    phases: list[str],
    feature_extractor,
    model,
    device: torch.device,
    batch_size: int,
    shard_size: int,
) -> dict[str, int]:
    counts   = {}
    is_train = split_name == "train"

    # ── clean windows ──────────────────────────────────────────────────────────
    writer     = ShardWriter(OUT_ROOT / split_name, shard_size)
    batch_wins: list[np.ndarray] = []
    batch_lbls: list[int]        = []

    for cls, path_str in tqdm(items, desc=f"  {split_name:<14} clean"):
        audio = load_mono(Path(path_str))
        if audio is None:
            continue
        audio = ebu_r128_normalize(audio)
        label = FOLDER_TO_LABEL[cls]
        for win in slice_windows(audio):
            batch_wins.append(win)
            batch_lbls.append(label)
            if len(batch_wins) >= batch_size:
                _flush_batch(batch_wins, batch_lbls, writer,
                             feature_extractor, model, device)

    _flush_batch(batch_wins, batch_lbls, writer,
                 feature_extractor, model, device)
    writer.close()
    counts["clean"] = writer.total
    print(f"    clean : {writer.total:,} embeddings")

    # ── augmented phases (train only) ──────────────────────────────────────────
    if not is_train or not phases:
        return counts

    for phase in phases:
        aug_writer = ShardWriter(OUT_ROOT / f"train_aug_{phase}", shard_size)
        batch_wins = []
        batch_lbls = []
        cfg  = PHASE_CFG[phase]
        desc = (f"  train_aug_{phase:<10}"
                f"(fx {cfg['min_fx']}–{cfg['max_fx']}, "
                f"sev {cfg['severity']}, ×{cfg['copies']})")

        for cls, path_str in tqdm(items, desc=desc):
            audio = load_mono(Path(path_str))
            if audio is None:
                continue
            audio = ebu_r128_normalize(audio)
            label = FOLDER_TO_LABEL[cls]
            for win in slice_windows(audio):
                for aug_win in compound_augment(win, phase):
                    batch_wins.append(aug_win)
                    batch_lbls.append(label)
                    if len(batch_wins) >= batch_size:
                        _flush_batch(batch_wins, batch_lbls, aug_writer,
                                     feature_extractor, model, device)

        _flush_batch(batch_wins, batch_lbls, aug_writer,
                     feature_extractor, model, device)
        aug_writer.close()
        counts[f"aug_{phase}"] = aug_writer.total
        print(f"    aug_{phase} : {aug_writer.total:,} embeddings")

    return counts


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Alertreck Stage 2b — W2V2-L2 embedding extraction")
    ap.add_argument(
        "--aug-phase", nargs="*", choices=["A", "B", "C"],
        default=[], metavar="PHASE",
        help="Curriculum phases to generate for train split (A B C). "
             "Default: none (clean splits only).")
    ap.add_argument(
        "--batch-size", type=int, default=64,
        help="Encoder batch size. Reduce to 8–16 on CPU or if GPU OOM. (default: 64)")
    ap.add_argument(
        "--shard-size", type=int, default=1000,
        help="Embeddings per .npz shard. (default: 1000)")
    ap.add_argument(
        "--device", default="auto",
        help="'cuda', 'cpu', or 'auto' (default — uses CUDA if available).")
    ap.add_argument(
        "--noise-clips", type=int, default=100,
        help="Number of background_wind_rain clips to load as noise pool. (default: 100)")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto"
              else torch.device(args.device))
    phases = args.aug_phase or []

    print("=== Alertreck Stage 2b — W2V2-L2 Embedding Extraction ===")
    print(f"SR         : {SR} Hz  |  clip: {CLIP_LEN}s ({CLIP_SAMPLES} samples)  "
          f"hop: {HOP_LEN}s ({HOP_SAMPLES} samples)")
    print(f"Encoder    : {W2V2_HUB}  →  layer {W2V2_LAYER} hidden states  →  "
          f"{EMBED_DIM}-dim L2-normalised")
    print(f"Device     : {device}  |  batch_size: {args.batch_size}")
    print(f"Phases     : {phases or 'none (clean only)'}")
    print(f"Output     : {OUT_ROOT.relative_to(REPO_ROOT)}")
    print()

    # ── validate splits.json ───────────────────────────────────────────────────
    if not SPLITS_JSON.exists():
        raise FileNotFoundError(
            f"splits.json not found at {SPLITS_JSON}\n"
            "Run audio_preprocessing.py first to generate stable file-level splits.")
    splits_raw = json.loads(SPLITS_JSON.read_text())
    splits     = {s: [(cls, path) for cls, path in recs]
                  for s, recs in splits_raw.items()}

    print("File-level splits (from splits.json):")
    for name, recs in splits.items():
        cnt = Counter(cls for cls, _ in recs)
        print(f"  {name:<6}  {len(recs):>5} files  " +
              "  ".join(f"{c}:{n}" for c, n in sorted(cnt.items())))
    print()

    # ── load noise pool at 16 kHz ──────────────────────────────────────────────
    noise_dir = DATASET_DIR / "background_wind_rain"
    if noise_dir.exists():
        noise_files = sorted(f for f in noise_dir.iterdir()
                             if f.suffix.lower() in AUDIO_EXTS)
        sampled = random.sample(noise_files, min(args.noise_clips, len(noise_files)))
        for p in tqdm(sampled, desc="Loading noise pool (16 kHz)"):
            a = load_mono(p)
            if a is not None:
                _noise_pool.append(ebu_r128_normalize(a))
        print(f"Noise pool : {len(_noise_pool)} clips\n")
    else:
        print(f"WARNING: noise directory not found at {noise_dir} — noise augmentation disabled\n")

    # ── load encoder ───────────────────────────────────────────────────────────
    feature_extractor, model = load_encoder(device)

    # ── process splits ─────────────────────────────────────────────────────────
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "seed":             SEED,
        "sample_rate":      SR,
        "clip_seconds":     CLIP_LEN,
        "hop_seconds":      HOP_LEN,
        "clip_samples":     CLIP_SAMPLES,
        "hop_samples":      HOP_SAMPLES,
        "embed_dim":        EMBED_DIM,
        "w2v2_model":       W2V2_HUB,
        "w2v2_layer":       W2V2_LAYER,
        "l2_normalised":    True,
        "curriculum_phases": phases,
        "label_map":        FOLDER_TO_LABEL,
        "splits":           {},
    }

    for split_name, items in splits.items():
        print(f"[{split_name.upper()}] — {len(items)} files")
        counts = process_split(
            split_name, items, phases,
            feature_extractor, model, device,
            args.batch_size, args.shard_size,
        )
        manifest["splits"][split_name] = counts
        print()

    manifest["script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()

    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest → {manifest_path.relative_to(REPO_ROOT)}")

    total = sum(v for counts in manifest["splits"].values()
                for v in counts.values())
    print(f"Total embeddings written: {total:,}")
    print("=== Stage 2b complete ===")


if __name__ == "__main__":
    main()
