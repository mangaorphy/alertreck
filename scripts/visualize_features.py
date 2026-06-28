#!/usr/bin/env python3
"""Feature visualisation for the Alertreck preprocessing pipeline.

Generates two kinds of figures under docs/figures/ :

  feature-samples   per-class log-mel + MFCC examples (read from the processed
                    val shards — no SpecAugment, so the features are clean)
  augmentation      clean vs curriculum Phase A/B/C of the SAME window, produced
                    by re-running the real pipeline functions on raw audio

Run with the env that has librosa (the same one used for preprocessing):
    /opt/anaconda3/bin/python scripts/visualize_features.py            # both
    /opt/anaconda3/bin/python scripts/visualize_features.py --what aug
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_preprocessing as ap   # reuse the real pipeline functions

REPO   = ap.REPO_ROOT
OUTDIR = REPO / "docs" / "figures" / "feature_samples"
AUGDIR = REPO / "docs" / "figures"
NAMES  = list(ap.FOLDER_TO_LABEL.keys())
SEED   = ap.SEED

# classes shown in the augmentation comparison (clear, illustrative signatures)
AUG_EXAMPLE_CLASSES = ["threat_chainsaw", "threat_gunshot", "threat_vehicle"]


# ── per-class log-mel + MFCC samples (from processed val shards) ───────────────
def feature_samples(n_per_class: int = 5) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    mel_shards  = sorted(glob.glob(str(REPO / "data/processed/mel/val/*.npz")))
    mfcc_shards = sorted(glob.glob(str(REPO / "data/processed/mfcc/val/*.npz")))
    if not mel_shards:
        print("  no val shards found — run audio_preprocessing.py first"); return
    assert len(mel_shards) == len(mfcc_shards), "mel/mfcc shard count mismatch"

    pool   = {c: [] for c in range(7)}
    counts = {c: 0 for c in range(7)}
    for mp, fp in zip(mel_shards, mfcc_shards):
        dm, df = np.load(mp), np.load(fp)
        assert np.array_equal(dm["y"], df["y"]), f"y mismatch in {os.path.basename(mp)}"
        for c in range(7):
            idx = np.where(dm["y"] == c)[0]
            counts[c] += len(idx)
            for k in idx:
                if len(pool[c]) < 60:
                    pool[c].append((dm["X"][k], df["X"][k]))

    for c, name in enumerate(NAMES):
        picks = rng.choice(len(pool[c]), size=min(n_per_class, len(pool[c])), replace=False)
        fig, axes = plt.subplots(2, len(picks), figsize=(4 * len(picks), 6.2))
        for j, p in enumerate(picks):
            mel, mfcc = pool[c][p]
            im0 = axes[0, j].imshow(mel, origin="lower", aspect="auto", cmap="magma")
            axes[0, j].set_title(f"sample {j + 1}", fontsize=10)
            axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
            im1 = axes[1, j].imshow(mfcc, origin="lower", aspect="auto", cmap="viridis")
            axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        axes[0, 0].set_ylabel("log-mel\n(128 bins)", fontsize=11)
        axes[1, 0].set_ylabel("MFCC+Δ+ΔΔ\n(120)", fontsize=11)
        fig.suptitle(f"{name}   —   log-mel (top) & MFCC (bottom)   ·   {counts[c]} val windows",
                     fontsize=14, fontweight="bold")
        fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.012, pad=0.01, label="dB")
        fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.012, pad=0.01)
        out = OUTDIR / f"{c}_{name}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print("saved", out.relative_to(REPO))

    fig, axes = plt.subplots(7, 2, figsize=(12, 22))
    for c, name in enumerate(NAMES):
        mel, mfcc = pool[c][0]
        axes[c, 0].imshow(mel, origin="lower", aspect="auto", cmap="magma")
        axes[c, 0].set_ylabel(name, fontsize=11, fontweight="bold")
        axes[c, 1].imshow(mfcc, origin="lower", aspect="auto", cmap="viridis")
        for k in (0, 1):
            axes[c, k].set_xticks([]); axes[c, k].set_yticks([])
    axes[0, 0].set_title("log-mel spectrogram (128×301)", fontsize=12)
    axes[0, 1].set_title("MFCC + Δ + ΔΔ (120×301)", fontsize=12)
    fig.suptitle("Processed features by class — overview", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = OUTDIR / "overview_all_classes.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", out.relative_to(REPO))


def _first_loadable(folder):
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in ap.AUDIO_EXTS:
            a = ap.load_mono(f)
            if a is not None and len(a) >= ap.CLIP_SAMPLES:
                return f, ap.ebu_r128_normalize(a)
    return None, None


# ── clean vs Phase A/B/C augmentation of the same window ──────────────────────
def augmentation_comparison() -> None:
    AUGDIR.mkdir(parents=True, exist_ok=True)
    import random
    random.seed(SEED); np.random.seed(SEED)
    # populate the noise pool with real wind/rain clips so the noise effect is realistic
    noise_files = sorted((ap.DATASET_DIR / "background_wind_rain").glob("*.wav"))[:8]
    ap._g_noise_pool = [ap.ebu_r128_normalize(ap.load_mono(p))
                        for p in noise_files if ap.load_mono(p) is not None]

    phases = ["A", "B", "C"]
    rows = [c for c in AUG_EXAMPLE_CLASSES if (ap.DATASET_DIR / c).is_dir()]
    fig, axes = plt.subplots(len(rows), 1 + len(phases),
                             figsize=(4 * (1 + len(phases)), 3.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]

    for r, cls in enumerate(rows):
        path, audio = _first_loadable(ap.DATASET_DIR / cls)
        if audio is None:
            continue
        windows = (ap.select_event_windows(audio) if cls in ap.EVENT_SELECT_CLASSES
                   else ap.slice_windows(audio))
        win = windows[len(windows) // 2]                       # a representative window

        clean = ap.extract_mel(win)
        axes[r, 0].imshow(clean, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
        axes[r, 0].set_ylabel(cls, fontsize=11, fontweight="bold")
        if r == 0:
            axes[r, 0].set_title("clean", fontsize=12)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

        for k, phase in enumerate(phases, start=1):
            aug = ap.compound_augment(win, phase)[0]           # first augmented copy
            mel = ap.filter_augment(ap.spec_augment(ap.extract_mel(aug)))
            axes[r, k].imshow(mel, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
            if r == 0:
                cfg = ap.PHASE_CFG[phase]
                axes[r, k].set_title(f"Phase {phase}\n{cfg['min_fx']}–{cfg['max_fx']} fx · "
                                     f"sev {cfg['severity']} · SNR {cfg['snr_lo']}–{cfg['snr_hi']} dB",
                                     fontsize=10)
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])

    fig.suptitle("Curriculum augmentation — same window, increasing difficulty (A→B→C)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = AUGDIR / "augmentation_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", out.relative_to(REPO))


def main() -> None:
    ap_ = argparse.ArgumentParser(description="Alertreck feature & augmentation visualisations")
    ap_.add_argument("--what", choices=["samples", "aug", "all"], default="all")
    args = ap_.parse_args()
    if args.what in ("samples", "all"):
        feature_samples()
    if args.what in ("aug", "all"):
        augmentation_comparison()


if __name__ == "__main__":
    main()
