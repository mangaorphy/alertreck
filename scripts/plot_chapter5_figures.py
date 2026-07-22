#!/usr/bin/env python3
"""
Generate the Chapter 5 figures from the trained-model results.

Reads models/*/results.json (the ground truth produced by the training notebooks)
and writes publication-quality PNGs to docs/figures/chapter5/.

    python3 scripts/plot_chapter5_figures.py

Every number plotted comes from results.json or the measured Pi benchmark below —
nothing is hard-coded from the write-up.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
OUT = REPO / "docs" / "figures" / "chapter5"

# ── palette (validated reference set; slots assigned in fixed order) ──────────
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

CLASSES = [
    "background_animals", "background_wind_rain", "threat_chainsaw",
    "threat_dog", "threat_gunshot", "threat_human", "threat_vehicle",
]
SHORT = {c: c.replace("background_", "bg: ").replace("threat_", "") for c in CLASSES}

# Training-window counts (from the shard `y` arrays; see Table 3.2)
TRAIN_N = {
    "background_animals": 3053, "background_wind_rain": 4195, "threat_chainsaw": 1499,
    "threat_dog": 751, "threat_gunshot": 2287, "threat_human": 2431, "threat_vehicle": 638,
}

# Measured on the deployment Pi 4 (50 forward passes, daemon quiesced)
LATENCY_MS = {"min": 153.9, "p50": 158.1, "mean": 190.0, "p95": 351.1, "max": 459.5}

F1_TARGET = 0.80


def setup() -> None:
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.titlepad": 12,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8.5,
    })
    OUT.mkdir(parents=True, exist_ok=True)


def load(model: str) -> dict:
    return json.loads((MODELS / model / "results.json").read_text())


def clean(ax, *, x_grid=False) -> None:
    """Recessive chrome: hide top/right spines, hairline grid on the value axis."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if x_grid else "y", linewidth=0.7, color=GRID)
    ax.grid(axis="y" if x_grid else "x", visible=False)


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


# ── Figure 5.0 — training curves with curriculum phases ──────────────────────
def fig_training_curves(cnn: dict) -> None:
    h = cnn["history"]
    epochs = np.arange(1, len(h["train_loss"]) + 1)
    phases = h["phase"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # phase boundaries
    bounds = [i for i in range(1, len(phases)) if phases[i] != phases[i - 1]]
    def mark_phases(ax):
        edges = [0, *bounds, len(phases)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            label = phases[lo]
            ax.axvspan(lo + 0.5, hi + 0.5, color=GRID, alpha=0.35, lw=0)
            ax.text((lo + hi) / 2 + 0.5, ax.get_ylim()[1], f" Phase {label} ",
                    ha="center", va="top", fontsize=8, color=MUTED)
        for b in bounds:
            ax.axvline(b + 0.5, color=BASELINE, lw=0.8, ls=":")

    ax1.plot(epochs, h["train_loss"], color=BLUE, lw=2, label="Train")
    ax1.plot(epochs, h["val_loss"], color=YELLOW, lw=2, label="Validation")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    clean(ax1); mark_phases(ax1); ax1.legend(loc="upper right")

    ax2.plot(epochs, h["train_f1"], color=BLUE, lw=2, label="Train")
    ax2.plot(epochs, h["val_f1"], color=YELLOW, lw=2, label="Validation")
    best = cnn["best_epoch"]
    ax2.axvline(best, color=CRITICAL, lw=1.2, ls="--")
    ax2.annotate(f"best epoch {best}", xy=(best, h["val_f1"][best - 1]),
                 xytext=(6, -14), textcoords="offset points",
                 fontsize=8, color=CRITICAL)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1")
    ax2.set_title("Macro F1")
    clean(ax2); mark_phases(ax2); ax2.legend(loc="lower right")

    fig.suptitle("Figure 5.0  AudioCNN training across the three-phase augmentation curriculum",
                 fontsize=11, fontweight="bold", color=INK, y=1.04, x=0.5)
    save(fig, "fig_5_0_training_curves.png")


# ── Figure 5.1 — model comparison ────────────────────────────────────────────
def fig_model_comparison(cnn, proto, w2v2, cae, ocsvm) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2]})

    names = ["AudioCNN", "ProtoNet", "Wav2Vec2-L2"]
    f1 = [cnn["test_macro_f1"], proto["test_macro_f1"], w2v2["test_macro_f1"]]
    auc = [cnn["macro_auc"], proto["macro_auc"], w2v2["macro_auc"]]

    x = np.arange(len(names)); w = 0.36
    b1 = ax1.bar(x - w / 2 - 0.01, f1, w, color=BLUE, label="Macro F1")
    b2 = ax1.bar(x + w / 2 + 0.01, auc, w, color=AQUA, label="Macro AUC")
    for bars in (b1, b2):
        ax1.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, color=INK2)
    ax1.set_xticks(x, names)
    ax1.set_ylim(0, 1.12); ax1.set_ylabel("Score")
    ax1.set_title("Discriminative models (7-class)")
    clean(ax1)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncols=2)

    # Single measure (binary AUC) → one hue; identity comes from the axis labels.
    anames = ["Conv-AE", "OC-SVM"]
    aauc = [cae["binary_auc"], ocsvm["binary_auc"]]
    bars = ax2.bar(anames, aauc, 0.5, color=BLUE)
    ax2.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, color=INK2)
    ax2.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax2.text(-0.45, 0.5, " chance", va="bottom", ha="left",
             fontsize=8, color=MUTED)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(0, 1.12); ax2.set_ylabel("Binary AUC")
    ax2.set_title("Anomaly detectors (threat vs background)")
    clean(ax2)

    fig.suptitle("Figure 5.1  Comparative model performance on the held-out test partition",
                 fontsize=11, fontweight="bold", color=INK, y=1.03)
    save(fig, "fig_5_1_model_comparison.png")


# ── Figure 5.2 — per-class F1 vs the 0.80 target ─────────────────────────────
def fig_per_class_f1(cnn: dict) -> None:
    pcf1 = cnn["per_class_f1"]
    items = sorted(CLASSES, key=lambda c: pcf1[c])          # ascending → best on top
    vals = [pcf1[c] for c in items]
    colors = [CRITICAL if v < F1_TARGET else BLUE for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh([SHORT[c] for c in items], vals, 0.62, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5, color=INK2)

    ax.axvline(F1_TARGET, color=MUTED, lw=1.2, ls="--")
    ax.text(F1_TARGET + 0.005, -0.75, "0.80 target", fontsize=8, color=MUTED)

    ax.set_xlim(0, 1.0); ax.set_xlabel("Test macro F1")
    ax.set_title("Figure 5.2  Per-class F1 of the deployed AudioCNN")
    clean(ax, x_grid=True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=CRITICAL)]
    ax.legend(handles, ["Meets 0.80 target", "Below target"], loc="lower right")
    save(fig, "fig_5_2_per_class_f1.png")


# ── Figure 5.3 — F1 vs AUC: the mis-thresholding evidence ────────────────────
def fig_f1_vs_auc(cnn: dict) -> None:
    pcf1, pcauc = cnn["per_class_f1"], cnn["per_class_auc"]
    order = sorted(CLASSES, key=lambda c: pcf1[c], reverse=True)
    f1 = [pcf1[c] for c in order]
    auc = [pcauc[c] for c in order]

    x = np.arange(len(order)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    b1 = ax.bar(x - w / 2 - 0.01, f1, w, color=BLUE, label="F1 (at 0.60 threshold)")
    b2 = ax.bar(x + w / 2 + 0.01, auc, w, color=AQUA, label="AUC (threshold-free)")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5, color=INK2)

    ax.axhline(F1_TARGET, color=MUTED, lw=1, ls="--")
    ax.text(-0.62, F1_TARGET, " 0.80\n F1 target", fontsize=7.5, color=MUTED,
            ha="left", va="center", linespacing=1.3)

    # call out the vehicle dissociation — aim at the middle of the low F1 bar
    vi = order.index("threat_vehicle")
    ax.annotate(
        "high AUC, low F1:\nmis-thresholded,\nnot mis-represented",
        xy=(vi - w / 2 - 0.01, pcf1["threat_vehicle"] / 2),
        xytext=(vi - 2.5, 1.02),
        fontsize=8, color=CRITICAL, ha="left",
        arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.1,
                        connectionstyle="arc3,rad=-0.3"),
    )

    ax.set_xticks(x, [SHORT[c] for c in order], rotation=20, ha="right")
    ax.set_xlim(-0.72, len(order) - 0.35)
    ax.set_ylim(0, 1.22); ax.set_ylabel("Score")
    ax.set_title("Figure 5.3  Per-class F1 against AUC — the vehicle dissociation")
    clean(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncols=2)
    save(fig, "fig_5_3_f1_vs_auc.png")


# ── Figure 5.4 — training volume vs performance ──────────────────────────────
def fig_volume_vs_f1(cnn: dict) -> None:
    pcf1 = cnn["per_class_f1"]
    xs = np.array([TRAIN_N[c] for c in CLASSES], dtype=float)
    ys = np.array([pcf1[c] for c in CLASSES])
    colors = [CRITICAL if y < F1_TARGET else BLUE for y in ys]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(xs, ys, s=90, c=colors, zorder=3, edgecolor=SURFACE, linewidth=1.5)

    for c, xv, yv in zip(CLASSES, xs, ys):
        dx, dy = (8, 4) if c != "threat_vehicle" else (10, -2)
        ax.annotate(SHORT[c], (xv, yv), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color=INK2)

    # trend line (log-x)
    m, b = np.polyfit(np.log10(xs), ys, 1)
    gx = np.linspace(np.log10(xs.min() * 0.85), np.log10(xs.max() * 1.15), 50)
    ax.plot(10 ** gx, m * gx + b, color=MUTED, lw=1.2, ls="--", zorder=1)

    ax.axhline(F1_TARGET, color=MUTED, lw=1, ls=":")
    ax.text(575, F1_TARGET + 0.005, "0.80 target", fontsize=7.5, color=MUTED,
            ha="left", va="bottom")

    ax.set_xscale("log")
    ticks = [600, 800, 1000, 1500, 2000, 3000, 4500]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(560, 5000)
    ax.set_xlabel("Training windows (log scale)")
    ax.set_ylabel("Test F1")
    ax.set_ylim(0.62, 0.92)
    ax.set_title("Figure 5.4  Per-class F1 against effective training volume")
    clean(ax); ax.grid(axis="x", linewidth=0.7, color=GRID)
    save(fig, "fig_5_4_volume_vs_f1.png")


# ── Figure 5.5 — per-class F1 across the three models ────────────────────────
def fig_cross_model(cnn, proto, w2v2) -> None:
    # W2V2 uses short keys — remap to the canonical class names
    w2v2_map = {
        "background_animals": "bg_animals", "background_wind_rain": "bg_wind_rain",
        "threat_chainsaw": "chainsaw", "threat_dog": "dog", "threat_gunshot": "gunshot",
        "threat_human": "human", "threat_vehicle": "vehicle",
    }
    order = sorted(CLASSES, key=lambda c: cnn["per_class_f1"][c], reverse=True)
    a = [cnn["per_class_f1"][c] for c in order]
    b = [proto["per_class_f1"][c] for c in order]
    d = [w2v2["per_class_f1"][w2v2_map[c]] for c in order]

    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bars = [
        ax.bar(x - w - 0.012, a, w, color=BLUE, label="AudioCNN"),
        ax.bar(x, b, w, color=AQUA, label="ProtoNet"),
        ax.bar(x + w + 0.012, d, w, color=YELLOW, label="Wav2Vec2-L2"),
    ]
    for bb in bars:
        ax.bar_label(bb, fmt="%.2f", padding=2, fontsize=6.8, color=INK2, rotation=90)

    ax.axhline(F1_TARGET, color=MUTED, lw=1, ls="--")
    ax.text(-0.68, F1_TARGET, " 0.80\n target", fontsize=7.5, color=MUTED,
            ha="left", va="center", linespacing=1.3)

    ax.set_xticks(x, [SHORT[c] for c in order], rotation=20, ha="right")
    ax.set_xlim(-0.75, len(order) - 0.35)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Test F1")
    ax.set_title("Figure 5.5  Per-class F1 across the three discriminative models")
    clean(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.40), ncols=3)
    save(fig, "fig_5_5_cross_model_f1.png")


# ── Figure 5.6 — anomaly detector per-class AUC ──────────────────────────────
def fig_anomaly(cae, ocsvm) -> None:
    threats = ["threat_human", "threat_gunshot", "threat_chainsaw",
               "threat_dog", "threat_vehicle"]
    a = [cae["per_class_auc"][c] for c in threats]
    b = [ocsvm["per_class_auc"][c] for c in threats]

    x = np.arange(len(threats)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    b1 = ax.bar(x - w / 2 - 0.01, a, w, color=BLUE, label="Conv-AE")
    b2 = ax.bar(x + w / 2 + 0.01, b, w, color=AQUA, label="OC-SVM")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=INK2)

    ax.axhline(0.5, color=CRITICAL, lw=1.2, ls="--")
    ax.text(len(threats) - 0.45, 0.52, "chance (0.5)",
            fontsize=8, color=CRITICAL, ha="right")

    # The one bar below chance — mark the bar itself, explain in the footnote.
    vx = 4 - w / 2 - 0.01
    ax.annotate("below chance", xy=(vx, cae["per_class_auc"]["threat_vehicle"] + 0.015),
                xytext=(vx, 0.30), ha="center", va="bottom",
                fontsize=8, color=CRITICAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.1))

    ax.set_xticks(x, [SHORT[c] for c in threats])
    ax.set_ylim(0, 1.12); ax.set_ylabel("Detection AUC")
    ax.set_title("Figure 5.6  Per-class detection AUC — unsupervised anomaly models")
    clean(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncols=2)

    fig.text(0.5, -0.10,
             "Conv-AE scores below chance on vehicle: engine noise is low-frequency and "
             "quasi-stationary, so the\nautoencoder reconstructs it as readily as the "
             "wind/rain background it was trained on.",
             ha="center", va="top", fontsize=8, color=INK2, linespacing=1.5)
    save(fig, "fig_5_6_anomaly_auc.png")


# ── Figure 5.7 — inference latency against the real-time budget ──────────────
def fig_latency() -> None:
    fig, ax = plt.subplots(figsize=(9, 2.9))

    window_ms = 3000
    ax.barh([0], [window_ms], height=0.42, color=GRID, zorder=1)
    ax.text(window_ms - 40, 0, "3,000 ms analysis window", va="center", ha="right",
            fontsize=8.5, color=INK2, zorder=4)

    lo, hi = LATENCY_MS["min"], LATENCY_MS["p95"]
    ax.barh([0], [hi - lo], left=lo, height=0.42, color=BLUE, zorder=2)
    ax.plot([LATENCY_MS["mean"]], [0], marker="|", ms=18, mew=2.4,
            color=SURFACE, zorder=3)

    # The measured range is tiny at this scale — label it once, off to the side,
    # with a single leader rather than three colliding callouts.
    ax.annotate(
        f"measured inference\nmin {lo:.0f} · mean {LATENCY_MS['mean']:.0f} · "
        f"p95 {hi:.0f} ms",
        xy=(hi, 0.22), xytext=(560, 0.72),
        fontsize=8.5, color=INK2, ha="left", va="center", linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.9,
                        connectionstyle="arc3,rad=0.2"),
    )

    ax.text(window_ms / 2, -0.78,
            f"≈{window_ms / LATENCY_MS['mean']:.0f}× real-time headroom",
            ha="center", fontsize=9.5, color=GOOD, fontweight="bold")

    ax.set_yticks([]); ax.set_ylim(-1.1, 0.9)
    ax.set_xlim(0, window_ms * 1.02)
    ax.set_xlabel("Milliseconds")
    ax.set_title("Figure 5.7  Measured inference latency against the 3-second window (Raspberry Pi 4, CPU)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis="x", linewidth=0.7, color=GRID)
    save(fig, "fig_5_7_latency.png")


def main() -> None:
    setup()
    cnn, proto = load("custom_cnn"), load("protonet")
    w2v2, cae, ocsvm = load("w2v2_l2"), load("conv_ae"), load("oc_svm")

    print(f"Writing Chapter 5 figures → {OUT.relative_to(REPO)}/")
    fig_training_curves(cnn)
    fig_model_comparison(cnn, proto, w2v2, cae, ocsvm)
    fig_per_class_f1(cnn)
    fig_f1_vs_auc(cnn)
    fig_volume_vs_f1(cnn)
    fig_cross_model(cnn, proto, w2v2)
    fig_anomaly(cae, ocsvm)
    fig_latency()
    print("Done — 8 figures.")


if __name__ == "__main__":
    main()
