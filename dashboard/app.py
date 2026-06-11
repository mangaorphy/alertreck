"""
Alertreck Explainability Dashboard
==================================
A Flask LAN dashboard that shows each detected sound with a Grad-CAM heatmap
overlaid on its mel spectrogram — visualising *why* the model flagged it.

Architecture:
  Pi (detector) ──saves──►  evidence/<date>/<class>/<name>.wav
                                                  <name>.mel.npy   (model input)
                                                  <name>.json      (class, probs, GPS, time)
        │  rsync (sync_events.sh)
        ▼
  Mac (this app) ──reads events──► Grad-CAM (torch) ──► overlay PNG ──► web page

Run (on the Mac, base conda env has torch + matplotlib):
    /opt/anaconda3/bin/python -m flask --app dashboard/app run --host 0.0.0.0 --port 5000
  or:
    /opt/anaconda3/bin/python dashboard/app.py

Then open  http://<mac-ip>:5000  from any device on the LAN.
"""

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template

from gradcam import GradCAM, CLASS_NAMES

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
REPO        = HERE.parent
EVENTS_DIR  = HERE / "events"                       # synced from the Pi (sync_events.sh)
OVERLAY_DIR = HERE / "static" / "overlays"          # cached Grad-CAM PNGs
CHECKPOINT  = REPO / "models" / "custom_cnn" / "best_model.pt"

OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Auto-sync from the Pi (background thread) ─────────────────────────────────
# Requires passwordless SSH (ssh-copy-id) so rsync never blocks on a prompt.
SYNC_ENABLED  = os.getenv("ALERTRECK_SYNC", "1") != "0"
SYNC_INTERVAL = float(os.getenv("ALERTRECK_SYNC_INTERVAL", "30"))   # seconds
PI_HOST       = os.getenv("ALERTRECK_PI", "alertreck@alertreck.local")
PI_REMOTE     = os.getenv("ALERTRECK_REMOTE", "~/alertreck/data/evidence/")
_sync_started = False


def _sync_once() -> None:
    """Pull new detection sidecars (json/mel/wav) from the Pi. Failures are non-fatal."""
    cmd = [
        "rsync", "-az", "--timeout=30",
        "--include=*/", "--include=*.json", "--include=*.mel.npy", "--include=*.wav",
        "--exclude=*",
        f"{PI_HOST}:{PI_REMOTE}", f"{EVENTS_DIR}/",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode != 0 and r.stderr:
            print(f"[sync] rsync rc={r.returncode}: {r.stderr.decode(errors='ignore').strip()[:160]}")
    except Exception as e:
        print(f"[sync] error: {e}")


def _sync_loop() -> None:
    while True:
        _sync_once()
        time.sleep(SYNC_INTERVAL)


def _start_sync() -> None:
    global _sync_started
    if SYNC_ENABLED and not _sync_started:
        _sync_started = True
        threading.Thread(target=_sync_loop, daemon=True).start()
        print(f"[sync] auto-sync every {SYNC_INTERVAL:.0f}s from {PI_HOST}")

# ── Grad-CAM engine (loaded once; guard with a lock — torch hooks aren't reentrant) ──
_cam = GradCAM(CHECKPOINT)
_cam_lock = threading.Lock()

app = Flask(__name__)


@app.template_filter("datetime")
def _fmt_datetime(epoch):
    try:
        return datetime.fromtimestamp(float(epoch)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown time"


def _overlay_path(stem: str) -> Path:
    return OVERLAY_DIR / f"{stem}.png"


def _render_overlay(mel: np.ndarray, record: dict, out_png: Path):
    """Run Grad-CAM and render the mel spectrogram with the heatmap overlaid."""
    with _cam_lock:
        cam, pred, probs = _cam(mel)

    pred_name = CLASS_NAMES[pred]
    conf = float(probs[pred])

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
              extent=[0, mel.shape[1], 0, mel.shape[0]])
    ax.imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.45,
              extent=[0, mel.shape[1], 0, mel.shape[0]])
    ax.set_title(f"{record.get('threat_type', pred_name)}  ·  Grad-CAM "
                 f"(model: {pred_name} {conf:.0%})", fontsize=10)
    ax.set_xlabel("Time frames (10 ms)")
    ax.set_ylabel("Mel bins")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return pred_name, conf, probs


def _load_events() -> list[dict]:
    """Scan the synced events dir, ensure each has an overlay, newest first."""
    events = []
    for jpath in EVENTS_DIR.rglob("*.json"):
        try:
            record = json.loads(jpath.read_text())
        except Exception:
            continue
        mel_path = jpath.parent / record.get("mel_file", "")
        if not mel_path.exists():
            continue

        stem = jpath.stem
        png = _overlay_path(stem)
        try:
            if not png.exists():
                mel = np.load(mel_path)
                _render_overlay(mel, record, png)
        except Exception as e:
            print(f"Grad-CAM failed for {stem}: {e}")
            continue

        probs = record.get("class_probabilities", {})
        events.append({
            "stem":        stem,
            "overlay":     f"overlays/{stem}.png",
            "threat_type": record.get("threat_type", "?"),
            "threat_level": record.get("threat_level", ""),
            "confidence":  record.get("confidence") or 0.0,
            "timestamp":   record.get("timestamp") or 0.0,
            "location":    record.get("location") or {},
            "probs":       sorted(probs.items(), key=lambda kv: kv[1], reverse=True),
            "mtime":       jpath.stat().st_mtime,
        })

    events.sort(key=lambda e: e["mtime"], reverse=True)
    return events


@app.route("/")
def index():
    events = _load_events()
    return render_template("index.html", events=events, n=len(events))


# Kick off the background Pi→Mac sync as soon as the app is imported,
# so it runs under both `python app.py` and `flask run`.
_start_sync()


if __name__ == "__main__":
    # Default to 8000 — macOS reserves 5000 for AirPlay Receiver. Override with PORT.
    port = int(os.getenv("PORT", "8000"))
    # threaded=False keeps Grad-CAM hooks deterministic; the lock also guards it.
    app.run(host="0.0.0.0", port=port, threaded=False)
