# Alertreck Explainability Dashboard

A Flask LAN dashboard that shows each detected sound with a **Grad-CAM** heatmap overlaid on its
mel spectrogram — visualising *why* the CNN flagged it as a threat.

It runs on a **separate machine** (your Mac), where PyTorch already lives, so the Raspberry Pi stays
ONNX-only and lean. Grad-CAM is computed on the **exact mel spectrogram the Pi classified** (the
`*.mel.npy` sidecar), so the explanation matches the edge model's real decision.

```
Pi (detector) ──saves──►  evidence/<date>/<class>/<name>.wav
                                                  <name>.mel.npy   ← model input
                                                  <name>.json      ← class, probs, GPS, time
      │  rsync  (sync_events.sh)
      ▼
Mac (this app) ──► Grad-CAM (torch, models/custom_cnn/best_model.pt) ──► overlay PNG ──► web page
```

## Setup (on the Mac)

The base conda env already has torch + matplotlib + numpy. Add Flask:

```bash
/opt/anaconda3/bin/pip install flask
```

## Run

```bash
cd dashboard
/opt/anaconda3/bin/python app.py
```

That's it — the app **auto-syncs from the Pi in the background** (every 30 s) and the page
auto-refreshes every 15 s, so new detections appear on their own. No manual sync needed.

Open **http://localhost:8000** on the Mac, or **http://<mac-LAN-ip>:8000** from any device on the
same network.

> Defaults to port **8000** because macOS reserves 5000 for AirPlay Receiver. Override with
> `PORT=5050 python app.py`.

> **Auto-sync needs passwordless SSH** — run `ssh-copy-id alertreck@alertreck.local` once, or the
> background rsync will silently time out on the password prompt.

### Tuning auto-sync (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `ALERTRECK_SYNC` | `1` | `0` disables auto-sync (use `./sync_events.sh` manually instead) |
| `ALERTRECK_SYNC_INTERVAL` | `30` | seconds between pulls |
| `ALERTRECK_PI` | `alertreck@alertreck.local` | Pi SSH target |
| `ALERTRECK_REMOTE` | `~/alertreck/data/evidence/` | remote evidence dir |

`./sync_events.sh` still works for a manual one-off pull.

## What each card shows

- **Grad-CAM overlay** — mel spectrogram (magma) with the heatmap (jet) marking the time-frequency
  region that drove the prediction. Red = most influential.
- **Class + level + confidence**, detection time, and GPS (if the SIM808 had a fix).
- **Top-4 class probabilities** as bars.

## Notes

- Overlays are cached in `static/overlays/<name>.png`; delete that folder to force regeneration.
- The Grad-CAM target layer is the final encoder feature map (`encoder[3].block[5]`, 256 channels).
- `events/` and `static/overlays/` are generated at runtime — safe to gitignore.
- Config via env: `ALERTRECK_PI` (default `alertreck@alertreck.local`),
  `ALERTRECK_REMOTE` (default `~/alertreck/data/evidence/`).
