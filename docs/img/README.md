# Screenshot assets for the Capstone Final Deliverable

Drop the captured PNGs here using these exact filenames (referenced from the root `README.md`):

| Filename | What to capture |
|---|---|
| `test-mic.png` | I2S mic unit-test output — `[PASS]` device + non-zero RMS on a clap vs near-zero on silence. |
| `daemon-run.png` | `python3 -m alertrack.main` startup log ("System ready") + onset lines with non-zero RMS. |
| `live-alert-sms.png` | Split screen: terminal `[ALERT] threat_gunshot` + GPS fix, and the ranger phone receiving the SMS. |
| `dashboard-classes.png` | Grad-CAM dashboard event list showing several different classes, each with confidence + spectrogram. |
| `pi-timing.png` | Pi terminal showing per-inference latency (≤ 80 ms) — daemon debug log or a timing snippet. |

Tips: terminal font ≥ 18 pt, dashboard zoomed, crop tightly. PNG preferred over JPG for text legibility.
