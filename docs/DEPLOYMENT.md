# ALERTRACK — Raspberry Pi Deployment Guide

---

## Part 1 — Flash a Fresh SD Card

**On your Mac:**

1. Download **Raspberry Pi Imager** from [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. Insert SD card (16 GB minimum, 32 GB recommended)
3. Open Imager → **Choose OS** → `Raspberry Pi OS Lite (64-bit)` (no desktop needed)
4. **Before writing** — click the gear icon (⚙️) and configure:
   - Enable SSH ✓
   - Set username: `alertreck` / password: (your choice — avoid special characters)
   - Set Wi-Fi SSID + password (your hotspot or router)
   - **Set Wi-Fi country** (e.g. `RW`) — without it the radio stays disabled and the Pi never joins
   - Set hostname: `alertreck`
5. Click **Write** and wait

---

## Part 2 — Find the Pi's IP Address

After inserting the SD card and booting the Pi, wait ~60 seconds then try one of these:

**Option A — mDNS hostname (easiest):**
```bash
ping alertreck.local
```
The IP will show in the response, e.g. `64 bytes from 192.168.1.88`

**Option B — Scan the network:**
```bash
# Install nmap if needed: brew install nmap
nmap -sn 192.168.1.0/24 | grep -i raspberry
```
> Replace `192.168.1` with your actual prefix. Check yours with: `ipconfig getifaddr en0`

**Option C — From your router:**
Log into your router admin page (usually `192.168.1.1`) → connected devices → look for `alertreck`

**Option D — Connect a monitor and run:**
```bash
hostname -I
```

---

## Part 3 — SSH into the Pi

```bash
ssh alertreck@alertreck.local
# or use the IP directly (changes on DHCP reboot — prefer the hostname above)
ssh alertreck@192.168.1.88
```

**First connection only** — type `yes` when asked about authenticity.

Username: `alertreck`  ·  Password: whatever you set in the Imager.

> Modern Raspberry Pi OS (Bookworm) has **no default `pi` user** — the account is the username
> you set when flashing. If SSH says "Permission denied", you are almost certainly using the wrong
> username (not the wrong password); SSH prompts for a password even when the user doesn't exist.

**SSH refused?** SSH was not enabled during flashing. Fix from your Mac:
```bash
touch /Volumes/bootfs/ssh
```
Eject, reinsert SD card into Pi, reboot.

**Set up a `ssh alertreck` alias + passwordless login (one-time, recommended):**
```bash
# On your Mac — add the alias to ~/.ssh/config
cat >> ~/.ssh/config <<'EOF'

Host alertreck
    HostName alertreck.local
    User alertreck
    StrictHostKeyChecking accept-new
EOF

# Copy your key so you never type the password again (asks for it once)
ssh-copy-id alertreck
ssh alertreck 'echo connected'    # should print "connected" with no prompt
```
> **After a reflash** the Pi gets a new SSH identity. Clear the stale key first or SSH will refuse:
> `ssh-keygen -R alertreck.local && ssh-keygen -R <old-ip>`, then `ssh-copy-id alertreck` again.

> ⚠️ **Never pull power from a running Pi.** Yanking the cable (e.g. while swapping power supplies)
> can corrupt the SD card and leave it unbootable — recoverable only by reflashing. Always
> `sudo shutdown -h now` and wait for the green LED to stop before unplugging.

---

## Part 4 — Set Up the Pi Environment

Run these commands **on the Pi** (inside the SSH session):

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv ffmpeg git libportaudio2 libsndfile1

# Create project folder
mkdir -p ~/alertreck/models/custom_cnn
cd ~/alertreck

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install numpy librosa sounddevice soundfile onnxruntime pyserial psutil
```

---

## Part 5 — Export & Deploy the Model

**On your Mac**, run from inside the `alertreck` project folder.

> 📦 **Don't have the trained model?** The checkpoints, ONNX exports, and `results.json` are on
> Google Drive: [Alertreck Data, Dataset & Models](https://drive.google.com/drive/folders/1U9BwIUNQ8Snl5RxR8LHthWfdOc_EdcTM?usp=sharing).
> Download `models/custom_cnn/` to skip training and go straight to deployment.

> **Environment:** the export needs `torch` + `onnx` + `onnxruntime`. These live in the **base**
> conda env (`/opt/anaconda3/bin/python`), *not* the `alertreck` env (which only has the runtime
> deps librosa/numpy). Use the base interpreter for export.
>
> **Checkpoint path:** the trained checkpoint is `models/custom_cnn/best_model.pt`. The script
> defaults to `models/best_model.pt`, so you **must** pass `--model` explicitly or it will fail
> with "model not found".

```bash
# Step 1 — Export the trained PyTorch checkpoint to ONNX (traces at 301 frames, dynamic width)
/opt/anaconda3/bin/python scripts/export_model.py \
  --model models/custom_cnn/best_model.pt \
  --out   models/custom_cnn/alertreck_cnn.onnx
# Expect: "ONNX validation OK  output shape: (1, 7)"
# Output : models/custom_cnn/alertreck_cnn.onnx
#          input mel_spec = [batch, 1, 128, frames]  (frames dynamic; training shape = 301)

# Step 2 — Copy the model to the Pi
scp models/custom_cnn/alertreck_cnn.onnx alertreck@alertreck.local:~/alertreck/models/custom_cnn/

# Step 3 — Copy the daemon code to the Pi
rsync -av --exclude='__pycache__' alertrack/ alertreck@alertreck.local:~/alertreck/alertrack/
```

> **Verify the export (optional but recommended)** before copying:
> ```bash
> /opt/anaconda3/bin/python -c "import onnxruntime as ort, numpy as np; \
> s=ort.InferenceSession('models/custom_cnn/alertreck_cnn.onnx'); i=s.get_inputs()[0]; \
> print(i.name, i.shape); \
> print('301 OK', s.run(None,{i.name:np.random.randn(1,1,128,301).astype(np.float32)})[0].shape)"
> ```
> Should print `mel_spec ['batch', 1, 128, 'frames']` and `301 OK (1, 7)`.

---

## Part 6 — Test the System on the Pi

```bash
# SSH into the Pi
ssh alertreck@alertreck.local

cd ~/alertreck
source venv/bin/activate

# Test model loads correctly
python3 -c "from alertrack.inference.model import ONNXModel; ONNXModel()"

# Test audio preprocessor
python3 -c "from alertrack.audio.preprocess import test_preprocessor; test_preprocessor()"

# Run the full system (Ctrl+C to stop).
# Must run as a module (-m) from ~/alertreck — main.py uses package-relative imports,
# so `python3 alertrack/main.py` fails with "attempted relative import with no known parent package".
python3 -m alertrack.main
```

> Run with `ALERTRACK_DEBUG=true python3 -m alertrack.main` to print the predicted class +
> confidence and the raw RMS every inference cycle — essential for the calibration in Part 6A.

---

## Part 6A — Microphone Calibration (critical — do this before trusting results)

The single biggest source of wrong predictions is the **microphone input**, not the model. Two
problems must be fixed on every new mic/site:

### 1. Find the mic's ALSA card number
USB audio devices re-enumerate — the card number **can change between reboots**.
```bash
arecord -l        # note "card N:" for "USB PnP Sound Device"
```
Use that `N` as `plughw:N,0` below. (If `main.py` can't find the mic after a reboot, the card number
moved — re-check here and update `MIC_DEVICE_INDEX` in `config.py` if needed.)

### 2. Set the capture gain so nothing clips
A clipped signal is destroyed *before* the model sees it — gunshots, speech and chainsaw all collapse
into noise/“vehicle”. Set the gain for the **loudest** sound you expect; EBU normalisation boosts quiet
sounds back up, so erring low costs nothing.

```bash
alsamixer
#  F6 → select "USB PnP Sound Device"
#  F4 → capture view
#  lower "Mic" capture; if there's an "Auto Gain Control", press M to turn it OFF
#  Esc to exit
```

**More reliable: set it directly with `amixer`** (control names are stable; `alsamixer`'s
simple-control names sometimes don't match). For the common PCM2902 USB mic on card 1:
```bash
amixer -c 1 cget numid=4                 # 'Auto Gain Control'  — check state
amixer -c 1 cset numid=4 off             # AGC OFF (auto-gain drives the mic to clipping)
amixer -c 1 cset numid=3 4               # 'Mic Capture Volume' low (0–16 scale; ~4 is a good start)
sudo alsactl store                       # persist across reboots (asks for password)
```
Re-check `numid` mapping with `amixer -c 1 contents` if the values above don't match your device.

Verify with a record-and-measure loop — make your loudest test sound during the 3 s:
```bash
arecord -D plughw:1,0 -d 3 -f S16_LE -r 44100 -c1 /tmp/cal.wav
python3 -c "import wave,numpy as np; w=wave.open('/tmp/cal.wav'); x=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16)/32768; print(f'peak={abs(x).max():.3f} RMS={np.sqrt((x**2).mean()):.4f}')"
```
**Target: `peak` < ~0.85 on your loudest source.** `peak=1.000` = clipping → lower the gain more.
Speech is "peaky" — test it from a realistic ~1 m, not pressed against the mic.

### 3. Mains hum (50/60 Hz) — diagnose, then fix at the RIGHT layer
Field mics pick up mains hum that the clean training audio never had; left in, it reads as
`threat_vehicle`. Diagnose it after setting the gain:
```bash
python3 -c "
import wave, numpy as np
w=wave.open('/tmp/cal.wav'); sr=w.getframerate()
x=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
X=np.abs(np.fft.rfft(x*np.hanning(len(x)))); f=np.fft.rfftfreq(len(x),1/sr); tot=X.sum()+1e-9
print(f'dominant={f[np.argmax(X)]:.1f}Hz  RMS={np.sqrt((x**2).mean()):.4f}  0-120Hz={100*X[f<120].sum()/tot:.1f}%')
"
```
A quiet room should read **`dominant` ≠ ~50 Hz and idle `RMS` < ~0.02**.

**Decide the fix by severity (this is the lesson from the field build):**

| Symptom | Cause | Fix |
|---|---|---|
| `dominant ≈ 50 Hz`, idle RMS ~0.02–0.05, no clipping | mild hum | **Software** — raise `HPF_CUTOFF_HZ` to 110 in `config.py`. The high-pass runs only at serve time, removing hum absent from training (reduces skew, doesn't add it). |
| `dominant ≈ 50 Hz`, idle RMS **0.3–0.9**, peak hits 1.0 (**clipping**) | **power-borne hum** — the Pi's USB 5 V rail injects mains hum into a cheap USB sound card | **Hardware — software CANNOT fix a clipped input.** See below. |

**Power-borne hum (the severe case): fix the power path.**
The 50 Hz comb (peaks at 50/150/250/350 Hz) enters through the Pi's USB power. Confirm by
running the Pi off a **USB-C power bank** (no mains) — if the hum vanishes, it's mains-borne. Permanent fixes, best first:
1. **USB isolator** (ADUM3160/ADUM4160) between Pi and mic — **but the mic side must be powered
   from a source other than the Pi** (a separate clean 5 V / charger fed into the isolator's output
   side). A bus-powered isolator passes the Pi's noisy 5 V through and does nothing.
2. **Powered USB hub** with its own wall adapter — mic on the hub, not the Pi.
3. **Clean / well-grounded Pi PSU** (the official 5 V·3 A supply; avoid cheap ungrounded chargers).
4. **I2S MEMS mic (INMP441)** wired to GPIO — no USB power path at all; immune to this entirely.

Target after the fix: idle **RMS < 0.02** and no 50 Hz dominance. Only then will onset detection
and classification work — a real chainsaw/gunshot must rise ~10 dB above the noise floor.

---

## Part 7 — Enable Auto-Start on Boot

The service file (`alertrack/alertrack.service`) is already configured for the `alertreck` user,
the `/home/alertreck/alertreck` project path, and the `-m alertrack.main` module entrypoint —
no editing needed. Just install and enable it:

```bash
# On the Pi — install the systemd service
sudo cp ~/alertreck/alertrack/alertrack.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable alertrack
sudo systemctl start alertrack

# Verify it is running
sudo systemctl status alertrack
```

> If you edit the daemon code later, re-`rsync` from the Mac then
> `sudo systemctl restart alertrack`. If you change `alertrack.service` itself,
> re-copy it and run `sudo systemctl daemon-reload` before restarting.

---

## Part 8 — Common Commands

| Task | Command (on Pi) |
|---|---|
| Check system status | `sudo systemctl status alertrack` |
| Watch live logs | `journalctl -u alertrack -f` |
| Restart after code update | `sudo systemctl restart alertrack` |
| Stop the system | `sudo systemctl stop alertrack` |
| Disable auto-start | `sudo systemctl disable alertrack` |

| Task | Command (on Mac) |
|---|---|
| Update model on Pi | `scp models/custom_cnn/alertreck_cnn.onnx alertreck@alertreck.local:~/alertreck/models/custom_cnn/` |
| Update daemon code | `rsync -av --exclude='__pycache__' alertrack/ alertreck@alertreck.local:~/alertreck/alertrack/` |
| Open SSH session | `ssh alertreck@alertreck.local` |

---

## Part 9 — Troubleshooting

| Problem | Fix |
|---|---|
| `onnxruntime not found` | Activate venv first: `source ~/alertreck/venv/bin/activate` |
| `No module named sounddevice` | `sudo apt install -y libportaudio2` then `pip install sounddevice` |
| `Permission denied` on SSH | SSH not enabled — create empty `ssh` file on boot partition (see Part 3) |
| `Permission denied` (password rejected) | Wrong **username** — use `alertreck@…`, not `pi@…` (no default `pi` user on Bookworm) |
| Pi not found at `alertreck.local` | Find IP via `ping alertreck.local`, then: `ssh alertreck@192.168.1.88` |
| Model not found error | Re-export on Mac (see Part 5, pass `--model models/custom_cnn/best_model.pt`), then copy to Pi |
| No audio device found | Check mic is plugged in: `python3 -c "import sounddevice; print(sounddevice.query_devices())"` |
| Mic stopped working after reboot | USB card number changed — re-run `arecord -l` (Part 6A) |
| Everything predicts `threat_vehicle` | Mains hum — see Part 6A step 3 (raise `HPF_CUTOFF_HZ`) |
| Gunshot/speech not detected | Mic is clipping — lower capture gain (Part 6A step 2) |

---

## Part 10 — Connect the SIM808 GPS / GSM Module

The SIM808 provides **GPS coordinates** (for geotagging alerts) and **GSM SMS** (to notify rangers),
both over a single UART serial link. This part covers GPS; SMS uses the same wiring.

### 1. Wire the SIM808 to the Pi 4 GPIO

| SIM808 pin | Pi 4 pin | Notes |
|---|---|---|
| `VCC` (module power) | **External 5V / ≥2A supply**, *not* the Pi's 5V | SIM808 draws ~2A bursts when the GSM radio transmits — powering it from the Pi will brown-out and reboot the Pi |
| `GND` | Pi GND (pin 6) **and** the external supply GND | Common ground is required |
| `TXD` | Pi `GPIO15 / RXD` (pin 10) | SIM808 TX → Pi RX |
| `RXD` | Pi `GPIO14 / TXD` (pin 8) | SIM808 RX → Pi TX |

Attach the **GPS antenna** (the active patch antenna) and place it with a clear view of the sky —
GPS will not get a fix indoors.

### 2. Enable the Pi's UART

The default `/dev/ttyAMA0` (PL011) on a Pi 4 is used by Bluetooth and the serial login console. Free it:
```bash
sudo raspi-config
#  Interface Options → Serial Port
#    "login shell over serial?"  → NO
#    "serial port hardware enabled?" → YES
```
Then disable Bluetooth so it doesn't claim the good UART, and reboot:
```bash
echo "dtoverlay=disable-bt" | sudo tee -a /boot/firmware/config.txt
sudo systemctl disable hciuart
sudo reboot
```
After reboot the SIM808 is on `/dev/ttyAMA0` at 9600 baud — matching `SIM808_PORT` in `config.py`.

### 3. Verify the serial link and get a GPS fix

Add yourself to the `dialout` group (one-time) so you can open the port without sudo:
```bash
sudo usermod -aG dialout alertreck   # log out / back in afterwards
```
Talk to the module and power on its GPS:
```bash
# 'AT' should echo 'OK'. Then power GPS on and stream NMEA:
python3 -c "
import serial, time
s = serial.Serial('/dev/ttyAMA0', 9600, timeout=2)
for cmd in (b'AT', b'AT+CGNSPWR=1', b'AT+CGNSOUT=1'):
    s.write(cmd + b'\r\n'); time.sleep(0.5)
    print(cmd, '->', s.read(s.in_waiting or 64))
"
```
Each command should return `OK`. Once the antenna has a sky view, a fix takes **30 s – several minutes**
the first time (cold start). Test the project's reader directly:
```bash
cd ~/alertreck
python3 -m alertrack.sensors.gps      # prints coordinates once it has a fix
```

### 4. Enable GPS (and GSM) in config

Edit `alertrack/config.py` on the Pi:
```python
GPS_ENABLED = True            # was False
NOTIFY_GSM  = True            # was False — enables SMS alerts
SIM808_PORT = "/dev/ttyAMA0"  # 9600 baud
RANGER_PHONE_NUMBERS = [
    "+250795607062",          # ranger number(s) in E.164 format
]
```

**Validate the GSM link before relying on it** (run on the Pi). Registration first:
```bash
python3 -c "
import serial, time
s=serial.Serial('/dev/ttyAMA0',9600,timeout=1)
def at(c): s.reset_input_buffer(); s.write((c+'\r\n').encode()); time.sleep(1); print(c,'->',s.read(s.in_waiting or 256).decode(errors='ignore').strip())
at('ATE0'); at('AT+CSQ'); at('AT+CREG?'); at('AT+COPS?')
"
# Want: CSQ first number >=10 ; CREG: 0,1 or 0,5 (registered) ; COPS shows the carrier
```
Then send a real test SMS to the ranger via the project's own sender:
```bash
python3 -c "
from alertrack.alerts.notifier import _sim808_send_sms
print('SMS sent:', _sim808_send_sms('+250795607062','ALERTRECK test - system online','/dev/ttyAMA0',9600,15.0))
"
```
If `AT` returns nothing, swap the TX/RX wires. If SMS reboots the module, the SIM808 power supply
is too weak (add a 1000 µF cap across VBAT, or use a stronger external supply).

Restart and watch the logs for a fix:
```bash
sudo systemctl restart alertrack
journalctl -u alertrack -f
```
When the GPS has a lock, alerts change from `Location: UNKNOWN (GPS unavailable)` to real coordinates,
and an SMS is sent to each ranger number.

> **GPS/SMS share one UART.** The daemon uses the poll-on-demand `SIM808AT` reader (it opens the port
> only to read a fix, then releases it) precisely so SMS and GPS never hold `/dev/ttyAMA0` at the same
> time. Don't switch back to a continuous NMEA reader, or SMS sends will fail with "port busy".

> **Tip:** until the antenna has a clear sky view, set `SIMULATE_GPS = True` in `config.py` to test the
> alert/SMS flow with placeholder coordinates — useful for indoor bench testing.

---

## Quick Reference

```
Pi address     : alertreck.local   (current IP 192.168.1.88 — DHCP, may change on reboot)
Pi hostname    : alertreck.local
SSH user       : alertreck
Project path   : /home/alertreck/alertreck/
Model path     : /home/alertreck/alertreck/models/custom_cnn/alertreck_cnn.onnx
Venv path      : /home/alertreck/alertreck/venv/
Service name   : alertrack
Mic            : USB PnP Sound Device (ALSA card varies — check `arecord -l`)
SIM808 (GPS/GSM): /dev/ttyAMA0 @ 9600 baud  (UART; GPS=AT+CGNSPWR/CGNSOUT)
```
