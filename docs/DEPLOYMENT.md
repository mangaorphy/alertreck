# ALERTRACK  Raspberry Pi Deployment Guide

---

## Part 1  Flash a Fresh SD Card

**On your Mac:**

1. Download **Raspberry Pi Imager** from [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. Insert SD card (16 GB minimum, 32 GB recommended)
3. Open Imager → **Choose OS** → `Raspberry Pi OS Lite (64-bit)` (no desktop needed)
4. **Before writing**  click the gear icon (⚙️) and configure:
   - Enable SSH ✓
   - Set username: `alertreck` / password: (your choice  avoid special characters)
   - Set Wi-Fi SSID + password (your hotspot or router)
   - **Set Wi-Fi country** (e.g. `RW`)  without it the radio stays disabled and the Pi never joins
   - Set hostname: `alertreck`
5. Click **Write** and wait

---

## Part 2  Find the Pi's IP Address

After inserting the SD card and booting the Pi, wait ~60 seconds then try one of these:

**Option A  mDNS hostname (easiest):**
```bash
ping alertreck.local
```
The IP will show in the response, e.g. `64 bytes from 192.168.1.88`

**Option B  Scan the network:**
```bash
# Install nmap if needed: brew install nmap
nmap -sn 192.168.1.0/24 | grep -i raspberry
```
> Replace `192.168.1` with your actual prefix. Check yours with: `ipconfig getifaddr en0`

**Option C  From your router:**
Log into your router admin page (usually `192.168.1.1`) → connected devices → look for `alertreck`

**Option D  Connect a monitor and run:**
```bash
hostname -I
```

---

## Part 3  SSH into the Pi

```bash
ssh alertreck@alertreck.local
# or use the IP directly (changes on DHCP reboot  prefer the hostname above)
ssh alertreck@192.168.1.88
```

**First connection only**  type `yes` when asked about authenticity.

Username: `alertreck`  ·  Password: whatever you set in the Imager.

> Modern Raspberry Pi OS (Bookworm) has **no default `pi` user**  the account is the username
> you set when flashing. If SSH says "Permission denied", you are almost certainly using the wrong
> username (not the wrong password); SSH prompts for a password even when the user doesn't exist.

**SSH refused?** SSH was not enabled during flashing. Fix from your Mac:
```bash
touch /Volumes/bootfs/ssh
```
Eject, reinsert SD card into Pi, reboot.

**Set up a `ssh alertreck` alias + passwordless login (one-time, recommended):**
```bash
# On your Mac  add the alias to ~/.ssh/config
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

---

## Part 4  Set Up the Pi Environment

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

## Part 5  Export & Deploy the Model

**On your Mac**, run from inside the `alertreck` project folder.

>**Don't have the trained model?** The checkpoints, ONNX exports, and `results.json` are on
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
# Step 1  Export the trained PyTorch checkpoint to ONNX (traces at 301 frames, dynamic width)
/opt/anaconda3/bin/python scripts/export_model.py \
  --model models/custom_cnn/best_model.pt \
  --out   models/custom_cnn/alertreck_cnn.onnx
# Expect: "ONNX validation OK  output shape: (1, 7)"
# Output : models/custom_cnn/alertreck_cnn.onnx
#          input mel_spec = [batch, 1, 128, frames]  (frames dynamic; training shape = 301)

# Step 2  Copy the model to the Pi
scp models/custom_cnn/alertreck_cnn.onnx alertreck@alertreck.local:~/alertreck/models/custom_cnn/

# Step 3  Copy the daemon code to the Pi
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

## Part 6  Test the System on the Pi

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
# Must run as a module (-m) from ~/alertreck  main.py uses package-relative imports,
# so `python3 alertrack/main.py` fails with "attempted relative import with no known parent package".
python3 -m alertrack.main
```

> Run with `ALERTRACK_DEBUG=true python3 -m alertrack.main` to print the predicted class +
> confidence and the raw RMS every inference cycle  essential for the calibration in Part 6A.

---

## Part 6A  Microphone Setup & Calibration (INMP441 I2S)

The mic is the single biggest source of wrong predictions. Alertreck uses the **INMP441 I2S MEMS**
microphone  a digital mic on the Pi's I2S bus: no USB, no analog gain control, and no mains hum.

### 1. Wire it and enable I2S
Full wiring table, `config.txt` overlay, and the ALSA config are in
**[../alertrack/deploy/INMP441_SETUP.md](../alertrack/deploy/INMP441_SETUP.md)**. In short:

- Wire VDD→3V3, GND→GND, SCK→GPIO18, WS→GPIO19, SD→GPIO20, L/R→GND (left channel). **3.3 V only.**
- Add to `/boot/firmware/config.txt`: `dtparam=i2s=on` and `dtoverlay=googlevoicehat-soundcard`, then reboot.
- `sudo cp alertrack/deploy/asound.conf /etc/asound.conf` so the mic is the ALSA **default**
  (resampled 48→44.1 kHz, downmixed to mono).
- `config.py` already has `MIC_DEVICE_INDEX = None` (use the ALSA default).

Confirm the mic card and the SIM808 UART both came up:
```bash
arecord -l            # card "sndrpigooglevoi"
ls -l /dev/ttyAMA0    # SIM808 UART
```

### 2. Verify capture and level
The INMP441 is a quiet mic  that is expected. EBU-R128 normalisation rescales loudness before the
model, so the absolute level does not need tuning (there is no gain control). Just confirm it is alive:
```bash
# on the Pi: record 5 s (clap/speak during it), then check the level
arecord -D default -f S16_LE -r 44100 -c 1 -d 5 /tmp/cal.wav
python3 alertrack/deploy/test_i2s_mic.py        # prints peak / rms, writes test_i2s.wav
```
A working mic shows `peak`/`rms` clearly rising on sound and near-zero on silence. The measured ambient
floor is ≈ 0.0003 and real events ≈ 0.02.

### 3. Onset / silence thresholds (already calibrated for the INMP441)
Because the INMP441 is quieter than the old USB mic, the absolute energy gates in `config.py` were
lowered to match its noise floor (the old `0.01` sat *above* real events on this mic and silenced them):
```python
SILENCE_THRESHOLD = 0.0015   # skip inference below this raw RMS  (floor ~= 0.0003, events ~= 0.02)
ONSET_MIN_RMS     = 0.0015   # onset never fires below this absolute floor
ONSET_TRIGGER_DB  = 7.0      # relative gate over the adaptive floor (mic-agnostic, unchanged)
```
Nudge `0.0015` up toward `0.002` if idle silence ever triggers, or down to `0.001` if faint/distant
events are missed. To recalibrate from scratch, record ~10 s of true silence and set the gate a few×
above the measured per-frame floor.

> **No mains hum.** The INMP441 is digital, so the 50/60 Hz mains hum that plagued the USB mic is gone
>  along with the USB-power-isolator / powered-hub workarounds it required. The `HPF_CUTOFF_HZ`
> high-pass in `config.py` is now largely redundant (it still harmlessly removes wind rumble); set
> `HPF_ENABLED = False` and compare field detections if you prefer.

---

## Part 7  Enable Auto-Start on Boot

The service file (`alertrack/alertrack.service`) is already configured for the `alertreck` user,
the `/home/alertreck/alertreck` project path, and the `-m alertrack.main` module entrypoint 
no editing needed. Just install and enable it:

```bash
# On the Pi  install the systemd service
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

## Part 8  Common Commands

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

## Part 9  Troubleshooting

| Problem | Fix |
|---|---|
| `onnxruntime not found` | Activate venv first: `source ~/alertreck/venv/bin/activate` |
| `No module named sounddevice` | `sudo apt install -y libportaudio2` then `pip install sounddevice` |
| `Permission denied` on SSH | SSH not enabled  create empty `ssh` file on boot partition (see Part 3) |
| `Permission denied` (password rejected) | Wrong **username**  use `alertreck@…`, not `pi@…` (no default `pi` user on Bookworm) |
| Pi not found at `alertreck.local` | Find IP via `ping alertreck.local`, then: `ssh alertreck@192.168.1.88` |
| Model not found error | Re-export on Mac (see Part 5, pass `--model models/custom_cnn/best_model.pt`), then copy to Pi |
| No audio device found | I2S overlay didn't load  check `arecord -l` shows `sndrpigooglevoi`; confirm `dtparam=i2s=on` + `dtoverlay=googlevoicehat-soundcard` in `config.txt`, reboot (Part 6A) |
| Mic captures silence (peak ≈ 0) | Re-check INMP441 wiring  L/R→GND and the SD/WS/SCK pins; a loose SD wire is the usual cause (`../alertrack/deploy/INMP441_SETUP.md`) |
| Real events never trigger | INMP441 floor is low  confirm `SILENCE_THRESHOLD` / `ONSET_MIN_RMS = 0.0015` in `config.py` (Part 6A step 3) |
| `arecord` works but Python is silent | PortAudio opened the wrong device  set `MIC_DEVICE_INDEX = "googlevoicehat"` in `config.py` |

---

## Part 10  Connect the SIM808 GPS / GSM Module

The SIM808 provides **GPS coordinates** (for geotagging alerts) and **GSM SMS** (to notify rangers),
both over a single UART serial link. This part covers GPS; SMS uses the same wiring.

### 1. Wire the SIM808 to the Pi 4 GPIO

| SIM808 pin | Pi 4 pin | Notes |
|---|---|---|
| `VCC` (module power) | **External 5V / ≥2A supply**, *not* the Pi's 5V | SIM808 draws ~2A bursts when the GSM radio transmits  powering it from the Pi will brown-out and reboot the Pi |
| `GND` | Pi GND (pin 6) **and** the external supply GND | Common ground is required |
| `TXD` | Pi `GPIO15 / RXD` (pin 10) | SIM808 TX → Pi RX |
| `RXD` | Pi `GPIO14 / TXD` (pin 8) | SIM808 RX → Pi TX |

Attach the **GPS antenna** (the active patch antenna) and place it with a clear view of the sky 
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
After reboot the SIM808 is on `/dev/ttyAMA0` at 9600 baud  matching `SIM808_PORT` in `config.py`.

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
NOTIFY_GSM  = True            # was False  enables SMS alerts
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
> alert/SMS flow with placeholder coordinates  useful for indoor bench testing.

---

## Quick Reference

```
Pi address     : alertreck.local   (current IP 192.168.1.88  DHCP, may change on reboot)
Pi hostname    : alertreck.local
SSH user       : alertreck
Project path   : /home/alertreck/alertreck/
Model path     : /home/alertreck/alertreck/models/custom_cnn/alertreck_cnn.onnx
Venv path      : /home/alertreck/alertreck/venv/
Service name   : alertrack
Mic            : INMP441 I2S MEMS (ALSA card `sndrpigooglevoi`; default via /etc/asound.conf)
SIM808 (GPS/GSM): /dev/ttyAMA0 @ 9600 baud  (UART; GPS=AT+CGNSPWR/CGNSOUT)
```
