# ALERTRACK — Raspberry Pi Deployment Guide

---

## Part 1 — Flash a Fresh SD Card

**On your Mac:**

1. Download **Raspberry Pi Imager** from [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. Insert SD card (16 GB minimum, 32 GB recommended)
3. Open Imager → **Choose OS** → `Raspberry Pi OS Lite (64-bit)` (no desktop needed)
4. **Before writing** — click the gear icon (⚙️) and configure:
   - Enable SSH ✓
   - Set username: `pi` / password: `alertreck2024`
   - Set Wi-Fi SSID + password (your hotspot or router)
   - Set hostname: `alertreck`
5. Click **Write** and wait

---

## Part 2 — Find the Pi's IP Address

After inserting the SD card and booting the Pi, wait ~60 seconds then try one of these:

**Option A — mDNS hostname (easiest):**
```bash
ping alertreck.local
```
The IP will show in the response, e.g. `64 bytes from 172.16.17.43`

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
ssh pi@alertreck.local
# or use the IP directly
ssh pi@172.16.17.43
```

**First connection only** — type `yes` when asked about authenticity.

Password: `alertreck2024` (or whatever you set during flashing)

**Default credentials (older Pi OS):**
- Username: `pi`
- Password: `raspberry`

**SSH refused?** SSH was not enabled during flashing. Fix from your Mac:
```bash
touch /Volumes/bootfs/ssh
```
Eject, reinsert SD card into Pi, reboot.

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

**On your Mac**, run from inside the `alertreck` project folder:

```bash
# Step 1 — Export the trained PyTorch model to ONNX
python3 scripts/export_model.py
# Output: models/custom_cnn/alertreck_cnn.onnx

# Step 2 — Copy the model to the Pi
scp models/custom_cnn/alertreck_cnn.onnx pi@172.16.17.43:~/alertreck/models/custom_cnn/

# Step 3 — Copy the daemon code to the Pi
rsync -av --exclude='__pycache__' alertrack/ pi@172.16.17.43:~/alertreck/alertrack/
```

---

## Part 6 — Test the System on the Pi

```bash
# SSH into the Pi
ssh pi@172.16.17.43

cd ~/alertreck
source venv/bin/activate

# Test model loads correctly
python3 -c "from alertrack.inference.model import ONNXModel; ONNXModel()"

# Test audio preprocessor
python3 -c "from alertrack.audio.preprocess import test_preprocessor; test_preprocessor()"

# Run the full system (Ctrl+C to stop)
python3 alertrack/main.py
```

---

## Part 7 — Enable Auto-Start on Boot

```bash
# On the Pi — install the systemd service
sudo cp ~/alertreck/alertrack/alertrack.service /etc/systemd/system/

# Edit the service to point to your virtual environment
sudo nano /etc/systemd/system/alertrack.service
```

Change the `ExecStart` line to:
```
ExecStart=/home/pi/alertreck/venv/bin/python3 /home/pi/alertreck/alertrack/main.py
```

Save (`Ctrl+O`, Enter, `Ctrl+X`) then enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable alertrack
sudo systemctl start alertrack

# Verify it is running
sudo systemctl status alertrack
```

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
| Update model on Pi | `scp models/custom_cnn/alertreck_cnn.onnx pi@172.16.17.43:~/alertreck/models/custom_cnn/` |
| Update daemon code | `rsync -av --exclude='__pycache__' alertrack/ pi@172.16.17.43:~/alertreck/alertrack/` |
| Open SSH session | `ssh pi@172.16.17.43` |

---

## Part 9 — Troubleshooting

| Problem | Fix |
|---|---|
| `onnxruntime not found` | Activate venv first: `source ~/alertreck/venv/bin/activate` |
| `No module named sounddevice` | `sudo apt install -y libportaudio2` then `pip install sounddevice` |
| `Permission denied` on SSH | SSH not enabled — create empty `ssh` file on boot partition (see Part 3) |
| Pi not found at `alertreck.local` | Use IP address directly: `ssh pi@172.16.17.43` |
| Model not found error | Run `python3 scripts/export_model.py` on Mac first, then copy to Pi |
| No audio device found | Check mic is plugged in: `python3 -c "import sounddevice; print(sounddevice.query_devices())"` |

---

## Quick Reference

```
Pi IP address  : 172.16.17.43
Pi hostname    : alertreck.local
SSH user       : pi
Project path   : ~/alertreck/
Model path     : ~/alertreck/models/custom_cnn/alertreck_cnn.onnx
Venv path      : ~/alertreck/venv/
Service name   : alertrack
```
