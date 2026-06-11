"""
ALERTRACK Configuration
=======================
Central configuration for the anti-poaching edge system.
All thresholds, paths, and constants are defined here.
"""

import os
from pathlib import Path

# ============================================================================
# SYSTEM IDENTIFICATION
# ============================================================================
DEVICE_ID = os.getenv("ALERTRACK_DEVICE_ID", "ALERTRACK_001")
DEVICE_LOCATION = os.getenv("ALERTRACK_LOCATION", "UNKNOWN_RESERVE")

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ALERTS_DIR = DATA_DIR / "alerts"
EVIDENCE_DIR = DATA_DIR / "evidence"
LOGS_DIR = DATA_DIR / "logs"

for directory in [DATA_DIR, ALERTS_DIR, EVIDENCE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_TYPE = os.getenv("ALERTRACK_MODEL", "custom_cnn")

MODEL_PATHS = {
    "custom_cnn": MODEL_DIR / "custom_cnn" / "alertreck_cnn.onnx",
}

MODEL_PATH = MODEL_PATHS.get(MODEL_TYPE)

# ── Fine-grained 7-class labels (must match training order) ──────────────────
CLASS_NAMES = [
    "background_animals",    # 0  — no alert
    "background_wind_rain",  # 1  — no alert
    "threat_chainsaw",       # 2  — alert HIGH
    "threat_dog",            # 3  — alert MEDIUM
    "threat_gunshot",        # 4  — alert HIGH
    "threat_human",          # 5  — alert HIGH
    "threat_vehicle",        # 6  — alert HIGH
]
CLASS_LABELS = CLASS_NAMES   # alias used by main.py
N_CLASSES = len(CLASS_NAMES)

# Classes that trigger alerts and their threat level
THREAT_CONFIG = {
    # class_name          : (threshold, level, cooldown_seconds)
    "threat_chainsaw":     (0.60, "HIGH",   300),
    "threat_dog":          (0.60, "MEDIUM", 300),
    "threat_gunshot":      (0.60, "HIGH",    60),
    "threat_human":        (0.60, "HIGH",   300),
    "threat_vehicle":      (0.60, "HIGH",   300),
}

# Background classes — never trigger alerts
BACKGROUND_CLASSES = {"background_animals", "background_wind_rain"}

# ============================================================================
# AUDIO CONFIGURATION  (must match scripts/audio_preprocessing.py)
# ============================================================================
SAMPLE_RATE   = 44_100   # Hz
CLIP_SECONDS  = 3.0      # seconds per inference window
BUFFER_SIZE   = int(SAMPLE_RATE * CLIP_SECONDS)   # 132 300 samples
CHANNELS      = 1        # mono
CHUNK_SIZE    = 1024     # samples per sounddevice read

# USB microphone device index — find yours with:
#   python3 -c "import sounddevice; print(sounddevice.query_devices())"
# Set to None to use system default (not recommended on Pi)
MIC_DEVICE_INDEX = 1  # e.g. set to 5 after running query_devices()

# ============================================================================
# PREPROCESSING CONFIGURATION  (must match scripts/audio_preprocessing.py)
# ============================================================================
N_MELS      = 128
N_FFT       = 2048
WIN_LENGTH  = 1102   # 25 ms @ 44.1 kHz
HOP_STFT    = 441    # 10 ms @ 44.1 kHz
FMIN        = 0
FMAX        = SAMPLE_RATE // 2   # 22 050 Hz
# Resulting spectrogram shape: 1 + floor(132300 / 441) = 301 frames (librosa center=True)
INPUT_SHAPE = (1, N_MELS, 301)   # (C, H, W) — channels-first, PyTorch convention

# Silence gate — skip inference if raw RMS is below this level
# Prevents amplified electronic noise from being classified as a threat.
# Tune this up if the mic is noisy, down if real sounds are being skipped.
SILENCE_THRESHOLD = 0.01    # raise if mic noise still triggers; lower if real sounds are skipped

# ── Mains-hum high-pass (field-mic fix, NOT part of training) ────────────────
# USB mics inject 50 Hz mains hum (60 Hz in the Americas) that the clean training
# audio never had. Left in, it dominates the low band and the model reads the
# rumble as `threat_vehicle`. A high-pass removes it, moving the served signal
# CLOSER to the (hum-free) training distribution. Applied before EBU normalisation.
HPF_ENABLED   = True
HPF_CUTOFF_HZ = 90.0    # pass above this; 50 Hz hum is fully removed. Set 110 if 100 Hz harmonic persists.
HPF_WIDTH_HZ  = 30.0    # raised-cosine transition width below the cutoff (reduces ringing)

# ============================================================================
# INFERENCE LOOP
# ============================================================================
INFERENCE_INTERVAL = 1.5     # seconds between inference runs (legacy timer mode)
STATS_INTERVAL     = 3600    # seconds between statistics log lines

# ── Onset-triggered inference ────────────────────────────────────────────────
# When enabled, the system does NOT classify on a blind timer. It watches for an
# energy onset above the adaptive background floor and classifies the 3 s window
# around the event — more precise, less CPU, and the event is centred in the window.
# Set ONSET_ENABLED = False to fall back to the fixed-interval timer loop.
ONSET_ENABLED       = True
ONSET_POLL_INTERVAL = 0.25   # seconds between onset checks (lower = more responsive)
ONSET_TRIGGER_DB    = 10.0   # dB above the adaptive noise floor required to fire
ONSET_REFRACTORY_S  = 2.0    # minimum seconds between triggers (debounce)
ONSET_SETTLE_S      = 1.0    # wait after onset so the event centres in the 3 s buffer
ONSET_FLOOR_ALPHA   = 0.10   # adaptive-floor EMA rate per poll (higher = faster tracking)
ONSET_RECENT_S      = 0.5    # window of "recent" audio used for the current level
ONSET_FRAME_MS      = 50     # frame size (ms) for energy measurement
ONSET_MIN_RMS       = 0.01   # absolute floor — below this is silence, never triggers

# ============================================================================
# SIM808 CONFIGURATION  (GSM SMS + GPS — shared UART, no data plan required)
# ============================================================================
# Hardware: SIM808 GSM/GPRS/GPS module connected to Pi 4 via UART serial
# GSM network: MTN or Airtel Rwanda (prepaid SIM, SMS only)
# SIM808 AT command port (same UART for SMS and GPS AT commands)
SIM808_PORT     = "/dev/ttyAMA0"   # Pi 4 UART; may be /dev/ttyUSB0 via USB-serial adapter
SIM808_BAUDRATE = 9600
SIM808_TIMEOUT  = 10.0             # seconds to wait for AT command response

# Ranger phone numbers for SMS alerts (E.164 format)
RANGER_PHONE_NUMBERS: list[str] = [
    # "+25078XXXXXXX",   # uncomment and fill in ranger numbers
]

# SMS retry configuration
SMS_MAX_RETRIES = 3
SMS_RETRY_DELAY = 5.0   # seconds between retry attempts

# ============================================================================
# GPS CONFIGURATION  (SIM808 built-in GPS, accessed via AT commands or NMEA)
# ============================================================================
# Enable NMEA output on SIM808: AT+CGNSPWR=1  then  AT+CGNSOUT=1
# The NMEA sentences ($GPGGA, $GPRMC) are then readable on SIM808_PORT
GPS_ENABLED  = False   # set True when SIM808 is physically connected
ENABLE_GPS   = GPS_ENABLED    # alias used by main.py
GPS_PORT     = SIM808_PORT    # SIM808 NMEA output shares UART with AT commands
GPS_BAUDRATE = SIM808_BAUDRATE
GPS_TIMEOUT  = 5.0

# ============================================================================
# ALERT CONFIGURATION
# ============================================================================
ALERT_RETENTION_DAYS = 90
NOTIFY_CONSOLE   = True
NOTIFY_LORA      = False
NOTIFY_GSM       = False
NOTIFY_SATELLITE = False

# ============================================================================
# EVIDENCE STORAGE
# ============================================================================
SAVE_EVIDENCE_AUDIO       = True
AUDIO_FORMAT              = "wav"
MAX_EVIDENCE_STORAGE_GB   = 100

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL       = "INFO"
LOG_FILE        = LOGS_DIR / "alertrack.log"
LOG_MAX_BYTES   = 10 * 1024 * 1024   # 10 MB
LOG_BACKUP_COUNT = 5

# ============================================================================
# PERFORMANCE & FAULT TOLERANCE
# ============================================================================
MAX_INFERENCE_TIME        = 2.0
MICROPHONE_RECONNECT_DELAY = 5.0
GPS_RECONNECT_DELAY        = 10.0
WATCHDOG_TIMEOUT           = 30.0

# ============================================================================
# DEBUG & TESTING
# ============================================================================
DEBUG_MODE     = os.getenv("ALERTRACK_DEBUG", "false").lower() == "true"
SIMULATE_GPS   = False
SIMULATE_AUDIO = False


# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    issues = []

    if not MODEL_PATH or not MODEL_PATH.exists():
        issues.append(f"Model not found: {MODEL_PATH}  (run: python3 scripts/export_model.py)")

    if not MODEL_DIR.exists():
        issues.append(f"Model directory not found: {MODEL_DIR}")

    for name, (thresh, level, cooldown) in THREAT_CONFIG.items():
        if not 0 < thresh <= 1:
            issues.append(f"Invalid threshold for {name}: {thresh}")

    if SAMPLE_RATE not in {16000, 22050, 44100, 48000}:
        issues.append(f"Unusual SAMPLE_RATE: {SAMPLE_RATE}")

    return issues


if __name__ == "__main__":
    print("ALERTRACK Configuration")
    print("=" * 60)
    print(f"Device ID    : {DEVICE_ID}")
    print(f"Model Type   : {MODEL_TYPE}")
    print(f"Model Path   : {MODEL_PATH}")
    print(f"Sample Rate  : {SAMPLE_RATE} Hz")
    print(f"Clip Length  : {CLIP_SECONDS}s  ({BUFFER_SIZE} samples)")
    print(f"Classes ({N_CLASSES}):")
    for i, name in enumerate(CLASS_NAMES):
        cfg = THREAT_CONFIG.get(name)
        tag = f"thresh={cfg[0]}  level={cfg[1]}  cooldown={cfg[2]}s" if cfg else "no alert"
        print(f"  [{i}] {name:<26}  {tag}")
    print("=" * 60)

    issues = validate_config()
    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration valid.")
