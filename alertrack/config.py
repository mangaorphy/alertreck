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

# Create directories if they don't exist
for directory in [DATA_DIR, ALERTS_DIR, EVIDENCE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Choose which model to use: 'yamnet', 'custom_cnn', or 'mobilenet'
MODEL_TYPE = os.getenv("ALERTRACK_MODEL", "yamnet")

MODEL_PATHS = {
    "yamnet": MODEL_DIR / "yamnet_e2e" / "yamnet_classifier.tflite",
    "custom_cnn": MODEL_DIR / "custom_cnn" / "threat_detection_custom_cnn.tflite",
    "mobilenet": MODEL_DIR / "threat_detection.tflite"
}

MODEL_PATH = MODEL_PATHS.get(MODEL_TYPE)

# Class names (must match training order)
CLASS_NAMES = ['BACKGROUND', 'THREAT_CONTEXT', 'THREAT']

# Threat class indices
BACKGROUND_IDX = 0
THREAT_CONTEXT_IDX = 1
THREAT_IDX = 2

# ============================================================================
# AUDIO CONFIGURATION
# ============================================================================
SAMPLE_RATE = 16000  # Hz (YAMNet uses 16kHz)
AUDIO_DURATION = 10.0  # seconds
BUFFER_SIZE = int(SAMPLE_RATE * AUDIO_DURATION)  # samples
CHANNELS = 1  # Mono

# Rolling buffer for continuous capture
CHUNK_SIZE = 1024  # samples per read
UPDATE_INTERVAL = 1.0  # seconds between inference runs

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================
# These must match your training preprocessing!
MEL_BANDS = 128
FFT_SIZE = 2048
HOP_LENGTH = 512
WINDOW_TYPE = 'hann'
FMIN = 20  # Hz
FMAX = SAMPLE_RATE // 2  # Nyquist frequency

# Target shape for model input (channels, height, width)
# Adjust based on your actual model input shape
INPUT_SHAPE = (128, 313, 1)  # (mel_bands, time_steps, channels)

# ============================================================================
# THREAT DETECTION THRESHOLDS
# ============================================================================
THREAT_THRESHOLD = 0.85  # 85% confidence for THREAT class
THREAT_CONTEXT_THRESHOLD = 0.75  # 75% confidence for THREAT_CONTEXT
BACKGROUND_THRESHOLD = 0.50  # Below this, everything is uncertain

# Cooldown: prevent repeated alerts for same threat type
COOLDOWN_SECONDS = 300  # 5 minutes

# ============================================================================
# GPS CONFIGURATION
# ============================================================================
GPS_ENABLED = True
GPS_PORT = "/dev/ttyUSB0"  # Common for USB GPS modules
GPS_BAUDRATE = 9600
GPS_TIMEOUT = 5.0  # seconds

# ============================================================================
# ALERT CONFIGURATION
# ============================================================================
ALERT_RETENTION_DAYS = 90  # Keep alerts for 90 days

# Notification methods (for future implementation)
NOTIFY_CONSOLE = True
NOTIFY_LORA = False  # Stub
NOTIFY_GSM = False  # Stub
NOTIFY_SATELLITE = False  # Stub

# ============================================================================
# EVIDENCE STORAGE
# ============================================================================
SAVE_EVIDENCE_AUDIO = True
AUDIO_FORMAT = "wav"  # or "mp3", "flac"
MAX_EVIDENCE_STORAGE_GB = 100  # Auto-cleanup old files if exceeded

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = LOGS_DIR / "alertrack.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# ============================================================================
# PERFORMANCE & FAULT TOLERANCE
# ============================================================================
MAX_INFERENCE_TIME = 2.0  # seconds (warn if exceeded)
MICROPHONE_RECONNECT_DELAY = 5.0  # seconds
GPS_RECONNECT_DELAY = 10.0  # seconds
WATCHDOG_TIMEOUT = 30.0  # seconds (restart if system hangs)

# ============================================================================
# DEBUG & TESTING
# ============================================================================
DEBUG_MODE = os.getenv("ALERTRACK_DEBUG", "false").lower() == "true"
SIMULATE_GPS = False  # Use fake GPS data for testing
SIMULATE_AUDIO = False  # Use test audio files instead of microphone

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration and check for common issues."""
    issues = []
    
    # Check model exists
    if not MODEL_PATH or not MODEL_PATH.exists():
        issues.append(f"Model not found: {MODEL_PATH}")
    
    # Check directories
    if not MODEL_DIR.exists():
        issues.append(f"Model directory not found: {MODEL_DIR}")
    
    # Check thresholds
    if not 0 <= THREAT_THRESHOLD <= 1:
        issues.append(f"Invalid THREAT_THRESHOLD: {THREAT_THRESHOLD}")
    
    if not 0 <= THREAT_CONTEXT_THRESHOLD <= 1:
        issues.append(f"Invalid THREAT_CONTEXT_THRESHOLD: {THREAT_CONTEXT_THRESHOLD}")
    
    # Check sample rate
    if SAMPLE_RATE not in [16000, 22050, 44100, 48000]:
        issues.append(f"Unusual SAMPLE_RATE: {SAMPLE_RATE}")
    
    return issues


if __name__ == "__main__":
    print("ALERTRACK Configuration")
    print("=" * 60)
    print(f"Device ID: {DEVICE_ID}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Audio Duration: {AUDIO_DURATION}s")
    print(f"Threat Threshold: {THREAT_THRESHOLD}")
    print(f"Threat Context Threshold: {THREAT_CONTEXT_THRESHOLD}")
    print(f"GPS Enabled: {GPS_ENABLED}")
    print("=" * 60)
    
    # Validate
    issues = validate_config()
    if issues:
        print("\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Configuration valid!")
