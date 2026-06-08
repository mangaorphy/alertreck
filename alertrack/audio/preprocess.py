"""
Audio Preprocessing Module
===========================
Converts a raw 3-second mono waveform to a log-mel spectrogram.

Parameters MUST exactly match scripts/audio_preprocessing.py (extract_mel + ebu_r128_normalize):
  SR         = 44 100 Hz
  N_MELS     = 128
  N_FFT      = 2 048
  WIN_LENGTH = 1 102   (25 ms @ 44.1 kHz)
  HOP_STFT   = 441     (10 ms @ 44.1 kHz)
  window     = "hann"
  fmax       = 22 050 Hz   (fmin defaults to 0)
  Loudness   = EBU R128 → −23 dBFS, clipped to [-1, 1]   (NOT plain RMS scaling)
  Output     = librosa.power_to_db(S, ref=np.max)   — dB relative to clip max
  Shape      = (1, 128, 301)  channels-first (PyTorch convention)

Mains-hum high-pass (the one deliberate deviation from training): real USB mics inject
50/60 Hz mains hum that the clean training audio never had. Removing it does NOT add
train/serve skew — it removes content that was absent in training, moving the served
signal closer to the training distribution. Controlled by HPF_* in config.py.
"""

import numpy as np
import librosa

from ..config import (
    SAMPLE_RATE, BUFFER_SIZE, N_MELS, N_FFT, WIN_LENGTH, HOP_STFT, FMAX,
    INPUT_SHAPE, DEBUG_MODE, SILENCE_THRESHOLD,
    HPF_ENABLED, HPF_CUTOFF_HZ, HPF_WIDTH_HZ,
)

# EBU R128 target loudness — must match EBU_TARGET in scripts/audio_preprocessing.py
_EBU_TARGET = 10 ** (-23.0 / 20.0)   # −23 dBFS ≈ 0.07079


def _ebu_r128_normalize(audio: np.ndarray) -> np.ndarray:
    """Scale to −23 dBFS and clip to [-1, 1] — identical to the training pipeline."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    return np.clip(audio * (_EBU_TARGET / rms), -1.0, 1.0).astype(np.float32)


def _highpass(audio: np.ndarray, sr: int,
              cutoff: float = HPF_CUTOFF_HZ, width: float = HPF_WIDTH_HZ) -> np.ndarray:
    """Zero-phase FFT high-pass with a raised-cosine transition (numpy-only, no scipy).

    Gain is 0 below (cutoff - width), ramps smoothly up to 1 at cutoff, and 1 above.
    Removes the 50/60 Hz mains-hum fundamental while preserving everything above the cutoff.
    The smooth transition avoids the time-domain ringing a brick-wall cut would cause.
    """
    n = len(audio)
    X = np.fft.rfft(audio)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    lo = max(0.0, cutoff - width)
    g = np.ones_like(f)
    g[f <= lo] = 0.0
    band = (f > lo) & (f < cutoff)
    g[band] = 0.5 * (1.0 - np.cos(np.pi * (f[band] - lo) / (cutoff - lo)))
    return np.fft.irfft(X * g, n=n).astype(np.float32)


class AudioPreprocessor:
    """Converts raw audio to a log-mel spectrogram ready for ONNXModel.predict()."""

    def __init__(self):
        print(f"AudioPreprocessor: SR={SAMPLE_RATE}  n_mels={N_MELS}  "
              f"n_fft={N_FFT}  win={WIN_LENGTH}  hop={HOP_STFT}  fmax={FMAX}")
        print(f"  Expected input : {BUFFER_SIZE} samples ({SAMPLE_RATE/1e3:.1f} kHz × 3 s)")
        print(f"  Loudness norm  : EBU R128 −23 dBFS (target {_EBU_TARGET:.5f})")
        print(f"  Output shape   : {INPUT_SHAPE}")

    def preprocess(self, audio: np.ndarray) -> np.ndarray | None:
        """
        Args:
            audio: float32 mono waveform, ~132 300 samples at 44.1 kHz

        Returns:
            np.ndarray shape (1, 128, 301) float32, or None on error
        """
        try:
            audio = audio.flatten().astype(np.float32)

            # Silence gate — skip inference if no real signal (uses raw RMS, pre-normalisation)
            rms = np.sqrt(np.mean(audio ** 2))
            if DEBUG_MODE:
                print(f"RMS={rms:.6f}  threshold={SILENCE_THRESHOLD}")
            if rms < SILENCE_THRESHOLD:
                if DEBUG_MODE:
                    print(f"Silence gate: skipping (RMS below threshold)")
                return None

            # Pad / trim to exactly BUFFER_SIZE samples, then normalise the fixed clip
            if len(audio) < BUFFER_SIZE:
                audio = np.pad(audio, (0, BUFFER_SIZE - len(audio)))
            else:
                audio = audio[:BUFFER_SIZE]

            # Remove 50/60 Hz mains hum BEFORE normalisation, so EBU scales the real
            # signal rather than the hum. Off → identical to training; on → field-mic fix.
            if HPF_ENABLED:
                audio = _highpass(audio, SAMPLE_RATE)

            # EBU R128 loudness normalisation — exactly as in training
            audio = _ebu_r128_normalize(audio)

            # Log-mel spectrogram — parameters identical to training extract_mel()
            S = librosa.feature.melspectrogram(
                y=audio,
                sr=SAMPLE_RATE,
                n_mels=N_MELS,
                n_fft=N_FFT,
                win_length=WIN_LENGTH,
                hop_length=HOP_STFT,
                window="hann",
                fmax=FMAX,
            )
            mel_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
            # shape: (128, 301)

            # Add channel dimension → (1, 128, 301)
            mel_db = np.expand_dims(mel_db, axis=0)

            if DEBUG_MODE:
                print(f"Preprocess: audio {audio.shape} → mel {mel_db.shape}  "
                      f"[{mel_db.min():.1f}, {mel_db.max():.1f}] dB")

            return mel_db

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None


def test_preprocessor():
    print("\nTesting AudioPreprocessor...")
    print("=" * 60)

    proc = AudioPreprocessor()
    audio = np.random.randn(BUFFER_SIZE).astype(np.float32)

    import time
    t0 = time.time()
    out = proc.preprocess(audio)
    elapsed = time.time() - t0

    if out is not None:
        print(f"Output shape   : {out.shape}  (expected {INPUT_SHAPE})")
        print(f"Value range    : [{out.min():.1f}, {out.max():.1f}] dB")
        print(f"Processing time: {elapsed*1000:.1f} ms")
        match = out.shape == INPUT_SHAPE
        print(f"Shape OK       : {match}")
    else:
        print("Preprocessing failed.")
    print("=" * 60)


if __name__ == "__main__":
    test_preprocessor()
