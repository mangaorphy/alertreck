"""
Audio Preprocessing Module
===========================
Converts a raw 3-second mono waveform to a log-mel spectrogram.

Parameters MUST exactly match scripts/audio_preprocessing.py:
  SR       = 44 100 Hz
  N_MELS   = 128
  N_FFT    = 2 048
  HOP_STFT = 512
  FMAX     = 22 050 Hz
  Output   = librosa.power_to_db(S, ref=np.max)   — raw dB, not normalised
  Shape    = (1, 128, 259)  channels-first (PyTorch convention)
"""

import numpy as np
import librosa
from scipy import signal as scipy_signal

from ..config import (
    SAMPLE_RATE, BUFFER_SIZE, N_MELS, N_FFT, HOP_STFT, FMIN, FMAX,
    INPUT_SHAPE, DEBUG_MODE, SILENCE_THRESHOLD
)

# High-pass filter coefficients — removes 50 Hz electrical hum (built once at import)
_HPF_B, _HPF_A = scipy_signal.butter(4, 120 / (SAMPLE_RATE / 2), btype="high")


class AudioPreprocessor:
    """Converts raw audio to a log-mel spectrogram ready for ONNXModel.predict()."""

    def __init__(self):
        print(f"AudioPreprocessor: SR={SAMPLE_RATE}  n_mels={N_MELS}  "
              f"n_fft={N_FFT}  hop={HOP_STFT}  fmax={FMAX}")
        print(f"  Expected input : {BUFFER_SIZE} samples ({SAMPLE_RATE/1e3:.1f} kHz × 3 s)")
        print(f"  Output shape   : {INPUT_SHAPE}")

    def preprocess(self, audio: np.ndarray) -> np.ndarray | None:
        """
        Args:
            audio: float32 mono waveform, ~132 300 samples at 44.1 kHz

        Returns:
            np.ndarray shape (1, 128, 259) float32, or None on error
        """
        try:
            audio = audio.flatten().astype(np.float32)

            # Silence gate — skip inference if no real signal
            rms = np.sqrt(np.mean(audio ** 2))
            if DEBUG_MODE:
                print(f"RMS={rms:.6f}  threshold={SILENCE_THRESHOLD}")
            if rms < SILENCE_THRESHOLD:
                if DEBUG_MODE:
                    print(f"Silence gate: skipping (RMS below threshold)")
                return None

            # High-pass filter — removes 50/60 Hz electrical hum from mic
            audio = scipy_signal.filtfilt(_HPF_B, _HPF_A, audio).astype(np.float32)

            # RMS-normalise to 0.1 (same as training pipeline)
            audio = audio * (0.1 / rms)

            # Pad / trim to exactly BUFFER_SIZE samples
            if len(audio) < BUFFER_SIZE:
                audio = np.pad(audio, (0, BUFFER_SIZE - len(audio)))
            else:
                audio = audio[:BUFFER_SIZE]

            # Log-mel spectrogram — exactly as in training
            S = librosa.feature.melspectrogram(
                y=audio,
                sr=SAMPLE_RATE,
                n_mels=N_MELS,
                n_fft=N_FFT,
                hop_length=HOP_STFT,
                fmin=FMIN,
                fmax=FMAX,
            )
            mel_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
            # shape: (128, 259)

            # Add channel dimension → (1, 128, 259)
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
