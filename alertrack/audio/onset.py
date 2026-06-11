"""
Onset Detector
==============
Decides *when* to run classification, instead of classifying on a blind timer.

It tracks an adaptive background noise floor and fires only when the recent audio
energy rises sharply above that floor — i.e. when something actually happens. This:
  • cuts false positives on pure background,
  • saves CPU (no inference on silence),
  • lets the main loop wait briefly after the onset so the event is centred in the
    3 s window the model expects (rather than clipped at a window edge).

It is train/serve-safe: it only changes *which* 3 s window gets classified, never the
audio content fed to the model.
"""

import time
import numpy as np

from ..config import (
    SAMPLE_RATE, ONSET_TRIGGER_DB, ONSET_REFRACTORY_S, ONSET_FLOOR_ALPHA,
    ONSET_RECENT_S, ONSET_FRAME_MS, ONSET_MIN_RMS, DEBUG_MODE,
)


class OnsetDetector:
    """Adaptive energy-onset detector over the recorder's rolling buffer."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr         = sample_rate
        self.frame_len  = max(1, int(sample_rate * ONSET_FRAME_MS / 1000))
        self.recent_len = max(self.frame_len, int(sample_rate * ONSET_RECENT_S))
        self.trigger_db = ONSET_TRIGGER_DB
        self.refractory = ONSET_REFRACTORY_S
        self.alpha      = ONSET_FLOOR_ALPHA
        self.min_rms    = ONSET_MIN_RMS

        self.floor_db: float | None = None   # adaptive background level (dBFS)
        self.last_trigger = 0.0

        print(f"OnsetDetector: trigger=+{self.trigger_db:.0f}dB over floor  "
              f"refractory={self.refractory:.1f}s  recent={ONSET_RECENT_S:.2f}s")

    def _frame_db(self, x: np.ndarray) -> np.ndarray:
        """Per-frame RMS in dBFS for a slice of audio."""
        n = len(x) // self.frame_len
        if n == 0:
            rms = np.sqrt(np.mean(x ** 2)) if len(x) else 0.0
            return np.array([20.0 * np.log10(rms + 1e-10)])
        x = x[: n * self.frame_len].reshape(n, self.frame_len)
        rms = np.sqrt(np.mean(x ** 2, axis=1))
        return 20.0 * np.log10(rms + 1e-10)

    def check(self, audio: np.ndarray) -> tuple[bool, dict]:
        """
        Examine the most recent slice of the rolling buffer.

        Returns (triggered, info). `triggered` is True when the loudest recent frame
        rises `trigger_db` above the adaptive floor, the signal is above the absolute
        silence floor, and we are past the refractory window.
        """
        recent  = audio[-self.recent_len:]
        db      = self._frame_db(recent)
        current = float(db.max())     # loudest recent frame = "now"
        quiet   = float(db.min())     # quietest recent frame feeds the floor estimate

        # Adaptive background floor (slow EMA toward the quiet level)
        if self.floor_db is None:
            self.floor_db = quiet
        else:
            self.floor_db += self.alpha * (quiet - self.floor_db)

        recent_rms = float(np.sqrt(np.mean(recent ** 2)))
        margin     = current - self.floor_db
        now        = time.time()
        in_refractory = (now - self.last_trigger) < self.refractory

        triggered = (
            recent_rms >= self.min_rms
            and margin >= self.trigger_db
            and not in_refractory
        )
        if triggered:
            self.last_trigger = now

        if DEBUG_MODE:
            state = "TRIGGER" if triggered else ("refractory" if in_refractory else "")
            print(f"onset: cur={current:6.1f}dB floor={self.floor_db:6.1f}dB "
                  f"margin={margin:5.1f}dB rms={recent_rms:.3f}  {state}")

        return triggered, {
            "current_db": current,
            "floor_db":   self.floor_db,
            "margin_db":  margin,
            "recent_rms": recent_rms,
        }

    def get_stats(self) -> dict:
        return {
            "floor_db":     self.floor_db,
            "last_trigger": self.last_trigger,
        }
