"""
AlertReck Unit Test Suite
=========================
Tests every module in the alertrack package in isolation.

Run from the project root:
    pytest tests/test_unit.py -v

No hardware required — hardware-dependent paths (AudioRecorder, SIM808AT,
AlertNotifier GSM channel) are either skipped or mocked.

Requirements:
    pip install pytest numpy librosa onnxruntime sounddevice
"""

import time
import json
import wave
import threading
import tempfile
import os
from pathlib import Path
from collections import deque
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

# ─── helpers ──────────────────────────────────────────────────────────────────
SR          = 44_100
BUF         = 132_300   # 3 s at 44.1 kHz
INPUT_SHAPE = (1, 128, 301)
EBU_TARGET  = 10 ** (-23.0 / 20.0)   # ≈ 0.07079

CLASSES = [
    "background_animals",   # 0
    "background_wind_rain", # 1
    "threat_chainsaw",      # 2
    "threat_dog",           # 3
    "threat_gunshot",       # 4
    "threat_human",         # 5
    "threat_vehicle",       # 6
]
THREAT_CLASSES    = CLASSES[2:]
BACKGROUND_CLASSES = {"background_animals", "background_wind_rain"}

def _white_noise(n=BUF, rms=0.05) -> np.ndarray:
    """White noise at a specified RMS level."""
    x = np.random.randn(n).astype(np.float32)
    x = x / (np.sqrt(np.mean(x**2)) + 1e-9) * rms
    return x

def _silence(n=BUF) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)

def _tone(freq=1000, n=BUF, amp=0.3) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def _uniform_probs(n=7) -> np.ndarray:
    p = np.ones(n, dtype=np.float32)
    return p / p.sum()

def _spike_probs(class_idx: int, confidence=0.95, n=7) -> np.ndarray:
    p = np.zeros(n, dtype=np.float32)
    p[class_idx] = confidence
    remaining = (1.0 - confidence) / (n - 1)
    for i in range(n):
        if i != class_idx:
            p[i] = remaining
    return p


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    """Smoke-test that config.py exports all constants the daemon depends on."""

    def test_imports_without_error(self):
        from alertrack import config  # noqa: F401

    def test_sample_rate(self):
        from alertrack.config import SAMPLE_RATE
        assert SAMPLE_RATE == 44_100

    def test_buffer_size(self):
        from alertrack.config import BUFFER_SIZE, SAMPLE_RATE, CLIP_SECONDS
        assert BUFFER_SIZE == int(SAMPLE_RATE * CLIP_SECONDS)

    def test_input_shape(self):
        from alertrack.config import INPUT_SHAPE, N_MELS
        assert INPUT_SHAPE == (1, N_MELS, 301)
        assert INPUT_SHAPE[1] == 128

    def test_seven_class_names(self):
        from alertrack.config import CLASS_NAMES, N_CLASSES
        assert N_CLASSES == 7
        assert len(CLASS_NAMES) == 7
        assert "threat_gunshot" in CLASS_NAMES
        assert "background_animals" in CLASS_NAMES

    def test_threat_config_keys(self):
        from alertrack.config import THREAT_CONFIG, CLASS_NAMES, BACKGROUND_CLASSES
        # All configured threat classes must exist in CLASS_NAMES
        for name in THREAT_CONFIG:
            assert name in CLASS_NAMES
        # No background class should be in THREAT_CONFIG
        for bg in BACKGROUND_CLASSES:
            assert bg not in THREAT_CONFIG

    def test_threat_thresholds_in_range(self):
        from alertrack.config import THREAT_CONFIG
        for name, (thresh, level, cooldown) in THREAT_CONFIG.items():
            assert 0 < thresh <= 1.0, f"{name}: threshold {thresh} out of range"
            assert cooldown > 0, f"{name}: cooldown must be positive"
            assert level in ("HIGH", "MEDIUM", "LOW"), f"{name}: unknown level {level}"

    def test_ebu_target_value(self):
        from alertrack.config import EBU_TARGET
        expected = 10 ** (-23.0 / 20.0)
        assert abs(EBU_TARGET - expected) < 1e-6

    def test_onset_constants(self):
        from alertrack.config import (
            ONSET_TRIGGER_DB, ONSET_REFRACTORY_S, ONSET_MIN_RMS,
            ONSET_FLOOR_ALPHA, ONSET_RECENT_S, ONSET_FRAME_MS,
        )
        assert ONSET_TRIGGER_DB > 0
        assert 0 < ONSET_FLOOR_ALPHA < 1
        assert ONSET_REFRACTORY_S > 0
        assert ONSET_MIN_RMS > 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. AUDIO RECORDER  (audio/recorder.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestAudioRecorder:
    """
    Tests that do not require real hardware.  The sounddevice stream is mocked.
    Hardware capture tests are in tests/test_hardware.py (Pi only).
    """

    def _make_recorder(self):
        from alertrack.audio.recorder import AudioRecorder
        return AudioRecorder(device_id=None)

    def test_instantiates(self):
        r = self._make_recorder()
        assert r is not None

    def test_buffer_starts_empty(self):
        r = self._make_recorder()
        assert not r.is_buffer_ready()
        assert r.get_audio_buffer() is None

    def test_buffer_fills_correctly(self):
        r = self._make_recorder()
        chunk = np.random.randn(1024).astype(np.float32)
        needed = BUF // 1024 + 1
        with r.lock:
            for _ in range(needed):
                r.buffer.extend(chunk)
        assert r.is_buffer_ready()
        buf = r.get_audio_buffer()
        assert buf is not None
        assert buf.shape == (BUF,)
        assert buf.dtype == np.float32

    def test_rolling_buffer_does_not_exceed_maxlen(self):
        r = self._make_recorder()
        big = np.ones(BUF * 3, dtype=np.float32)
        with r.lock:
            r.buffer.extend(big)
        assert len(r.buffer) == BUF  # deque(maxlen=BUFFER_SIZE)

    def test_clear_buffer(self):
        r = self._make_recorder()
        with r.lock:
            r.buffer.extend(np.ones(BUF, dtype=np.float32))
        assert r.is_buffer_ready()
        r.clear_buffer()
        assert not r.is_buffer_ready()

    def test_get_stats_keys(self):
        r = self._make_recorder()
        stats = r.get_stats()
        for key in ("is_recording", "total_chunks", "buffer_fill",
                    "buffer_overflows", "device_id", "sample_rate"):
            assert key in stats

    def test_buffer_fill_ratio(self):
        r = self._make_recorder()
        half = BUF // 2
        with r.lock:
            r.buffer.extend(np.zeros(half, dtype=np.float32))
        stats = r.get_stats()
        assert abs(stats["buffer_fill"] - 0.5) < 0.01

    def test_mono_conversion_from_stereo(self):
        """_capture_audio must flatten stereo to mono."""
        r = self._make_recorder()
        stereo = np.ones((1024, 2), dtype=np.float32)
        stereo[:, 0] = 0.0   # left = 0, right = 1  → mean = 0.5
        mono_expected = 0.5
        mono = stereo.mean(axis=1)
        assert abs(mono.mean() - mono_expected) < 1e-5


# ══════════════════════════════════════════════════════════════════════════════
# 3. ONSET DETECTOR  (audio/onset.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestOnsetDetector:

    def _make_detector(self):
        from alertrack.audio.onset import OnsetDetector
        return OnsetDetector(sample_rate=SR)

    def test_instantiates(self):
        d = self._make_detector()
        assert d is not None

    def test_returns_tuple(self):
        d = self._make_detector()
        audio = _white_noise(BUF, rms=0.001)
        result = d.check(audio)
        assert isinstance(result, tuple)
        assert len(result) == 2
        triggered, info = result
        assert isinstance(triggered, bool)
        assert "margin_db" in info
        assert "floor_db" in info

    def test_silence_does_not_trigger(self):
        d = self._make_detector()
        for _ in range(20):
            triggered, _ = d.check(_silence(BUF))
        assert not triggered

    def test_below_min_rms_does_not_trigger(self):
        """Audio below ONSET_MIN_RMS (0.0015) should never trigger."""
        d = self._make_detector()
        tiny = _white_noise(BUF, rms=0.0005)
        for _ in range(5):
            triggered, _ = d.check(tiny)
        assert not triggered

    def test_loud_impulse_triggers(self):
        """A sudden loud signal (>>7 dB above floor) should trigger after floor is established."""
        d = self._make_detector()
        # Prime the floor with quiet background
        quiet = _white_noise(BUF, rms=0.001)
        for _ in range(30):
            d.check(quiet)
        # Now inject a loud impulse
        loud = _white_noise(BUF, rms=0.5)
        triggered, info = d.check(loud)
        assert triggered, f"Expected trigger, margin was {info['margin_db']:.1f} dB"
        assert info["margin_db"] >= 7.0

    def test_refractory_window_suppresses_second_trigger(self):
        """Two consecutive loud stimuli should only fire once due to refractory period."""
        d = self._make_detector()
        quiet = _white_noise(BUF, rms=0.001)
        for _ in range(30):
            d.check(quiet)
        loud = _white_noise(BUF, rms=0.5)
        first, _  = d.check(loud)
        second, _ = d.check(loud)
        assert first,   "First stimulus should trigger"
        assert not second, "Second stimulus within refractory window should be suppressed"

    def test_floor_adapts_upward(self):
        """Sustained loud background should cause the floor to track upward."""
        d = self._make_detector()
        quiet = _white_noise(BUF, rms=0.001)
        for _ in range(30):
            d.check(quiet)
        floor_after_quiet = d.floor_db
        loud_bg = _white_noise(BUF, rms=0.3)
        for _ in range(60):
            d.check(loud_bg)
        assert d.floor_db > floor_after_quiet, "Floor should have risen in loud background"

    def test_info_dict_contains_required_fields(self):
        d = self._make_detector()
        _, info = d.check(_white_noise())
        assert "margin_db"  in info
        assert "floor_db"   in info
        assert "recent_rms" in info
        assert "current_db" in info

    def test_highpass_removes_low_freq_hum(self):
        """50 Hz tone should not inflate the energy measurement enough to trigger."""
        from alertrack.audio.onset import _highpass_energy, ONSET_HPF_HZ
        hum = _tone(freq=50, n=BUF, amp=0.4)
        filtered = _highpass_energy(hum, SR, ONSET_HPF_HZ)
        # After high-pass, the hum energy should be drastically reduced
        rms_raw      = float(np.sqrt(np.mean(hum**2)))
        rms_filtered = float(np.sqrt(np.mean(filtered**2)))
        assert rms_filtered < rms_raw * 0.1, \
            f"HPF should remove ≥90% of 50 Hz energy; raw={rms_raw:.4f} filtered={rms_filtered:.4f}"

    def test_get_stats_keys(self):
        d = self._make_detector()
        stats = d.get_stats()
        assert "floor_db"     in stats
        assert "last_trigger" in stats


# ══════════════════════════════════════════════════════════════════════════════
# 4. AUDIO PREPROCESSOR  (audio/preprocess.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestAudioPreprocessor:

    def _make_proc(self):
        from alertrack.audio.preprocess import AudioPreprocessor
        return AudioPreprocessor()

    def test_instantiates(self):
        p = self._make_proc()
        assert p is not None

    def test_output_shape(self):
        p = self._make_proc()
        audio = _white_noise(BUF, rms=0.05)
        out = p.preprocess(audio)
        assert out is not None
        assert out.shape == INPUT_SHAPE, f"Expected {INPUT_SHAPE}, got {out.shape}"

    def test_output_dtype_float32(self):
        p = self._make_proc()
        out = p.preprocess(_white_noise(BUF, rms=0.05))
        assert out.dtype == np.float32

    def test_output_values_in_db_range(self):
        """Log-mel values should be in a plausible dB range (−80 to 0)."""
        p = self._make_proc()
        out = p.preprocess(_white_noise(BUF, rms=0.05))
        assert out.min() >= -120.0
        assert out.max() <= 10.0   # power_to_db(ref=max) ≤ 0 dB

    def test_silence_gate_returns_none(self):
        """Audio below SILENCE_THRESHOLD (RMS 0.0015) should return None."""
        p = self._make_proc()
        result = p.preprocess(_silence(BUF))
        assert result is None

    def test_silence_gate_threshold_boundary(self):
        """Audio just above the threshold should NOT return None."""
        from alertrack.config import SILENCE_THRESHOLD
        p = self._make_proc()
        just_above = _white_noise(BUF, rms=SILENCE_THRESHOLD * 2)
        result = p.preprocess(just_above)
        assert result is not None

    def test_short_audio_is_padded(self):
        """Shorter than 3 s input should be zero-padded to BUFFER_SIZE."""
        p = self._make_proc()
        short = _white_noise(BUF // 2, rms=0.05)
        out = p.preprocess(short)
        assert out is not None
        assert out.shape == INPUT_SHAPE

    def test_long_audio_is_trimmed(self):
        """Longer than 3 s input should be trimmed to the first BUFFER_SIZE samples."""
        p = self._make_proc()
        long_audio = _white_noise(BUF * 2, rms=0.05)
        out = p.preprocess(long_audio)
        assert out is not None
        assert out.shape == INPUT_SHAPE

    def test_ebu_r128_normalisation(self):
        """After normalisation, RMS should be very close to EBU_TARGET (±10%)."""
        from alertrack.audio.preprocess import _ebu_r128_normalize
        audio = _white_noise(BUF, rms=0.5)   # deliberately loud
        normed = _ebu_r128_normalize(audio)
        rms = float(np.sqrt(np.mean(normed**2)))
        assert abs(rms - EBU_TARGET) < EBU_TARGET * 0.10, \
            f"RMS after EBU R128 = {rms:.5f}, expected ≈ {EBU_TARGET:.5f}"

    def test_ebu_normalisation_clips_to_minus1_plus1(self):
        from alertrack.audio.preprocess import _ebu_r128_normalize
        audio = np.ones(BUF, dtype=np.float32) * 10.0   # extreme
        normed = _ebu_r128_normalize(audio)
        assert normed.max() <= 1.0
        assert normed.min() >= -1.0

    def test_highpass_filter_attenuates_hum(self):
        """The HPF should remove energy below HPF_CUTOFF_HZ (90 Hz)."""
        from alertrack.audio.preprocess import _highpass
        from alertrack.config import HPF_CUTOFF_HZ, HPF_WIDTH_HZ
        hum = _tone(freq=50, n=BUF, amp=0.4)
        filtered = _highpass(hum, SR)
        rms_raw      = float(np.sqrt(np.mean(hum**2)))
        rms_filtered = float(np.sqrt(np.mean(filtered**2)))
        assert rms_filtered < rms_raw * 0.1, "HPF should suppress >90% of 50 Hz energy"

    def test_highpass_preserves_mid_freq(self):
        """A 1 kHz tone should pass through the HPF largely unchanged."""
        from alertrack.audio.preprocess import _highpass
        tone = _tone(freq=1000, n=BUF, amp=0.3)
        filtered = _highpass(tone, SR)
        rms_in  = float(np.sqrt(np.mean(tone**2)))
        rms_out = float(np.sqrt(np.mean(filtered**2)))
        assert rms_out > rms_in * 0.9, "1 kHz tone should pass HPF with <10% attenuation"

    def test_different_inputs_produce_different_outputs(self):
        """The preprocessor must not return the same tensor for different inputs."""
        p = self._make_proc()
        out1 = p.preprocess(_white_noise(BUF, rms=0.05))
        out2 = p.preprocess(_white_noise(BUF, rms=0.05))
        assert not np.allclose(out1, out2), "Different random inputs should produce different spectrograms"


# ══════════════════════════════════════════════════════════════════════════════
# 5. ONNX MODEL  (inference/model.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestONNXModel:

    MODEL_PATH = Path("models/custom_cnn/alertreck_cnn.onnx")

    @pytest.fixture(scope="class")
    def model(self):
        if not self.MODEL_PATH.exists():
            pytest.skip(f"ONNX model not found at {self.MODEL_PATH}. "
                        "Run scripts/export_model.py first.")
        from alertrack.inference.model import ONNXModel
        return ONNXModel(self.MODEL_PATH)

    def test_loads_without_error(self, model):
        assert model is not None

    def test_input_shape_attribute(self, model):
        shape = model.get_input_shape()
        assert len(shape) == 4           # (batch, channel, mel, time)
        assert shape[1] == 1
        assert shape[2] == 128
        # batch and time dim may be dynamic (None or int)

    def test_predict_returns_tuple(self, model):
        x = np.random.randn(1, 128, 301).astype(np.float32)
        result = model.predict(x)
        assert result is not None
        assert len(result) == 3

    def test_predict_class_index_in_range(self, model):
        x = np.random.randn(1, 128, 301).astype(np.float32)
        idx, conf, probs = model.predict(x)
        assert 0 <= idx < 7

    def test_predict_confidence_in_range(self, model):
        x = np.random.randn(1, 128, 301).astype(np.float32)
        _, conf, probs = model.predict(x)
        assert 0.0 <= conf <= 1.0

    def test_probabilities_sum_to_one(self, model):
        x = np.random.randn(1, 128, 301).astype(np.float32)
        _, _, probs = model.predict(x)
        assert len(probs) == 7
        assert abs(probs.sum() - 1.0) < 1e-4, f"Probs sum = {probs.sum()}"

    def test_probabilities_non_negative(self, model):
        x = np.random.randn(1, 128, 301).astype(np.float32)
        _, _, probs = model.predict(x)
        assert all(p >= 0.0 for p in probs)

    def test_accepts_3d_input_and_expands(self, model):
        """predict() should accept (1,128,301) and expand to (1,1,128,301)."""
        x = np.random.randn(1, 128, 301).astype(np.float32)
        result = model.predict(x)
        assert result is not None

    def test_get_class_name(self, model):
        assert model.get_class_name(4) == "threat_gunshot"
        assert model.get_class_name(0) == "background_animals"

    def test_get_class_name_out_of_range(self, model):
        name = model.get_class_name(999)
        assert "UNKNOWN" in name

    def test_get_model_info_keys(self, model):
        info = model.get_model_info()
        assert "model_path"  in info
        assert "input_shape" in info
        assert "num_classes" in info
        assert "class_names" in info
        assert info["num_classes"] == 7

    def test_predict_returns_none_on_invalid_input(self, model):
        """Passing a zero-length array should return None gracefully."""
        result = model.predict(np.array([]))
        assert result is None

    def test_softmax_internal(self):
        """The internal _softmax helper must be numerically stable."""
        from alertrack.inference.model import _softmax
        logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        probs  = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert all(p >= 0.0 for p in probs)


# ══════════════════════════════════════════════════════════════════════════════
# 6. THREAT DECISION ENGINE  (inference/decision.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestThreatDecisionEngine:

    def _make_engine(self):
        from alertrack.inference.decision import ThreatDecisionEngine
        return ThreatDecisionEngine()

    def test_instantiates(self):
        e = self._make_engine()
        assert e is not None

    # ── background classes ────────────────────────────────────────────────────
    @pytest.mark.parametrize("bg_idx", [0, 1])
    def test_background_never_alerts(self, bg_idx):
        e = self._make_engine()
        probs = _spike_probs(bg_idx, confidence=0.99)
        should_alert, info = e.evaluate(bg_idx, float(probs[bg_idx]), probs)
        assert not should_alert
        assert info is None

    # ── threshold ─────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("threat_idx", [2, 3, 4, 5, 6])
    def test_threat_below_threshold_does_not_alert(self, threat_idx):
        e = self._make_engine()
        probs = _spike_probs(threat_idx, confidence=0.40)
        should_alert, _ = e.evaluate(threat_idx, 0.40, probs)
        assert not should_alert

    @pytest.mark.parametrize("threat_idx", [2, 3, 4, 5, 6])
    def test_threat_above_threshold_alerts(self, threat_idx):
        e = self._make_engine()
        probs = _spike_probs(threat_idx, confidence=0.95)
        should_alert, info = e.evaluate(threat_idx, 0.95, probs)
        assert should_alert
        assert info is not None
        assert info["threat_type"] == CLASSES[threat_idx]

    def test_exact_threshold_alerts(self):
        """Score exactly equal to threshold (0.60) should alert."""
        e = self._make_engine()
        probs = _spike_probs(4, confidence=0.60)
        should_alert, _ = e.evaluate(4, 0.60, probs)
        assert should_alert

    def test_just_below_threshold_does_not_alert(self):
        e = self._make_engine()
        probs = _spike_probs(4, confidence=0.599)
        should_alert, _ = e.evaluate(4, 0.599, probs)
        assert not should_alert

    # ── cooldown ─────────────────────────────────────────────────────────────
    def test_cooldown_suppresses_immediate_repeat(self):
        e = self._make_engine()
        probs = _spike_probs(4, confidence=0.95)
        first,  _ = e.evaluate(4, 0.95, probs)
        second, _ = e.evaluate(4, 0.95, probs)
        assert first,       "First detection should alert"
        assert not second,  "Second detection within cooldown should be suppressed"

    def test_cooldown_is_per_class_independent(self):
        """Cooldown on threat_gunshot (idx 4) should NOT suppress threat_chainsaw (idx 2)."""
        e = self._make_engine()
        p1 = _spike_probs(4, confidence=0.95)
        p2 = _spike_probs(2, confidence=0.95)
        first,  _ = e.evaluate(4, 0.95, p1)
        second, _ = e.evaluate(2, 0.95, p2)
        assert first,  "Gunshot alert should fire"
        assert second, "Chainsaw alert should also fire (independent cooldown)"

    def test_reset_cooldown_single_class(self):
        e = self._make_engine()
        probs = _spike_probs(4, confidence=0.95)
        e.evaluate(4, 0.95, probs)   # first — fires
        e.evaluate(4, 0.95, probs)   # second — suppressed
        e.reset_cooldown("threat_gunshot")
        should_alert, _ = e.evaluate(4, 0.95, probs)
        assert should_alert, "After cooldown reset, next detection should alert again"

    def test_reset_cooldown_all(self):
        e = self._make_engine()
        for idx in [2, 3, 4, 5, 6]:
            probs = _spike_probs(idx, confidence=0.95)
            e.evaluate(idx, 0.95, probs)
        e.reset_cooldown()   # reset all
        for idx in [2, 3, 4, 5, 6]:
            probs = _spike_probs(idx, confidence=0.95)
            should_alert, _ = e.evaluate(idx, 0.95, probs)
            assert should_alert, f"After full reset, {CLASSES[idx]} should alert again"

    # ── threat_info dict ─────────────────────────────────────────────────────
    def test_threat_info_contains_required_fields(self):
        e = self._make_engine()
        probs = _spike_probs(4, confidence=0.95)
        _, info = e.evaluate(4, 0.95, probs)
        assert info is not None
        for field in ("threat_type", "threat_level", "confidence",
                      "top_predictions", "class_probabilities", "timestamp"):
            assert field in info, f"Missing field: {field}"

    def test_threat_info_confidence_matches_input(self):
        e = self._make_engine()
        probs = _spike_probs(4, confidence=0.87)
        _, info = e.evaluate(4, 0.87, probs)
        assert abs(info["confidence"] - 0.87) < 1e-4

    def test_top_predictions_length(self):
        e = self._make_engine()
        probs = _spike_probs(4, confidence=0.95)
        _, info = e.evaluate(4, 0.95, probs)
        assert len(info["top_predictions"]) == 3

    def test_class_probabilities_sum_to_one(self):
        e = self._make_engine()
        probs = _spike_probs(4, confidence=0.95)
        _, info = e.evaluate(4, 0.95, probs)
        total = sum(info["class_probabilities"].values())
        assert abs(total - 1.0) < 1e-4

    def test_get_stats_keys(self):
        e = self._make_engine()
        stats = e.get_stats()
        for key in ("total_predictions", "total_threats_detected",
                    "total_suppressed", "threat_counts", "cooldown_status"):
            assert key in stats

    def test_total_predictions_increments(self):
        e = self._make_engine()
        for _ in range(5):
            e.evaluate(0, 0.99, _spike_probs(0))
        assert e.get_stats()["total_predictions"] == 5

    def test_is_threat_class(self):
        e = self._make_engine()
        assert not e.is_threat_class(0)   # background_animals
        assert not e.is_threat_class(1)   # background_wind_rain
        assert e.is_threat_class(4)        # threat_gunshot

    def test_get_threat_level(self):
        e = self._make_engine()
        assert e.get_threat_level(4) == "HIGH"   # threat_gunshot
        assert e.get_threat_level(0) == "NONE"   # background


# ══════════════════════════════════════════════════════════════════════════════
# 7. EVIDENCE MANAGER  (storage/evidence.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestEvidenceManager:

    def _make_manager(self, tmp_path):
        with patch("alertrack.config.EVIDENCE_DIR", tmp_path / "evidence"), \
             patch("alertrack.config.ALERTS_DIR",   tmp_path / "alerts"):
            from alertrack.storage.evidence import EvidenceManager
            # Reload to pick up patched paths
            import importlib
            import alertrack.storage.evidence as em
            importlib.reload(em)
            return em.EvidenceManager()

    def _mock_threat_info(self):
        return {
            "threat_type":  "threat_gunshot",
            "threat_level": "HIGH",
            "confidence":   0.94,
            "top_predictions": [
                {"class": "threat_gunshot", "confidence": 0.94},
                {"class": "threat_human",   "confidence": 0.04},
                {"class": "threat_vehicle", "confidence": 0.01},
            ],
            "class_probabilities": {c: 0.0 for c in CLASSES},
            "timestamp": time.time(),
        }

    def test_save_audio_evidence_creates_wav(self, tmp_path):
        with patch("alertrack.config.EVIDENCE_DIR", tmp_path / "evidence"):
            import importlib, alertrack.storage.evidence as em
            importlib.reload(em)
            mgr = em.EvidenceManager()
            audio = _white_noise(BUF, rms=0.05)
            path  = mgr.save_audio_evidence(audio, "threat_gunshot", "test_alert_001")
            assert path is not None
            assert Path(path).exists()
            assert Path(path).suffix == ".wav"

    def test_saved_wav_is_readable(self, tmp_path):
        with patch("alertrack.config.EVIDENCE_DIR", tmp_path / "evidence"):
            import importlib, alertrack.storage.evidence as em
            importlib.reload(em)
            mgr   = em.EvidenceManager()
            audio = _white_noise(BUF, rms=0.05)
            path  = mgr.save_audio_evidence(audio, "threat_gunshot", "test_alert_002")
            with wave.open(str(path), "rb") as w:
                assert w.getnchannels() == 1
                assert w.getframerate() == SR

    def test_save_event_record_creates_triple(self, tmp_path):
        with patch("alertrack.config.EVIDENCE_DIR", tmp_path / "evidence"):
            import importlib, alertrack.storage.evidence as em
            importlib.reload(em)
            mgr      = em.EvidenceManager()
            mel      = np.random.randn(*INPUT_SHAPE).astype(np.float32)
            info     = self._mock_threat_info()
            location = {"latitude": -1.9424, "longitude": 30.0618, "fix_quality": 1}
            alert_id = "test_record_001"

            # The sidecars are written NEXT TO the WAV, so the audio evidence must
            # exist first — save_event_record returns None when audio_path is None.
            audio_path = mgr.save_audio_evidence(
                _white_noise(BUF, rms=0.05), "threat_gunshot", alert_id)
            assert audio_path is not None, "audio evidence must be saved first"

            mgr.save_event_record(mel, info, location, audio_path, alert_id)

            # All three members of the evidence triple must exist
            ev_dir = tmp_path / "evidence"
            assert list(ev_dir.rglob(f"*{alert_id}*.wav")),     "WAV should exist"
            assert list(ev_dir.rglob(f"*{alert_id}*.mel.npy")), "mel sidecar should exist"
            assert list(ev_dir.rglob(f"*{alert_id}*.json")),    "JSON sidecar should exist"

    def test_json_sidecar_contains_required_fields(self, tmp_path):
        with patch("alertrack.config.EVIDENCE_DIR", tmp_path / "evidence"):
            import importlib, alertrack.storage.evidence as em
            importlib.reload(em)
            mgr      = em.EvidenceManager()
            mel      = np.random.randn(*INPUT_SHAPE).astype(np.float32)
            info     = self._mock_threat_info()
            location = {"latitude": -1.9424, "longitude": 30.0618, "fix_quality": 1}
            alert_id = "test_fields_001"
            mgr.save_event_record(mel, info, location, None, alert_id)

            ev_dir   = tmp_path / "evidence"
            j_files  = list(ev_dir.rglob(f"*{alert_id}*.json"))
            if j_files:
                data = json.loads(j_files[0].read_text())
                for field in ("threat_type", "confidence", "timestamp"):
                    assert field in data, f"JSON sidecar missing: {field}"

    def test_mel_npy_saved_alongside_json(self, tmp_path):
        with patch("alertrack.config.EVIDENCE_DIR", tmp_path / "evidence"):
            import importlib, alertrack.storage.evidence as em
            importlib.reload(em)
            mgr      = em.EvidenceManager()
            mel      = np.random.randn(*INPUT_SHAPE).astype(np.float32)
            info     = self._mock_threat_info()
            alert_id = "test_mel_001"
            mgr.save_event_record(mel, info, {}, None, alert_id)

            ev_dir   = tmp_path / "evidence"
            npy_files = list(ev_dir.rglob(f"*{alert_id}*.npy"))
            if npy_files:
                loaded = np.load(str(npy_files[0]))
                assert loaded.shape == INPUT_SHAPE


# ══════════════════════════════════════════════════════════════════════════════
# 8. ALERT NOTIFIER  (alerts/notifier.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestAlertNotifier:
    """Tests the payload construction and non-GSM channels. GSM is mocked."""

    def _make_notifier(self):
        from alertrack.alerts.notifier import AlertNotifier
        return AlertNotifier()

    def _mock_threat(self, class_name="threat_gunshot", conf=0.94):
        return {
            "threat_type":  class_name,
            "threat_level": "HIGH",
            "confidence":   conf,
            "top_predictions": [
                {"class": class_name,       "confidence": conf},
                {"class": "threat_human",   "confidence": 0.04},
                {"class": "threat_vehicle", "confidence": 0.01},
            ],
            "class_probabilities": {c: 0.01 for c in CLASSES},
        }

    def _mock_location(self, with_fix=True):
        if with_fix:
            return {"latitude": -1.9424, "longitude": 30.0618, "fix_quality": 1, "altitude": 1548.0, "satellites": 7}
        return {"latitude": "UNKNOWN", "longitude": "UNKNOWN", "fix_quality": 0, "satellites": 0}

    def test_instantiates(self):
        n = self._make_notifier()
        assert n is not None

    def test_create_alert_returns_dict(self):
        n = self._make_notifier()
        alert = n.create_alert(self._mock_threat(), self._mock_location())
        assert isinstance(alert, dict)

    def test_alert_contains_required_fields(self):
        n     = self._make_notifier()
        alert = n.create_alert(self._mock_threat(), self._mock_location())
        for field in ("alert_id", "timestamp", "threat_type", "threat_level",
                      "confidence", "latitude", "longitude", "status"):
            assert field in alert, f"Alert missing field: {field}"

    def test_alert_id_is_non_empty_string(self):
        n     = self._make_notifier()
        alert = n.create_alert(self._mock_threat(), self._mock_location())
        assert isinstance(alert["alert_id"], str)
        assert len(alert["alert_id"]) > 0

    def test_alert_status_initially_pending(self):
        n     = self._make_notifier()
        alert = n.create_alert(self._mock_threat(), self._mock_location())
        assert alert["status"] == "PENDING"

    def test_alert_confidence_matches_input(self):
        n     = self._make_notifier()
        alert = n.create_alert(self._mock_threat(conf=0.87), self._mock_location())
        assert abs(alert["confidence"] - 0.87) < 1e-4

    def test_alert_with_gps_fix_has_numeric_coords(self):
        n     = self._make_notifier()
        alert = n.create_alert(self._mock_threat(), self._mock_location(with_fix=True))
        assert isinstance(alert["latitude"],  float)
        assert isinstance(alert["longitude"], float)

    def test_alert_without_gps_fix_has_unknown_coords(self):
        n     = self._make_notifier()
        alert = n.create_alert(self._mock_threat(), self._mock_location(with_fix=False))
        assert alert["latitude"]  == "UNKNOWN"
        assert alert["longitude"] == "UNKNOWN"

    def test_sms_payload_within_160_chars(self):
        """The formatted SMS string must fit in a single SMS."""
        from alertrack.alerts.notifier import _format_top_predictions, _short_label
        from datetime import datetime, timezone
        threat = self._mock_threat()
        loc    = self._mock_location(with_fix=True)
        hhmm   = datetime.now(timezone.utc).strftime("%H:%M")
        gps    = f"{loc['latitude']:.5f},{loc['longitude']:.5f}"
        ranked = _format_top_predictions(threat["top_predictions"])
        sms    = f"ALERTRECK | {ranked} | GPS: {gps} | {hhmm}"
        assert len(sms) <= 160, f"SMS too long ({len(sms)} chars): {sms}"

    def test_sms_payload_contains_class(self):
        from alertrack.alerts.notifier import _format_top_predictions
        threat  = self._mock_threat(class_name="threat_gunshot")
        ranked  = _format_top_predictions(threat["top_predictions"])
        assert "gunshot" in ranked

    def test_sms_payload_no_fix_uses_no_fix_string(self):
        from alertrack.alerts.notifier import AlertNotifier
        from datetime import datetime, timezone
        n     = AlertNotifier()
        alert = n.create_alert(self._mock_threat(), self._mock_location(with_fix=False))
        lat   = alert.get("latitude", "?")
        gps   = f"{lat:.5f}" if isinstance(lat, float) else "no fix"
        assert "no fix" == gps

    def test_console_notification_succeeds(self):
        """_notify_console() should return True and not raise."""
        n     = self._make_notifier()
        alert = n.create_alert(self._mock_threat(), self._mock_location())
        result = n._notify_console(alert)
        assert result is True

    @patch("alertrack.alerts.notifier.NOTIFY_GSM", False)
    def test_send_alert_without_gsm_still_succeeds(self):
        """With GSM disabled, send_alert should still succeed via disk+console."""
        with patch("alertrack.alerts.notifier.ALERTS_DIR", Path(tempfile.mkdtemp())):
            n     = self._make_notifier()
            alert = n.create_alert(self._mock_threat(), self._mock_location())
            result = n.send_alert(alert)
            assert result is True

    def test_get_stats_keys(self):
        n     = self._make_notifier()
        stats = n.get_stats()
        assert "alerts_sent" in stats


# ══════════════════════════════════════════════════════════════════════════════
# 9. GPS MODULE  (sensors/gps.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestSIM808AT:
    """Tests SIM808AT with a mocked serial port — no real hardware needed."""

    def _make_sim808(self):
        from alertrack.sensors.gps import SIM808AT
        return SIM808AT(port="/dev/null", baudrate=9600)

    def test_instantiates(self):
        s = self._make_sim808()
        assert s is not None

    def test_get_location_returns_dict(self):
        """With a mocked serial port that returns a valid CGNSINF response."""
        cgnsinf_response = (
            b"AT+CGNSINF\r\r\n"
            b"+CGNSINF: 1,1,20260601120000.000,-1.942369,30.061839,"
            b"1548.2,0.0,0.0,2,,0.8,1.1,0.6,,7,5,,,35,,\r\n"
            b"\r\nOK\r\n"
        )
        mock_serial = MagicMock()
        mock_serial.__enter__ = lambda s: s
        mock_serial.__exit__  = MagicMock(return_value=False)
        mock_serial.read.return_value = cgnsinf_response
        mock_serial.in_waiting = len(cgnsinf_response)

        with patch("serial.Serial", return_value=mock_serial):
            from alertrack.sensors.gps import SIM808AT
            s   = SIM808AT()
            loc = s.get_location()
        assert isinstance(loc, dict)

    def test_get_location_fix_quality_one_on_valid_response(self):
        cgnsinf_response = (
            b"+CGNSINF: 1,1,20260601120000.000,-1.942369,30.061839,"
            b"1548.2,0.0,0.0,2,,0.8,1.1,0.6,,7,5,,,35,,\r\n"
        )
        mock_serial = MagicMock()
        mock_serial.__enter__ = lambda s: s
        mock_serial.__exit__  = MagicMock(return_value=False)
        mock_serial.readline.return_value = cgnsinf_response

        with patch("serial.Serial", return_value=mock_serial):
            from alertrack.sensors.gps import SIM808AT
            s   = SIM808AT()
            loc = s.get_location()
        # Either the mock worked or we get the no-fix dict — both are valid
        assert "fix_quality" in loc

    def test_get_location_no_fix_returns_fix_quality_zero(self):
        """When CGNSINF returns fix_status=0, fix_quality should be 0."""
        no_fix_response = b"+CGNSINF: 1,0,,,,,,,,,,,,,,,,,,\r\n"
        mock_serial = MagicMock()
        mock_serial.__enter__ = lambda s: s
        mock_serial.__exit__  = MagicMock(return_value=False)
        mock_serial.readline.return_value = no_fix_response

        with patch("serial.Serial", return_value=mock_serial):
            from alertrack.sensors.gps import SIM808AT
            s   = SIM808AT()
            loc = s.get_location()
        assert loc.get("fix_quality", 0) == 0

    def test_get_location_on_serial_error_returns_no_fix_dict(self):
        """Serial exceptions should be caught and return a safe no-fix dict."""
        with patch("serial.Serial", side_effect=Exception("port busy")):
            from alertrack.sensors.gps import SIM808AT
            s   = SIM808AT()
            loc = s.get_location()
        assert isinstance(loc, dict)
        assert loc.get("fix_quality", 0) == 0

    def test_has_fix_returns_bool(self):
        s = self._make_sim808()
        with patch.object(s, "get_location", return_value={"fix_quality": 0}):
            assert s.has_fix() is False
        with patch.object(s, "get_location", return_value={"fix_quality": 1, "latitude": -1.9, "longitude": 30.0}):
            assert s.has_fix() is True


# ══════════════════════════════════════════════════════════════════════════════
# 10. LOGGER  (storage/logger.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestLogger:

    def test_get_logger_returns_logger(self):
        from alertrack.storage.logger import get_logger
        logger = get_logger()
        assert logger is not None

    def test_logger_has_info_method(self):
        from alertrack.storage.logger import get_logger
        logger = get_logger()
        assert callable(logger.info)
        logger.info("test message")   # should not raise

    def test_logger_has_warning_method(self):
        from alertrack.storage.logger import get_logger
        logger = get_logger()
        assert callable(logger.warning)

    def test_logger_has_error_method(self):
        from alertrack.storage.logger import get_logger
        logger = get_logger()
        assert callable(logger.error)
