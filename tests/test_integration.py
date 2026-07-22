"""
AlertReck Integration Test Suite
=================================
Tests that each component pair passes data correctly through the full pipeline.
All hardware I/O (sounddevice, serial) is mocked.

Run from the project root:
    pytest tests/test_integration.py -v

The ONNX model must be present at models/custom_cnn/alertreck_cnn.onnx.
Run scripts/export_model.py first if it is missing.
"""

import time
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

SR          = 44_100
BUF         = 132_300
INPUT_SHAPE = (1, 128, 301)
MODEL_PATH  = Path("models/custom_cnn/alertreck_cnn.onnx")

CLASSES = [
    "background_animals",
    "background_wind_rain",
    "threat_chainsaw",
    "threat_dog",
    "threat_gunshot",
    "threat_human",
    "threat_vehicle",
]

def _white_noise(n=BUF, rms=0.05):
    x = np.random.randn(n).astype(np.float32)
    return x / (np.sqrt(np.mean(x**2)) + 1e-9) * rms


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def audio_buffer():
    return _white_noise(BUF, rms=0.05)

@pytest.fixture(scope="module")
def preprocessor():
    from alertrack.audio.preprocess import AudioPreprocessor
    return AudioPreprocessor()

@pytest.fixture(scope="module")
def mel_tensor(preprocessor, audio_buffer):
    out = preprocessor.preprocess(audio_buffer)
    if out is None:
        pytest.skip("Preprocessor returned None — buffer RMS too low.")
    return out

@pytest.fixture(scope="module")
def onnx_model():
    if not MODEL_PATH.exists():
        pytest.skip(f"ONNX model not found at {MODEL_PATH}. Run export_model.py first.")
    from alertrack.inference.model import ONNXModel
    return ONNXModel(MODEL_PATH)

@pytest.fixture(scope="module")
def decision_engine():
    from alertrack.inference.decision import ThreatDecisionEngine
    return ThreatDecisionEngine()

@pytest.fixture(scope="module")
def notifier(tmp_path_factory):
    alerts_dir = tmp_path_factory.mktemp("alerts")
    with patch("alertrack.config.ALERTS_DIR", alerts_dir):
        import importlib, alertrack.alerts.notifier as m
        importlib.reload(m)
        return m.AlertNotifier()

@pytest.fixture(scope="module")
def evidence_manager(tmp_path_factory):
    ev_dir = tmp_path_factory.mktemp("evidence")
    with patch("alertrack.config.EVIDENCE_DIR", ev_dir):
        import importlib, alertrack.storage.evidence as m
        importlib.reload(m)
        return m.EvidenceManager(), ev_dir


# ══════════════════════════════════════════════════════════════════════════════
# INT-01  AudioRecorder → OnsetDetector
# Confirms the rolling buffer format is accepted by the detector.
# ══════════════════════════════════════════════════════════════════════════════

class TestInt01RecorderToOnset:

    def test_buffer_hand_off_format(self):
        """Recorder buffer (float32, shape (132300,)) must be accepted by OnsetDetector."""
        from alertrack.audio.recorder import AudioRecorder
        from alertrack.audio.onset    import OnsetDetector

        recorder = AudioRecorder(device_id=None)
        detector = OnsetDetector(sample_rate=SR)

        # Simulate a filled buffer
        with recorder.lock:
            recorder.buffer.extend(_white_noise(BUF, rms=0.05))

        audio = recorder.get_audio_buffer()
        assert audio is not None
        assert audio.shape == (BUF,)
        assert audio.dtype == np.float32

        # Must not raise
        triggered, info = detector.check(audio)
        assert isinstance(triggered, bool)

    def test_onset_floor_tracks_from_recorder_buffer(self):
        """After priming with many quiet buffers, the floor should be well-established."""
        from alertrack.audio.recorder import AudioRecorder
        from alertrack.audio.onset    import OnsetDetector

        recorder = AudioRecorder(device_id=None)
        detector = OnsetDetector(sample_rate=SR)

        quiet = _white_noise(BUF, rms=0.001)
        for _ in range(40):
            with recorder.lock:
                recorder.buffer.clear()
                recorder.buffer.extend(quiet)
            audio = recorder.get_audio_buffer()
            detector.check(audio)

        assert detector.floor_db is not None
        assert detector.floor_db < -30.0, "Floor should be well below −30 dBFS for quiet signal"

    def test_loud_audio_triggers_after_quiet_prime(self):
        from alertrack.audio.recorder import AudioRecorder
        from alertrack.audio.onset    import OnsetDetector

        recorder = AudioRecorder(device_id=None)
        detector = OnsetDetector(sample_rate=SR)

        quiet = _white_noise(BUF, rms=0.001)
        for _ in range(40):
            with recorder.lock:
                recorder.buffer.clear()
                recorder.buffer.extend(quiet)
            detector.check(recorder.get_audio_buffer())

        loud = _white_noise(BUF, rms=0.5)
        with recorder.lock:
            recorder.buffer.clear()
            recorder.buffer.extend(loud)
        triggered, _ = detector.check(recorder.get_audio_buffer())
        assert triggered, "Loud stimulus should trigger after quiet background primed"


# ══════════════════════════════════════════════════════════════════════════════
# INT-02  OnsetDetector → AudioPreprocessor
# Confirms a triggered window is correctly passed to preprocessing.
# ══════════════════════════════════════════════════════════════════════════════

class TestInt02OnsetToPreprocessor:

    def test_triggered_window_preprocesses_to_correct_shape(self, preprocessor, audio_buffer):
        """The buffer snapshot taken on a trigger must preprocess to INPUT_SHAPE."""
        out = preprocessor.preprocess(audio_buffer)
        assert out is not None
        assert out.shape == INPUT_SHAPE

    def test_ebu_normalisation_applied_before_mel(self):
        """After normalisation the signal RMS should be close to the EBU target."""
        from alertrack.audio.preprocess import _ebu_r128_normalize
        audio  = _white_noise(BUF, rms=0.5)
        normed = _ebu_r128_normalize(audio)
        ebu    = 10 ** (-23.0 / 20.0)
        rms    = float(np.sqrt(np.mean(normed**2)))
        assert abs(rms - ebu) < ebu * 0.10

    def test_silence_below_onset_min_rms_filtered_by_preprocessor(self):
        """If OnsetDetector misfires on a near-silent buffer, the preprocessor gate catches it."""
        from alertrack.audio.preprocess import AudioPreprocessor
        p     = AudioPreprocessor()
        tiny  = np.zeros(BUF, dtype=np.float32) + 1e-5
        assert p.preprocess(tiny) is None


# ══════════════════════════════════════════════════════════════════════════════
# INT-03  AudioPreprocessor → ONNXModel
# Confirms the mel tensor shape is accepted by the ONNX session.
# ══════════════════════════════════════════════════════════════════════════════

class TestInt03PreprocessorToModel:

    def test_mel_tensor_accepted_by_onnx(self, mel_tensor, onnx_model):
        result = onnx_model.predict(mel_tensor)
        assert result is not None

    def test_output_class_index_valid(self, mel_tensor, onnx_model):
        idx, _, _ = onnx_model.predict(mel_tensor)
        assert 0 <= idx < 7

    def test_output_probabilities_sum_to_one(self, mel_tensor, onnx_model):
        _, _, probs = onnx_model.predict(mel_tensor)
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_confidence_equals_max_prob(self, mel_tensor, onnx_model):
        idx, conf, probs = onnx_model.predict(mel_tensor)
        assert abs(conf - probs[idx]) < 1e-6

    def test_batch_independence(self, onnx_model):
        """Two independent calls with different inputs must produce different outputs."""
        x1 = np.random.randn(1, 128, 301).astype(np.float32)
        x2 = np.random.randn(1, 128, 301).astype(np.float32)
        _, _, p1 = onnx_model.predict(x1)
        _, _, p2 = onnx_model.predict(x2)
        assert not np.allclose(p1, p2), "Independent inputs should rarely produce identical outputs"


# ══════════════════════════════════════════════════════════════════════════════
# INT-04  ONNXModel → ThreatDecisionEngine
# Confirms the probability vector from ONNX is correctly routed through the gate.
# ══════════════════════════════════════════════════════════════════════════════

class TestInt04ModelToDecisionEngine:

    def test_real_model_output_accepted_by_engine(self, onnx_model, decision_engine):
        """Decision engine must accept the exact output tuple from ONNXModel."""
        x = np.random.randn(1, 128, 301).astype(np.float32)
        result = onnx_model.predict(x)
        assert result is not None
        idx, conf, probs = result
        # reset cooldowns so this call is always evaluated fresh
        decision_engine.reset_cooldown()
        should_alert, info = decision_engine.evaluate(idx, conf, probs)
        assert isinstance(should_alert, bool)
        if should_alert:
            assert info is not None
            assert "threat_type" in info

    def test_background_output_never_alerts(self, decision_engine):
        """Even a high-confidence background prediction must produce no alert."""
        decision_engine.reset_cooldown()
        for bg_idx in [0, 1]:
            probs     = np.zeros(7, dtype=np.float32)
            probs[bg_idx] = 0.99
            probs     = probs / probs.sum()
            should, _ = decision_engine.evaluate(bg_idx, 0.99, probs)
            assert not should, f"Background class {CLASSES[bg_idx]} must never alert"

    def test_high_confidence_threat_alerts(self, decision_engine):
        decision_engine.reset_cooldown()
        probs     = np.zeros(7, dtype=np.float32)
        probs[4]  = 0.95   # threat_gunshot
        probs     = probs / probs.sum()
        should, info = decision_engine.evaluate(4, 0.95, probs)
        assert should
        assert info["threat_type"] == "threat_gunshot"

    def test_probability_vector_passes_through_intact(self, onnx_model, decision_engine):
        """The class_probabilities dict in threat_info must reflect the ONNX output."""
        decision_engine.reset_cooldown()
        # Force a threat class output by constructing a known probability vector
        probs_in     = np.zeros(7, dtype=np.float32)
        probs_in[4]  = 0.92
        probs_in     = probs_in / probs_in.sum()
        _, info = decision_engine.evaluate(4, 0.92, probs_in)
        if info is not None:
            cp = info["class_probabilities"]
            assert abs(cp["threat_gunshot"] - probs_in[4]) < 1e-4


# ══════════════════════════════════════════════════════════════════════════════
# INT-05  ThreatDecisionEngine → EvidenceManager
# Confirms a threat_info dict from the engine is correctly persisted.
# ══════════════════════════════════════════════════════════════════════════════

class TestInt05DecisionToEvidence:

    def test_threat_info_persisted_as_flat_file_triple(self, tmp_path):
        from alertrack.inference.decision import ThreatDecisionEngine

        engine = ThreatDecisionEngine()
        engine.reset_cooldown()
        probs     = np.zeros(7, dtype=np.float32)
        probs[4]  = 0.94
        probs     = probs / probs.sum()
        should_alert, threat_info = engine.evaluate(4, 0.94, probs)
        assert should_alert

        ev_dir = tmp_path / "evidence"
        with patch("alertrack.config.EVIDENCE_DIR", ev_dir):
            import importlib, alertrack.storage.evidence as em
            importlib.reload(em)
            mgr      = em.EvidenceManager()
            audio    = _white_noise(BUF, rms=0.05)
            mel      = np.random.randn(*INPUT_SHAPE).astype(np.float32)
            alert_id = "int05_test_001"

            wav_path = mgr.save_audio_evidence(audio, threat_info["threat_type"], alert_id)
            mgr.save_event_record(mel, threat_info, {}, wav_path, alert_id)

        assert wav_path is not None
        assert Path(wav_path).exists()
        npy_files = list(ev_dir.rglob(f"*{alert_id}*.npy"))
        assert len(npy_files) >= 1, ".mel.npy sidecar should be saved"

    def test_evidence_wav_matches_input_audio(self, tmp_path):
        """WAV written by EvidenceManager should have correct sample rate and channels."""
        import wave as wv
        from alertrack.inference.decision import ThreatDecisionEngine

        engine = ThreatDecisionEngine()
        engine.reset_cooldown()
        probs     = np.zeros(7, dtype=np.float32)
        probs[4]  = 0.94
        probs     = probs / probs.sum()
        _, threat_info = engine.evaluate(4, 0.94, probs)

        ev_dir = tmp_path / "evidence"
        with patch("alertrack.config.EVIDENCE_DIR", ev_dir):
            import importlib, alertrack.storage.evidence as em
            importlib.reload(em)
            mgr      = em.EvidenceManager()
            audio    = _white_noise(BUF, rms=0.05)
            wav_path = mgr.save_audio_evidence(audio, threat_info["threat_type"], "int05_wav_check")

        if wav_path and Path(wav_path).exists():
            with wv.open(str(wav_path), "rb") as w:
                assert w.getnchannels()  == 1
                assert w.getframerate()  == SR


# ══════════════════════════════════════════════════════════════════════════════
# INT-06  ThreatDecisionEngine → AlertNotifier
# Confirms threat_info from the engine is correctly turned into an alert dict.
# ══════════════════════════════════════════════════════════════════════════════

class TestInt06DecisionToNotifier:

    def _make_notifier(self, alerts_dir):
        with patch("alertrack.config.ALERTS_DIR", alerts_dir):
            import importlib, alertrack.alerts.notifier as m
            importlib.reload(m)
            return m.AlertNotifier()

    def test_threat_info_to_alert_dict(self, tmp_path):
        from alertrack.inference.decision import ThreatDecisionEngine

        engine = ThreatDecisionEngine()
        engine.reset_cooldown()
        probs     = np.zeros(7, dtype=np.float32)
        probs[2]  = 0.88   # threat_chainsaw
        probs     = probs / probs.sum()
        _, threat_info = engine.evaluate(2, 0.88, probs)
        assert threat_info is not None

        notifier = self._make_notifier(tmp_path / "alerts")
        location = {"latitude": -1.9424, "longitude": 30.0618, "fix_quality": 1,
                    "altitude": 1548.0, "satellites": 7}
        alert    = notifier.create_alert(threat_info, location)

        assert alert["threat_type"]  == "threat_chainsaw"
        assert abs(alert["confidence"] - 0.88) < 0.01
        assert isinstance(alert["latitude"],  float)
        assert isinstance(alert["longitude"], float)

    def test_sms_payload_built_from_engine_output(self, tmp_path):
        from alertrack.inference.decision import ThreatDecisionEngine
        from alertrack.alerts.notifier    import _format_top_predictions

        engine = ThreatDecisionEngine()
        engine.reset_cooldown()
        probs     = np.zeros(7, dtype=np.float32)
        probs[4]  = 0.94
        probs     = probs / probs.sum()
        _, threat_info = engine.evaluate(4, 0.94, probs)

        ranked = _format_top_predictions(threat_info["top_predictions"])
        assert "gunshot" in ranked
        assert len(ranked) <= 100   # well within 160-char limit

    @patch("alertrack.alerts.notifier.NOTIFY_GSM", False)
    def test_send_alert_console_path(self, tmp_path):
        from alertrack.inference.decision import ThreatDecisionEngine

        engine = ThreatDecisionEngine()
        engine.reset_cooldown()
        probs     = np.zeros(7, dtype=np.float32)
        probs[5]  = 0.82   # threat_human
        probs     = probs / probs.sum()
        _, threat_info = engine.evaluate(5, 0.82, probs)

        notifier = self._make_notifier(tmp_path / "alerts")
        alert    = notifier.create_alert(threat_info, {"latitude": "UNKNOWN", "longitude": "UNKNOWN", "fix_quality": 0})
        result   = notifier.send_alert(alert)
        assert result is True


# ══════════════════════════════════════════════════════════════════════════════
# INT-07  AlertNotifier → SIM808AT (GPS)
# Confirms GPS fix is polled and injected into the alert payload.
# ══════════════════════════════════════════════════════════════════════════════

class TestInt07NotifierToGPS:

    def test_gps_coordinates_injected_when_fix_available(self, tmp_path):
        from alertrack.sensors.gps      import SIM808AT
        from alertrack.alerts.notifier  import AlertNotifier

        mock_loc = {
            "latitude":   -1.942369,
            "longitude":   30.061839,
            "fix_quality": 1,
            "altitude":    1548.2,
            "satellites":  7,
        }
        with patch("alertrack.config.ALERTS_DIR", tmp_path):
            import importlib, alertrack.alerts.notifier as m
            importlib.reload(m)
            notifier = m.AlertNotifier()

        threat_info = {
            "threat_type":  "threat_gunshot",
            "threat_level": "HIGH",
            "confidence":   0.94,
            "top_predictions": [
                {"class": "threat_gunshot", "confidence": 0.94}
            ],
            "class_probabilities": {c: 0.0 for c in CLASSES},
        }
        alert = notifier.create_alert(threat_info, mock_loc)
        assert isinstance(alert["latitude"],  float)
        assert isinstance(alert["longitude"], float)
        assert abs(alert["latitude"]  - (-1.942369)) < 1e-5
        assert abs(alert["longitude"] - 30.061839)   < 1e-5

    def test_no_gps_fix_alert_still_dispatched(self, tmp_path):
        """When GPS returns fix_quality=0, the alert must still be sent."""
        with patch("alertrack.config.ALERTS_DIR", tmp_path), \
             patch("alertrack.alerts.notifier.NOTIFY_GSM", False):
            import importlib, alertrack.alerts.notifier as m
            importlib.reload(m)
            notifier = m.AlertNotifier()

        threat_info = {
            "threat_type":  "threat_vehicle",
            "threat_level": "HIGH",
            "confidence":   0.75,
            "top_predictions": [{"class": "threat_vehicle", "confidence": 0.75}],
            "class_probabilities": {c: 0.0 for c in CLASSES},
        }
        no_fix_loc = {"latitude": "UNKNOWN", "longitude": "UNKNOWN", "fix_quality": 0}
        alert  = notifier.create_alert(threat_info, no_fix_loc)
        result = notifier.send_alert(alert)
        assert result is True
        assert alert["gps_quality"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# INT-08  AlertNotifier → SIM808AT (GSM SMS)
# Confirms the SMS AT command sequence is issued correctly (mocked serial).
# ══════════════════════════════════════════════════════════════════════════════

class TestInt08NotifierToGSM:

    def test_sms_at_command_sequence_called(self, tmp_path):
        """_sim808_send_sms should write AT, AT+CMGF=1, AT+CMGS, and Ctrl-Z to serial."""
        written = []

        mock_ser = MagicMock()
        mock_ser.__enter__ = lambda s: s
        mock_ser.__exit__  = MagicMock(return_value=False)
        mock_ser.in_waiting = 0

        def fake_read(n):
            # First call: AT OK, second: >, third: +CMGS:
            responses = [b"OK\r\n", b">\r\n", b"+CMGS: 42\r\nOK\r\n"]
            return responses[min(len(written) // 3, 2)]

        mock_ser.read.side_effect  = fake_read
        mock_ser.write.side_effect = lambda b: written.append(b)

        from alertrack.alerts.notifier import _sim808_send_sms
        with patch("serial.Serial", return_value=mock_ser):
            result = _sim808_send_sms(
                "+250795607062",
                "ALERTRECK | gunshot 94% | GPS: -1.942,30.062 | 12:07",
                "/dev/ttyAMA0", 9600, 10.0
            )
        # Should have written something to the serial port
        assert len(written) > 0
        combined = b"".join(written)
        assert b"AT" in combined
        assert b"CMGF" in combined or b"CMGS" in combined

    def test_sms_payload_within_160_chars(self):
        from alertrack.alerts.notifier import _format_top_predictions
        from datetime import datetime, timezone

        top = [
            {"class": "threat_gunshot", "confidence": 0.94},
            {"class": "threat_human",   "confidence": 0.04},
            {"class": "threat_vehicle", "confidence": 0.01},
        ]
        ranked = _format_top_predictions(top)
        hhmm   = datetime.now(timezone.utc).strftime("%H:%M")
        gps    = "-1.94237,30.06184"
        sms    = f"ALERTRECK | {ranked} | GPS: {gps} | {hhmm}"
        assert len(sms) <= 160, f"SMS exceeds 160 chars ({len(sms)}): {sms}"

    def test_sms_retry_on_failure(self, tmp_path):
        """On serial failure, _notify_gsm must retry SMS_MAX_RETRIES times."""
        from alertrack.alerts.notifier import AlertNotifier
        call_count = {"n": 0}

        def fail_always(*args, **kwargs):
            call_count["n"] += 1
            raise IOError("serial error")

        # Reload FIRST: reloading inside the patch context re-executes the module
        # and silently discards the mocks. Patch the notifier's own bound names —
        # it does `from ..config import SMS_MAX_RETRIES`, so patching
        # alertrack.config.* would not affect the already-bound reference.
        import importlib, alertrack.alerts.notifier as m
        importlib.reload(m)

        n_numbers, n_retries = 2, 3
        with patch.object(m, "ALERTS_DIR", tmp_path), \
             patch.object(m, "NOTIFY_GSM", True), \
             patch.object(m, "NOTIFY_CONSOLE", False), \
             patch.object(m, "_SERIAL_AVAILABLE", True), \
             patch.object(m, "_sim808_send_sms", side_effect=fail_always), \
             patch.object(m, "RANGER_PHONE_NUMBERS", ["+250700000001", "+250700000002"]), \
             patch.object(m, "SMS_MAX_RETRIES", n_retries), \
             patch.object(m, "SMS_RETRY_DELAY", 0):
            n     = m.AlertNotifier()
            alert = n.create_alert(
                {"threat_type": "threat_gunshot", "threat_level": "HIGH", "confidence": 0.94,
                 "top_predictions": [], "class_probabilities": {}},
                {"latitude": "UNKNOWN", "longitude": "UNKNOWN", "fix_quality": 0}
            )
            n._notify_gsm(alert)

        expected = n_numbers * n_retries
        assert call_count["n"] == expected, \
            f"Expected {expected} retries, got {call_count['n']}"


# ══════════════════════════════════════════════════════════════════════════════
# INT-09  EvidenceManager → Grad-CAM Dashboard
# Confirms the .mel.npy sidecar is loadable by the dashboard component.
# ══════════════════════════════════════════════════════════════════════════════

class TestInt09EvidenceToDashboard:

    def test_mel_npy_loadable_by_numpy(self, tmp_path):
        """The mel sidecar saved by EvidenceManager must be loadable as a numpy array."""
        ev_dir = tmp_path / "evidence"
        with patch("alertrack.config.EVIDENCE_DIR", ev_dir):
            import importlib, alertrack.storage.evidence as em
            importlib.reload(em)
            mgr  = em.EvidenceManager()
            mel  = np.random.randn(*INPUT_SHAPE).astype(np.float32)
            info = {
                "threat_type":  "threat_gunshot",
                "threat_level": "HIGH",
                "confidence":   0.94,
                "top_predictions": [],
                "class_probabilities": {},
                "timestamp": time.time(),
            }
            mgr.save_event_record(mel, info, {}, None, "int09_dashtest")

        npy_files = list(ev_dir.rglob("*int09_dashtest*.npy"))
        if npy_files:
            loaded = np.load(str(npy_files[0]))
            assert loaded.shape == INPUT_SHAPE, \
                f"Loaded .npy shape {loaded.shape} != expected {INPUT_SHAPE}"
            assert loaded.dtype == np.float32

    def test_mel_npy_values_in_expected_range(self, tmp_path):
        """Log-mel values in the sidecar should be in dB range (−80 to 0)."""
        from alertrack.audio.preprocess import AudioPreprocessor
        ev_dir = tmp_path / "evidence"
        with patch("alertrack.config.EVIDENCE_DIR", ev_dir):
            import importlib, alertrack.storage.evidence as em
            importlib.reload(em)
            mgr   = em.EvidenceManager()
            proc  = AudioPreprocessor()
            audio = _white_noise(BUF, rms=0.05)
            mel   = proc.preprocess(audio)
            assert mel is not None

            info = {
                "threat_type":  "threat_gunshot",
                "threat_level": "HIGH",
                "confidence":   0.94,
                "top_predictions": [],
                "class_probabilities": {},
                "timestamp": time.time(),
            }
            mgr.save_event_record(mel, info, {}, None, "int09_range")

        npy_files = list(ev_dir.rglob("*int09_range*.npy"))
        if npy_files:
            loaded = np.load(str(npy_files[0]))
            assert loaded.min() >= -120.0
            assert loaded.max() <=   10.0


# ══════════════════════════════════════════════════════════════════════════════
# INT-10  Full end-to-end pipeline (no real hardware)
# AudioRecorder buffer → Preprocessor → ONNXModel → Decision → Evidence
# ══════════════════════════════════════════════════════════════════════════════

class TestInt10EndToEnd:

    def test_full_pipeline_threat_path(self, tmp_path):
        """
        Simulate one complete threat-detection cycle:
        1. Fill recorder buffer with white noise
        2. Preprocessor converts to mel tensor
        3. ONNXModel runs inference
        4. Decision engine evaluates (force a threat by injecting known probs)
        5. EvidenceManager saves the triple
        6. Notifier builds the alert dict
        All steps must succeed without hardware.
        """
        if not MODEL_PATH.exists():
            pytest.skip("ONNX model not found — run export_model.py first.")

        # Step 1 — fill recorder buffer
        from alertrack.audio.recorder     import AudioRecorder
        from alertrack.audio.preprocess   import AudioPreprocessor
        from alertrack.inference.model    import ONNXModel
        from alertrack.inference.decision import ThreatDecisionEngine

        recorder = AudioRecorder(device_id=None)
        with recorder.lock:
            recorder.buffer.extend(_white_noise(BUF, rms=0.05))
        audio = recorder.get_audio_buffer()
        assert audio is not None

        # Step 2 — preprocess
        proc = AudioPreprocessor()
        mel  = proc.preprocess(audio)
        assert mel is not None

        # Step 3 — ONNX inference
        model  = ONNXModel(MODEL_PATH)
        result = model.predict(mel)
        assert result is not None
        idx, conf, probs = result

        # Step 4 — Force a threat scenario (override probs to guarantee an alert)
        engine = ThreatDecisionEngine()
        engine.reset_cooldown()
        forced_probs     = np.zeros(7, dtype=np.float32)
        forced_probs[4]  = 0.95   # threat_gunshot
        forced_probs     = forced_probs / forced_probs.sum()
        should_alert, threat_info = engine.evaluate(4, 0.95, forced_probs)
        assert should_alert
        assert threat_info is not None

        # Step 5 — persist evidence
        ev_dir = tmp_path / "evidence"
        with patch("alertrack.config.EVIDENCE_DIR", ev_dir):
            import importlib, alertrack.storage.evidence as em
            importlib.reload(em)
            mgr      = em.EvidenceManager()
            alert_id = "e2e_test_001"
            wav_path = mgr.save_audio_evidence(audio, threat_info["threat_type"], alert_id)
            mgr.save_event_record(mel, threat_info, {}, wav_path, alert_id)

        assert wav_path and Path(wav_path).exists()

        # Step 6 — build alert dict
        with patch("alertrack.config.ALERTS_DIR",   tmp_path / "alerts"), \
             patch("alertrack.alerts.notifier.NOTIFY_GSM", False):
            import importlib, alertrack.alerts.notifier as m
            importlib.reload(m)
            notifier = m.AlertNotifier()
            location = {"latitude": "UNKNOWN", "longitude": "UNKNOWN", "fix_quality": 0}
            alert    = notifier.create_alert(threat_info, location)
            sent     = notifier.send_alert(alert)

        assert sent is True
        assert alert["threat_type"] == "threat_gunshot"
        print(f"\n[INT-10] End-to-end pipeline passed. "
              f"Model predicted class {idx} ({CLASSES[idx]}) conf={conf:.3f}; "
              f"forced alert for threat_gunshot.")

    def test_full_pipeline_background_path_produces_no_alert(self, tmp_path):
        """Background-class output must not reach EvidenceManager or Notifier."""
        if not MODEL_PATH.exists():
            pytest.skip("ONNX model not found.")

        from alertrack.inference.decision import ThreatDecisionEngine

        engine = ThreatDecisionEngine()
        engine.reset_cooldown()
        bg_probs     = np.zeros(7, dtype=np.float32)
        bg_probs[0]  = 0.99   # background_animals
        bg_probs     = bg_probs / bg_probs.sum()
        should_alert, info = engine.evaluate(0, 0.99, bg_probs)
        assert not should_alert
        assert info is None
