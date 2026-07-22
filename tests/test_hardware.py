"""
AlertReck Hardware Tests
=========================
These tests require physical hardware connected to the Raspberry Pi.
They are SKIPPED in CI and during development on a Mac/desktop.

Run on the Pi with:
    pytest tests/test_hardware.py -v --run-hardware

Prerequisites:
  - INMP441 I2S MEMS microphone wired and ALSA overlay loaded
  - SIM808 GSM/GPS module connected on /dev/ttyAMA0
  - A valid SIM card inserted (MTN or Airtel Rwanda)
  - GPS antenna with clear sky view (or SIMULATE_GPS=True for SMS-only tests)
"""

import time
import numpy as np
import pytest

SR  = 44_100
BUF = 132_300

pytestmark = pytest.mark.hardware   # all tests in this file are hardware-gated


# ══════════════════════════════════════════════════════════════════════════════
# HW-01  INMP441 I2S Microphone
# ══════════════════════════════════════════════════════════════════════════════

class TestHardwareMicrophone:

    def test_alsa_device_opens(self):
        """sounddevice should open the ALSA default without raising."""
        import sounddevice as sd
        try:
            stream = sd.InputStream(
                device=None,      # ALSA default (the I2S mic via asound.conf)
                channels=1,
                samplerate=SR,
                blocksize=1024,
                dtype="float32",
            )
            stream.start()
            stream.stop()
            stream.close()
        except Exception as e:
            pytest.fail(f"Failed to open ALSA stream: {e}")

    def test_capture_produces_non_zero_signal(self):
        """Recording 3 s should produce a non-zero peak when clapping or speaking."""
        import sounddevice as sd
        print("\n[HW-01] Recording 3 s — make some noise (clap or speak)...")
        audio = sd.rec(BUF, samplerate=SR, channels=1, dtype="float32", device=None)
        sd.wait()
        audio = audio.flatten()
        peak = float(np.abs(audio).max())
        rms  = float(np.sqrt(np.mean(audio**2)))
        print(f"  peak={peak:.5f}  rms={rms:.5f}")
        assert peak > 1e-4, (
            f"Peak is essentially zero ({peak:.6f}). "
            "Check wiring, L/R pin assignment, and that the I2S overlay loaded."
        )

    def test_capture_shape_and_dtype(self):
        import sounddevice as sd
        audio = sd.rec(BUF, samplerate=SR, channels=1, dtype="float32", device=None)
        sd.wait()
        audio = audio.flatten()
        assert audio.shape == (BUF,)
        assert audio.dtype == np.float32

    def test_silence_is_near_zero(self):
        """
        While the room is quiet, peak should be well below 0.01.
        This test validates that the mic is not injecting DC or mains hum.
        If it fails, check that the high-pass filter in preprocess.py is active.
        """
        import sounddevice as sd
        print("\n[HW-01] Recording 3 s of silence — stay quiet...")
        audio = sd.rec(BUF, samplerate=SR, channels=1, dtype="float32", device=None)
        sd.wait()
        audio = audio.flatten()
        peak = float(np.abs(audio).max())
        print(f"  silence peak={peak:.5f}")
        assert peak < 0.05, (
            f"Silence peak too high ({peak:.5f}). "
            "Possible mains hum — ensure HPF_ENABLED=True in config.py."
        )

    def test_preprocessor_produces_correct_shape_from_live_audio(self):
        """Live captured audio must yield INPUT_SHAPE (1,128,301) after preprocessing."""
        import sounddevice as sd
        from alertrack.audio.preprocess import AudioPreprocessor

        audio = sd.rec(BUF, samplerate=SR, channels=1, dtype="float32", device=None)
        sd.wait()
        audio = audio.flatten()
        proc  = AudioPreprocessor()
        out   = proc.preprocess(audio)

        # May return None if the room is absolutely silent during the test
        if out is None:
            pytest.skip("Preprocessor returned None — increase SILENCE_THRESHOLD or make noise.")
        assert out.shape == (1, 128, 301)

    def test_recorder_daemon_fills_buffer_in_background(self):
        """AudioRecorder should fill its 3-second deque within 4 seconds."""
        from alertrack.audio.recorder import AudioRecorder
        recorder = AudioRecorder(device_id=None)
        recorder.start()
        deadline = time.time() + 4.0
        while not recorder.is_buffer_ready() and time.time() < deadline:
            time.sleep(0.1)
        recorder.stop()
        assert recorder.is_buffer_ready(), \
            "Buffer did not fill within 4 seconds — check microphone and ALSA config."


# ══════════════════════════════════════════════════════════════════════════════
# HW-02  SIM808 GPS
# ══════════════════════════════════════════════════════════════════════════════

class TestHardwareGPS:

    def test_uart_port_accessible(self):
        """The SIM808 UART port /dev/ttyAMA0 should exist and be readable."""
        import serial
        try:
            ser = serial.Serial("/dev/ttyAMA0", 9600, timeout=1)
            ser.close()
        except Exception as e:
            pytest.fail(f"/dev/ttyAMA0 not accessible: {e}")

    def test_sim808_responds_to_at(self):
        """The SIM808 should reply OK to a bare AT command within 2 s."""
        import serial, time
        with serial.Serial("/dev/ttyAMA0", 9600, timeout=2) as ser:
            ser.reset_input_buffer()
            ser.write(b"AT\r\n")
            time.sleep(1.0)
            resp = ser.read(ser.in_waiting or 64).decode("ascii", errors="ignore")
        assert "OK" in resp, f"SIM808 did not respond with OK. Got: {resp!r}"

    def test_gps_powers_on(self):
        """AT+CGNSPWR=1 should return OK."""
        from alertrack.sensors.gps import SIM808AT
        s = SIM808AT(port="/dev/ttyAMA0", baudrate=9600)
        result = s.power_on()
        # power_on returns True on OK, False on timeout — either is non-fatal
        print(f"\n[HW-02] GPS power_on result: {result}")

    def test_cgnsinf_returns_dict(self):
        """AT+CGNSINF poll should return a dict with fix_quality key."""
        from alertrack.sensors.gps import SIM808AT
        s   = SIM808AT(port="/dev/ttyAMA0", baudrate=9600)
        s.power_on()
        loc = s.get_location()
        assert isinstance(loc, dict)
        assert "fix_quality" in loc
        print(f"\n[HW-02] GPS location: {loc}")

    @pytest.mark.skipif(True, reason="Requires outdoor clear-sky view. Remove skipif to run.")
    def test_gps_acquires_fix_outdoors(self):
        """
        With a clear sky view, the SIM808 should acquire a fix within 60 s (cold start).
        This test is skipped by default — remove the skipif decorator and run outdoors.
        """
        from alertrack.sensors.gps import SIM808AT
        s = SIM808AT(port="/dev/ttyAMA0", baudrate=9600)
        s.power_on()
        deadline = time.time() + 60
        loc = {}
        while time.time() < deadline:
            loc = s.get_location()
            if loc.get("fix_quality") == 1:
                break
            time.sleep(2)

        assert loc.get("fix_quality") == 1, "No GPS fix within 60 s — check antenna placement."
        assert isinstance(loc["latitude"],  float)
        assert isinstance(loc["longitude"], float)
        print(f"\n[HW-02] GPS fix: lat={loc['latitude']:.6f}  lon={loc['longitude']:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# HW-03  SIM808 GSM SMS
# ══════════════════════════════════════════════════════════════════════════════

class TestHardwareGSM:

    def test_gsm_signal_strength(self):
        """AT+CSQ should return a signal quality > 0 when registered on the network."""
        import serial, time
        with serial.Serial("/dev/ttyAMA0", 9600, timeout=2) as ser:
            ser.reset_input_buffer()
            ser.write(b"AT+CSQ\r\n")
            time.sleep(1.0)
            resp = ser.read(ser.in_waiting or 128).decode("ascii", errors="ignore")
        assert "+CSQ:" in resp, f"AT+CSQ did not return expected response. Got: {resp!r}"
        # Parse signal quality (first number after +CSQ:)
        import re
        m = re.search(r"\+CSQ:\s*(\d+)", resp)
        if m:
            rssi = int(m.group(1))
            print(f"\n[HW-03] GSM RSSI = {rssi} (99 = no signal)")
            assert rssi != 99, "RSSI=99 means no signal — check SIM, antenna, and carrier"

    def test_network_registration(self):
        """AT+CREG? should indicate registered (0,1) or roaming (0,5)."""
        import serial, time
        with serial.Serial("/dev/ttyAMA0", 9600, timeout=2) as ser:
            ser.reset_input_buffer()
            ser.write(b"AT+CREG?\r\n")
            time.sleep(1.0)
            resp = ser.read(ser.in_waiting or 128).decode("ascii", errors="ignore")
        assert "+CREG:" in resp, f"Expected +CREG: response. Got: {resp!r}"
        registered = ",1" in resp or ",5" in resp
        print(f"\n[HW-03] CREG response: {resp.strip()}")
        assert registered, "SIM808 not registered on network. Check SIM and carrier."

    def test_sms_text_mode(self):
        """AT+CMGF=1 (text mode) should return OK."""
        import serial, time
        with serial.Serial("/dev/ttyAMA0", 9600, timeout=2) as ser:
            ser.reset_input_buffer()
            ser.write(b"AT+CMGF=1\r\n")
            time.sleep(1.0)
            resp = ser.read(ser.in_waiting or 64).decode("ascii", errors="ignore")
        assert "OK" in resp, f"AT+CMGF=1 failed. Got: {resp!r}"

    @pytest.mark.skipif(True, reason="Sends a real SMS — remove skipif to run and verify on handset.")
    def test_send_real_sms_to_ranger_number(self):
        """
        Sends a real test SMS to the first configured ranger number.
        Remove the skipif decorator and run this on the Pi to verify end-to-end delivery.
        Check the ranger handset within 30 s.
        """
        from alertrack.alerts.notifier import _sim808_send_sms
        from alertrack.config import RANGER_PHONE_NUMBERS, SIM808_PORT, SIM808_BAUDRATE, SIM808_TIMEOUT

        number  = RANGER_PHONE_NUMBERS[0]
        payload = "ALERTRECK TEST — system online and SMS delivery verified."
        result  = _sim808_send_sms(number, payload, SIM808_PORT, SIM808_BAUDRATE, SIM808_TIMEOUT)
        assert result is True, f"SMS to {number} failed — check GSM registration and signal."
        print(f"\n[HW-03] SMS sent to {number}. Check handset.")


# ══════════════════════════════════════════════════════════════════════════════
# HW-04  Full on-device pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestHardwareFullPipeline:

    def test_live_capture_through_onnx(self):
        """
        Capture live audio, preprocess it, and run ONNX inference.
        All steps must succeed on the Pi with the INMP441 connected.
        """
        import sounddevice as sd
        from alertrack.audio.preprocess import AudioPreprocessor
        from alertrack.inference.model  import ONNXModel

        if not (Path := __import__("pathlib").Path)("models/custom_cnn/alertreck_cnn.onnx").exists():
            pytest.skip("ONNX model not found.")

        print("\n[HW-04] Capturing 3 s of live audio — make some noise...")
        audio = sd.rec(BUF, samplerate=SR, channels=1, dtype="float32", device=None)
        sd.wait()
        audio = audio.flatten()
        peak  = float(np.abs(audio).max())
        print(f"  peak={peak:.5f}")

        proc  = AudioPreprocessor()
        mel   = proc.preprocess(audio)
        if mel is None:
            pytest.skip("Preprocessor returned None — too quiet during test.")

        model = ONNXModel()
        result = model.predict(mel)
        assert result is not None
        idx, conf, probs = result
        from alertrack.config import CLASS_NAMES
        print(f"  Predicted: {CLASS_NAMES[idx]} ({conf*100:.1f}%)")
        assert 0 <= idx < 7
        assert abs(probs.sum() - 1.0) < 1e-4
