#!/usr/bin/env python3
"""
ALERTRACK Main Entry Point
===========================
Integrates all modules into a continuous real-time threat detection system.
Runs 24/7 on Raspberry Pi with full fault tolerance.
"""

import sys
import time
import signal
import traceback
from pathlib import Path
from typing import Optional

from .config import (
    MODEL_PATH,
    INFERENCE_INTERVAL,
    ENABLE_GPS,
    STATS_INTERVAL,
    CLASS_LABELS,
    ONSET_ENABLED,
    ONSET_POLL_INTERVAL,
    ONSET_SETTLE_S,
    ONSET_TRIGGER_DB,
    ONSET_REFRACTORY_S,
    validate_config,
)
from .audio.recorder import AudioRecorder
from .audio.preprocess import AudioPreprocessor
from .audio.onset import OnsetDetector
from .inference.model import ONNXModel
from .inference.decision import ThreatDecisionEngine
from .sensors.gps import SIM808AT
from .alerts.notifier import AlertNotifier
from .storage.logger import get_logger
from .storage.evidence import EvidenceManager
from .utils import PerformanceTimer, get_timestamp


shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    get_logger().info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True


class ALERTRACKSystem:
    """Main system class integrating all components."""

    def __init__(self):
        self.logger = get_logger()
        self.logger.info("=" * 80)
        self.logger.info("ALERTRACK Anti-Poaching Edge System v2.0")
        self.logger.info(f"Classes: {CLASS_LABELS}")
        self.logger.info("=" * 80)

        self.recorder: Optional[AudioRecorder] = None
        self.preprocessor: Optional[AudioPreprocessor] = None
        self.onset: Optional[OnsetDetector] = None
        self.model: Optional[ONNXModel] = None
        self.decision_engine: Optional[ThreatDecisionEngine] = None
        self.gps: Optional[SIM808AT] = None
        self.notifier: Optional[AlertNotifier] = None
        self.evidence_manager: Optional[EvidenceManager] = None

        self.total_inferences = 0
        self.total_threats    = 0
        self.start_time       = time.time()
        self.last_stats_time  = time.time()

        self._initialize_components()

    def _initialize_components(self):
        try:
            self.logger.info("Validating configuration...")
            issues = validate_config()
            if issues:
                for issue in issues:
                    self.logger.warning(f"Config issue: {issue}")
            else:
                self.logger.info("Configuration OK")

            self.evidence_manager = EvidenceManager()
            self.logger.info("Evidence manager ready")

            self.logger.info(f"Loading ONNX model: {MODEL_PATH}")
            self.model = ONNXModel(MODEL_PATH)
            info = self.model.get_model_info()
            self.logger.info(f"Model loaded: {info['input_shape']} → {info['num_classes']} classes")

            self.preprocessor = AudioPreprocessor()
            self.logger.info("Preprocessor ready")

            if ONSET_ENABLED:
                self.onset = OnsetDetector()
                self.logger.info("Onset detector ready")

            self.decision_engine = ThreatDecisionEngine()
            self.logger.info("Decision engine ready")

            if ENABLE_GPS:
                try:
                    # Poll-on-demand GPS: shares the SIM808 UART with SMS, so it must
                    # NOT hold the port open. It opens/closes only when polled.
                    self.gps = SIM808AT()
                    if self.gps.power_on():
                        self.logger.info("SIM808 GPS powered on (poll-on-demand)")
                    else:
                        self.logger.warning("SIM808 GPS power-on unconfirmed — will retry on poll")
                except Exception as e:
                    self.logger.warning(f"GPS unavailable: {e} — continuing without GPS")
                    self.gps = None
            else:
                self.logger.info("GPS disabled")
                self.gps = None

            self.notifier = AlertNotifier()
            self.logger.info("Alert notifier ready")

            self.recorder = AudioRecorder()
            self.recorder.start()
            self.logger.info("Audio recorder started")

            self.logger.info("=" * 80)
            self.logger.info("System ready — listening for threats.")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"Initialisation failed: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _process_audio_chunk(self):
        try:
            if not self.recorder.is_buffer_ready():
                return

            audio_buffer = self.recorder.get_audio_buffer()
            if audio_buffer is None:
                return

            mel_spec = self.preprocessor.preprocess(audio_buffer)
            if mel_spec is None:
                return

            result = self.model.predict(mel_spec)
            if result is None:
                return

            class_idx, confidence, probabilities = result
            class_name = self.model.get_class_name(class_idx)
            self.total_inferences += 1

            self.logger.debug(
                f"#{self.total_inferences}  {class_name}  {confidence:.1%}"
            )

            should_alert, threat_info = self.decision_engine.evaluate(
                class_idx, confidence, probabilities
            )

            if should_alert:
                self._handle_threat(threat_info, audio_buffer, mel_spec)

        except Exception as e:
            self.logger.error(f"Audio processing error: {e}")
            self.logger.debug(traceback.format_exc())

    def _handle_threat(self, threat_info: dict, audio_buffer, mel_spec=None):
        try:
            threat_type  = threat_info["threat_type"]
            confidence   = threat_info["confidence"]
            threat_level = threat_info["threat_level"]
            self.total_threats += 1

            self.logger.warning(
                f"THREAT #{self.total_threats}: {threat_type}  "
                f"{confidence:.1%}  level={threat_level}"
            )

            # Generate alert_id first so evidence filename matches alert
            from .utils import generate_alert_id
            alert_id = generate_alert_id()

            # Resolve location with a single GPS poll (one UART open) — used by both
            # the event record and the alert. Done before SMS so the port is free.
            location = self.gps.get_location() if self.gps else {}

            evidence_path = None
            try:
                evidence_path = self.evidence_manager.save_audio_evidence(
                    audio_buffer,
                    threat_type,
                    alert_id,
                )
                self.logger.info(f"Evidence saved: {evidence_path}")
            except Exception as e:
                self.logger.error(f"Evidence save failed: {e}")

            # Save mel + metadata sidecars for the Grad-CAM dashboard
            if mel_spec is not None:
                try:
                    self.evidence_manager.save_event_record(
                        mel_spec, threat_info, location, evidence_path, alert_id
                    )
                except Exception as e:
                    self.logger.error(f"Event record save failed: {e}")

            try:
                alert = self.notifier.create_alert(
                    threat_info=threat_info,
                    location=location,
                    audio_path=str(evidence_path) if evidence_path else None,
                )
                self.notifier.send_alert(alert)
                self.logger.info(f"Alert sent: {alert['alert_id']}")
            except Exception as e:
                self.logger.error(f"Alert send failed: {e}")

        except Exception as e:
            self.logger.error(f"Threat handling error: {e}")

    def _log_statistics(self):
        try:
            uptime       = time.time() - self.start_time
            uptime_hours = uptime / 3600 or 1e-9

            rec_stats = self.recorder.get_stats()          if self.recorder          else {}
            dec_stats = self.decision_engine.get_stats()   if self.decision_engine   else {}
            evi_stats = self.evidence_manager.get_storage_stats() if self.evidence_manager else {}

            self.logger.info("=" * 80)
            self.logger.info("SYSTEM STATISTICS")
            self.logger.info(f"Uptime          : {uptime/3600:.1f} h")
            self.logger.info(f"Inferences      : {self.total_inferences}  "
                             f"({self.total_inferences/uptime_hours:.0f}/h)")
            self.logger.info(f"Threats fired   : {self.total_threats}")
            if dec_stats:
                self.logger.info(f"Suppressed      : {dec_stats.get('total_suppressed', 0)}")
                counts = dec_stats.get("threat_counts", {})
                if counts:
                    self.logger.info(f"Per-class counts: {counts}")
            if rec_stats:
                self.logger.info(f"Buffer overflows: {rec_stats.get('buffer_overflows', 0)}")
            if evi_stats:
                self.logger.info(f"Evidence        : {evi_stats.get('total_files', 0)} files  "
                                 f"{evi_stats.get('total_size_mb', 0):.0f} MB")
            if self.gps:
                loc = self.gps.get_location()   # single poll
                if loc.get("fix_quality") == 1:
                    self.logger.info(f"GPS             : {loc['latitude']:.6f}, {loc['longitude']:.6f}  "
                                     f"({loc.get('satellites', 0)} sats)")
                else:
                    self.logger.info("GPS             : searching...")
            self.logger.info("=" * 80)
            self.last_stats_time = time.time()

        except Exception as e:
            self.logger.error(f"Stats log error: {e}")

    def run(self):
        try:
            if ONSET_ENABLED and self.onset is not None:
                self._run_onset_loop()
            else:
                self._run_timer_loop()
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Fatal loop error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.shutdown()

    def _run_timer_loop(self):
        """Legacy mode — classify on a fixed interval regardless of content."""
        global shutdown_requested
        self.logger.info(f"Inference loop started  (timer mode, interval={INFERENCE_INTERVAL}s)")

        while not shutdown_requested:
            self._process_audio_chunk()

            if time.time() - self.last_stats_time >= STATS_INTERVAL:
                self._log_statistics()

            time.sleep(INFERENCE_INTERVAL)

    def _run_onset_loop(self):
        """Onset-triggered mode — classify only when an energy onset is detected."""
        global shutdown_requested
        self.logger.info(
            f"Inference loop started  (onset mode, poll={ONSET_POLL_INTERVAL}s, "
            f"trigger=+{ONSET_TRIGGER_DB:.0f}dB, refractory={ONSET_REFRACTORY_S:.1f}s)"
        )

        while not shutdown_requested:
            if self.recorder.is_buffer_ready():
                audio = self.recorder.get_audio_buffer()
                if audio is not None:
                    triggered, info = self.onset.check(audio)
                    if triggered:
                        self.logger.info(
                            f"Onset detected (+{info['margin_db']:.0f}dB over floor) — "
                            f"settling {ONSET_SETTLE_S:.1f}s, then classifying"
                        )
                        # Let the event move toward the centre of the rolling buffer
                        time.sleep(ONSET_SETTLE_S)
                        self._process_audio_chunk()

            if time.time() - self.last_stats_time >= STATS_INTERVAL:
                self._log_statistics()

            time.sleep(ONSET_POLL_INTERVAL)

    def shutdown(self):
        self.logger.info("Shutting down ALERTRACK...")
        self._log_statistics()

        for component, name in [
            (self.recorder, "recorder"),
            (self.gps, "GPS"),
        ]:
            if component:
                try:
                    component.stop()
                    self.logger.info(f"{name} stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping {name}: {e}")

        if self.evidence_manager:
            try:
                deleted = self.evidence_manager.cleanup_old_evidence()
                if deleted:
                    self.logger.info(f"Cleaned {deleted} old evidence files")
            except Exception as e:
                self.logger.error(f"Evidence cleanup error: {e}")

        self.logger.info(
            f"Shutdown complete — runtime={((time.time()-self.start_time)/3600):.1f}h  "
            f"inferences={self.total_inferences}  threats={self.total_threats}"
        )


def main():
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger = get_logger()
    try:
        ALERTRACKSystem().run()
    except Exception as e:
        logger.error(f"Fatal: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
