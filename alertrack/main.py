#!/usr/bin/env python3
"""
ALERTRACK Main Entry Point

Integrates all modules into a continuous real-time threat detection system.
Runs 24/7 on Raspberry Pi with full fault tolerance.
"""

import sys
import time
import signal
import traceback
from pathlib import Path
from typing import Optional

# Local imports
from config import (
    MODEL_PATH,
    INFERENCE_INTERVAL,
    ENABLE_GPS,
    STATS_INTERVAL,
    validate_config,
    CLASS_LABELS
)
from audio.recorder import AudioRecorder
from audio.preprocess import AudioPreprocessor
from inference.model import TFLiteModel
from inference.decision import ThreatDecisionEngine
from sensors.gps import GPSReader
from alerts.notifier import AlertNotifier
from storage.logger import get_logger
from storage.evidence import EvidenceManager
from utils import PerformanceTimer, get_timestamp


# Global shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger = get_logger()
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True


class ALERTRACKSystem:
    """Main system class integrating all components."""
    
    def __init__(self):
        """Initialize all system components."""
        self.logger = get_logger()
        self.logger.info("=" * 80)
        self.logger.info("ALERTRACK Anti-Poaching Edge System v1.0")
        self.logger.info("=" * 80)
        
        # Component references
        self.recorder: Optional[AudioRecorder] = None
        self.preprocessor: Optional[AudioPreprocessor] = None
        self.model: Optional[TFLiteModel] = None
        self.decision_engine: Optional[ThreatDecisionEngine] = None
        self.gps: Optional[GPSReader] = None
        self.notifier: Optional[AlertNotifier] = None
        self.evidence_manager: Optional[EvidenceManager] = None
        
        # Runtime state
        self.total_inferences = 0
        self.total_threats = 0
        self.start_time = time.time()
        self.last_stats_time = time.time()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components with error handling."""
        try:
            # Validate configuration first
            self.logger.info("Validating system configuration...")
            validate_config()
            self.logger.info("✓ Configuration validated")
            
            # Initialize logger and evidence manager first
            self.logger.info("Initializing storage components...")
            self.evidence_manager = EvidenceManager()
            self.logger.info("✓ Evidence manager initialized")
            
            # Initialize model
            self.logger.info(f"Loading TFLite model from: {MODEL_PATH}")
            self.model = TFLiteModel(MODEL_PATH)
            model_info = self.model.get_model_info()
            self.logger.info(f"✓ Model loaded: {model_info['input_shape']} → {model_info['output_shape']}")
            
            # Initialize preprocessor
            self.logger.info("Initializing audio preprocessor...")
            target_shape = model_info['input_shape'][1:3]  # Remove batch and channel dims
            self.preprocessor = AudioPreprocessor(target_shape=target_shape)
            self.logger.info(f"✓ Preprocessor initialized (target shape: {target_shape})")
            
            # Initialize decision engine
            self.logger.info("Initializing threat decision engine...")
            self.decision_engine = ThreatDecisionEngine()
            self.logger.info("✓ Decision engine initialized")
            
            # Initialize GPS if enabled
            if ENABLE_GPS:
                self.logger.info("Initializing GPS reader...")
                try:
                    self.gps = GPSReader()
                    self.gps.start()
                    self.logger.info("✓ GPS reader started")
                except Exception as e:
                    self.logger.warning(f"GPS initialization failed: {e}. Continuing without GPS.")
                    self.gps = None
            else:
                self.logger.info("GPS disabled in configuration")
                self.gps = None
            
            # Initialize alert notifier
            self.logger.info("Initializing alert notifier...")
            self.notifier = AlertNotifier(gps_reader=self.gps)
            self.logger.info("✓ Alert notifier initialized")
            
            # Initialize audio recorder (start last)
            self.logger.info("Initializing audio recorder...")
            self.recorder = AudioRecorder()
            self.recorder.start()
            self.logger.info("✓ Audio recorder started")
            
            self.logger.info("=" * 80)
            self.logger.info("All components initialized successfully!")
            self.logger.info("System ready for threat detection.")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _process_audio_chunk(self):
        """Process one audio chunk through the inference pipeline."""
        try:
            # Check if buffer is ready
            if not self.recorder.is_buffer_ready():
                return
            
            # Get audio buffer
            with PerformanceTimer() as timer:
                audio_buffer = self.recorder.get_audio_buffer()
            
            if audio_buffer is None:
                self.logger.warning("Failed to get audio buffer")
                return
            
            # Preprocess audio to mel spectrogram
            with PerformanceTimer() as preprocess_timer:
                mel_spec = self.preprocessor.preprocess(audio_buffer)
            
            # Run inference
            with PerformanceTimer() as inference_timer:
                class_idx, confidence, probabilities = self.model.predict(mel_spec)
            
            class_name = self.model.get_class_name(class_idx)
            self.total_inferences += 1
            
            # Log inference result
            self.logger.debug(
                f"Inference #{self.total_inferences}: "
                f"{class_name} ({confidence:.1%}) | "
                f"Preprocess: {preprocess_timer.elapsed:.3f}s | "
                f"Inference: {inference_timer.elapsed:.3f}s"
            )
            
            # Evaluate threat decision
            should_alert, threat_info = self.decision_engine.evaluate(
                class_idx, confidence, probabilities
            )
            
            if should_alert:
                self._handle_threat(threat_info, audio_buffer)
            
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")
            self.logger.debug(traceback.format_exc())
    
    def _handle_threat(self, threat_info: dict, audio_buffer):
        """Handle detected threat: save evidence and send alert."""
        try:
            threat_type = threat_info['threat_type']
            confidence = threat_info['confidence']
            threat_level = threat_info['threat_level']
            
            self.total_threats += 1
            
            self.logger.warning(
                f"🚨 THREAT DETECTED #{self.total_threats}: "
                f"{threat_type} ({confidence:.1%}) - Level: {threat_level}"
            )
            
            # Save audio evidence
            evidence_path = None
            try:
                evidence_path = self.evidence_manager.save_audio_evidence(
                    audio_buffer,
                    threat_type,
                    metadata={
                        'confidence': confidence,
                        'threat_level': threat_level,
                        'class_probabilities': threat_info['class_probabilities']
                    }
                )
                self.logger.info(f"✓ Evidence saved: {evidence_path}")
            except Exception as e:
                self.logger.error(f"Failed to save evidence: {e}")
            
            # Create and send alert
            try:
                alert = self.notifier.create_alert(
                    threat_type=threat_type,
                    threat_level=threat_level,
                    confidence=confidence,
                    class_probabilities=threat_info['class_probabilities'],
                    audio_evidence=evidence_path
                )
                
                self.notifier.send_alert(alert)
                self.logger.info(f"✓ Alert sent: {alert['alert_id']}")
                
            except Exception as e:
                self.logger.error(f"Failed to send alert: {e}")
        
        except Exception as e:
            self.logger.error(f"Error handling threat: {e}")
            self.logger.debug(traceback.format_exc())
    
    def _log_statistics(self):
        """Log system statistics periodically."""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Get component statistics
            recorder_stats = self.recorder.get_stats() if self.recorder else {}
            decision_stats = self.decision_engine.get_stats() if self.decision_engine else {}
            evidence_stats = self.evidence_manager.get_storage_stats() if self.evidence_manager else {}
            
            # Calculate rates
            uptime_hours = uptime / 3600
            inferences_per_hour = self.total_inferences / uptime_hours if uptime_hours > 0 else 0
            threats_per_hour = self.total_threats / uptime_hours if uptime_hours > 0 else 0
            
            self.logger.info("=" * 80)
            self.logger.info("SYSTEM STATISTICS")
            self.logger.info("=" * 80)
            self.logger.info(f"Uptime: {uptime/3600:.1f} hours")
            self.logger.info(f"Total Inferences: {self.total_inferences} ({inferences_per_hour:.1f}/hour)")
            self.logger.info(f"Total Threats: {self.total_threats} ({threats_per_hour:.2f}/hour)")
            
            if recorder_stats:
                self.logger.info(f"Audio Chunks: {recorder_stats.get('total_chunks', 0)}")
                self.logger.info(f"Buffer Overflows: {recorder_stats.get('buffer_overflows', 0)}")
            
            if decision_stats:
                self.logger.info(f"Detections: {decision_stats.get('total_detections', 0)}")
                self.logger.info(f"Suppressed (cooldown): {decision_stats.get('suppressed_by_cooldown', 0)}")
            
            if evidence_stats:
                self.logger.info(f"Evidence Files: {evidence_stats.get('total_files', 0)}")
                self.logger.info(f"Evidence Storage: {evidence_stats.get('total_size_mb', 0):.1f} MB")
            
            # GPS status
            if self.gps:
                has_fix = self.gps.has_fix()
                coords = self.gps.get_coordinates()
                if has_fix and coords[0] is not None:
                    self.logger.info(f"GPS: Fix acquired ({coords[0]:.6f}, {coords[1]:.6f})")
                else:
                    self.logger.info("GPS: Searching for fix...")
            
            self.logger.info("=" * 80)
            
            self.last_stats_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error logging statistics: {e}")
    
    def run(self):
        """Main system loop."""
        global shutdown_requested
        
        self.logger.info("Starting main inference loop...")
        self.logger.info(f"Inference interval: {INFERENCE_INTERVAL}s")
        
        try:
            while not shutdown_requested:
                # Process audio chunk
                self._process_audio_chunk()
                
                # Log statistics periodically
                if time.time() - self.last_stats_time >= STATS_INTERVAL:
                    self._log_statistics()
                
                # Sleep before next inference
                time.sleep(INFERENCE_INTERVAL)
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all components."""
        self.logger.info("=" * 80)
        self.logger.info("Shutting down ALERTRACK system...")
        self.logger.info("=" * 80)
        
        # Log final statistics
        self._log_statistics()
        
        # Stop components in reverse order
        if self.recorder:
            try:
                self.logger.info("Stopping audio recorder...")
                self.recorder.stop()
                self.logger.info("✓ Audio recorder stopped")
            except Exception as e:
                self.logger.error(f"Error stopping recorder: {e}")
        
        if self.gps:
            try:
                self.logger.info("Stopping GPS reader...")
                self.gps.stop()
                self.logger.info("✓ GPS reader stopped")
            except Exception as e:
                self.logger.error(f"Error stopping GPS: {e}")
        
        # Cleanup old evidence files
        if self.evidence_manager:
            try:
                self.logger.info("Cleaning up old evidence files...")
                deleted = self.evidence_manager.cleanup_old_evidence()
                if deleted > 0:
                    self.logger.info(f"✓ Deleted {deleted} old evidence files")
            except Exception as e:
                self.logger.error(f"Error cleaning up evidence: {e}")
        
        self.logger.info("=" * 80)
        self.logger.info("ALERTRACK system shutdown complete")
        self.logger.info(f"Total runtime: {(time.time() - self.start_time)/3600:.1f} hours")
        self.logger.info(f"Total inferences: {self.total_inferences}")
        self.logger.info(f"Total threats detected: {self.total_threats}")
        self.logger.info("=" * 80)


def main():
    """Main entry point."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger = get_logger()
    
    try:
        # Create and run system
        system = ALERTRACKSystem()
        system.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
