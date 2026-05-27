"""
Alert Notifier Module
======================
Generates and dispatches alerts via console, disk, and SIM808 GSM SMS.
SMS payload: ALERTRECK | <class> | Conf: x% | GPS: lat,lon | HH:MM
"""

import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path

from ..config import (
    ALERTS_DIR, DEVICE_ID, DEVICE_LOCATION,
    NOTIFY_CONSOLE, NOTIFY_LORA, NOTIFY_GSM, NOTIFY_SATELLITE,
    SIM808_PORT, SIM808_BAUDRATE, SIM808_TIMEOUT,
    RANGER_PHONE_NUMBERS, SMS_MAX_RETRIES, SMS_RETRY_DELAY,
)
from ..utils import get_timestamp, generate_alert_id, save_json, format_confidence, format_coords

try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


def _sim808_at(ser, cmd: str, expected: str = "OK", timeout: float = 5.0) -> bool:
    """Send one AT command and wait for expected response."""
    ser.write((cmd + "\r\n").encode())
    deadline = time.time() + timeout
    buf = ""
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1).decode("ascii", errors="ignore")
        buf += chunk
        if expected in buf:
            return True
        if "ERROR" in buf:
            return False
    return False


def _sim808_send_sms(number: str, text: str,
                     port: str, baudrate: int, timeout: float) -> bool:
    """Send one SMS via SIM808 AT commands. Returns True on success."""
    with serial.Serial(port, baudrate, timeout=2.0) as ser:
        time.sleep(0.5)
        if not _sim808_at(ser, "AT"):
            raise RuntimeError("SIM808 not responding to AT")
        if not _sim808_at(ser, "AT+CMGF=1"):   # text mode
            raise RuntimeError("Could not set SMS text mode")
        ser.write(f'AT+CMGS="{number}"\r'.encode())
        time.sleep(0.5)
        buf = ser.read(ser.in_waiting or 1).decode("ascii", errors="ignore")
        if ">" not in buf:
            raise RuntimeError("No SMS prompt received")
        ser.write(text.encode() + bytes([26]))   # message + Ctrl-Z
        deadline = time.time() + timeout
        buf = ""
        while time.time() < deadline:
            chunk = ser.read(ser.in_waiting or 1).decode("ascii", errors="ignore")
            buf += chunk
            if "+CMGS:" in buf:
                return True
            if "ERROR" in buf:
                return False
        return False


class AlertNotifier:
    """
    Manages alert generation and notification dispatch.
    """
    
    def __init__(self):
        """Initialize alert notifier."""
        self.device_id = DEVICE_ID
        self.device_location = DEVICE_LOCATION
        self.alerts_sent = 0
        
        # Ensure alerts directory exists
        ALERTS_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"AlertNotifier initialized:")
        print(f"  Device ID: {self.device_id}")
        print(f"  Location: {self.device_location}")
        print(f"  Console: {NOTIFY_CONSOLE}")
        print(f"  LoRaWAN: {NOTIFY_LORA}")
        print(f"  GSM: {NOTIFY_GSM}")
        print(f"  Satellite: {NOTIFY_SATELLITE}")
    
    def create_alert(
        self,
        threat_info: Dict[str, Any],
        location: Dict[str, Any],
        audio_path: str = None
    ) -> Dict[str, Any]:
        """
        Create alert object from threat detection.
        
        Args:
            threat_info: Threat information from decision engine
            location: GPS location data
            audio_path: Path to saved audio evidence
            
        Returns:
            Complete alert dictionary
        """
        alert = {
            # Alert metadata
            'alert_id': generate_alert_id(),
            'timestamp': get_timestamp(),
            'device_id': self.device_id,
            'device_location': self.device_location,
            
            # Threat information
            'threat_type': threat_info['threat_type'],
            'threat_level': threat_info['threat_level'],
            'confidence': threat_info['confidence'],
            'class_probabilities': threat_info['class_probabilities'],
            
            # Location
            'latitude': location.get('latitude', 'UNKNOWN'),
            'longitude': location.get('longitude', 'UNKNOWN'),
            'altitude': location.get('altitude', 'UNKNOWN'),
            'gps_quality': location.get('fix_quality', 0),
            'gps_satellites': location.get('satellites', 0),
            
            # Evidence
            'audio_evidence': audio_path,
            
            # Status
            'status': 'PENDING',  # PENDING, ACKNOWLEDGED, RESOLVED
            'notified_channels': []
        }
        
        return alert
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Send alert through all enabled notification channels.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if at least one channel succeeded
        """
        success = False
        
        # Save alert to disk (always)
        if self._save_alert_to_disk(alert):
            alert['notified_channels'].append('DISK')
            success = True
        
        # Console notification
        if NOTIFY_CONSOLE:
            if self._notify_console(alert):
                alert['notified_channels'].append('CONSOLE')
                success = True
        
        # LoRaWAN notification (stub)
        if NOTIFY_LORA:
            if self._notify_lora(alert):
                alert['notified_channels'].append('LORA')
                success = True
        
        # GSM notification (stub)
        if NOTIFY_GSM:
            if self._notify_gsm(alert):
                alert['notified_channels'].append('GSM')
                success = True
        
        # Satellite notification (stub)
        if NOTIFY_SATELLITE:
            if self._notify_satellite(alert):
                alert['notified_channels'].append('SATELLITE')
                success = True
        
        if success:
            self.alerts_sent += 1
        
        return success
    
    def _save_alert_to_disk(self, alert: Dict[str, Any]) -> bool:
        """Save alert as JSON file."""
        try:
            filename = f"{alert['alert_id']}_{alert['timestamp'][:10]}.json"
            filepath = ALERTS_DIR / filename
            
            return save_json(alert, filepath)
        
        except Exception as e:
            print(f"❌ Failed to save alert to disk: {e}")
            return False
    
    def _notify_console(self, alert: Dict[str, Any]) -> bool:
        """Print alert to console."""
        try:
            print("\n" + "=" * 70)
            print("🚨 THREAT ALERT 🚨")
            print("=" * 70)
            print(f"Alert ID:      {alert['alert_id']}")
            print(f"Timestamp:     {alert['timestamp']}")
            print(f"Device:        {alert['device_id']} ({alert['device_location']})")
            print(f"Threat Type:   {alert['threat_type']}")
            print(f"Threat Level:  {alert['threat_level']}")
            print(f"Confidence:    {format_confidence(alert['confidence'])}")
            
            # Location
            lat = alert['latitude']
            lon = alert['longitude']
            if lat != 'UNKNOWN' and lon != 'UNKNOWN':
                print(f"Location:      {format_coords(lat, lon)}")
                print(f"Altitude:      {alert['altitude']}m")
                print(f"GPS Quality:   {alert['gps_quality']} ({alert['gps_satellites']} satellites)")
            else:
                print(f"Location:      UNKNOWN (GPS unavailable)")
            
            # Evidence
            if alert['audio_evidence']:
                print(f"Audio Evidence: {alert['audio_evidence']}")
            
            print("=" * 70)
            print()
            
            return True
        
        except Exception as e:
            print(f"❌ Console notification failed: {e}")
            return False
    
    def _notify_lora(self, alert: Dict[str, Any]) -> bool:
        """
        Send alert via LoRaWAN.
        
        STUB IMPLEMENTATION - To be completed with actual LoRaWAN hardware.
        
        For real implementation:
        - Use LoRaWAN module (e.g., RFM95W)
        - Encode alert as compact payload
        - Send to LoRaWAN gateway
        - Handle acknowledgments
        """
        print(f"📡 [LoRaWAN STUB] Would send alert {alert['alert_id']}")
        
        # Simulate success for testing
        return True
    
    def _notify_gsm(self, alert: Dict[str, Any]) -> bool:
        """Send SMS alert via SIM808 GSM module using AT commands."""
        if not RANGER_PHONE_NUMBERS:
            print("GSM: no ranger phone numbers configured in config.py")
            return False

        lat  = alert.get("latitude",  "?")
        lon  = alert.get("longitude", "?")
        conf = int(alert["confidence"] * 100)
        hhmm = datetime.now(timezone.utc).strftime("%H:%M")
        gps_str = f"{lat:.5f},{lon:.5f}" if isinstance(lat, float) else "no fix"
        sms = (
            f"ALERTRECK | {alert['threat_type']} | "
            f"Conf: {conf}% | GPS: {gps_str} | {hhmm}"
        )

        if not _SERIAL_AVAILABLE:
            print(f"GSM: pyserial not installed — would send: {sms}")
            return False

        success = False
        for number in RANGER_PHONE_NUMBERS:
            for attempt in range(1, SMS_MAX_RETRIES + 1):
                try:
                    if _sim808_send_sms(number, sms, SIM808_PORT,
                                        SIM808_BAUDRATE, SIM808_TIMEOUT):
                        print(f"GSM: SMS sent to {number} ({attempt} attempt(s))")
                        success = True
                        break
                except Exception as exc:
                    print(f"GSM: attempt {attempt} failed — {exc}")
                if attempt < SMS_MAX_RETRIES:
                    time.sleep(SMS_RETRY_DELAY)
            else:
                print(f"GSM: all {SMS_MAX_RETRIES} attempts failed for {number}")
        return success
    
    def _notify_satellite(self, alert: Dict[str, Any]) -> bool:
        """
        Send alert via satellite communication.
        
        STUB IMPLEMENTATION - To be completed with actual satellite modem.
        
        For real implementation:
        - Use satellite modem (e.g., Iridium, Globalstar)
        - Send short burst data (SBD)
        - Optimize payload size (very limited bandwidth)
        - Handle retries and timeouts
        """
        print(f"🛰️  [Satellite STUB] Would send alert {alert['alert_id']}")
        
        # Simulate success for testing
        return True
    
    def get_stats(self) -> Dict:
        """Get notifier statistics."""
        return {
            'alerts_sent': self.alerts_sent,
            'console_enabled': NOTIFY_CONSOLE,
            'lora_enabled': NOTIFY_LORA,
            'gsm_enabled': NOTIFY_GSM,
            'satellite_enabled': NOTIFY_SATELLITE
        }


def test_notifier():
    """Test the alert notifier."""
    print("\n🔔 Testing AlertNotifier...")
    print("=" * 60)
    
    # Create notifier
    notifier = AlertNotifier()
    
    # Create test alert
    threat_info = {
        'threat_type': 'THREAT',
        'threat_level': 'HIGH',
        'confidence': 0.92,
        'class_probabilities': {
            'BACKGROUND': 0.03,
            'THREAT_CONTEXT': 0.05,
            'THREAT': 0.92
        }
    }
    
    location = {
        'latitude': -1.373333,
        'longitude': 36.857782,
        'altitude': 1700.0,
        'fix_quality': 1,
        'satellites': 8
    }
    
    audio_path = "/data/evidence/2024-12-29/gunshot_20241229_143022.wav"
    
    # Create alert
    alert = notifier.create_alert(threat_info, location, audio_path)
    
    print("\n📋 Created Alert:")
    print(json.dumps(alert, indent=2))
    
    # Send alert
    print("\n📤 Sending alert...")
    success = notifier.send_alert(alert)
    
    print(f"\nAlert sent: {success}")
    print(f"Notified channels: {alert['notified_channels']}")
    
    # Show stats
    print("\n📊 Notifier Statistics:")
    stats = notifier.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")


if __name__ == "__main__":
    test_notifier()
