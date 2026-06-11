"""
Evidence Storage Module
========================
Saves audio evidence to disk with organization by date and threat type.
Manages storage limits and cleanup.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

from ..config import (
    EVIDENCE_DIR, SAMPLE_RATE, AUDIO_FORMAT,
    SAVE_EVIDENCE_AUDIO, MAX_EVIDENCE_STORAGE_GB,
    ALERT_RETENTION_DAYS
)
from ..utils import (
    get_timestamp_filename, get_date_folder,
    ensure_disk_space, cleanup_old_files
)


class EvidenceManager:
    """
    Manages storage of audio evidence for detected threats.
    Organizes files by date and threat type, handles cleanup.
    """
    
    def __init__(self):
        """Initialize evidence manager."""
        self.evidence_dir = EVIDENCE_DIR
        self.sample_rate = SAMPLE_RATE
        self.audio_format = AUDIO_FORMAT
        self.save_enabled = SAVE_EVIDENCE_AUDIO
        self.max_storage_gb = MAX_EVIDENCE_STORAGE_GB
        
        # Ensure evidence directory exists
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.files_saved = 0
        self.total_bytes_saved = 0
        self.save_failures = 0
        
        print(f"EvidenceManager initialized:")
        print(f"  Evidence dir: {self.evidence_dir}")
        print(f"  Format: {self.audio_format}")
        print(f"  Save enabled: {self.save_enabled}")
        print(f"  Max storage: {self.max_storage_gb} GB")
    
    def save_audio_evidence(
        self,
        audio: np.ndarray,
        threat_type: str,
        alert_id: str
    ) -> Optional[str]:
        """
        Save audio clip as evidence.
        
        Args:
            audio: Audio samples
            threat_type: Type of threat detected
            alert_id: Unique alert identifier
            
        Returns:
            Path to saved file, or None if disabled/failed
        """
        if not self.save_enabled:
            return None
        
        try:
            # Create directory structure: evidence/YYYY-MM-DD/threat_type/
            date_folder = get_date_folder()
            threat_folder = self.evidence_dir / date_folder / threat_type
            threat_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = get_timestamp_filename()
            filename = f"{threat_type}_{timestamp}_{alert_id}.{self.audio_format}"
            filepath = threat_folder / filename
            
            # Save audio file
            sf.write(
                str(filepath),
                audio,
                self.sample_rate,
                subtype='PCM_16'  # 16-bit PCM
            )
            
            # Update statistics
            file_size = filepath.stat().st_size
            self.files_saved += 1
            self.total_bytes_saved += file_size
            
            # Check storage limits
            self._check_storage_limits()
            
            print(f"💾 Evidence saved: {filepath.name} ({file_size / 1024:.1f} KB)")
            
            return str(filepath)
        
        except Exception as e:
            print(f"❌ Failed to save evidence: {e}")
            self.save_failures += 1
            return None
    
    def save_event_record(
        self,
        mel_spec: np.ndarray,
        threat_info: dict,
        location: dict,
        audio_path: Optional[str],
        alert_id: str,
    ) -> Optional[str]:
        """
        Save the model input (mel spectrogram) and a metadata JSON alongside the
        audio evidence, for the Grad-CAM explainability dashboard.

        Writes two sidecar files next to the WAV:
          <base>.mel.npy  — the exact (128, 301) mel the model classified
          <base>.json     — class, confidence, per-class probabilities, location, time

        Returns the JSON path, or None if disabled / failed.
        """
        if not self.save_enabled or audio_path is None:
            return None

        try:
            import json
            audio_p = Path(audio_path)
            folder  = audio_p.parent
            stem    = audio_p.stem

            mel = np.asarray(mel_spec, dtype=np.float32)
            if mel.ndim == 3:          # (1, 128, 301) → (128, 301)
                mel = mel[0]
            mel_path = folder / f"{stem}.mel.npy"
            np.save(mel_path, mel)

            record = {
                "alert_id":            alert_id,
                "threat_type":         threat_info.get("threat_type"),
                "threat_level":        threat_info.get("threat_level"),
                "confidence":          threat_info.get("confidence"),
                "class_probabilities": threat_info.get("class_probabilities", {}),
                "timestamp":           threat_info.get("timestamp"),
                "location":            location or {},
                "audio_file":          audio_p.name,
                "mel_file":            mel_path.name,
            }
            json_path = folder / f"{stem}.json"
            json_path.write_text(json.dumps(record, indent=2))

            return str(json_path)

        except Exception as e:
            print(f"❌ Failed to save event record: {e}")
            return None

    def _check_storage_limits(self):
        """Check and enforce storage limits."""
        try:
            # Ensure disk space
            if not ensure_disk_space(self.evidence_dir, self.max_storage_gb):
                print(f"⚠️  Evidence storage cleanup performed")
            
        except Exception as e:
            print(f"❌ Error checking storage limits: {e}")
    
    def cleanup_old_evidence(self, days: int = None) -> int:
        """
        Remove evidence files older than specified days.
        
        Args:
            days: Age threshold (uses config default if None)
            
        Returns:
            Number of files deleted
        """
        days = days or ALERT_RETENTION_DAYS
        
        try:
            deleted = cleanup_old_files(self.evidence_dir, days)
            
            if deleted > 0:
                print(f"🗑️  Cleaned up {deleted} old evidence files (>{days} days)")
            
            return deleted
        
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return 0
    
    def get_storage_stats(self) -> dict:
        """Get evidence storage statistics."""
        from ..utils import get_directory_size, bytes_to_gb
        
        total_size_bytes = get_directory_size(self.evidence_dir)
        total_size_gb = bytes_to_gb(total_size_bytes)
        usage_percent = (total_size_gb / self.max_storage_gb) * 100 if self.max_storage_gb > 0 else 0
        
        return {
            'files_saved': self.files_saved,
            'total_size_gb': total_size_gb,
            'max_storage_gb': self.max_storage_gb,
            'usage_percent': usage_percent,
            'save_failures': self.save_failures,
            'evidence_dir': str(self.evidence_dir)
        }
    
    def list_evidence_files(self, threat_type: str = None, date: str = None) -> list:
        """
        List evidence files filtered by threat type and/or date.
        
        Args:
            threat_type: Filter by threat type (None = all)
            date: Filter by date (YYYY-MM-DD format, None = all)
            
        Returns:
            List of file paths
        """
        files = []
        
        try:
            if date:
                search_dir = self.evidence_dir / date
                if not search_dir.exists():
                    return []
            else:
                search_dir = self.evidence_dir
            
            if threat_type:
                pattern = f"**/{threat_type}/*.{self.audio_format}"
            else:
                pattern = f"**/*.{self.audio_format}"
            
            files = list(search_dir.glob(pattern))
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        except Exception as e:
            print(f"❌ Error listing evidence files: {e}")
        
        return [str(f) for f in files]


def test_evidence_manager():
    """Test the evidence manager."""
    print("\n💾 Testing EvidenceManager...")
    print("=" * 60)
    
    # Create evidence manager
    manager = EvidenceManager()
    
    # Generate test audio (5 seconds of white noise)
    duration = 5.0
    audio = np.random.randn(int(SAMPLE_RATE * duration)).astype(np.float32)
    
    print(f"\nTest audio: {audio.shape} samples, {duration}s")
    
    # Save evidence
    print("\n💾 Saving evidence...")
    filepath = manager.save_audio_evidence(
        audio=audio,
        threat_type="THREAT",
        alert_id="test_alert_001"
    )
    
    if filepath:
        print(f"✅ Saved to: {filepath}")
    else:
        print("❌ Save failed")
    
    # Save another one
    filepath2 = manager.save_audio_evidence(
        audio=audio * 0.5,
        threat_type="THREAT_CONTEXT",
        alert_id="test_alert_002"
    )
    
    # Get storage stats
    print("\n📊 Storage Statistics:")
    stats = manager.get_storage_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # List evidence files
    print("\n📁 Evidence Files:")
    files = manager.list_evidence_files()
    for f in files:
        print(f"  {f}")
    
    # Test cleanup (files from today won't be deleted)
    print("\n🗑️  Testing cleanup (0 days - won't delete today's files)...")
    deleted = manager.cleanup_old_evidence(days=0)
    print(f"  Deleted: {deleted} files")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")


if __name__ == "__main__":
    test_evidence_manager()
