"""
ALERTRACK Utilities
===================
Common utility functions used across the system.
"""

import os
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


def get_timestamp() -> str:
    """Get current timestamp in ISO 8601 format."""
    return datetime.utcnow().isoformat() + "Z"


def get_timestamp_filename() -> str:
    """Get timestamp suitable for filenames (no special characters)."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def get_date_folder() -> str:
    """Get current date for folder organization (YYYY-MM-DD)."""
    return datetime.utcnow().strftime("%Y-%m-%d")


def generate_alert_id() -> str:
    """Generate unique alert ID using timestamp and random component."""
    timestamp = str(time.time()).encode()
    random_component = os.urandom(8)
    combined = timestamp + random_component
    return hashlib.sha256(combined).hexdigest()[:16]


def save_json(data: Dict[Any, Any], filepath: Path) -> bool:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")
        return False


def load_json(filepath: Path) -> Optional[Dict[Any, Any]]:
    """
    Load JSON file.
    
    Args:
        filepath: Path to load from
        
    Returns:
        Dictionary if successful, None otherwise
    """
    try:
        if not filepath.exists():
            return None
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {e}")
        return None


def get_directory_size(directory: Path) -> int:
    """
    Get total size of directory in bytes.
    
    Args:
        directory: Directory to measure
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                if filepath.exists():
                    total_size += filepath.stat().st_size
    except Exception as e:
        print(f"Error calculating directory size: {e}")
    return total_size


def bytes_to_gb(bytes_count: int) -> float:
    """Convert bytes to gigabytes."""
    return bytes_count / (1024 ** 3)


def cleanup_old_files(directory: Path, days_old: int) -> int:
    """
    Remove files older than specified days.
    
    Args:
        directory: Directory to clean
        days_old: Remove files older than this many days
        
    Returns:
        Number of files deleted
    """
    if not directory.exists():
        return 0
    
    deleted_count = 0
    current_time = time.time()
    max_age_seconds = days_old * 24 * 60 * 60
    
    try:
        for filepath in directory.rglob('*'):
            if filepath.is_file():
                file_age = current_time - filepath.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        filepath.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"Failed to delete {filepath}: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    return deleted_count


def ensure_disk_space(directory: Path, max_gb: float) -> bool:
    """
    Ensure directory doesn't exceed size limit.
    If exceeded, delete oldest files until under limit.
    
    Args:
        directory: Directory to check
        max_gb: Maximum size in GB
        
    Returns:
        True if under limit or successfully cleaned
    """
    try:
        current_size_bytes = get_directory_size(directory)
        current_size_gb = bytes_to_gb(current_size_bytes)
        
        if current_size_gb <= max_gb:
            return True
        
        # Get all files sorted by modification time (oldest first)
        files = []
        for filepath in directory.rglob('*'):
            if filepath.is_file():
                files.append((filepath.stat().st_mtime, filepath))
        
        files.sort()  # Sort by modification time
        
        # Delete oldest files until under limit
        for mtime, filepath in files:
            try:
                file_size = filepath.stat().st_size
                filepath.unlink()
                current_size_bytes -= file_size
                current_size_gb = bytes_to_gb(current_size_bytes)
                
                if current_size_gb <= max_gb:
                    return True
            except Exception as e:
                print(f"Failed to delete {filepath}: {e}")
                continue
        
        return current_size_gb <= max_gb
    
    except Exception as e:
        print(f"Error ensuring disk space: {e}")
        return False


def retry_on_failure(func, max_attempts: int = 3, delay: float = 1.0):
    """
    Retry a function on failure.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        
    Returns:
        Result of function or None if all attempts fail
    """
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"All {max_attempts} attempts failed.")
                return None
    return None


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation", warn_threshold: Optional[float] = None):
        self.name = name
        self.warn_threshold = warn_threshold
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        
        if self.warn_threshold and self.elapsed > self.warn_threshold:
            print(f"⚠️  {self.name} took {self.elapsed:.3f}s (threshold: {self.warn_threshold}s)")
        
        return False  # Don't suppress exceptions


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage string."""
    return f"{confidence * 100:.1f}%"


def format_coords(lat: float, lon: float) -> str:
    """Format GPS coordinates as string."""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.6f}°{lat_dir}, {abs(lon):.6f}°{lon_dir}"


if __name__ == "__main__":
    print("ALERTRACK Utilities Test")
    print("=" * 60)
    print(f"Timestamp: {get_timestamp()}")
    print(f"Filename timestamp: {get_timestamp_filename()}")
    print(f"Date folder: {get_date_folder()}")
    print(f"Alert ID: {generate_alert_id()}")
    print(f"Confidence format: {format_confidence(0.8765)}")
    print(f"Coords format: {format_coords(-1.292066, 36.821945)}")
    print("=" * 60)
