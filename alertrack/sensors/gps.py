"""
GPS Module
==========
Handles GPS coordinate reading from UART/USB GPS modules.
Parses NMEA sentences and provides location data for alerts.
"""

import time
from typing import Optional, Tuple, Dict
import threading

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️  pyserial not available - GPS will be simulated")

from ..config import (
    GPS_ENABLED, GPS_PORT, GPS_BAUDRATE, GPS_TIMEOUT,
    GPS_RECONNECT_DELAY, SIMULATE_GPS, DEBUG_MODE
)


class GPSReader:
    """
    Reads GPS coordinates from NMEA-compliant GPS module.
    Supports UART and USB GPS devices.
    """
    
    def __init__(self, port: str = None, baudrate: int = None):
        """
        Initialize GPS reader.
        
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0')
            baudrate: Baud rate (typically 9600 or 4800)
        """
        self.port = port or GPS_PORT
        self.baudrate = baudrate or GPS_BAUDRATE
        self.timeout = GPS_TIMEOUT
        
        self.serial_port = None
        self.is_running = False
        self.read_thread = None
        
        # Latest GPS data
        self.lock = threading.Lock()
        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.timestamp = None
        self.fix_quality = 0
        self.num_satellites = 0
        
        # Statistics
        self.total_reads = 0
        self.successful_reads = 0
        self.failed_reads = 0
        
        if SIMULATE_GPS:
            print("🌍 GPS simulation mode enabled")
            self._simulate_gps_data()
        elif not GPS_ENABLED:
            print("🌍 GPS disabled in config")
        elif not SERIAL_AVAILABLE:
            print("🌍 pyserial not available - using simulated GPS")
            self._simulate_gps_data()
        else:
            print(f"🌍 GPS initialized: {self.port} @ {self.baudrate} baud")
    
    def start(self):
        """Start GPS reading in background thread."""
        if not GPS_ENABLED or SIMULATE_GPS or not SERIAL_AVAILABLE:
            return
        
        if self.is_running:
            print("⚠️  GPS already running")
            return
        
        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        print("✅ GPS reading started")
    
    def stop(self):
        """Stop GPS reading."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        
        if self.serial_port:
            try:
                self.serial_port.close()
            except:
                pass
            self.serial_port = None
        
        print("🛑 GPS reading stopped")
    
    def _read_loop(self):
        """Main GPS reading loop - runs in background thread."""
        while self.is_running:
            try:
                self._open_port()
                self._read_gps_data()
            except Exception as e:
                print(f"❌ GPS error: {e}")
                print(f"Reconnecting in {GPS_RECONNECT_DELAY}s...")
                time.sleep(GPS_RECONNECT_DELAY)
            finally:
                if self.serial_port:
                    try:
                        self.serial_port.close()
                    except:
                        pass
                    self.serial_port = None
    
    def _open_port(self):
        """Open serial port."""
        if self.serial_port:
            return
        
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            if DEBUG_MODE:
                print(f"📡 GPS port opened: {self.port}")
        
        except Exception as e:
            print(f"❌ Failed to open GPS port: {e}")
            raise
    
    def _read_gps_data(self):
        """Read and parse NMEA sentences."""
        while self.is_running and self.serial_port:
            try:
                # Read line from GPS
                line = self.serial_port.readline().decode('ascii', errors='ignore').strip()
                
                if not line:
                    continue
                
                self.total_reads += 1
                
                # Parse NMEA sentence
                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    # Global Positioning System Fix Data
                    self._parse_gga(line)
                elif line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                    # Recommended Minimum Specific GPS/Transit Data
                    self._parse_rmc(line)
                
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Error reading GPS data: {e}")
                self.failed_reads += 1
                continue
    
    def _parse_gga(self, sentence: str):
        """
        Parse GGA sentence: Global Positioning System Fix Data.
        Format: $GPGGA,time,lat,N/S,lon,E/W,quality,satellites,hdop,altitude,M,...
        """
        try:
            parts = sentence.split(',')
            
            if len(parts) < 10:
                return
            
            # Fix quality (0=invalid, 1=GPS, 2=DGPS)
            fix_quality = int(parts[6]) if parts[6] else 0
            
            if fix_quality == 0:
                return  # No valid fix
            
            # Latitude
            lat_str = parts[2]
            lat_dir = parts[3]
            if lat_str and lat_dir:
                lat = self._convert_to_degrees(lat_str, lat_dir)
            else:
                return
            
            # Longitude
            lon_str = parts[4]
            lon_dir = parts[5]
            if lon_str and lon_dir:
                lon = self._convert_to_degrees(lon_str, lon_dir)
            else:
                return
            
            # Altitude
            alt = float(parts[9]) if parts[9] else None
            
            # Number of satellites
            num_sats = int(parts[7]) if parts[7] else 0
            
            # Update stored data
            with self.lock:
                self.latitude = lat
                self.longitude = lon
                self.altitude = alt
                self.fix_quality = fix_quality
                self.num_satellites = num_sats
                self.timestamp = time.time()
            
            self.successful_reads += 1
            
            if DEBUG_MODE:
                print(f"GPS: {lat:.6f}, {lon:.6f}, alt={alt}m, sats={num_sats}")
        
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error parsing GGA: {e}")
            self.failed_reads += 1
    
    def _parse_rmc(self, sentence: str):
        """
        Parse RMC sentence: Recommended Minimum Specific GPS/Transit Data.
        Format: $GPRMC,time,status,lat,N/S,lon,E/W,speed,track,date,...
        """
        try:
            parts = sentence.split(',')
            
            if len(parts) < 8:
                return
            
            # Status (A=active, V=void)
            status = parts[2]
            if status != 'A':
                return  # Not active
            
            # Latitude
            lat_str = parts[3]
            lat_dir = parts[4]
            if lat_str and lat_dir:
                lat = self._convert_to_degrees(lat_str, lat_dir)
            else:
                return
            
            # Longitude
            lon_str = parts[5]
            lon_dir = parts[6]
            if lon_str and lon_dir:
                lon = self._convert_to_degrees(lon_str, lon_dir)
            else:
                return
            
            # Update stored data
            with self.lock:
                self.latitude = lat
                self.longitude = lon
                self.timestamp = time.time()
            
            self.successful_reads += 1
        
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error parsing RMC: {e}")
            self.failed_reads += 1
    
    @staticmethod
    def _convert_to_degrees(coord_str: str, direction: str) -> float:
        """
        Convert NMEA coordinate to decimal degrees.
        
        Args:
            coord_str: Coordinate string (e.g., '4807.038' for latitude)
            direction: Direction (N/S for lat, E/W for lon)
            
        Returns:
            Decimal degrees
        """
        # Latitude: DDMM.MMMM
        # Longitude: DDDMM.MMMM
        
        if not coord_str:
            return 0.0
        
        # Split degrees and minutes
        if len(coord_str) > 7:  # Longitude (3 digit degrees)
            degrees = float(coord_str[:3])
            minutes = float(coord_str[3:])
        else:  # Latitude (2 digit degrees)
            degrees = float(coord_str[:2])
            minutes = float(coord_str[2:])
        
        # Convert to decimal degrees
        decimal = degrees + (minutes / 60.0)
        
        # Apply direction
        if direction in ['S', 'W']:
            decimal = -decimal
        
        return decimal
    
    def get_coordinates(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get latest GPS coordinates.
        
        Returns:
            Tuple of (latitude, longitude) or (None, None) if unavailable
        """
        with self.lock:
            if self.latitude is not None and self.longitude is not None:
                # Check if data is recent (within last 30 seconds)
                if self.timestamp and (time.time() - self.timestamp) < 30:
                    return self.latitude, self.longitude
        
        return None, None
    
    def get_location_dict(self) -> Dict:
        """
        Get location data as dictionary.
        
        Returns:
            Dictionary with location info
        """
        lat, lon = self.get_coordinates()
        
        return {
            'latitude': lat if lat is not None else 'UNKNOWN',
            'longitude': lon if lon is not None else 'UNKNOWN',
            'altitude': self.altitude if self.altitude is not None else 'UNKNOWN',
            'fix_quality': self.fix_quality,
            'satellites': self.num_satellites,
            'timestamp': self.timestamp
        }
    
    def has_fix(self) -> bool:
        """Check if GPS has valid fix."""
        lat, lon = self.get_coordinates()
        return lat is not None and lon is not None
    
    def _simulate_gps_data(self):
        """Set simulated GPS data for testing."""
        with self.lock:
            # Example coordinates: Nairobi National Park
            self.latitude = -1.373333
            self.longitude = 36.857782
            self.altitude = 1700.0
            self.fix_quality = 1
            self.num_satellites = 8
            self.timestamp = time.time()
        
        print(f"🌍 Simulated GPS: {self.latitude}, {self.longitude}")
    
    def get_stats(self) -> Dict:
        """Get GPS reader statistics."""
        success_rate = 0.0
        if self.total_reads > 0:
            success_rate = self.successful_reads / self.total_reads
        
        return {
            'is_running': self.is_running,
            'has_fix': self.has_fix(),
            'total_reads': self.total_reads,
            'successful_reads': self.successful_reads,
            'failed_reads': self.failed_reads,
            'success_rate': success_rate,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'satellites': self.num_satellites
        }


if __name__ == "__main__":
    print("\n🌍 Testing GPSReader...")
    print("=" * 60)
    
    # Create GPS reader (will use simulation if no hardware)
    gps = GPSReader()
    
    try:
        # Start reading
        gps.start()
        
        # Read coordinates for a few seconds
        for i in range(5):
            time.sleep(1)
            
            lat, lon = gps.get_coordinates()
            location = gps.get_location_dict()
            
            print(f"\n[{i+1}] GPS Data:")
            print(f"  Coordinates: {lat}, {lon}")
            print(f"  Has fix: {gps.has_fix()}")
            print(f"  Location dict: {location}")
        
        # Show stats
        print("\n📊 GPS Statistics:")
        stats = gps.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    finally:
        gps.stop()
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
