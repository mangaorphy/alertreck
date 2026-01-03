"""
Audio Recorder Module
=====================
Handles continuous audio capture from microphone using a rolling buffer.
Fault-tolerant with automatic reconnection on device failures.
"""

import numpy as np
import sounddevice as sd
import time
import threading
from collections import deque
from typing import Optional, Callable

from ..config import (
    SAMPLE_RATE, BUFFER_SIZE, CHANNELS, CHUNK_SIZE,
    MICROPHONE_RECONNECT_DELAY, DEBUG_MODE
)


class AudioRecorder:
    """
    Continuous audio recorder with rolling buffer.
    Captures audio in chunks and maintains a fixed-size buffer.
    """
    
    def __init__(self, device_id: Optional[int] = None):
        """
        Initialize audio recorder.
        
        Args:
            device_id: Specific microphone device ID (None for default)
        """
        self.device_id = device_id
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.chunk_size = CHUNK_SIZE
        self.buffer_size = BUFFER_SIZE
        
        # Rolling buffer to store audio samples
        self.buffer = deque(maxlen=self.buffer_size)
        
        # Thread control
        self.is_recording = False
        self.record_thread = None
        self.lock = threading.Lock()
        
        # Stream object
        self.stream = None
        
        # Statistics
        self.total_chunks = 0
        self.buffer_overflows = 0
        
        print(f"AudioRecorder initialized: {SAMPLE_RATE}Hz, {CHANNELS}ch, buffer={BUFFER_SIZE} samples")
    
    def start(self):
        """Start continuous audio recording in background thread."""
        if self.is_recording:
            print("⚠️  Recording already in progress")
            return
        
        self.is_recording = True
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()
        print("✅ Audio recording started")
    
    def stop(self):
        """Stop audio recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        print("🛑 Audio recording stopped")
    
    def _record_loop(self):
        """Main recording loop - runs in background thread."""
        while self.is_recording:
            try:
                self._open_stream()
                self._capture_audio()
            except Exception as e:
                print(f"❌ Microphone error: {e}")
                print(f"Reconnecting in {MICROPHONE_RECONNECT_DELAY}s...")
                time.sleep(MICROPHONE_RECONNECT_DELAY)
            finally:
                if self.stream:
                    try:
                        self.stream.stop()
                        self.stream.close()
                    except:
                        pass
                    self.stream = None
    
    def _open_stream(self):
        """Open audio stream."""
        if self.stream:
            return
        
        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype='float32'
            )
            self.stream.start()
            
            if DEBUG_MODE:
                print(f"📡 Stream opened: device={self.device_id}")
        
        except Exception as e:
            print(f"❌ Failed to open audio stream: {e}")
            raise
    
    def _capture_audio(self):
        """Capture audio chunks and add to buffer."""
        while self.is_recording and self.stream:
            try:
                # Read audio chunk
                audio_chunk, overflowed = self.stream.read(self.chunk_size)
                
                if overflowed:
                    self.buffer_overflows += 1
                    if DEBUG_MODE:
                        print(f"⚠️  Buffer overflow detected (total: {self.buffer_overflows})")
                
                # Convert to mono if needed
                if audio_chunk.shape[1] > 1:
                    audio_chunk = np.mean(audio_chunk, axis=1)
                else:
                    audio_chunk = audio_chunk.flatten()
                
                # Add to rolling buffer
                with self.lock:
                    self.buffer.extend(audio_chunk)
                
                self.total_chunks += 1
                
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Error reading audio chunk: {e}")
                raise
    
    def get_audio_buffer(self) -> Optional[np.ndarray]:
        """
        Get current audio buffer as numpy array.
        
        Returns:
            Audio samples as float32 array, or None if buffer not full
        """
        with self.lock:
            if len(self.buffer) < self.buffer_size:
                return None
            
            # Convert deque to numpy array
            audio = np.array(self.buffer, dtype=np.float32)
            return audio
    
    def is_buffer_ready(self) -> bool:
        """Check if buffer contains enough samples."""
        with self.lock:
            return len(self.buffer) >= self.buffer_size
    
    def get_stats(self) -> dict:
        """Get recorder statistics."""
        with self.lock:
            buffer_fill = len(self.buffer) / self.buffer_size if self.buffer_size > 0 else 0
            
            return {
                'is_recording': self.is_recording,
                'total_chunks': self.total_chunks,
                'buffer_fill': buffer_fill,
                'buffer_overflows': self.buffer_overflows,
                'device_id': self.device_id,
                'sample_rate': self.sample_rate
            }
    
    def clear_buffer(self):
        """Clear the audio buffer."""
        with self.lock:
            self.buffer.clear()
    
    @staticmethod
    def list_devices():
        """List available audio input devices."""
        print("\n📱 Available Audio Devices:")
        print("=" * 60)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"[{i}] {device['name']}")
                print(f"    Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}Hz")
        print("=" * 60)


def test_recorder(duration: float = 10.0):
    """
    Test the audio recorder.
    
    Args:
        duration: How long to test (seconds)
    """
    print(f"\n🎤 Testing AudioRecorder for {duration} seconds...")
    
    # List devices
    AudioRecorder.list_devices()
    
    # Create recorder
    recorder = AudioRecorder()
    
    try:
        # Start recording
        recorder.start()
        
        # Wait for buffer to fill
        while not recorder.is_buffer_ready():
            time.sleep(0.1)
            print(".", end="", flush=True)
        
        print("\n✅ Buffer ready!")
        
        # Get some buffers
        start_time = time.time()
        buffer_count = 0
        
        while time.time() - start_time < duration:
            audio = recorder.get_audio_buffer()
            if audio is not None:
                buffer_count += 1
                print(f"Buffer {buffer_count}: shape={audio.shape}, "
                      f"min={audio.min():.3f}, max={audio.max():.3f}, "
                      f"mean={audio.mean():.3f}")
            time.sleep(1.0)
        
        # Show stats
        stats = recorder.get_stats()
        print("\n📊 Recorder Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    finally:
        recorder.stop()
        print("\n✅ Test complete!")


if __name__ == "__main__":
    test_recorder(duration=15.0)
