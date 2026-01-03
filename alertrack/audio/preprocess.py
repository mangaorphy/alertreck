"""
Audio Preprocessing Module
===========================
Converts raw audio to mel spectrograms matching training preprocessing.
CRITICAL: Parameters must exactly match those used during model training.
"""

import numpy as np
import librosa
from typing import Optional

from ..config import (
    SAMPLE_RATE, MEL_BANDS, FFT_SIZE, HOP_LENGTH,
    WINDOW_TYPE, FMIN, FMAX, INPUT_SHAPE, DEBUG_MODE
)


class AudioPreprocessor:
    """
    Preprocesses audio for model inference.
    Converts raw waveform to mel spectrogram using same parameters as training.
    """
    
    def __init__(self):
        """Initialize preprocessor with configuration from training."""
        self.sample_rate = SAMPLE_RATE
        self.n_mels = MEL_BANDS
        self.n_fft = FFT_SIZE
        self.hop_length = HOP_LENGTH
        self.window = WINDOW_TYPE
        self.fmin = FMIN
        self.fmax = FMAX
        self.input_shape = INPUT_SHAPE
        
        print(f"AudioPreprocessor initialized:")
        print(f"  Sample Rate: {self.sample_rate} Hz")
        print(f"  Mel Bands: {self.n_mels}")
        print(f"  FFT Size: {self.n_fft}")
        print(f"  Hop Length: {self.hop_length}")
        print(f"  Target Shape: {self.input_shape}")
    
    def preprocess(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert raw audio to mel spectrogram.
        
        Args:
            audio: Raw audio samples (float32, mono)
            
        Returns:
            Mel spectrogram ready for model input, or None on error
        """
        try:
            # Ensure audio is 1D and float32
            if len(audio.shape) > 1:
                audio = audio.flatten()
            audio = audio.astype(np.float32)
            
            # Normalize audio to [-1, 1]
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                fmin=self.fmin,
                fmax=self.fmax,
                power=2.0  # Power spectrogram
            )
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1]
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            # Adjust to target shape if needed
            mel_spec_resized = self._resize_to_target(mel_spec_norm)
            
            # Add channel dimension if needed (for CNN models)
            # Shape should be (height, width, channels) or (channels, height, width)
            if len(self.input_shape) == 3:
                # Assuming (height, width, channels) format
                if mel_spec_resized.shape != self.input_shape[:2]:
                    if DEBUG_MODE:
                        print(f"⚠️  Shape mismatch: {mel_spec_resized.shape} vs {self.input_shape[:2]}")
                
                # Add channel dimension
                mel_spec_resized = np.expand_dims(mel_spec_resized, axis=-1)
            
            if DEBUG_MODE:
                print(f"Preprocessing: audio={audio.shape} → mel_spec={mel_spec_resized.shape}")
            
            return mel_spec_resized.astype(np.float32)
        
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None
    
    def _resize_to_target(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Resize mel spectrogram to match model input shape.
        
        Args:
            mel_spec: Mel spectrogram (mels, time)
            
        Returns:
            Resized spectrogram
        """
        target_height, target_width = self.input_shape[0], self.input_shape[1]
        current_height, current_width = mel_spec.shape
        
        # Pad or crop height (mel bands)
        if current_height < target_height:
            pad_height = target_height - current_height
            mel_spec = np.pad(mel_spec, ((0, pad_height), (0, 0)), mode='constant')
        elif current_height > target_height:
            mel_spec = mel_spec[:target_height, :]
        
        # Pad or crop width (time steps)
        if current_width < target_width:
            pad_width = target_width - current_width
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        elif current_width > target_width:
            mel_spec = mel_spec[:, :target_width]
        
        return mel_spec
    
    def preprocess_batch(self, audio_list: list) -> np.ndarray:
        """
        Preprocess multiple audio samples.
        
        Args:
            audio_list: List of audio arrays
            
        Returns:
            Batch of preprocessed spectrograms (batch_size, height, width, channels)
        """
        spectrograms = []
        
        for audio in audio_list:
            spec = self.preprocess(audio)
            if spec is not None:
                spectrograms.append(spec)
        
        if not spectrograms:
            return np.array([])
        
        return np.array(spectrograms, dtype=np.float32)
    
    def get_expected_audio_length(self) -> int:
        """Get expected audio length in samples."""
        # Calculate from target width and hop length
        expected_samples = (self.input_shape[1] - 1) * self.hop_length + self.n_fft
        return expected_samples


def test_preprocessor():
    """Test the audio preprocessor."""
    print("\n🔧 Testing AudioPreprocessor...")
    print("=" * 60)
    
    # Create preprocessor
    preprocessor = AudioPreprocessor()
    
    # Generate test audio (10 seconds of white noise)
    duration = 10.0
    audio = np.random.randn(int(SAMPLE_RATE * duration)).astype(np.float32)
    
    print(f"\nTest audio: shape={audio.shape}, duration={duration}s")
    
    # Preprocess
    import time
    start = time.time()
    mel_spec = preprocessor.preprocess(audio)
    elapsed = time.time() - start
    
    if mel_spec is not None:
        print(f"\n✅ Preprocessing successful!")
        print(f"  Input shape: {audio.shape}")
        print(f"  Output shape: {mel_spec.shape}")
        print(f"  Expected shape: {INPUT_SHAPE}")
        print(f"  Processing time: {elapsed*1000:.2f}ms")
        print(f"  Min value: {mel_spec.min():.4f}")
        print(f"  Max value: {mel_spec.max():.4f}")
        print(f"  Mean value: {mel_spec.mean():.4f}")
        
        # Check shape matches
        if mel_spec.shape == INPUT_SHAPE:
            print("\n✅ Shape matches expected input!")
        else:
            print(f"\n⚠️  Shape mismatch!")
            print(f"  Expected: {INPUT_SHAPE}")
            print(f"  Got: {mel_spec.shape}")
    else:
        print("\n❌ Preprocessing failed!")
    
    # Test batch processing
    print("\n🔧 Testing batch processing...")
    audio_batch = [audio, audio * 0.5, audio * 0.25]
    batch = preprocessor.preprocess_batch(audio_batch)
    print(f"  Batch shape: {batch.shape}")
    print(f"  Expected: (3, {INPUT_SHAPE[0]}, {INPUT_SHAPE[1]}, {INPUT_SHAPE[2]})")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")


if __name__ == "__main__":
    test_preprocessor()
