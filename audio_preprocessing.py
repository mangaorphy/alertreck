"""
Audio Preprocessing Pipeline for Threat Detection Model
========================================================
This module handles preprocessing of audio files for training a threat detection model
to identify poaching activities and notify rangers.

Features:
- Audio loading and validation
- Standardization (sample rate, duration, channels)
- Feature extraction (Mel-spectrograms, MFCCs, raw waveforms)
- Data augmentation (time/pitch shifting, noise injection)
- Train/validation/test split
- Three-tier threat level encoding
"""

import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Comprehensive audio preprocessing pipeline for threat detection.
    """
    
    def __init__(
        self,
        data_dir: str,
        target_sr: int = 22050,
        duration: float = 10.0,
        n_mels: int = 128,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
        random_seed: int = 42
    ):
        """
        Initialize the audio preprocessor.
        
        Args:
            data_dir: Root directory containing THREAT, THREAT_CONTEXT, BACKGROUND folders
            target_sr: Target sample rate for all audio files
            duration: Target duration in seconds (clips will be padded/trimmed)
            n_mels: Number of mel bands for mel-spectrogram
            n_mfcc: Number of MFCCs to extract
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.random_seed = random_seed
        
        # Target length in samples
        self.target_length = int(target_sr * duration)
        
        # Three-tier threat categorization
        self.threat_levels = {
            'THREAT': 2,           # High priority - immediate threat
            'THREAT_CONTEXT': 1,   # Medium priority - potential threat indicator
            'BACKGROUND': 0        # Low priority - normal environmental sounds
        }
        
        # Subcategory mapping
        self.categories = {
            'THREAT': ['gunshot', 'chainsaw', 'human_voice'],
            'THREAT_CONTEXT': ['dog_bark'],
            'BACKGROUND': ['animal_sound', 'wind_rain', 'ambient_noise']
        }
        
        np.random.seed(random_seed)
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return waveform and sample rate.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio waveform, sample rate)
        """
        try:
            # Use soundfile instead of librosa.load to avoid numba dependency
            audio, sr = sf.read(file_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            return audio, self.target_sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def standardize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Standardize audio to target duration by padding or trimming.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Standardized audio of fixed length
        """
        if len(audio) > self.target_length:
            # Trim audio to target length
            audio = audio[:self.target_length]
        elif len(audio) < self.target_length:
            # Pad audio with zeros
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram from audio.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Mel-spectrogram (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmax=self.target_sr // 2
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCCs from audio.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            MFCCs (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.target_sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfcc
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract multiple features from audio.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Dictionary containing different feature representations
        """
        features = {
            'waveform': audio,
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'mfcc': self.extract_mfcc(audio)
        }
        
        return features
    
    def augment_audio(self, audio: np.ndarray, augmentation_type: str) -> np.ndarray:
        """
        Apply data augmentation to audio.
        
        Args:
            audio: Input audio waveform
            augmentation_type: Type of augmentation to apply
            
        Returns:
            Augmented audio
        """
        if augmentation_type == 'time_stretch':
            try:
                # Random time stretching (0.8x to 1.2x speed)
                rate = np.random.uniform(0.8, 1.2)
                audio = librosa.effects.time_stretch(audio, rate=rate)
                audio = self.standardize_audio(audio)
            except Exception as e:
                # Skip if numba not available
                if 'numba' in str(e):
                    pass
                else:
                    raise
            
        elif augmentation_type == 'pitch_shift':
            try:
                # Random pitch shifting (-2 to +2 semitones)
                n_steps = np.random.uniform(-2, 2)
                audio = librosa.effects.pitch_shift(
                    audio, sr=self.target_sr, n_steps=n_steps
                )
            except Exception as e:
                # Skip if numba not available
                if 'numba' in str(e):
                    pass
                else:
                    raise
            
        elif augmentation_type == 'noise':
            # Add white noise (SNR between 10-30 dB)
            noise_factor = np.random.uniform(0.001, 0.01)
            noise = np.random.randn(len(audio)) * noise_factor
            audio = audio + noise
            
        elif augmentation_type == 'time_shift':
            # Random time shifting
            shift = np.random.randint(-self.target_sr, self.target_sr)
            audio = np.roll(audio, shift)
        
        elif augmentation_type == 'environmental_mix':
            # Mix with environmental sounds (wind, rain) for realistic field conditions
            audio = self._mix_environmental_sound(audio)
            
        return audio
    
    def _mix_environmental_sound(self, audio: np.ndarray) -> np.ndarray:
        """
        Mix audio with environmental sounds (wind/rain) to simulate forest conditions.
        
        Args:
            audio: Input audio waveform (threat sound)
            
        Returns:
            Audio mixed with environmental background
        """
        # Collect environmental sound files (wind_rain category)
        env_dir = self.data_dir / 'BACKGROUND' / 'wind_rain'
        
        if not env_dir.exists():
            # If wind_rain not available, try ambient_noise
            env_dir = self.data_dir / 'BACKGROUND' / 'ambient_noise'
            if not env_dir.exists():
                return audio  # Return original if no environmental sounds available
        
        # Get list of environmental sound files
        env_files = list(env_dir.glob('*.wav'))
        
        if len(env_files) == 0:
            return audio  # No environmental sounds available
        
        # Randomly select an environmental sound
        env_file = np.random.choice(env_files)
        
        try:
            # Load environmental sound
            env_audio, sr = sf.read(str(env_file), dtype='float32')
            
            # Convert to mono if stereo
            if len(env_audio.shape) > 1:
                env_audio = np.mean(env_audio, axis=1)
            
            # Resample if needed
            if sr != self.target_sr:
                env_audio = librosa.resample(env_audio, orig_sr=sr, target_sr=self.target_sr)
            
            env_audio = self.standardize_audio(env_audio)
            env_audio = self.normalize_audio(env_audio)
            
            # Random mixing ratio (environmental sound should be quieter than threat)
            # SNR between 5-15 dB (threat sound louder than environment)
            snr_db = np.random.uniform(5, 15)
            
            # Calculate power of signals
            signal_power = np.mean(audio ** 2)
            env_power = np.mean(env_audio ** 2)
            
            # Calculate scaling factor for environmental sound
            snr_linear = 10 ** (snr_db / 10)
            scale = np.sqrt(signal_power / (snr_linear * env_power))
            
            # Mix signals
            mixed_audio = audio + (env_audio * scale)
            
            # Normalize to prevent clipping
            mixed_audio = self.normalize_audio(mixed_audio)
            
            return mixed_audio
            
        except Exception as e:
            # If mixing fails, return original audio
            return audio
    
    def process_file(
        self,
        file_path: Path,
        threat_level: str,
        subcategory: str,
        apply_augmentation: bool = False
    ) -> Optional[Dict]:
        """
        Process a single audio file and extract features.
        
        Args:
            file_path: Path to audio file
            threat_level: Threat level (THREAT, THREAT_CONTEXT, BACKGROUND)
            subcategory: Subcategory (e.g., gunshot, dog_bark, wind_rain)
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Dictionary containing processed features and labels
        """
        # Load audio
        audio, sr = self.load_audio(str(file_path))
        if audio is None:
            return None
        
        # Standardize and normalize
        audio = self.standardize_audio(audio)
        audio = self.normalize_audio(audio)
        
        # Apply augmentation if requested
        if apply_augmentation:
            # Smart augmentation selection based on threat level
            if threat_level in ['THREAT', 'THREAT_CONTEXT']:
                # Higher priority threats: prioritize environmental mixing (2x weight)
                aug_type = np.random.choice([
                    'time_stretch', 'pitch_shift', 'noise', 'time_shift',
                    'environmental_mix', 'environmental_mix', None
                ])
            else:
                # Background sounds: standard augmentation (no environmental mixing needed)
                aug_type = np.random.choice([
                    'time_stretch', 'pitch_shift', 'noise', 'time_shift', None
                ])
            
            if aug_type:
                audio = self.augment_audio(audio, aug_type)
        
        # Extract features
        features = self.extract_features(audio)
        
        # Create label
        label = {
            'threat_level': self.threat_levels[threat_level],
            'threat_level_name': threat_level,
            'subcategory': subcategory,
            'file_name': file_path.name
        }
        
        return {
            'features': features,
            'label': label
        }
    
    def collect_dataset(self) -> List[Tuple[Path, str, str]]:
        """
        Collect all audio file paths with their labels.
        
        Returns:
            List of tuples (file_path, threat_level, subcategory)
        """
        dataset = []
        
        for threat_level in self.threat_levels.keys():
            threat_dir = self.data_dir / threat_level
            
            if not threat_dir.exists():
                print(f"Warning: {threat_dir} does not exist. Skipping.")
                continue
            
            for subcategory in self.categories[threat_level]:
                subcat_dir = threat_dir / subcategory
                
                if not subcat_dir.exists():
                    print(f"Warning: {subcat_dir} does not exist. Skipping.")
                    continue
                
                # Collect all .wav files (including subdirectories for animal_sound)
                wav_files = list(subcat_dir.glob('**/*.wav'))
                
                for wav_file in wav_files:
                    dataset.append((wav_file, threat_level, subcategory))
        
        return dataset
    
    def preprocess_dataset(
        self,
        output_dir: str,
        test_size: float = 0.15,
        val_size: float = 0.15,
        apply_augmentation: bool = True,
        augmentation_factor: int = 2
    ):
        """
        Preprocess entire dataset and save to disk.
        
        Args:
            output_dir: Directory to save preprocessed data
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            apply_augmentation: Whether to apply data augmentation
            augmentation_factor: How many augmented versions to create per sample
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Collecting dataset...")
        dataset = self.collect_dataset()
        print(f"Found {len(dataset)} audio files")
        
        # Display dataset statistics
        print("\nDataset Statistics:")
        for threat_level in self.threat_levels.keys():
            count = sum(1 for _, tl, _ in dataset if tl == threat_level)
            print(f"  {threat_level}: {count} files")
            for subcategory in self.categories[threat_level]:
                subcat_count = sum(
                    1 for _, tl, sc in dataset if tl == threat_level and sc == subcategory
                )
                print(f"    - {subcategory}: {subcat_count} files")
        
        # Split dataset into train/val/test
        print("\nSplitting dataset...")
        train_val, test = train_test_split(
            dataset, test_size=test_size, random_state=self.random_seed, shuffle=True
        )
        train, val = train_test_split(
            train_val, test_size=val_size, random_state=self.random_seed, shuffle=True
        )
        
        print(f"Train: {len(train)} files")
        print(f"Validation: {len(val)} files")
        print(f"Test: {len(test)} files")
        
        # Process each split
        splits = {
            'train': train,
            'val': val,
            'test': test
        }
        
        for split_name, split_data in splits.items():
            print(f"\nProcessing {split_name} split...")
            
            processed_data = []
            
            # Process original files
            for file_path, threat_level, subcategory in tqdm(split_data, desc=f"{split_name} (original)"):
                result = self.process_file(file_path, threat_level, subcategory, apply_augmentation=False)
                if result:
                    processed_data.append(result)
            
            # Apply augmentation only to training data
            if split_name == 'train' and apply_augmentation:
                print(f"Applying augmentation (factor: {augmentation_factor})...")
                for _ in range(augmentation_factor):
                    for file_path, threat_level, subcategory in tqdm(split_data, desc=f"{split_name} (augmented)"):
                        result = self.process_file(file_path, threat_level, subcategory, apply_augmentation=True)
                        if result:
                            processed_data.append(result)
            
            # Save processed data
            split_output = output_path / f"{split_name}_data.pkl"
            with open(split_output, 'wb') as f:
                pickle.dump(processed_data, f)
            
            print(f"Saved {len(processed_data)} samples to {split_output}")
        
        # Save preprocessing configuration
        config = {
            'target_sr': self.target_sr,
            'duration': self.duration,
            'target_length': self.target_length,
            'n_mels': self.n_mels,
            'n_mfcc': self.n_mfcc,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'threat_levels': self.threat_levels,
            'categories': self.categories,
            'dataset_stats': {
                'total_files': len(dataset),
                'train_size': len(train) * (1 + augmentation_factor if apply_augmentation else 1),
                'val_size': len(val),
                'test_size': len(test)
            }
        }
        
        config_path = output_path / 'preprocessing_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nPreprocessing complete! Configuration saved to {config_path}")
        print(f"Preprocessed data saved to {output_path}")


def main():
    """
    Main function to run preprocessing pipeline.
    """
    # Configuration
    DATA_DIR = "/Users/cococe/Desktop/AUDIOSET METADATA"
    OUTPUT_DIR = "/Users/cococe/Desktop/AUDIOSET METADATA/preprocessed_data"
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        data_dir=DATA_DIR,
        target_sr=22050,      # Standard audio sample rate
        duration=10.0,         # 10-second clips
        n_mels=128,           # 128 mel bands for detailed frequency representation
        n_mfcc=40,            # 40 MFCCs for compact representation
        n_fft=2048,           # FFT window size
        hop_length=512,       # Hop length for STFT
        random_seed=42        # For reproducibility
    )
    
    # Run preprocessing
    preprocessor.preprocess_dataset(
        output_dir=OUTPUT_DIR,
        test_size=0.15,           # 15% for testing
        val_size=0.15,            # 15% of training for validation
        apply_augmentation=True,  # Apply data augmentation
        augmentation_factor=2     # Create 2x augmented versions per training sample
    )
    
    print("\n✅ Audio preprocessing completed successfully!")
    print(f"📁 Preprocessed data ready at: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Review the preprocessing_config.json file")
    print("2. Load the preprocessed data for model training")
    print("3. Build and train your threat detection model")


if __name__ == "__main__":
    main()
