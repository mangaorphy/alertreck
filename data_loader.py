"""
Data Loader and Visualization Utilities
========================================
Helper functions to load and inspect preprocessed audio data.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
from typing import Dict, List, Tuple
import json


class ThreatDataLoader:
    """
    Load and manage preprocessed audio data for model training.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing preprocessed data files
        """
        self.data_dir = Path(data_dir)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load preprocessing configuration."""
        config_path = self.data_dir / 'preprocessing_config.json'
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_split(self, split: str = 'train') -> List[Dict]:
        """
        Load a data split (train/val/test).
        
        Args:
            split: Split name ('train', 'val', or 'test')
            
        Returns:
            List of processed samples
        """
        split_path = self.data_dir / f'{split}_data.pkl'
        with open(split_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def get_batch(
        self,
        split: str = 'train',
        batch_size: int = 32,
        feature_type: str = 'mel_spectrogram'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a random batch of data.
        
        Args:
            split: Data split to sample from
            batch_size: Number of samples in batch
            feature_type: Type of features to extract
            
        Returns:
            Tuple of (features, labels)
        """
        data = self.load_split(split)
        
        # Random sampling
        indices = np.random.choice(len(data), size=batch_size, replace=False)
        batch_data = [data[i] for i in indices]
        
        # Extract features and labels
        features = np.array([
            sample['features'][feature_type] for sample in batch_data
        ])
        labels = np.array([
            sample['label']['threat_level'] for sample in batch_data
        ])
        
        return features, labels
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'config': self.config,
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            try:
                data = self.load_split(split)
                
                # Count samples per threat level
                threat_counts = {}
                for sample in data:
                    threat = sample['label']['threat_level_name']
                    threat_counts[threat] = threat_counts.get(threat, 0) + 1
                
                # Count samples per subcategory
                subcat_counts = {}
                for sample in data:
                    subcat = sample['label']['subcategory']
                    subcat_counts[subcat] = subcat_counts.get(subcat, 0) + 1
                
                stats['splits'][split] = {
                    'total_samples': len(data),
                    'threat_distribution': threat_counts,
                    'subcategory_distribution': subcat_counts
                }
            except FileNotFoundError:
                stats['splits'][split] = {'error': 'Split not found'}
        
        return stats
    
    def visualize_sample(
        self,
        split: str = 'train',
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Visualize a sample from the dataset.
        
        Args:
            split: Data split to sample from
            sample_idx: Index of sample to visualize
            save_path: Optional path to save visualization
        """
        data = self.load_split(split)
        sample = data[sample_idx]
        
        features = sample['features']
        label = sample['label']
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot waveform
        waveform = features['waveform']
        time = np.arange(len(waveform)) / self.config['target_sr']
        axes[0].plot(time, waveform)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f"Waveform - {label['threat_level_name']} ({label['subcategory']})")
        axes[0].grid(True, alpha=0.3)
        
        # Plot mel-spectrogram
        mel_spec = features['mel_spectrogram']
        img1 = librosa.display.specshow(
            mel_spec,
            sr=self.config['target_sr'],
            hop_length=self.config['hop_length'],
            x_axis='time',
            y_axis='mel',
            ax=axes[1],
            cmap='viridis'
        )
        axes[1].set_title('Mel-Spectrogram')
        fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')
        
        # Plot MFCCs
        mfcc = features['mfcc']
        img2 = librosa.display.specshow(
            mfcc,
            sr=self.config['target_sr'],
            hop_length=self.config['hop_length'],
            x_axis='time',
            ax=axes[2],
            cmap='coolwarm'
        )
        axes[2].set_title('MFCCs')
        axes[2].set_ylabel('MFCC Coefficient')
        fig.colorbar(img2, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Print sample information
        print(f"\nSample Information:")
        print(f"  File: {label['file_name']}")
        print(f"  Threat Level: {label['threat_level_name']} (class {label['threat_level']})")
        print(f"  Subcategory: {label['subcategory']}")
        print(f"  Waveform shape: {waveform.shape}")
        print(f"  Mel-spectrogram shape: {mel_spec.shape}")
        print(f"  MFCC shape: {mfcc.shape}")


def print_dataset_summary(data_dir: str):
    """
    Print a comprehensive summary of the preprocessed dataset.
    
    Args:
        data_dir: Directory containing preprocessed data
    """
    loader = ThreatDataLoader(data_dir)
    stats = loader.get_statistics()
    
    print("=" * 80)
    print("PREPROCESSED DATASET SUMMARY")
    print("=" * 80)
    
    print("\nPreprocessing Configuration:")
    print(f"  Sample Rate: {stats['config']['target_sr']} Hz")
    print(f"  Duration: {stats['config']['duration']} seconds")
    print(f"  Target Length: {stats['config']['target_length']} samples")
    print(f"  Mel Bands: {stats['config']['n_mels']}")
    print(f"  MFCCs: {stats['config']['n_mfcc']}")
    print(f"  FFT Size: {stats['config']['n_fft']}")
    print(f"  Hop Length: {stats['config']['hop_length']}")
    
    print("\nThreat Level Mapping:")
    for level, code in stats['config']['threat_levels'].items():
        print(f"  {level}: {code}")
    
    print("\nDataset Statistics:")
    for split_name, split_stats in stats['splits'].items():
        if 'error' in split_stats:
            print(f"\n{split_name.upper()}: {split_stats['error']}")
            continue
        
        print(f"\n{split_name.upper()} Split:")
        print(f"  Total Samples: {split_stats['total_samples']}")
        
        print(f"  Threat Level Distribution:")
        for threat, count in split_stats['threat_distribution'].items():
            percentage = (count / split_stats['total_samples']) * 100
            print(f"    {threat}: {count} ({percentage:.1f}%)")
        
        print(f"  Subcategory Distribution:")
        for subcat, count in sorted(split_stats['subcategory_distribution'].items()):
            percentage = (count / split_stats['total_samples']) * 100
            print(f"    {subcat}: {count} ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)


def main():
    """
    Demonstrate data loading and visualization.
    """
    DATA_DIR = "/Users/cococe/Desktop/AUDIOSET METADATA/preprocessed_data"
    
    # Load data and print summary
    print_dataset_summary(DATA_DIR)
    
    # Create visualizations for sample from each threat level
    print("\nGenerating sample visualizations...")
    loader = ThreatDataLoader(DATA_DIR)
    
    try:
        train_data = loader.load_split('train')
        
        # Find one sample from each threat level
        threat_samples = {}
        for idx, sample in enumerate(train_data):
            threat_level = sample['label']['threat_level_name']
            if threat_level not in threat_samples:
                threat_samples[threat_level] = idx
            if len(threat_samples) == 3:
                break
        
        # Visualize samples
        viz_dir = Path(DATA_DIR) / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        for threat_level, idx in threat_samples.items():
            save_path = viz_dir / f'sample_{threat_level.lower()}.png'
            loader.visualize_sample('train', idx, str(save_path))
        
        print(f"\n✅ Visualizations saved to {viz_dir}")
        
    except Exception as e:
        print(f"\n⚠️  Could not generate visualizations: {e}")
        print("Please run audio_preprocessing.py first to create preprocessed data.")


if __name__ == "__main__":
    main()
