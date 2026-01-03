#!/usr/bin/env python3
"""
Split large train_data.pkl into smaller chunks for Colab.
Run this locally where you have enough RAM (59GB free).
"""

import pickle
import os
from pathlib import Path

# Paths
PREPROCESSED_DIR = Path("/Users/cococe/Desktop/AUDIOSET METADATA/preprocessed_data")
CHUNKS_DIR = PREPROCESSED_DIR / "train_chunks"
TRAIN_FILE = PREPROCESSED_DIR / "train_data.pkl"

# Create chunks directory
CHUNKS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("SPLITTING TRAINING DATA INTO CHUNKS")
print("=" * 80)

# Load training data
print(f"\n📥 Loading {TRAIN_FILE.name}...")
print("⏳ This may take a minute...\n")

with open(TRAIN_FILE, 'rb') as f:
    train_data = pickle.load(f)

total_samples = len(train_data)
print(f"✅ Loaded {total_samples:,} training samples")

# Split into chunks (1,600 samples each = ~2GB)
chunk_size = 1600
num_chunks = (total_samples + chunk_size - 1) // chunk_size

print(f"\n📦 Creating {num_chunks} chunks ({chunk_size} samples each)...")
print()

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, total_samples)
    chunk = train_data[start_idx:end_idx]
    
    chunk_path = CHUNKS_DIR / f"train_chunk_{i:02d}.pkl"
    
    with open(chunk_path, 'wb') as f:
        pickle.dump(chunk, f)
    
    size_mb = chunk_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Chunk {i+1:2d}/{num_chunks}: {len(chunk):,} samples → {chunk_path.name} ({size_mb:.1f} MB)")

print(f"\n✅ All chunks created in: {CHUNKS_DIR}")

# Show directory contents
print("\n" + "=" * 80)
print("CHUNK DIRECTORY CONTENTS")
print("=" * 80)
os.system(f'ls -lh "{CHUNKS_DIR}"')

print("\n" + "=" * 80)
print("NEXT STEP: Upload chunks to S3")
print("=" * 80)
print("\nRun this command:")
print(f'aws s3 sync "{CHUNKS_DIR}" s3://alertreck/preprocessed_data/train_chunks/')
