#!/usr/bin/env python3
import shutil
from pathlib import Path

BASE = Path('/Users/cococe/Downloads/Master Tracks')
TARGET = BASE / 'All Audio'
TARGET.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.aiff', '.m4a', '.aac', '.ogg', '.wma'}

# Collect audio files, excluding the target directory
files = []
for p in BASE.rglob('*'):
    if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.is_relative_to(TARGET):
        files.append(p)

# Sort deterministically (by path) for predictable selection
files.sort(key=lambda x: str(x))

copied = 0
skipped = 0
for src in files:
    if copied >= 800:
        break
    try:
        parent = src.parent.name
        base = src.name
        dest = TARGET / f"{parent} - {base}"
        if dest.exists():
            # If a file with the same target name already exists, skip to avoid duplicates on re-runs
            continue
        shutil.copy2(src, dest)
        copied += 1
    except Exception:
        skipped += 1

final_count = sum(1 for _ in TARGET.glob('*'))
print(f"Copied {copied} files. Skipped {skipped}. Total in target: {final_count}")
print(f"Target: {TARGET}")
