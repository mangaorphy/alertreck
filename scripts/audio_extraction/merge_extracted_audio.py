#!/usr/bin/env python3
"""Merge extracted_audio/ into dataset/ under the 6-class taxonomy.

Sources:
    extracted_audio/clips/         -> Mozilla Common Voice (English)  -> threat_human
    extracted_audio/dog_bark/      -> UrbanSound8K class 3 (dog_bark)  -> threat_dog
    extracted_audio/engine_idling/ -> UrbanSound8K class 5 (engine)    -> threat_vehicle

Files are copied (not moved) with a source prefix so they won't collide with
ESC-50 or future extractions. Stage 2 preprocessing handles stereo->mono and
resampling to 44.1 kHz, so mp3 + stereo wav inputs are both fine.
"""
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "extracted_audio"
DST_ROOT = REPO_ROOT / "dataset"

MERGES = [
    ("clips",         "threat_human",   "cv__"),
    ("dog_bark",      "threat_dog",     "us8k__"),
    ("engine_idling", "threat_vehicle", "us8k__"),
]


def main() -> None:
    for src_name, dst_class, prefix in MERGES:
        src_dir = SRC_ROOT / src_name
        dst_dir = DST_ROOT / dst_class
        if not src_dir.exists():
            print(f"[skip] {src_dir} not found")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)

        copied = skipped = 0
        for src in sorted(src_dir.iterdir()):
            if not src.is_file() or src.name.startswith("."):
                continue
            dst = dst_dir / f"{prefix}{src.name}"
            if dst.exists():
                skipped += 1
                continue
            shutil.copy2(src, dst)
            copied += 1

        total = sum(1 for _ in dst_dir.iterdir())
        print(f"{src_name:16s} -> {dst_class:16s}  copied={copied:4d} skipped={skipped:4d}  total in class={total}")


if __name__ == "__main__":
    main()
