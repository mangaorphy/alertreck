"""
Randomly sample 200 audio files per source from DATASET02 and copy
them into the correct dataset/ folders.

Sources → destination:
  Animals/elephant  → background_animals/
  Animals/lion      → background_animals/
  Animals/horse     → background_animals/
  Environment/rainfall → background_wind_rain/
  Environment/wind     → background_wind_rain/
  Birds/*           → background_animals/  (all species pooled, 200 total each)
"""

import random
import shutil
from pathlib import Path

SEED = 42
N = 200
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

REPO_ROOT   = Path(__file__).resolve().parents[2]
DS02        = REPO_ROOT / "DATASET02"
DATASET_DIR = REPO_ROOT / "dataset"

# (source_dir, destination_folder, label_prefix)
# For Birds we pool all sub-species then sample 200 per species.
SOURCES = [
    (DS02 / "Animals" / "elephant",    "background_animals",   "elephant"),
    (DS02 / "Animals" / "lion",        "background_animals",   "lion"),
    (DS02 / "Animals" / "horse",       "background_animals",   "horse"),
    (DS02 / "Environment" / "rainfall","background_wind_rain",  "rainfall"),
    (DS02 / "Environment" / "wind",    "background_wind_rain",  "wind"),
]

BIRD_SPECIES = ["sparrow", "crow", "peacock", "parrot"]


def audio_files(folder: Path) -> list[Path]:
    return [f for f in folder.rglob("*") if f.suffix.lower() in AUDIO_EXTS]


def sample_and_copy(files: list[Path], dest_dir: Path, prefix: str, n: int) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    chosen = random.sample(files, min(n, len(files)))
    for f in chosen:
        dst = dest_dir / f"ds02_{prefix}__{f.name}"
        if not dst.exists():
            shutil.copy2(f, dst)
    return len(chosen)


def main():
    random.seed(SEED)
    print(f"{'SOURCE':<28}  {'DEST':<22}  {'COPIED':>6}")
    print("-" * 62)

    for src_dir, dest_name, prefix in SOURCES:
        files = audio_files(src_dir)
        dest  = DATASET_DIR / dest_name
        copied = sample_and_copy(files, dest, prefix, N)
        print(f"  {prefix:<26}  {dest_name:<22}  {copied:>6}")

    # Birds: 200 per species
    for species in BIRD_SPECIES:
        src_dir = DS02 / "Birds" / species
        files   = audio_files(src_dir)
        dest    = DATASET_DIR / "background_animals"
        copied  = sample_and_copy(files, dest, f"bird_{species}", N)
        print(f"  bird_{species:<22}  {'background_animals':<22}  {copied:>6}")

    print("-" * 62)
    print("Done.")


if __name__ == "__main__":
    main()
