"""
Randomly sample 500 bird-positive clips from the ff1010bird dataset
and copy them to dataset/birds/ for manual review.
"""

import csv
import random
import shutil
from pathlib import Path

SEED = 42
N = 500
CSV_PATH = Path(__file__).parents[2] / "META_DATA_CSV" / "ff1010bird_metadata_2018.csv"
WAV_DIR = Path(__file__).parents[2] / "wav"
OUT_DIR = Path(__file__).parents[2] / "dataset" / "birds"


def main():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    bird_ids = [r["itemid"] for r in rows if r["hasbird"] == "1"]
    print(f"Bird-positive clips in CSV: {len(bird_ids)}")

    # Only keep ids whose wav file actually exists
    available = [id_ for id_ in bird_ids if (WAV_DIR / f"{id_}.wav").exists()]
    print(f"Available on disk: {len(available)}")

    random.seed(SEED)
    sampled = random.sample(available, min(N, len(available)))
    print(f"Sampling: {len(sampled)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for id_ in sampled:
        src = WAV_DIR / f"{id_}.wav"
        dst = OUT_DIR / f"{id_}.wav"
        shutil.copy2(src, dst)

    print(f"Done. {len(sampled)} files copied to {OUT_DIR}")


if __name__ == "__main__":
    main()
