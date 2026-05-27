"""
Download 500 chainsaw clips from AudioSet (unbalanced_train_segments.csv)
directly into dataset/threat_chainsaw/.

Label: Chainsaw = /m/01j4z9  (1,667 entries available)
Skips files that already exist in the output folder.
"""

import csv
import random
import subprocess
import time
from pathlib import Path

SEED        = 42
GOAL        = 500
TARGET_ID   = "/m/01j4z9"
REPO_ROOT   = Path(__file__).resolve().parents[2]
CSV_PATH    = REPO_ROOT / "META_DATA_CSV" / "unbalanced_train_segments.csv"
OUT_DIR     = REPO_ROOT / "dataset" / "threat_chainsaw"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_csv(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open() as f:
        for line in f:
            if line.startswith("#") or line.startswith("---"):
                continue
            parts = [p.strip().strip('"') for p in line.strip().split(",", 3)]
            if len(parts) < 4:
                continue
            ytid, start, end, labels = parts
            if TARGET_ID in labels:
                rows.append({"ytid": ytid, "start": float(start), "end": float(end)})
    return rows


def main():
    rows = parse_csv(CSV_PATH)
    print(f"Chainsaw entries in CSV: {len(rows)}")

    random.seed(SEED)
    random.shuffle(rows)

    success = 0
    attempted = 0

    for row in rows:
        if success >= GOAL:
            break

        ytid  = row["ytid"]
        start = row["start"]
        end   = row["end"]
        out   = OUT_DIR / f"audioset_chainsaw__{ytid}.wav"

        if out.exists():
            print(f"  skip (exists): {ytid}")
            success += 1
            continue

        url = f"https://www.youtube.com/watch?v={ytid}"
        cmd = [
            "yt-dlp", "-x",
            "--audio-format", "wav",
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "-o", str(out),
            "--quiet", "--no-warnings",
            url,
        ]

        attempted += 1
        try:
            subprocess.run(cmd, check=True, timeout=60)
            success += 1
            print(f"  [{success}/{GOAL}] ok: {ytid}")
            time.sleep(1)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print(f"  failed: {ytid}")

    print(f"\nDone. Downloaded {success} files to {OUT_DIR}")
    print(f"Attempted: {attempted} | Already existed: {success - attempted}")


if __name__ == "__main__":
    main()
