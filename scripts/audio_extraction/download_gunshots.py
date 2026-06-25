"""
Download diverse gunshot/gunfire clips from AudioSet (unbalanced_train_segments.csv)
directly into dataset/threat_gunshot/.

Pulls from FIVE gunfire-related AudioSet labels to widen the distribution (different
weapon types, automatic fire, distant artillery) — improving real-world generalisation,
which the same-distribution test set does not expose.

  /m/032s66  — Gunshot, gunfire
  /m/04zjc   — Machine gun
  /m/02z32qm — Fusillade
  /m/0_1c    — Artillery fire
  /m/073cg4  — Cap gun

Skips files that already exist. Mirrors scripts/audio_extraction/download_chainsaws_500.py.

Usage:
  python3 scripts/audio_extraction/download_gunshots.py            # default 500
  python3 scripts/audio_extraction/download_gunshots.py --goal 800
  python3 scripts/audio_extraction/download_gunshots.py --per-label 150   # cap per class
"""

import argparse
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

SEED      = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH  = REPO_ROOT / "META_DATA_CSV" / "unbalanced_train_segments.csv"
OUT_DIR   = REPO_ROOT / "dataset" / "threat_gunshot"

# AudioSet label -> short tag used in the filename
TARGET_LABELS = {
    "/m/032s66":  "gunshot",
    "/m/04zjc":   "machinegun",
    "/m/02z32qm": "fusillade",
    "/m/0_1c":    "artillery",
    "/m/073cg4":  "capgun",
}


def parse_csv(csv_path: Path) -> list[dict]:
    """Return rows matching any target label, tagged with the first match."""
    rows = []
    with csv_path.open() as f:
        for line in f:
            if line.startswith("#") or line.startswith("---"):
                continue
            parts = [p.strip().strip('"') for p in line.strip().split(",", 3)]
            if len(parts) < 4:
                continue
            ytid, start, end, labels = parts
            tag = next((t for lid, t in TARGET_LABELS.items() if lid in labels), None)
            if tag:
                rows.append({"ytid": ytid, "start": float(start),
                             "end": float(end), "tag": tag})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal",      type=int, default=500, help="total clips to fetch")
    ap.add_argument("--per-label", type=int, default=0,   help="cap per label (0 = no cap)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = parse_csv(CSV_PATH)
    by_tag = defaultdict(int)
    for r in rows:
        by_tag[r["tag"]] += 1
    print("Available entries per label:")
    for tag, n in sorted(by_tag.items(), key=lambda x: -x[1]):
        print(f"  {tag:12s} {n:>6}")
    print(f"Total: {len(rows)}\n")

    random.seed(SEED)
    random.shuffle(rows)

    success, attempted = 0, 0
    got_per_tag = defaultdict(int)

    for row in rows:
        if success >= args.goal:
            break
        if args.per_label and got_per_tag[row["tag"]] >= args.per_label:
            continue

        ytid, start, end, tag = row["ytid"], row["start"], row["end"], row["tag"]
        out = OUT_DIR / f"audioset_{tag}__{ytid}.wav"

        if out.exists():
            success += 1
            got_per_tag[tag] += 1
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
            got_per_tag[tag] += 1
            print(f"  [{success}/{args.goal}] ok ({tag}): {ytid}")
            time.sleep(1)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print(f"  failed: {ytid}")

    print(f"\nDone. {success} files in {OUT_DIR}")
    print(f"Per label: {dict(got_per_tag)}")
    print(f"Attempted downloads: {attempted}")


if __name__ == "__main__":
    main()
