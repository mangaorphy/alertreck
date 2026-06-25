"""
Top up every dataset class to a target count (default 2000) from AudioSet
(unbalanced_train_segments.csv), downloading directly into dataset/<class>/.

For each class it:
  • counts audio files already on disk,
  • skips the class if it already meets the target,
  • downloads from a curated set of AudioSet labels until the target is reached,
  • dedups against prior downloads by YouTube ID (any historical filename),
  • samples across the class's sub-labels to keep the mix diverse.

Curated labels reuse the IDs from the per-class scripts in this folder, plus
new dog/vehicle sets pulled from META_DATA_CSV/class_labels_indices.csv.

NOTE: AudioSet has only ~1,667 chainsaw clips total, so threat_chainsaw cannot
reach 2,000 — it will grab as many as remain and stop.

Usage:
  python3 scripts/audio_extraction/download_topup_2000.py                  # all classes -> 2000
  python3 scripts/audio_extraction/download_topup_2000.py --target 2000
  python3 scripts/audio_extraction/download_topup_2000.py --classes threat_dog threat_vehicle
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
DATASET   = REPO_ROOT / "dataset"
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

# class -> {AudioSet label id : short tag used in the filename}
CLASS_LABELS: dict[str, dict[str, str]] = {
    "background_wind_rain": {
        "/m/03m9d0z": "wind", "/m/06mb1": "rain", "/m/07r10fb": "raindrop",
    },
    "threat_chainsaw": {
        "/m/01j4z9": "chainsaw",
    },
    "threat_dog": {
        "/m/05tny_": "bark", "/m/07rc7d9": "bowwow", "/m/07r_k2n": "yip",
        "/m/0ghcn6": "growl", "/t/dd00136": "whimperdog", "/m/07qf0zm": "howl",
        "/m/01z5f": "canidae",
    },
    "threat_human": {
        "/m/09x0r": "speech", "/m/05zppz": "malespeech", "/m/02zsn": "femalespeech",
        "/m/01h8n0": "conversation", "/m/02qldy": "narration", "/m/07p6fty": "shout",
        "/m/07q4ntr": "bellow", "/m/07rwj3x": "whoop",
    },
    "threat_vehicle": {
        "/m/07yv9": "vehicle", "/m/012f08": "motorveh", "/m/0k4j": "car",
        "/m/07r04": "truck", "/m/04_sv": "motorcycle", "/m/0btp2": "traffic",
        "/m/02mk9": "engine", "/t/dd00066": "medengine", "/t/dd00067": "heavyengine",
        "/t/dd00130": "enginestart",
    },
}


def parse_csv(label_map: dict[str, str]) -> list[dict]:
    """Rows whose labels match any id in label_map, tagged with the first match."""
    rows = []
    with CSV_PATH.open() as f:
        for line in f:
            if line.startswith("#") or line.startswith("---"):
                continue
            parts = [p.strip().strip('"') for p in line.strip().split(",", 3)]
            if len(parts) < 4:
                continue
            ytid, start, end, labels = parts
            tag = next((t for lid, t in label_map.items() if lid in labels), None)
            if tag:
                rows.append({"ytid": ytid, "start": float(start),
                             "end": float(end), "tag": tag})
    return rows


def existing_ytids(out_dir: Path) -> set[str]:
    """All filenames already present, so we can skip any clip we've fetched before."""
    if not out_dir.exists():
        return set()
    return {f.name for f in out_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS}


def count_audio(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    return sum(1 for f in out_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS)


def topup_class(cls: str, target: int) -> None:
    label_map = CLASS_LABELS[cls]
    out_dir   = DATASET / cls
    out_dir.mkdir(parents=True, exist_ok=True)

    have = count_audio(out_dir)
    print(f"\n=== {cls} ===  on disk: {have}  target: {target}")
    if have >= target:
        print("  already at target — skipping")
        return

    rows = parse_csv(label_map)
    random.seed(SEED)
    random.shuffle(rows)

    names = existing_ytids(out_dir)            # filenames already present
    seen  = "\n".join(names)                   # cheap substring membership for old names

    need = target - have
    got, attempted = 0, 0
    per_tag = defaultdict(int)
    print(f"  candidates: {len(rows)}   need ~{need} more")

    for row in rows:
        if have + got >= target:
            break
        ytid, start, end, tag = row["ytid"], row["start"], row["end"], row["tag"]

        out = out_dir / f"audioset_{tag}__{ytid}.wav"
        if out.name in names or ytid in seen:   # already have this clip
            continue

        cmd = [
            "yt-dlp", "-x", "--audio-format", "wav",
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "-o", str(out), "--quiet", "--no-warnings",
            f"https://www.youtube.com/watch?v={ytid}",
        ]
        attempted += 1
        try:
            subprocess.run(cmd, check=True, timeout=60)
            got += 1
            per_tag[tag] += 1
            print(f"  [{have + got}/{target}] ok ({tag}): {ytid}")
            time.sleep(1)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

    print(f"  done {cls}: +{got} new (now {have + got}); attempted {attempted}; mix {dict(per_tag)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",  type=int, default=2000)
    ap.add_argument("--classes", nargs="*", default=list(CLASS_LABELS),
                    choices=list(CLASS_LABELS))
    args = ap.parse_args()

    print(f"Top-up target: {args.target} per class")
    print(f"Classes: {args.classes}")
    for cls in args.classes:
        topup_class(cls, args.target)
    print("\nAll done.")


if __name__ == "__main__":
    main()
