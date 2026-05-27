"""
Generate data/processed/manifest.csv — one row per audio file in dataset/.

Columns: path, class, source, duration_s, sample_rate, licence

Source and licence are inferred from filename prefix:
  esc50__*            → ESC-50             CC BY-NC 3.0
  ds02_bird_*         → DATASET02/Birds     unknown (local collection)
  ds02_*              → DATASET02           unknown (local collection)
  audioset_*          → AudioSet/YouTube    CC BY 4.0 (metadata); audio © uploaders
  ff1010bird__*       → ff1010bird          CC mixed (FreeSound)
  us8k__*             → UrbanSound8K        CC BY-NC
  cv__*               → Common Voice        CC0
  (anything else)     → unknown             unknown

Run from repo root:
  python3 scripts/data_manifest.py
"""

import csv
import sys
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sys.exit("soundfile not installed — run: pip install soundfile")

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
OUT_PATH    = REPO_ROOT / "data" / "processed" / "manifest.csv"

# folder name → BACKGROUND or THREAT label used in model config
CLASS_KIND = {
    "background_animals":  "BACKGROUND",
    "background_wind_rain": "BACKGROUND",
    "birds":               "BACKGROUND",
    "threat_gunshot":      "THREAT",
    "threat_chainsaw":     "THREAT",
    "threat_vehicle":      "THREAT",
    "threat_human":        "THREAT",
    "threat_dog":          "THREAT_CONTEXT",
}

PREFIX_META: list[tuple[str, str, str]] = [
    ("esc50__",         "ESC-50",           "CC BY-NC 3.0"),
    ("ds02_bird_",      "DATASET02/Birds",   "unknown"),
    ("ds02_",           "DATASET02",         "unknown"),
    ("audioset_",       "AudioSet/YouTube",  "CC BY 4.0 (metadata); audio © uploaders"),
    ("ff1010bird__",    "ff1010bird",        "CC mixed (FreeSound)"),
    ("us8k__",          "UrbanSound8K",      "CC BY-NC"),
    ("cv__",            "Common Voice",      "CC0"),
]

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def infer_source(filename: str) -> tuple[str, str]:
    for prefix, source, licence in PREFIX_META:
        if filename.startswith(prefix):
            return source, licence
    return "unknown", "unknown"


def audio_info(path: Path) -> tuple[float, int]:
    try:
        info = sf.info(str(path))
        return round(info.duration, 3), info.samplerate
    except Exception:
        return -1.0, -1


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing_classes: set[str] = set()

    for folder in sorted(DATASET_DIR.iterdir()):
        if not folder.is_dir():
            continue
        cls = folder.name
        if cls not in CLASS_KIND:
            missing_classes.add(cls)

        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in AUDIO_EXTS:
                continue
            source, licence = infer_source(f.name)
            duration, sr    = audio_info(f)
            rows.append({
                "path":       str(f.relative_to(REPO_ROOT)),
                "class":      cls,
                "kind":       CLASS_KIND.get(cls, "unknown"),
                "source":     source,
                "duration_s": duration,
                "sample_rate": sr,
                "licence":    licence,
            })

    if missing_classes:
        print(f"Warning: folders not in CLASS_KIND mapping: {missing_classes}")

    fieldnames = ["path", "class", "kind", "source", "duration_s", "sample_rate", "licence"]
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    from collections import Counter
    by_class = Counter(r["class"] for r in rows)
    print(f"\nManifest written → {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Total files: {len(rows)}\n")
    print(f"{'CLASS':<26}  {'FILES':>6}  {'DURATION (h)':>13}")
    print("-" * 50)
    for cls in sorted(by_class):
        cls_rows = [r for r in rows if r["class"] == cls]
        total_h  = sum(r["duration_s"] for r in cls_rows if r["duration_s"] > 0) / 3600
        print(f"  {cls:<24}  {by_class[cls]:>6}  {total_h:>12.2f}h")
    print("-" * 50)
    total_h = sum(r["duration_s"] for r in rows if r["duration_s"] > 0) / 3600
    print(f"  {'TOTAL':<24}  {len(rows):>6}  {total_h:>12.2f}h")


if __name__ == "__main__":
    main()
