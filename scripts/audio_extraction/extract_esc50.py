#!/usr/bin/env python3
"""Extract all relevant ESC-50 classes into dataset/ under the 6-class taxonomy.

Source : ESC-50 — Piczak (2015)
         https://github.com/karolpiczak/ESC-50  (CC BY-NC 3.0)
CSV    : ESC-50-master/meta/esc50.csv
Audio  : ESC-50-master/audio/*.wav  (44.1 kHz mono, 5 s per clip)

ESC-50 → Alertreck taxonomy mapping
--------------------------------------
THREAT CLASSES
  chainsaw              -> threat_chainsaw
  engine                -> threat_vehicle
  dog                   -> threat_dog
  footsteps             -> threat_human

NORMAL CLASS  (natural ambience)
  rain, sea_waves, crackling_fire, crickets, chirping_birds,
  water_drops, wind, pouring_water, thunderstorm

NORMAL CLASS  (non-threat animals)
  cat, cow, crow, frog, hen, insects, pig, rooster, sheep
"""
import csv
import shutil
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ESC50_ROOT = REPO_ROOT / "ESC-50-master"
CSV_PATH   = ESC50_ROOT / "meta" / "esc50.csv"
AUDIO_DIR  = ESC50_ROOT / "audio"
DATASET_DIR = REPO_ROOT / "dataset"

CATEGORY_TO_CLASS: dict[str, str] = {
    # ── threats ──────────────────────────────────────────────────────────
    "chainsaw":         "threat_chainsaw",
    "engine":           "threat_vehicle",
    "dog":              "threat_dog",
    "footsteps":        "threat_human",

    # ── background · weather / natural ambience ───────────────────────────
    "rain":             "background_wind_rain",
    "sea_waves":        "background_wind_rain",
    "crackling_fire":   "background_wind_rain",
    "water_drops":      "background_wind_rain",
    "wind":             "background_wind_rain",
    "pouring_water":    "background_wind_rain",
    "thunderstorm":     "background_wind_rain",

    # ── background · non-threat animals ──────────────────────────────────
    "crickets":         "background_animals",
    "chirping_birds":   "background_animals",
    "cat":              "background_animals",
    "cow":              "background_animals",
    "crow":             "background_animals",
    "frog":             "background_animals",
    "hen":              "background_animals",
    "insects":          "background_animals",
    "pig":              "background_animals",
    "rooster":          "background_animals",
    "sheep":            "background_animals",
}


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"ESC-50 metadata not found at {CSV_PATH}")
    if not AUDIO_DIR.exists():
        raise SystemExit(f"ESC-50 audio dir not found at {AUDIO_DIR}")

    counts:  defaultdict[str, int] = defaultdict(int)   # class -> copied
    by_cat:  defaultdict[str, int] = defaultdict(int)   # category -> copied
    missing = 0

    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            category    = row["category"]
            target_class = CATEGORY_TO_CLASS.get(category)
            if target_class is None:
                continue                                 # unmapped category — skip

            src = AUDIO_DIR / row["filename"]
            if not src.exists():
                missing += 1
                continue

            out_dir = DATASET_DIR / target_class
            out_dir.mkdir(parents=True, exist_ok=True)

            dest = out_dir / f"esc50__{row['filename']}"
            if not dest.exists():
                shutil.copy2(src, dest)

            counts[target_class] += 1
            by_cat[category]     += 1

    # ── summary ──────────────────────────────────────────────────────────
    print("ESC-50 extraction complete\n")
    print(f"{'CLASS':<18}  {'COUNT':>5}   CATEGORIES")
    print("-" * 60)
    for cls in sorted(counts):
        cats = [c for c, cc in CATEGORY_TO_CLASS.items() if cc == cls]
        cat_detail = ", ".join(f"{c}({by_cat[c]})" for c in sorted(cats) if by_cat[c])
        print(f"  {cls:<16}  {counts[cls]:>5}   {cat_detail}")

    total = sum(counts.values())
    print("-" * 60)
    print(f"  {'TOTAL':<16}  {total:>5}")
    if missing:
        print(f"\n  warning: {missing} rows skipped (audio file not found)")


if __name__ == "__main__":
    main()
