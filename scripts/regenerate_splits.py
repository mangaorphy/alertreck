#!/usr/bin/env python3
"""Regenerate data/processed/splits.json with leak-free, source-aware grouping.

Overwrites the existing file-level split with one where no parent recording spans
train/val/test, prints a before/after leakage report, and backs up the old split.

After running this, re-run  python scripts/audio_preprocessing.py  to re-shard
against the new split, then retrain.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grouping import group_key, group_aware_split

SPLITS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'splits.json'))
SEED = 42
SPLIT_NAMES = ('train', 'val', 'test')


def leak_report(splits, label):
    seen = defaultdict(set)
    for sp in SPLIT_NAMES:
        for cls, path in splits[sp]:
            seen[(cls, group_key(path))].add(sp)
    crossing = [g for g, s in seen.items() if len(s) > 1]
    n_files = sum(len(splits[sp]) for sp in SPLIT_NAMES)
    leaked = sum(len(g) for g in crossing) if crossing else 0
    print(f"  [{label}] files={n_files}  parent-recordings={len(seen)}  "
          f"recordings spanning >1 split = {len(crossing)}")
    return crossing


def main():
    old = json.loads(open(SPLITS).read())
    old = {sp: [(c, p) for c, p in old[sp]] for sp in SPLIT_NAMES}
    items = [(c, p) for sp in SPLIT_NAMES for c, p in old[sp]]

    print("Leakage BEFORE (current file-level split):")
    leak_report(old, 'before')

    new = group_aware_split(items, seed=SEED)

    print("\nLeakage AFTER (group-aware split):")
    crossing = leak_report(new, 'after')
    assert not crossing, "group-aware split still leaks — bug in group_key"

    print("\nPer-class file counts (train / val / test):")
    cc = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    for sp in SPLIT_NAMES:
        for c, _ in new[sp]:
            cc[c][sp] += 1
    for c in sorted(cc):
        d = cc[c]; t = d['train'] + d['val'] + d['test']
        print(f"  {c:24} {d['train']:5} / {d['val']:5} / {d['test']:5}   "
              f"({d['train']/t:.0%}/{d['val']/t:.0%}/{d['test']/t:.0%})")

    tot = {sp: len(new[sp]) for sp in SPLIT_NAMES}
    print(f"\nTotal files: train={tot['train']}  val={tot['val']}  test={tot['test']}")

    bak = SPLITS + '.fileLevel.bak'
    with open(bak, 'w') as f:
        json.dump({sp: [[c, str(p)] for c, p in old[sp]] for sp in SPLIT_NAMES}, f)
    with open(SPLITS, 'w') as f:
        json.dump({sp: [[c, str(p)] for c, p in new[sp]] for sp in SPLIT_NAMES}, f, indent=2)
    print(f"\nBacked up old split → {bak}")
    print(f"Wrote group-aware split → {SPLITS}")
    print("\nNext: re-run  python scripts/audio_preprocessing.py  to re-shard, then retrain.")


if __name__ == '__main__':
    main()
