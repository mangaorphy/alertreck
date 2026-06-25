"""Source-aware grouping for leak-free train/val/test splits.

Many clips are segments of the same parent recording (e.g. gunshots122_00013_1
and gunshots122_00009_1 are different shots from the same session; us8k slices
share a Freesound ID). If such segments land in different splits the model
memorises the recording rather than the sound class, inflating test scores.
group_key() maps each file to its parent recording so whole groups stay within
one split.
"""
import os
import re
import random
from collections import defaultdict


def group_key(path) -> str:
    """Parent-recording identifier for a clip path (dataset-aware)."""
    stem = os.path.basename(str(path)).rsplit('.', 1)[0]

    # UrbanSound8K: <fsID>-<class>-<occurrence>-<slice> → group by Freesound ID
    m = re.match(r'(us8k__\d+)', stem)
    if m:
        return m.group(1)
    # ESC-50: <fold>-<clipID>-<take>-<target> → group by source clip ID
    m = re.match(r'esc50__\d+-(\d+)', stem)
    if m:
        return f'esc50__{m.group(1)}'
    # gunshots<NN>_<seg>_<sub> → group by recording bank
    m = re.match(r'(gunshots\d+)', stem)
    if m:
        return m.group(1)
    # ds02 …_part_<M> chunks → strip the chunk index, keep the recording
    m = re.match(r'(.+?)_part_\d+$', stem)
    if m:
        return m.group(1)
    # locally-recorded human clips m_<phrase>_<rec> → group by phrase id
    m = re.match(r'(m_[0-9a-z]+)_', stem)
    if m:
        return m.group(1)
    # AudioSet (…__<ytid>), Common Voice, ff1010bird: each file is its own source
    return stem


def group_aware_split(items, seed=42, fracs=(0.60, 0.20, 0.20)):
    """Split [(cls, path), …] into train/val/test with no parent recording
    spanning two splits. Groups are assigned per class (so classes stay
    stratified) by a deterministic greedy fill toward the target file
    fractions. Returns {'train': [...], 'val': [...], 'test': [...]}."""
    f_tr, f_va, f_te = fracs
    rng = random.Random(seed)

    by_cls = defaultdict(lambda: defaultdict(list))
    for cls, path in items:
        by_cls[cls][group_key(path)].append((cls, path))

    out = {'train': [], 'val': [], 'test': []}
    for cls in sorted(by_cls):
        groups = list(by_cls[cls].items())                  # [(gkey, [items])]
        rng.shuffle(groups)
        groups.sort(key=lambda g: len(g[1]), reverse=True)  # pack big groups first
        n = sum(len(v) for _, v in groups)
        target = {'train': f_tr * n, 'val': f_va * n, 'test': f_te * n}
        count  = {'train': 0, 'val': 0, 'test': 0}
        for _, members in groups:
            # assign whole group to the split with the largest remaining deficit
            split = max(('train', 'val', 'test'), key=lambda s: target[s] - count[s])
            out[split].extend(members)
            count[split] += len(members)
    return out
