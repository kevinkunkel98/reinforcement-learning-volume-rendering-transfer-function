"""Medians of eval_v3.py shards: summarize_v3.py TAG SPLIT [TAG SPLIT ...]"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from rl.vis_eval import EVAL_KINDS, wilcoxon  # noqa: E402

O = "out/rl_mb_v3/eval"
METHODS = ("policy", "warm30", "warm200", "plan", "plan_warm30", "plan_warm200", "cold200")


def load(tag, split):
    rows = []
    for k in range(6):
        path = os.path.join(O, f"{tag}_{split}_{k}.json")
        with open(path) as stream:
            rows += json.load(stream)
    assert len(rows) == 300, (tag, split, len(rows))
    return sorted(rows, key=lambda r: r["i"])


def med(values):
    values = [v for v in values if v is not None]
    return f"{np.median(values):+.2f}" if values else "  n/a"


args = sys.argv[1:]
for tag, split in zip(args[::2], args[1::2]):
    rows = load(tag, split)
    print(f"\n== {tag} {split}  (start 'other' median {np.median([r['start_other'] for r in rows]):.2f})")
    print(f"{'method':12s} {'att':>6s} {'pp':>6s}  " + "  ".join(f"{k[:8]:>13s}" for k in EVAL_KINDS))
    for m in METHODS:
        if m not in rows[0]:
            continue
        cells = [f"{med([r[m] for r in rows if r['kind'] == k])}/{med([r[m + '_pp'] for r in rows if r['kind'] == k])}"
                 for k in EVAL_KINDS]
        print(f"{m:12s} {med([r[m] for r in rows]):>6s} {med([r[m + '_pp'] for r in rows]):>6s}  "
              + "  ".join(f"{c:>13s}" for c in cells))
    if "cold200" in rows[0]:
        for m in [m for m in METHODS[:-1] if m in rows[0]]:
            p = wilcoxon([r[m] for r in rows], [r["cold200"] for r in rows])["p_value"]
            print(f"  {m} vs cold200: p = {p:.2g}")
