"""Medians of eval_v3.py shards: [O=dir] [COLD=TAG] summarize_v3.py TAG SPLIT [TAG SPLIT ...]

Every tag scores the same fixed 300 episodes per split, so a tag evaluated
without --cold borrows cold200 from the COLD tag for its paired tests.
A tag evaluated with --keep also gets a keep-repair table: attainment
median / mean / share >= 0.5, median drift of the unmentioned classes, and
the repair's mean evaluations and milliseconds.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from rl.vis_eval import EVAL_KINDS, wilcoxon  # noqa: E402

O = os.environ.get("O", "out/rl_mb_v3/eval")
COLD = os.environ.get("COLD")
METHODS = ("policy", "warm30", "warm200", "plan", "plan_warm30", "plan_warm200", "cold200")
KEEP_BASES = ("policy", "plan", "warm30", "warm200", "plan_warm30", "plan_warm200")


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
    if "cold200" not in rows[0] and COLD:
        for row, cold in zip(rows, load(COLD, split)):
            assert row["i"] == cold["i"] and row["volume"] == cold["volume"]
            row["cold200"], row["cold200_pp"] = cold["cold200"], cold["cold200_pp"]
    print(f"\n== {tag} {split}  (start 'other' median {np.median([r['start_other'] for r in rows]):.2f})")
    print(f"{'method':12s} {'att':>6s} {'pp':>6s}  " + "  ".join(f"{k[:8]:>13s}" for k in EVAL_KINDS))
    for m in METHODS:
        if m not in rows[0]:
            continue
        cells = [f"{med([r[m] for r in rows if r['kind'] == k])}/{med([r[m + '_pp'] for r in rows if r['kind'] == k])}"
                 for k in EVAL_KINDS]
        print(f"{m:12s} {med([r[m] for r in rows]):>6s} {med([r[m + '_pp'] for r in rows]):>6s}  "
              + "  ".join(f"{c:>13s}" for c in cells))
    for a, b in (("plan", "policy"), ("plan_warm30", "warm30"), ("plan_warm200", "warm200")):
        if a in rows[0] and b in rows[0]:
            p = wilcoxon([r[a] for r in rows], [r[b] for r in rows])["p_value"]
            print(f"  {a} vs {b}: p = {p:.2g}")
    if "cold200" in rows[0]:
        for m in [m for m in METHODS[:-1] if m in rows[0]]:
            p = wilcoxon([r[m] for r in rows], [r["cold200"] for r in rows])["p_value"]
            print(f"  {m} vs cold200: p = {p:.2g}")
    keep_bases = [m for m in KEEP_BASES if m + "_keep" in rows[0]]
    if keep_bases:
        print(f"  {'keep repair':18s} {'median':>7s} {'mean':>7s} {'>=0.5':>6s} {'drift':>6s} {'bdrift':>6s}"
              f" {'evals':>6s} {'ms':>6s}  p(vs without)")
        for m in keep_bases:
            for name in (m, m + "_keep"):
                att = np.array([r[name] for r in rows])
                extra = (f" {np.mean([r[m + '_keep_evals'] for r in rows]):6.1f}"
                         f" {np.mean([r[m + '_keep_ms'] for r in rows]):6.0f}"
                         f"  {wilcoxon([r[name] for r in rows], [r[m] for r in rows])['p_value']:.2g}"
                         if name.endswith("_keep") else "")
                print(f"  {name:18s} {np.median(att):+7.3f} {np.mean(att):+7.3f} {np.mean(att >= 0.5):6.2f}"
                      f" {np.median([r[name + '_drift'] for r in rows]):6.2f}"
                      f" {np.median([r[name + '_bdrift'] for r in rows]):6.2f}" + extra)
