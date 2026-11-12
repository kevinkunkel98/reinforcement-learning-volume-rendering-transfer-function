"""Patient-level summary of rl.evaluate result shards.

Episodes of one patient are not independent, so a test over all episodes
overstates n. Here the unit is the patient: each method's median attainment
per patient, the median over patients with a bootstrap 95 % CI that resamples
patients (and episodes within them), and paired Wilcoxon tests on the
per-patient medians.

    python docs/experiments/2026-11-12-scripts/patient_stats.py out/eval_extra/s1_test_extra_*.json
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from rl.evaluate import METHODS, wilcoxon  # noqa: E402

PAIRS = (("plan", "policy"), ("plan_warm30", "warm30"), ("plan_warm30", "cold200"),
         ("plan_warm200", "cold200"))
BOOTSTRAP = 2000


def load(paths) -> list:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            rows += json.load(stream)
    return sorted({r["i"]: r for r in rows}.values(), key=lambda r: r["i"])


def by_patient(rows, method) -> dict:
    groups = collections.defaultdict(list)
    for row in rows:
        if method in row:
            groups[row["volume"]].append(row[method])
    return {volume: np.asarray(values) for volume, values in sorted(groups.items())}


def cluster_bootstrap(groups: dict, rng) -> tuple:
    """95 % CI of the median over patients of the per-patient medians."""
    patients = list(groups.values())
    stats = []
    for _ in range(BOOTSTRAP):
        picked = rng.integers(0, len(patients), len(patients))
        stats.append(np.median([np.median(rng.choice(patients[p], len(patients[p]))) for p in picked]))
    return tuple(np.percentile(stats, [2.5, 97.5]))


def main(paths):
    rows = load(paths)
    rng = np.random.default_rng(0)
    methods = [m for m in METHODS if m in rows[0]]
    patients = sorted({r["volume"] for r in rows})
    print(f"{len(rows)} episodes, {len(patients)} patients "
          f"({min(collections.Counter(r['volume'] for r in rows).values())}-"
          f"{max(collections.Counter(r['volume'] for r in rows).values())} episodes each)")
    print(f"{'method':<14}{'episode median':>16}{'patient median':>16}{'95% CI (patients)':>22}")
    medians = {}
    for method in methods:
        groups = by_patient(rows, method)
        medians[method] = {v: float(np.median(x)) for v, x in groups.items()}
        lo, hi = cluster_bootstrap(groups, rng)
        print(f"{method:<14}{np.median([r[method] for r in rows]):>16.3f}"
              f"{np.median(list(medians[method].values())):>16.3f}{f'[{lo:.3f}, {hi:.3f}]':>22}")
    print("paired Wilcoxon on per-patient medians:")
    for a, b in PAIRS:
        if a in medians and b in medians:
            x = [medians[a][v] for v in patients]
            y = [medians[b][v] for v in patients]
            test = wilcoxon(x, y)
            wins = sum(p > q for p, q in zip(x, y))
            print(f"  {a} vs {b}: p = {test['p_value']:.2g}, {wins}/{len(patients)} patients higher")


if __name__ == "__main__":
    main(sys.argv[1:])
