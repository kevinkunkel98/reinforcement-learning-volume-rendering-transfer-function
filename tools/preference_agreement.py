"""How often the visibility metric picks the candidate the rater picked.

This is the check the whole preference dataset exists to make: the policy is
trained and scored on `visibility.py`, which is a proxy for "does this render
answer the instruction". If raters disagree with it often, the proxy is the
problem, and that is the motivation for a preference-trained reward. No model
is fitted here -- every row already carries the metric's own pick.
"""
import argparse
import collections
import json
import math

PREF_PATH = "out/vis_preferences.jsonl"
DECIDED = ("a", "b")


def _wilson(agree: int, n: int) -> tuple:
    """Wilson score interval: behaves at 0/n and n/n, where the textbook
    normal approximation gives a zero-width interval."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    p = agree / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _summary(rows: list) -> dict:
    decided = [r for r in rows if r["choice"] in DECIDED]
    agree = sum(r["choice"] == r["objective_choice"] for r in decided)
    n = len(decided)
    return {
        "n": n,
        "agree": agree,
        "rate": (agree / n) if n else None,
        "ci95": _wilson(agree, n),
    }


def agreement(rows: list) -> dict:
    # Rows written before flag_assisted_rows.py existed have no "assisted"
    # key at all; missing means unassisted, not excluded.
    assisted_excluded = sum(bool(r.get("assisted")) for r in rows)
    usable = [r for r in rows if not r.get("assisted")]

    result = _summary(usable)
    result["ties"] = sum(r["choice"] == "equal" for r in usable)
    result["skipped"] = sum(r["choice"] == "skip" for r in usable)
    result["assisted_excluded"] = assisted_excluded

    by_kind = collections.defaultdict(list)
    for row in usable:
        by_kind[row["instruction"]["kind"]].append(row)
    result["per_kind"] = {kind: _summary(kind_rows) for kind, kind_rows in sorted(by_kind.items())}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=PREF_PATH)
    args = ap.parse_args()
    with open(args.path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    result = agreement(rows)

    rate = "n/a" if result["rate"] is None else f"{result['rate']:.3f}"
    lo, hi = result["ci95"]
    print(f"metric-vs-human agreement: {rate} (95% CI {lo:.3f}-{hi:.3f}) on n={result['n']}")
    print(f"  ties={result['ties']} skipped={result['skipped']} "
          f"assisted excluded={result['assisted_excluded']}")
    for kind, summary in result["per_kind"].items():
        kind_rate = "n/a" if summary["rate"] is None else f"{summary['rate']:.3f}"
        print(f"  {kind:<12} {kind_rate} on n={summary['n']}")


if __name__ == "__main__":
    main()
