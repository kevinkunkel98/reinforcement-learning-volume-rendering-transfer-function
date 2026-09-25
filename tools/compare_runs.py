"""Compare two groups of `rl.vis_eval` result files -- typically the three
seeds of one training configuration against the three of another -- on the
statistic the write-up quotes for v2 against v3:

    .venv/bin/python -m tools.compare_runs \
        --a "out/rl_v2/eval_rerun_v2_seed*.json" \
        --b "out/rl_v2/eval_rerun_v3_seed*.json"

Each group's seeds are averaged *per episode* first, and the median is taken
over those averaged episodes. That is a different number from the median of
the seeds' own medians, and the two must not be crossed: on the corrected
pipeline v2/v3 is +0.201 against +0.286 seed-averaged, and +0.247 against
+0.275 as medians of medians. Both are honest; only one goes with the paired
p-value this tool reports, because the pairing is per episode.

Every arm is scored on identical episodes (`rl.vis_eval.fixed_episodes` is
deterministic in `--seed`), so the comparison is paired: episode i of every
file is the same volume, start state and instruction. This tool verifies that
rather than assuming it, and refuses files whose episodes disagree.

It reads stored per-episode rows and never re-runs a policy, so it cannot
drift from the result files the way a number computed in a throwaway session
can -- which is what happened to this comparison the first time.
"""
import argparse
import glob
import json

import numpy as np

import provenance
from rl.vis_eval import wilcoxon

DEFAULT_ARM = "policy"


def load_arm(path: str, arm: str = DEFAULT_ARM) -> list:
    """One result file's per-episode rows for `arm`.

    Files written before `episodes_detail` existed carry summaries only. A
    median cannot be un-aggregated, so there is nothing to fall back to and
    this raises rather than reporting a comparison of the wrong statistic."""
    with open(path) as stream:
        payload = json.load(stream)
    detail = payload.get("episodes_detail")
    if not detail:
        raise ValueError(
            f"{path}: no per-episode rows -- written before rl.vis_eval stored them. "
            "Re-run the evaluation; a stored median cannot be re-aggregated.")
    if arm not in detail:
        raise ValueError(f"{path}: no arm {arm!r} (has: {', '.join(sorted(detail))})")
    return detail[arm], payload.get("provenance")


def seed_average(groups: list) -> list:
    """One row per episode, its attainment averaged across `groups` (the
    seeds of one configuration). An episode is `None` if any seed reports
    `None` there -- a partly-scored episode is not an average."""
    first = groups[0]
    averaged = []
    for index in range(len(first)):
        values = [group[index]["attainment"] for group in groups]
        mean = None if any(v is None for v in values) else float(np.mean(values))
        averaged.append({"attainment": mean, "kind": first[index]["kind"],
                         "volume": first[index]["volume"]})
    return averaged


def _check_aligned(named_rows: list) -> None:
    """Every file must describe the same episodes in the same order."""
    reference_name, reference = named_rows[0]
    for name, rows in named_rows[1:]:
        if len(rows) != len(reference):
            raise ValueError(
                f"{name}: {len(rows)} episodes against {len(reference)} in {reference_name}")
        for index, (row, expected) in enumerate(zip(rows, reference)):
            identity = (row.get("kind"), row.get("volume"), row.get("start_params"),
                        row.get("targets"), row.get("goal"))
            expected_identity = (expected.get("kind"), expected.get("volume"),
                                 expected.get("start_params"), expected.get("targets"),
                                 expected.get("goal"))
            if identity != expected_identity:
                raise ValueError(
                    f"episode {index} differs between {reference_name} and {name}: "
                    f"{expected['volume']}/{expected['kind']} against "
                    f"{row['volume']}/{row['kind']} -- these runs were not scored on "
                    "the same episodes, so they cannot be paired")


def _median(values) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def compare_groups(a_paths: list, b_paths: list, arm: str = DEFAULT_ARM) -> dict:
    """Paired comparison of group `b` against group `a` on seed-averaged
    per-episode attainment: each side's median, their difference, the share
    of episodes `b` wins, a paired Wilcoxon signed-rank test, and the same
    per instruction kind."""
    loaded_a = [load_arm(path, arm) for path in a_paths]
    loaded_b = [load_arm(path, arm) for path in b_paths]
    provenance_records = [record for _, record in loaded_a + loaded_b]
    if any(record is not None for record in provenance_records):
        if any(record is None for record in provenance_records):
            raise ValueError("result provenance is missing from one or more paired files")
        reference_provenance = provenance_records[0]
        expected_layers = reference_provenance.get("anatomy_layers")
        for record in provenance_records[1:]:
            report = provenance.compare(record, current=reference_provenance,
                                        expected_layers=expected_layers)
            if report["stale"]:
                raise ValueError("result provenance is incompatible: "
                                 + "; ".join(report["reasons"]))
    a_groups = [rows for rows, _ in loaded_a]
    b_groups = [rows for rows, _ in loaded_b]
    _check_aligned(list(zip(a_paths, a_groups)) + list(zip(b_paths, b_groups)))

    a_rows, b_rows = seed_average(a_groups), seed_average(b_groups)
    pairs = [(a, b) for a, b in zip(a_rows, b_rows)
             if a["attainment"] is not None and b["attainment"] is not None]
    a_values = [a["attainment"] for a, _ in pairs]
    b_values = [b["attainment"] for _, b in pairs]

    by_kind = {}
    for kind in sorted({a["kind"] for a, _ in pairs}):
        kind_a = [a["attainment"] for a, _ in pairs if a["kind"] == kind]
        kind_b = [b["attainment"] for a, b in pairs if a["kind"] == kind]
        by_kind[kind] = {"a": _median(kind_a), "b": _median(kind_b),
                         "delta": _median(kind_b) - _median(kind_a), "n": len(kind_a)}

    test = wilcoxon(a_values, b_values)
    return {
        "arm": arm,
        "a": {"files": list(a_paths), "median": _median(a_values), "n_seeds": len(a_groups)},
        "b": {"files": list(b_paths), "median": _median(b_values), "n_seeds": len(b_groups)},
        "delta": _median(b_values) - _median(a_values),
        "share_b_ahead": float(np.mean(np.asarray(b_values) > np.asarray(a_values))),
        "p_value": test["p_value"],
        "n_episodes": len(pairs),
        "by_kind": by_kind,
    }


# --- CLI -----------------------------------------------------------------------

def _print_report(result: dict) -> None:
    a, b = result["a"], result["b"]
    print(f"\narm: {result['arm']}   episodes: {result['n_episodes']}   "
          f"seeds: {a['n_seeds']} against {b['n_seeds']}")
    print("\nseed-averaged per-episode attainment, median over episodes:")
    print(f"  a  {a['median']:+.4f}   ({len(a['files'])} files)")
    print(f"  b  {b['median']:+.4f}   ({len(b['files'])} files)")
    print(f"  delta {result['delta']:+.4f}   b ahead on {result['share_b_ahead']*100:.0f} % "
          f"of episodes   paired Wilcoxon p = {result['p_value']:.4g}")
    print("\nper instruction kind:")
    print(f"  {'kind':12} {'a':>9} {'b':>9} {'delta':>9} {'n':>5}")
    for kind, stats in result["by_kind"].items():
        print(f"  {kind:12} {stats['a']:+9.4f} {stats['b']:+9.4f} "
              f"{stats['delta']:+9.4f} {stats['n']:5d}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs="+", required=True,
                         help="result files (or a glob) for the reference group, e.g. the v2 seeds")
    parser.add_argument("--b", nargs="+", required=True,
                         help="result files (or a glob) for the group being tested, e.g. the v3 seeds")
    parser.add_argument("--arm", default=DEFAULT_ARM,
                         help="which arm to compare (default: %(default)s)")
    parser.add_argument("--out", default=None, help="also write the report as JSON")
    return parser.parse_args(argv)


def _expand(patterns: list) -> list:
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched or [pattern])
    return paths


def main(argv=None):
    args = parse_args(argv)
    result = compare_groups(_expand(args.a), _expand(args.b), arm=args.arm)
    _print_report(result)
    if args.out:
        with open(args.out, "w") as stream:
            json.dump(result, stream, indent=2)
        print(f"\n[compare_runs] wrote {args.out}")


if __name__ == "__main__":
    main()
