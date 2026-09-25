"""First baseline attainment table on real volumes.

Samples N instructions per volume (`goals.sample_instruction`), runs every
non-learned baseline (`rl.baselines.BASELINES`) on each, scores the result
with `goals.attainment`, and reports mean attainment per baseline and per
instruction kind. This is the table a learned policy (Plan 5) has to beat --
if B1 (today's executor) already attains ~1.0 on most instructions, the
instruction set is too easy to show any value from training, and that has to
be known before training starts.

    .venv/bin/python -m tools.baseline_report --volumes ts_s1379 ts_s1337 ts_s0454 --instructions 30
"""
import argparse
import json
import os

import numpy as np

import goals
import provenance
import visibility
from rl.baselines import BASELINES

OUT_PATH = "out/baseline_report.json"


def _attainment_or_none(goal, start_agg, final_agg):
    """None when the start state already meets the goal (distance(start) is
    ~0): attainment's denominator would be zero, and the row has nothing to
    say about whether a baseline moved the needle."""
    start_distance = goals.distance(goal, start_agg, start_agg)
    if start_distance <= 1e-9:
        return None
    return goals.attainment(goal, start_agg, final_agg)


def run_volume(name: str, n_instructions: int, seed: int = 0, active_layers=None) -> list:
    """One row per (instruction, baseline) pair for this volume."""
    model = visibility.for_volume(name)
    start_params = goals.starting_params()
    start_agg = goals.aggregate(goals.features(model, start_params, active_layers), active_layers)
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n_instructions):
        instruction = goals.sample_instruction(name, model, start_agg, rng, active_layers)
        for baseline_name, baseline_fn in BASELINES.items():
            final_params = baseline_fn(model, start_params, instruction,
                                       active_layers=active_layers)
            final_features = goals.features(model, final_params, active_layers)
            final_agg = goals.aggregate(final_features, active_layers)
            attainment = _attainment_or_none(instruction["goal"], start_agg, final_agg)
            rows.append({"volume": name, "kind": instruction["kind"],
                         "baseline": baseline_name, "attainment": attainment})
    return rows


def summarise(rows: list) -> dict:
    """Robust attainment stats per baseline (`goals.summarise_attainment`:
    median, mean clipped to [-1, 1], raw mean, share of episodes that
    improved on doing nothing, n), and a plain mean per (baseline,
    instruction kind); None rows are not counted."""
    by_baseline: dict = {}
    by_kind: dict = {}
    for row in rows:
        attainment = row["attainment"]
        if attainment is None:
            continue
        by_baseline.setdefault(row["baseline"], []).append(attainment)
        by_kind.setdefault(row["baseline"], {}).setdefault(row["kind"], []).append(attainment)

    baseline_summary = {name: goals.summarise_attainment(values) for name, values in by_baseline.items()}
    kind_summary = {name: {kind: {"mean": float(np.mean(values)), "n": len(values)}
                           for kind, values in kinds.items()}
                    for name, kinds in by_kind.items()}
    return {"by_baseline": baseline_summary, "by_kind": kind_summary}


def _print_table(summary: dict) -> None:
    print(f"{'baseline':22s} {'median':>8s} {'mean_clip':>10s} {'mean_raw':>10s} {'share+':>8s} {'n':>6s}")
    for name in sorted(summary["by_baseline"]):
        entry = summary["by_baseline"][name]
        print(f"{name:22s} {entry['median']:8.3f} {entry['mean_clipped']:10.3f} "
              f"{entry['mean_raw']:10.3f} {entry['share_positive']:8.3f} {entry['n']:6d}")

    print()
    print(f"{'baseline':22s} {'kind':12s} {'mean attainment':>16s} {'n':>6s}")
    for name in sorted(summary["by_kind"]):
        for kind in sorted(summary["by_kind"][name]):
            entry = summary["by_kind"][name][kind]
            print(f"{name:22s} {kind:12s} {entry['mean']:16.3f} {entry['n']:6d}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volumes", nargs="+", required=True)
    parser.add_argument("--instructions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=OUT_PATH)
    layers = parser.add_mutually_exclusive_group()
    layers.add_argument("--layers", help="active anatomy layers as inline JSON")
    layers.add_argument("--layers-file", help="JSON file containing active anatomy layers")
    args = parser.parse_args(argv)
    active_layers = provenance.load_active_layers(args.layers, args.layers_file)
    if os.path.exists(args.out):
        with open(args.out) as stream:
            previous = json.load(stream)
        if previous.get("provenance") is None:
            raise ValueError("existing baseline report has no provenance; remove it before regeneration")
        report = provenance.compare(previous["provenance"], expected_layers=active_layers)
        if report["stale"]:
            raise ValueError("existing baseline report provenance is incompatible: "
                             + "; ".join(report["reasons"]))

    rows = []
    for name in args.volumes:
        rows += run_volume(name, args.instructions, args.seed, active_layers)

    summary = summarise(rows)
    _print_table(summary)

    high = [name for name, entry in summary["by_baseline"].items()
            if name.startswith("B1") and entry["median"] is not None and entry["median"] >= 0.9]
    if high:
        print()
        print("WARNING: B1 (today's executor) already attains >= 0.9 mean attainment -- "
              "the instruction set may be too easy for a learned policy to add value. "
              "Check before Plan 5 trains anything.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump({"volumes": args.volumes, "instructions": args.instructions, "seed": args.seed,
                   "provenance": provenance.result_provenance(active_layers),
                   "summary": summary, "rows": rows}, stream, indent=2)
    print(f"\n[baseline_report] wrote {args.out}")


if __name__ == "__main__":
    main()
