"""Analyze out/preferences.jsonl: pair counts, agreement rate, disagreements."""
import argparse
import json
from collections import defaultdict

DEFAULT_PATH = "out/preferences.jsonl"


def load(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    args = ap.parse_args()

    entries = load(args.path)
    if not entries:
        print(f"no preference pairs found in {args.path}")
        return

    by_tissue = defaultdict(list)
    for e in entries:
        tissue = e["cmd_dict"].get("target") or "unknown"
        by_tissue[tissue].append(e)

    print("pairs per target tissue:")
    for tissue, items in sorted(by_tissue.items()):
        print(f"  {tissue}: {len(items)}")

    def agreement(items):
        agree = sum(1 for e in items if e["human_verdict"] == e["objective_verdict"])
        return agree, len(items)

    total_agree, total_n = agreement(entries)
    print(f"\noverall agreement: {total_agree}/{total_n} ({100 * total_agree / total_n:.1f}%)")
    print("agreement per tissue:")
    for tissue, items in sorted(by_tissue.items()):
        a, n = agreement(items)
        print(f"  {tissue}: {a}/{n} ({100 * a / n:.1f}%)")

    disagreements = [e for e in entries if e["human_verdict"] != e["objective_verdict"]]
    print(f"\ndisagreements ({len(disagreements)}):")
    for e in disagreements:
        print(f"  [{e['cmd_dict'].get('target')}] human={e['human_verdict']} "
              f"objective={e['objective_verdict']}  {e['before_png']} -> {e['after_png']}")


if __name__ == "__main__":
    main()
