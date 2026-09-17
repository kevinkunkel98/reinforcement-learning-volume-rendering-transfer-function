"""Mark preference rows that were rated with Claude commenting on the pair.

Those judgments are not independent: the rater saw an argument for one
candidate before choosing. They stay in the dataset -- they are still a
rater's answers -- but anything used to *evaluate* the reward model or a
policy must exclude them, or the evaluation measures the assistant as much as
the rater.
"""
import argparse
import datetime
import json
import os
import tempfile

PREF_PATH = "out/vis_preferences.jsonl"


def _parse(stamp: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(stamp)


def flag_rows(rows: list, start: str, end: str) -> list:
    """Every row gets an explicit `assisted` bool; an existing True is kept."""
    lo, hi = _parse(start), _parse(end)
    flagged = []
    for row in rows:
        row = dict(row)
        inside = lo <= _parse(row["timestamp"]) <= hi
        row["assisted"] = bool(row.get("assisted", False) or inside)
        flagged.append(row)
    return flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=PREF_PATH)
    ap.add_argument("--start", required=True, help="ISO timestamp, inclusive")
    ap.add_argument("--end", required=True, help="ISO timestamp, inclusive")
    args = ap.parse_args()

    with open(args.path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    flagged = flag_rows(rows, args.start, args.end)

    # Written through a temporary file in the same directory: a half-written
    # preference file would lose judgments that cannot be re-collected.
    directory = os.path.dirname(os.path.abspath(args.path))
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as tmp:
        for row in flagged:
            tmp.write(json.dumps(row) + "\n")
        temporary = tmp.name
    os.replace(temporary, args.path)
    print(f"{sum(r['assisted'] for r in flagged)}/{len(flagged)} rows flagged assisted")


if __name__ == "__main__":
    main()
