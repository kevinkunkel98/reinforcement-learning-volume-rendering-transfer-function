"""Builds out/rlhf_preferences.jsonl from already-logged chat-UI usage
(out/feedback.jsonl ratings + out/log.jsonl's per-command rendered
features) instead of requiring a dedicated rl/collect_preferences.py
session -- see docs/superpowers/specs/2026-09-11-ingest-feedback-design.md."""
import argparse
import json
import os

FEEDBACK_PATH = "out/feedback.jsonl"
LOG_PATH = "out/log.jsonl"
OUT_PATH = "out/rlhf_preferences.jsonl"
RATING_TO_LABEL = {"up": 1, "down": -1}


def ingest_feedback(feedback_path: str = FEEDBACK_PATH, log_path: str = LOG_PATH,
                     out_path: str = OUT_PATH) -> int:
    by_params_after = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                row = json.loads(line)
                by_params_after[tuple(row["params_after"])] = row

    total = 0
    rows = []
    if os.path.exists(feedback_path):
        with open(feedback_path) as f:
            for line in f:
                total += 1
                fb = json.loads(line)
                if fb.get("rating") not in RATING_TO_LABEL:
                    continue
                cmd = fb.get("cmd_dict") or {}
                if cmd.get("attribute") != "opacity" or cmd.get("direction") not in ("increase", "decrease") \
                        or not isinstance(cmd.get("target"), str):
                    continue
                log_row = by_params_after.get(tuple(fb["params"]))
                if log_row is None:
                    continue
                rows.append({
                    "timestamp": fb["timestamp"],
                    "rater_id": fb.get("session_id", "unknown"),
                    "target_tissue": cmd["target"],
                    "direction": cmd["direction"],
                    "delta": None,
                    "before_features": log_row["features_before"],
                    "after_features": log_row["features_after"],
                    "label": RATING_TO_LABEL[fb["rating"]],
                })

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"[ingest_feedback] wrote {len(rows)} rows from {total} feedback entries ({out_path})")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild out/rlhf_preferences.jsonl from chat-UI feedback + command logs.")
    parser.add_argument("--feedback", type=str, default=FEEDBACK_PATH)
    parser.add_argument("--log", type=str, default=LOG_PATH)
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()
    ingest_feedback(args.feedback, args.log, args.out)


if __name__ == "__main__":
    main()
