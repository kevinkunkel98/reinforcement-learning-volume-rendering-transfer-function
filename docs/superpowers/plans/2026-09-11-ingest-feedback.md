# Ingest Chat-UI Feedback into RLHF Preference Data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `out/rlhf_preferences.jsonl` from already-logged chat-UI usage (`out/feedback.jsonl` ratings + `out/log.jsonl`'s per-command rendered features) instead of requiring a dedicated `rl/collect_preferences.py` labeling session.

**Architecture:** One new module, `rl/ingest_feedback.py`, joins the two existing logs by exact `params` match, filters to the single-tissue-opacity-increase/decrease command shape the reward model's MDP models, and rebuilds `out/rlhf_preferences.jsonl` from scratch each run. `rl/reward_model.py`/`rl/reward_model_env.py`/`rl/reward_model_train.py` are untouched — the output schema is identical to what `rl/collect_preferences.py` already produces.

**Tech Stack:** Python, standard library only (`json`, `argparse`, `os`).

**Full design:** `docs/superpowers/specs/2026-09-11-ingest-feedback-design.md`

---

## Task 1: `rl/ingest_feedback.py`

**Files:**
- Create: `rl/ingest_feedback.py`
- Test: `tests/test_rl_ingest_feedback.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Fast tests for rl/ingest_feedback.py -- synthetic logs, no real rendering."""
import json
import os

from rl.ingest_feedback import ingest_feedback


def _write_jsonl(path, rows):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _features(seed):
    return {"mean": 10.0 + seed, "std": 5.0, "coverage": 0.2, "entropy": 1.0}


def test_ingest_feedback_joins_and_labels_opacity_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    log_rows = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "command": {"target": "bone", "attribute": "opacity", "direction": "increase"},
            "params_before": [0.0] * 24, "params_after": [1.0] * 24,
            "features_before": _features(0), "features_after": _features(1),
            "verdict": None,
        },
        {
            "timestamp": "2026-01-01T00:01:00",
            "command": {"target": "fat", "attribute": "opacity", "direction": "decrease"},
            "params_before": [1.0] * 24, "params_after": [2.0] * 24,
            "features_before": _features(1), "features_after": _features(2),
            "verdict": None,
        },
    ]
    feedback_rows = [
        {  # matches log row 0, opacity increase, up
            "timestamp": "2026-01-01T00:00:05", "session_id": "sess-a", "step_id": 1,
            "cmd_text": "increase opacity for bone",
            "cmd_dict": {"target": "bone", "attribute": "opacity", "direction": "increase", "strength": None},
            "params": [1.0] * 24, "rating": "up",
        },
        {  # matches log row 1, opacity decrease, down
            "timestamp": "2026-01-01T00:01:05", "session_id": "sess-a", "step_id": 2,
            "cmd_text": "decrease opacity for fat",
            "cmd_dict": {"target": "fat", "attribute": "opacity", "direction": "decrease", "strength": None},
            "params": [2.0] * 24, "rating": "down",
        },
        {  # camera command -- must be filtered out
            "timestamp": "2026-01-01T00:02:00", "session_id": "sess-a", "step_id": 3,
            "cmd_text": "rotate left",
            "cmd_dict": {"camera": {"action": "rotate", "direction": "left", "strength": "moderately"}},
            "params": [2.0] * 24, "rating": "up",
        },
        {  # no matching log row -- must be filtered out
            "timestamp": "2026-01-01T00:03:00", "session_id": "sess-a", "step_id": 4,
            "cmd_text": "increase opacity for spongy",
            "cmd_dict": {"target": "spongy", "attribute": "opacity", "direction": "increase", "strength": None},
            "params": [9.0] * 24, "rating": "up",
        },
    ]
    _write_jsonl("out/log.jsonl", log_rows)
    _write_jsonl("out/feedback.jsonl", feedback_rows)

    n = ingest_feedback(out_path="out/rlhf_preferences.jsonl")

    assert n == 2
    with open("out/rlhf_preferences.jsonl") as f:
        written = [json.loads(line) for line in f]
    assert len(written) == 2
    assert written[0]["target_tissue"] == "bone"
    assert written[0]["direction"] == "increase"
    assert written[0]["label"] == 1
    assert written[0]["before_features"] == _features(0)
    assert written[0]["after_features"] == _features(1)
    assert written[1]["target_tissue"] == "fat"
    assert written[1]["label"] == -1


def test_ingest_feedback_rebuilds_not_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl("out/log.jsonl", [{
        "timestamp": "t", "command": {}, "params_before": [0.0] * 24, "params_after": [1.0] * 24,
        "features_before": _features(0), "features_after": _features(1), "verdict": None,
    }])
    _write_jsonl("out/feedback.jsonl", [{
        "timestamp": "t", "session_id": "s", "step_id": 1, "cmd_text": "x",
        "cmd_dict": {"target": "bone", "attribute": "opacity", "direction": "increase", "strength": None},
        "params": [1.0] * 24, "rating": "up",
    }])

    ingest_feedback(out_path="out/rlhf_preferences.jsonl")
    ingest_feedback(out_path="out/rlhf_preferences.jsonl")

    with open("out/rlhf_preferences.jsonl") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 1
```

Save this to `tests/test_rl_ingest_feedback.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rl_ingest_feedback.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.ingest_feedback'`

- [ ] **Step 3: Write the implementation**

```python
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
```

Save this to `rl/ingest_feedback.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rl_ingest_feedback.py -v`
Expected: PASS (2 tests, fast — no rendering, synthetic JSONL fixtures only)

- [ ] **Step 5: Commit**

```bash
git add rl/ingest_feedback.py tests/test_rl_ingest_feedback.py
git commit -m "Add rl/ingest_feedback.py: build RLHF preference data from chat-UI usage logs"
```
Do NOT add any `Co-Authored-By`/`Claude-Session` trailer — this repo's convention is no AI attribution in commit messages.

---

## Task 2: Run it for real

No new code — runs Task 1's script against the real logs already on disk from prior chat-UI usage.

- [ ] **Step 1: Run the full fast suite to confirm nothing broke**

Run: `.venv/bin/python -m pytest -m "not slow"`
Expected: PASS, no new failures

- [ ] **Step 2: Run the real ingestion**

Run: `.venv/bin/python -m rl.ingest_feedback`
Expected: prints `[ingest_feedback] wrote N rows from 16 feedback entries (out/rlhf_preferences.jsonl)` — based on the real logs inspected during design, `N` should be 15 (one camera-command row filtered out). `out/rlhf_preferences.jsonl` exists afterward with 15 lines.

- [ ] **Step 3: Train the reward model on the real ingested data**

Run: `.venv/bin/python -m rl.reward_model --preferences out/rlhf_preferences.jsonl`
Expected: prints `[reward_model] train_acc=... val_acc=... (n_train=12, n_val=3)` (80/20 split of 15 rows). Report both numbers honestly — this is still a small-sample pilot (15 labels from ordinary use, not a dedicated 50-label session). `out/rl_models/reward_model.pt` exists afterward.

- [ ] **Step 4: Report results — nothing to commit**

`out/` is gitignored. Report the train/val accuracy from Step 3, and how many rows were actually usable (`N` from Step 2) vs. total feedback collected, back to the user.

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the design's "New module" and "Testing" sections exactly (both filter cases — non-opacity command, unmatched params — and the rebuild-not-duplicate behavior are tested). Task 2 covers the "Data flow" end-to-end against real data.
- **Type/signature consistency:** `ingest_feedback(feedback_path, log_path, out_path) -> int` is called identically in both tests and in Task 2's CLI usage.
- **No placeholders.**
- **Scope check:** one new file + one test file; no existing file modified, matching the spec's explicit non-goals.
