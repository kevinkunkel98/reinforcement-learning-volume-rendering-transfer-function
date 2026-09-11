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
