import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from rl.extract_pairs import extract_pairs


FEATURES = {"mean": 10.0, "std": 2.0, "coverage": 0.5, "entropy": 1.0}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def observation(seed=0):
    return {
        "before_features": {**FEATURES, "mean": float(seed)},
        "after_features": {**FEATURES, "mean": float(seed + 1)},
        "before_image": None,
        "after_image": None,
    }


def step(session, episode, step_id, *, accepted=None, ended=None, parent=None,
         carried=None, seed=0, target="bone", direction="increase"):
    row = {
        "session_id": session,
        "episode_id": episode,
        "step_id": step_id,
        "command": {"attribute": "opacity", "target": target, "direction": direction},
        "accepted": accepted,
        "ended": ended,
        **observation(seed),
    }
    if parent is not None:
        row["parent_step_id"] = parent
    if carried is not None:
        row["carried_forward"] = carried
    return row


def test_extracts_recoverable_branches_and_trajectory_pairs(tmp_path):
    log = tmp_path / "canonical.jsonl"
    out = tmp_path / "pairs.jsonl"
    write_jsonl(log, [
        step("s", "e", 1, seed=1),
        step("s", "e", 2, accepted=True, parent=1, carried=True, seed=2),
        step("s", "e", 3, ended=True, seed=3),
    ])

    rows, stats = extract_pairs(log_path=log, out_path=out)

    assert len(rows) == 2
    assert {row["source"] for row in rows} == {"branch", "trajectory"}
    branch = next(row for row in rows if row["source"] == "branch")
    assert branch["weight"] == 1.0
    assert branch["label"] == 1
    assert branch["target_tissue"] == "bone"
    assert branch["direction"] == "increase"
    assert branch["command"] == {"attribute": "opacity", "target": "bone", "direction": "increase"}
    assert branch["observation_a"]["before_features"] == {**FEATURES, "mean": 2.0}
    trajectory = next(row for row in rows if row["source"] == "trajectory")
    assert trajectory["weight"] == 0.7
    assert stats["branch"] == 1


def test_rejects_symlinked_input_output_collision(tmp_path):
    log = tmp_path / "log.jsonl"
    output = tmp_path / "pairs.jsonl"
    alias = tmp_path / "alias.jsonl"
    write_jsonl(log, [])
    alias.symlink_to(log)

    with pytest.raises(ValueError, match="collide"):
        extract_pairs(log_path=alias, out_path=log)


def test_branch_requires_explicit_recoverable_metadata_and_gap_is_two(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "pairs.jsonl"
    write_jsonl(log, [
        step("s", "e", 1, seed=1),
        step("s", "e", 2, accepted=True, seed=2),
        step("s", "e", 3, accepted=True, seed=3),
        step("s", "e", 4, accepted=True, parent=1, carried=False, seed=4),
    ])

    rows, stats = extract_pairs(log_path=log, out_path=out)

    assert len(rows) == 3
    assert all(row["source"] == "trajectory" for row in rows)
    assert stats["branch"] == 0


def test_thumbs_join_by_session_step_then_unique_params_fallback(tmp_path):
    log = tmp_path / "log.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    out = tmp_path / "pairs.jsonl"
    first = step("s", "e", 5, seed=5)
    first["params_after"] = [1, 2]
    second = step("s", "e", 6, seed=6)
    second["params_after"] = [3, 4]
    write_jsonl(log, [first, second])
    write_jsonl(feedback, [
        {"session_id": "s", "step_id": 5, "rating": "up"},
        {"session_id": "other", "params": [3, 4], "rating": "down"},
    ])

    rows, _ = extract_pairs(log_path=log, feedback_path=feedback, out_path=out)

    thumbs = [row for row in rows if row["source"] == "thumbs"]
    assert len(thumbs) == 2
    assert {row["weight"] for row in thumbs} == {0.4}
    assert {row["label"] for row in thumbs} == {1, -1}


def test_recovers_features_from_images_and_preserves_paths(tmp_path):
    before = tmp_path / "before.png"
    after = tmp_path / "after.jpg"
    Image.fromarray(np.full((2, 2, 3), 10, dtype=np.uint8)).save(before)
    Image.fromarray(np.full((2, 2, 3), 20, dtype=np.uint8)).save(after)
    log = tmp_path / "log.jsonl"
    out = tmp_path / "pairs.jsonl"
    row = step("s", "e", 1, seed=1)
    row["before_features"] = None
    row["after_features"] = None
    row["before_image"] = str(before)
    row["after_image"] = str(after)
    row["accepted"] = True
    write_jsonl(log, [row])
    feedback = tmp_path / "feedback.jsonl"
    write_jsonl(feedback, [{"session_id": "s", "step_id": 1, "rating": "up"}])

    rows, stats = extract_pairs(log_path=log, feedback_path=feedback, out_path=out)

    assert len(rows) == 1
    assert rows[0]["source"] == "thumbs"
    assert rows[0]["observation_a"]["before_features"]["mean"] == 20.0
    assert rows[0]["observation_b"]["before_features"]["mean"] == 10.0
    assert rows[0]["observation_a"] != rows[0]["observation_b"]
    assert rows[0]["observation_a"]["before_image"] == str(after)
    assert stats["skipped"] == 0
    assert stats["total"] == 1


def test_empty_inputs_write_valid_empty_output_and_cli_summary(tmp_path):
    out = tmp_path / "nested" / "pairs.jsonl"
    result = subprocess.run([
        sys.executable, "-m", "rl.extract_pairs", "--out", str(out),
    ], capture_output=True, text=True, check=True)

    assert out.read_text() == ""
    assert "source=" in result.stdout
    assert "target=" in result.stdout
    assert "total=0" in result.stdout


def test_grouping_and_dedupe_include_full_normalized_command(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "pairs.jsonl"
    write_jsonl(log, [
        step("s", "e", 1, seed=1),
        step("s", "e", 3, accepted=True, seed=3),
        step("s", "e", 5, accepted=True, seed=5),
    ])
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    rows[1]["command"]["target"] = "fat"
    rows[2]["command"]["target"] = "fat"
    write_jsonl(log, rows)

    pairs, _ = extract_pairs(log_path=log, out_path=out)

    assert len(pairs) == 1
    assert pairs[0]["command"] == {
        "attribute": "opacity", "target": "fat", "direction": "increase"
    }


def test_non_opacity_commands_are_skipped(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "pairs.jsonl"
    row = step("s", "e", 1, seed=1)
    row["command"]["attribute"] = "width"
    write_jsonl(log, [row])

    pairs, stats = extract_pairs(log_path=log, out_path=out)

    assert pairs == []
    assert stats["skipped"] == 1


def test_non_opacity_feedback_is_skipped_instead_of_using_log_command(tmp_path):
    log = tmp_path / "log.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    out = tmp_path / "pairs.jsonl"
    write_jsonl(log, [step("s", "e", 1, seed=1)])
    write_jsonl(feedback, [{
        "session_id": "s", "step_id": 1, "rating": "up",
        "command": {"attribute": "width", "target": "bone", "direction": "increase"},
    }])

    pairs, stats = extract_pairs(log_path=log, feedback_path=feedback, out_path=out)

    assert pairs == []
    assert stats["skipped"] == 1


def test_input_output_path_collision_is_rejected(tmp_path):
    log = tmp_path / "log.jsonl"
    write_jsonl(log, [step("s", "e", 1, seed=1)])

    with pytest.raises(ValueError, match="input and output|collision"):
        extract_pairs(log_path=log, feedback_path=log, out_path=log)


def test_branch_parent_must_match_full_command_context(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "pairs.jsonl"
    parent = step("s", "e", 1, seed=1)
    parent["command"]["attribute"] = "width"
    child = step("s", "e", 2, accepted=True, parent=1, carried=True, seed=2)
    child["command"]["attribute"] = "opacity"
    write_jsonl(log, [parent, child])

    pairs, stats = extract_pairs(log_path=log, out_path=out)

    assert not any(pair["source"] == "branch" for pair in pairs)
    assert stats["branch"] == 0


def test_aliases_are_canonicalized_and_unknown_tissues_rejected(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "pairs.jsonl"
    alias = step("s", "e", 1, seed=1, target="bones")
    unknown = step("s", "e", 2, seed=2, target="marrow")
    write_jsonl(log, [alias, unknown])

    pairs, stats = extract_pairs(log_path=log, out_path=out)

    assert stats["skipped"] == 1
    assert pairs == []


def test_current_log_feedback_schema_remains_supported(tmp_path):
    log = tmp_path / "log.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    out = tmp_path / "pairs.jsonl"
    row = {
        "session_id": "s",
        "step_id": 7,
        "command": {"target": "bone", "attribute": "opacity", "direction": "increase"},
        "params_after": [1, 2],
        "features_before": FEATURES,
        "features_after": {**FEATURES, "mean": 11.0},
    }
    write_jsonl(log, [row])
    write_jsonl(feedback, [{
        "session_id": "s", "step_id": 7, "params": [1, 2], "rating": "up",
        "cmd_dict": {"target": "bone", "attribute": "opacity", "direction": "increase"},
    }])

    pairs, _ = extract_pairs(log_path=log, feedback_path=feedback, out_path=out)

    assert len(pairs) == 1
    assert pairs[0]["source"] == "thumbs"
    assert pairs[0]["observation_a"]["before_features"]["mean"] == 11.0


def test_canonical_preference_records_are_emitted_directly(tmp_path):
    preferences = tmp_path / "preferences.jsonl"
    out = tmp_path / "pairs.jsonl"
    record = {
        "observation_a": observation(10),
        "observation_b": observation(20),
        "command": {"attribute": "opacity", "target": "bone", "direction": "increase"},
        "target_tissue": "bone",
        "direction": "increase",
        "label": -1,
        "source": "branch",
        "weight": 1.0,
        "session_id": "web-session",
        "episode_id": "web-episode",
        "step_id": 4,
    }
    write_jsonl(preferences, [record])

    pairs, stats = extract_pairs(
        log_path=tmp_path / "missing-log.jsonl",
        feedback_path=tmp_path / "missing-feedback.jsonl",
        preferences_path=preferences,
        out_path=out,
    )

    assert pairs == [record]
    assert stats["branch"] == 1
    assert stats["total"] == 1


def test_thumb_pair_has_nonzero_training_signal_and_rating_semantics(tmp_path):
    log = tmp_path / "log.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    out = tmp_path / "pairs.jsonl"
    row = step("s", "e", 7, seed=1)
    write_jsonl(log, [row])
    write_jsonl(feedback, [{"session_id": "s", "step_id": 7, "rating": "down"}])

    pairs, _ = extract_pairs(log_path=log, feedback_path=feedback, out_path=out)

    pair = pairs[0]
    assert pair["label"] == -1
    assert pair["observation_a"] != pair["observation_b"]
    assert pair["observation_a"]["before_features"] == row["after_features"]
    assert pair["observation_b"]["before_features"] == row["before_features"]
    assert pair["observation_a"]["before_features"]["mean"] != pair["observation_b"]["before_features"]["mean"]


def test_non_finite_features_are_skipped(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "pairs.jsonl"
    row = step("s", "e", 1, seed=1)
    row["after_features"]["mean"] = float("nan")
    write_jsonl(log, [row])

    pairs, stats = extract_pairs(log_path=log, out_path=out)

    assert pairs == []
    assert stats["skipped"] == 1


def test_malformed_nonnumeric_features_are_skipped_and_counted(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "pairs.jsonl"
    row = step("s", "e", 1, seed=1)
    row["before_features"]["mean"] = "not-a-number"
    row["after_features"]["std"] = {"malformed": True}
    write_jsonl(log, [row])

    pairs, stats = extract_pairs(log_path=log, out_path=out)

    assert pairs == []
    assert stats["skipped"] == 1


def test_extracts_human_verdict_rows_and_recovers_before_png_after_png(tmp_path):
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    Image.fromarray(np.full((2, 2, 3), 10, dtype=np.uint8)).save(before)
    Image.fromarray(np.full((2, 2, 3), 20, dtype=np.uint8)).save(after)
    log = tmp_path / "preferences.jsonl"
    out = tmp_path / "pairs.jsonl"
    row = {
        "session_id": "s", "step_id": 1,
        "cmd_dict": {"attribute": "opacity", "target": "bone", "direction": "increase"},
        "features_before": None, "features_after": None,
        "before_png": str(before), "after_png": str(after),
        "human_verdict": "better",
    }
    write_jsonl(log, [row])

    pairs, _ = extract_pairs(log_path=log, feedback_path=None, out_path=out)

    assert len(pairs) == 1
    assert pairs[0]["source"] == "thumbs"
    assert pairs[0]["label"] == 1
    assert pairs[0]["observation_a"]["before_features"]["mean"] == 20.0


def test_extracts_inline_human_verdict_from_preferences_as_thumbs(tmp_path):
    preferences = tmp_path / "preferences.jsonl"
    out = tmp_path / "pairs.jsonl"
    row = step("s", "e", 2, seed=2)
    row["human_verdict"] = "worse"
    write_jsonl(preferences, [row])

    pairs, stats = extract_pairs(
        log_path=None,
        feedback_path=None,
        preferences_path=preferences,
        out_path=out,
    )

    assert len(pairs) == 1
    assert pairs[0]["source"] == "thumbs"
    assert pairs[0]["label"] == -1
    assert stats["thumbs"] == 1


def test_extract_pairs_can_use_preferences_default(tmp_path, monkeypatch):
    preferences = tmp_path / "preferences.jsonl"
    out = tmp_path / "pairs.jsonl"
    write_jsonl(preferences, [])
    monkeypatch.setattr("rl.extract_pairs.DEFAULT_PREFERENCES", str(preferences))

    pairs, stats = extract_pairs(log_path=None, feedback_path=None, out_path=out)

    assert pairs == []
    assert stats["total"] == 0


def test_default_parser_path_defines_preferences_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["rl.extract_pairs"])

    import rl.extract_pairs as extract_module
    extract_module.main()

    assert (tmp_path / "out" / "pairs.jsonl").read_text() == ""
