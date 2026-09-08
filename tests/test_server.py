"""Tests exercise Session directly, in-process, on pytest's own (main) thread.

FastAPI's TestClient runs the ASGI app on a background portal thread, and VTK's
Cocoa render window can only be created on the true process main thread on macOS
-- routing it through TestClient crashes the interpreter. Session has no FastAPI
dependency, so calling it directly sidesteps that entirely; the HTTP routing layer
itself is thin enough to verify by running the real server (see the plan's Task 6).
"""
import os

import numpy as np
import pytest

from server import Session

TEST_SESSION_PATH = "/tmp/test_ui_session.json"


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # Session now saves rendered PNGs to out/ui_images/<session_id>/ (relative
    # to cwd) on every command -- run every test in this module from a scratch
    # directory so that never touches the project's real out/ directory.
    monkeypatch.chdir(tmp_path)


def _fresh_session():
    if os.path.exists(TEST_SESSION_PATH):
        os.remove(TEST_SESSION_PATH)
    return Session(TEST_SESSION_PATH)


def test_initial_state_has_one_step_at_cursor_zero():
    s = _fresh_session()
    state = s.state()
    assert state["cursor"] == 0
    assert state["total"] == 1
    assert state["current"]["cmd_text"] is None
    assert len(state["current"]["params"]) == 24


def test_command_appends_a_step():
    s = _fresh_session()
    state = s.command("increase opacity for bone strongly", parser="rule", search=False)
    assert state["total"] == 2
    assert state["cursor"] == 1
    assert state["current"]["cmd_dict"]["target"] == "bone"
    assert state["current"]["image_b64"]


def test_unparseable_command_raises_and_does_not_append():
    s = _fresh_session()
    with pytest.raises(ValueError):
        s.command("make the skeleton pop", parser="rule", search=False)
    assert s.state()["total"] == 1


def test_back_and_forward_move_cursor():
    s = _fresh_session()
    s.command("increase opacity for bone strongly", parser="rule", search=False)
    assert s.back()["cursor"] == 0
    assert s.forward()["cursor"] == 1


def test_new_command_after_going_back_truncates_forward_history():
    s = _fresh_session()
    s.command("increase opacity for bone strongly", parser="rule", search=False)
    s.back()
    s.command("increase opacity for fat strongly", parser="rule", search=False)
    state = s.state()
    assert state["total"] == 2
    assert state["current"]["cmd_dict"]["target"] == "fat"


def test_objective_search_appends_one_final_step():
    s = _fresh_session()
    state = s.command("increase opacity for bone strongly", parser="rule",
                       search=True, evaluator="objective", steps=5)
    assert state["total"] == 2  # one new step, not one per iteration
    assert state["current"]["search"] is True
    assert state["current"]["masses"]["bone"] > 220.0


def test_policy_search_raises_when_no_trained_model():
    # _isolate_cwd (autouse) has already chdir'd to a scratch tmp_path, so
    # out/rl_models/sac_tf_agent.zip does not exist here even if the real
    # repo has a trained model on disk -- this is the fresh-checkout/CI path.
    s = _fresh_session()
    with pytest.raises(ValueError, match="no trained model"):
        s.command("increase opacity for bone strongly", parser="rule",
                   search=True, evaluator="policy", steps=3)
    assert s.state()["total"] == 1  # nothing appended on failure


def test_policy_search_appends_one_final_step(monkeypatch):
    import server as server_module
    from commands import _find_or_create_peak

    def fake_run_policy(params, tissue, direction, idx, steps):
        moved = params.copy()
        moved[idx * 3 + 2] = min(1.0, moved[idx * 3 + 2] + 0.1)
        return moved

    monkeypatch.setattr(server_module, "run_policy", fake_run_policy)

    s = _fresh_session()
    before_params = s.state()["current"]["params"]
    _, idx = _find_or_create_peak(np.array(before_params, dtype=np.float64), "bone")
    expected_height = min(1.0, before_params[idx * 3 + 2] + 0.1)

    state = s.command("increase opacity for bone strongly", parser="rule",
                       search=True, evaluator="policy", steps=5)
    assert state["total"] == 2  # one new step, not one per iteration
    assert state["current"]["search"] is True
    assert state["current"]["params"][idx * 3 + 2] == pytest.approx(expected_height)


def test_human_search_returns_pending_and_judge_advances_it():
    s = _fresh_session()
    state = s.command("increase opacity for bone strongly", parser="rule",
                       search=True, evaluator="human", steps=5)
    assert state["pending"] is not None
    assert state["pending"]["before_image_b64"]
    assert state["pending"]["after_image_b64"]
    assert state["total"] == 1  # not appended yet

    state2 = s.judge("better")
    assert state2["pending"] is not None or state2["total"] == 2


def test_human_search_converges_and_appends_final_step():
    s = _fresh_session()
    s.command("increase opacity for bone strongly", parser="rule",
              search=True, evaluator="human", steps=2)
    s.judge("better")
    state = s.judge("worse")
    assert state["pending"] is None
    assert state["total"] == 2
    assert state["current"]["search"] is True


def test_search_requested_for_non_opacity_attribute_falls_back_to_direct_apply():
    # "sharpen bone" parses to attribute="width", direction="decrease". search.propose_step
    # is hardcoded to mutate the height/opacity parameter, so search must not run for width
    # (or brightness/center) commands even when search=True is requested -- the command
    # should still be applied directly, just without the hill-climbing loop.
    s = _fresh_session()
    state = s.command("sharpen bone", parser="rule", search=True, evaluator="objective", steps=5)
    assert state["current"]["cmd_dict"]["attribute"] == "width"
    assert state["current"]["search"] is False
    assert state["pending"] is None


def test_judge_without_pending_raises():
    s = _fresh_session()
    with pytest.raises(ValueError):
        s.judge("better")


def test_preferences_logged_with_both_verdicts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("out", exist_ok=True)
    s = Session(str(tmp_path / "session.json"))
    s.command("increase opacity for bone strongly", parser="rule",
              search=True, evaluator="human", steps=1)
    s.judge("worse")

    import json
    lines = (tmp_path / "out" / "preferences.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["human_verdict"] == "worse"
    assert entry["objective_verdict"] in ("better", "worse")
    assert entry["cmd_dict"]["target"] == "bone"


def test_switch_dataset_resets_history():
    s = _fresh_session()
    s.command("increase opacity for bone strongly", parser="rule", search=False)
    assert s.state()["total"] == 2

    state = s.switch_dataset("synthetic")
    assert state["total"] == 1
    assert state["cursor"] == 0
    assert state["current"]["cmd_text"] is None
    assert state["dataset"] == "synthetic"


def test_switch_dataset_unknown_name_raises_and_leaves_history_alone():
    s = _fresh_session()
    s.command("increase opacity for bone strongly", parser="rule", search=False)
    with pytest.raises(ValueError):
        s.switch_dataset("not_a_real_dataset")
    assert s.state()["total"] == 2  # untouched


def test_feedback_sets_step_field_and_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("out", exist_ok=True)
    s = Session(str(tmp_path / "session.json"))
    s.command("increase opacity for bone strongly", parser="rule", search=False)
    step_id = s.state()["current"]["id"]

    state = s.feedback(step_id, "up")
    assert state["current"]["feedback"] == "up"

    import json
    lines = (tmp_path / "out" / "feedback.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["rating"] == "up"
    assert entry["step_id"] == step_id
    assert entry["cmd_dict"]["target"] == "bone"
    assert entry["session_id"] == s.session_id


def test_feedback_invalid_rating_raises():
    s = _fresh_session()
    step_id = s.state()["current"]["id"]
    with pytest.raises(ValueError):
        s.feedback(step_id, "sideways")


def test_feedback_unknown_step_id_raises():
    s = _fresh_session()
    with pytest.raises(ValueError):
        s.feedback(99999, "up")


def test_session_id_persists_across_reload():
    s = _fresh_session()
    original_id = s.session_id
    s2 = Session(TEST_SESSION_PATH)
    assert s2.session_id == original_id


def test_command_saves_image_file_matching_image_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("out", exist_ok=True)
    s = Session(str(tmp_path / "session.json"))
    state = s.command("increase opacity for bone strongly", parser="rule", search=False)

    image_path = state["current"]["image_path"]
    assert os.path.exists(image_path)
    with open(image_path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_human_search_pair_images_saved_and_referenced_in_preferences(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("out", exist_ok=True)
    s = Session(str(tmp_path / "session.json"))
    s.command("increase opacity for bone strongly", parser="rule",
              search=True, evaluator="human", steps=1)
    s.judge("better")

    import json
    lines = (tmp_path / "out" / "preferences.jsonl").read_text().strip().splitlines()
    entry = json.loads(lines[0])
    assert os.path.exists(entry["before_png"])
    assert os.path.exists(entry["after_png"])


def test_camera_command_does_not_change_params():
    session = _fresh_session()
    before_params = session.history[session.cursor]["params"]
    session.command("rotate right")
    after_params = session.history[session.cursor]["params"]
    assert before_params == after_params


def test_camera_command_changes_camera_state():
    session = _fresh_session()
    before_camera = session.history[session.cursor]["camera"]
    session.command("rotate right")
    after_camera = session.history[session.cursor]["camera"]
    assert after_camera["azimuth"] != before_camera["azimuth"]


def test_camera_state_is_per_step_and_restored_by_back():
    session = _fresh_session()
    session.command("rotate right")
    rotated_camera = session.history[session.cursor]["camera"]
    session.back()
    original_camera = session.history[session.cursor]["camera"]
    assert original_camera != rotated_camera
    session.forward()
    assert session.history[session.cursor]["camera"] == rotated_camera


def test_new_session_starts_with_default_camera():
    from camera import DEFAULT_CAMERA
    session = _fresh_session()
    assert session.history[0]["camera"] == DEFAULT_CAMERA


def test_camera_persists_through_human_search_convergence():
    session = _fresh_session()
    session.command("rotate right")
    rotated_camera = session.history[session.cursor]["camera"]
    session.command("increase opacity for bone strongly", parser="rule",
                     search=True, evaluator="human", steps=2)
    assert session.pending["camera"] == rotated_camera
    session.judge("better")
    state = session.judge("worse")
    assert state["current"]["camera"] == rotated_camera


def test_camera_change_during_pending_judgment_does_not_corrupt_final_camera():
    session = _fresh_session()
    session.command("rotate right")
    rotated_camera = session.history[session.cursor]["camera"]
    session.command("increase opacity for bone strongly", parser="rule",
                     search=True, evaluator="human", steps=2)
    # Nothing locks out further camera commands while a judgment is pending --
    # rotating again here must not corrupt the camera the human actually judged
    # against (rotated_camera), even though it becomes the new cursor's camera.
    session.command("rotate left")
    second_camera = session.history[session.cursor]["camera"]
    assert second_camera != rotated_camera
    session.judge("better")
    state = session.judge("worse")
    assert state["current"]["camera"] == rotated_camera
