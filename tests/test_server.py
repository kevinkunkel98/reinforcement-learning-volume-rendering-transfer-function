"""Tests exercise Session directly, in-process, on pytest's own (main) thread.

FastAPI's TestClient runs the ASGI app on a background portal thread, and VTK's
Cocoa render window can only be created on the true process main thread on macOS
-- routing it through TestClient crashes the interpreter. Session has no FastAPI
dependency, so calling it directly sidesteps that entirely; the HTTP routing layer
itself is thin enough to verify by running the real server (see the plan's Task 6).
"""
import os
import asyncio
import time
import json

import numpy as np
import pytest

from fastapi import HTTPException
from fastapi.responses import Response

import goals
import server
import transfer
import visibility
from rl.baselines import CONTROLLABLE
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


def _scene(scene_id, parent_scene_id):
    return {
        "scene_id": scene_id,
        "parent_scene_id": parent_scene_id,
        "session_id": "session-1",
        "client": "web",
        "dataset": "synthetic",
        "dataset_version": "synthetic-v1",
        "volume": {
            "dimensions": [2, 2, 2],
            "spacing": [1, 1, 1],
            "scalar_type": "float32",
            "orientation": "dataset-normalized",
        },
        "transfer_function": [0] * transfer.TOTAL_PARAMS,
        "camera": {
            "position": [0, 0, 1],
            "focal_point": [0, 0, 0],
            "view_up": [0, 1, 0],
            "zoom": 1,
        },
        "command": {"attribute": "camera"},
    }


def test_dataset_metadata_route_returns_transport_metadata(monkeypatch):
    metadata = {
        "name": "synthetic",
        "version": "synthetic-v1",
        "dimensions": [2, 2, 2],
        "spacing": [1.0, 1.0, 1.0],
        "scalar_type": "float32",
        "chunk_count": 1,
        "chunks": [{"index": 0, "byte_length": 32}],
    }
    monkeypatch.setattr(server, "dataset_metadata", lambda name: metadata)

    result = asyncio.run(server.dataset_metadata_route("synthetic"))

    assert result == metadata


def test_dataset_metadata_route_maps_unknown_dataset_to_404(monkeypatch):
    def unknown(name):
        raise ValueError("unknown dataset 'missing'")

    monkeypatch.setattr(server, "dataset_metadata", unknown)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.dataset_metadata_route("missing"))
    assert exc.value.status_code == 404
    assert "unknown dataset" in exc.value.detail


def test_dataset_metadata_route_maps_existing_dataset_validation_to_422(monkeypatch):
    monkeypatch.setattr(server, "list_datasets", lambda: ["synthetic"])
    monkeypatch.setattr(server, "dataset_metadata", lambda name: (_ for _ in ()).throw(
        ValueError("spacing must contain three positive finite values")))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.dataset_metadata_route("synthetic"))
    assert exc.value.status_code == 422
    assert "spacing" in exc.value.detail


def test_dataset_metadata_route_maps_loader_failure_to_500(monkeypatch):
    monkeypatch.setattr(server, "list_datasets", lambda: ["synthetic"])
    monkeypatch.setattr(server, "dataset_metadata", lambda name: (_ for _ in ()).throw(
        OSError("dataset file is unreadable")))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.dataset_metadata_route("synthetic"))
    assert exc.value.status_code == 500
    assert "unreadable" in exc.value.detail


def test_dataset_chunk_route_returns_binary_response(monkeypatch):
    monkeypatch.setattr(server, "get_volume_chunk", lambda name, index: b"\x00\x01")

    result = asyncio.run(server.dataset_chunk("synthetic", 0))

    assert isinstance(result, Response)
    assert result.media_type == "application/octet-stream"
    assert result.body == b"\x00\x01"
    assert result.headers["content-type"] == "application/octet-stream"
    assert result.headers["content-length"] == "2"


def test_dataset_chunk_route_maps_bad_index_to_400(monkeypatch):
    def bad_index(name, index):
        raise IndexError("chunk index out of range: 3")

    monkeypatch.setattr(server, "get_volume_chunk", bad_index)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.dataset_chunk("synthetic", 3))
    assert exc.value.status_code == 422
    assert "out of range" in exc.value.detail


def test_dataset_chunk_route_maps_non_integer_index_to_400():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.dataset_chunk("synthetic", "not-an-index"))
    assert exc.value.status_code == 422
    assert "integer" in exc.value.detail


def test_scene_transition_route_normalizes_and_appends_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = _scene("scene-0", None)
    after = _scene("scene-1", "scene-0")

    result = asyncio.run(server.scene_transition_route({"before": before, "after": after}))

    assert result["scene_id"] == "scene-1"
    assert result["parent_scene_id"] == "scene-0"
    lines = (tmp_path / "out" / "scene_transitions.jsonl").read_text().splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["after_scene"] == result
    assert logged["before_scene"] == before
    assert logged["event_id"]
    assert logged["dedupe_key"]


def test_scene_transition_route_is_idempotent_by_event_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = _scene("scene-0", None)
    after = _scene("scene-1", "scene-0")
    payload = {"before": before, "after": after, "event_id": "event-1", "dedupe_key": "dedupe-1"}

    first = asyncio.run(server.scene_transition_route(payload))
    second = asyncio.run(server.scene_transition_route(payload))

    assert first == second
    assert len((tmp_path / "out" / "scene_transitions.jsonl").read_text().splitlines()) == 1


def test_scene_transition_route_rejects_conflicting_event_id_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = _scene("scene-0", None)
    after = _scene("scene-1", "scene-0")
    asyncio.run(server.scene_transition_route({"before": before, "after": after, "event_id": "event-1"}))

    conflicting = _scene("scene-2", "scene-0")
    with pytest.raises(HTTPException, match="event_id"):
        asyncio.run(server.scene_transition_route({"before": before, "after": conflicting, "event_id": "event-1"}))


@pytest.mark.parametrize("field", ["event_id", "dedupe_key"])
def test_scene_transition_route_rejects_empty_persistence_keys(tmp_path, monkeypatch, field):
    monkeypatch.chdir(tmp_path)
    before = _scene("scene-0", None)
    after = _scene("scene-1", "scene-0")

    with pytest.raises(HTTPException, match=field):
        asyncio.run(server.scene_transition_route({"before": before, "after": after, field: "  "}))


def test_dataset_boundary_requires_root_neutral_scene(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scene = _scene("dataset-root", "old-scene")
    scene["command"] = {"attribute": "neutral", "kind": "dataset_boundary"}

    with pytest.raises(HTTPException, match="parent_scene_id"):
        asyncio.run(server.scene_transition_route({"after": scene, "boundary": True}))


def test_dataset_boundary_accepts_neutral_command_in_scene(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scene = _scene("dataset-root", None)
    scene["command"] = {"attribute": "neutral", "kind": "dataset_boundary"}

    result = asyncio.run(server.scene_transition_route({"after": scene, "boundary": True,
                                                        "event_id": "boundary-1", "dedupe_key": "boundary-1"}))

    assert result["command"] == {"attribute": "neutral", "kind": "dataset_boundary"}


def test_scene_transition_route_rejects_self_transition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scene = _scene("scene-0", None)

    with pytest.raises(HTTPException, match="distinct"):
        asyncio.run(server.scene_transition_route({"before": scene, "after": scene}))


def test_scene_transition_route_adds_extractor_metadata_and_preserves_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = _scene("scene-0", None)
    before.update({"step_id": 0, "features_after": {"mean": 1, "std": 2, "coverage": 3, "entropy": 4}})
    after = _scene("scene-1", "scene-0")
    after.update({
        "step_id": 1,
        "parent_step_id": 0,
        "carried_forward": True,
        "accepted": True,
        "ended": False,
        "features_before": {"mean": 1, "std": 2, "coverage": 3, "entropy": 4},
        "features_after": {"mean": 5, "std": 6, "coverage": 7, "entropy": 8},
    })

    result = asyncio.run(server.scene_transition_route({"before": before, "after": after}))

    assert result["dataset"] == "synthetic"
    assert result["dataset_version"] == "synthetic-v1"
    assert result["camera"] == after["camera"]
    assert result["parent_step_id"] == 0
    assert result["carried_forward"] is True
    assert result["accepted"] is True
    assert result["step_id"] == 1


def test_scene_transition_route_rejects_mismatched_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = _scene("scene-0", None)
    after = _scene("scene-1", "other-scene")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.scene_transition_route({"before": before, "after": after}))
    assert exc.value.status_code == 400
    assert "parent_scene_id" in exc.value.detail


def test_initial_state_has_one_step_at_cursor_zero():
    s = _fresh_session()
    state = s.state()
    assert state["cursor"] == 0
    assert state["total"] == 1
    assert state["current"]["cmd_text"] is None
    assert len(state["current"]["params"]) == transfer.TOTAL_PARAMS


def test_render_step_includes_a_transfer_curve():
    """The histogram/curve visual guide (README, "the hard part") needs the
    opacity+colour curve sampled over the HU range on every step, computed
    fresh from that step's own params -- not the volume, so it works even for
    datasets with no visibility model."""
    s = _fresh_session()
    curve = s.state()["current"]["curve"]
    assert len(curve["hu"]) == server.CURVE_SAMPLES
    assert len(curve["opacity"]) == server.CURVE_SAMPLES
    assert len(curve["rgb"]) == server.CURVE_SAMPLES
    assert all(0.0 <= o <= 1.0 for o in curve["opacity"])
    assert all(len(rgb) == 3 for rgb in curve["rgb"])
    assert all(0.0 <= c <= 1.0 for rgb in curve["rgb"] for c in rgb)


def test_render_step_includes_the_volume_histogram():
    s = _fresh_session()
    histogram = s.state()["current"]["histogram"]
    assert len(histogram["counts"]) == visibility.HISTOGRAM_BINS
    assert histogram["range"] == list(visibility.CENTER_RANGE)
    assert abs(sum(histogram["counts"]) - 1.0) < 1e-6


def test_state_and_steps_have_no_judgment_fields():
    s = _fresh_session()
    state = s.command("increase opacity for bone strongly", parser="rule", search=False)
    assert "pending" not in state
    assert "verdict" not in state["current"]
    assert "feedback" not in state["current"]
    assert not hasattr(s, "judge")
    assert not hasattr(s, "feedback")


def test_command_appends_a_step():
    s = _fresh_session()
    state = s.command("increase opacity for bone strongly", parser="rule", search=False)
    assert state["total"] == 2
    assert state["cursor"] == 1
    assert state["current"]["cmd_dict"]["target"] == "skeleton"
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
    s.command("increase opacity for lungs strongly", parser="rule", search=False)
    state = s.state()
    assert state["total"] == 2
    assert state["current"]["cmd_dict"]["target"] == "lungs"


def test_search_appends_one_final_step():
    s = _fresh_session()
    state = s.command("increase opacity for bone strongly", parser="rule",
                       search=True, steps=5)
    assert state["total"] == 2  # one new step, not one per iteration
    assert state["current"]["search"] is True
    assert state["current"]["masses"]["bone"] > 220.0


def test_search_keeps_current_camera():
    s = _fresh_session()
    s.command("rotate right")
    cam = s.history[s.cursor]["camera"]
    state = s.command("increase opacity for bone strongly", parser="rule", search=True, steps=3)
    assert state["current"]["search"] is True
    assert state["current"]["camera"] == cam


def test_search_requested_for_non_opacity_attribute_falls_back_to_direct_apply():
    # "sharpen bone" parses to attribute="width", direction="decrease", which
    # goals.goal_from_command does not recognize as a goal (only show_only,
    # opacity and brightness commands are) -- so search=True still falls back
    # to exact application, just via the goal-construction ValueError rather
    # than the old opacity/direction-only gate.
    s = _fresh_session()
    state = s.command("sharpen bone", parser="rule", search=True, steps=5)
    assert state["current"]["cmd_dict"]["attribute"] == "width"
    assert state["current"]["search"] is False


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


# --- Task 3: policy mode -----------------------------------------------------
# mode="policy" builds a goal from the parsed command (goals.goal_from_command)
# and runs the trained one-shot policy on it; a stub in place of the real SB3
# checkpoint keeps these tests fast and deterministic. Camera commands bypass
# the whole mechanism (handled earlier in `command()`, same as every other
# mode). When the policy is unavailable, or the command isn't a goal at all,
# it falls back to exact application and records why.

class _StubPolicy:
    """Stands in for an SB3 model: same `.predict(obs, deterministic=...)`
    interface, always returns the same fixed action."""

    def __init__(self, action):
        self._action = np.asarray(action, dtype=np.float64)

    def predict(self, observation, deterministic=True):
        return self._action, None


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _use_real_totalseg_manifest(monkeypatch):
    """`_isolate_cwd` chdirs into a scratch tmp_path so these tests never write
    into the real project's out/; `totalseg`'s manifest and volume/label paths
    are repo-root-relative, so patch `totalseg._subjects` -- the one function
    `subject`, `has_labels`, `classes_present`, `is_contrast`,
    `available_names` and `split_names` all read -- to resolve them against
    the real repo root, for the one real TotalSegmentator volume
    (`ts_s1379`) the policy-mode tests below exercise.

    Patching `subject` alone (an earlier version of this fixture) left
    `has_labels` reading `_subjects()` directly: from a chdir'd tmp_path it
    always found no manifest and reported no labels, which silently switched
    `visibility.for_volume` onto its intensity-only fallback -- under which
    `vessels` has no ceiling at all, invisible unless a test happens to ask
    about vessels (e.g. via "show only"). It also left `available_names`/
    `split_names` returning `[]`, so `datasets.list_datasets()` omitted every
    ts_* volume inside `ts_session`. Patching `_subjects` fixes the whole
    family at once, and `subject()` keeps raising `ValueError` on an unknown
    name (its own, un-patched behaviour) rather than a `KeyError`."""
    import totalseg
    with open(os.path.join(_REPO_ROOT, "data/totalseg_manifest.json")) as f:
        subjects = json.load(f)["subjects"]

    def _absolute(entry):
        entry = dict(entry)
        entry["path"] = os.path.join(_REPO_ROOT, entry["path"])
        if entry.get("labels_path"):
            entry["labels_path"] = os.path.join(_REPO_ROOT, entry["labels_path"])
        return entry

    resolved = {s["name"]: _absolute(s) for s in subjects}
    monkeypatch.setattr(totalseg, "_subjects", lambda: resolved)


@pytest.fixture
def ts_session(monkeypatch):
    """A fresh Session switched to the real `ts_s1379` TotalSegmentator
    volume -- the one dataset these policy-mode tests can run a real goal
    check against (see `_use_real_totalseg_manifest`). `server._dataset_name`
    is module-level global state that outlives any one test, so this restores
    whatever dataset was active beforehand once the test is done, regardless
    of outcome."""
    _use_real_totalseg_manifest(monkeypatch)
    original_dataset = server._dataset_name
    s = _fresh_session()
    s.switch_dataset("ts_s1379")
    try:
        yield s
    finally:
        server.set_dataset(original_dataset)


def test_policy_mode_applies_the_policys_action(ts_session):
    s = ts_session
    # Stays clear of ±1 so no colour channel saturates: a peak's channels sit
    # at their hue offsets around the group mean, and clipping one of them
    # would shift that mean off the action value (see the clipping test).
    action = np.linspace(-0.4, 0.4, len(CONTROLLABLE))
    s.policy_provider = lambda: _StubPolicy(action)

    before_params = np.array(s.history[s.cursor]["params"], dtype=np.float64)
    state = s.command("more bone", mode="policy")

    assert state["current"]["mode"] == "policy"
    assert state["current"]["message"] is None
    params = np.array(state["current"]["params"], dtype=np.float64)
    for group, value in zip(CONTROLLABLE, action):
        # Each group lands on the action value as its mean; the (r, g, b)
        # group keeps its channel offsets so the render stays coloured.
        assert float(np.mean([params[i] for i in group])) == pytest.approx(float(value), abs=1e-6)
    controllable_indices = {i for group in CONTROLLABLE for i in group}
    for i in range(len(before_params)):
        if i not in controllable_indices:
            assert params[i] == pytest.approx(before_params[i])  # centres untouched


def test_policy_mode_camera_command_still_moves_camera(ts_session):
    s = ts_session
    s.policy_provider = lambda: _StubPolicy(np.zeros(len(CONTROLLABLE)))
    before_camera = s.history[s.cursor]["camera"]

    state = s.command("rotate right", mode="policy")

    assert state["current"]["camera"]["azimuth"] != before_camera["azimuth"]
    assert state["current"]["mode"] == "camera"


def test_policy_mode_without_checkpoint_falls_back_to_exact_and_says_so():
    s = _fresh_session()
    # The default policy_provider looks for out/rl_v2/oneshot_v3_seed0/best.zip
    # relative to cwd; this test's cwd (tmp_path, via _isolate_cwd) has none.
    state = s.command("increase opacity for bone strongly", mode="policy")

    assert state["current"]["mode"] == "exact"
    assert state["current"]["message"]
    assert state["current"]["cmd_dict"]["target"] == "skeleton"
    # falls back to exactly what apply_command would have produced
    direct = _fresh_session()
    direct_state = direct.command("increase opacity for bone strongly", mode="exact")
    assert state["current"]["params"] == direct_state["current"]["params"]


def test_policy_mode_non_goal_command_falls_back_with_message(ts_session):
    s = ts_session
    s.policy_provider = lambda: _StubPolicy(np.zeros(len(CONTROLLABLE)))

    state = s.command("reset", mode="policy")

    assert state["current"]["mode"] == "exact"
    assert state["current"]["message"]


def test_run_policy_arm_matches_session_run_policy(ts_session):
    """`Session._run_policy` is a thin delegation to `run_policy_arm` -- call
    both with the same inputs and pin that they cannot drift apart."""
    s = ts_session
    action = np.linspace(-0.4, 0.4, len(CONTROLLABLE))
    s.policy_provider = lambda: _StubPolicy(action)
    cmd, _ = server.parse_command_with_meta("more bone", parser="rule")
    params = np.array(s.history[s.cursor]["params"], dtype=np.float64)

    model = s.model_for_volume(server._dataset_name)
    policy = s.policy_provider()
    start_agg = goals.aggregate(model.features(params))
    instruction = goals.goal_from_command(cmd, model, start_agg, volume=server._dataset_name)

    delegated_params, delegated_text = s._run_policy(cmd, params.copy())
    direct_params = server.run_policy_arm(model, policy, instruction, start_agg, params.copy())

    assert delegated_params.tolist() == pytest.approx(direct_params.tolist())
    assert delegated_text == instruction["text"]
    for group, value in zip(CONTROLLABLE, action):
        assert float(np.mean([direct_params[i] for i in group])) == pytest.approx(float(value), abs=1e-6)


def test_run_policy_arm_without_a_checkpoint_raises_the_user_visible_message(ts_session):
    s = ts_session
    model = s.model_for_volume(server._dataset_name)
    params = np.array(s.history[s.cursor]["params"], dtype=np.float64)
    start_agg = goals.aggregate(model.features(params))
    cmd, _ = server.parse_command_with_meta("more bone", parser="rule")
    instruction = goals.goal_from_command(cmd, model, start_agg, volume=server._dataset_name)

    with pytest.raises(ValueError) as exc:
        server.run_policy_arm(model, None, instruction, start_agg, params)

    # This message ends up in the step's user-visible `message` field, so its
    # exact wording matters, not just that some ValueError was raised. The
    # literal is intentional here (not `server.NO_POLICY_CHECKPOINT_MESSAGE`)
    # -- comparing the code to itself would keep this test green through an
    # accidental wording change.
    assert str(exc.value) == "no trained policy checkpoint found -- applied the command directly instead"


def test_policy_mode_falls_back_to_exact_when_the_volume_has_no_visibility_cache():
    """Regression: `model_for_volume` (`visibility.for_volume`) is not total --
    it raises `FileNotFoundError` for a volume with no visibility cache -- and
    `Session.command` only catches `ValueError`. The no-checkpoint check must
    run before `model_for_volume` is ever called, or this propagates instead
    of falling back to exact application."""
    if os.path.exists(TEST_SESSION_PATH):
        os.remove(TEST_SESSION_PATH)

    def _raise(name):
        raise FileNotFoundError(f"no visibility cache for {name}")

    s = Session(TEST_SESSION_PATH, policy_provider=lambda: None, model_for_volume=_raise)

    state = s.command("more bone", mode="policy")

    assert state["current"]["mode"] == "exact"
    assert state["current"]["message"]


def test_search_mode_searches_on_a_non_opacity_instruction(ts_session, monkeypatch):
    """The old objective search only fired for opacity increase/decrease, so
    "show only the lungs" silently fell through to exact application while the
    UI still showed search as active. It must now actually search."""
    s = ts_session
    calls = {}

    def _spy(model, start_params, instruction, evaluations=200, initial_step=0.2):
        calls["evaluations"] = evaluations
        return np.asarray(start_params, dtype=np.float64).copy()

    monkeypatch.setattr(server, "hill_climb", _spy)

    state = s.command("show only the lungs", mode="search", steps=10)

    assert calls["evaluations"] == 10
    assert state["current"]["mode"] == "search"


def test_search_mode_falls_back_to_exact_when_the_volume_has_no_visibility_cache():
    """The search twin of
    test_policy_mode_falls_back_to_exact_when_the_volume_has_no_visibility_cache:
    `model_for_volume` (`visibility.for_volume`) is not total -- it raises
    `FileNotFoundError` for a volume with no visibility cache -- and the old
    search branch caught only `ValueError`, so this escaped as an unhandled
    exception (a 500) instead of falling back to exact application."""
    if os.path.exists(TEST_SESSION_PATH):
        os.remove(TEST_SESSION_PATH)

    def _raise(name):
        raise FileNotFoundError(f"no visibility cache for {name}")

    s = Session(TEST_SESSION_PATH, model_for_volume=_raise)

    state = s.command("increase opacity for bone strongly", mode="search", steps=10)

    assert state["current"]["mode"] == "exact"
    assert state["current"]["message"]


def test_search_mode_reports_when_the_budget_is_too_small_to_move():
    """`hill_climb` spends 1 evaluation scoring the start state and 1 per
    proposal; a full coordinate sweep is 12 groups x 2 signs = 24, so a
    budget below ~25 (the UI's `#steps-input` allows as little as 1) can
    return the start state completely untouched. `search` must say so rather
    than report mode="search" with no visible effect and no explanation --
    the exact failure this task exists to eliminate. Uses the default
    dataset's intensity model (not `ts_session`): whether 5 evaluations
    happen to find an improving move is data-dependent, and this combination
    is pinned to not move."""
    s = _fresh_session()

    state = s.command("increase opacity for bone strongly", mode="search", steps=5)

    assert state["current"]["mode"] == "search"
    assert state["current"]["message"] == (
        "search spent 5 evaluations without improving on the start; "
        "a full sweep needs about 25")


def test_search_mode_actually_reduces_distance_to_the_goal(ts_session):
    """A stubbed `hill_climb` (as in
    test_search_mode_searches_on_a_non_opacity_instruction) can return the
    start state unchanged and still look like a pass if the only assertion
    is a mass threshold `transfer.default_params()` already clears at rest.
    This drives the real hill-climb with a budget past a full sweep (~24
    evaluations) and checks it actually gets closer to the goal."""
    s = ts_session
    cmd, _ = server.parse_command_with_meta("more bone", parser="rule")
    current_params = np.array(s.history[s.cursor]["params"], dtype=np.float64)
    model = s.model_for_volume(server._dataset_name)
    start_agg = goals.aggregate(model.features(current_params))
    instruction = goals.goal_from_command(cmd, model, start_agg, volume=server._dataset_name)
    start_distance = goals.distance(instruction["goal"], start_agg, start_agg)

    state = s.command("more bone", mode="search", steps=25)

    new_params = np.array(state["current"]["params"], dtype=np.float64)
    new_agg = goals.aggregate(model.features(new_params))
    end_distance = goals.distance(instruction["goal"], start_agg, new_agg)
    assert end_distance < start_distance


# --- compare_arms: one command, four answers, one start ----------------------

def _compare_inputs(s):
    """The shared context `compare_arms` takes: model, start params, camera,
    parsed command, start aggregate and goal -- built once, exactly as the
    route will build them, so every arm is scored against the same start."""
    model = s.model_for_volume(server._dataset_name)
    params = np.array(s.history[s.cursor]["params"], dtype=np.float64)
    camera = dict(s.history[s.cursor]["camera"])
    cmd, _ = server.parse_command_with_meta("more bone", parser="rule")
    start_agg = goals.aggregate(model.features(params))
    instruction = goals.goal_from_command(cmd, model, start_agg, volume=server._dataset_name)
    return model, policy_stub(), cmd, instruction, params, camera


def policy_stub():
    return _StubPolicy(np.zeros(len(CONTROLLABLE)))


def test_compare_arms_runs_four_arms_from_the_same_start(ts_session):
    model, policy, cmd, instruction, params, camera = _compare_inputs(ts_session)

    arms = server.compare_arms(model, policy, cmd, instruction, params, camera,
                                cheap=4, thorough=8)

    assert set(arms) == {"exact", "search_cheap", "search_thorough", "policy"}
    assert arms["exact"]["evaluations"] == 0
    assert arms["search_cheap"]["evaluations"] == 4
    assert arms["search_thorough"]["evaluations"] == 8
    assert arms["policy"]["evaluations"] == 0
    for name, arm in arms.items():
        assert arm["attainment"] is not None, name
        assert set(arm["class_visibility"]) == set(goals.GOAL_CLASSES), name
        assert arm["image_b64"] and not arm["image_b64"].startswith("data:"), name
        assert arm["elapsed_ms"] >= 0, name
        assert arm["unchanged"] in (True, False), name


def test_compare_arms_scores_every_arm_against_the_shared_start(ts_session):
    """The point of one start state: an arm's attainment must be what it would
    be if that arm alone had answered, not something relative to another arm."""
    model, policy, cmd, instruction, params, camera = _compare_inputs(ts_session)
    start_agg = goals.aggregate(model.features(params))

    arms = server.compare_arms(model, policy, cmd, instruction, params, camera,
                                cheap=4, thorough=8)

    for name, arm in arms.items():
        replayed = np.array(arm["params"], dtype=np.float64)
        expected = goals.attainment(instruction["goal"], start_agg,
                                     goals.aggregate(model.features(replayed)))
        assert arm["attainment"] == pytest.approx(expected), name


def test_compare_arms_marks_an_arm_that_did_not_move(ts_session, monkeypatch):
    """An arm returning its own input is not an arm agreeing with the start --
    it is an arm that did not move, and the panel must not read those the same.
    At B3's budget the search arm can legitimately land here: 10 evaluations is
    less than half of one 24-evaluation coordinate sweep."""
    model, policy, cmd, instruction, params, camera = _compare_inputs(ts_session)
    monkeypatch.setattr(server, "run_search_arm",
                         lambda model, start_params, instruction, evaluations: start_params.copy())

    arms = server.compare_arms(model, policy, cmd, instruction, params, camera,
                                cheap=4, thorough=8)

    assert arms["search_cheap"]["unchanged"] is True
    assert arms["search_thorough"]["unchanged"] is True


def test_compare_arms_degrades_one_arm_without_killing_the_others(ts_session):
    """Losing one column must not end a live demo."""
    model, _policy, cmd, instruction, params, camera = _compare_inputs(ts_session)

    arms = server.compare_arms(model, None, cmd, instruction, params, camera,
                                cheap=4, thorough=8)

    assert arms["policy"]["unavailable"]
    assert arms["exact"]["attainment"] is not None
    assert arms["search_cheap"]["attainment"] is not None


def test_compare_arms_times_only_the_arm_not_the_render(ts_session, monkeypatch):
    """`elapsed_ms` is what the arm cost to answer, not what it cost to draw.

    The timing column exists to show that the policy answers for free while
    thorough search costs seconds. Rendering is the same price for every arm,
    so folding it in inflates the cheap arms towards the expensive one and
    makes the policy's number stop being the policy's number."""
    model, policy, cmd, instruction, params, camera = _compare_inputs(ts_session)
    real_render = server._render_image_b64

    def _slow_render(p, c):
        time.sleep(1.0)
        return real_render(p, c)

    monkeypatch.setattr(server, "_render_image_b64", _slow_render)

    arms = server.compare_arms(model, policy, cmd, instruction, params, camera,
                                cheap=4, thorough=8)

    # A full second of render per arm must not appear in any arm's cost.
    # For scale: a real render+encode is ~620 ms, while the policy arm's own
    # work is ~210 ms (dominated by the solo_max ceiling probes) and exact is
    # near-instant -- so folding the render in would have flattened three of
    # the four columns onto each other.
    assert arms["policy"]["elapsed_ms"] < 500
    assert arms["exact"]["elapsed_ms"] < 500


def test_compare_arms_gives_a_degraded_arm_the_same_keys(ts_session):
    """A consumer iterating the arms must not have to special-case a failed
    one before reading a field every other arm has."""
    model, policy, cmd, instruction, params, camera = _compare_inputs(ts_session)

    arms = server.compare_arms(model, None, cmd, instruction, params, camera,
                                cheap=4, thorough=8)

    # A failed arm carries the successful arm's keys plus its own two: the
    # reader-facing `unavailable` and the developer-facing `detail`.
    assert set(arms["policy"]) == set(arms["exact"]) | {"unavailable", "detail"}
    assert arms["policy"]["params"] is None


def test_compare_arms_does_not_leak_raw_exception_text_into_a_reason(ts_session, monkeypatch):
    """`str(exc)` is not user-facing copy: a KeyError stringifies to the bare
    repr of its key. In a column an examiner is reading closely, "'lungs'" is
    not an explanation."""
    model, policy, cmd, instruction, params, camera = _compare_inputs(ts_session)

    def _raise_key_error(*args, **kwargs):
        raise KeyError("lungs")

    monkeypatch.setattr(server, "run_search_arm", _raise_key_error)
    arms = server.compare_arms(model, policy, cmd, instruction, params, camera,
                                cheap=4, thorough=8)

    reason = arms["search_cheap"]["unavailable"]
    assert reason != "'lungs'"
    assert "visibility cache" in reason


def test_compare_arms_reports_brightness_beside_visibility(ts_session):
    """A brightness instruction moves the brightness channel, not the
    visibility one. Plotting only visibility makes four columns whose bars are
    near-identical while their attainment ranges from 0.00 to 0.94 -- the
    panel then contradicts itself."""
    model, policy, cmd, instruction, params, camera = _compare_inputs(ts_session)

    arms = server.compare_arms(model, policy, cmd, instruction, params, camera,
                                cheap=4, thorough=8)

    for name, arm in arms.items():
        assert set(arm["class_brightness"]) == set(goals.GOAL_CLASSES), name


def test_goal_channels_names_which_channel_the_instruction_targets():
    """The panel plots the channel the instruction is about."""
    vis_only = np.zeros(4 * len(goals.GOAL_CLASSES))
    vis_only[len(goals.GOAL_CLASSES) + goals.GOAL_CLASSES.index("skeleton")] = 1.0
    assert server.goal_channels(vis_only) == {"vis": ["skeleton"], "bright": []}

    bright_only = np.zeros(4 * len(goals.GOAL_CLASSES))
    bright_only[3 * len(goals.GOAL_CLASSES) + goals.GOAL_CLASSES.index("lungs")] = 1.0
    assert server.goal_channels(bright_only) == {"vis": [], "bright": ["lungs"]}


def test_compare_route_says_which_channel_to_plot(ts_session, monkeypatch):
    s = ts_session
    s.policy_provider = lambda: policy_stub()
    monkeypatch.setattr(server, "session", s)

    result = asyncio.run(server.compare_route(
        server.CompareRequest(text="brighten the skeleton", parser="rule", cheap=4, thorough=8)))

    assert result["applicable"] is True
    assert result["channels"]["bright"] == ["skeleton"]
    assert set(result["start"]["class_brightness"]) == set(goals.GOAL_CLASSES)


def test_compare_arms_warms_the_ceiling_before_timing_any_arm(ts_session, monkeypatch):
    """`model.solo_max` probes four single-peak transfer functions and memoises
    per model instance -- about 250 ms the first time, free afterwards. Run
    inside the policy arm's stopwatch it makes the first comparison of a
    session report the policy as slower than the cheap search it is supposed to
    beat, for a reason that has nothing to do with the policy. And the first
    comparison is the one a demo audience watches."""
    model, policy, cmd, instruction, params, camera = _compare_inputs(ts_session)
    model._solo_max.clear()
    warm_at_arm = {}

    real_arm = server.run_policy_arm

    def _spy(model_, policy_, instruction_, start_agg_, start_params_):
        warm_at_arm["classes"] = set(model_._solo_max)
        return real_arm(model_, policy_, instruction_, start_agg_, start_params_)

    monkeypatch.setattr(server, "run_policy_arm", _spy)
    server.compare_arms(model, policy, cmd, instruction, params, camera, cheap=4, thorough=8)

    measured = {m for c in goals.GOAL_CLASSES for m in goals.MEASURED_FOR_GOAL[c]}
    assert measured <= warm_at_arm["classes"]


def test_compare_arms_does_not_claim_a_fallback_it_did_not_perform(ts_session):
    """With no checkpoint the policy column is blank. Saying "applied the
    command directly instead" -- copy written for `Session.command`, which
    really does fall back -- makes the panel assert something false about its
    own behaviour, in the column a reader looks at hardest."""
    model, _policy, cmd, instruction, params, camera = _compare_inputs(ts_session)

    arms = server.compare_arms(model, None, cmd, instruction, params, camera,
                                cheap=4, thorough=8)

    reason = arms["policy"]["unavailable"]
    assert "applied the command directly" not in reason
    assert "checkpoint" in reason
    assert arms["policy"]["detail"]


def test_compare_arms_leaves_the_start_state_untouched(ts_session):
    """One array feeds four arms and is the reference for every `unchanged`
    flag. It survives today only because three independent callees happen to
    copy; an in-place clip in any of them would corrupt arms 2-4 silently."""
    model, policy, cmd, instruction, params, camera = _compare_inputs(ts_session)
    before = params.copy()

    server.compare_arms(model, policy, cmd, instruction, params, camera, cheap=4, thorough=8)

    assert np.array_equal(params, before)


def test_compare_arms_reports_a_scoring_failure_as_itself(ts_session, monkeypatch):
    """A render or scoring failure is not a missing visibility cache. Reporting
    one as the other is a confident, specific, wrong diagnosis with no
    traceback to follow."""
    model, policy, cmd, instruction, params, camera = _compare_inputs(ts_session)

    def _boom(*a, **k):
        raise KeyError("render exploded")

    monkeypatch.setattr(server, "_render_image_b64", _boom)

    with pytest.raises(KeyError):
        server.compare_arms(model, policy, cmd, instruction, params, camera,
                             cheap=4, thorough=8)


# --- sweep_arms: the distribution, not one draw -------------------------------

def test_sweep_arms_summarises_every_arm_over_many_instructions(ts_session):
    """One comparison is a single draw from a distribution; the thesis claim is
    a median over many. The sweep is what makes the panel show the claim."""
    s = ts_session
    model = s.model_for_volume(server._dataset_name)
    params = np.array(s.history[s.cursor]["params"], dtype=np.float64)

    result = server.sweep_arms(model, policy_stub(), server._dataset_name, params,
                                episodes=4, seed=0)

    assert set(result["arms"]) == {"exact", "search_cheap", "policy"}
    for name, arm in result["arms"].items():
        assert arm["n"] == 4, name
        assert arm["median"] is not None, name
        assert 0.0 <= arm["share_positive"] <= 1.0, name
    assert result["episodes"] == 4


def test_sweep_arms_is_deterministic_in_its_seed(ts_session):
    """A demo that gives different numbers each press is not evidence."""
    s = ts_session
    model = s.model_for_volume(server._dataset_name)
    params = np.array(s.history[s.cursor]["params"], dtype=np.float64)

    first = server.sweep_arms(model, policy_stub(), server._dataset_name, params, episodes=3, seed=7)
    second = server.sweep_arms(model, policy_stub(), server._dataset_name, params, episodes=3, seed=7)

    assert first["arms"]["policy"]["median"] == second["arms"]["policy"]["median"]


def test_sweep_arms_can_include_thorough_search_when_asked(ts_session):
    """Excluded by default: 200 evaluations x 20 episodes is about a minute."""
    s = ts_session
    model = s.model_for_volume(server._dataset_name)
    params = np.array(s.history[s.cursor]["params"], dtype=np.float64)

    result = server.sweep_arms(model, policy_stub(), server._dataset_name, params,
                                episodes=2, seed=0, thorough=4)

    assert "search_thorough" in result["arms"]
    assert result["arms"]["search_thorough"]["n"] == 2


def test_sweep_arms_handles_absolute_instructions(ts_session, monkeypatch):
    """`goals.sample_instruction` wants the *aggregated* features, keyed by goal
    class, not the raw per-material ones. Only the "absolute" branch indexes
    them by goal class, so passing the wrong shape survives most seeds and then
    raises KeyError('soft') on about one instruction in ten."""
    s = ts_session
    model = s.model_for_volume(server._dataset_name)
    params = np.array(s.history[s.cursor]["params"], dtype=np.float64)
    monkeypatch.setattr(goals, "_sample_kind", lambda rng: "absolute")
    # Raw features are keyed by MATERIAL (skeleton, lungs, organs, muscle,
    # vessels) and three goal classes share those names -- so the wrong shape
    # only raises when the sampler picks `soft`, the one goal class that has no
    # material of the same name. Force it.
    monkeypatch.setattr(goals, "reachable_goal_classes", lambda name, model: ["soft"])

    result = server.sweep_arms(model, policy_stub(), server._dataset_name, params,
                                episodes=2, seed=0)

    assert result["kinds"] == ["absolute"]
    assert result["arms"]["policy"]["n"] == 2


# --- the /api/compare route ---------------------------------------------------

def test_compare_route_leaves_session_history_untouched(ts_session, monkeypatch):
    """Comparing is a side quest, not a step: the panel answers four ways from
    where the session already is, and leaves it exactly there."""
    s = ts_session
    s.policy_provider = lambda: policy_stub()
    monkeypatch.setattr(server, "session", s)
    before_len, before_cursor = len(s.history), s.cursor

    result = asyncio.run(server.compare_route(
        server.CompareRequest(text="more bone", parser="rule", cheap=4, thorough=8)))

    assert result["applicable"] is True
    assert set(result["arms"]) == {"exact", "search_cheap", "search_thorough", "policy"}
    assert len(s.history) == before_len
    assert s.cursor == before_cursor


def test_compare_route_reports_a_camera_command_as_not_applicable(ts_session, monkeypatch):
    """Camera, reset, width and centre are not goals, so there is nothing to
    score four ways -- say so rather than erroring."""
    s = ts_session
    s.policy_provider = lambda: policy_stub()
    monkeypatch.setattr(server, "session", s)

    result = asyncio.run(server.compare_route(
        server.CompareRequest(text="rotate right", parser="rule", cheap=4, thorough=8)))

    assert result["applicable"] is False
    assert result["reason"]
    assert "arms" not in result
    # The reason is shown in the panel: no module paths, no dumped dicts.
    assert "commands.apply_command" not in result["reason"]
    assert "{" not in result["reason"]
    # The developer-facing text is still available, just not as the copy.
    assert "commands.apply_command" in result["detail"]


def test_compare_route_rejects_an_unparseable_instruction(ts_session, monkeypatch):
    s = ts_session
    s.policy_provider = lambda: policy_stub()
    monkeypatch.setattr(server, "session", s)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.compare_route(
            server.CompareRequest(text="qwertyuiop", parser="rule", cheap=4, thorough=8)))

    assert exc.value.status_code == 400


def test_compare_route_reports_the_start_state_beside_the_arms(ts_session, monkeypatch):
    """The panel draws a tick at each class's starting visibility, so "did the
    mentioned class move and did the others stay put" is one glance."""
    s = ts_session
    s.policy_provider = lambda: policy_stub()
    monkeypatch.setattr(server, "session", s)

    result = asyncio.run(server.compare_route(
        server.CompareRequest(text="more bone", parser="rule", cheap=4, thorough=8)))

    assert set(result["start"]["class_visibility"]) == set(goals.GOAL_CLASSES)
    assert result["budgets"] == {"cheap": 4, "thorough": 8}


@pytest.fixture
def ct_chest_session(monkeypatch):
    """A fresh Session switched to the real `ct_chest` Slicer CT -- the chat
    UI's default dataset (see server._resolve_dataset_name), which has no
    TotalSegmentator anatomy labels at all (unlike `ts_s1379` above). This
    exercises goals.goal_classes_for_volume's intensity-label fallback
    end to end through the real trained checkpoint, not a stub. Both
    `datasets.DATA_DIR` and `server.POLICY_PATH` are relative to cwd, and
    `_isolate_cwd` chdirs into a scratch tmp_path, so both are patched to
    the real repo paths -- same trick `_use_real_totalseg_manifest` plays
    for the manifest. `_policy_state` is a process-wide cache (like
    `server._dataset_name`), so it's reset around the test too: an earlier
    test's `_load_policy()` call (e.g. the "no checkpoint" test above) may
    have already cached a `None` policy from a cwd with no checkpoint."""
    import datasets
    import policy
    monkeypatch.setattr(datasets, "DATA_DIR", os.path.join(_REPO_ROOT, "data"))
    # policy.py owns the path now, so patch it there -- patching the re-export
    # on `server` would leave the loader reading the unpatched original.
    monkeypatch.setattr(policy, "POLICY_PATH", os.path.join(_REPO_ROOT, "out/rl_v2/oneshot_v3_seed0/best.zip"))
    policy.reset_cache()
    original_dataset = server._dataset_name
    s = _fresh_session()
    s.switch_dataset("ct_chest")
    try:
        yield s
    finally:
        server.set_dataset(original_dataset)
        policy.reset_cache()


def test_policy_mode_on_ct_chest_produces_a_real_policy_answer(ct_chest_session):
    # ct_chest has no anatomy labels, so this only works because
    # goals.goal_classes_for_volume falls back to what the intensity-label
    # visibility model can measure (skeleton, lungs, soft) instead of
    # raising -- confirming policy mode isn't limited to TotalSegmentator
    # volumes any more.
    s = ct_chest_session
    before_params = np.array(s.history[s.cursor]["params"], dtype=np.float64)

    state = s.command("more bone", mode="policy")

    assert state["current"]["mode"] == "policy"
    assert state["current"]["message"] is None
    params = np.array(state["current"]["params"], dtype=np.float64)
    assert not np.array_equal(params, before_params)


def test_command_records_its_mode():
    s = _fresh_session()
    exact_state = s.command("increase opacity for bone strongly", parser="rule", search=False)
    assert exact_state["current"]["mode"] == "exact"
    assert exact_state["current"]["message"] is None

    search_state = s.command("increase opacity for bone strongly", parser="rule",
                              search=True, steps=3)
    assert search_state["current"]["mode"] == "search"


# --- viewer telemetry and parser provenance (RL v2 vocabulary) -------------
# The viewer's readout and its parser badge have to speak the same four goal
# classes the policy does. `masses` (opacity over the retired fat/air/spongy
# bands) stays in the step for the search tests above, but it is no longer
# what the UI shows.

def test_step_reports_visibility_for_the_four_goal_classes():
    s = _fresh_session()
    vis = s.state()["current"]["class_visibility"]
    assert set(vis) == set(goals.GOAL_CLASSES)
    # every supported class reports a number; unsupported ones (vessels on a
    # volume without contrast) report None rather than a misleading 0.0
    assert vis["skeleton"] is not None and vis["skeleton"] >= 0.0


def test_step_visibility_tracks_a_command_that_hides_a_class():
    s = _fresh_session()
    before = s.state()["current"]["class_visibility"]["skeleton"]
    after = s.command("show only lungs", parser="rule")["current"]["class_visibility"]["skeleton"]
    assert after < before


def test_llm_parser_fallback_is_reported_to_the_viewer(monkeypatch):
    # Ollama down: the command still executes via the rule parser, but the UI
    # must be able to say so instead of silently claiming an LLM parse.
    monkeypatch.setattr("commands.OLLAMA_HOST", "http://127.0.0.1:1", raising=False)
    s = _fresh_session()
    step = s.command("more bone", parser="llm")["current"]
    assert step["parser_requested"] == "llm"
    assert step["parser_used"] == "rule"
    assert step["parser_fallback"]


def test_rule_parser_reports_itself_without_a_fallback_reason():
    s = _fresh_session()
    step = s.command("more bone", parser="rule")["current"]
    assert step["parser_requested"] == "rule"
    assert step["parser_used"] == "rule"
    assert step["parser_fallback"] is None


def test_policy_path_points_at_a_checkpoint_trained_on_the_corrected_observation():
    # v2 was trained with the lung ceiling probed at the retired band layout's
    # fat peak and with the colour action collapsing r=g=b (fixed in 62b5720 /
    # cc167af). The viewer must not demo that checkpoint when a v3-or-later
    # one exists. v4 (current) adds goals.distance's "other" keep term on top
    # of v3's observation fix -- it doesn't regress this property.
    assert "oneshot_v3" in server.POLICY_PATH or "oneshot_v4" in server.POLICY_PATH


def test_framing_is_measured_once_per_dataset_not_per_command():
    # The viewer's before/after comparison only works if the camera holds
    # still while the transfer function changes.
    server._frame_bounds_cache["dataset"] = None
    s = _fresh_session()
    first = server._frame_bounds()
    s.command("show only lungs", parser="rule")
    assert server._frame_bounds() is first


def test_collector_and_viewer_load_the_same_checkpoint():
    # These were separate constants until 2026-09-17, and they drifted: the
    # viewer moved to v3 when the observation fixes landed and the collector
    # kept sampling v2, so every preference pair compared candidates from the
    # superseded checkpoint.
    import collect
    import policy

    assert server.POLICY_PATH == policy.POLICY_PATH
    assert collect.POLICY_PATH == policy.POLICY_PATH
    assert "oneshot_v3" in policy.POLICY_PATH or "oneshot_v4" in policy.POLICY_PATH
