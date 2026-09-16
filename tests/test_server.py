"""Tests exercise Session directly, in-process, on pytest's own (main) thread.

FastAPI's TestClient runs the ASGI app on a background portal thread, and VTK's
Cocoa render window can only be created on the true process main thread on macOS
-- routing it through TestClient crashes the interpreter. Session has no FastAPI
dependency, so calling it directly sidesteps that entirely; the HTTP routing layer
itself is thin enough to verify by running the real server (see the plan's Task 6).
"""
import os
import asyncio
import json

import pytest

from fastapi import HTTPException
from fastapi.responses import Response

import server
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
        "transfer_function": [0] * 24,
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
    assert len(state["current"]["params"]) == 24


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


def test_objective_search_appends_one_final_step():
    s = _fresh_session()
    state = s.command("increase opacity for bone strongly", parser="rule",
                       search=True, steps=5)
    assert state["total"] == 2  # one new step, not one per iteration
    assert state["current"]["search"] is True
    assert state["current"]["masses"]["bone"] > 220.0


def test_objective_search_keeps_current_camera():
    s = _fresh_session()
    s.command("rotate right")
    cam = s.history[s.cursor]["camera"]
    state = s.command("increase opacity for bone strongly", parser="rule", search=True, steps=3)
    assert state["current"]["search"] is True
    assert state["current"]["camera"] == cam


def test_search_requested_for_non_opacity_attribute_falls_back_to_direct_apply():
    # "sharpen bone" parses to attribute="width", direction="decrease". search.propose_step
    # is hardcoded to mutate the height/opacity parameter, so search must not run for width
    # (or brightness/center) commands even when search=True is requested -- the command
    # should still be applied directly, just without the hill-climbing loop.
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
