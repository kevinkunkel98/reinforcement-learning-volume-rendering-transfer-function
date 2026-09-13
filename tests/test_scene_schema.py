import copy

import pytest

from scene_schema import normalize_scene, scene_transition


def valid_scene(**overrides):
    scene = {
        "scene_id": "s:1",
        "parent_scene_id": "s:0",
        "session_id": "session-1",
        "client": "web",
        "dataset": "ct_cardio",
        "dataset_version": "sha256:x",
        "volume": {
            "dimensions": [4, 5, 6],
            "spacing": [0.7, 0.7, 1.0],
            "scalar_type": "float32",
            "orientation": "dataset-normalized",
        },
        "transfer_function": [0.0] * 24,
        "camera": {
            "position": [0, 0, 1],
            "focal_point": [0, 0, 0],
            "view_up": [0, 1, 0],
            "zoom": 1.0,
        },
        "goal": {"target": "bone", "direction": "increase"},
        "command": {
            "attribute": "opacity",
            "target": "bone",
            "direction": "increase",
        },
    }
    scene.update(overrides)
    return scene


def test_normalize_scene_preserves_goal_camera_and_transfer_function():
    source = valid_scene()

    scene = normalize_scene(source)

    assert scene["client"] == "web"
    assert scene["volume"]["dimensions"] == [4, 5, 6]
    assert len(scene["transfer_function"]) == 24
    assert scene["camera"] == source["camera"]
    assert scene["goal"] == source["goal"]
    assert scene["command"] == source["command"]
    assert scene is not source
    assert scene["volume"] is not source["volume"]


@pytest.mark.parametrize(
    "change",
    [
        {"scene_id": ""},
        {"parent_scene_id": 3},
        {"session_id": ""},
        {"client": "desktop"},
        {"dataset": ""},
        {"dataset_version": ""},
        {"volume": {"dimensions": [0, 1, 1]}},
        {"volume": {"dimensions": [1.5, 1, 1]}},
        {"volume": {"spacing": [0.7, 0, 1.0]}},
        {"volume": {"scalar_type": "uint8"}},
        {"transfer_function": [0.0] * 23},
        {"transfer_function": [float("nan")] * 24},
        {"camera": {"position": [0, 0], "focal_point": [0, 0, 0], "view_up": [0, 1, 0], "zoom": 1}},
        {"camera": {"position": [0, 0, 1], "focal_point": [0, 0, 0], "view_up": [0, 1, 0], "zoom": 0}},
        {"goal": {"target": "", "direction": "increase"}},
        {"command": {"attribute": "color", "target": "bone", "direction": "increase"}},
    ],
)
def test_normalize_scene_rejects_invalid_contract(change):
    scene = valid_scene()
    for path, value in change.items():
        if path == "volume":
            scene["volume"].update(value)
        elif path == "camera":
            scene["camera"] = value
        elif path == "goal":
            scene["goal"] = value
        elif path == "command":
            scene["command"] = value
        else:
            scene[path] = value

    with pytest.raises(ValueError):
        normalize_scene(scene)


def test_normalize_scene_rejects_non_json_numbers_and_wrong_shapes():
    for field, value in [
        ("transfer_function", [float("inf")] * 24),
        ("transfer_function", ["0"] * 24),
        ("camera", {"position": [0, 0, 1], "focal_point": [0, 0, 0], "view_up": [0, 1, 0], "zoom": float("nan")}),
        ("volume", {"dimensions": [1, 2, 3], "spacing": [0.7, 0.7, 1.0], "scalar_type": "float32"}),
    ]:
        scene = valid_scene()
        if field == "volume":
            scene[field] = value
        else:
            scene[field] = value
        with pytest.raises(ValueError):
            normalize_scene(scene)


def test_scene_transition_requires_parent_and_preserves_optional_verdict_fields():
    before = normalize_scene(valid_scene(scene_id="s:0", parent_scene_id=None))
    after = valid_scene(scene_id="s:1", parent_scene_id="s:0")

    transition = scene_transition(
        before,
        after,
        verdict="accepted",
        accepted=True,
        ended=False,
    )

    assert transition["scene_id"] == "s:1"
    assert transition["parent_scene_id"] == "s:0"
    assert transition["verdict"] == "accepted"
    assert transition["accepted"] is True
    assert transition["ended"] is False


def test_scene_transition_rejects_mismatched_parent():
    before = normalize_scene(valid_scene(scene_id="s:0", parent_scene_id=None))
    after = valid_scene(scene_id="s:1", parent_scene_id="other")

    with pytest.raises(ValueError):
        scene_transition(before, after)


def test_normalize_scene_does_not_mutate_input():
    source = valid_scene()
    original = copy.deepcopy(source)

    normalize_scene(source)

    assert source == original
