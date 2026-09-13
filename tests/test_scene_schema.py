import copy
import json
from pathlib import Path

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


@pytest.mark.parametrize("target", ["air", "fat", "soft", "spongy", "bone"])
@pytest.mark.parametrize("direction", ["increase", "decrease"])
def test_normalize_scene_accepts_canonical_tissues_and_directions(target, direction):
    scene = valid_scene(
        goal={"target": target, "direction": direction},
        command={"attribute": "opacity", "target": target, "direction": direction},
    )

    normalized = normalize_scene(scene)

    assert normalized["goal"] == {"target": target, "direction": direction}
    assert normalized["command"]["target"] == target


@pytest.mark.parametrize("field,value", [("goal", {"target": "bones", "direction": "increase"}),
                                          ("goal", {"target": "bone", "direction": "up"}),
                                          ("command", {"attribute": "opacity", "target": "marrow", "direction": "increase"}),
                                          ("command", {"attribute": "opacity", "target": "bone", "direction": "up"})])
def test_normalize_scene_rejects_noncanonical_tissues_and_directions(field, value):
    scene = valid_scene(**{field: value})

    with pytest.raises(ValueError):
        normalize_scene(scene)


@pytest.mark.parametrize("verdict", ["better", "worse", "tie", "A", "B"])
def test_normalize_scene_accepts_approved_verdicts(verdict):
    scene = normalize_scene(valid_scene(verdict=verdict))

    assert scene["verdict"] == verdict


def test_normalize_scene_accepts_accepted_verdict_for_ui_compatibility():
    assert normalize_scene(valid_scene(verdict="accepted"))["verdict"] == "accepted"


@pytest.mark.parametrize("verdict", ["rejected", "maybe", 1, True])
def test_normalize_scene_rejects_unapproved_verdicts(verdict):
    with pytest.raises(ValueError):
        normalize_scene(valid_scene(verdict=verdict))


def test_normalize_scene_rejects_nonfinite_nested_optional_fields_and_unknown_fields():
    with pytest.raises(ValueError):
        normalize_scene(valid_scene(client_metadata={"render": {"samples": float("inf")}}))
    with pytest.raises(ValueError):
        normalize_scene(valid_scene(unexpected_field={"ok": True}))


def test_normalize_scene_rejects_unknown_nested_fixed_object_keys():
    for field, value in (
        ("volume", {**valid_scene()["volume"], "extra": True}),
        ("camera", {**valid_scene()["camera"], "extra": True}),
        ("goal", {**valid_scene()["goal"], "extra": True}),
        ("command", {**valid_scene()["command"], "extra": True}),
    ):
        with pytest.raises(ValueError):
            normalize_scene(valid_scene(**{field: value}))


def test_normalize_scene_preserves_extensible_client_metadata():
    metadata = {"renderer": "vtk.js", "settings": {"samples": 32}}

    scene = normalize_scene(valid_scene(client_metadata=metadata))

    assert scene["client_metadata"] == metadata


def test_normalize_scene_accepts_camera_only_scene_without_goal():
    scene = valid_scene()
    scene.pop("goal")
    scene["command"] = {"attribute": "camera"}

    normalized = normalize_scene(scene)

    assert "goal" not in normalized
    assert normalized["command"] == {"attribute": "camera"}


def test_normalize_scene_preserves_extractor_compatibility_mapping():
    scene = normalize_scene(valid_scene(parent_step_id=7, carried_forward=True))

    assert scene["parent_step_id"] == 7
    assert scene["carried_forward"] is True


def test_scene_fixture_documents_current_extractor_gap():
    fixture_path = Path(__file__).parent / "fixtures" / "scene_transition.json"
    records = json.loads(fixture_path.read_text())

    normalized = [normalize_scene(record) for record in records]

    assert normalized[1]["parent_scene_id"] == normalized[0]["scene_id"]
    assert normalized[1]["parent_step_id"] == 1
    assert normalized[1]["carried_forward"] is True
    # rl.extract_pairs still consumes step_id/parent_step_id rows, not scene IDs.
    assert "step_id" not in normalized[1]


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


@pytest.mark.parametrize(
    "change",
    [
        {"session_id": "other"},
        {"client": "vrui"},
        {"dataset": "other"},
        {"dataset_version": "sha256:other"},
        {"scene_id": "s:0"},
    ],
)
def test_scene_transition_rejects_incompatible_or_duplicate_scene(change):
    before = normalize_scene(valid_scene(scene_id="s:0", parent_scene_id=None))
    after = valid_scene(scene_id="s:1", parent_scene_id="s:0", **change)

    with pytest.raises(ValueError):
        scene_transition(before, after)


def test_scene_transition_rejects_volume_metadata_changes_but_allows_goal_changes():
    before = normalize_scene(valid_scene(scene_id="s:0", parent_scene_id=None))
    after = valid_scene(
        scene_id="s:1",
        parent_scene_id="s:0",
        goal={"target": "fat", "direction": "decrease"},
        volume={**valid_scene()["volume"], "spacing": [0.8, 0.7, 1.0]},
    )

    with pytest.raises(ValueError):
        scene_transition(before, after)

    after["volume"] = before["volume"]
    transition = scene_transition(before, after)
    assert transition["goal"] == {"target": "fat", "direction": "decrease"}


def test_scene_transition_accepts_camera_only_after_opacity_scene():
    before = normalize_scene(valid_scene(scene_id="s:0", parent_scene_id=None))
    after = valid_scene(scene_id="s:1", parent_scene_id="s:0")
    after.pop("goal")
    after["command"] = {"attribute": "camera"}

    transition = scene_transition(before, after)

    assert transition["command"] == {"attribute": "camera"}
    assert "goal" not in transition


def test_normalize_scene_does_not_mutate_input():
    source = valid_scene()
    original = copy.deepcopy(source)

    normalize_scene(source)

    assert source == original
