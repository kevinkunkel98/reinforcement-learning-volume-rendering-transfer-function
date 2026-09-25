import copy
import json
from pathlib import Path

import pytest

import transfer
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
        "transfer_function": [0.0] * transfer.TOTAL_PARAMS,
        "camera": {
            "position": [0, 0, 1],
            "focal_point": [0, 0, 0],
            "view_up": [0, 1, 0],
            "zoom": 1.0,
        },
        "goal": {"target": "skeleton", "direction": "increase"},
        "command": {
            "attribute": "opacity",
            "target": "skeleton",
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
    assert len(scene["transfer_function"]) == transfer.TOTAL_PARAMS
    assert scene["camera"] == source["camera"]
    assert scene["goal"] == source["goal"]
    assert scene["command"] == source["command"]
    assert scene is not source
    assert scene["volume"] is not source["volume"]


def test_normalize_scene_defaults_optional_anatomy_layers_for_legacy_scene():
    scene = normalize_scene(valid_scene())

    from anatomy_layers import default_layers

    assert scene["anatomy_layers"] == default_layers()
    assert "label_layout" not in scene


def test_normalize_scene_accepts_anatomy_layers_and_label_layout():
    scene = normalize_scene(valid_scene(
        anatomy_layers={"liver": {"opacity": 0.25}},
        label_layout="anatomy-v2",
    ))

    assert scene["anatomy_layers"]["liver"]["opacity"] == 0.25
    assert scene["label_layout"] == "anatomy-v2"


@pytest.mark.parametrize("label_layout", ["anatomy-v1", "", 3, None])
def test_normalize_scene_rejects_invalid_label_layout(label_layout):
    with pytest.raises(ValueError):
        normalize_scene(valid_scene(label_layout=label_layout))


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
        {"transfer_function": [0.0] * (transfer.TOTAL_PARAMS - 1)},
        {"transfer_function": [float("nan")] * transfer.TOTAL_PARAMS},
        {"camera": {"position": [0, 0], "focal_point": [0, 0, 0], "view_up": [0, 1, 0], "zoom": 1}},
        {"camera": {"position": [0, 0, 1], "focal_point": [0, 0, 0], "view_up": [0, 1, 0], "zoom": 0}},
        {"goal": {"target": "", "direction": "increase"}},
        {"command": {"attribute": "color", "target": "skeleton", "direction": "increase"}},
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
        ("transfer_function", [float("inf")] * transfer.TOTAL_PARAMS),
        ("transfer_function", ["0"] * transfer.TOTAL_PARAMS),
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


def test_normalize_scene_rejects_legacy_transfer_function_length():
    scene = valid_scene(transfer_function=[0.0] * 24)

    with pytest.raises(ValueError, match=str(transfer.TOTAL_PARAMS)):
        normalize_scene(scene)


@pytest.mark.parametrize("target", ["skeleton", "lungs", "soft", "vessels"])
@pytest.mark.parametrize("direction", ["increase", "decrease"])
def test_normalize_scene_accepts_canonical_tissues_and_directions(target, direction):
    scene = valid_scene(
        goal={"target": target, "direction": direction},
        command={"attribute": "opacity", "target": target, "direction": direction},
    )

    normalized = normalize_scene(scene)

    assert normalized["goal"] == {"target": target, "direction": direction}
    assert normalized["command"]["target"] == target


@pytest.mark.parametrize("field,value", [("goal", {"target": "spongy", "direction": "increase"}),
                                          ("goal", {"target": "skeleton", "direction": "up"}),
                                          ("command", {"attribute": "opacity", "target": "marrow", "direction": "increase"}),
                                          ("command", {"attribute": "opacity", "target": "skeleton", "direction": "up"})])
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


@pytest.mark.parametrize("kind", ["root", "dataset_boundary"])
def test_normalize_scene_accepts_explicit_neutral_commands_without_goal(kind):
    scene = valid_scene(parent_scene_id=None, command={"attribute": "neutral", "kind": kind})
    scene.pop("goal")

    normalized = normalize_scene(scene)

    assert normalized["command"] == {"attribute": "neutral", "kind": kind}


def test_normalize_scene_accepts_non_extractable_command_with_parent():
    scene = valid_scene(
        command={"attribute": "neutral", "kind": "non_extractable"},
        parent_scene_id="s:0",
        client_metadata={"original_command": {"direction": "reset"}},
    )
    scene.pop("goal")

    normalized = normalize_scene(scene)

    assert normalized["command"] == {"attribute": "neutral", "kind": "non_extractable"}
    assert normalized["parent_scene_id"] == "s:0"


def test_normalize_scene_rejects_neutral_command_with_opacity_goal():
    scene = valid_scene(command={"attribute": "neutral", "kind": "root"})

    with pytest.raises(ValueError):
        normalize_scene(scene)


def test_scene_transition_rejects_dataset_changes_without_boundary():
    before = normalize_scene(valid_scene(scene_id="s:0", parent_scene_id=None))
    after = valid_scene(scene_id="s:1", parent_scene_id="s:0", dataset="other")

    with pytest.raises(ValueError, match="boundary"):
        scene_transition(before, after)


def test_normalize_scene_preserves_extractor_compatibility_mapping():
    scene = normalize_scene(valid_scene(parent_step_id=7, carried_forward=True, step_id=8,
                                        features_before={"mean": 1}, features_after={"mean": 2}))

    assert scene["parent_step_id"] == 7
    assert scene["carried_forward"] is True
    assert scene["step_id"] == 8
    assert scene["features_before"] == {"mean": 1}


def test_scene_fixture_documents_current_extractor_gap():
    fixture_path = Path(__file__).parent / "fixtures" / "scene_transition.json"
    records = json.loads(fixture_path.read_text())

    normalized = [normalize_scene(record) for record in records]

    assert normalized[1]["parent_scene_id"] == normalized[0]["scene_id"]
    assert normalized[1]["parent_step_id"] == 1
    assert normalized[1]["carried_forward"] is True
    # Normalized scenes carry scene_id/parent_scene_id, not legacy step_id rows.
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
        {"scene_id": "s:0"},
    ],
)
def test_scene_transition_rejects_incompatible_or_duplicate_scene(change):
    before = normalize_scene(valid_scene(scene_id="s:0", parent_scene_id=None))
    after_overrides = {"scene_id": "s:1", "parent_scene_id": "s:0"}
    after_overrides.update(change)
    after = valid_scene(**after_overrides)

    with pytest.raises(ValueError):
        scene_transition(before, after)


def test_scene_transition_allows_dataset_volume_changes_and_goal_changes():
    before = normalize_scene(valid_scene(scene_id="s:0", parent_scene_id=None))
    after = valid_scene(
        scene_id="s:1",
        parent_scene_id="s:0",
        goal={"target": "vessels", "direction": "decrease"},
        volume={**valid_scene()["volume"], "spacing": [0.8, 0.7, 1.0]},
    )

    transition = scene_transition(before, after)
    assert transition["goal"] == {"target": "vessels", "direction": "decrease"}
    assert transition["volume"]["spacing"] == [0.8, 0.7, 1.0]


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


def test_the_four_goal_classes_are_accepted_as_scene_targets():
    # _TISSUES was the retired HU-band vocabulary (air/fat/soft/spongy/bone), so
    # every scene transition naming skeleton, lungs or vessels -- three of the
    # four classes the parser actually emits -- was rejected with a 400. The
    # live log confirms it: out/scene_transitions.jsonl has never held one.
    import goals
    from scene_schema import _TISSUES

    for goal_class in goals.GOAL_CLASSES:
        assert goal_class in _TISSUES, f"{goal_class} is a goal class the parser emits"


@pytest.mark.parametrize("target", ["heart", "vessels", "liver", "kidneys", "spleen"])
def test_normalize_scene_accepts_registry_anatomy_targets(target):
    scene = normalize_scene(valid_scene(
        goal={"target": target, "direction": "increase"},
        command={"attribute": "opacity", "target": target, "direction": "increase"},
    ))
    assert scene["goal"]["target"] == target
    assert scene["command"]["target"] == target


def test_retired_band_names_are_no_longer_canonical_targets():
    from scene_schema import _TISSUES

    assert not ({"air", "fat", "spongy"} & _TISSUES)
