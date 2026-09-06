import json

from transfer import default_params
from commands import apply_command
from evaluate import objective, jsonl_append


def test_objective_positive_for_correct_increase():
    params = default_params()
    cmd = {"target": "bone", "attribute": "opacity",
           "direction": "increase", "strength": "strongly"}
    after = apply_command(cmd, params)
    assert objective(params, after, cmd) == 1


def test_objective_negative_when_no_change():
    params = default_params()
    cmd = {"target": "bone", "attribute": "opacity",
           "direction": "increase", "strength": "strongly"}
    assert objective(params, params, cmd) == -1


def test_objective_negative_for_wrong_direction():
    params = default_params()
    cmd_inc = {"target": "bone", "attribute": "opacity",
               "direction": "increase", "strength": "strongly"}
    after = apply_command(cmd_inc, params)
    cmd_dec = {**cmd_inc, "direction": "decrease"}
    assert objective(params, after, cmd_dec) == -1


def test_objective_show_only_positive_even_when_target_raw_mass_drops():
    # bone already saturated (height=1.0) before show_only sets it back to
    # the fixed 0.7 isolate-height — raw bone mass drops, but every other
    # peak gets crushed too, so bone's *share* of the image still rises.
    from transfer import PARAMS_PER_PEAK
    before = default_params().copy()
    before[3 * PARAMS_PER_PEAK + 2] = 1.0  # bone height external -> internal 1.0
    cmd = {"target": "bone", "attribute": "opacity", "direction": "show_only", "strength": None}
    after = apply_command(cmd, before)
    assert objective(before, after, cmd) == 1


def test_objective_show_only_negative_when_already_isolated():
    params = default_params()
    cmd = {"target": "bone", "attribute": "opacity", "direction": "show_only", "strength": None}
    isolated = apply_command(cmd, params)
    after = apply_command(cmd, isolated)  # re-applying changes nothing
    assert objective(isolated, after, cmd) == -1


def test_objective_always_accepts_width_brightness_center():
    from transfer import default_params
    params = default_params()
    for attribute in ("width", "brightness", "center"):
        cmd = {"target": "bone", "attribute": attribute, "direction": "increase"}
        assert objective(params, params, cmd) == 1  # even a no-op change is accepted


def test_objective_always_accepts_camera_commands():
    from transfer import default_params
    params = default_params()
    cmd = {"camera": {"action": "rotate", "direction": "left", "strength": "moderately"}}
    assert objective(params, params, cmd) == 1


def test_jsonl_append_writes_one_line(tmp_path):
    path = tmp_path / "log.jsonl"
    jsonl_append(str(path), {"a": 1})
    jsonl_append(str(path), {"b": 2})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
