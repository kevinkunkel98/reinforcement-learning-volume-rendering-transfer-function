import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import commands
from commands import parse_command_rule, apply_command, parse_command_llm
from transfer import default_params, peak_internal, N_PEAKS


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # parse_command_llm now logs every call to out/llm_requests.jsonl -- run
    # every test in this module from a scratch directory so that never touches
    # the project's real out/ directory.
    monkeypatch.chdir(tmp_path)


def test_parse_increase_with_strength():
    cmd = parse_command_rule("increase opacity for bone strongly")
    assert cmd == {"target": "skeleton", "attribute": "opacity",
                    "direction": "increase", "strength": "strongly"}


def test_parse_decrease_default_strength():
    cmd = parse_command_rule("decrease opacity for lungs")
    assert cmd["target"] == "lungs"
    assert cmd["direction"] == "decrease"
    assert cmd["strength"] == "moderately"


def test_parse_show_only():
    cmd = parse_command_rule("show only soft tissue")
    assert cmd["direction"] == "show_only"
    assert cmd["target"] == "soft"


def test_layer_sentences_go_through_the_transfer_function():
    # The viewer's layer panel is gone; these used to set layer opacity instead.
    assert parse_command_rule("show only liver")["direction"] == "show_only"
    assert "layer_action" not in parse_command_rule("hide the kidneys")
    assert parse_command_rule("show heart and spleen")["target"] == ["heart", "spleen"]


_SHOW_ONLY_LIVER = {"attribute": "opacity", "direction": "show_only", "target": "liver", "strength": None,
                    "layer_action": {"action": "show_only", "targets": ["liver"]}}


def test_layer_commands_do_not_change_hounsfield_parameters():
    params = default_params()
    layers = commands.default_layers()
    before = params.copy()

    params_after, layers_after = commands.apply_command_state(
        _SHOW_ONLY_LIVER, params, layers
    )

    np.testing.assert_array_equal(params_after, before)
    assert layers_after["liver"]["opacity"] == 1.0
    assert all(layer["opacity"] == 0.0 for name, layer in layers_after.items() if name != "liver")


def test_hide_and_show_layer_commands_are_deterministic():
    layers = commands.default_layers()
    _, hidden = commands.apply_command_state(
        {"layer_action": {"action": "hide", "targets": ["kidneys"]}}, default_params(), layers
    )
    assert hidden["kidneys"]["opacity"] == 0.0

    _, shown = commands.apply_command_state(
        {"layer_action": {"action": "show", "targets": ["heart", "spleen"]}}, default_params(), hidden
    )
    assert shown["heart"]["opacity"] == 1.0
    assert shown["spleen"]["opacity"] == 1.0


def test_restore_all_anatomy_returns_layer_defaults():
    layers = commands.default_layers()
    layers["liver"]["opacity"] = 0.0
    _, restored = commands.apply_command_state(
        {"attribute": "layers", "action": "restore", "targets": []}, default_params(), layers
    )
    assert restored == commands.default_layers()


def test_parse_reset():
    cmd = parse_command_rule("reset")
    assert cmd == {"target": None, "attribute": None,
                    "direction": "reset", "strength": None}


def test_parse_unrecognized_raises():
    with pytest.raises(ValueError):
        parse_command_rule("make the weather nicer")


def test_apply_command_increases_target_height():
    params = default_params()
    before_h = peak_internal(params, 3)["height"]  # bone is peak index 3
    cmd = {"target": "bone", "attribute": "opacity",
           "direction": "increase", "strength": "strongly"}
    after = apply_command(cmd, params)
    after_h = peak_internal(after, 3)["height"]
    assert after_h > before_h


def test_apply_command_increase_never_fully_saturates_on_repeat():
    # Regression: a fixed +0.6 step for "strongly" used to hit the height=1.0
    # ceiling in one command, after which every further "increase" was a
    # silent no-op with no feedback. Repeated commands should keep making
    # (shrinking but nonzero) progress instead of hard-stopping.
    params = default_params()
    cmd = {"target": "bone", "attribute": "opacity",
           "direction": "increase", "strength": "strongly"}
    heights = []
    for _ in range(5):
        params = apply_command(cmd, params)
        heights.append(peak_internal(params, 3)["height"])
    assert all(b < a < 1.0 for b, a in zip([0.6] + heights, heights))


def test_apply_command_decrease_never_fully_saturates_on_repeat():
    params = default_params()
    cmd = {"target": "bone", "attribute": "opacity",
           "direction": "decrease", "strength": "strongly"}
    heights = []
    for _ in range(5):
        params = apply_command(cmd, params)
        heights.append(peak_internal(params, 3)["height"])
    assert all(b > a > 0.0 for b, a in zip([0.6] + heights, heights))


def test_apply_command_reset_returns_default():
    params = default_params()
    params[2] = 1.0  # perturb
    cmd = {"target": None, "attribute": None, "direction": "reset", "strength": None}
    after = apply_command(cmd, params)
    np.testing.assert_array_equal(after, default_params())


def test_apply_command_show_only_suppresses_others():
    params = default_params()
    cmd = {"target": "bone", "attribute": "opacity",
           "direction": "show_only", "strength": None}
    after = apply_command(cmd, params)
    heights = [peak_internal(after, i)["height"] for i in range(N_PEAKS)]
    assert heights[3] > 0.5
    assert all(h < 0.05 for i, h in enumerate(heights) if i != 3)


def test_apply_command_show_only_narrows_a_wide_shown_peak():
    """Skeleton's default width (280 HU) still has real opacity reaching into
    neighbouring tissue's HU range -- boosting its height alone lights up
    organ/muscle voxels too. "Show only" must also cap the shown peak's width
    (see transfer.SHOW_ONLY_MAX_WIDTH_HU)."""
    params = default_params()
    cmd = {"target": "bone", "attribute": "opacity",
           "direction": "show_only", "strength": None}
    after = apply_command(cmd, params)
    assert peak_internal(after, 3)["width"] == pytest.approx(150.0)


def test_apply_command_show_only_does_not_widen_an_already_narrow_peak():
    params = default_params()
    cmd = {"target": "lungs", "attribute": "opacity",
           "direction": "show_only", "strength": None}
    after = apply_command(cmd, params)
    assert peak_internal(after, 0)["width"] == pytest.approx(60.0)


def _fake_response(payload: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps({"response": json.dumps(payload)}).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_parse_command_llm_valid_response():
    payload = {"target": "skeleton", "attribute": "opacity",
               "direction": "increase", "strength": "strongly"}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("make the skeleton pop")
    assert cmd == payload


def test_parse_command_llm_falls_back_on_connection_error(capsys):
    with patch("commands.urlopen", side_effect=OSError("connection refused")):
        cmd = parse_command_llm("increase opacity for bone strongly")
    assert cmd["target"] == "skeleton"  # fallback re-parses with the rule parser's new vocabulary
    assert cmd["direction"] == "increase"
    assert "falling back to rule parser" in capsys.readouterr().out


def test_parse_command_llm_falls_back_on_invalid_schema(capsys):
    with patch("commands.urlopen", return_value=_fake_response({"foo": "bar"})):
        cmd = parse_command_llm("increase opacity for bone strongly")
    assert cmd["target"] == "skeleton"
    assert "falling back to rule parser" in capsys.readouterr().out


def test_parse_command_llm_falls_back_when_target_missing_for_non_reset(capsys):
    payload = {"target": None, "attribute": "opacity", "direction": "show_only", "strength": None}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("increase opacity for bone strongly")
    assert cmd["target"] == "skeleton"
    assert "falling back to rule parser" in capsys.readouterr().out


def test_parse_command_llm_normalizes_alias_target_instead_of_falling_back(capsys):
    payload = {"target": "bones", "attribute": "opacity", "direction": "increase", "strength": "moderately"}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("increase opacity for the bones")
    assert cmd["target"] == "skeleton"
    assert "falling back to rule parser" not in capsys.readouterr().out


def test_parse_command_llm_logs_successful_request(tmp_path):
    payload = {"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": "strongly"}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        parse_command_llm("increase opacity for bone strongly")

    lines = (tmp_path / "out" / "llm_requests.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["text"] == "increase opacity for bone strongly"
    assert entry["final_cmd"] == payload
    assert entry["fell_back"] is False
    assert entry["fallback_reason"] is None


def test_parse_command_llm_logs_fallback_with_reason(tmp_path):
    with patch("commands.urlopen", side_effect=OSError("connection refused")):
        parse_command_llm("increase opacity for bone strongly")

    lines = (tmp_path / "out" / "llm_requests.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["fell_back"] is True
    assert "connection refused" in entry["fallback_reason"]
    assert entry["final_cmd"]["target"] == "skeleton"


# ---------- multi-target show_only ----------

def test_parse_show_only_multiple_tissues():
    cmd = parse_command_rule("show only bone and lungs")
    assert cmd["direction"] == "show_only"
    assert sorted(cmd["target"]) == ["lungs", "skeleton"]


def test_parse_show_only_single_tissue_still_returns_plain_string():
    cmd = parse_command_rule("show only bone")
    assert cmd["target"] == "skeleton"


def test_apply_command_show_only_multi_target_keeps_both_visible():
    params = default_params()
    cmd = {"target": ["bone", "spongy"], "attribute": "opacity",
           "direction": "show_only", "strength": None}
    after = apply_command(cmd, params)
    heights = [peak_internal(after, i)["height"] for i in range(N_PEAKS)]
    assert heights[3] > 0.5  # bone
    assert heights[2] > 0.5  # spongy
    assert all(h < 0.05 for i, h in enumerate(heights) if i not in (2, 3))


# ---------- compound "set level" commands ----------

def test_parse_single_level_command_not_wrapped_in_compound():
    cmd = parse_command_rule("high opacity for bone")
    assert cmd == {"target": "skeleton", "attribute": "opacity", "direction": "set", "level": "high"}


def test_parse_compound_level_command():
    cmd = parse_command_rule("high opacity vessels, low opacity bone")
    assert "compound" in cmd
    by_target = {c["target"]: c["level"] for c in cmd["compound"]}
    assert by_target == {"vessels": "high", "skeleton": "low"}
    assert all(c["direction"] == "set" for c in cmd["compound"])


def test_apply_command_set_assigns_absolute_level_regardless_of_current():
    params = default_params()
    cmd = {"target": "bone", "attribute": "opacity", "direction": "set", "level": "low"}
    after = apply_command(cmd, params)
    # bone's default height (0.6) is well above "low" (0.15) -- set must move
    # it down to the absolute level, not just push it further up.
    assert peak_internal(after, 3)["height"] == pytest.approx(0.15, abs=1e-6)


def test_apply_command_compound_sets_each_tissue_to_its_own_level():
    params = default_params()
    cmd = {"compound": [
        {"target": "spongy", "attribute": "opacity", "direction": "set", "level": "high"},
        {"target": "bone", "attribute": "opacity", "direction": "set", "level": "low"},
    ]}
    after = apply_command(cmd, params)
    assert peak_internal(after, 2)["height"] == pytest.approx(0.85, abs=1e-6)  # spongy
    assert peak_internal(after, 3)["height"] == pytest.approx(0.15, abs=1e-6)  # bone


def test_parse_command_llm_multi_target_show_only():
    payload = {"target": ["bone", "lungs"], "attribute": "opacity",
               "direction": "show_only", "strength": None}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("show bone and lungs")
    assert sorted(cmd["target"]) == ["lungs", "skeleton"]


def test_parse_command_llm_compound_response():
    payload = {"compound": [
        {"target": "vessels", "attribute": "opacity", "direction": "set", "level": "high"},
        {"target": "skeleton", "attribute": "opacity", "direction": "set", "level": "low"},
    ]}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("high opacity vessels, low opacity skeleton")
    assert cmd == payload


def test_parse_command_llm_compound_normalizes_alias_targets(capsys):
    """A one-clause compound is folded to the bare command the rule parser
    would produce for the same sentence (`_normalise_cmd`), with its alias
    target ("bones") still normalised to the canonical class."""
    payload = {"compound": [
        {"target": "bones", "attribute": "opacity", "direction": "set", "level": "low"},
    ]}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("low opacity for the bones")
    assert cmd["target"] == "skeleton"
    assert "falling back to rule parser" not in capsys.readouterr().out


def test_parse_command_llm_normalizes_alias_targets_inside_a_real_compound(capsys):
    payload = {"compound": [
        {"target": "bones", "attribute": "opacity", "direction": "set", "level": "low"},
        {"target": "pulmonary", "attribute": "opacity", "direction": "set", "level": "high"},
    ]}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("low opacity for the bones, high for the pulmonary tissue")
    assert [sub["target"] for sub in cmd["compound"]] == ["skeleton", "lungs"]
    assert "falling back to rule parser" not in capsys.readouterr().out


# ---------- Task 4: the LLM parser speaks the new vocabulary ----------

def test_validate_accepts_compound_of_relative_subcommands():
    from commands import _validate_cmd
    good = {"compound": [
        {"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": "moderately"},
        {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "slightly"},
    ]}
    assert _validate_cmd(good) is True


def test_validate_rejects_retired_class_target():
    from commands import _validate_cmd
    bad = {"target": "spongy", "attribute": "opacity", "direction": "increase", "strength": "moderately"}
    assert _validate_cmd(bad) is False


def test_validate_rejects_compound_subcommand_mixing_level_and_strength():
    from commands import _validate_cmd
    bad = {"compound": [
        {"target": "skeleton", "attribute": "opacity", "direction": "set",
         "level": "high", "strength": "strongly"},
    ]}
    assert _validate_cmd(bad) is False


def test_validate_set_cmd_shape_unchanged():
    from commands import _validate_set_cmd
    good = {"target": "skeleton", "attribute": "opacity", "direction": "set", "level": "high"}
    assert _validate_set_cmd(good) is True


def test_validate_single_cmd_shape_unchanged():
    from commands import _validate_single_cmd
    good = {"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": "moderately"}
    assert _validate_single_cmd(good) is True


def test_parse_command_llm_compound_with_relative_subcommands():
    payload = {"compound": [
        {"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": "moderately"},
        {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "slightly"},
    ]}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("more bone, a bit less soft tissue")
    assert cmd == payload


def test_parse_command_llm_rejects_retired_target_via_rule_parser_fallback():
    payload = {"target": "spongy", "attribute": "opacity", "direction": "increase", "strength": "moderately"}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        with pytest.raises(ValueError, match="skeleton, lungs, heart, vessels"):
            parse_command_llm("decrease opacity for spongy bone")


# ---------- width / brightness / center attributes ----------

def test_apply_command_width_increase_widens_peak():
    params = default_params()
    before_w = peak_internal(params, 3)["width"]  # bone is peak index 3
    cmd = {"target": "bone", "attribute": "width", "direction": "increase", "strength": "strongly"}
    after = apply_command(cmd, params)
    after_w = peak_internal(after, 3)["width"]
    assert after_w > before_w


def test_apply_command_width_decrease_narrows_peak():
    params = default_params()
    before_w = peak_internal(params, 3)["width"]
    cmd = {"target": "bone", "attribute": "width", "direction": "decrease", "strength": "strongly"}
    after = apply_command(cmd, params)
    after_w = peak_internal(after, 3)["width"]
    assert after_w < before_w


def test_apply_command_brightness_increase_raises_all_channels():
    params = default_params()
    before_rgb = peak_internal(params, 3)["rgb"]
    cmd = {"target": "bone", "attribute": "brightness", "direction": "increase", "strength": "strongly"}
    after = apply_command(cmd, params)
    after_rgb = peak_internal(after, 3)["rgb"]
    for before_c, after_c in zip(before_rgb, after_rgb):
        assert after_c >= before_c


def test_apply_command_brightness_decrease_lowers_all_channels():
    params = default_params()
    before_rgb = peak_internal(params, 3)["rgb"]
    cmd = {"target": "bone", "attribute": "brightness", "direction": "decrease", "strength": "strongly"}
    after = apply_command(cmd, params)
    after_rgb = peak_internal(after, 3)["rgb"]
    for before_c, after_c in zip(before_rgb, after_rgb):
        assert after_c <= before_c


def test_apply_command_center_increase_shifts_center_up():
    params = default_params()
    before_c = peak_internal(params, 3)["center"]  # bone, defaults near 900 HU
    cmd = {"target": "bone", "attribute": "center", "direction": "increase", "strength": "slightly"}
    after = apply_command(cmd, params)
    after_c = peak_internal(after, 3)["center"]
    assert after_c > before_c


def test_apply_command_center_clamps_to_own_tissue_band():
    from transfer import TISSUE_BANDS
    params = default_params()
    cmd = {"target": "bone", "attribute": "center", "direction": "increase", "strength": "strongly"}
    for _ in range(50):
        params = apply_command(cmd, params)
    final_c = peak_internal(params, 3)["center"]
    lo, hi = TISSUE_BANDS["bone"]
    assert final_c <= hi + 1e-6
    assert final_c >= lo - 1e-6


def test_apply_command_set_level_width():
    params = default_params()
    cmd = {"target": "bone", "attribute": "width", "direction": "set", "level": "high"}
    after = apply_command(cmd, params)
    from transfer import WIDTH_RANGE
    after_w = peak_internal(after, 3)["width"]
    # LEVEL_WORDS["high"] = 0.85 of the way through WIDTH_RANGE
    expected = WIDTH_RANGE[0] + 0.85 * (WIDTH_RANGE[1] - WIDTH_RANGE[0])
    assert abs(after_w - expected) < 1.0


def test_apply_command_rejects_center_with_set_direction():
    params = default_params()
    cmd = {"target": "fat", "attribute": "center", "direction": "set", "level": "high"}
    with pytest.raises(ValueError):
        apply_command(cmd, params)


def test_parse_sharpen_verb():
    cmd = parse_command_rule("sharpen the bone peak")
    assert cmd["target"] == "skeleton"
    assert cmd["attribute"] == "width"
    assert cmd["direction"] == "decrease"


def test_parse_soften_verb():
    cmd = parse_command_rule("soften soft tissue")
    assert cmd["target"] == "soft"
    assert cmd["attribute"] == "width"
    assert cmd["direction"] == "increase"


def test_parse_brighten_verb():
    cmd = parse_command_rule("brighten bone strongly")
    assert cmd["target"] == "skeleton"
    assert cmd["attribute"] == "brightness"
    assert cmd["direction"] == "increase"
    assert cmd["strength"] == "strongly"


def test_parse_darken_verb():
    cmd = parse_command_rule("darken lungs")
    assert cmd["target"] == "lungs"
    assert cmd["attribute"] == "brightness"
    assert cmd["direction"] == "decrease"


def test_parse_hide_verb_defaults_to_strongly():
    """"Hide"/"remove" have no dedicated direction (the LLM prompt already
    says so: "decrease" with strength "strongly"), but the rule parser had no
    vocabulary for them at all -- "hide the lungs" raised, it didn't parse as
    a strong decrease."""
    cmd = parse_command_rule("hide the lungs")
    assert cmd["target"] == "lungs"
    assert cmd["attribute"] == "opacity"
    assert cmd["direction"] == "decrease"
    assert cmd["strength"] == "strongly"


def test_parse_remove_verb_defaults_to_strongly():
    cmd = parse_command_rule("remove the skeleton")
    assert cmd["target"] == "skeleton"
    assert cmd["attribute"] == "opacity"
    assert cmd["direction"] == "decrease"
    assert cmd["strength"] == "strongly"


def test_parse_hide_verb_respects_an_explicit_strength():
    cmd = parse_command_rule("hide the lungs slightly")
    assert cmd["strength"] == "slightly"


def test_parse_shift_center_up_with_apostrophe():
    cmd = parse_command_rule("shift bone's center up")
    assert cmd["target"] == "skeleton"
    assert cmd["attribute"] == "center"
    assert cmd["direction"] == "increase"


def test_parse_move_down_without_center_word():
    cmd = parse_command_rule("move lungs down")
    assert cmd["target"] == "lungs"
    assert cmd["attribute"] == "center"
    assert cmd["direction"] == "decrease"


def test_parse_increase_width_generalized():
    cmd = parse_command_rule("increase width for lungs")
    assert cmd["target"] == "lungs"
    assert cmd["attribute"] == "width"
    assert cmd["direction"] == "increase"


def test_parse_increase_sharpness_maps_to_width():
    cmd = parse_command_rule("increase sharpness for vessels strongly")
    assert cmd["target"] == "vessels"
    assert cmd["attribute"] == "width"


def test_parse_low_brightness_absolute():
    cmd = parse_command_rule("low brightness for vessels")
    assert cmd == {"target": "vessels", "attribute": "brightness", "direction": "set", "level": "low"}


def test_parse_high_sharpness_absolute_maps_to_width():
    cmd = parse_command_rule("high sharpness bone")
    assert cmd["attribute"] == "width"
    assert cmd["level"] == "high"


def test_validate_rejects_center_with_set_direction():
    from commands import _validate_cmd
    bad = {"compound": [
        {"target": "bone", "attribute": "center", "direction": "set", "level": "high"},
    ]}
    assert _validate_cmd(bad) is False


def test_validate_accepts_width_set():
    from commands import _validate_cmd
    good = {"compound": [
        {"target": "skeleton", "attribute": "width", "direction": "set", "level": "high"},
    ]}
    assert _validate_cmd(good) is True


def test_parse_rotate_right():
    cmd = parse_command_rule("rotate right")
    assert cmd == {"camera": {"action": "rotate", "direction": "right", "strength": "moderately"}}


def test_parse_turn_left_with_strength():
    cmd = parse_command_rule("turn left strongly")
    assert cmd == {"camera": {"action": "rotate", "direction": "left", "strength": "strongly"}}


def test_parse_tilt_up():
    cmd = parse_command_rule("tilt up")
    assert cmd == {"camera": {"action": "tilt", "direction": "up", "strength": "moderately"}}


def test_parse_tilt_down_slightly():
    cmd = parse_command_rule("tilt down slightly")
    assert cmd == {"camera": {"action": "tilt", "direction": "down", "strength": "slightly"}}


def test_parse_zoom_in():
    cmd = parse_command_rule("zoom in")
    assert cmd == {"camera": {"action": "zoom", "direction": "in", "strength": "moderately"}}


def test_parse_zoom_out():
    cmd = parse_command_rule("zoom out")
    assert cmd == {"camera": {"action": "zoom", "direction": "out", "strength": "moderately"}}


def test_validate_accepts_well_formed_camera_command():
    from commands import _validate_cmd
    good = {"camera": {"action": "rotate", "direction": "left", "strength": "moderately"}}
    assert _validate_cmd(good) is True


def test_validate_rejects_camera_command_with_mismatched_direction():
    from commands import _validate_cmd
    bad = {"camera": {"action": "rotate", "direction": "up", "strength": "moderately"}}
    assert _validate_cmd(bad) is False


def test_validate_rejects_camera_command_with_unknown_action():
    from commands import _validate_cmd
    bad = {"camera": {"action": "pan", "direction": "left", "strength": "moderately"}}
    assert _validate_cmd(bad) is False


def test_validate_rejects_camera_command_with_invalid_strength():
    from commands import _validate_cmd
    bad = {"camera": {"action": "rotate", "direction": "left", "strength": "very much"}}
    assert _validate_cmd(bad) is False


def test_validate_rejects_camera_command_with_null_strength():
    from commands import _validate_cmd
    bad = {"camera": {"action": "rotate", "direction": "left", "strength": None}}
    assert _validate_cmd(bad) is False


# ---------- Task 1: anatomical vocabulary in the parser ----------

from commands import CLASS_SYNONYMS


@pytest.mark.parametrize("cls,words", list(CLASS_SYNONYMS.items()))
def test_every_synonym_resolves_to_its_goal_class(cls, words):
    for word in words:
        cmd = parse_command_rule(f"increase opacity for {word} strongly")
        assert cmd["target"] == cls, (word, cmd)


def test_parse_more_bone_defaults_to_moderate_increase():
    cmd = parse_command_rule("more bone")
    assert cmd == {"target": "skeleton", "attribute": "opacity",
                    "direction": "increase", "strength": "moderately"}


def test_parse_less_bone_defaults_to_moderate_decrease():
    cmd = parse_command_rule("less bone")
    assert cmd == {"target": "skeleton", "attribute": "opacity",
                    "direction": "decrease", "strength": "moderately"}


def test_parse_a_bit_less_soft_tissue_is_slight_decrease():
    cmd = parse_command_rule("a bit less soft tissue")
    assert cmd == {"target": "soft", "attribute": "opacity",
                    "direction": "decrease", "strength": "slightly"}


def test_parse_much_less_is_strong_decrease():
    cmd = parse_command_rule("much less vessels")
    assert cmd["target"] == "vessels"
    assert cmd["direction"] == "decrease"
    assert cmd["strength"] == "strongly"


@pytest.mark.parametrize(
    ("text", "target", "direction"),
    [
        ("show more liver", "liver", "increase"),
        ("hide the kidneys", "kidneys", "decrease"),
        ("brighten the spleen", "spleen", "increase"),
        ("show heart", "heart", "show_only"),
        ("reduce aorta", "vessels", "decrease"),
    ],
)
def test_parse_eight_class_anatomy_vocabulary(text, target, direction):
    cmd = parse_command_rule(text)
    assert cmd["target"] == target
    assert cmd["direction"] == direction


def test_llm_prompt_describes_promoted_anatomy_classes():
    for term in ("heart", "cardiac", "liver", "kidneys", "renal", "spleen", "aorta", "vena cava"):
        assert term in commands._SYSTEM_PROMPT


def test_parse_compound_relative_clauses():
    cmd = parse_command_rule("more bone, a bit less soft tissue")
    assert "compound" in cmd
    assert cmd["compound"] == [
        {"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": "moderately"},
        {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "slightly"},
    ]


def test_parse_compound_relative_clauses_raises_if_one_clause_unparseable():
    with pytest.raises(ValueError):
        parse_command_rule("more bone, make the weather nicer")


def test_parse_show_me_the_class():
    cmd = parse_command_rule("show me the lungs")
    assert cmd["direction"] == "show_only"
    assert cmd["target"] == "lungs"


def test_parse_rejects_fat():
    with pytest.raises(ValueError, match="skeleton, lungs, heart, vessels"):
        parse_command_rule("decrease opacity for fat")


def test_parse_rejects_air():
    with pytest.raises(ValueError, match="skeleton, lungs, heart, vessels"):
        parse_command_rule("decrease opacity for air")


def test_parse_rejects_spongy():
    with pytest.raises(ValueError, match="skeleton, lungs, heart, vessels"):
        parse_command_rule("decrease opacity for spongy bone")


# ---------- camera, reset, width and centre commands still parse unchanged ----------

def test_parse_reset_still_parses_unchanged():
    assert parse_command_rule("reset") == {
        "target": None, "attribute": None, "direction": "reset", "strength": None}


def test_parse_camera_commands_still_parse_unchanged():
    assert parse_command_rule("rotate left") == {
        "camera": {"action": "rotate", "direction": "left", "strength": "moderately"}}
    assert parse_command_rule("tilt up") == {
        "camera": {"action": "tilt", "direction": "up", "strength": "moderately"}}
    assert parse_command_rule("zoom in") == {
        "camera": {"action": "zoom", "direction": "in", "strength": "moderately"}}


def test_parse_width_and_centre_commands_still_parse_unchanged():
    cmd = parse_command_rule("increase width for bone strongly")
    assert cmd["target"] == "skeleton"
    assert cmd["attribute"] == "width"
    assert cmd["direction"] == "increase"

    cmd = parse_command_rule("shift bone's center up")
    assert cmd["target"] == "skeleton"
    assert cmd["attribute"] == "center"
    assert cmd["direction"] == "increase"


# --- LLM parser normalisation and determinism ---------------------------------

def test_single_element_compound_is_unwrapped():
    """The rule parser already folds a one-clause sentence to a bare command
    (`parse_command_rule`: `subcommands[0] if len(subcommands) == 1`), and the
    LLM is asked for the same shape but often wraps one clause in a compound
    anyway. Downstream code -- goals.goal_from_command, the eval harness --
    then sees a different shape for a semantically identical instruction."""
    got = commands._normalise_cmd({"compound": [
        {"target": "skeleton", "attribute": "opacity",
         "direction": "increase", "strength": "moderately"}]})

    assert got == {"target": "skeleton", "attribute": "opacity",
                   "direction": "increase", "strength": "moderately"}


def test_multi_element_compound_is_left_alone():
    cmd = {"compound": [
        {"target": "skeleton", "attribute": "opacity",
         "direction": "increase", "strength": "moderately"},
        {"target": "soft", "attribute": "opacity",
         "direction": "decrease", "strength": "slightly"}]}

    assert commands._normalise_cmd(cmd) == cmd


def test_plain_command_is_left_alone():
    cmd = {"target": "lungs", "attribute": "opacity",
           "direction": "decrease", "strength": "slightly"}

    assert commands._normalise_cmd(cmd) == cmd


def test_llm_request_pins_temperature_and_seed():
    """Parsing must be reproducible: the same instruction has to yield the same
    command every time, or the viewer answers a spoken command differently on
    a retry and the parser evaluation is not reproducible. Ollama's default
    temperature is 0.8."""
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"response": json.dumps(
                {"target": "skeleton", "attribute": "opacity",
                 "direction": "increase", "strength": "moderately"})}).encode()

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse()

    original = commands.urlopen
    commands.urlopen = _fake_urlopen
    try:
        commands.parse_command_llm("make the bone pop")
    finally:
        commands.urlopen = original

    options = captured["body"].get("options", {})
    assert options.get("temperature") == 0
    assert "seed" in options


def test_one_clause_compound_holding_a_show_only_is_not_a_fallback(capsys):
    """qwen2.5:7b answers "show only the lungs" with the show_only clause
    wrapped in a one-element compound. `_normalise_cmd` unwraps exactly that
    shape, but validation used to run first and reject it -- a compound
    sub-command may only be a set/relative clause -- so a correct parse was
    thrown away and the rule parser silently answered instead."""
    payload = {"compound": [
        {"target": ["lungs"], "attribute": None, "direction": "show_only", "strength": None},
    ]}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("show only the lungs")
    assert cmd["direction"] == "show_only"
    assert cmd["target"] == "lungs" or cmd["target"] == ["lungs"]
    assert "falling back to rule parser" not in capsys.readouterr().out


def test_multi_clause_compound_still_rejects_a_show_only_sub_command():
    from commands import _validate_cmd
    assert not _validate_cmd({"compound": [
        {"target": ["lungs"], "attribute": None, "direction": "show_only", "strength": None},
        {"target": "skeleton", "attribute": "opacity", "direction": "set", "level": "low"},
    ]})


def test_reset_returns_the_given_start_tf():
    start = default_params() * 0.5
    reset = parse_command_rule("reset")
    np.testing.assert_array_equal(apply_command(reset, default_params() * 0.1, reset_params=start), start)
    np.testing.assert_array_equal(apply_command(reset, start), default_params())


# ---------- everyday phrasings (data/parser_eval_set.json has the full set) ----------

def _rel(target, direction, strength="moderately", attribute="opacity"):
    return {"target": target, "attribute": attribute, "direction": direction, "strength": strength}


@pytest.mark.parametrize("text,expected", [
    ("add bones", _rel("skeleton", "increase")),
    ("increase opacity of soft tissues strongly", _rel("soft", "increase", "strongly")),
    ("decrease soft tissue opacity strongly.", _rel("soft", "decrease", "strongly")),
    ("decrease opacity of soft tissue a lot", _rel("soft", "decrease", "strongly")),
    ("make the skeleton pop", _rel("skeleton", "increase")),
    ("the lungs are barely visible", _rel("lungs", "increase")),
    ("let me see through the muscle", _rel("soft", "decrease")),
    ("tone down the veins a bit", _rel("vessels", "decrease", "slightly")),
    ("I need to see the pulmonary tissue better", _rel("lungs", "increase")),
    ("darken langs extremely", _rel("lungs", "decrease", "strongly", "brightness")),
    ("brighten skeleton a lot", _rel("skeleton", "increase", "strongly", "brightness")),
    ("decrease opacity of soft tissue so I can see the bones.", _rel("soft", "decrease")),
])
def test_rule_parser_everyday_phrasings(text, expected):
    assert parse_command_rule(text) == expected


@pytest.mark.parametrize("text,level,target", [
    ("high spleen", "high", "spleen"), ("set the spleen to high", "high", "spleen"),
    ("liver opacity medium", "medium", "liver"), ("make the kidneys low opacity", "low", "kidneys"),
])
def test_rule_parser_absolute_levels_in_any_order(text, level, target):
    assert parse_command_rule(text) == {"target": target, "attribute": "opacity",
                                        "direction": "set", "level": level}


@pytest.mark.parametrize("text,target", [
    ("only the kidneys", "kidneys"), ("isolate the liver", "liver"),
    ("I only want to see the vessels", "vessels"), ("nothing but bone", "skeleton"),
    ("display just the lungs and heart", ["lungs", "heart"]),
])
def test_rule_parser_show_only_phrasings(text, target):
    cmd = parse_command_rule(text)
    assert cmd["direction"] == "show_only" and cmd["target"] == target


def test_rule_parser_keeps_every_clause_of_a_compound():
    # Before: the greedy verb regex read this as one darken of soft tissue.
    cmd = parse_command_rule("darken bone extremely and decrease opacity of soft tissue strongly.")
    assert cmd == {"compound": [_rel("skeleton", "decrease", "strongly", "brightness"),
                                _rel("soft", "decrease", "strongly")]}
    # Before: "show ... less" became show_only.
    assert parse_command_rule("show a lot less bones") == _rel("skeleton", "decrease", "strongly")


def test_rule_parser_pronoun_clause_refers_to_the_previous_class():
    assert parse_command_rule("the aorta is dominating the view, pull it back") == _rel("vessels", "decrease")


def test_rule_parser_folds_a_modifier_only_clause_into_its_neighbour():
    assert parse_command_rule("Darken the skeleton, extremely.") == _rel("skeleton", "decrease", "strongly",
                                                                          "brightness")


@pytest.mark.parametrize("text", ["revert", "start over", "go back to the default"])
def test_rule_parser_reset_words(text):
    assert parse_command_rule(text)["direction"] == "reset"


@pytest.mark.parametrize("text", ["thank you.", "bye.", "you", "okay"])
def test_rule_parser_refuses_speech_noise(text):
    with pytest.raises(ValueError):
        parse_command_rule(text)


def test_parse_command_llm_repairs_a_missing_strength_and_attribute():
    payload = {"target": "soft", "attribute": None, "direction": "decrease"}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("hide the soft tissue completely")
    assert cmd == _rel("soft", "decrease", "strongly")


def test_parse_command_llm_maps_an_off_schema_strength():
    payload = {"target": "lungs", "attribute": "brightness", "direction": "decrease", "strength": "extremely"}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        assert parse_command_llm("darken the lungs extremely")["strength"] == "strongly"


def test_parse_command_llm_accepts_a_bare_set_command():
    payload = {"attribute": "opacity", "direction": "set", "level": "high", "target": "vessels"}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("high opacity vessels")
    assert cmd == {"target": "vessels", "attribute": "opacity", "direction": "set", "level": "high"}


def test_parse_command_llm_accepts_a_wrapped_show_only_without_strength():
    payload = {"compound": [{"attribute": None, "direction": "show_only", "target": "skeleton"}]}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("show only bones")
    assert cmd["direction"] == "show_only" and cmd["target"] == "skeleton"


def test_parse_command_llm_not_an_instruction_raises():
    with patch("commands.urlopen", return_value=_fake_response({"none": True})):
        with pytest.raises(ValueError):
            parse_command_llm("thank you.")


def test_parse_command_llm_not_an_instruction_still_tries_the_rule_parser():
    with patch("commands.urlopen", return_value=_fake_response({"none": True})):
        assert parse_command_llm("start over")["direction"] == "reset"


@pytest.mark.parametrize("text,payload", [
    ("bye.", {"target": None, "attribute": None, "direction": "reset", "strength": None}),
    ("you", {"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": "moderately"}),
])
def test_parse_command_llm_refuses_a_command_the_text_does_not_ask_for(text, payload):
    reasons = []
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        with pytest.raises(ValueError):
            parse_command_llm(text, _on_fallback=reasons.append)
    assert reasons and reasons[0].startswith("ungrounded")


def test_lay_anatomy_words_from_the_lexicon_name_classes():
    assert parse_command_rule("show only the ribcage")["target"] == "skeleton"
    assert parse_command_rule("less guts")["target"] == "soft"
    assert parse_command_rule("add vasculature")["target"] == "vessels"


def test_new_strength_words_from_the_lexicon():
    assert parse_command_rule("a smidge more spleen")["strength"] == "slightly"


def test_retired_class_raises_its_own_error_type():
    with pytest.raises(commands.RetiredClassError, match="no longer a supported class"):
        parse_command_rule("show more fat")


@pytest.mark.parametrize("text, expected", [
    ("mute the bones", ("skeleton", "opacity", "decrease", "strongly")),
    ("make the liver translucent", ("liver", "opacity", "decrease", "moderately")),
    ("tighter peak on the skeleton", ("skeleton", "width", "decrease", "moderately")),
    ("crisper kidneys", ("kidneys", "width", "decrease", "moderately")),
    ("bring back the lungs", ("lungs", "opacity", "increase", "moderately")),
    ("kill the soft tissue", ("soft", "opacity", "decrease", "strongly")),
    ("ghost the muscles a bit", ("soft", "opacity", "decrease", "slightly")),
])
def test_lexicon_cue_phrases(text, expected):
    cmd = parse_command_rule(text)
    assert (cmd["target"], cmd["attribute"], cmd["direction"], cmd["strength"]) == expected


def test_show_only_and_reset_cues():
    assert parse_command_rule("solo vessels")["direction"] == "show_only"
    assert parse_command_rule("focus on the liver")["target"] == "liver"
    assert parse_command_rule("undo everything")["direction"] == "reset"


def _llm_stub(monkeypatch, answer=None, fail=False):
    calls = []

    def fake(text, model=None, host=None, _on_fallback=None):
        calls.append(text)
        if fail:
            if _on_fallback:
                _on_fallback("request failed: <urlopen error refused>")
            return commands.parse_command_rule(text)
        return answer
    monkeypatch.setattr(commands, "parse_command_llm", fake)
    return calls


def test_hybrid_answers_covered_sentences_without_the_llm(monkeypatch):
    calls = _llm_stub(monkeypatch)
    cmd, meta = commands.parse_command_hybrid("Darkin' Skeleton")
    assert cmd["target"] == "skeleton" and cmd["direction"] == "decrease"
    assert meta["path"] == "rule" and meta["normalized"] == "darken Skeleton"
    assert calls == []


def test_hybrid_asks_the_llm_when_words_are_left_over(monkeypatch):
    answer = {"target": "skeleton", "attribute": "opacity", "direction": "show_only", "strength": None}
    calls = _llm_stub(monkeypatch, answer=answer)
    cmd, meta = commands.parse_command_hybrid("everything except the skeleton should go away")
    assert cmd == answer and meta["path"] == "llm"
    assert meta["uncovered"] == ["except", "go", "away"]
    assert len(calls) == 1


def test_hybrid_reports_llm_offline_when_nothing_parses(monkeypatch):
    _llm_stub(monkeypatch, fail=True)
    with pytest.raises(ValueError, match="LLM offline"):
        commands.parse_command_hybrid("everything except the skeleton should go away")


def test_hybrid_refuses_retired_classes_without_the_llm(monkeypatch):
    calls = _llm_stub(monkeypatch)
    with pytest.raises(commands.RetiredClassError):
        commands.parse_command_hybrid("show more fat")
    assert calls == []


def test_hybrid_logs_every_utterance(monkeypatch):
    _llm_stub(monkeypatch)
    commands.parse_command_hybrid("add kidneys")
    with pytest.raises(ValueError):
        commands.parse_command_hybrid("show more fat")
    rows = [json.loads(line) for line in open(commands.UTTERANCE_LOG_PATH)]
    assert [r["text"] for r in rows] == ["add kidneys", "show more fat"]
    assert rows[0]["path"] == "rule" and rows[0]["cmd"]["target"] == "kidneys"
    assert rows[1]["cmd"] is None and "no longer a supported class" in rows[1]["error"]
    assert rows[0]["parse_ms"] >= 0


def test_with_meta_hybrid_carries_the_path(monkeypatch):
    _llm_stub(monkeypatch)
    cmd, meta = commands.parse_command_with_meta("add kidneys", parser="hybrid")
    assert meta["parser_requested"] == "hybrid" and meta["parser_used"] == "rule"
    assert meta["normalized"] == "add kidneys"


@pytest.mark.parametrize("text, attribute, direction", [
    ("make the lungs less transparent", "opacity", "increase"),
    ("the bones are too transparent", "opacity", "increase"),
    ("the bones are too solid", "opacity", "decrease"),
    ("make the liver less dim", "brightness", "increase"),
    ("the liver is too dim", "brightness", "increase"),
    ("make the lungs more transparent", "opacity", "decrease"),
    ("less opaque bones", "opacity", "decrease"),
])
def test_negators_before_a_cue_flip_its_direction(text, attribute, direction):
    cmd = parse_command_rule(text)
    assert (cmd["attribute"], cmd["direction"]) == (attribute, direction)


def test_hybrid_sends_raw_text_to_the_llm_and_keeps_its_own_rule_parse_on_fallback(monkeypatch):
    # The raw text does not parse ("Leitenlangs"), the normalized one does,
    # but "brother" stays uncovered -- so the LLM is asked, with the raw text.
    calls = _llm_stub(monkeypatch, fail=True)
    cmd, meta = commands.parse_command_hybrid("Leitenlangs please brother")
    assert calls == ["Leitenlangs please brother"]
    assert meta["normalized"] == "lighten lungs please brother" and meta["uncovered"] == ["brother"]
    assert cmd["target"] == "lungs" and cmd["attribute"] == "brightness" and cmd["direction"] == "increase"
    assert meta["path"] == "rule" and meta["fallback"].startswith("request failed")


def test_hybrid_prefers_its_normalized_rule_parse_when_the_llm_falls_back(monkeypatch):
    def fake(text, model=None, host=None, _on_fallback=None):
        _on_fallback("invalid schema: {}")
        return {"target": "heart", "attribute": "opacity", "direction": "increase", "strength": "moderately"}
    monkeypatch.setattr(commands, "parse_command_llm", fake)
    cmd, meta = commands.parse_command_hybrid("Leitenlangs please brother")
    assert cmd["target"] == "lungs" and meta["path"] == "rule" and meta["fallback"] == "invalid schema: {}"


def test_hybrid_reports_llm_offline_when_the_request_really_fails():
    with patch("commands.urlopen", side_effect=OSError("refused")):
        with pytest.raises(ValueError, match="LLM offline"):
            commands.parse_command_hybrid("everything except the skeleton should go away")
    rows = [json.loads(line) for line in open("out/llm_requests.jsonl")]
    assert len(rows) == 1
    assert rows[0]["final_cmd"] is None and rows[0]["fell_back"]
    assert rows[0]["fallback_reason"].startswith("request failed")


def test_hybrid_rejection_message_keeps_the_reason(monkeypatch):
    def fake(text, model=None, host=None, _on_fallback=None):
        _on_fallback("model: not an instruction")
        return commands.parse_command_rule(text)
    monkeypatch.setattr(commands, "parse_command_llm", fake)
    with pytest.raises(ValueError, match=r"could not parse 'thank you for watching' \(model: not an instruction\)"):
        commands.parse_command_hybrid("thank you for watching")


def test_hybrid_still_parses_when_the_utterance_log_cannot_be_written(monkeypatch, tmp_path, capsys):
    _llm_stub(monkeypatch)
    monkeypatch.setattr(commands, "UTTERANCE_LOG_PATH", str(tmp_path))  # a directory: open() fails
    cmd, meta = commands.parse_command_hybrid("add kidneys")
    assert cmd["target"] == "kidneys" and meta["path"] == "rule"
    assert "utterance log" in capsys.readouterr().out
