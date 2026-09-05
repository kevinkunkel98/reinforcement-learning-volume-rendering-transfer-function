import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

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
    assert cmd == {"target": "bone", "attribute": "opacity",
                    "direction": "increase", "strength": "strongly"}


def test_parse_decrease_default_strength():
    cmd = parse_command_rule("decrease opacity for fat")
    assert cmd["target"] == "fat"
    assert cmd["direction"] == "decrease"
    assert cmd["strength"] == "moderately"


def test_parse_show_only():
    cmd = parse_command_rule("show only spongy bone")
    assert cmd["direction"] == "show_only"
    assert cmd["target"] == "spongy"


def test_parse_reset():
    cmd = parse_command_rule("reset")
    assert cmd == {"target": None, "attribute": None,
                    "direction": "reset", "strength": None}


def test_parse_unrecognized_raises():
    with pytest.raises(ValueError):
        parse_command_rule("make the skeleton pop")


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


def _fake_response(payload: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps({"response": json.dumps(payload)}).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_parse_command_llm_valid_response():
    payload = {"target": "bone", "attribute": "opacity",
               "direction": "increase", "strength": "strongly"}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("make the skeleton pop")
    assert cmd == payload


def test_parse_command_llm_falls_back_on_connection_error(capsys):
    with patch("commands.urlopen", side_effect=OSError("connection refused")):
        cmd = parse_command_llm("increase opacity for bone strongly")
    assert cmd["target"] == "bone"
    assert cmd["direction"] == "increase"
    assert "falling back to rule parser" in capsys.readouterr().out


def test_parse_command_llm_falls_back_on_invalid_schema(capsys):
    with patch("commands.urlopen", return_value=_fake_response({"foo": "bar"})):
        cmd = parse_command_llm("increase opacity for bone strongly")
    assert cmd["target"] == "bone"
    assert "falling back to rule parser" in capsys.readouterr().out


def test_parse_command_llm_falls_back_when_target_missing_for_non_reset(capsys):
    payload = {"target": None, "attribute": "opacity", "direction": "show_only", "strength": None}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("increase opacity for bone strongly")
    assert cmd["target"] == "bone"
    assert "falling back to rule parser" in capsys.readouterr().out


def test_parse_command_llm_normalizes_alias_target_instead_of_falling_back(capsys):
    payload = {"target": "bones", "attribute": "opacity", "direction": "increase", "strength": "moderately"}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("increase opacity for the bones")
    assert cmd["target"] == "bone"
    assert "falling back to rule parser" not in capsys.readouterr().out


def test_parse_command_llm_logs_successful_request(tmp_path):
    payload = {"target": "bone", "attribute": "opacity", "direction": "increase", "strength": "strongly"}
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
    assert entry["final_cmd"]["target"] == "bone"


# ---------- multi-target show_only ----------

def test_parse_show_only_multiple_tissues():
    cmd = parse_command_rule("show only bone and spongy")
    assert cmd["direction"] == "show_only"
    assert sorted(cmd["target"]) == ["bone", "spongy"]


def test_parse_show_only_single_tissue_still_returns_plain_string():
    cmd = parse_command_rule("show only bone")
    assert cmd["target"] == "bone"


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
    assert cmd == {"target": "bone", "attribute": "opacity", "direction": "set", "level": "high"}


def test_parse_compound_level_command():
    cmd = parse_command_rule("high opacity spongy, low opacity bone")
    assert "compound" in cmd
    by_target = {c["target"]: c["level"] for c in cmd["compound"]}
    assert by_target == {"spongy": "high", "bone": "low"}
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
    payload = {"target": ["bone", "spongy"], "attribute": "opacity",
               "direction": "show_only", "strength": None}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("show bone and spongy")
    assert sorted(cmd["target"]) == ["bone", "spongy"]


def test_parse_command_llm_compound_response():
    payload = {"compound": [
        {"target": "spongy", "attribute": "opacity", "direction": "set", "level": "high"},
        {"target": "bone", "attribute": "opacity", "direction": "set", "level": "low"},
    ]}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("high opacity spongy, low opacity bone")
    assert cmd == payload


def test_parse_command_llm_compound_normalizes_alias_targets(capsys):
    payload = {"compound": [
        {"target": "bones", "attribute": "opacity", "direction": "set", "level": "low"},
    ]}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("low opacity for the bones")
    assert cmd["compound"][0]["target"] == "bone"
    assert "falling back to rule parser" not in capsys.readouterr().out


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
