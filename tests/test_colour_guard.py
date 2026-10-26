import numpy as np
import pytest

import goals
import transfer
from rl.colour_guard import colour_guard, hue_matches, luminance, scale_hue

P = transfer.PARAMS_PER_PEAK
REFERENCE = goals.starting_params("v3")


def _rgb(params, goal_class):
    return np.asarray(transfer.peak_internal(params, transfer.ANATOMICAL_PEAK_INDEX[goal_class])["rgb"])


def _set_rgb(params, goal_class, rgb):
    params = np.asarray(params, dtype=np.float64).copy()
    base = transfer.ANATOMICAL_PEAK_INDEX[goal_class] * P
    params[base + 3:base + 6] = np.asarray(rgb, dtype=np.float64) * 2.0 - 1.0
    return params


def _set_height(params, goal_class, height):
    params = np.asarray(params, dtype=np.float64).copy()
    params[transfer.ANATOMICAL_PEAK_INDEX[goal_class] * P + 2] = height * 2.0 - 1.0
    return params


def _show_only_skeleton_goal():
    # vis targets on every class, no brightness anywhere.
    targets = {c: {"vis": -2.0} for c in goals.GOAL_CLASSES if c != "skeleton"}
    targets["skeleton"] = {"vis": 0.5}
    return goals.goal_vector(targets)


def _brightness_goal(goal_class, amount):
    return goals.goal_vector({goal_class: {"bright": amount}})


def test_luminance_is_the_visibility_models_rec709():
    import visibility
    rgb = np.array([0.3, 0.6, 0.9])
    assert luminance(rgb) == pytest.approx(float(rgb @ visibility.LUMINANCE))


@pytest.mark.parametrize("level", [0.0, 0.1, 0.36, 0.6, 0.9, 1.0])
def test_scale_hue_hits_the_luminance_exactly_and_stays_in_range(level):
    heart = np.array(transfer.ANATOMICAL_COLOURS["heart"])
    rgb = scale_hue(heart, level)
    assert luminance(rgb) == pytest.approx(level, abs=1e-9)
    assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0 + 1e-12)


def test_scale_hue_keeps_red_red_when_brightened():
    heart = np.array(transfer.ANATOMICAL_COLOURS["heart"])
    bright = scale_hue(heart, 0.7)
    assert bright[0] == pytest.approx(1.0)
    assert bright[0] - bright[1] > 0.3          # still clearly red, not white
    np.testing.assert_allclose(scale_hue(heart, 1.0), [1.0, 1.0, 1.0])


def test_hue_matches_rejects_white_grey_and_black_for_red():
    heart = np.array(transfer.ANATOMICAL_COLOURS["heart"])
    assert hue_matches(heart, heart)
    assert hue_matches(scale_hue(heart, 0.2), heart)
    assert not hue_matches(np.array([1.0, 1.0, 1.0]), heart)
    assert not hue_matches(np.array([0.8, 0.8, 0.8]), heart)
    assert not hue_matches(np.array([0.0, 0.0, 0.0]), heart)


def test_hidden_class_whitened_by_the_policy_keeps_its_pre_step_colour():
    """The live bug: "show only skeleton" whitened the hidden heart."""
    start = REFERENCE.copy()
    answer = _set_height(_set_rgb(start, "heart", (1.0, 1.0, 1.0)), "heart", 0.0)
    guarded = colour_guard(_show_only_skeleton_goal(), start, answer, REFERENCE)
    np.testing.assert_allclose(_rgb(guarded, "heart"), _rgb(start, "heart"))
    # opacity (and centre, width) are the answer's
    base = transfer.ANATOMICAL_PEAK_INDEX["heart"] * P
    np.testing.assert_array_equal(guarded[base:base + 3], answer[base:base + 3])


def test_pre_step_colour_with_the_reference_hue_is_kept_exactly():
    """A darker, same-hue heart before the step stays exactly as it was."""
    heart = np.array(transfer.ANATOMICAL_COLOURS["heart"])
    start = _set_rgb(REFERENCE, "heart", scale_hue(heart, 0.2))
    answer = _set_rgb(start, "heart", (0.0, 0.0, 0.0))
    guarded = colour_guard(_show_only_skeleton_goal(), start, answer, REFERENCE)
    np.testing.assert_allclose(_rgb(guarded, "heart"), _rgb(start, "heart"))


@pytest.mark.parametrize("garbled", [(1.0, 1.0, 1.0), (0.8, 0.8, 0.8), (0.0, 0.0, 0.0)])
def test_garbled_pre_step_colour_comes_back_at_the_reference(garbled):
    """A heart already white/grey/black before the step (an earlier,
    unguarded answer) returns to the reference colour, at the reference's
    luminance -- not white's or black's."""
    start = _set_rgb(REFERENCE, "heart", garbled)
    guarded = colour_guard(_show_only_skeleton_goal(), start, start, REFERENCE)
    np.testing.assert_allclose(_rgb(guarded, "heart"), _rgb(REFERENCE, "heart"))


def test_brightness_target_keeps_the_answers_luminance_with_the_reference_hue():
    start = REFERENCE.copy()
    answer = _set_rgb(start, "heart", (0.6, 0.6, 0.6))   # brightened to grey, hue lost
    guarded = colour_guard(_brightness_goal("heart", 0.2), start, answer, REFERENCE)
    rgb = _rgb(guarded, "heart")
    assert luminance(rgb) == pytest.approx(luminance(_rgb(answer, "heart")), abs=1e-9)
    np.testing.assert_allclose(rgb, scale_hue(transfer.ANATOMICAL_COLOURS["heart"], 0.6))
    assert rgb[0] - rgb[1] > 0.3                          # red again


def test_brightness_target_only_frees_that_class():
    start = REFERENCE.copy()
    answer = _set_rgb(_set_rgb(start, "heart", (1.0, 0.9, 0.9)), "skeleton", (0.0, 0.0, 0.0))
    guarded = colour_guard(_brightness_goal("heart", 0.4), start, answer, REFERENCE)
    np.testing.assert_allclose(_rgb(guarded, "skeleton"), _rgb(start, "skeleton"))


def test_only_rgb_is_touched():
    rng = np.random.default_rng(0)
    start = REFERENCE.copy()
    answer = np.clip(start + rng.uniform(-0.5, 0.5, size=start.shape), -1.0, 1.0)
    guarded = colour_guard(_show_only_skeleton_goal(), start, answer, REFERENCE)
    for i in range(transfer.N_PEAKS):
        np.testing.assert_array_equal(guarded[i * P:i * P + 3], answer[i * P:i * P + 3])
    assert not np.array_equal(guarded, answer)


def test_inputs_are_not_mutated_and_guard_is_idempotent():
    start = REFERENCE.copy()
    answer = _set_rgb(start, "liver", (1.0, 1.0, 1.0))
    answer_copy = answer.copy()
    once = colour_guard(_show_only_skeleton_goal(), start, answer, REFERENCE)
    np.testing.assert_array_equal(answer, answer_copy)
    twice = colour_guard(_show_only_skeleton_goal(), once, once, REFERENCE)
    np.testing.assert_allclose(once, twice)


def test_guard_leaves_brightness_features_unchanged_for_brightness_targets():
    """The visibility model's `bright` is Rec. 709 luminance of the peak's rgb
    (label-scoped: one peak per class), so keeping luminance keeps it."""
    import visibility
    start = REFERENCE.copy()
    answer = _set_rgb(start, "heart", (1.0, 0.9, 0.9))
    guarded = colour_guard(_brightness_goal("heart", 0.4), start, answer, REFERENCE)
    opacity_answer, lum_answer = visibility.label_transfer_tables(answer)
    opacity_guarded, lum_guarded = visibility.label_transfer_tables(guarded)
    np.testing.assert_allclose(lum_guarded, lum_answer, atol=1e-6)
    np.testing.assert_array_equal(opacity_guarded, opacity_answer)
