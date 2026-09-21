import math

import numpy as np
import pytest

import goals
import totalseg
import visibility
from transfer import peak_internal


def _features(skeleton=(0.0, 0.0), lungs=(0.0, 0.0), organs=(0.0, 0.0),
              muscle=(0.0, 0.0), vessels=(0.0, 0.0), other=0.0, coverage=0.5):
    """A raw visibility.features()-shaped dict: each arg is (vis, bright).
    `other` is the unlabeled-tissue vis (visibility.py's "other" bucket) --
    no brightness, since nothing ever targets its colour."""
    values = {"skeleton": skeleton, "lungs": lungs, "organs": organs,
              "muscle": muscle, "vessels": vessels}
    vis = {c: v[0] for c, v in values.items()}
    vis["other"] = other
    return {"vis": vis,
            "bright": {c: v[1] for c, v in values.items()},
            "coverage": coverage}


def _aggregated(skeleton=0.0, lungs=0.0, soft=0.0, vessels=0.0, other=0.0,
                 skeleton_b=0.0, lungs_b=0.0, soft_b=0.0, vessels_b=0.0, coverage=0.5):
    """An aggregate()-shaped dict directly, for tests that don't need aggregate() itself."""
    return {"vis": {"skeleton": skeleton, "lungs": lungs, "soft": soft, "vessels": vessels, "other": other},
            "bright": {"skeleton": skeleton_b, "lungs": lungs_b, "soft": soft_b, "vessels": vessels_b},
            "coverage": coverage}


def test_starting_params_places_one_peak_per_goal_class():
    params = goals.starting_params()
    assert params.shape == (24,)
    for goal_class, index in goals.PEAK_INDEX.items():
        centre = peak_internal(params, index)["center"]
        assert centre == pytest.approx(goals.PEAK_CENTRES_HU[goal_class], abs=1.0)


def test_aggregate_sums_soft_and_weights_its_brightness():
    features = _features(skeleton=(0.3, 0.9), organs=(0.2, 0.4), muscle=(0.6, 0.8))
    aggregated = goals.aggregate(features)
    assert aggregated["vis"]["soft"] == pytest.approx(0.8)
    assert aggregated["bright"]["soft"] == pytest.approx(0.7)
    assert aggregated["vis"]["skeleton"] == pytest.approx(0.3)
    assert aggregated["bright"]["skeleton"] == pytest.approx(0.9)


def test_aggregate_soft_brightness_is_zero_when_invisible():
    features = _features(organs=(0.0, 0.0), muscle=(0.0, 0.0))
    aggregated = goals.aggregate(features)
    assert aggregated["vis"]["soft"] == pytest.approx(0.0)
    assert aggregated["bright"]["soft"] == 0.0


def test_aggregate_passes_through_other():
    """`other` (unlabeled tissue) isn't a goal class, but a transfer function
    can still render it, and `distance` needs it to detect a search that
    hides behind unclassified material instead of answering the instruction."""
    features = _features(skeleton=(0.3, 0.9), other=0.2)
    aggregated = goals.aggregate(features)
    assert aggregated["vis"]["other"] == pytest.approx(0.2)


def test_goal_vector_layout():
    vector = goals.goal_vector({"skeleton": {"vis": 0.3}})
    assert vector.shape == (16,)
    d, m, e, n = vector[0:4], vector[4:8], vector[8:12], vector[12:16]
    skeleton = goals.GOAL_CLASSES.index("skeleton")
    assert d[skeleton] == pytest.approx(0.3)
    assert m[skeleton] == 1.0
    rest = np.delete(vector, [skeleton, 4 + skeleton])
    assert np.all(rest == 0.0)


def test_progress_is_log_change_for_visibility_and_plain_change_for_brightness():
    start = _aggregated(skeleton=0.001, skeleton_b=0.4)
    current = _aggregated(skeleton=0.011, skeleton_b=0.55)
    c, b = goals.progress(start, current)
    assert c["skeleton"] == pytest.approx(0.778, abs=1e-3)
    assert b["skeleton"] == pytest.approx(0.15)


def test_distance_is_zero_when_the_goal_is_met_exactly():
    start = _aggregated(skeleton=0.01)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    new_vis = 10 ** (0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON
    current = _aggregated(skeleton=new_vis)
    assert goals.distance(goal, start, current) == pytest.approx(0.0, abs=1e-9)


def test_distance_penalises_unmentioned_classes_drifting():
    start = _aggregated(skeleton=0.01, lungs=0.05)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    new_vis = 10 ** (0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON
    without_drift = _aggregated(skeleton=new_vis, lungs=0.05)
    with_drift = _aggregated(skeleton=new_vis, lungs=0.08)
    c_lungs = math.log10(0.08 + goals.EPSILON) - math.log10(0.05 + goals.EPSILON)
    # drift beyond KEEP_TOLERANCE is penalised; the tolerated part is free
    extra = goals.LAMBDA_KEEP * max(0.0, abs(c_lungs) - goals.KEEP_TOLERANCE)
    d0 = goals.distance(goal, start, without_drift)
    d1 = goals.distance(goal, start, with_drift)
    assert d1 - d0 == pytest.approx(extra)


def test_distance_tolerates_small_keep_drift():
    """An unmentioned class drifting by exactly KEEP_TOLERANCE costs nothing
    (inaction on it looks the same as a drift that small); drifting by
    2xKEEP_TOLERANCE costs exactly LAMBDA_KEEP * KEEP_TOLERANCE more."""
    start = _aggregated(skeleton=0.01, lungs=0.05)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    new_skeleton = 10 ** (0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON

    def _lungs_at(c_target):
        return 10 ** (c_target + math.log10(0.05 + goals.EPSILON)) - goals.EPSILON

    no_drift = _aggregated(skeleton=new_skeleton, lungs=0.05)
    at_tolerance = _aggregated(skeleton=new_skeleton, lungs=_lungs_at(goals.KEEP_TOLERANCE))
    beyond_tolerance = _aggregated(skeleton=new_skeleton, lungs=_lungs_at(2 * goals.KEEP_TOLERANCE))

    d_no_drift = goals.distance(goal, start, no_drift)
    d_at_tolerance = goals.distance(goal, start, at_tolerance)
    d_beyond = goals.distance(goal, start, beyond_tolerance)

    assert d_at_tolerance == pytest.approx(d_no_drift)
    assert d_beyond - d_no_drift == pytest.approx(goals.LAMBDA_KEEP * goals.KEEP_TOLERANCE)


def test_distance_penalises_other_drifting_beyond_tolerance():
    """`other` (unlabeled tissue) is never nameable by an instruction, so it
    is always an unmentioned class -- drift beyond KEEP_TOLERANCE there costs
    the same LAMBDA_KEEP penalty any other unmentioned class's drift does."""
    start = _aggregated(skeleton=0.01, other=0.05)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    new_vis = 10 ** (0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON
    without_drift = _aggregated(skeleton=new_vis, other=0.05)
    with_drift = _aggregated(skeleton=new_vis, other=0.08)
    c_other = math.log10(0.08 + goals.EPSILON) - math.log10(0.05 + goals.EPSILON)
    extra = goals.LAMBDA_KEEP * max(0.0, abs(c_other) - goals.KEEP_TOLERANCE)
    d0 = goals.distance(goal, start, without_drift)
    d1 = goals.distance(goal, start, with_drift)
    assert d1 - d0 == pytest.approx(extra)


def test_distance_tolerates_small_other_drift():
    start = _aggregated(skeleton=0.01, other=0.05)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    new_skeleton = 10 ** (0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON

    def _other_at(c_target):
        return 10 ** (c_target + math.log10(0.05 + goals.EPSILON)) - goals.EPSILON

    no_drift = _aggregated(skeleton=new_skeleton, other=0.05)
    at_tolerance = _aggregated(skeleton=new_skeleton, other=_other_at(goals.KEEP_TOLERANCE))
    beyond_tolerance = _aggregated(skeleton=new_skeleton, other=_other_at(2 * goals.KEEP_TOLERANCE))

    d_no_drift = goals.distance(goal, start, no_drift)
    d_at_tolerance = goals.distance(goal, start, at_tolerance)
    d_beyond = goals.distance(goal, start, beyond_tolerance)

    assert d_at_tolerance == pytest.approx(d_no_drift)
    assert d_beyond - d_no_drift == pytest.approx(goals.LAMBDA_KEEP * goals.KEEP_TOLERANCE)


def test_distance_penalises_hiding_behind_unlabeled_material():
    """The bug this whole fix exists for: a transfer function that suppresses
    every named class but renders an opaque wall of unclassified tissue
    instead must not score as well as suppressing them cleanly -- otherwise
    a search (or a policy trained on this reward) can satisfy "show only X"
    by rendering unlabeled material instead of X."""
    start = _aggregated(skeleton=0.01, other=0.01)
    goal = goals.goal_vector({"skeleton": {"vis": -goals.HIDE_STRENGTH}})
    hidden_cleanly = _aggregated(skeleton=0.0001, other=0.01)
    hidden_behind_other = _aggregated(skeleton=0.0001, other=0.9)
    d_clean = goals.distance(goal, start, hidden_cleanly)
    d_gamed = goals.distance(goal, start, hidden_behind_other)
    assert d_gamed > d_clean


def test_attainment_is_one_when_reached_and_negative_when_worse():
    start = _aggregated(skeleton=0.01)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    reached_vis = 10 ** (0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON
    reached = _aggregated(skeleton=reached_vis)
    assert goals.attainment(goal, start, reached) == pytest.approx(1.0)

    worse_vis = 10 ** (-0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON
    worse = _aggregated(skeleton=worse_vis)
    assert goals.attainment(goal, start, worse) < 0.0


def test_summarise_attainment_reports_robust_stats_and_ignores_none():
    values = [-17, 0.5, 0.6, 0.7, None]
    summary = goals.summarise_attainment(values)
    assert summary["median"] == pytest.approx(0.55)
    assert summary["mean_clipped"] == pytest.approx(0.2)
    assert summary["mean_raw"] == pytest.approx(-3.8)
    assert summary["share_positive"] == pytest.approx(0.75)
    assert summary["n"] == 4


def test_summarise_attainment_handles_all_none():
    summary = goals.summarise_attainment([None, None])
    assert summary["n"] == 0
    assert summary["median"] is None
    assert summary["mean_clipped"] is None
    assert summary["mean_raw"] is None
    assert summary["share_positive"] is None


def test_is_useless_detects_empty_and_opaque_states():
    empty = _aggregated(skeleton=0.2, coverage=0.0)
    assert goals.is_useless(empty) is True

    opaque = _aggregated(skeleton=0.0, lungs=0.0, soft=0.0, vessels=0.0, coverage=1.0)
    assert goals.is_useless(opaque) is True

    normal = _aggregated(skeleton=0.1, lungs=0.05, soft=0.2, vessels=0.02, coverage=0.5)
    assert goals.is_useless(normal) is False


@pytest.mark.slow
def test_real_check_starting_params_on_ts_s1379():
    model = visibility.for_volume("ts_s1379")
    aggregated = goals.aggregate(model.features(goals.starting_params()))
    print(aggregated["vis"])
    assert aggregated["vis"]["skeleton"] > 0.0
    assert aggregated["vis"]["lungs"] > 0.0
    assert aggregated["vis"]["soft"] > 0.0
    assert goals.is_useless(aggregated) is False


# --- Task 2: instruction sampler and text ------------------------------------

class _StubModel:
    """A model whose solo_max is fixed, for tests that don't need real visibility."""

    def __init__(self, solo=0.5):
        self._solo = solo

    def solo_max(self, name):
        return self._solo


def _all_classes_present():
    return ["skeleton", "lungs", "organs", "muscle", "vessels"]


def test_goal_classes_exclude_vessels_without_contrast(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: False)
    assert "vessels" not in goals.goal_classes_for_volume("fake")


def test_goal_classes_include_vessels_with_contrast(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    assert "vessels" in goals.goal_classes_for_volume("fake")


def test_goal_classes_exclude_absent_classes(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present",
                        lambda name: ["skeleton", "organs", "muscle", "vessels"])
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    supported = goals.goal_classes_for_volume("fake")
    assert "lungs" not in supported
    assert "skeleton" in supported and "soft" in supported and "vessels" in supported


def test_goal_classes_for_volume_falls_back_for_volumes_without_anatomical_labels():
    # ct_chest is a real Slicer CT, not a ts_* TotalSegmentator subject, so it
    # has no anatomy labelling -- goal_classes_for_volume must still return
    # what visibility's intensity fallback can measure instead of raising.
    supported = goals.goal_classes_for_volume("ct_chest")
    assert set(supported) == {"skeleton", "lungs", "soft"}


def test_goal_classes_for_volume_fallback_never_includes_vessels():
    assert "vessels" not in goals.goal_classes_for_volume("ct_chest")
    assert "vessels" not in goals.goal_classes_for_volume("mri_head")


def test_sample_instruction_mix_matches_the_declared_shares(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    model = _StubModel()
    start = _aggregated(skeleton=0.05, lungs=0.05, soft=0.05, vessels=0.05)
    rng = np.random.default_rng(0)
    counts = {}
    n = 400
    for _ in range(n):
        instruction = goals.sample_instruction("fake", model, start, rng)
        counts[instruction["kind"]] = counts.get(instruction["kind"], 0) + 1
    for kind, share in goals.INSTRUCTION_MIX:
        assert abs(counts.get(kind, 0) / n - share) <= 0.07, (kind, counts)


def test_sampled_goals_only_target_supported_classes(monkeypatch):
    # non-contrast: vessels present in labels but not targetable
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: False)
    model = _StubModel()
    start = _aggregated(skeleton=0.05, lungs=0.05, soft=0.05, vessels=0.05)
    rng = np.random.default_rng(1)
    supported = set(goals.goal_classes_for_volume("fake"))
    assert "vessels" not in supported
    for _ in range(200):
        instruction = goals.sample_instruction("fake", model, start, rng)
        assert set(instruction["targets"]) <= supported


def test_show_only_hides_the_other_classes(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    model = _StubModel()
    start = _aggregated(skeleton=0.05, lungs=0.05, soft=0.05, vessels=0.05)
    rng = np.random.default_rng(2)
    supported = goals.goal_classes_for_volume("fake")
    found = False
    for _ in range(400):
        instruction = goals.sample_instruction("fake", model, start, rng)
        if instruction["kind"] != "show_only":
            continue
        found = True
        shown = [c for c, t in instruction["targets"].items() if t.get("vis", 0.0) > 0.0]
        for goal_class in supported:
            if goal_class in shown:
                continue
            assert instruction["targets"][goal_class]["vis"] <= -goals.VISIBILITY_STRENGTH["strongly"]
    assert found


def test_absolute_level_target_is_relative_to_solo_max():
    model = _StubModel(solo=0.5)
    delta = goals._absolute_target_delta(model, "skeleton", "high", 0.05)
    expected = math.log10(0.4 + goals.EPSILON) - math.log10(0.05 + goals.EPSILON)
    assert delta == pytest.approx(expected)


def test_every_sampled_instruction_has_text_and_a_goal_vector(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    model = _StubModel()
    start = _aggregated(skeleton=0.05, lungs=0.05, soft=0.05, vessels=0.05)
    rng = np.random.default_rng(3)
    for _ in range(50):
        instruction = goals.sample_instruction("fake", model, start, rng)
        assert instruction["text"]
        assert instruction["goal"].shape == (16,)
        assert instruction["goal"][4:8].sum() + instruction["goal"][12:16].sum() > 0.0


def test_sampling_is_deterministic_for_a_seed(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    model = _StubModel()
    start = _aggregated(skeleton=0.05, lungs=0.05, soft=0.05, vessels=0.05)

    def _run():
        rng = np.random.default_rng(42)
        return [goals.sample_instruction("fake", model, start, rng) for _ in range(20)]

    first, second = _run(), _run()
    for a, b in zip(first, second):
        assert a["kind"] == b["kind"]
        assert a["text"] == b["text"]
        assert np.array_equal(a["goal"], b["goal"])


@pytest.mark.slow
def test_real_check_sample_instructions_on_contrast_and_non_contrast():
    for name in ("ts_s1379", "ts_s0357"):
        model = visibility.for_volume(name)
        start = goals.aggregate(model.features(goals.starting_params()))
        rng = np.random.default_rng(0)
        print(f"\n--- {name} (contrast={totalseg.is_contrast(name)}) ---")
        for _ in range(12):
            instruction = goals.sample_instruction(name, model, start, rng)
            nonzero = {k: v for k, v in instruction["targets"].items()}
            print(instruction["kind"], "|", instruction["text"], "|", nonzero)
            if not totalseg.is_contrast(name):
                assert "vessels" not in instruction["targets"]


# --- Task 2: commands become goals -------------------------------------------

def _patch_contrast_volume(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)


def test_goal_from_command_relative_increase_matches_visibility_strength(monkeypatch):
    _patch_contrast_volume(monkeypatch)
    model = _StubModel()
    start = _aggregated(skeleton=0.05)
    cmd = {"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": "strongly"}
    result = goals.goal_from_command(cmd, model, start, volume="fake")
    assert result["kind"] == "relative"
    assert result["targets"] == {"skeleton": {"vis": goals.VISIBILITY_STRENGTH["strongly"]}}
    assert result["text"]
    np.testing.assert_array_equal(result["goal"], goals.goal_vector(result["targets"]))


def test_goal_from_command_relative_decrease_is_negative(monkeypatch):
    _patch_contrast_volume(monkeypatch)
    model = _StubModel()
    start = _aggregated(soft=0.05)
    cmd = {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "slightly"}
    result = goals.goal_from_command(cmd, model, start, volume="fake")
    assert result["targets"] == {"soft": {"vis": -goals.VISIBILITY_STRENGTH["slightly"]}}


def test_goal_from_command_brightness_matches_brightness_strength(monkeypatch):
    _patch_contrast_volume(monkeypatch)
    model = _StubModel()
    start = _aggregated(lungs=0.05)
    cmd = {"target": "lungs", "attribute": "brightness", "direction": "increase", "strength": "moderately"}
    result = goals.goal_from_command(cmd, model, start, volume="fake")
    assert result["kind"] == "brightness"
    assert result["targets"] == {"lungs": {"bright": goals.BRIGHTNESS_STRENGTH["moderately"]}}


def test_goal_from_command_absolute_level_matches_absolute_target_delta(monkeypatch):
    _patch_contrast_volume(monkeypatch)
    model = _StubModel(solo=0.5)
    start = _aggregated(vessels=0.05)
    cmd = {"target": "vessels", "attribute": "opacity", "direction": "set", "level": "high"}
    result = goals.goal_from_command(cmd, model, start, volume="fake")
    assert result["kind"] == "absolute"
    expected = goals._absolute_target_delta(model, "vessels", "high", 0.05)
    assert result["targets"]["vessels"]["vis"] == pytest.approx(expected)


def test_goal_from_command_show_only_hides_the_rest(monkeypatch):
    _patch_contrast_volume(monkeypatch)
    model = _StubModel()
    start = _aggregated(skeleton=0.05, lungs=0.05, soft=0.05, vessels=0.05)
    cmd = {"target": ["skeleton", "lungs"], "attribute": "opacity", "direction": "show_only", "strength": None}
    result = goals.goal_from_command(cmd, model, start, volume="fake")
    assert result["kind"] == "show_only"
    assert result["targets"]["skeleton"]["vis"] == goals.HIDE_STRENGTH
    assert result["targets"]["lungs"]["vis"] == goals.HIDE_STRENGTH
    assert result["targets"]["soft"]["vis"] == -goals.HIDE_STRENGTH
    assert result["targets"]["vessels"]["vis"] == -goals.HIDE_STRENGTH


def test_goal_from_command_show_only_single_string_target(monkeypatch):
    _patch_contrast_volume(monkeypatch)
    model = _StubModel()
    start = _aggregated(skeleton=0.05, lungs=0.05, soft=0.05, vessels=0.05)
    cmd = {"target": "skeleton", "attribute": "opacity", "direction": "show_only", "strength": None}
    result = goals.goal_from_command(cmd, model, start, volume="fake")
    assert result["targets"]["skeleton"]["vis"] == goals.HIDE_STRENGTH
    assert result["targets"]["lungs"]["vis"] == -goals.HIDE_STRENGTH


def test_goal_from_command_compound_merges_subcommands(monkeypatch):
    _patch_contrast_volume(monkeypatch)
    model = _StubModel()
    start = _aggregated(skeleton=0.05, soft=0.05)
    cmd = {"compound": [
        {"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": "moderately"},
        {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "slightly"},
    ]}
    result = goals.goal_from_command(cmd, model, start, volume="fake")
    assert result["kind"] == "compound"
    assert result["targets"] == {
        "skeleton": {"vis": goals.VISIBILITY_STRENGTH["moderately"]},
        "soft": {"vis": -goals.VISIBILITY_STRENGTH["slightly"]},
    }
    np.testing.assert_array_equal(result["goal"], goals.goal_vector(result["targets"]))


@pytest.mark.parametrize("cmd", [
    {"target": None, "attribute": None, "direction": "reset", "strength": None},
    {"target": "skeleton", "attribute": "width", "direction": "increase", "strength": "strongly"},
    {"target": "skeleton", "attribute": "center", "direction": "increase", "strength": "slightly"},
    {"camera": {"action": "rotate", "direction": "left", "strength": "moderately"}},
])
def test_goal_from_command_non_goal_commands_raise_pointing_at_apply_command(monkeypatch, cmd):
    _patch_contrast_volume(monkeypatch)
    model = _StubModel()
    start = _aggregated(skeleton=0.05)
    with pytest.raises(ValueError, match="apply_command"):
        goals.goal_from_command(cmd, model, start, volume="fake")


def test_goal_from_command_vessels_without_contrast_raises(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: False)
    model = _StubModel()
    start = _aggregated(vessels=0.05)
    cmd = {"target": "vessels", "attribute": "opacity", "direction": "increase", "strength": "moderately"}
    with pytest.raises(ValueError):
        goals.goal_from_command(cmd, model, start, volume="fake")


def test_goal_from_command_class_absent_from_volume_raises(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present",
                        lambda name: ["skeleton", "organs", "muscle", "vessels"])
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    model = _StubModel()
    start = _aggregated(lungs=0.05)
    cmd = {"target": "lungs", "attribute": "opacity", "direction": "increase", "strength": "moderately"}
    with pytest.raises(ValueError):
        goals.goal_from_command(cmd, model, start, volume="fake")


# --- reachability gate --------------------------------------------------------

class _PerClassModel:
    """A model whose solo_max differs per measured class, so a goal class can
    be labelled-present yet unreachable -- the real case on this dataset, where
    26 of 30 volumes are abdomen/pelvis scans whose lungs are clipped bases
    behind an opaque body wall (ceiling ~0.0015, i.e. 0.15% of the image)."""

    def __init__(self, per_class):
        self._per_class = per_class

    def solo_max(self, name):
        return self._per_class.get(name, 0.0)


def _reachable_everything():
    return {m: 0.1 for m in visibility.CLASSES}


def test_unreachable_class_is_not_a_goal_class(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    per_class = _reachable_everything()
    per_class["lungs"] = 0.0003          # below VISIBLE_CEILING: invisible in a render
    model = _PerClassModel(per_class)

    got = goals.reachable_goal_classes("ts_fake", model)

    assert "lungs" not in got
    assert "skeleton" in got and "soft" in got


def test_reachable_classes_match_goal_classes_when_everything_is_visible(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    model = _PerClassModel(_reachable_everything())

    assert goals.reachable_goal_classes("ts_fake", model) == goals.goal_classes_for_volume("ts_fake")


def test_reachability_sums_the_measured_classes_behind_a_goal_class(monkeypatch):
    """`soft` is organs + muscle: neither alone need clear the threshold."""
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    per_class = {m: 0.0 for m in visibility.CLASSES}
    per_class["skeleton"] = 0.1
    per_class["organs"] = goals.VISIBLE_CEILING * 0.6
    per_class["muscle"] = goals.VISIBLE_CEILING * 0.6
    model = _PerClassModel(per_class)

    assert "soft" in goals.reachable_goal_classes("ts_fake", model)


def test_sampled_instructions_never_name_an_unreachable_class(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    per_class = _reachable_everything()
    per_class["lungs"] = 0.0003
    model = _PerClassModel(per_class)
    rng = np.random.default_rng(0)
    start = {"vis": {c: 0.01 for c in goals.GOAL_CLASSES},
             "bright": {c: 0.5 for c in goals.GOAL_CLASSES}, "coverage": 0.5}

    for _ in range(200):
        instruction = goals.sample_instruction("ts_fake", model, start, rng)
        assert "lungs" not in instruction["targets"], instruction["text"]
        assert "lung" not in instruction["text"], instruction["text"]


def test_goal_from_command_rejects_a_class_the_scan_cannot_show(monkeypatch):
    """Typing "more lungs" on an abdomen scan must say so rather than build a
    goal nothing can reach -- otherwise policy mode silently returns a transfer
    function that cannot possibly satisfy the request."""
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    per_class = _reachable_everything()
    per_class["lungs"] = 0.0003
    model = _PerClassModel(per_class)
    start = {"vis": {c: 0.01 for c in goals.GOAL_CLASSES},
             "bright": {c: 0.5 for c in goals.GOAL_CLASSES}, "coverage": 0.5}
    command = {"target": "lungs", "attribute": "opacity",
               "direction": "increase", "strength": "slightly"}

    with pytest.raises(ValueError, match="lungs"):
        goals.goal_from_command(command, model, start, volume="ts_fake")


def test_goal_from_command_allows_a_class_the_scan_can_show(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    model = _PerClassModel(_reachable_everything())
    start = {"vis": {c: 0.01 for c in goals.GOAL_CLASSES},
             "bright": {c: 0.5 for c in goals.GOAL_CLASSES}, "coverage": 0.5}
    command = {"target": "skeleton", "attribute": "opacity",
               "direction": "increase", "strength": "slightly"}

    got = goals.goal_from_command(command, model, start, volume="ts_fake")

    assert "skeleton" in got["targets"]
