"""The objective: goals, distance and attainment, and what a scan can show."""
import math

import numpy as np
import pytest

import commands
import goals
import transfer


def _agg(vis, bright=None):
    vis = {c: vis.get(c, 0.05) for c in goals.GOAL_CLASSES} | {"other": vis.get("other", 0.05)}
    return {"vis": vis, "bright": {c: (bright or {}).get(c, 0.5) for c in goals.GOAL_CLASSES}, "coverage": 0.5}


def test_attainment_is_one_at_the_goal_and_zero_without_change():
    start = _agg({"liver": 0.01})
    targets = {"liver": {"vis": goals.VISIBILITY_STRENGTH["moderately"]}}
    goal = goals.goal_vector(targets)
    reached = _agg({"liver": 10 ** (math.log10(0.01 + goals.EPSILON) + 0.3) - goals.EPSILON})
    assert goals.attainment(goal, start, reached) == pytest.approx(1.0)
    assert goals.attainment(goal, start, start) == pytest.approx(0.0)


def test_unmentioned_drift_is_free_within_the_tolerance_and_costs_beyond_it():
    start = _agg({"liver": 0.01, "heart": 0.02})
    goal = goals.goal_vector({"liver": {"vis": 0.3}})
    small = _agg({"liver": 0.01, "heart": 0.02 * 10 ** 0.1})
    large = _agg({"liver": 0.01, "heart": 0.02 * 10 ** 0.5})
    assert goals.distance(goal, start, small) == pytest.approx(goals.distance(goal, start, start))
    assert goals.distance(goal, start, large) > goals.distance(goal, start, start)


def test_show_only_weights_balance_the_two_groups():
    weights = goals.show_only_weights({"lungs"}, goals.GOAL_CLASSES)
    assert weights["lungs"] == 1.0
    assert sum(w for c, w in weights.items() if c != "lungs") == pytest.approx(1.0)


def test_goal_from_command(model):
    start = goals.aggregate(model.features(goals.starting_params()))
    cmd = commands.parse_command_with_meta("more liver and less lungs", parser="rule")[0]
    with pytest.raises(ValueError):    # a camera command is not a goal
        goals.goal_from_command({"camera": {"azimuth": 10}}, model, start, "ts_s9001")
    instruction = goals.goal_from_command(commands.parse_command_with_meta("a bit more liver")[0],
                                          model, start, "ts_s9001")
    assert instruction["kind"] == "relative" and instruction["targets"] == {"liver": {"vis": 0.15}}
    assert goals.goal_from_command(cmd, model, start, "ts_s9001")["kind"] == "compound"


def test_vessels_are_a_goal_only_with_contrast():
    assert "vessels" in goals.goal_classes_for_volume("ts_s9001")
    assert "vessels" not in goals.goal_classes_for_volume("ts_s9002")


def test_sampled_instructions_are_well_posed_and_seeded(model):
    params = goals.starting_params()
    start = goals.aggregate(model.features(params))
    a = goals.sample_instruction("ts_s9001", model, start, np.random.default_rng(3))
    b = goals.sample_instruction("ts_s9001", model, start, np.random.default_rng(3))
    assert a["text"] == b["text"] and np.array_equal(a["goal"], b["goal"])
    kinds = {goals.sample_instruction("ts_s9001", model, start, np.random.default_rng(s))["kind"] for s in range(40)}
    assert kinds == {"relative", "compound", "show_only", "absolute", "brightness"}
    shown = goals.well_posed(goals.goal_from_command(
        commands.parse_command_with_meta("show only the lungs")[0], model, start, "ts_s9001"), model, params, start)
    assert shown["weights"]["other"] == 0.0        # hide_fully: the unlabelled shell may fade


def test_occluders_of_a_hidden_target_are_found(model):
    params = goals.starting_params()
    lungs = transfer.ANATOMICAL_PEAK_INDEX["lungs"] * transfer.PARAMS_PER_PEAK
    params[lungs + 2] = 1.0                        # opaque lungs in front of the heart
    instruction = {"targets": {"heart": {"vis": 0.3}}}
    assert "lungs" in goals.find_occluders(model, params, instruction)
