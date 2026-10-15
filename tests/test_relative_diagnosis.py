import math

import goals
from tools.relative_diagnosis import solvable


def test_a_decrease_is_solvable_only_if_its_target_stays_above_the_log_floor():
    eps = goals.EPSILON
    # "slightly less" (0.15) needs vis + eps to shrink by 10**0.15 without going below eps.
    needed = (10 ** 0.15 - 1) * eps
    assert solvable("decrease", 0.15, needed * 1.01)
    assert not solvable("decrease", 0.15, needed * 0.99)
    assert not solvable("decrease", 0.6, 0.0)


def test_an_increase_needs_a_visible_class_to_start_from():
    assert solvable("increase", 0.6, goals.EPSILON)
    assert not solvable("increase", 0.15, 0.0)


def test_floor_matches_goals_progress():
    # The bound used above is exactly the most negative change goals.progress can report.
    start = {"vis": {"liver": 0.002}, "bright": {}}
    final = {"vis": {"liver": 0.0}, "bright": {}}
    c, _ = goals.progress(start, final)
    assert math.isclose(c["liver"], math.log10(goals.EPSILON) - math.log10(0.002 + goals.EPSILON))
