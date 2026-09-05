"""Tests for the pure-logic pieces of rl/eval.py -- the arithmetic behind
the thesis's headline comparison numbers, tested without any RL/rendering
dependency."""
from rl.eval import _steps_to_90pct


def test_reaches_threshold_partway_through_increasing_sequence():
    # start=0.0, best=1.0 -> 90% threshold is 0.9; reached at index 3
    fractions = [0.0, 0.5, 0.8, 0.95, 1.0]
    assert _steps_to_90pct(fractions, start=0.0, best=1.0) == 3


def test_best_equals_start_returns_zero():
    fractions = [0.4, 0.4, 0.4]
    assert _steps_to_90pct(fractions, start=0.4, best=0.4) == 0


def test_decreasing_sequence_reaches_threshold():
    # start=1.0, best=0.0 -> 90% threshold is 0.1; reached at index 2
    fractions = [1.0, 0.5, 0.05, 0.0]
    assert _steps_to_90pct(fractions, start=1.0, best=0.0) == 2


def test_never_reaches_threshold_falls_back_to_last_index():
    # start=0.0, best=1.0 but the sequence stalls at 0.5, never reaching 0.9
    fractions = [0.0, 0.2, 0.4, 0.5, 0.5]
    assert _steps_to_90pct(fractions, start=0.0, best=1.0) == len(fractions) - 1
