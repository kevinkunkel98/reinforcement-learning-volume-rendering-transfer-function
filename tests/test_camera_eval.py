"""Tests for the pure-logic pieces of rl/camera_eval.py."""
import numpy as np

from rl.camera_env import _view_direction
from rl.camera_eval import _run_hill_climb


def test_hill_climb_alignment_never_decreases_step_to_step():
    rng = np.random.default_rng(0)
    target_direction = rng.normal(size=3)
    target_direction /= np.linalg.norm(target_direction)
    episode = {
        "azimuth": 10.0,
        "elevation": -20.0,
        "target_tissue": "bone",
        "target_direction": target_direction,
    }
    alignments = _run_hill_climb(episode)
    for before, after in zip(alignments, alignments[1:]):
        assert after >= before - 1e-9


def test_hill_climb_reaches_near_optimal_alignment_within_step_budget():
    # Regression guard for the `resize_step(step, accepted)` bug: without
    # `max_step=MAX_DELTA_DEGREES`, resize_step's default max_step=1.0 (tuned
    # for the opacity-transfer-function domain) silently caps the hill-climb
    # step size at 1 degree after the first accepted move, instead of letting
    # it keep exploring in ~30 degree increments. That doesn't break the
    # never-decreases invariant above, but it makes convergence far too slow
    # to cover a large azimuth/elevation gap within MAX_STEPS. Starting far
    # from the target (150 deg azimuth, 60 deg elevation away) only reaches
    # ~0.37 alignment when capped at 1 deg steps, vs ~1.0 with a 30 deg step
    # budget, so this cleanly distinguishes the two.
    target_direction = _view_direction(150.0, 60.0)
    episode = {
        "azimuth": 0.0,
        "elevation": 0.0,
        "target_tissue": "bone",
        "target_direction": target_direction,
    }
    alignments = _run_hill_climb(episode)
    assert alignments[-1] > 0.95
