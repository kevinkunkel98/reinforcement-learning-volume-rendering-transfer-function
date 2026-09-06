"""Tests for the pure-logic pieces of rl/camera_eval.py."""
import numpy as np

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
