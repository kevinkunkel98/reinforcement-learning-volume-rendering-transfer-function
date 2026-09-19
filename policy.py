"""The trained one-shot policy: one checkpoint path, one lazy loader.

This module exists because the path was written down twice. `server.py` was
moved to the v3 checkpoint when the observation fixes landed -- v2 was trained
with the reachable ceiling probed at the retired band layout's fat peak, and
with a colour action that collapsed r, g and b to one scalar -- but `collect.py`
kept its own copy of the constant and went on sampling the v2 policy. Every
preference pair judged before 2026-09-17 therefore compared candidates from the
superseded checkpoint, which is a quiet corruption of the data the reward model
is meant to learn from.

Two callers, one constant. Anything that needs the policy imports it here.
"""
import os

# v3: trained on the corrected observation. Held-out attainment +0.275 against
# v2's +0.247, as the median of the three seeds' own medians -- the statistic
# the README and pipeline tables report.
#
# The paired v2/v3 test uses a different statistic and the two must not be
# crossed: on the median of the seed-averaged per-episode attainment it is v3
# +0.286 against v2 +0.201, Wilcoxon p = 0.0064. Same checkpoints, same 200
# episodes; +0.028 and +0.086 are the same result aggregated two ways.
POLICY_PATH = "out/rl_v2/oneshot_v3_seed0/best.zip"

_state = {"loaded": False, "policy": None}


def load_policy(path: str = None):
    """The checkpoint, loaded on first use and cached after.

    Lazy, and `None` when no checkpoint exists: the viewer and the collection
    page both have to work before a training run has finished, falling back to
    exact command application and to the non-policy candidate sources
    respectively.
    """
    path = path or POLICY_PATH
    if not _state["loaded"]:
        _state["loaded"] = True
        if os.path.exists(path):
            from stable_baselines3 import SAC
            _state["policy"] = SAC.load(path)
    return _state["policy"]


def reset_cache():
    """Forget the cached checkpoint. For tests that change `POLICY_PATH` or the
    working directory -- the path is relative, so cwd decides what it finds."""
    _state["loaded"] = False
    _state["policy"] = None
