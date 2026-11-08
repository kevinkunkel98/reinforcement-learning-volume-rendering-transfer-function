"""The trained policy the viewer and /collect use: one path, one lazy loader.

The default is the policy behind the thesis results (task v3, seed 1, best on
val with planning), a copy of out/rl_mb_v3c/clb_r3_seed1 under models/ so a
fresh clone has it. POLICY_PATH points elsewhere.
"""
import functools
import os

from rl.episodes import policy_metadata

POLICY_PATH = os.environ.get("POLICY_PATH", "models/v3c_seed1/checkpoint.pt")


@functools.lru_cache(maxsize=4)
def _load(path: str):
    from rl.model_based import ModelBasedPolicy
    policy_metadata(path)
    return ModelBasedPolicy.load(path)


def load_policy(path: str = None):
    """The checkpoint, loaded on first use and cached; None when there is
    none, so the viewer falls back to applying commands exactly."""
    path = os.path.realpath(os.path.abspath(path or POLICY_PATH))
    return _load(path) if os.path.exists(path) else None
