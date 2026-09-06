"""Pure hill-climbing search-step math, shared by mvp.py and server.py.

Opacity-only: propose_step always mutates the height parameter; callers
must gate entry to this module on attribute == "opacity". resize_step
itself is domain-agnostic and takes an optional max_step to cap growth
at the natural scale of whatever quantity is being stepped.
"""
import numpy as np

from transfer import PARAMS_PER_PEAK


def propose_step(params: np.ndarray, peak_idx: int, sign: float, step: float) -> np.ndarray:
    proposed = params.copy()
    base = peak_idx * PARAMS_PER_PEAK
    internal = (proposed[base + 2] + 1.0) / 2.0
    internal = float(np.clip(internal + sign * step, 0.0, 1.0))
    proposed[base + 2] = internal * 2.0 - 1.0
    return proposed


def resize_step(step: float, accepted: bool, max_step: float = 1.0) -> float:
    return min(step * 1.2, max_step) if accepted else step * 0.5
