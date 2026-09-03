"""Pure hill-climbing search-step math, shared by mvp.py and server.py."""
import numpy as np

from transfer import PARAMS_PER_PEAK


def propose_step(params: np.ndarray, peak_idx: int, sign: float, step: float) -> np.ndarray:
    proposed = params.copy()
    base = peak_idx * PARAMS_PER_PEAK
    internal = (proposed[base + 2] + 1.0) / 2.0
    internal = float(np.clip(internal + sign * step, 0.0, 1.0))
    proposed[base + 2] = internal * 2.0 - 1.0
    return proposed


def resize_step(step: float, accepted: bool) -> float:
    return min(step * 1.2, 1.0) if accepted else step * 0.5
