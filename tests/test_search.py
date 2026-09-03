from transfer import default_params, PARAMS_PER_PEAK
from commands import _find_or_create_peak
from search import propose_step, resize_step


def test_propose_step_moves_target_height_up():
    params = default_params()
    _, idx = _find_or_create_peak(params, "bone")
    base = idx * PARAMS_PER_PEAK
    before_h = params[base + 2]
    proposed = propose_step(params, idx, sign=1.0, step=0.35)
    assert proposed[base + 2] > before_h
    assert proposed.shape == params.shape


def test_propose_step_clamps_at_ceiling():
    params = default_params()
    _, idx = _find_or_create_peak(params, "bone")
    base = idx * PARAMS_PER_PEAK
    proposed = propose_step(params, idx, sign=1.0, step=5.0)
    assert proposed[base + 2] == 1.0


def test_resize_step_grows_on_accept_and_shrinks_on_reject():
    assert resize_step(0.35, accepted=True) == 0.35 * 1.2
    assert resize_step(0.35, accepted=False) == 0.35 * 0.5
    assert resize_step(0.9, accepted=True) == 1.0
