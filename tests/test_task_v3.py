import numpy as np
import pytest

import goals
import transfer
from anatomy import CANONICAL_CLASSES
from rl.oneshot_env import OneShotEnv
from test_task_v2 import V9, _OccludingModel


def test_unlabelled_opacity_scales_only_the_unlabelled_row():
    params = goals.starting_params("v2")
    hu = np.linspace(*transfer.CENTER_RANGE, 64)
    full = transfer.label_scoped_tables(params, hu)
    transfer.set_unlabelled_opacity(0.2)
    faded = transfer.label_scoped_tables(params, hu)
    np.testing.assert_allclose(faded[0][0], 0.2 * full[0][0])
    np.testing.assert_allclose(faded[0][1], full[0][1])
    for a, b in zip(full[1:], faded[1:]):
        np.testing.assert_allclose(a[0], b[0])
    # The global HU lookup ignores labels, so the shell setting does not apply.
    np.testing.assert_allclose(transfer.label_scoped_tables(params, hu, mode="global")[0][0],
                               transfer._opacity_and_color_at(params, hu)[0])


def test_unlabelled_opacity_rejects_out_of_range():
    with pytest.raises(ValueError):
        transfer.set_unlabelled_opacity(1.5)


def test_task_v3_starts_like_v2_with_the_shell_faded():
    np.testing.assert_allclose(goals.starting_params("v3"), goals.starting_params("v2"))
    goals.use_task_version("v3")
    assert transfer.UNLABELLED_OPACITY == 0.2
    goals.use_task_version("v2")
    assert transfer.UNLABELLED_OPACITY == 1.0
    with pytest.raises(ValueError):
        goals.use_task_version("v9")


def test_v3_env_fades_the_shell_and_samples_visible_brightness(monkeypatch):
    seen = []

    def fake_sample(name, model, start, rng, layers, counts, balance, **kwargs):
        seen.append(kwargs)
        return {"kind": "relative", "text": "more bone", "targets": {"skeleton": {"vis": 0.3}},
                "goal": goals.goal_vector({"skeleton": {"vis": 0.3}})}

    monkeypatch.setattr(goals, "sample_instruction", fake_sample)
    env = OneShotEnv(["fake"], model_for_volume=lambda name: _OccludingModel(), **V9, task_version="v3")
    assert transfer.UNLABELLED_OPACITY == 0.2
    assert env.solvable_instructions
    env.reset(seed=0)
    assert seen[-1] == {"solvable_only": True, "visible_brightness": True}
    OneShotEnv(["fake"], model_for_volume=lambda name: _OccludingModel(), **V9, task_version="v2")
    assert transfer.UNLABELLED_OPACITY == 1.0


def test_visible_brightness_never_asks_an_invisible_class(monkeypatch):
    monkeypatch.setattr(goals, "_sample_kind", lambda rng: "brightness")
    monkeypatch.setattr(goals, "reachable_goal_classes", lambda name, model, layers=None: ["heart", "liver"])
    start = {"vis": {c: 0.0 for c in CANONICAL_CLASSES}, "bright": {}, "coverage": 0.5}
    start["vis"]["liver"] = 0.02
    rng = np.random.default_rng(0)
    for _ in range(20):
        inst = goals.sample_instruction("fake", _OccludingModel(), start, rng, visible_brightness=True)
        assert inst["kind"] == "brightness" and list(inst["targets"]) == ["liver"]
    start["vis"]["liver"] = 0.0
    inst = goals.sample_instruction("fake", _OccludingModel(), start, rng, visible_brightness=True)
    assert inst["kind"] == "absolute"
