"""Tests for rl/serve.py -- the live-serving counterpart to rl/eval.py's
offline comparison. Uses a fake model (not the real trained SAC agent) so
these run fast and don't depend on out/rl_models/sac_tf_agent.zip existing."""
import numpy as np
import pytest

import rl.serve as serve
from commands import _find_or_create_peak
from transfer import default_params


def test_run_policy_raises_when_model_missing(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.zip")
    params = default_params()
    with pytest.raises(ValueError, match="no trained model"):
        serve.run_policy(params, "bone", "increase", peak_idx=0, steps=3, model_path=missing_path)


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return np.array([0.5], dtype=np.float32), None


def test_run_policy_moves_target_peak_and_does_not_mutate_input(monkeypatch):
    monkeypatch.setattr(serve, "_load_model", lambda model_path=serve.MODEL_PATH: _FakeModel())
    params, peak_idx = _find_or_create_peak(default_params(), "bone")
    original = params.copy()

    result = serve.run_policy(params, "bone", "increase", peak_idx, steps=3, model_path="unused")

    assert result.shape == params.shape
    assert not np.array_equal(result, original)  # the fake model's action moved the height
    assert np.array_equal(params, original)  # input array itself untouched


def test_run_policy_caches_loaded_model(monkeypatch, tmp_path):
    calls = []

    class _CountingFakeModel(_FakeModel):
        pass

    def fake_sac_load(path):
        calls.append(path)
        return _CountingFakeModel()

    monkeypatch.setattr(serve.SAC, "load", staticmethod(fake_sac_load))
    model_zip = tmp_path / "fake.zip"
    model_zip.write_text("not a real model, never opened by the fake loader")
    serve._model_cache.clear()

    params, peak_idx = _find_or_create_peak(default_params(), "bone")
    serve.run_policy(params, "bone", "increase", peak_idx, steps=1, model_path=str(model_zip))
    serve.run_policy(params, "bone", "increase", peak_idx, steps=1, model_path=str(model_zip))

    assert len(calls) == 1  # second call reused the cached model, did not call SAC.load again
