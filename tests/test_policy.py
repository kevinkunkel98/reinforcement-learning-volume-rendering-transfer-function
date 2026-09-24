import json

import pytest

import policy


class _Space:
    def __init__(self, shape):
        self.shape = shape


class _LoadedPolicy:
    observation_space = _Space((57,))
    action_space = _Space((12,))


def test_load_policy_rejects_old_checkpoint_dimensions(monkeypatch, tmp_path):
    checkpoint = tmp_path / "old.zip"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(policy, "POLICY_PATH", str(checkpoint))
    monkeypatch.setattr(policy, "_load_sac", lambda path: _LoadedPolicy())
    policy.reset_cache()

    with pytest.raises(ValueError, match="old one-shot checkpoint.*57.*12"):
        policy.load_policy()

    policy.reset_cache()


def test_policy_metadata_round_trips_as_json(tmp_path):
    path = tmp_path / "metadata.json"
    policy.write_metadata(str(path))

    metadata = json.loads(path.read_text())
    assert metadata["observation_size"] == 97
    assert metadata["action_size"] == 24
    assert metadata["anatomy_layout"] == "anatomy-v2"
