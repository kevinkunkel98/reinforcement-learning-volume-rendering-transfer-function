import plots.qualitative as qualitative


def test_qualitative_uses_validating_policy_loader(monkeypatch, tmp_path):
    loaded = object()
    seen = []
    monkeypatch.setattr(qualitative.policy_module, "load_policy",
                        lambda path: seen.append(path) or loaded)
    result = qualitative._load_policy("out/rl_v3/oneshot_seed0/best.zip")

    assert seen == ["out/rl_v3/oneshot_seed0/best.zip"]
    assert result is loaded
