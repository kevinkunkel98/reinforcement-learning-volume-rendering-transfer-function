import csv
import os

import numpy as np
import pytest

import goals
import transfer
from rl import oneshot_train
from rl.oneshot_env import ACTION_SIZE, POLICY_VERSION, OneShotEnv


class _StubModel:
    """Same shape as the stub in test_oneshot_env.py: deterministic and
    cheap, but responsive to every controllable value."""

    MEASURED = {"skeleton": "skeleton", "lungs": "lungs", "soft": "organs", "vessels": "vessels"}

    def __init__(self):
        self.histogram = np.full(16, 1.0 / 16.0, dtype=np.float32)

    def features(self, params) -> dict:
        vis, bright = {}, {}
        for goal_class, measured in self.MEASURED.items():
            idx = goals.PEAK_INDEX[goal_class]
            peak = transfer.peak_internal(params, idx)
            vis[measured] = max(float(peak["height"]), 0.0)
            bright[measured] = float(sum(peak["rgb"]) / 3.0)
        vis["muscle"] = 0.0
        bright["muscle"] = 0.0
        coverage = min(1.0, sum(vis.values()))
        return {"vis": vis, "bright": bright, "coverage": coverage}

    def solo_max(self, name: str) -> float:
        return 1.0


CLASSES_PRESENT = {"stub_a": ["skeleton", "lungs", "organs", "muscle", "vessels"],
                    "stub_b": ["skeleton", "lungs", "organs", "muscle", "vessels"]}


def _patch_totalseg(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: CLASSES_PRESENT[name])
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)


def _stub_env(monkeypatch, volume_ids=("stub_a", "stub_b")):
    _patch_totalseg(monkeypatch)
    return OneShotEnv(list(volume_ids), model_for_volume=lambda name: _StubModel())


class _FixedActionModel:
    """A stand-in for an SB3 model: `predict` always returns the same
    deterministic action, so `evaluate_policy_on` can be tested without
    training anything."""

    def __init__(self, action=None):
        self.action = action if action is not None else np.zeros(ACTION_SIZE, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        assert deterministic is True
        return self.action, None


# --- build_env / build_eval_env -------------------------------------------------

def test_build_env_uses_the_training_split(monkeypatch):
    fake_train = ["train_a", "train_b", "train_c"]
    monkeypatch.setattr(oneshot_train.datasets, "volumes_for_split", lambda split: {
        "train": fake_train, "val": ["val_a"]}[split])

    env = oneshot_train.build_env()

    assert env.volume_ids == fake_train


def test_build_eval_env_uses_the_validation_split(monkeypatch):
    fake_val = ["val_a", "val_b"]
    monkeypatch.setattr(oneshot_train.datasets, "volumes_for_split", lambda split: {
        "train": ["train_a"], "val": fake_val}[split])

    env = oneshot_train.build_eval_env()

    assert env.volume_ids == fake_val


def test_build_env_accepts_an_injected_volume_list(monkeypatch):
    env = oneshot_train.build_env(volume_ids=["only_this_one"])
    assert env.volume_ids == ["only_this_one"]


# --- evaluate_policy_on -----------------------------------------------------------

def test_evaluate_policy_on_returns_per_episode_attainment(monkeypatch):
    env = _stub_env(monkeypatch, volume_ids=("stub_a",))
    episodes = oneshot_train.validation_episodes(env, count=5, seed_base=0)
    model = _FixedActionModel()

    results = oneshot_train.evaluate_policy_on(episodes, model)

    assert len(results) == 5
    for row in results:
        assert "attainment" in row and "kind" in row
        assert isinstance(row["attainment"], float)


def test_evaluate_policy_on_matches_a_manual_rollout(monkeypatch):
    env = _stub_env(monkeypatch, volume_ids=("stub_a",))
    action = np.full(ACTION_SIZE, 0.3, dtype=np.float32)
    model = _FixedActionModel(action=action)

    [result] = oneshot_train.evaluate_policy_on([(env, 123)], model)

    # Reproduce the same episode by hand.
    env.reset(seed=123)
    obs, reward, terminated, truncated, info = env.step(action)
    assert result["attainment"] == pytest.approx(info["attainment"])
    assert result["kind"] == info["kind"]


def test_summarize_eval_aggregates_overall_and_per_kind():
    results = [
        {"attainment": 0.5, "kind": "relative"},
        {"attainment": 1.0, "kind": "relative"},
        {"attainment": 0.0, "kind": "compound"},
    ]
    summary = oneshot_train.summarize_eval(results)
    assert summary["mean_attainment"] == pytest.approx(0.5)
    assert summary["median_attainment"] == pytest.approx(0.5)
    assert summary["mean_clipped_attainment"] == pytest.approx(0.5)
    assert summary["share_positive"] == pytest.approx(2.0 / 3.0)
    assert summary["by_kind"]["relative"] == pytest.approx(0.75)
    assert summary["by_kind"]["compound"] == pytest.approx(0.0)
    assert summary["by_kind"]["show_only"] is None


# --- run directory guard -----------------------------------------------------------

def test_ensure_run_dir_raises_on_an_existing_directory(tmp_path):
    existing = tmp_path / "run"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        oneshot_train.ensure_run_dir(str(existing))


def test_ensure_run_dir_creates_a_new_directory(tmp_path):
    fresh = tmp_path / "run"
    oneshot_train.ensure_run_dir(str(fresh))
    assert fresh.is_dir()


# --- end-to-end smoke run with a stub env -----------------------------------------

def test_smoke_run_writes_eval_progress_csv_with_expected_columns(monkeypatch, tmp_path):
    _patch_totalseg(monkeypatch)
    train_env = OneShotEnv(["stub_a", "stub_b"], model_for_volume=lambda name: _StubModel())
    eval_env = OneShotEnv(["stub_a", "stub_b"], model_for_volume=lambda name: _StubModel())
    out = str(tmp_path / "smoke_run")

    result = oneshot_train.run_training(
        out=out, timesteps=200, seed=0, eval_interval=100,
        train_env=train_env, eval_env=eval_env, eval_episode_count=5)

    eval_path = result["eval_progress_path"]
    assert os.path.exists(eval_path)
    with open(eval_path, newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == oneshot_train.CSV_FIELDS
        rows = list(reader)
    assert len(rows) == 2  # 200 timesteps / 100 eval-interval
    assert os.path.exists(result["best_path"])


def test_best_checkpoint_is_selected_by_median_not_mean(monkeypatch, tmp_path):
    # Round 1 has a high mean (0.575) but a low median (-0.9), dragged up by
    # one outlier; round 2 has a lower mean (0.3) but a higher median (0.3).
    # Selecting by median must treat round 2 as the new best, even though
    # its mean is worse -- that is the whole point of the fix (median is
    # robust to the single-episode outliers attainment produces).
    _patch_totalseg(monkeypatch)
    train_env = OneShotEnv(["stub_a"], model_for_volume=lambda name: _StubModel())
    eval_env = OneShotEnv(["stub_a"], model_for_volume=lambda name: _StubModel())
    out = str(tmp_path / "median_run")

    round1 = [{"attainment": a, "kind": "relative"} for a in (-0.9, -0.9, -0.9, 5.0)]
    round2 = [{"attainment": a, "kind": "relative"} for a in (0.3, 0.3, 0.3, 0.3)]
    rounds = iter([round1, round2])
    monkeypatch.setattr(oneshot_train, "evaluate_policy_on", lambda episodes, model: next(rounds))

    saved_paths = []
    monkeypatch.setattr(oneshot_train.SAC, "save", lambda self, path: saved_paths.append(path))

    oneshot_train.run_training(
        out=out, timesteps=200, seed=0, eval_interval=100,
        train_env=train_env, eval_env=eval_env, eval_episode_count=4)

    best_path = os.path.join(out, oneshot_train.BEST_NAME)
    # best.zip is saved once for round 1 (nothing to compare against yet)
    # and again for round 2, because round 2's median beats round 1's.
    assert saved_paths.count(best_path) == 2


# --- CLI defaults -------------------------------------------------------------

def test_parse_args_defaults_match_the_plan():
    args = oneshot_train.parse_args([])
    assert args.timesteps == 150_000
    assert args.seed == 0
    assert args.eval_interval == 10_000
    assert args.out is None


def test_default_out_template_includes_the_seed():
    assert oneshot_train.DEFAULT_OUT_TEMPLATE.format(seed=3) == "out/rl_v2/oneshot_seed3"


def test_learning_starts_matches_the_plan():
    assert oneshot_train.LEARNING_STARTS == 500


def test_parse_args_accepts_v7_short_experiment_options():
    args = oneshot_train.parse_args(["--policy-version", "oneshot-v7",
                                     "--action-mode", "residual", "--hindsight-ratio", "0.4", "--balance-classes",
                                     "--reward-mode", "target"])
    assert args.policy_version == "oneshot-v7"
    assert args.action_mode == "residual"
    assert args.hindsight_ratio == pytest.approx(0.4)
    assert args.balance_classes is True
    assert args.reward_mode == "target"


def test_parse_args_rejects_v7_without_explicit_action_mode():
    with pytest.raises(SystemExit):
        oneshot_train.parse_args(["--policy-version", "oneshot-v7"])


def test_short_experiment_preset_is_bounded():
    assert oneshot_train.SHORT_EXPERIMENT["timesteps"] <= 1_000
    assert oneshot_train.SHORT_EXPERIMENT["eval_episode_count"] <= 10


def test_short_experiment_cli_uses_bounded_validation_count(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(oneshot_train, "run_training",
                        lambda **kwargs: captured.update(kwargs) or {
                            "eval_progress_path": "eval.csv", "best_path": "best.zip"})
    oneshot_train.main(["--short", "--out", str(tmp_path / "run")])
    assert captured["eval_episode_count"] == oneshot_train.SHORT_EXPERIMENT["eval_episode_count"]


def test_run_writes_one_shot_contract_metadata(monkeypatch, tmp_path):
    _patch_totalseg(monkeypatch)
    out = str(tmp_path / "metadata_run")
    result = oneshot_train.run_training(
        out=out, timesteps=10, seed=0, eval_interval=10,
        train_env=OneShotEnv(["stub_a"], model_for_volume=lambda name: _StubModel()),
        eval_env=OneShotEnv(["stub_a"], model_for_volume=lambda name: _StubModel()),
        eval_episode_count=1)

    assert result["metadata"]["observation_size"] == 97
    assert result["metadata"]["action_size"] == 24
    assert result["metadata"]["anatomy_layout"] == "anatomy-v2"
    assert os.path.exists(os.path.join(out, "metadata.json"))


def test_v6_contract_defaults_remain_absolute_attainment():
    assert POLICY_VERSION == "oneshot-v6"
    env = OneShotEnv(["stub_a"], model_for_volume=lambda name: _StubModel())
    assert env.policy_version == "oneshot-v6"
    assert env.action_mode == "absolute"
    assert env.reward_mode == "attainment"


def test_v6_rejects_residual_action_mode_at_init():
    with pytest.raises(ValueError, match="v6"):
        OneShotEnv(["stub_a"], model_for_volume=lambda name: _StubModel(), action_mode="residual")


def test_load_policy_metadata_falls_back_to_v6(tmp_path):
    from rl.oneshot_env import load_policy_metadata
    metadata = load_policy_metadata(str(tmp_path / "missing.zip"))
    assert metadata == {"policy_version": "oneshot-v6", "action_mode": "absolute",
                        "reward_mode": "attainment"}


def test_load_policy_metadata_reads_sidecar(tmp_path):
    from rl.oneshot_env import load_policy_metadata
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(
        '{"policy_version": "oneshot-v7", "action_mode": "residual", "reward_mode": "target"}')
    assert load_policy_metadata(str(run / "best.zip"))["action_mode"] == "residual"


@pytest.mark.parametrize("metadata", [
    {"policy_version": "oneshot-v7", "action_mode": "absolute", "reward_mode": "target"},
    {"policy_version": "oneshot-v7", "action_mode": "residual", "reward_mode": "attainment"},
])
def test_v7_metadata_requires_residual_target_contract(metadata):
    from rl.oneshot_env import resolve_policy_metadata
    with pytest.raises(ValueError):
        resolve_policy_metadata(metadata)


def test_v7_environment_requires_target_reward_at_init():
    with pytest.raises(ValueError, match="target reward"):
        OneShotEnv(["stub_a"], model_for_volume=lambda name: _StubModel(),
                   policy_version="oneshot-v7", action_mode="residual",
                   reward_mode="attainment")


def test_v7_residual_action_is_added_to_start(monkeypatch):
    _patch_totalseg(monkeypatch)
    env = OneShotEnv(["stub_a"], model_for_volume=lambda name: _StubModel(),
                     policy_version="oneshot-v7", action_mode="residual", reward_mode="target")
    env.reset(seed=2)
    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    action[0] = 0.1
    expected = np.clip(env._controllable_values(env._start_params) + action, -1.0, 1.0)
    env.step(action)
    assert env._controllable_values(env._params) == pytest.approx(expected)


def test_v7_hindsight_ratio_one_exposes_oracle_action(monkeypatch):
    _patch_totalseg(monkeypatch)
    env = OneShotEnv(["stub_a"], model_for_volume=lambda name: _StubModel(),
                     policy_version="oneshot-v7", action_mode="residual", reward_mode="target",
                     hindsight_ratio=1.0)
    _, info = env.reset(seed=3)
    assert info["goal_source"] == "hindsight"
    assert env.hindsight_action() is not None


def test_v7_target_reward_reports_progress_and_drift(monkeypatch):
    _patch_totalseg(monkeypatch)
    env = OneShotEnv(["stub_a"], model_for_volume=lambda name: _StubModel(),
                     policy_version="oneshot-v7", action_mode="residual", reward_mode="target")
    env.reset(seed=4)
    _, reward, _, _, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))
    assert isinstance(reward, float)
    assert "target_progress" in info
    assert "drift" in info


def test_v7_target_reward_drift_includes_brightness(monkeypatch):
    _patch_totalseg(monkeypatch)
    env = OneShotEnv(["stub_a"], model_for_volume=lambda name: _StubModel(),
                     policy_version="oneshot-v7", action_mode="residual", reward_mode="target")
    env.reset(seed=4)
    env._instruction = {"kind": "brightness", "text": "test",
                        "targets": {"skeleton": {"bright": 0.2}},
                        "goal": goals.goal_vector({"skeleton": {"bright": 0.2}})}
    monkeypatch.setattr(goals, "progress", lambda start, current:
                        ({**{c: 0.0 for c in goals.GOAL_CLASSES}, "other": 0.0},
                         {"skeleton": 0.0, "lungs": 0.2, "soft": 0.0, "vessels": 0.0,
                          "heart": 0.0, "liver": 0.0, "kidneys": 0.0, "spleen": 0.0}))
    env.step(np.zeros(ACTION_SIZE, dtype=np.float32))
    assert env._last_drift > 0.0
