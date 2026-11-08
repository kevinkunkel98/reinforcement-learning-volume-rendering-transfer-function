"""Episodes, the visibility estimate, the search, the model-based learner and the evaluation."""
import numpy as np
import pytest
import torch

import goals
import transfer
import visibility
from rl import closed_loop, model_based
from rl.episodes import ACTION_SIZE, OBSERVATION_SIZE, EpisodeSampler, fixed_episodes, policy_metadata
from rl.evaluate import evaluate_episode, wilcoxon
from rl.search import hill_climb, rule_executor

SCANS = ["ts_s9001", "ts_s9002"]


def test_visibility_follows_the_transfer_function(model):
    params = goals.starting_params()
    lungs = transfer.ANATOMICAL_PEAK_INDEX["lungs"] * transfer.PARAMS_PER_PEAK
    before = model.features(params)["vis"]["lungs"]
    params[lungs + 2] = 1.0
    assert model.features(params)["vis"]["lungs"] > before
    hidden = params.copy()
    hidden[lungs + 2] = -1.0
    assert model.features(hidden)["vis"]["lungs"] == 0.0
    assert model.vis_only(params)["lungs"] == pytest.approx(model.features(params)["vis"]["lungs"])


def test_visibility_cache_round_trips(model):
    import scans
    path = model.save_cache(scans.version("ts_s9001"), cache_dir="out/cache/roundtrip",
                            label_identity=model.label_identity)
    loaded = visibility.VisibilityModel.load_cache("ts_s9001", scans.version("ts_s9001"), "out/cache/roundtrip",
                                                   label_identity=model.label_identity)
    assert path and loaded.features(goals.starting_params()) == model.features(goals.starting_params())


def test_episodes_are_seeded():
    a, b = EpisodeSampler(SCANS).reset(7), EpisodeSampler(SCANS).reset(7)
    assert a.volume == b.volume and np.array_equal(a.start_params, b.start_params)
    assert np.array_equal(a.instruction["goal"], b.instruction["goal"])
    assert a.observation().shape == (OBSERVATION_SIZE,)
    assert [e.volume for e in fixed_episodes("test", 3)] == ["ts_s9001"] * 3


def test_search_improves_on_the_start_and_the_rule_executor_answers():
    episode = EpisodeSampler(SCANS).reset(1)
    found = hill_climb(episode.model, episode.start_params, episode.instruction, evaluations=60)
    assert episode.attainment(found) >= 0.0
    assert rule_executor(episode.model, episode.start_params, episode.instruction).shape == (transfer.TOTAL_PARAMS,)


def test_shipped_checkpoint_matches_the_task(policy):
    metadata = policy_metadata("models/v3c_seed1/checkpoint.pt")
    assert metadata["task_version"] == "v3" and policy.dynamics is not None
    action, _ = policy.predict(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
    assert action.shape == (ACTION_SIZE,) and np.all(np.abs(action) <= 1.0)


def test_planning_lowers_the_models_loss(policy):
    episode = EpisodeSampler(SCANS).reset(2)
    action, _ = policy.predict(episode.observation())
    from rl.evaluate import planned
    refined = planned(policy, episode, action)
    context = model_based._tensor(np.concatenate([
        episode.model.histogram, model_based.solo_max_log(episode.model),
        model_based.controllable_values(episode.start_params), model_based.state_vector(episode.start_agg)]))[None]
    goal = model_based._tensor(episode.instruction["goal"])[None]
    weights = model_based._tensor(model_based.weight_row(episode.instruction.get("weights")))[None]
    start = model_based.goal_distance(goal, weights, torch.zeros(1, model_based.STATE_SIZE))
    with torch.no_grad():
        losses = [float(model_based.plan_loss(policy.dynamics, context, goal, weights,
                                              model_based._tensor(a)[None], start)) for a in (action, refined)]
    assert losses[1] <= losses[0]


def test_evaluation_row_has_every_method(policy):
    episode = EpisodeSampler(SCANS).reset(4)
    row = evaluate_episode(policy, episode, 0, budgets=(10,), cold=True, keep=True, colour=True)
    for key in ("policy", "plan", "warm10", "plan_warm10", "cold200", "plan_keep", "plan_keep_colour"):
        assert np.isfinite(row[key]), key
    assert row["warm10"] >= row["policy"]


def test_wilcoxon_matches_a_hand_computed_example():
    result = wilcoxon([3, -1, 2, -4, 5], [0, 0, 0, 0, 0])
    assert result["statistic"] == 5.0 and result["p_value"] == pytest.approx(0.58964, abs=1e-4)


def test_model_based_learner_end_to_end(tmp_path):
    data = model_based.collect_starts(EpisodeSampler(SCANS, balance_classes=True), range(6), actions_per_start=4)
    assert data["act"].shape == (6, 4, ACTION_SIZE)
    train, val = model_based.split_starts(6)
    ensemble, _report = model_based.train_dynamics(data, train, val, epochs=1, log=lambda *_: None)
    policy, _ = model_based.train_policy(ensemble, data, train, steps=5, log=lambda *_: None)
    policy, _ = closed_loop.finetune(policy, ensemble, data, train, steps=3, log=lambda *_: None)
    path = policy.save(str(tmp_path), ensemble)
    reloaded = model_based.ModelBasedPolicy.load(path)
    assert policy_metadata(path)["task_version"] == "v3" and reloaded.dynamics is not None
    episode = EpisodeSampler(SCANS).reset(0)
    assert len(closed_loop.rollout(reloaded, episode, horizon=2)) == 2
