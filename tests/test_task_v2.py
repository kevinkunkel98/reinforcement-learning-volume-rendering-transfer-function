import math

import numpy as np
import pytest

import goals
import transfer
from anatomy import CANONICAL_CLASSES
from rl.oneshot_env import OneShotEnv, observation_metadata, resolve_policy_metadata

V9 = {"policy_version": "oneshot-v9", "action_mode": "residual", "reward_mode": "target"}


def _height(params, goal_class):
    return max(float(transfer.peak_internal(params, goals.PEAK_INDEX[goal_class])["height"]), 0.0)


class _OccludingModel:
    """Soft tissue sits in front of the heart; soft also shades the unlabelled
    shell ("other"). Every other class shows its own peak height."""

    def __init__(self):
        self.histogram = np.full(16, 1.0 / 16.0, dtype=np.float32)

    def features(self, params) -> dict:
        soft = _height(params, "soft")
        vis = {c: _height(params, c) * 0.1 for c in CANONICAL_CLASSES}
        vis["heart"] = _height(params, "heart") * 0.1 * (1.0 - soft) ** 4
        vis["other"] = 0.5 * soft
        bright = {c: 0.5 for c in CANONICAL_CLASSES}
        return {"vis": vis, "bright": bright, "coverage": 0.5}

    def solo_max(self, name: str) -> float:
        return 0.1


def _agg(**vis):
    full = {c: vis.get(c, 0.01) for c in list(goals.GOAL_CLASSES) + ["other"]}
    return {"vis": full, "bright": {c: 0.5 for c in goals.GOAL_CLASSES}, "coverage": 0.5}


def test_other_weight_zero_drops_only_the_other_keep_term():
    goal = goals.goal_vector({"heart": {"vis": 0.3}})
    start, current = _agg(other=0.5), _agg(other=0.05)
    charged = goals.distance(goal, start, current)
    exempt = goals.distance(goal, start, current, weights={"other": 0.0})
    c, _ = goals.progress(start, current)
    assert charged - exempt == pytest.approx(goals.LAMBDA_KEEP * (abs(c["other"]) - goals.KEEP_TOLERANCE))
    assert goals.distance(goal, start, current, weights={}) == pytest.approx(charged)


def test_occlusion_aware_exempts_occluders_from_keep_terms():
    instruction = {"kind": "relative", "targets": {"heart": {"vis": 0.3}},
                   "goal": goals.goal_vector({"heart": {"vis": 0.3}})}
    aware = goals.occlusion_aware(instruction, {"soft", "other"})
    start = _agg(heart=0.001, soft=0.2, other=0.5)
    faded = _agg(heart=0.002, soft=0.001, other=0.01)
    kw = {"weights": aware["weights"]}
    # fading soft and the shell costs keep-drift under v1, nothing under v2
    assert goals.distance(aware["goal"], start, faded, **kw) < goals.distance(instruction["goal"], start, faded)
    assert aware["weights"]["soft"] == 0.0 and aware["weights"]["other"] == 0.0
    assert aware["weights"]["heart"] == 1.0
    assert aware["occluders"] == ["other", "soft"]
    assert goals.occlusion_aware(instruction, set()) is instruction


def test_find_occluders_detects_the_class_and_shell_in_front_of_a_target():
    model = _OccludingModel()
    params = goals.starting_params()
    params[goals._height_index("soft")] = 0.4   # unit 0.7: soft hides the heart
    instruction = {"kind": "relative", "targets": {"heart": {"vis": 0.3}},
                   "goal": goals.goal_vector({"heart": {"vis": 0.3}})}
    found = goals.find_occluders(model, params, instruction)
    assert "soft" in found and "other" in found
    assert "lungs" not in found and "skeleton" not in found
    decrease = {"kind": "relative", "targets": {"heart": {"vis": -0.3}},
                "goal": goals.goal_vector({"heart": {"vis": -0.3}})}
    assert goals.find_occluders(model, params, decrease) == set()


def test_reachable_targets_clip_to_floor_and_ceiling():
    model = _OccludingModel()           # ceiling 0.1 for every class
    start = _agg(lungs=0.0, heart=0.05)
    goal = goals.goal_vector({"lungs": {"vis": -0.6}, "heart": {"vis": 1.0}})
    clipped = goals.reachable_targets({"goal": goal}, model, start)["goal"]
    n = len(goals.GOAL_CLASSES)
    lungs, heart = goals.GOAL_CLASSES.index("lungs"), goals.GOAL_CLASSES.index("heart")
    assert clipped[lungs] == pytest.approx(0.0)        # already at the floor: nothing left to lower
    assert clipped[heart] == pytest.approx(math.log10(0.1 + goals.EPSILON) - math.log10(0.05 + goals.EPSILON))
    assert clipped[n + lungs] == 1.0 and clipped[n + heart] == 1.0


def test_v2_start_shows_the_organs():
    v1, v2 = goals.starting_params(), goals.starting_params("v2")
    for goal_class, height in transfer.VISIBLE_ORGAN_HEIGHTS.items():
        assert v2[goals._height_index(goal_class)] == pytest.approx(height * 2 - 1)
    assert v2[goals._height_index("soft")] < v1[goals._height_index("soft")]
    assert v2[goals._height_index("heart")] > v1[goals._height_index("heart")]
    np.testing.assert_allclose(v1, transfer.anatomical_params())
    with pytest.raises(ValueError):
        goals.starting_params("v9")


def test_task_version_is_recorded_only_for_v2():
    assert "task_version" not in observation_metadata(**V9)
    assert observation_metadata(**V9, task_version="v2")["task_version"] == "v2"
    assert resolve_policy_metadata(V9) == V9
    assert resolve_policy_metadata({**V9, "task_version": "v2"}) == {**V9, "task_version": "v2"}
    with pytest.raises(ValueError):
        resolve_policy_metadata({**V9, "task_version": "v9"})


def test_v2_env_starts_visible_and_samples_well_posed_instructions(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: list(CANONICAL_CLASSES))
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    env = OneShotEnv(["fake"], model_for_volume=lambda name: _OccludingModel(), **V9, task_version="v2")
    assert env.solvable_instructions
    assert env.contract_metadata()["task_version"] == "v2"
    for seed in range(20):
        env.reset(seed=seed)
        n = len(goals.GOAL_CLASSES)
        for i, goal_class in enumerate(goals.GOAL_CLASSES):
            if env._instruction["goal"][n + i] and (env._instruction.get("weights") or {}).get(goal_class, 1.0):
                start = math.log10(env._start_agg["vis"][goal_class] + goals.EPSILON)
                assert start + env._instruction["goal"][i] >= math.log10(goals.EPSILON) - 1e-9
    with pytest.raises(ValueError):
        OneShotEnv(["fake"], model_for_volume=lambda name: _OccludingModel(), **V9, task_version="v9")


def test_for_task_version_is_identity_for_v1_and_applies_both_v2_transforms():
    model = _OccludingModel()
    params = goals.starting_params()
    params[goals._height_index("soft")] = 0.4
    instruction = {"kind": "relative", "targets": {"heart": {"vis": 3.0}},
                   "goal": goals.goal_vector({"heart": {"vis": 3.0}})}
    start = goals.aggregate(model.features(params))
    assert goals.for_task_version(instruction, model, params, start, "v1") is instruction
    v2 = goals.for_task_version(instruction, model, params, start, "v2")
    expected = goals.reachable_targets(
        goals.occlusion_aware(instruction, goals.find_occluders(model, params, instruction)), model, start)
    np.testing.assert_allclose(v2["goal"], expected["goal"])
    assert v2["weights"]["other"] == 0.0
    with pytest.raises(ValueError):
        goals.for_task_version(instruction, model, params, start, "v9")



def test_target_reward_survives_an_instruction_clipped_to_nothing(monkeypatch):
    # reachable_targets can clip every target to 0 (all at the floor or the
    # ceiling already); the episode then asks for nothing, like attainment's
    # floor case, instead of dividing by a zero start distance.
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: list(CANONICAL_CLASSES))
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    env = OneShotEnv(["fake"], model_for_volume=lambda name: _OccludingModel(), **V9, task_version="v2")
    env.reset(seed=0)
    nothing = goals.goal_vector({"heart": {"vis": 0.0}})
    env._instruction = {**env._instruction, "targets": {"heart": {"vis": 0.0}}, "goal": nothing,
                        "weights": None}
    env._start_distance = goals.distance(nothing, env._start_agg, env._start_agg)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert info["attainment"] == 0.0 and math.isfinite(reward)
