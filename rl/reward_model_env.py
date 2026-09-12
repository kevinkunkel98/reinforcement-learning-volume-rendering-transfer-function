"""TFEnv variant whose reward comes from a learned reward model over
rendered image features, instead of the automatic mass_fraction metric. See
rl/reward_model.py for the reward model itself, and
docs/superpowers/specs/2026-09-09-rlhf-reward-model-design.md for why."""
from camera import DEFAULT_CAMERA
from rl.env import TFEnv
from rl.reward_model import ensemble_reward, predict_reward

import render


DEFAULT_COVERAGE_THRESHOLD = 0.01
DEFAULT_MEAN_OPACITY_THRESHOLD = 250.0
DEFAULT_HARD_PENALTY = -1.0


class RewardModelTFEnv(TFEnv):
    """Reuses TFEnv's sampling/action/observation logic entirely unchanged
    (via super().reset()/super().step()) and only replaces the reward.
    One render per step, not two: each step's "after" features become next
    step's cached "before" features, seeded once at reset()."""

    def __init__(self, volume, spacing, reward_model, seed=None, camera=None,
                 alpha=0.7, coverage_threshold=DEFAULT_COVERAGE_THRESHOLD,
                 mean_opacity_threshold=DEFAULT_MEAN_OPACITY_THRESHOLD,
                 hard_penalty=DEFAULT_HARD_PENALTY):
        super().__init__(seed=seed)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.volume = volume
        self.spacing = spacing
        self.reward_model = reward_model
        self.camera = camera or dict(DEFAULT_CAMERA)
        self.alpha = float(alpha)
        self.coverage_threshold = float(coverage_threshold)
        self.mean_opacity_threshold = float(mean_opacity_threshold)
        self.hard_penalty = float(hard_penalty)
        self._prev_features = None

    def _render_features(self, params):
        win = render.render(self.volume, params, self.spacing, self.camera)
        return render.features(render.grab(win))

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_features = self._render_features(self.params)
        return obs, info

    def step(self, action):
        obs, automatic_reward, terminated, truncated, info = super().step(action)
        after_features = self._render_features(self.params)
        models = (self.reward_model,) if not isinstance(self.reward_model, (list, tuple)) else self.reward_model
        model_rewards = [predict_reward(model, self._prev_features, after_features,
                                         self.target_tissue, self.direction)
                         for model in models]
        model_reward = ensemble_reward(model_rewards)
        reward = self.alpha * model_reward + (1.0 - self.alpha) * automatic_reward
        if (after_features["coverage"] < self.coverage_threshold or
                after_features["mean"] > self.mean_opacity_threshold):
            reward += self.hard_penalty
        self._prev_features = after_features
        info["automatic_mass_fraction_reward"] = automatic_reward
        return obs, reward, terminated, truncated, info
