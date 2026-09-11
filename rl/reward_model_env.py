"""TFEnv variant whose reward comes from a learned reward model over
rendered image features, instead of the automatic mass_fraction metric. See
rl/reward_model.py for the reward model itself, and
docs/superpowers/specs/2026-09-09-rlhf-reward-model-design.md for why."""
from camera import DEFAULT_CAMERA
from rl.env import TFEnv
from rl.reward_model import predict_reward

import render


class RewardModelTFEnv(TFEnv):
    """Reuses TFEnv's sampling/action/observation logic entirely unchanged
    (via super().reset()/super().step()) and only replaces the reward.
    One render per step, not two: each step's "after" features become next
    step's cached "before" features, seeded once at reset()."""

    def __init__(self, volume, spacing, reward_model, seed=None, camera=None):
        super().__init__(seed=seed)
        self.volume = volume
        self.spacing = spacing
        self.reward_model = reward_model
        self.camera = camera or dict(DEFAULT_CAMERA)
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
        reward = predict_reward(self.reward_model, self._prev_features, after_features,
                                 self.target_tissue, self.direction)
        self._prev_features = after_features
        info["automatic_mass_fraction_reward"] = automatic_reward
        return obs, reward, terminated, truncated, info
