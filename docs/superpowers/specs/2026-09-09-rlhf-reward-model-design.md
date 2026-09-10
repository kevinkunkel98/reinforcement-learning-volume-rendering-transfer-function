# RLHF Reward Model for the Transfer-Function RL Policy — Design

## Purpose

`rl/online_train.py` (previous phase) trains against `mass_fraction` — an exact,
free-to-compute proxy for "did the target tissue's opacity move the way the
words asked," but it never looks at a rendered image, so it can't capture
whether the *result* actually looks good. This phase closes that gap: a
small reward model is trained from a human's better/worse judgments of
rendered before/after pairs, and that learned reward — not `mass_fraction`
— then drives RL training. This is the RLHF pattern (reward model from human
preference, RL trained against the reward model) applied to transfer-function
opacity control.

Scope: an **MVP single-rater pilot** demonstrating the mechanism works, not a
generalizable multi-user study. `out/rlhf_preferences.jsonl` records a
`rater_id` per label (constant for this phase, always the person running
`rl/collect_preferences.py`) purely so a later multi-rater extension (e.g.
routing the existing chat UI's Better/Worse flow at this same schema) doesn't
require a schema migration — no multi-user collection mechanism is built in
this phase.

## Grounding in existing code

- **Render latency, measured directly** (not assumed): `render.render()` +
  `render.grab()` + `render.features()` on the synthetic phantom (96³,
  `datasets.load_dataset("synthetic")`) took **109.0 ms/call**, averaged over
  20 calls, GPU raycast mapper. This is why the RL step budget in this phase
  is ~2,000, not ~50,000 like the previous phase — every step now needs a
  real render, there is no way around it once the reward depends on image
  features.
- `render.features(rgb) -> dict` already exists, returns
  `{"mean": ..., "std": ..., "coverage": ..., "entropy": ...}` — confirmed by
  direct call. Reused unmodified.
- `evaluate.human(before_png, after_png) -> int` already exists: prints both
  paths, prompts `"Besser oder schlechter? (b/s): "`, returns `1`/`-1`. Reused
  unmodified — this *is* the labeling prompt, no need to reimplement it.
- `evaluate.jsonl_append(path, entry)` already exists, reused unmodified for
  writing `out/rlhf_preferences.jsonl`.
- `rl.eval._build_episodes(n, seed)` already exists and, via
  `TFEnv(seed=seed).reset()`, produces exactly the `(params, target_tissue,
  direction, peak_idx)` tuples this phase needs as base cases — reused
  unmodified rather than re-implementing sampling.
- `search.propose_step(params, peak_idx, sign, step)` and
  `camera.DEFAULT_CAMERA` (`{"azimuth": 30.0, "elevation": 20.0, "zoom":
  1.0}`, copy before use per its own docstring) — reused unmodified.
- **`TFEnv` subclassing verified directly against the real class** (not
  assumed): after `env.reset()`, `env.params`/`env.target_tissue`/
  `env.direction` are plain accessible attributes; `env.step(action)` mutates
  `env.params` in place and returns `(obs, reward, terminated, truncated,
  {"mass_fraction_after": ...})`. This means a subclass can call
  `super().step(action)` to get the *real* sampling/mutation/observation
  machinery for free, discard just the returned `reward`, and substitute a
  render-based one — no need to duplicate `TFEnv`'s reset/step/`_obs` logic,
  and `rl/env.py` itself is never touched.

## New module: `rl/collect_preferences.py`

Samples `n_pairs` (default 50) base cases via `rl.eval._build_episodes`, for
each draws one random action in `[-1, 1]` (same distribution an untrained
policy explores), applies it via `search.propose_step` exactly as `TFEnv`
does, renders before/after on the synthetic phantom, asks
`evaluate.human()`, and logs one row per labeled pair:

```python
{
  "timestamp": "...", "rater_id": "default",
  "target_tissue": "bone", "direction": "increase", "delta": 0.083,
  "before_features": {"mean": ..., "std": ..., "coverage": ..., "entropy": ...},
  "after_features": {...},
  "label": 1
}
```

to `out/rlhf_preferences.jsonl`. Rendered PNGs are written to
`out/rlhf_labeling/<i>_before.png` / `<i>_after.png` (the paths
`evaluate.human()` prints and the person opens themselves — same convention
the existing hill-climb human-judging flow already uses).

CLI: `--n-pairs` (default 50), `--rater-id` (default `"default"`), `--seed`
(default 0), `--out` (default `out/rlhf_preferences.jsonl`).

## New module: `rl/reward_model.py`

A tiny PyTorch MLP. Input (10 dims) = Δ(mean, std, coverage, entropy) between
after/before features (4) + target-tissue one-hot in `rl.env.TISSUES` order
(5) + direction flag (1). Output: a logit, trained with
`BCEWithLogitsLoss` against the label (`1` → `1.0`, `-1` → `0.0`).

```python
FEATURE_KEYS = ("mean", "std", "coverage", "entropy")
INPUT_DIM = len(FEATURE_KEYS) + len(TISSUES) + 1  # 10

def featurize(before_features, after_features, target_tissue, direction) -> np.ndarray:
    delta = [after_features[k] - before_features[k] for k in FEATURE_KEYS]
    onehot = [1.0 if t == target_tissue else 0.0 for t in TISSUES]
    direction_flag = 1.0 if direction == "increase" else -1.0
    return np.array(delta + onehot + [direction_flag], dtype=np.float32)

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)
```

`train_reward_model(preferences_path, model_path, epochs=200, lr=1e-2, seed=0)`:
loads all labeled rows, builds `X`/`y` via `featurize`, does an 80/20
train/val split (shuffled, fixed seed), full-batch Adam for `epochs` steps
(the dataset is tiny — no minibatching needed), and **prints both train and
val accuracy** — reported honestly even if mediocre, per the explicit
methodological caveat this phase is designed around (50 labels is a pilot,
not enough to expect strong generalization, and the thesis should say so).
Saves `state_dict` to `model_path` (default
`out/rl_models/reward_model.pt`).

`load_reward_model(model_path)` / `predict_reward(model, before_features,
after_features, target_tissue, direction) -> float`: loads weights, runs
`featurize` → forward → `2·sigmoid(logit) − 1`, mapping to a signed
`[-1, 1]` reward (consistent in *shape*, not necessarily scale, with the
existing `mass_fraction`-delta reward — SAC does not require matching
scales).

## New module: `rl/reward_model_env.py`

```python
class RewardModelTFEnv(TFEnv):
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
        self._prev_features = self._render_features(self.params)  # seeds the cache
        return obs, info

    def step(self, action):
        obs, automatic_reward, terminated, truncated, info = super().step(action)
        after_features = self._render_features(self.params)
        reward = predict_reward(self.reward_model, self._prev_features, after_features,
                                 self.target_tissue, self.direction)
        self._prev_features = after_features  # this step's "after" is next step's "before" -- one render/step, not two
        info["automatic_mass_fraction_reward"] = automatic_reward
        return obs, reward, terminated, truncated, info
```

Everything except the reward computation — sampling, action scaling,
`propose_step` application, the 31-dim observation (which still includes the
*free* `mass_fraction` value, since observing it costs nothing; only the
*reward* needs a render) — comes from `TFEnv` unchanged. `rl/env.py` is not
modified. `info["automatic_mass_fraction_reward"]` is carried through for
free at every step specifically so training/eval can report *both* signals
side by side: is optimizing the learned reward also moving the true metric
sensibly?

## Training + evaluation

Mirrors `rl/online_train.py`'s chunked-`model.learn()`-plus-checkpoint-eval
shape, at a much smaller scale given the per-step render cost:

- Default `--timesteps 2000` (≈ 2000 × 109ms ≈ 3.6 min at the measured rate,
  single render per step via the caching above), `--eval-interval 500` (4
  checkpoints).
- Held-out eval set: 5 episodes (not 20 — each held-out eval step also
  renders, so this is deliberately small; noted explicitly as a cost-driven
  tradeoff).
- Each checkpoint logs, to a CSV mirroring `eval_progress.csv`'s shape: mean
  learned-reward-model score *and* mean `mass_fraction` delta (the automatic
  metric) across the held-out set — both signals, not just one.
- At the end of the run, render and save a handful (e.g. 3) of held-out
  before/after PNG pairs from the final policy, so the result can be
  eyeballed directly, not only read as numbers — this is the one piece of
  this phase's deliverable that actually answers "does it look good," same
  as the reason this whole phase exists.

Exact module name/CLI flags for this piece are finalized in the
implementation plan (mirrors `rl/online_train.py`'s structure closely enough
that no further design decisions are needed here).

## Data flow

```
python -m rl.collect_preferences --n-pairs 50
  → out/rlhf_preferences.jsonl (50 labeled rows)
  → out/rlhf_labeling/*.png (rendered pairs, for the rater to view)

python -m rl.reward_model --preferences out/rlhf_preferences.jsonl
  → out/rl_models/reward_model.pt
  → prints train/val accuracy

python -m rl.train_reward_model_policy --timesteps 2000   # name finalized in plan
  → out/rl_models/sac_tf_reward_model_agent.zip
  → out/rl_logs/reward_model_run_seed{seed}/eval_progress.csv (learned-reward + automatic-metric, per checkpoint)
  → out/rl_logs/reward_model_run_seed{seed}/*.png (final qualitative before/after pairs)
```

## Error handling

Consistent with the rest of `rl/` — no defensive handling beyond what
PyTorch/VTK/Gymnasium already raise. `train_reward_model` does not
special-case a too-small dataset; with `n_pairs=50` the 80/20 split always
leaves at least a few validation rows, which is sufficient for this phase's
purposes (an honest, if noisy, accuracy number).

## Testing

- `tests/test_rl_reward_model.py` — fast, no rendering: synthetic
  `preferences.jsonl` fixture (a handful of hand-written rows with real
  `render.features()`-shaped dicts), asserts `featurize()`'s output shape/
  values, and that `train_reward_model()` produces a model whose train
  accuracy is well above chance on a tiny separable synthetic set (a
  correctness check on the training loop, not a claim about the real data).
- `tests/test_rl_reward_model_env.py` — fast: monkeypatches
  `rl.reward_model_env.render.render`/`.grab`/`.features` to return
  synthetic data (no real VTK), and a stub reward model, to verify the
  before/after caching behavior specifically (the second step's "before"
  must equal the first step's "after" features, with `render` called exactly
  once per step, not twice) — plus one `@pytest.mark.slow` real end-to-end
  step (real render, real tiny reward model) as an integration smoke test.
- `tests/test_rl_collect_preferences.py` — fast: monkeypatches
  `evaluate.human` (fixed return, no real console prompt) and render calls,
  asserts the written JSONL has the right row count/schema.
- No test exercises the training/eval script's real 2000-step run beyond a
  tiny `@pytest.mark.slow` smoke test (same pattern as
  `tests/test_rl_online_train.py`), covered in the plan.

## New dependencies

None — `torch` is already a dependency (pulled in by `stable-baselines3`).

## Explicitly out of scope (this phase)

- **Multi-rater / multi-user preference collection** — `rater_id` is logged
  for forward-compatibility only; no mechanism to route the existing chat
  UI's Better/Worse flow at this schema is built now. Documented as the
  natural next step / thesis future-work section.
- **Live chat-UI wiring** of the reward-model-trained policy — same
  deferral as the previous phase.
- **Camera-viewpoint RLHF** — TF-opacity only.
- **Vision-model / LLM-as-judge reward** — the "automat" in this phase is
  the small learned reward model, not a vision-language model. A future
  extension could swap the reward source without changing the RL side of
  this architecture, but that swap is not built here.
- **Real CT datasets** — synthetic phantom only, for speed and to avoid the
  download dependency, consistent with how the offline/online opacity RL
  already trains.
