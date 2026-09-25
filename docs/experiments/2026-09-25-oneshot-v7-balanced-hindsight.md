# One-Shot v7 Policy Experiment

## Purpose

Train policy against expanded eight-class anatomy after segmentation-aware
rendering. Existing `oneshot-v6` remains baseline; v7 checkpoint must not be
loaded as v6.

## Configuration

```text
policy_version: oneshot-v7
action_mode: residual
reward_mode: target
hindsight_ratio: 0.25
balance_classes: true
observation_size: 97
action_size: 24
goal_classes: skeleton, lungs, heart, vessels, liver, kidneys, spleen, soft
```

Residual action semantics:

```text
new_controllable = start_controllable + policy_delta
```

Values are clipped to `[-1, 1]`. Current RGB offsets remain preserved.

## Reward

Primary signal is target distance reduction:

```text
target_progress = start_distance - final_distance
reward = clip(target_progress, -1, 1)
          - 0.1 * unmentioned_drift
          - useless_render_penalty
```

`goals.distance()` already includes keep penalties. The auxiliary drift weight
is therefore `0.1`, avoiding double punishment of ordinary occlusion side
effects. Drift includes visibility and brightness, with brightness weighted by
`goals.KAPPA`.

## Sampling

- Instructions are restricted to supported and reachable classes.
- Reachable class mentions are balanced over training episodes.
- Hindsight episodes use a reachable transfer-function target and oracle action.
- `hindsight_ratio=0.25` keeps most training distribution close to spoken
  instructions while providing supervised reachable examples.

## Reproducible Commands

Short smoke:

```bash
.venv/bin/python -m rl.oneshot_train \
  --short \
  --policy-version oneshot-v7 \
  --action-mode residual \
  --reward-mode target \
  --hindsight-ratio 0.25 \
  --balance-classes \
  --seed 0 \
  --out out/rl_v4/smoke_v7_rewardfix_seed0
```

Controlled experiment:

```bash
.venv/bin/python -m rl.oneshot_train \
  --timesteps 10000 \
  --eval-interval 2000 \
  --policy-version oneshot-v7 \
  --action-mode residual \
  --reward-mode target \
  --hindsight-ratio 0.25 \
  --balance-classes \
  --seed 0 \
  --out out/rl_v4/exp_v7_balanced_hindsight_seed0
```

Evaluation:

```bash
.venv/bin/python -m rl.vis_eval \
  --policy out/rl_v4/exp_v7_balanced_hindsight_seed0/best.zip \
  --split test --episodes 200 --seed 0 --formulation one_shot \
  --out out/rl_v4/exp_v7_balanced_hindsight_seed0/eval_test.json
```

Per-class and baseline comparison:

```bash
.venv/bin/python -m tools.per_class_eval \
  out/rl_v4/exp_v7_balanced_hindsight_seed0/best.zip \
  --split test --episodes 200 --seed 0 \
  --baseline expanded_hill_climb \
  --out out/rl_v4/exp_v7_balanced_hindsight_seed0/per_class.json
```

## Evaluation Gates

Do not launch overnight training unless the controlled run is directionally
better than v6 and not obviously worse than cheap search:

- Median attainment improves from v6 reference `-0.028` on the same evaluation
  protocol.
- Share-positive exceeds v6 reference `35%`.
- New organ classes have nonzero reachable sample counts.
- Policy does not collapse to negative reward or useless renders.
- Compare against `do_nothing`, executor, hill climb 10, and hill climb 200.

The 2,000-step reward-fix smoke reached median `-0.032` and `40%` positive on
its small validation set. This is a smoke signal, not a held-out result. The
previous v7 configuration with drift weight `1.0` reached roughly `-0.54` to
`-0.67` median and `15%` positive, motivating the reward correction.

## Artifacts

Outputs belong under ignored `out/rl_v4/`. Each run must retain `metadata.json`,
`eval_progress.csv`, checkpoints, and evaluation JSON. Metadata must include
policy version, action mode, reward mode, seed, dimensions, and anatomy layout.
