# Preference-aligned transfer functions — design

Date: 2026-09-17
Status: agreed, not started. Improvement-month work; the MVP due 2026-09-22 is
unaffected.

## Why this exists

The thesis currently claims: a policy can be trained to adjust a transfer
function from an instruction, and it beats every non-search baseline on unseen
patients (p < 0.001, three seeds, 200 held-out instructions).

That claim has one soft spot. Everything the policy is trained and scored on is
`visibility.py`, which is cheap, deterministic and — as of the investigation on
2026-09-17 — very nearly differentiable: the resampled index cube is constant
with respect to the transfer function, and the parameters enter only through
`transfer_tables`, a Gaussian over 256 LUT values. Port that one function to
torch and attainment has an analytic gradient, at which point a few gradient
steps should beat both the policy and the 200-evaluation hill-climb, and the
obvious question becomes "why reinforcement learning at all?".

This design answers that question by moving the target: a policy aligned to
**human preference**, which is neither differentiable nor cheap to query. The
programmatic objective trains a competent policy; human judgments make it good.

## Goal

A two-stage policy:

1. **Stage 1 (no human):** hindsight-goal episodes, attainment reward. Produces
   `π₁`, a competent transfer-function policy.
2. **Stage 2 (human):** the same environment with the reward replaced by a
   Bradley-Terry model fitted to collected preference judgments, KL-anchored to
   `π₁`. Produces `π₂`.

Headline result: a blind A/B of `π₁` against `π₂` on unseen instructions,
reported as a win rate with a binomial confidence interval.

## Non-goals

- VR headset integration. Parked until after the MVP; see
  `docs/vr-integration.md` for the hardware constraints.
- Replacing `visibility.py`. It stays as the stage-1 reward and as the
  measurement apparatus for evaluation.
- Training a policy from preference data alone. A few hundred judgments from a
  single non-expert rater cannot carry that, and the KL anchor exists precisely
  to prevent it.

## Architecture

Four units. Two exist, one is extended, two are new.

| unit | status | responsibility |
|---|---|---|
| `rl/oneshot_env.py` | exists | goal → observation → action → attainment |
| hindsight episode sampler | new, inside the env | generate episodes from a sampled target transfer function |
| `reward_model.py` | new | Bradley-Terry scorer over measured features |
| `rl/oneshot_train.py` | extended | stage 2: swapped reward, KL penalty to `π₁` |

### Hindsight episode sampler

`OneShotEnv.reset()` currently samples an instruction and hopes the volume can
answer it. That is how "a bit more lungs" reached a rater on ts_s1371, a scan
whose voxels are 1.0 % lung against 6.9 % on a chest scan.

Inverted:

1. Sample a start transfer function (as now).
2. Sample a **target** transfer function.
3. Measure both with the volume's visibility model.
4. Derive the goal vector from the difference between the two aggregates,
   using the existing `goals.goal_vector` encoding.

Every episode is answerable by construction: the target *is* a solution. No
reachability filtering, no unanswerable instructions, unlimited episodes.

Instruction text is not needed during training at all — the policy consumes the
16-value goal vector. Language enters only at inference, through the LLM parser
in `commands.py`.

**Distribution-shift check (required, not optional):** a goal derived from a
random target may not resemble one derived from "show only the lungs". Stage 1
mixes hindsight episodes with instruction-derived episodes, and attainment is
reported per source. If hindsight-only training degrades attainment on
instruction-derived goals, the mixture ratio is the knob.

### Reward model (`reward_model.py`)

Input: the goal vector, plus per-class visibility and brightness before and
after — roughly 20 numbers, all already computed by
`goals.aggregate(model.features(params))` in about 15 ms.

Fit: Bradley-Terry on preference pairs from `out/vis_preferences.jsonl`. `equal`
judgments are used as ties; `skip` rows are dropped.

Rationale for features rather than image embeddings: with a few hundred pairs, a
768-dimensional DINOv2 embedding overfits, and it puts GPU inference inside the
RL loop. A feature model is also interpretable — its weights state which classes
the rater actually rewards, which is a result in itself. Image embeddings stay
in the backlog as an ablation once the pair count is in the thousands.

Interface:

```python
score(goal_vector, before_aggregate, after_aggregate) -> float
```

### Stage 2 training

Same environment, same observation, reward becomes:

```
reward = reward_model.score(goal, before, after) - beta * KL(pi || pi_stage1)
```

`beta` is swept and reported. This is the safeguard that lets a non-expert
rater's judgments tilt a competent policy without rebuilding it.

## Data

- **Source:** `out/vis_preferences.jsonl`, collected through `collect.py` at
  `127.0.0.1:8000/collect`.
- **Rater metadata:** already recorded per row (`id`, `role`, `experience`), so
  supervisor judgments later merge into the same pipeline without changes.
- **Assisted rows:** judgments made on 2026-09-17 while Claude commented on the
  pairs are flagged `assisted` and excluded from any set used to *evaluate* the
  reward model or the policies. They may still be used for training.
- **Split:** preference pairs are split by *instruction*, not by row, so the
  same instruction cannot appear in both fit and evaluation.

## Evaluation

All arms measured on the same held-out episodes (six unseen patients, as now).

| arm | what it tests |
|---|---|
| hill-climb, 200 evaluations | the oracle ceiling |
| `π₁` | can programmatic RL train a transfer function |
| `π₂` | does preference finetuning improve human-judged quality |
| `π₁` + rerank by reward model | is finetuning needed, or is scoring enough |
| metric-optimal render | does the proxy disagree with the human |

Two numbers matter:

1. **Metric-vs-human agreement.** How often the metric's preferred candidate
   matches the rater's. Needs no reward model and no training — computable as
   soon as enough unassisted judgments exist. High agreement validates the
   proxy; low agreement is the motivation for stage 2.
2. **Blind A/B win rate**, `π₂` over `π₁`, with a binomial confidence interval.

Predictions are pre-registered in a dated experiment document before the run, as
was done for `docs/experiments/2026-09-16-retrain-after-measurement-fixes.md`.

## Claim scope

With a single non-expert rater the claim is *"the policy can be aligned to a
rater's preferences"* — not *"the policy produces clinically better renders"*.
The write-up uses the weak form. Supervisor judgments, collected through the
same pipeline, are what would upgrade it.

## Risks

| risk | mitigation |
|---|---|
| Null result at n ≈ 300 (seed noise exceeded the effect in the v2/v3 comparison) | Pre-register; report the null honestly; widen n with supervisor sessions |
| Hindsight goals drift from instruction-derived goals | Mixed episodes, per-source attainment reported |
| Reward model overfits a small, single-rater dataset | Feature-based model, split by instruction, KL anchor in stage 2 |
| A differentiable optimiser outperforms everything on the proxy | Include it as a baseline; it cannot optimise preference, which is the point |

## Sequence

1. Collect unassisted preference judgments (in progress; on the critical path).
2. Metric-vs-human agreement analysis — no new training required.
3. Hindsight sampler in `OneShotEnv`, with the mixture check.
4. Stage-1 training, longer than 150k steps: the v3 seeds were still improving
   when the budget ran out (seed 1 best at its final step, seed 2 at 140k).
5. `reward_model.py` plus the reranking arm.
6. Stage-2 finetune, beta sweep.
7. Blind A/B.

Steps 1 and 2 are useful whatever happens to the rest.

## Measured after implementation (2026-09-17)

The distribution-shift check the design required is done, on the synthetic
volume, 200 episodes per arm. Hindsight targets are drawn as the start's
controllable values plus U(-0.25, 0.25) per group (`HINDSIGHT_NOISE`),
calibrated so the median requested change lands in the band real instructions
ask for.

| largest |delta| per episode (log10) | median | p90 | share > 1.0 |
|---|---|---|---|
| hindsight | 0.44 | 2.38 | 28 % |
| `goals.sample_instruction` | 0.60 | 1.00 | 2 % |

The medians agree. The tails do not, and this is not a step-size artefact:
shrinking the noise further pulls the median out of band while leaving the tail
in place. The cause is the `goals.EPSILON` floor — a perturbation that takes a
class from effectively invisible to visible produces a large log10 delta however
small the parameter step was. So hindsight training over-represents "make an
invisible class appear", which instructions rarely ask for.

Recorded rather than tuned away. The mitigation is the mixture the design
already requires: stage 1 mixes hindsight with instruction-derived episodes and
reports attainment per source, using the `goal_source` field in the env's info
dict. If a floor-crossing rejection turns out to be needed, that is the honest
lever, not the noise scale.

A second measurement worth keeping: the action that produces a hindsight target
scores median attainment 0.985 (p10 0.818), against a median of -2.343 for a
random action on the same episodes. The shortfall from 1.0 is `goals.distance`'s
keep term on classes the goal does not mention, whose peaks the target also
moved. The mentioned term is exactly 0 at the target, so the goal encoding and
the reward measure one quantity.
