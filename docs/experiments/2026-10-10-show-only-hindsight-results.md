# Show-Only Hindsight Demonstrations: Results (seeds 0-2, 150k steps)

## Purpose

`OneShotEnv._sample_hindsight_show_only` (`docs/superpowers/specs/2026-10-09-show-only-hindsight-design.md`)
adds a `--hindsight-show-only` share: hindsight demonstrations shaped like a
real `show_only` instruction (1-2 classes pushed toward high visibility,
every other class pushed toward low), mirroring the brightness-hindsight
mechanism (`docs/experiments/2026-10-07-search-distilled-pretraining-results.md`)
that fixed a severe brightness regression. This run tests whether the same
mechanism helps `show_only` attainment the way it helped brightness.

## Setup

E4 = the E3 recipe (`docs/experiments/2026-10-07-search-distilled-pretraining-results.md`)
plus `--hindsight-show-only 0.1`, both at pretrain time and continuously
during the 150k RL steps. Same hyperparameters otherwise: `oneshot-v9`,
`--action-mode residual --reward-mode target`, `--hindsight-ratio 0.25
--balance-classes --hindsight-brightness 0.1`, search-distilled pretrain
(1000 ep @ 200 evaluations), 200-episode validation eval every 10k steps,
each seed with its own freshly collected `hindsight_v9_bright_showonly_seed{0,1,2}.npz`.

| run | pretrain | out dir |
|---|---|---|
| E3 (baseline) | hindsight+brightness + search-distilled | `out/rl_v4/e3_v9_searchmix_bright_150k_seed{0,1,2}` |
| E4 | E3 + `--hindsight-show-only 0.1` | `out/rl_v4/e4_v9_searchmix_bright_showonly_150k_seed{0,1,2}` |

Metrics below use this project's own convention: each seed's value is the
mean of its last 3 validation evals (steps 130k/140k/150k), to reduce
single-eval noise from the 200-episode sample; median-of-3 is the median
across seeds. **Note:** an earlier look at this data (mid-run, before all 3
seeds had finished) compared raw single-eval (150k-only) numbers instead and
suggested `show_only` was getting *worse*, not better. That read doesn't
survive switching to the project's own noise-reduced convention below — the
150k-only eval for `show_only` is apparently a noisy one for several of
these runs. The numbers in this document are the ones that should be
trusted.

## Results

| metric | E3 s0 | E3 s1 | E3 s2 | **E3 med** | E4 s0 | E4 s1 | E4 s2 | **E4 med** | seeds improved |
|---|---|---|---|---|---|---|---|---|---|
| median_attainment | 0.180 | 0.143 | 0.153 | **0.153** | 0.165 | 0.160 | 0.118 | **0.160** | 1/3 |
| absolute | 0.176 | 0.187 | 0.073 | **0.176** | 0.099 | 0.179 | 0.129 | **0.129** | 1/3 |
| relative | -0.068 | -0.102 | -0.064 | **-0.068** | -0.088 | 0.001 | -0.030 | **-0.030** | 2/3 |
| compound | 0.088 | 0.039 | 0.080 | **0.080** | 0.034 | 0.012 | -0.046 | **0.012** | 0/3 |
| show_only | 0.096 | 0.055 | 0.061 | **0.061** | 0.089 | 0.062 | 0.087 | **0.087** | 2/3 |
| brightness | 0.246 | 0.219 | -0.010 | **0.219** | 0.282 | 0.243 | 0.269 | **0.269** | 3/3 |
| share_positive | 0.718 | 0.708 | 0.677 | **0.708** | 0.725 | 0.718 | 0.700 | **0.718** | 3/3 |

"Seeds improved" = how many of the 3 seed-matched pairs (same seed, same
pretrain data otherwise) have E4 > E3 for that metric.

## Findings

1. **`show_only` attainment improves modestly, but not cleanly.** Median-of-3
   goes from +0.061 to +0.087, and 2 of 3 seeds individually improve (seed 1:
   +0.055 -> +0.062; seed 2: +0.061 -> +0.087). Seed 0 is the exception,
   going slightly backward (+0.096 -> +0.089). This is a real but small
   effect — nothing like the brightness-hindsight fix's 2-3x swing from
   strongly negative to strongly positive. It does not mirror brightness's
   result.

2. **`compound` consistently regresses** (0.080 -> 0.012 median-of-3, worse
   in all 3 seeds, seed 2 even flips negative). This is the clearest, most
   consistent effect in this dataset. Likely cause: `compound` instructions
   mix a visibility goal with other goal types in one instruction, and the
   show_only-anchored hindsight data (which sets an explicit target on
   *every* class, all 8, every episode) is the most different in shape from
   both compound's own goal format and from the hindsight-only/brightness
   data the actor saw in E3 — diluting the fit on compound specifically.

3. **`absolute` is mixed-to-worse** (0.176 -> 0.129 median-of-3, only 1/3
   seeds improved, seed 0 drops substantially). Same general diagnosis as
   compound: a non-trivial fraction of pretrain+training signal is now
   spent on show_only-shaped data instead.

4. **`brightness` and `share_positive` both improve consistently** (3/3
   seeds each). This is the most unexpected result and should not be
   over-attributed to the show_only mechanism itself — introducing
   `--hindsight-show-only` also changes the overall mix of demonstrations
   the actor is pretrained and anchored on (a confound: it's not an
   isolated change, it shifts the composition of every pretrain/hindsight
   batch). A plausible story is that an even broader mix of fully-specified,
   explicit per-class targets (show_only's hindsight shape sets a target on
   all 8 classes, unlike brightness's or the generic sampler's partial
   targets) generalizes better to brightness instructions too, but this
   dataset doesn't distinguish that from other explanations (e.g. the
   show_only data simply displacing noisier generic-hindsight episodes).
   Worth a dedicated follow-up if this matters later; not claimed here.

5. **Overall `median_attainment` is roughly a wash** (0.153 -> 0.160
   median-of-3, but only 1/3 seeds individually better — the median-of-3
   uptick is driven mostly by seed 1's balanced small gains, while seed 0 and
   seed 2 both slightly decline). This is consistent with findings 1-3: a
   small `show_only` gain and a large `brightness` gain, partially offset by
   a clear `compound` loss and a mixed `absolute` result.

## Decision

**Do not adopt `--hindsight-show-only 0.1` as part of the default training
recipe.** It does not reproduce brightness-hindsight's clean, large win on
its own target metric — the `show_only` improvement is real but small and
inconsistent (2/3 seeds) — and it costs a consistent, clear regression on
`compound` (0/3 seeds improved) plus a mixed result on `absolute`. The
`brightness` and `share_positive` gains are real on this data but
confounded with the overall pretrain-mix change, so they shouldn't be used
to justify shipping this flag either.

This suggests the hindsight-demonstration mechanism that worked for
brightness doesn't transfer cleanly to `show_only` — plausibly because
`show_only`'s hindsight shape (an explicit target on every one of 8 classes,
every episode) is a much larger and more specific slice of the action space
to anchor than brightness's single-class nudge, and at a 10% share it both
under-delivers on its own goal and crowds out signal for other instruction
kinds. Two directions worth considering before trying this again:

- **Reward shaping instead of hindsight demonstrations** for `show_only`
  specifically (an idea raised earlier in this project: reward maximizing
  the shown tissue and minimizing all else directly, rather than teaching it
  via relabelled demonstrations) — this was the original idea before the
  hindsight approach was chosen, and this result is a reason to revisit it.
- **A smaller `--hindsight-show-only` share** (e.g. 0.03-0.05) to anchor the
  behavior without displacing as much of the other instruction kinds'
  training signal, if the hindsight-demonstration approach is retried.

The code itself (the `_sample_hindsight_show_only` sampler, its tests, and
the CLI wiring) stays in the codebase — it's correct and tested, just not
turned on by default; `--hindsight-show-only` defaults to `0.0` already, so
no revert is needed, only a decision not to add it to the recommended
command line.

Open, as with E3: no held-out *test*-split evaluation has been run for any
of these configs, only validation-split `eval_progress.csv` numbers.
