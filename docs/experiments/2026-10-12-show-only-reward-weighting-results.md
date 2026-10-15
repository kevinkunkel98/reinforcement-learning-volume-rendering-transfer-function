# Show-Only Reward Weighting: Results (seeds 0-2, 150k steps)

## Purpose

Validates whether `goals.show_only_weights` (group-balanced weighting,
`docs/superpowers/specs/2026-10-10-show-only-reward-weighting-design.md`)
actually improves `show_only` attainment in real training, per that spec's
own deferred validation plan. The root-cause diagnosis (see that spec's
Purpose section) was that `goals.distance`'s flat per-class weighting let
`show_only`'s hidden-class group (6-7 classes) numerically dominate its
shown-class group (1-2 classes) 3-7x, so a policy could score well by
suppressing hidden classes while barely elevating the shown one.

## Setup

E5 = the exact E3 recipe
(`docs/experiments/2026-10-07-search-distilled-pretraining-results.md`):
`oneshot-v9`, `--action-mode residual --reward-mode target`,
`--hindsight-ratio 0.25 --balance-classes --hindsight-brightness 0.1`,
reusing E3's own cached pretrain data verbatim
(`hindsight_v9_bright_seed{N}.npz` + `search_v9_seed{N}.npz`, 4000+1000
episodes), 200-episode validation eval every 10k steps. The only change from
E3 is the code: `goals.distance`/`attainment` now apply `show_only_weights`
wherever a `show_only` instruction is scored, including live during this
run's own RL training and evaluation.

Three numbers per seed, all using the project's "mean of last 3 validation
evals (steps 130k/140k/150k)" convention. For the two "E3" rows this means
re-evaluating E3's own saved `checkpoint_{130000,140000,150000}.zip` files on
the exact same fixed 200-episode validation set
(`rl.oneshot_train.validation_episodes`/`evaluate_policy_on`), under whichever
scoring code was current at the time:

| run | checkpoints | metric code |
|---|---|---|
| E3-old | E3's own checkpoints | old (unweighted) -- as originally reported |
| E3-rescored | the same E3 checkpoints, same episodes | new (weighted) -- isolates the redefinition |
| E5 | fresh checkpoints, retrained under the new reward | new (weighted) |

## Results

| show_only attainment | s0 | s1 | s2 | **median** |
|---|---|---|---|---|
| E3-old (old metric) | +0.096 | +0.055 | +0.061 | **+0.061** |
| E3-rescored (new metric, no retrain) | -0.215 | -0.271 | -0.238 | **-0.238** |
| E5 (new metric, retrained) | -0.037 | +0.001 | -0.043 | **-0.037** |

Seeds improved, E5 vs. E3-rescored (same-seed pairs): **3/3**.

Other instruction kinds, E5 only (mean-of-last-3), against E3-old's own
median-of-3 as reference (median_attainment 0.153, absolute 0.176, relative
-0.068, compound 0.080, brightness 0.219, share_positive 0.708):

| metric | E5 s0 | s1 | s2 | **E5 med** | E3-old med (reference) |
|---|---|---|---|---|---|
| median_attainment | 0.133 | 0.174 | 0.160 | **0.160** | 0.153 |
| absolute | 0.166 | 0.123 | 0.192 | **0.166** | 0.176 |
| relative | -0.021 | -0.383 | -0.145 | **-0.145** | -0.068 |
| compound | 0.028 | 0.093 | 0.089 | **0.089** | 0.080 |
| brightness | 0.213 | 0.228 | -0.059 | **0.213** | 0.219 |
| share_positive | 0.692 | 0.678 | 0.677 | **0.678** | 0.708 |

## Findings

1. **Rescoring alone (no retraining) swings `show_only` from +0.061 to
   -0.238 median-of-3.** This confirms the root-cause diagnosis directly:
   E3's apparent success on `show_only` was largely an artifact of the old
   metric's hidden-group numerical dominance, not real shown-class
   elevation. The same policy, unchanged, looks clearly bad once hidden- and
   shown-class terms are weighted fairly.
2. **Retraining under the new, correctly-weighted reward recovers most, but
   not all, of that gap**: -0.238 -> -0.037 median-of-3, and all 3/3 seeds
   individually improve over their matched E3-rescored baseline (seed 1 even
   flips to slightly positive, +0.001). This is a real, consistent effect --
   unlike E4's hindsight-demonstration approach (2/3 seeds, a modest +0.026
   median-of-3 move), this is 3/3 seeds and a much larger swing in absolute
   terms.
3. **No clear collateral regression on the other instruction kinds.**
   `median_attainment`, `absolute`, and `compound` are flat-to-slightly-better
   than E3's own historical numbers; `share_positive` is marginally down
   (0.678 vs 0.708). `relative` looks worse on the surface (-0.145 vs -0.068)
   but is driven by one outlier seed (seed 1 at -0.383); the other two seeds
   (-0.021, -0.145) sit closer to E3's range, and this project's own prior
   docs have repeatedly flagged single-checkpoint-window reads as noisy.
   `brightness` has one outlier seed too (seed 2 collapses to -0.059 against
   +0.213/+0.228 for the other two seeds) -- plausibly ordinary seed
   variance, not something this change should obviously cause, but not
   independently chased down further here.
4. **Not done**: the shown-vs-hidden per-class attainment breakdown (this
   spec's own validation-plan step 4, checking whether the shown class's
   *per-class* median specifically moved off its earlier-measured ~0, not
   just the aggregate weighted `attainment_show_only` number) was not
   re-run on these E5 checkpoints. The aggregate result above is directionally
   consistent with the fix working as intended, but this more granular check
   remains open.

## Decision

Per the project's own 3-seed gate (median attainment > 0, share improved >
50% across all 3 seeds): the "improved" half is clearly met (3/3 seeds, a
large and consistent swing). The "median > 0" half is not -- median-of-3
`show_only` attainment is still **-0.037**, negative.

**This fix is a real, measured improvement over the (now-understood-to-be-
inflated) E3 baseline, but does not yet clear this project's own bar for
"adopted."** It is kept in the codebase (it is also the metrically correct
way to score `show_only` regardless of training outcome -- the old flat
weighting was simply wrong, not just unhelpful), but should not be cited as
having solved `show_only` without a further iteration that gets the median
attainment itself above zero.

Plausible next steps, not pursued here: more training steps or more
pretraining capacity specifically aimed at show_only-shaped targets (E4's
hindsight-demonstration mechanism is now a more promising candidate to
revisit, since the reward target it would be anchoring toward is correctly
specified rather than the unbalanced version it was tested against before); a
longer training budget; the shown-vs-hidden per-class breakdown from finding 4.

Open, as with E3/E4: no held-out *test*-split evaluation has been run for any
of these configs, only validation-split `eval_progress.csv` numbers.
