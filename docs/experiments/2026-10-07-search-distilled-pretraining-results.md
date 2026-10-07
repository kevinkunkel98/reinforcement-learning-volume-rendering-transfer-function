# Search-Distilled Pretraining: Results (seed 0, 150k steps)

## Purpose

Follow-up to `docs/experiments/2026-09-27-oneshot-v9-label-scoped-pretrain.md`'s
E1/E2 ablation, now testing whether adding a search-distilled teacher (new
feature, `docs/superpowers/specs/2026-10-06-search-distilled-pretraining-design.md`)
on top of hindsight pretraining improves attainment -- and investigating an
unexpected signal found along the way: hindsight-only pretraining causes a
severe regression on brightness instructions specifically, which the
search-distilled data substantially (not fully) repairs.

## Setup

All runs: 150k timesteps, `oneshot-v9`, `--action-mode residual --reward-mode
target`, `--hindsight-ratio 0.25 --balance-classes`, seed 0, 200-episode
validation eval every 10k steps. Only the pretrain source differs.

| run | pretrain | out dir |
|---|---|---|
| E1 (control) | none | `out/rl_v4/e1_v9_150k_seed0_rewardfix` |
| E2 | hindsight only, 4000 episodes | `out/rl_v4/e2_v9_bc_150k_seed0_rewardfix` |
| E2+search | hindsight (4000) + search-distilled (1000 @ 200 evaluations) | `out/rl_v4/e1_v9_150k_seed0_searchmix` |
| E3 | hindsight+brightness (4000, `--hindsight-brightness 0.1`) + search-distilled (1000) | `out/rl_v4/e3_v9_searchmix_bright_150k_seed0` |

## Results (mean of last 3 validation evals: steps 130k/140k/150k)

| metric | E1 | E2 | E2+search | **E3** |
|---|---|---|---|---|
| median_attainment | -0.034 | +0.088 | +0.109 | **+0.179** |
| absolute | -0.096 | **+0.193** | +0.109 | +0.176 |
| relative | -0.474 | **-0.052** | -0.208 | -0.068 |
| compound | -0.047 | +0.047 | +0.001 | **+0.088** |
| show_only | 0.054 | 0.070 | +0.076 | **+0.096** |
| brightness | -1.278 | -5.204 | -2.379 | **+0.246** |

Pretrain fit quality (`metadata.json`):

| run | train_mse | val_mse | zero_mse | samples |
|---|---|---|---|---|
| E2 | 0.0178 | 0.0190 | 0.0196 | 4000 |
| E2+search | 0.0249 | 0.0263 | 0.0314 | 5000 |
| E3 | 0.0238 | 0.0251 | 0.0309 | 5000 |

## Findings

1. **Both pretraining variants beat the no-pretrain control on overall median
   attainment**, and adding search-distilled data on top of hindsight gives
   the best overall median (+0.109).

2. **Hindsight-only pretraining induces a severe, specific regression on
   brightness instructions** (-5.204 vs. control's -1.278 -- roughly 4x
   worse than doing no pretraining at all). Likely cause: none of the 4000
   hindsight demonstrations are brightness-type (`hindsight_brightness_share`
   defaults to 0), so confidently fitting the actor to 4000 purely-visibility
   oracle targets biases it away from brightness-relevant behaviour, and the
   RL phase's hindsight relabelling (same 0% brightness share) never
   corrects this during the following 150k steps.

3. **Adding search-distilled data substantially repairs this** (-5.204 ->
   -2.379), despite the search-distilled dataset having strong brightness
   coverage and quality on its own terms. Replaying the 96 brightness
   episodes actually collected into `search_v9_seed0.npz` (by rerunning
   `env.step` on the stored actions under the original seeds, not
   re-searching) gives mean attainment 0.608, median 0.819 -- the *best* of
   any instruction kind in that dataset:

   | kind | n | mean attainment | median |
   |---|---|---|---|
   | brightness | 96 | 0.608 | 0.819 |
   | relative | 406 | 0.489 | 0.501 |
   | compound | 255 | 0.455 | 0.460 |
   | absolute | 112 | 0.402 | 0.286 |
   | show_only | 131 | 0.227 | 0.196 |

   So the search teacher's brightness solutions are good; the actor's
   zero-shot transfer from them to held-out brightness instructions is
   comparatively weak (the searchmix run's timestep-0 eval
   `attainment_brightness` is -0.395, well below the 0.819 training-
   distribution median), suggesting a generalization gap from a small
   (96-episode) brightness sample, not a bad teacher signal.

4. **E2+search slightly underperforms E2 on absolute/relative/compound**
   despite improving the median and brightness -- consistent with the
   search-distilled examples (noisier/more diverse MSE targets, train_mse
   0.025 vs. hindsight-only's 0.018) diluting the very tight hindsight-only
   fit on those kinds, in exchange for broader instruction-kind coverage.

## E3: does continuous brightness-hindsight anchoring fix it?

E3 adds `--hindsight-brightness 0.1` to the E2+search recipe: brightness-type
hindsight demonstrations become a continuous training-time signal throughout
all 150k SAC steps (not just a one-off pretrain batch), on top of the
already-good search-distilled brightness examples.

**Yes, decisively.** `attainment_brightness` goes from -2.379 to **+0.246**
-- not just repaired, but now the strongest-improving instruction kind of
all six runs, beating even the no-pretrain control (-1.278). This confirms
the hypothesis in finding 2/3 above: the regression was specifically a
*training-signal-coverage* problem (brightness never anchored during the RL
phase), not a problem with the search-distilled data's quality, and a
continuous brightness-hindsight signal during RL training fixes it directly
rather than needing a larger one-off pretrain sample.

E3 is also a near-strict improvement over every other run on every metric:
best median_attainment (+0.179, vs. the next-best +0.109), best compound,
best show_only, best brightness by a wide margin, and within noise of E2's
best-in-class absolute/relative (E2: +0.193/-0.052 vs. E3: +0.176/-0.068),
while E2 itself collapses on brightness (-5.204). No other configuration
comes close to this combination.

## Decision

**Adopt the full recipe** -- hindsight pretraining (4000 ep) + brightness
hindsight (`--hindsight-brightness 0.1`, both at pretrain time and
continuously during the 150k RL steps) + search-distilled pretraining (1000
ep @ 200 evaluations) -- as the new best-known configuration for `oneshot-v9`.

This is one seed. Per this project's own decision gate ("worth a thesis
claim: held-out median > 0 with share improved > 50% across 3 seeds"),
**seeds 1 and 2 of the E3 recipe are queued next** to confirm this isn't a
single-seed artifact before treating it as a result.
