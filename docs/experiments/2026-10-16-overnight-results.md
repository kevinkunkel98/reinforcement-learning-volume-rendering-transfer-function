# 2026-10-16: SAC arms, warm-start search, and model-based RL

**Summary.** No one-change SAC arm (solvable sampling, no drift charge,
start variety) moves the policy off ~0; only 2x training does (E9, +0.09).
Model-based RL -- a learned renderer model and a policy trained through it
with the exact goal metric -- reaches +0.23 alone on `vis_eval` (3 seeds,
p = 7e-14 vs E3, 7e-8 vs E9), positive on every instruction kind, and is now
the viewer's default policy. A 2-step closed loop (the policy re-observes
the real render once) adds +0.05 in every seed. As a warm start, 30 renders
from its proposal match 200-render search. More data, more patients (90
instead of 20) and policy capacity no longer help: about 30 % of val
instructions cannot be improved by any method under the current metric,
which is the next thing to address.

Follow-up to [the relative diagnosis](2026-10-15-relative-diagnosis.md).
Every arm is the E3 recipe (150k steps, `oneshot-v9`, residual/target,
`--hindsight-ratio 0.25 --balance-classes --hindsight-brightness 0.1`, the
same hindsight + search pretrain files per seed) with **one** change, 3 seeds.

| arm | change vs E3 | out dirs |
|---|---|---|
| E6 | `--solvable-instructions` | `out/rl_v4/e6_v9_searchmix_bright_solvable_150k_seed{0,1,2}` |
| E7 | E6 + `--target-drift-weight 0` (reward = clipped attainment) | `out/rl_v4/e7_v9_solvable_nodrift_150k_seed{0,1,2}` |
| E8 | E6 + `--start-variety 0.5` | `out/rl_v4/e8_v9_solvable_startvar_150k_seed{0,1,2}` |
| E9 | E3 at 300k steps | `out/rl_v4/e9_v9_searchmix_bright_300k_seed{0,1,2}` |

Evaluation: `rl.vis_eval --split val --episodes 200` on each `best.zip`
(identical episodes for every arm, full instruction mix including
unsolvable ones), compared with `tools.compare_runs` (seed-averaged
per-episode medians, paired Wilcoxon).

**Provenance note.** `rl/vis_eval.py` was edited (refinement only, commit
5fa822d) while the E6 evaluations were running, so the E3/E6/E7 result files
carry different scoring fingerprints and `compare_runs` refuses them. The
non-refinement scoring path did not change: the deterministic baseline rows
(B3, B4, B5) are bit-identical across all files. The comparisons below
bypass the fingerprint check for that reason only.

## Results (200 val episodes, seed-averaged median attainment)

| | overall | relative | compound | show_only | absolute | brightness |
|---|---|---|---|---|---|---|
| E3 | +0.002 | +0.040 | +0.125 | −0.183 | +0.006 | +0.074 |
| E6 | −0.011 | +0.015 | +0.077 | −0.139 | +0.021 | +0.338 |
| E7 | −0.013 | +0.029 | +0.096 | −0.130 | −0.006 | +0.287 |
| hill_climb 200 | 0.333 | 0.387 | 0.460 | 0.152 | 0.256 | 0.939 |

- E6 vs E3: Δ −0.013, p = 0.02 (E3 better overall); brightness +0.26.
- E7 vs E6: Δ −0.002, p = 0.20 — **the double drift charge is not what
  holds the policy back.**
- E7 vs E3: Δ −0.014, p = 0.22.

Relative split (`tools.relative_diagnosis`, 200 relative episodes, 90
solvable / 110 unsolvable), median per seed:

| | solvable | unsolvable |
|---|---|---|
| E3 | 0.32 / 0.23 / 0.26 | 0.00 / 0.00 / 0.00 |
| E6 | 0.32 / 0.28 / 0.29 | −0.08 / −0.06 / −0.07 |

Solvable-only training buys a little on solvable instructions and loses on
the unsolvable ones it never saw. It is a cleaner task definition, not a
learning improvement.

## Probes on E3 (150 real val instructions, no hindsight)

1. **Scaling the policy's action** by k: k = 0.5 beats k = 1.0 in all three
   seeds (overall median +0.045 vs +0.017 for seed 0). The policy is not
   timid; it overshoots and moves too much at once. Masking the action to
   the mentioned classes' peaks *hurts* relative/compound (occluder changes
   are needed). show_only is best at k = 0: any policy action makes it worse.
2. **More search-distilled pretraining data does not help.** The fit to
   search actions improves steadily with data (val/zero MSE 0.66 → 0.55 from
   300 to 2600 samples), but every pretrain-only actor (5 mixes, incl. all
   3000 search episodes) scores a *negative* median (−0.07 … −0.25). MSE
   imitation of a multi-modal teacher averages solutions into a blurry
   action that moves everything a bit. The positive numbers come from RL.
3. **Expert iteration does not help either.** Refining E3's own proposals
   with 60-evaluation search (+0.18 median gain per episode) and
   fine-tuning the actor on the refined actions (1500 train episodes, fit
   val/zero 0.10) leaves the policy alone flat (+0.017 → +0.007/+0.009).
4. **The policy is a strong warm start for search.** `hill_climb` from the
   policy's proposal, measuring the instruction from the episode's own start
   (new `reference_params`, step 0.1):

   | budget | cold search (from start) | warm search (from policy) |
   |---|---|---|
   | 30 evaluations | 0.046 | **0.184** |
   | 60 evaluations | 0.128 | **0.267** |
   | 200 (B4, step 0.2) | 0.333 | — |

   Relative with 60: 0.66 warm vs 0.42 cold (0.39 for 200-evaluation
   search). This is the strongest positive result for "RL amortises search".
   Splitting 60 evaluations into 30 warm + 30 cold and keeping the better
   gives 0.246 overall (show_only +0.04 instead of +0.00), so plain warm
   search is the better default.
   `vis_eval --refine N` reported numbers before commit 5fa822d measured
   the goal from the proposal (applying relative goals twice) and are
   invalid.

## Training curves

Validation median at 150k is still rising in every E3/E6/E7 seed (E6 seed 0:
0.17 at 70k, 0.41 at 150k), hence E9.

## E8 and E9

Same 200 val episodes, `tools.compare_runs` (fingerprint bypass as above):

| | overall | relative | compound | show_only | absolute | brightness |
|---|---|---|---|---|---|---|
| E8 (start variety) | −0.007 | −0.022 | +0.066 | −0.059 | −0.015 | +0.267 |
| E9 (E3 at 300k) | **+0.093** | +0.144 | +0.201 | −0.058 | +0.002 | +0.432 |

- E8 vs E3: Δ −0.008, p = 0.43; vs E6: Δ +0.004, p = 0.31. Start variety
  is neutral, like solvable sampling and the drift weight.
- **E9 vs E3: Δ +0.091, p = 4e-8.** Twice the training is the only SAC
  change that helped: SAC is not stuck, it is sample-hungry. show_only stays
  negative.

## Closed-loop SAC (negative)

Re-observing after each step with the remaining goal (instruction minus
progress so far; unmentioned classes that drifted past `KEEP_TOLERANCE`
become "undo" targets):

- Zero-shot, E3 applied repeatedly: k = 1 diverges after 2 steps; k = 0.5
  with the undo targets lifts relative from 0.13 to 0.26 over 4 steps but
  show_only falls, overall flat (150 episodes).
- Fine-tuning E3 on a 2- and 3-step env (critic-only warm-up 5k steps,
  gamma 0.95): both collapse once the actor trains (3-step to −2.9, 2-step
  slides to −0.15 by 30k). Stopped.

## Model-based RL (new default)

`rl/model_based.py`. The renderer's physics does not depend on the
instruction, so learn it once and train the policy through it with the
exact goal metric:

1. **Dynamics:** 5-MLP ensemble, (histogram, solo_max, start controllable
   values, start state, residual action) → change in per-class log10
   visibility, brightness, coverage, plus a useless logit. Data: 3000
   training-split starts × 20 random TF changes (dense small, sparse large,
   no-op), 60k renders, ~10 min on 3 workers.
2. **Policy:** MLP on the one-shot observation layout, residual actions,
   trained by backpropagating a torch twin of `goals.distance` through the
   frozen ensemble (+ disagreement penalty β = 1, + predicted-useless
   penalty). Zero renders at inference.
3. **Dyna:** render the policy's own proposals (+ noise) on 1500 new starts,
   refit model and policy.

Model fit on held-out starts (rank correlation of predicted vs real
attainment for unseen start/action/instruction triples): 0.91 (round 0),
0.93 (round 1), 0.945 (round 2).

Policy alone, 200 val episodes (seeds 50 000+, scratch scripts, same
episodes for every row; SAC E3 re-scored on them), seed-averaged median:

| | overall | absolute | brightness | compound | relative | show_only | vs E3 |
|---|---|---|---|---|---|---|---|
| SAC E3 | +0.001 | +0.021 | +0.115 | +0.045 | +0.046 | −0.290 | |
| MB round 0 (60k renders) | +0.115 | +0.101 | +0.363 | +0.127 | +0.115 | +0.106 | 71 %, p = 2e-9 |
| **MB round 1 (+30k)** | **+0.182** | +0.142 | +0.618 | +0.150 | +0.346 | +0.116 | 79 %, p = 5e-20 |
| MB round 2 (+30k) | +0.174 | +0.122 | +0.621 | +0.193 | +0.296 | +0.146 | 77 %, p = 2e-18 |

Per seed (each seed's own median, a different statistic from the table's
seed-averaged per-episode median): round 0 0.123 / 0.098 / 0.133, round 1 0.215 / 0.224 / 0.199, round 2 0.182 / 0.168 / 0.183. One Dyna round gives the gain; a
second does not add to it. The model is still optimistic (it predicts
0.36–0.43 for its own policy), so the remaining gap is model exploitation.

Every kind is positive, including show_only, which every SAC arm made
worse. About 90k renders against E9's 300k environment steps for half the
score.

**Policy + warm search** (round-0 policy, seed 0, `hill_climb` from the
proposal measured from the episode start, step 0.1, better of
{proposal, refined}; 200 episodes):

| renders | cold search | SAC E3 + search | MB + search |
|---|---|---|---|
| 0 | 0 | +0.011 | +0.123 |
| 30 | +0.051 | +0.186 | **+0.327** |
| 60 | +0.134 | +0.279 | **+0.406** |
| 200 cold (B4) | 0.333 | | |

30 renders from the model-based proposal match 200-render search; 60 beat
it (absolute 0.58 vs 0.26, show_only 0.29 vs 0.15). A better learned start
gives better search at every budget, which also answers "would any warm
start do": the SAC start is clearly worse.

These numbers come from scratch scripts on a fixed seed range
(`2026-10-16-scripts/`); the `rl.vis_eval` rows below are the official ones.

### Official scores (`rl.vis_eval`, `rl.model_based` checkpoints)

`rl.model_based train` on rounds 0 + 1 (90k renders), 3 seeds
(`out/rl_mb/r1_seed{0,1,2}`, per-seed val medians 0.165 / 0.202 / 0.191),
200 `vis_eval` val episodes, `tools.compare_runs` against the SAC arms
(fingerprints differ only because `vis_eval.py` gained the `.pt` loader;
bypassed as above):

| | overall | absolute | brightness | compound | relative | show_only | p |
|---|---|---|---|---|---|---|---|
| SAC E3 | +0.002 | +0.006 | +0.074 | +0.125 | +0.040 | −0.183 | |
| SAC E9 (300k) | +0.093 | +0.002 | +0.432 | +0.201 | +0.144 | −0.058 | |
| **Model-based round 1** | **+0.233** | +0.206 | +0.722 | +0.268 | +0.212 | +0.148 | 7e-14 vs E3, 7e-8 vs E9 |

Seed 1 ships as the viewer's default (`policy.py`).

### What does not move the pure policy

All on the same 200 val episodes, policy alone, own median per run:

- **Policy-side knobs** (seed-1 dynamics reused, `--dynamics`): baseline
  0.202, β = 0 0.185, β = 0.3 0.181, β = 3 0.197, 512×3 network 0.206,
  20k steps 0.213. Within noise; the policy already optimises the model.
- **More Dyna rounds:** round 2 0.182 / 0.168 / 0.183 (no gain over round 1).
- **Start variety in the model data** (1500 starts, half with organs made
  visible): 0.217 vs 0.202, seed 1 -- within noise.
- **More patients.** `tools.extend_totalseg` adds the zip's other 70
  eligible subjects as split `train_extra` (val/test unchanged). Same data
  size, 4.5x the patients: 60k random renders on 90 patients 0.121 vs on 20
  patients 0.115. All data (210k renders, 90 patients): 0.170 / 0.194 /
  0.187 vs round 1's 0.165 / 0.202 / 0.191. Model rank correlation
  0.89 → 0.94, policy unchanged.

Val is only 4 patients, so ±0.02 is noise.

### Ceiling: what is achievable at all

Best of {model-based proposal + 1000-evaluation `hill_climb`, cold
1000-evaluation `hill_climb`} per episode (33 val episodes,
`2026-10-16-scripts/ceiling.py`), against the policy on the same episodes:

| kind | solvable | policy | ceiling | policy / ceiling |
|---|---|---|---|---|
| absolute | – | 0.24 | 0.82 | 0.46 |
| brightness | – | 0.82 | 1.00 | 0.89 |
| compound | yes | −0.15 | 0.77 | – |
| compound | no | −0.01 | 0.10 | – |
| relative | yes | 0.54 | 1.00 | 0.57 |
| relative | no | 0.00 | 0.01 | – |
| show_only | – | 0.03 | 0.41 | 0.26 |

10 of 33 episodes have a ceiling below 0.1: no method reaches them. They
are instructions about a class the start hides (`goals.instruction_solvable`
false): revealing it means fading its occluders, which `goals.distance`
charges as keep-term drift by more than the target gains. On the solvable
ones the policy gets about half of what is achievable.

### Closed loop with the model-based policy (pure RL, 2 renders)

The policy takes K steps; after each one the real render is re-observed
with the *remaining* goal (instruction minus progress so far; unmentioned
classes drifted past `KEEP_TOLERANCE` become undo targets). Trained by
backpropagating the final normalised distance through a K-step rollout in
the learned model, starting from the one-shot all-data checkpoints
(`2026-10-16-scripts/mb_multistep.py`, 6000 steps).

- Zero-shot (one-shot policy applied repeatedly) does not work: 0.19 →
  0.07 → −0.03 → −0.20 for k = 1; damped k = 0.5 peaks at 0.16.
- Trained, 200 val episodes:

| seed | one-shot | closed loop, step 1 | closed loop, step 2 |
|---|---|---|---|
| 0 | 0.170 | 0.209 | **0.220** |
| 1 | 0.194 | 0.224 | **0.245** |
| 2 | 0.187 | 0.249 | **0.241** |
| 1, 20k training steps | | **0.264** | 0.255 |
| 1, K = 3 | | 0.211 | 0.197, 0.231 (step 3) |

Seed 1, step 2 by kind: absolute 0.25, brightness 0.90, compound 0.33,
relative 0.40, show_only 0.08. +0.05 over one-shot in every seed, compound
0.13 → 0.33, brightness 0.72 → 0.90. The multi-step-trained policy's
*first* step is also better than the one-shot policy (it learns to leave
room for the correction). show_only falls at step 2 in every run (0.16 →
0.08); clamping the remaining goal to what is reachable (not below the
visibility floor) does not fix it (0.194 overall).

## Where this leaves the pure policy

| | overall median |
|---|---|
| SAC E3 | 0.00 |
| SAC E9 (2x training) | 0.09 |
| model-based, one-shot | 0.18 – 0.23 |
| model-based, 2-step closed loop | 0.22 – 0.25 |

The remaining limit is the task definition, not the learner: about 30 % of
val episodes cannot be improved by any method under `goals.distance`, and
more data, patients or policy capacity no longer move the score. Median 0.5
on every kind is out of reach for any method while that holds.

**Next (thesis design question): make hidden-organ instructions solvable.**

1. *Default TF with visible organs.* Start from a transfer function where
   organs are not hidden behind soft tissue (the viewer's default and the
   env's start distribution), so more instructions are solvable at all.
2. *Occlusion-aware metric.* When a target class is hidden, fading its
   occluders is what the user asked for, not drift: exempt a mentioned
   class's occluders from the keep terms. A metric change, so it is reported
   next to the current metric, never instead of it.

Both are re-scored with the ceiling probe first, so the achievable level is
known before any training.

## Pending

- Test split, once, for the final configuration.
- Port the closed-loop trainer (`mb_multistep.py`) into `rl/` if it stays.
