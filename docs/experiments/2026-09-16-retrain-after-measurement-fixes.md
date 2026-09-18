# Retrain after the measurement fixes

Written **before** the run, so the analysis cannot be chosen to fit whatever
comes out. Recorded predictions, the protocol, and what would count as the
fixes not having helped.

Date: 2026-09-16. Code state: `40895be`.

## What changed since the v2 checkpoints

Four defects, each committed with its evidence:

| commit | defect | what it touched |
|---|---|---|
| `cc167af` | the policy's colour action wrote one scalar into r, g and b, forcing every policy render grey | preference collection was not blind |
| `a0b009a` | instructions were sampled from label presence, not from whether a render could show the tissue | unanswerable items shown to raters and scored in evaluation |
| `62b5720` | `solo_max` probed with the retired band layout, whose peak 0 is fat (−100 HU) not lungs (−800 HU) | **the policy's observation**, absolute-level targets, and render validation |
| `40895be` | the viewer started and reset from the band layout while the policy was trained on the anatomical one | policy mode in the UI adjusted the fat peak for "more lungs" |

`62b5720` is the one that matters for training: the log₁₀ reachable ceiling is
part of the observation, so every v2 seed was trained being told lungs were
unreachable on every volume. Measured correctly, the lung ceiling's median
over the 30 volumes is 0.0354, not 0.0015, and lungs are unreachable on 2
volumes rather than 26.

Consequence for the instruction distribution actually trained on (400 sampled
instructions over the 20 training volumes):

| | skeleton | lungs | soft | vessels |
|---|---|---|---|---|
| before | 238 | 45 | 258 | 27 |
| after | 204 | **194** | 202 | 24 |

## Why the old numbers cannot be reused

> **Correction, written after the run.** This section was wrong. The episode
> set did *not* change: B3, B4 and B5 score bit-identically (to four decimals,
> over 200 episodes) before and after the fixes, which deterministic search
> baselines could not do on different episodes. The gate removes nothing on the
> test split once the ceiling is measured correctly, so the sampled
> instructions coincide. The old and new figures are therefore directly
> comparable, and the decomposition of the published +0.194 in the README
> depends on that. The reasoning below stands as the precaution it was — it
> cost one extra evaluation arm and would have been necessary had the gate
> bitten — but its premise did not hold.

`rl.vis_eval.fixed_episodes` is deterministic given the code, but the sampler
changed, so the episode set it produces now is not the one behind the recorded
+0.194 / +0.218 / +0.660. Comparing a new figure against those would confound
the policy change with a change in the questions being asked.

**Both checkpoints are therefore scored on the same, current episode set.**
The v2 arm is "the checkpoint we would have shipped, run through the corrected
system" — not v2 as originally measured. That distinction goes in the write-up.

## Predictions

1. **Overall attainment improves**, but modestly — most instructions name
   skeleton or soft, which the broken ceiling did not affect.
2. **The improvement concentrates in lung instructions.** This is the sharp
   prediction: if the ceiling fix matters, it matters here. Lungs go from ~11%
   to ~48% of sampled instructions, and v2 was trained with a lung ceiling of
   ~0.
3. **Absolute-level instructions improve more than relative ones**, since
   "high lungs" was 0.8 × a ceiling of ~0 and is now 0.8 × a real ceiling.
4. **No regression on skeleton or soft.** If these get worse, the gate has
   narrowed the training distribution in a way that hurts, and that is a
   finding to report rather than tune away.
5. Search (`B4`, 200 evaluations) still beats the one-shot policy on median
   attainment. Nothing here closes that gap; the argument for a policy is cost
   per instruction, not peak quality.

Achievability check already run on the gated sampler: 200-evaluation search
reaches median attainment **0.840** over 24 instructions on 4 training
volumes, with none below 0.1. The objective is satisfiable, so a flat learning
curve would indicate a learning problem, not an impossible task.

## Protocol

- 3 seeds (0, 1, 2), 150k steps, `out/rl_v2/oneshot_v3_seed{0,1,2}`. The v2
  checkpoints are kept, not overwritten.
- Best checkpoint per seed by median validation attainment (existing rule).
- Evaluation: 200 episodes, `--split test`, `--formulation one_shot`, held-out
  patients, identical episodes for every arm: v2 seeds, v3 seeds, and
  baselines B0–B5.
- Report median attainment (robust to the unbounded-below outliers a mean is
  vulnerable to), share of instructions improved, per-kind and **per-class**
  breakdown, and paired Wilcoxon against the relevant baseline.
- Report all three seeds, not the best one.

## What would falsify the claim that the fixes helped

- No improvement on lung instructions specifically → the ceiling was not the
  binding constraint, and the observation's solo_max channel is doing less
  work than assumed. Worth reporting either way, since it bears directly on
  whether that tier-1 observation feature earns its place.
- Improvement spread evenly across all classes → suspect something other than
  the ceiling fix (e.g. the gate removing hard-but-possible items and
  flattering the average). Check by scoring v3 on ungated episodes.
- Overall attainment down → the gate narrowed training too far.

## Status

- [x] smoke run (10k steps) — plumbing check
- [x] 3 seeds trained (150k steps each, no errors; final validation medians
      +0.250, +0.246, +0.333)
- [x] policy arms scored on identical episodes (`tools/per_class_eval.py`)
- [ ] baselines + refinement arms (`rl.vis_eval`) — running
- [ ] README and `docs/rl-v2-pipeline.typ` tables updated from the re-run

## Result: the predictions mostly failed

200 held-out episodes, identical for every arm, policy alone (no refinement):

| | overall | lungs | skeleton | soft | vessels |
|---|---|---|---|---|---|
| v2 seed0 | +0.158 | +0.131 | +0.162 | +0.155 | +0.156 |
| v2 seed1 | +0.182 | +0.154 | +0.173 | +0.173 | +0.250 |
| v2 seed2 | +0.181 | +0.158 | +0.212 | +0.191 | +0.140 |
| v3 seed0 | +0.164 | +0.166 | +0.163 | +0.160 | +0.108 |
| v3 seed1 | +0.209 | +0.182 | +0.191 | +0.191 | +0.209 |
| v3 seed2 | +0.169 | +0.104 | +0.171 | +0.127 | +0.097 |

Episodes per class: lungs 85, skeleton 95, soft 111, vessels 17.

Paired Wilcoxon on the seed-averaged per-episode attainment:

| comparison | v2 | v3 | p |
|---|---|---|---|
| all episodes | +0.1732 | +0.1743 | **0.85** |
| seed 0 vs seed 0 | +0.158 | +0.164 | 0.93 |
| seed 1 vs seed 1 | +0.182 | +0.209 | 0.20 |
| seed 2 vs seed 2 | +0.181 | +0.169 | 0.73 |
| lung episodes only (n=85) | +0.1348 | +0.1598 | 0.081 |

**Prediction 1 (overall improves): not supported.** p = 0.85; the difference
is +0.001.

**Prediction 2 (improvement concentrates in lungs): not supported, though the
direction is right.** +0.025 median on lung episodes, p = 0.081 — a trend, not
a result. The spread between seeds within each arm (v3 lungs ranges
0.104–0.182) is larger than the gap between arms.

**Prediction 4 (no regression on skeleton or soft): holds.** Neither class
moved outside the seed-to-seed spread.

### What this means

Retraining with a corrected observation produced **no detectable change in
held-out attainment at n = 3**. The honest reading is that the log₁₀ reachable
ceiling — one of the two "tier 1" observation features — is doing less work
than assumed: a policy trained believing lungs were unreachable everywhere
performs the same as one trained with the true ceiling, once both are fed
correct observations at test time. That bears directly on whether that feature
earns its place, and is worth stating in the write-up rather than burying.

It does **not** mean the fixes were unnecessary. Attainment cannot see what
they were for:

- the colour fix made preference pairs blind (the policy's candidate was
  identifiable on sight, which would have invalidated every judgment);
- the reachability gate stopped raters being shown instructions no method can
  satisfy;
- the ceiling fix is why lungs can be validated against real renders at all.

None of those show up in an attainment number, and all of them determine
whether the collected preference data means anything.

### Caveat on power

n = 3 seeds. With this much seed-to-seed variance, only a large effect would
be detectable; "no difference" here means "no difference we could see", not
"no difference". More seeds would settle the lung trend (p = 0.081) one way or
the other, and that is a good use of a GPU cluster.

---

## Addendum, 2026-09-17: this experiment's measurement was stale

Every number above was produced by a batch job that started before the ceiling
fix in `62b5720` landed (22:48 on 2026-09-16) and wrote its result files at
03:25-03:26 the next morning. Python had already imported the pre-fix
`goals.py`/`visibility.py`, so the whole batch scored on the code it loaded, not
the code in the tree. The files' timestamps say otherwise, which is how this went
unnoticed for a day.

Confirmed by checking out `a0b009a` (the commit before the fix) into a worktree
and re-scoring the same v3 seed-0 checkpoint against it: median +0.142,
per-kind mean for absolute instructions -22.2 -- reproducing the stored
+0.164 / -23.5 almost exactly, while the current code gives +0.231 / +0.35 on the
identical episodes.

### What the conclusion becomes

Re-measured on the corrected pipeline, same checkpoints, same 200 held-out
episodes:

| | seed-averaged median | per-seed |
|---|---|---|
| v2 (pre-fix observation) | +0.2008 | 0.193 / 0.247 / 0.249 |
| v3 (corrected observation) | +0.2863 | 0.231 / 0.307 / 0.275 |

Paired Wilcoxon over seed-averaged per-episode attainment: **p = 0.0064**, v3
ahead on 59 % of episodes. The pre-registered prediction that retraining would
help was therefore right, and the null reported above was an artefact.

Per instruction kind, the gain lands where the mechanism predicts:

| kind | v2 | v3 | delta |
|---|---|---|---|
| absolute | +0.366 | +0.489 | +0.122 |
| brightness | +0.345 | +0.411 | +0.066 |
| relative | +0.185 | +0.239 | +0.054 |
| compound | +0.017 | +0.055 | +0.038 |
| show_only | +0.164 | +0.181 | +0.017 |

Absolute instructions are the ones whose targets are set from the reachable
ceiling, and they gain most from the channel that `62b5720` fixed. The reading
recorded above -- that the log10 reachable-ceiling channel "is doing less work in
the observation than assumed" -- is withdrawn. It does the work; the measurement
could not see it.

### Caveat that still stands

Three checkpoints per arm. The p-value is a paired test over 200 episodes, not
over training runs, so it speaks to consistency across instructions rather than
across seeds. The honest sentence is: on these seeds, the corrected observation
improves median attainment by about 0.086, concentrated in absolute instructions.

### What changed so this cannot recur

`provenance.py` records the git commit, whether the tree was dirty, and a
fingerprint of the *imported* scoring modules into every result file, captured at
import time rather than write time. `rl/vis_eval.py` prints a staleness banner
when a result's fingerprint no longer matches the code on disk.
