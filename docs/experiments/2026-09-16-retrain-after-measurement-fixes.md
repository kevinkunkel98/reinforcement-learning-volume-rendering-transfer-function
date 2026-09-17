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

- [ ] smoke run (10k steps) — plumbing check
- [ ] 3 seeds trained
- [ ] all arms evaluated on identical episodes
- [ ] README and `docs/rl-v2-pipeline.typ` tables updated from the re-run
