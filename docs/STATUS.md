# Status — 2026-09-20

MVP due in 2 days. The results have not moved since the 18th and do not need
to. What changed this weekend is that the prototype became something you can
*operate* rather than describe: one instruction, answered four ways at once,
with the cost of each answer on screen.

## Where things stand

| | state |
|---|---|
| Measurement (`visibility.py`) | validated against real renders, four of five classes |
| Policy | 3 seeds on the corrected pipeline (`oneshot_v3_seed{0,1,2}`) |
| Held-out evaluation | 200 episodes, 3 seeds, baselines B0–B5, provenance recorded |
| Reproducibility | result files carry per-episode rows; `tools.compare_runs` recomputes the v2/v3 comparison from them |
| Viewer | four-mode comparison panel and a 20-instruction sweep |
| Language → goal → policy | end to end, LLM parser default |
| Figures | frontier, reliability, per-kind curves, qualitative, SAC diagnostics |
| **Clean preference judgments** | **0** — the 54 collected are quarantined as a pilot |
| Tests | 633 passing |

## Results (re-measured 2026-09-17, re-derived 2026-09-19)

200 instructions, six unseen patients. Attainment is the median of the three
seeds' medians; "improved" is their mean. Baseline rows do not vary by seed —
the episode seed is fixed at 0, so every arm is scored on identical episodes
and only the policy differs between the three files.

| Method | Evaluations | Median attainment | Improved |
|---|---|---|---|
| hill-climb (thorough) | 200 | +0.730 | 100 % |
| **policy + 3 refinements** | **4** | **+0.316** | 76 % |
| **policy alone** | **0** | **+0.275** | 73 % |
| hill-climb (cheap) | 10 | +0.263 | 93 % |
| do nothing | 0 | 0.000 | — |
| rule-based executor | 0 | −0.022 | 37 % |
| random | 0 | −0.040 | 39 % |
| occlusion heuristic | 0 | −0.191 | 30 % |

**The claim:** the policy answers with no evaluations at the level of a
hill-climber allowed ten, and beats every non-search baseline at p < 0.001.
Search remains more reliable (93 % vs 73 %). Per seed the paired test against
cheap search splits — seed 0 favours search (p = 3.0e-04), seeds 1 and 2 favour
the policy (p = 0.054, p = 0.029) — so "matches" is defensible and "beats" is
not.

Per instruction kind (median, seed 1): absolute +0.50, brightness +0.48,
relative +0.30, show-only +0.20, **compound +0.09**. Compound is 25 % of the
sampled mix (43 of the 200 test episodes) and the policy essentially fails it.

On 2026-09-19 all six result files were re-run and reproduced their stored
summaries exactly, to six decimal places, now carrying per-episode rows.
`tools.compare_runs` recomputes the v2-against-v3 comparison from those files:
+0.2008 against +0.2863 seed-averaged, p = 0.0064, v3 ahead on 59 % of
episodes, the gain concentrated in absolute instructions (+0.1223).

## The viewer, as of this weekend

**compare** answers one instruction four ways from the same start state —
applied directly, hill-climbed at 10 evaluations (B3) and 200 (B4), and by the
policy — reporting each arm's attainment, evaluation count, wall clock and
per-class effect beside its render. The search arms call
`rl.baselines.hill_climb` and attainment is `goals.attainment`, so the panel
computes the same quantities as the table above rather than a lookalike. It
plots whichever channel the instruction names: a brightness goal barely moves
visibility, so plotting visibility for one would show four near-identical bars
beside attainments ranging from +0.00 to +0.94.

**sweep 20** runs twenty instructions sampled from the same grammar and reports
each method's median attainment and how often it improved on doing nothing. One
comparison is a single draw; the claim above is a median, and the reliability
gap means roughly one instruction in four has the policy not improving.

Measured on `ts_s0477`, a held-out test subject, reset to the default transfer
function before every measurement:

| instruction | exact | search·10 | search·200 | policy |
|---|---|---|---|---|
| more bone | −0.001 | +0.348 | +0.582 | **+0.627** |
| brighten the skeleton | +0.025 | +0.000 | +0.941 | **+0.819** |
| show only the lungs | **+0.680** | +0.430 | +0.442 | +0.254 |

| sweep seed | exact | search·10 | policy |
|---|---|---|---|
| 0 | −0.209 (37 %) | +0.342 (100 %) | **+0.425 (85 %)** |
| 1 | −0.263 (22 %) | +0.387 (100 %) | **+0.405 (95 %)** |
| 2 | +0.041 (60 %) | +0.384 (100 %) | **+0.340 (85 %)** |

Exact and policy spend 0 evaluations and answer in 0 ms; search·10 takes
~140 ms, search·200 ~2.7 s.

## Demo notes

- **Press Reset before measuring anything.** Both panels start from wherever
  the session is, so comparing an instruction you have already applied measures
  from its own result — a different question, with worse-looking numbers.
- **Drive brightness instructions.** The policy beats cheap search on 8 of 9 of
  them, and cheap search often scores exactly +0.000 there, because ten
  evaluations is less than half of one 24-evaluation coordinate sweep.
- **Avoid "a bit less soft tissue"** (policy −2.8 on two subjects) and compound
  instructions (median −0.25 in the sweep).
- **Four of six test subjects have no lungs.** Lung instructions are correctly
  refused there; that is not a bug.

## Defects found and fixed this weekend

- The viewer's **search mode was not the search the thesis measures** — a
  bespoke coordinate stepper over `evaluate.objective` rather than
  `rl.baselines.hill_climb` — and it only engaged for opacity commands, so
  "show only the lungs" with search on silently did nothing.
- The **policy fallback died on a missing visibility cache** instead of
  degrading to exact application.
- Every **`ts_session` test ran against the intensity-only visibility model**,
  because the fixture patched `totalseg.subject` but not the `_subjects`
  chokepoint that `has_labels` reads.
- **Visibility models were rebuilt on every call**, discarding `solo_max`'s
  memo and repaying ~190 ms each time.
- The **comparison panel timed the render, not the arm**, which made the first
  press of a session report the policy as slower than cheap search.

## Cheapest improvements, in order

1. **Fix compound instructions** — a quarter of the mix at +0.09, the clearest
   capability gap and the one a user notices first.
2. **Train longer.** Worth trying, but not the evidence-backed first move: the
   per-kind curves plateau rather than run out of steps. ~3 h per seed.
3. **Distil search into the policy** — supervised pretraining on hill-climb
   solutions before RL. The standard way to close an amortisation gap.
4. **Make `visibility.py` differentiable.** The index cube is constant with
   respect to the transfer function; only `transfer_tables` needs porting to
   torch. Gives an analytic ceiling, a teacher for distillation, and an honest
   new baseline.

## Known issues

- **The "policy, no ceiling input" ablation** was never validly run: the stored
  result file is byte-identical to its own control, and no ablation switch has
  ever existed in the code. Withdrawn rather than repaired.
- **The viewer ships the seed-0 checkpoint**, the weakest of the three on
  held-out data (+0.231 against +0.307 and +0.275). Fixed before the held-out
  numbers were known, and not revisited.
- **`evaluations` in the comparison panel is the budget granted**, not what
  search spent — `hill_climb` can stop early, so B4 reports 200 where it spends
  about 193.
- **The per-class analysis of which anatomy benefits from retraining** was
  measured in the stale batch and has not been re-run; per-episode rows record
  instruction kind and volume, not goal class.
- **Anchor-pool and image caches** under `out/cache/` are serialisation
  contracts across raters. Do not change their key formats without invalidating
  them deliberately.
