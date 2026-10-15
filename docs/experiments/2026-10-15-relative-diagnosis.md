# Why `relative` scores near zero — diagnosis (2026-10-15)

**Question:** `relative` (40 % of instructions) has been negative in every
8-class experiment (E1–E5, e.g. E3 −0.068, E5 −0.145), while the 4-class v4
policy scored +0.354 on it. Why?

**Short answer:** mostly the task, not the learner. **55 % of `relative`
instructions cannot be met by any method from their start state**, because
the named class is practically invisible there (vis < `goals.EPSILON` =
1e-3). On those, every policy is a coin flip whose drift makes the *mean*
negative. On the solvable 45 % the policy is clearly positive, but well
below search — that gap is the real learning problem.

Reproduce: `.venv/bin/python -m tools.relative_diagnosis out/rl_v4/e3_v9_searchmix_bright_150k_seed{0,1,2}/best.zip out/rl_v4/e5_v9_searchmix_bright_150k_seed0/best.zip --episodes 200 --hill 200`
(validation split, fresh seeds from 50 000, no hindsight episodes).

## Evidence (200 `relative` episodes, validation split)

| arm | mean | median | > 0 |
|---|---|---|---|
| E3 seed 0 / 1 / 2 | 0.02 / 0.01 / −0.02 | 0.07 / 0.03 / 0.03 | 64 / 63 / 58 % |
| E5 seed 0 | −0.02 | 0.01 | 56 % |
| `hill_climb` (200 evaluations) | 0.45 | 0.39 | 96 % |

Split by whether the target is reachable at all from the start state
(`tools.relative_diagnosis.solvable`):

| | solvable (n = 90) | unsolvable (n = 110) |
|---|---|---|
| E3 seeds, median | 0.23–0.32 (70–79 % > 0) | 0.00 (~50 % > 0) |
| E3 seeds, mean | 0.09–0.17 | −0.07 … −0.11 |
| E5, mean | 0.13 | −0.15 |
| `hill_climb`, median | 0.80 | 0.06 |

**Why the unsolvable ones are unsolvable.** The goal is a change of
`log10(vis + 1e-3)`. At the start transfer function most organs sit behind
soft tissue: median start vis kidneys 0.0000, lungs 0.0001, spleen 0.0004,
heart 0.0010, liver 0.0011 (skeleton 0.0035).
- *Decrease*: vis cannot go below 0, so the change is bounded by
  `log10(1e-3) − log10(vis₀ + 1e-3)`. **61 % of decreases ask for more than
  that** (median best-possible attainment 0.12 even with zero drift).
- *Increase* of a hidden class needs its occluders removed, which the keep
  terms charge as drift. Even 200-evaluation search reaches a median of 0.06.

**Why v4 was fine.** Its 4–5 classes were mostly visible; 5 of today's 8
classes are occluded organs. The v4-like classes still do well today
(skeleton median 0.45, vessels 0.44 for E3 seed 0).

**Why "E5 made relative worse" is mostly noise.** On the solvable half E5
is inside the E3 range (mean 0.13 vs 0.09–0.17); the difference is in the
unsolvable half. The training-time figures were also a **mean over ~12
episodes** (40 validation episodes, 25 % of them hindsight and not counted,
40 % of the rest `relative`).

## Secondary findings

1. **The policy undershoots.** It moves the right way in 90 % of episodes but
   achieves only 16–43 % (median by strength/direction) of the requested
   change; 71–76 % of the start distance remains as unmet goal, drift adds
   ~26 %. Search leaves 50 % unmet and almost no drift.
2. **Training reward and evaluation metric charge drift differently.** The
   `target` reward (`OneShotEnv.step`) sums |Δ| over unmentioned classes
   *without* the 0.15 tolerance (weight 0.1) and excludes the goal class's
   brightness; `goals.distance` applies the tolerance (weight 0.3) and
   counts the goal class's brightness as drift. A plausible cause of the
   timidity — not tested yet.
3. Per-kind evaluation numbers are means; with attainment unbounded below and
   a start distance of 0.15–0.6 for `relative`, a few episodes dominate them.

## Options (not implemented — they change the task definition)

1. **Only sample solvable `relative` instructions** (decrease: target above
   the log floor; increase: start vis ≥ 1e-3), re-drawing the class
   otherwise. Arguably what a user would mean too: "a bit less kidney" when no
   kidney is visible is not a meaningful request. Needs retraining; the
   numbers stop being comparable to E1–E5 (report both).
2. Treat an invisible class as an **absolute** request ("show X") instead.
3. **Report** `relative` split into solvable / unsolvable, as medians, on more
   than 40 validation episodes.
4. Then attack the learning gap on the solvable half (finding 1), starting
   with the reward/metric drift mismatch (finding 2).
