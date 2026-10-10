# Show-Only Group-Balanced Reward: Design

## Purpose

A diagnostic on the current best policy (`out/rl_v4/e3_v9_searchmix_bright_150k_seed0/best.zip`,
346 held-out `show_only` episodes) found a clear, structural asymmetry:

| group | n targets | median per-class attainment |
|---|---|---|
| shown (elevate) | 485 | **-0.005** |
| hidden (suppress) | 1486 | **0.119** |

The policy makes essentially no net progress elevating the 1-2 classes a
`show_only` instruction names as "shown"; it only succeeds at suppressing the
6-7 "hidden" ones. `goals.distance()` weights every targeted class equally
(weight 1.0 each), so a `show_only` goal's hidden group (6-7 classes) carries
3-7x the total weight of its shown group (1-2 classes) purely from having
more terms -- the optimizer gets more total reward from suppressing the many
than from elevating the few, independent of per-class difficulty. This is a
structural reward-shaping problem, not a sampling or capacity issue, and it
was found to not be fixable by the previously-tried hindsight-demonstration
approach (`docs/experiments/2026-10-10-show-only-hindsight-results.md`:
`show_only` attainment improved only modestly and inconsistently, and cost a
consistent `compound` regression).

This spec fixes the asymmetry directly in the objective: `show_only`'s
shown-class group and hidden-class group get **equal total weight** (1.0
each) regardless of how many classes are in each, rather than one class
automatically outweighing many just by count.

## Non-goals

- Not changing the weighting of any other instruction kind (`relative`,
  `compound`, `absolute`, `brightness`). Their goals always have weight 1.0
  on every mentioned class today (1-2 classes), so a correctly-scoped fix
  must leave them bit-for-bit unchanged.
- Not reverting or removing `--hindsight-show-only` (stays at default 0.0,
  unused for now, per the prior experiment's conclusion).
- Not changing the observation / policy input shape. `instruction["goal"]`
  (`goals.goal_vector`'s 32-value output) is fed directly into
  `OneShotEnv._build_observation` and `VisibilityTFEnv`'s observation --
  changing its shape or the meaning of an existing slot would touch every
  policy checkpoint's input contract. This design adds a wholly separate,
  non-observed field instead.

## Mechanism

### 1. `goals.distance`/`attainment`/`scoring` gain an optional `weights` parameter

```python
def distance(goal: np.ndarray, start: dict, current: dict, weights: dict = None) -> float:
```

`weights` maps `goal_class -> float` multiplier, default `None` (every class
weight 1.0 -- today's exact behavior). When provided, the two mentioned-class
terms (`m[i] * abs(c - d)` and `n[i] * abs(b - e)`) are each additionally
scaled by `weights.get(goal_class, 1.0)`. The keep-tolerance terms for
genuinely unmentioned classes (the `(1 - m[i])`/`(1 - n[i])` branches) are
untouched -- `weights` only ever applies where a class already has `m[i]` or
`n[i]` set to 1, so it cannot be confused with "unmentioned."

`attainment(goal, start, final, active_layers=None, weights=None)` and
`scoring(goal, start_features, final_features, active_layers=None,
weights=None)` both gain the same passthrough parameter and forward it to
`distance`. No existing call site that omits `weights` changes behavior.

### 2. `goals.show_only_weights(shown, classes)` -- the shared formula

```python
def show_only_weights(shown: set, classes) -> dict:
    """Equal total weight for the shown group and the hidden group,
    regardless of how many classes are in each -- 1/n_shown per shown class,
    1/n_hidden per hidden class."""
    hidden = [c for c in classes if c not in shown]
    weights = {c: 1.0 / len(shown) for c in shown}
    weights.update({c: 1.0 / len(hidden) for c in hidden})
    return weights
```

`classes` is the full class list the instruction was built against (`shown |
hidden`, exactly `targets.keys()` for a `show_only` instruction). Both groups
are always non-empty for a `show_only` instruction (1-2 shown out of 8 total,
so hidden is 6-7) -- no division-by-zero case exists given how
`_sample_show_only`/`_goal_show_only` construct `shown`.

### 3. Every `show_only` construction site attaches `weights` to the instruction dict

Three places build a `show_only`-shaped `targets` dict today; all three call
`show_only_weights` and attach the result as a new `"weights"` key on the
returned instruction dict, alongside the existing `"kind"`, `"text"`,
`"targets"`, `"goal"`:

- `goals._sample_show_only` (random training instructions) -- called from
  `goals.sample_instruction`.
- `goals._goal_show_only` (parsed real user commands, e.g. the production
  server path in `server.py`) -- called from `goals._goal_from_command`.
- `OneShotEnv._sample_hindsight_show_only` (hindsight demonstrations) --
  already computes `shown`/`hidden` sets internally; adds the same `weights`
  key to its returned instruction dict.

Every other instruction-construction function (`_sample_relative`,
`_sample_compound`, `_sample_absolute`, `_sample_brightness`, and their
`_goal_*`/hindsight counterparts) is unchanged and never sets `"weights"` --
their instruction dicts simply lack the key, and every scoring call site
treats a missing key the same as `weights=None` (via `instruction.get
("weights")`).

### 4. Every scoring call site threads `instruction.get("weights")` through

So the RL reward, the eval attainment metric, the search baseline's
objective, and the production server's scoring all stay consistent with each
other (a repeat of the hindsight feature's own principle: the search teacher
used for distillation must optimize the same objective the policy is
trained and judged on). Each change is the same one-line pattern -- add
`weights=instruction.get("weights")` to an existing call -- at:

| file | call sites |
|---|---|
| `rl/oneshot_env.py` | `OneShotEnv._attainment` (line ~358); `OneShotEnv.step`'s `target_progress` branch, both `start_distance` and `final_distance` (lines ~565-566); the `reset()` `start_distance` call (line ~530) |
| `rl/baselines.py` | `_evaluate` (used by `hill_climb`'s search loop); `hill_climb`'s own `best_distance` initialization |
| `rl/vis_eval.py` | `_frozen_episode_env`/`_frozen_one_shot_env`'s `start_distance` calls; `_attainment_or_zero` |
| `rl/vis_env.py` | `VisibilityTFEnv._attainment`; `reset()`'s `start_distance`; `step()`'s `distance` call |
| `rl/candidates.py` | the `a_distance`/`b_distance` objective-choice comparison |
| `server.py` | the live-inference `"attainment"` field; the sweep/compare endpoint's `_score` closure |

All of these already have `instruction` in scope, so most sites' change is
simply `instruction["goal"]` -> `instruction["goal"],
weights=instruction.get("weights")`. `rl/baselines.py` is the one exception:
`hill_climb` already receives the full `instruction` and extracts
`weights = instruction.get("weights")` once at the top (mirroring how it
already extracts `goal = instruction["goal"]`), then passes `weights=weights`
on its own `best_distance` call and on each `_evaluate` call in its search
loop; `_evaluate` itself only ever receives the bare `goal` vector today (not
the full `instruction`), so it gains a new `weights=None` parameter forwarded
to `goals.distance`.

### 5. Always on, not a flag

Unlike `--hindsight-show-only`, this changes how `show_only` is *scored*
system-wide -- there is no meaningful "old objective" to preserve behind a
flag; the old unweighted scoring was the bug. Once implemented, every
`show_only` instruction (sampled, hindsight, or a real parsed command) is
scored with the group-balanced weights unconditionally.

## Data flow example

A `show_only` instruction naming 1 shown class out of 8:
`shown = {"skeleton"}`, `hidden = {"lungs", "soft", "vessels", ...}` (7
classes). `show_only_weights` gives `{"skeleton": 1.0, "lungs": 1/7, "soft":
1/7, ...}`. In `distance()`, the skeleton term contributes
`1.0 * |c_skeleton - d_skeleton|` exactly as before (group of 1, weight
unchanged), while each hidden-class term contributes `(1/7) * |c_i - d_i|`
instead of `1.0 * |c_i - d_i|` -- so the 7 hidden terms' combined weight
drops from 7.0 to 1.0, matching the shown group's 1.0, instead of
outweighing it 7:1.

## Testing strategy

- `tests/test_goals.py`: new tests for `distance`/`attainment`/`scoring`
  with an explicit `weights` dict (confirm a weighted term's contribution
  scales exactly as expected, confirm omitting `weights` reproduces today's
  existing passing tests bit-for-bit, confirm an unmentioned class's
  keep-tolerance term is unaffected by `weights` even when a `weights` entry
  happens to exist for that class).
- `tests/test_goals.py`: new tests for `show_only_weights` directly (1
  shown/7 hidden -> `{shown: 1.0, each hidden: 1/7}`; 2 shown/6 hidden ->
  `{each shown: 0.5, each hidden: 1/6}`; weights sum per group to exactly
  1.0).
- `tests/test_goals.py`: extend the existing `test_show_only_hides_the_other_
  classes` / `test_goal_from_command_show_only_hides_the_rest` style tests to
  also assert the returned instruction's `"weights"` key matches
  `show_only_weights(shown, targets.keys())`.
- `tests/test_oneshot_env.py`: extend the existing
  `test_hindsight_show_only_targets_every_class_with_correct_signs`-style
  tests to assert `_sample_hindsight_show_only`'s returned instruction also
  carries the correct `"weights"` key.
- `tests/test_oneshot_env.py`: a new test confirming `OneShotEnv.step`'s
  reward for a `show_only` instruction actually reflects the new weighting
  (construct a stub/fake model where the shown-class error and a single
  hidden-class error are both nonzero and of a known magnitude; assert the
  reward matches hand-computed weighted `target_progress`, not the old
  unweighted value).
- `tests/test_baselines.py`: confirm `hill_climb`/`_evaluate` accept and
  apply `weights` the same way, and that a `show_only` instruction's search
  now treats the shown class's error as co-equal to the hidden group's, not
  swamped by it.
- Run the full existing suite after each change -- every test that doesn't
  pass `weights` must still pass unchanged (the core backward-compatibility
  guarantee of this design).

## Validation plan (after implementation)

Because this redefines what `attainment_show_only` *means*, a direct
before/after comparison needs two steps, not one:

1. **Rescore, don't retrain, first.** Run `rl.vis_eval` against the existing
   E3 checkpoints (seeds 0-2) under the *new* weighted metric, with no
   retraining. This isolates how much of any change is just the metric
   being redefined (an already-good or already-bad transfer function might
   score differently under the new weights even though the policy itself
   didn't change).
2. **Retrain.** Train a fresh 3-seed run (E5) with the exact E3 recipe
   (search-distilled pretrain + continuous brightness-hindsight anchoring,
   `--hindsight-show-only` left at its default 0.0 per the prior
   experiment's conclusion) -- the only difference from E3 is that `goals.
   distance` now applies `show_only_weights` wherever `show_only`
   instructions are scored, including during RL training itself.
3. **Compare three numbers per seed:** E3-old-checkpoint-under-old-metric
   (already have this), E3-old-checkpoint-under-new-metric (step 1), and
   E5-new-checkpoint-under-new-metric (step 2) -- so "did rescoring alone
   move the number" and "did retraining under the new reward actually change
   behavior" are kept separate, not conflated.
4. Re-run the shown-vs-hidden diagnostic (the same per-class attainment
   breakdown this spec's Purpose section used) on the E5 checkpoints, to
   directly check whether the shown-class median attainment moved off its
   current ~0, not just whether the aggregate `attainment_show_only` number
   improved.
5. Apply this project's own 3-seed decision gate (median attainment > 0,
   share improved > 50% across all 3 seeds) before calling this adopted,
   same discipline as every other experiment in `docs/experiments/`.

This validation work itself is out of scope for the implementation plan this
spec feeds into -- it happens afterward, the same way E3's and E4's 3-seed
confirmations followed their own implementation work.
