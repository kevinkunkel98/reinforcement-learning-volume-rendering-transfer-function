# `show_only` hindsight demonstrations — design

## Goal

Give the RL training loop continuous, in-training-time hindsight
demonstrations shaped like `show_only` instructions, mirroring the existing
`--hindsight-brightness` mechanism. `search-distilled-pretraining-design.md`
explicitly scoped "targeting specific weak instruction kinds (`show_only`,
compound)" out of its own work; this spec is that planned follow-up for
`show_only` specifically.

Motivation, from this session's E1-E3 ablation
(`docs/experiments/2026-10-07-search-distilled-pretraining-results.md`):
hindsight-only pretraining drove `attainment_brightness` from -1.278 (no
pretrain) to -5.204, because none of the 4000 hindsight demonstrations were
brightness-shaped and nothing during RL training ever anchored that
behaviour. Adding `--hindsight-brightness` (continuous brightness-shaped
hindsight demonstrations throughout RL training, not just a one-off pretrain
batch) fixed it outright: +0.246, median-of-3 +0.219. `show_only` is
independently documented (`docs/STATUS.md`) as a chronically weak
instruction kind across every era of this project, and -- unlike
`compound`, which the generic hindsight sampler already produces by
chance roughly half the time it mentions 2 classes -- `show_only`'s goal
vector is structurally unique: it is the only instruction kind that sets an
*explicit negative target on every unmentioned class*, a shape the generic
hindsight sampler (which only ever sets targets on the 1-2 classes it
picked) can never produce, however lucky the random noise gets. This closes
that specific, structural coverage gap.

Out of scope: `compound` (no evidence it has the same coverage-gap problem;
revisit only if data says otherwise). Changing `goals.distance`'s reward
formula itself -- the reward already scores `show_only` correctly
(`goals._sample_show_only`/`_goal_show_only`); the gap is purely that the
policy never sees a training-time demonstration of that shape.

## Context

`OneShotEnv._sample_hindsight_brightness` (`rl/oneshot_env.py:378-400`) is
the direct precedent: pick a class, nudge one controllable group with small
random noise, measure the *real* resulting change, use that measured change
as the ground-truth goal. It's gated by `hindsight_brightness_share`
(constructor param + `--hindsight-brightness` CLI flag), checked in
`_sample_hindsight_goal` before falling through to the generic
visibility-noise sampler.

`show_only`'s real-instruction shape (`goals._sample_show_only`,
`goals.py:422-434`): pick `n = min(randint(1, 3), len(classes))` "shown"
classes; every shown class gets `{"vis": HIDE_STRENGTH}` (positive), every
other canonical class gets `{"vis": -HIDE_STRENGTH}` (explicit negative) --
note this targets only the 8 canonical `GOAL_CLASSES`, not unclassified
("other") tissue; suppressing material that belongs to none of the eight
classes is handled by `goals.distance`'s keep penalty, not the goal vector,
for both real and hindsight `show_only` episodes alike, so this design
needs no special handling for "other".

A small noise nudge (brightness's approach) cannot reach a `show_only`-like
state: showing 1-2 classes while suppressing every other one is a large,
deliberate change across many controllable groups at once, not a one-group
nudge. The oracle action here has to be *constructed* directly, not found
by chance.

## Approach

**New method `OneShotEnv._sample_hindsight_show_only(self, rng,
start_params, start_agg)`**, same return shape and placement convention as
`_sample_hindsight_brightness` (defined right after it):

1. Pick `n = int(rng.integers(1, 3))` shown classes (mirrors
   `_sample_show_only`'s own distribution) via `rng.choice(GOAL_CLASSES,
   size=n, replace=False)`.
2. Build one action from `start_controllable`: for *every* canonical
   class's *height* group (`CONTROLLABLE.index((base + 2,))` where `base =
   goals.PEAK_INDEX[goal_class] * transfer.PARAMS_PER_PEAK` -- the same
   indexing brightness already uses for its own group, just the height
   index instead of the colour one), push it **relative to its own current
   value** by a randomized amount -- up for shown classes, down for every
   other class (`np.clip(start_controllable[group] +/- rng.uniform(*
   HINDSIGHT_SHOW_ONLY_PUSH_RANGE), -1.0, 1.0)`) -- mirroring
   `_sample_hindsight_brightness`'s own relative-nudge pattern, not an
   absolute snap to a fixed region. This matters: an *absolute* "low"
   region (e.g. real height 0-0.35) is not safely below every class's
   actual starting height -- `transfer.ANATOMICAL_HEIGHTS` ranges from
   0.05 (lungs) to 0.60 (skeleton), so a fixed absolute low region sits
   *above* most classes' own baseline, which would push a "hidden" class
   brighter instead of dimmer. A relative push is directionally correct
   regardless of where each class's height actually starts. Randomized
   magnitude, not a fixed snap to the literal extremes, for the same
   reason as before -- real anatomy overlaps, so one exact knob position
   would teach a narrower pattern than the task has.
3. Apply the action, measure the real resulting `vis` for *every* canonical
   class via `goals.aggregate(self._features(apply_controllable(start_params,
   action)), self._active_layers)` (mirrors brightness's `target_agg`
   computation, just reading `vis` for all 8 classes instead of `bright`
   for one).
4. `targets = {c: {"vis": delta_c} for c in GOAL_CLASSES}`, same
   `log10(target + eps) - log10(start + eps)` convention already used
   throughout the file, for every class -- this explicit-target-on-every-
   class shape is the entire point.
5. Retry up to `HINDSIGHT_MAX_TRIES` (the existing shared constant) unless
   **both** sides clear a minimum delta: every shown class's measured
   change `>= HINDSIGHT_SHOW_ONLY_MIN_DELTA`, *and* every hidden class's
   measured change `<= -HINDSIGHT_SHOW_ONLY_MIN_DELTA` (same role as
   `HINDSIGHT_BRIGHTNESS_MIN_DELTA`, checked on both sides, not just the
   shown one -- checking only the shown side was the bug the relative-push
   fix above closes, but validating the actual measured outcome on both
   sides is what actually guarantees it, the same "trust the measurement,
   not the construction" principle the whole hindsight system already
   runs on). Guards against picking a class with ~0 reachable ceiling (or
   an action that clipped to a no-op) on this particular volume, which
   would produce a degenerate, incorrectly-signed training example.
6. Return `None` if no try clears the threshold (same failure contract as
   `_sample_hindsight_brightness`); the caller already handles `None` by
   falling through.

**Wiring**, identical pattern to brightness:
- `hindsight_show_only_share: float = 0.0` constructor param on
  `OneShotEnv` (validated range `[0, 1]`, same as
  `hindsight_brightness_share`).
- One more independent branch in `_sample_hindsight_goal`
  (`rl/oneshot_env.py:402-407`), checked alongside (not instead of) the
  existing brightness branch -- each its own independent coin-flip against
  the same episode, falling through to the generic sampler if neither hits
  or both return `None`. Order (brightness checked before show-only, or
  vice versa) has no effect when both shares are small/typical; brightness
  stays first for a minimal diff.
- `rl/oneshot_train.py`: new `--hindsight-show-only` CLI flag, threaded
  into `OneShotEnv` construction and into the pretrain-collection contract
  dict exactly where `--hindsight-brightness` already is
  (`load_or_collect_pretrain`'s `expected`/`contract` dicts in
  `rl/oneshot_train.py`), including the same `stored.setdefault(...,  0.0)`
  backward-compatibility default so datasets collected before this change
  still load.

## Cost

Pure in-process env computation (one extra `apply_controllable` +
`features`/`aggregate` call per sampled episode, same order of cost as the
existing brightness path) -- no new `hill_climb` search, no new
multiprocessing collection step. Negligible throughput impact, unlike
search-distilled pretraining's ~2.7s/episode collection cost.

## Known limitation this design does not claim to fix

This closes the *hindsight-coverage* gap for `show_only` -- it does not by
itself guarantee the specific failure mode `docs/STATUS.md` documents
(`show_only` winning by suppressing everything, including hiding behind
unclassified "other" tissue, rather than actually raising the named class)
is resolved. That failure mode is a property of the reward's keep-penalty
weighting (`LAMBDA_KEEP`/`KEEP_TOLERANCE`) interacting with what the policy
has learned, which this hindsight signal should help but can't be assumed
to resolve from design alone. Validate empirically with an E-series
ablation (mirroring this session's E1-E3 pattern: with vs. without
`--hindsight-show-only`, per-kind `attainment_show_only` breakdown, 3
seeds) before claiming it fixed, the same way brightness's fix was
validated rather than assumed.

## Testing

Use the `"fake_a"` + `model_for_volume=lambda name: _StubModel()` fixture
(already used by ~20 other tests in this file), not `"synthetic"` --
`"synthetic"` uses `visibility.py`'s intensity-only fallback, under which
`heart`/`vessels`/`liver`/`kidneys`/`spleen` read exactly 0.0 visibility
regardless of any transfer-function parameter (the fallback cannot identify
them), so a test asserting every one of the 8 classes gets a meaningful
explicit target can never pass against it. Production training never hits
this: `datasets.volumes_for_split` only returns real TotalSegmentator
subjects with full labels.

Mirror the existing brightness hindsight tests in `tests/test_oneshot_env.py`:

- **Structural**: every canonical class has an explicit target in the
  sampled goal; shown classes' targets are positive, every other class's
  is negative (the property that makes this shape distinct from every
  other hindsight kind).
- **Behavioural**: applying the sampled hindsight action to a real volume's
  start state actually raises the shown class(es)' measured visibility and
  lowers every other class's, mirroring
  `test_hindsight_goal_is_exactly_what_the_target_achieved`'s pattern of
  checking the *real* measured outcome, not just the constructed intent.
- **Share-gating**: `hindsight_show_only_share=0.0` (the default) never
  produces a `show_only`-shaped hindsight goal; a share of `1.0` always
  does (when it doesn't return `None`), mirroring the existing brightness
  share tests.
- Full project suite green before commit, same bar as every other task
  this session.
