# Visibility `features()` throughput — design

## Goal

Raise RL training throughput (resets/steps per second) by removing redundant
computation inside `visibility.VisibilityModel.features()` and the hindsight
goal-sampling loop in `rl/oneshot_env.py`, with **zero change to any numeric
result** on the paths that don't already use active layers — this is a pure
performance refactor, not a behavior change. The one path that trades exact
per-trial reproducibility for speed (the hindsight retry loop, see Approach)
only changes RNG draw *order*, not the acceptance criterion or the resulting
goal distribution, and only under `--hindsight-ratio > 0`.

Out of scope for this version: vectorizing/batching the hindsight retry loop
across trials (rejected below), moving any computation to MPS/GPU (the grid
is small enough that CPU dispatch, not FLOPs, dominates — see Measurements),
touching `transfer.label_scoped_tables` or any other consumer of
`visibility.py` outside `rl/`.

## Measurements (before)

Profiled on `ts_s0357` (`.venv/bin/python`, `cProfile`, 30 resets after a
warm-up reset, M1 Pro):

| config                 | resets/s | ms/reset |
|------------------------|---------:|---------:|
| `hindsight_ratio=0.0`  |     16.4 |     61.1 |
| `hindsight_ratio=1.0`  |      9.6 |    103.6 |

`cProfile` on the `hindsight_ratio=1.0` case: `visibility.py:features()`
accounts for 3.09s of 3.14s total (60 calls over 30 resets — i.e. the
hindsight retry loop needs on average only ~1 extra call on this volume, not
many). Inside `features()`, `VisibilityModel._weights()` is called twice per
`features()` call (120 calls for 60 `features()` calls): once for the "plain"
weights, once for "effective" (layer-masked) weights — identical work when
`layers is None` (true for every non-v8 policy, which includes the current
v9 training). The costliest ops are `torch.where` (0.61s / 1500 calls),
`pow` (0.53s / 120 calls — the `(1-alpha)**step_mm` term), `sum` (0.37s /
1766 calls) and `cumprod` (0.16s / 120 calls) — all paid twice per
`features()` call for work that, outside v8, produces the same result both
times.

## Approach

**A/A+ — stop computing "effective" weights twice.** Split
`VisibilityModel._weights` into three pieces:

- `_alpha_luminance(params, transfer_mode)` — the expensive, layer-independent
  chain (LUT build, index lookup, the `pow` term).
- `_layer_mask(layers)` — the 8-class `torch.where` loop that builds the
  layer-opacity mask (only runs when layers are given).
- `_composite(alpha)` — the `cumprod`/`before * alpha` chain.

`features()` calls `_alpha_luminance` **once**. If `layers is None`, the
"effective" vis/bright/coverage are set equal to the already-computed plain
values — no second composite, no mask loop. If layers are given (v8), only
`_composite` and `_layer_mask` run a second time; `_alpha_luminance` (the
`pow`-bearing part) is never duplicated, v8 included. `_weights()` stays as a
thin wrapper over the same three pieces, so `per_view_visibility` (its only
other caller) is unaffected.

This is exact, not approximate: for `layers=None` the old code's second
`_weights` call used `normalize_layers({})` — an all-opacity-1.0, no-op
mask — so its "effective" output was already mathematically equal to the
plain pass; the new code just reuses the one computed value instead of
running the same math again. (If anything this removes a theoretical source
of tiny divergence: two separate calls into the same torch reduction ops
are not guaranteed bit-identical across threads, so today's two calls could
already disagree in the last bit or two — the new code can't, since there's
only one value.) For `layers != None` (v8), output values are unchanged
because the same `_alpha_luminance`/`_layer_mask`/`_composite` math runs,
just without recomputing `_alpha_luminance` a second time.

**B — a `vis`-only path for the hindsight visibility search loop.** Add
`VisibilityModel.vis_only(params, layers=None, transfer_mode=...) -> dict`
(per-class vis, keyed like `features()["vis"]`, `"other"` included): skips
the luminance-table lookup, the per-class `bright` division and `coverage`
entirely — it calls `_alpha_luminance`/`_layer_mask`/`_composite` the same
way `features()` does, just never touches luminance. Factor the
goal-class-aggregation loop inside `goals.aggregate` into a shared
`goals._aggregate_vis(vis: dict) -> dict` (handles the `soft`/legacy
`organs`+`muscle` fallback once), used by both `goals.aggregate` and a new
`goals.aggregate_vis(vis: dict) -> dict` that wraps `vis_only`'s output the
same way. `OneShotEnv` gets `self._vis(params)` (mirrors `self._features`),
and `_sample_hindsight_goal`'s retry loop uses it instead of
`self._features(...)` + `goals.aggregate(...)` — it only ever read
`target_agg["vis"][goal_class]`, never `bright`/`coverage`.
`_sample_hindsight_brightness` is **not** changed: it needs per-class
brightness, so it keeps using `self._features` (and still benefits from
A/A+ automatically, since that's inside `features()` itself).

**Rejected: C — batch the retry loop's trials into one vectorized call.**
Would require giving `transfer.label_scoped_tables` (and the LUT gather in
`_alpha_luminance`) a batch dimension — a bigger, riskier change to a
function also used by the non-RL rendering path (`render.py`/`server.py`).
The measurement above shows the retry loop on this volume/instruction mix
needs on average ~1 extra `features()`-equivalent call, not many — so there
are few trials to batch, and always evaluating a fixed batch size (e.g. 20)
would cost *more* than today's early-exit-after-~1 loop in the common case.
Revisit only if post-A/A+/B profiling on other volumes/instruction kinds (or
with `--hindsight-brightness > 0`) shows the loop actually retrying many
times on average — with real numbers, not a guess.

## Testing

- `tests/test_visibility.py`: add a case asserting `features(params, layers=None)`'s
  `effective_vis`/`effective_bright`/`effective_coverage` equal its own
  `vis`/`bright`/`coverage` (the new short-circuit), and a case with a
  non-default `layers` dict asserting output values match the pre-refactor
  values captured once before the change (regression pin).
- New test: `vis_only(params, layers)` matches `features(params,
  layers)["vis"]` exactly, both for `layers=None` and a non-default `layers`
  dict.
- `tests/test_oneshot_train.py`: a test that `_sample_hindsight_goal`'s
  deltas (computed via the new `self._vis` path) match a reference computed
  by hand through `self._features` + `goals.aggregate` for the same sampled
  action, at a fixed seed with a stub/mocked model.
- All existing tests in `tests/test_oneshot_train.py` and the visibility
  suite must stay green unmodified otherwise — this is the regression
  backstop for "no accuracy loss".
- Re-run the profiling script (kept out-of-repo, in scratch) after the
  change on the same volume/config as the "before" table above, and report
  the new resets/s numbers.
