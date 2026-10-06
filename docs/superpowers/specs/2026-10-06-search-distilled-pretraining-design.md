# Search-distilled pretraining — design

## Goal

Give the one-shot SAC actor a supervised warm start from `hill_climb`
search solutions on *real* sampled instructions, as a second teacher signal
alongside the existing hindsight warm start (`rl/pretrain.py`). This is
`docs/STATUS.md`'s long-standing "cheapest improvement #3" ("Distil search
into the policy — supervised pretraining on hill-climb solutions before RL,
the standard way to close an amortisation gap"), not yet implemented on
`main`.

Out of scope for this version: targeting specific weak instruction kinds
(`show_only`, compound) — teacher episodes are drawn from the normal
`goals.sample_instruction` mix, same as real training episodes, no biasing
knob. Also out of scope: replacing hindsight pretraining, or any new mixing
logic inside `rl/pretrain.py` itself (see Approach).

## Context

An abandoned, uncommitted prototype of this idea exists in the
`.worktrees/rl-optimization` worktree (`rl/teacher_data.py` +
`rl/distill.py`, branched 2026-09-20). It predates `oneshot-v9`, implements
its own `action_mode="residual"` support that `main` has since superseded
with a more complete, policy-version-aware design, and reimplements
observation-building, action-encoding, and a from-scratch actor network
offline, guarded by ~240 lines of provenance-contract validation to catch
drift between that offline pipeline and the live code. It is not being
merged or rebased — the architecture it targets no longer exists on `main`
— but its core idea (hill-climb solutions as supervised targets) is sound
and is what this spec reimplements, cleanly, against current `main`.

## Approach

**Reuse `rl/pretrain.py`'s existing machinery instead of a parallel
pipeline.** `pretrain_actor`/`prefill_replay_buffer`/`save_dataset`/
`load_dataset` only ever consume `{"obs", "actions", "rewards"}` arrays —
they don't know or care how those triples were produced. So "distill
search" only needs a new *producer* of that same shape, collected and
consumed inside one run the same way hindsight data already is, with no new
offline file format and no provenance-contract re-validation layer: the
data is used in the same process that just generated it (or a cached `.npz`
from an earlier run of the *same* collection code), never a cross-codebase
offline artifact the way the old prototype's JSONL was.

**`OneShotEnv.search_policy_action(evaluations=200, initial_step=0.2)`**
(new method, placed right after `hindsight_policy_action()` in
`rl/oneshot_env.py`): runs `rl.baselines.hill_climb` on the episode's own
*real, sampled* instruction (not a hindsight-constructed one) and encodes
the result in the policy's own action space, mirroring
`hindsight_policy_action()` exactly — residual offset from the start state,
zero-padded for v8's layer actions. Returns `None` when the episode is a
hindsight episode (symmetry with `hindsight_policy_action`; never actually
hit by the collector below, which always runs with `hindsight_ratio=0.0`).

**`rl.pretrain.collect_search(volume_ids, contract, count, evaluations=200,
seed=0, workers=1)`** (new function, alongside `collect_hindsight`): same
shape and multiprocessing structure as `collect_hindsight`/`_collect_chunk`
— a new `_collect_search_chunk` module-level function (not a shared closure
parameter: closures aren't picklable for `multiprocessing`'s spawn
context, and the codebase already tolerates this kind of small duplication,
e.g. `hill_climb_10`/`hill_climb_200` in `rl/baselines.py`) that builds an
`OneShotEnv(..., hindsight_ratio=0.0, **contract)`, and per episode:
`env.reset(seed=...)` → `env.search_policy_action(evaluations=evaluations)`
→ `env.step(action)`, collecting `(obs, action, reward)` exactly like
hindsight collection does. `save_dataset`/`load_dataset` need no changes.

**CLI (`rl/oneshot_train.py`):** three new flags, `--search-pretrain-data`
(path, same load-or-collect-and-cache pattern as `--pretrain-data`),
`--search-pretrain-episodes` (default **1000** — deliberately lower than
hindsight's 4000 default; see Cost below), `--search-evaluations` (hill-climb
budget per episode, default 200, matching baseline `B4_hill_climb_200`'s
convention). A new `load_or_collect_search_pretrain(args)` mirrors
`load_or_collect_pretrain`, with its own provenance dict (adds
`"evaluations": args.search_evaluations` to the matched keys, so a cached
file collected under a different hill-climb budget is correctly rejected
rather than silently reused).

**Mixing the two sources is a `main()`-level concatenation, not new
library code.** When both `--pretrain-data` and `--search-pretrain-data`
are given, `main()` builds each dataset dict independently (each through its
own `load_or_collect_*` / cache file) and merges them with one
`np.concatenate` per key (`obs`, `actions`, `rewards`) before passing the
combined dict to `run_training`'s existing `pretrain_data` parameter.
`pretrain_actor`/`prefill_replay_buffer` run once, over the union, exactly
as they do today over hindsight-only data — no new code path inside
`rl/pretrain.py` for "mixed" data, because a merged dict of the same three
arrays *is* a valid `rl.pretrain` dataset by construction.

## Cost

`hill_climb` at `evaluations=200` costs ~2.7 s/episode (measured and
recorded in the pre-v9 `docs/STATUS.md`). For 1000 episodes at 6 workers,
collection is ~7-8 minutes, one-time per `--seed`/`--search-evaluations`
combination (cached afterward, same as hindsight data). This is why the
default episode count is lower than hindsight's 4000: hindsight collection
is itself not free (`rl/pretrain.py`'s own docstring: "collection is the
slow part"), but a hindsight episode's "search" is a single perturbation and
reachability check, not a multi-step coordinate-descent optimization — the
two defaults are not meant to be equal.

## Known pre-existing risk this design inherits

`tests/test_pretrain.py::test_collect_is_deterministic_and_independent_of_workers`
is a currently-flaky test (confirmed via bisection during the preceding
throughput work: passes in isolation, intermittently fails — tiny float32
divergence, ~3e-8 — when run inside the full suite). The likely cause:
`torch.set_num_threads(1)` inside `_collect_chunk` only reliably constrains
intra-op parallelism if called before that process has done any other torch
work; the `workers=1` path runs `_collect_chunk` in the *same* process as
the rest of the test suite (which has already touched torch), while the
`workers>1` path runs it in fresh spawned processes where it's the first
torch call. This is pre-existing and not caused by this design, but
`_collect_search_chunk` reproduces the identical pattern, so an equivalent
determinism test for `collect_search` will likely inherit the same
full-suite-only flakiness. Not blocking — noted so it isn't mistaken for a
new regression when it's seen again.

## Testing

- `tests/test_oneshot_env.py`: `search_policy_action()` returns `None` on a
  hindsight episode; on a real instruction, the returned action — replayed
  via `apply_controllable`/`goals.distance` — improves on the start state by
  a clear margin (same pattern as the existing
  `test_hindsight_oracle_beats_a_random_action_by_a_clear_margin`).
- `tests/test_pretrain.py`: `collect_search` is deterministic and
  worker-count-independent (mirrors the existing hindsight test, with the
  flakiness risk above noted inline as a comment, not papered over);
  `collect_search`'s reward matches replaying `search_policy_action()`
  through `env.step()` directly (mirrors
  `test_collected_reward_is_the_oracle_reward`).
- `tests/test_oneshot_train.py`: `load_or_collect_search_pretrain` caches,
  loads, and rejects a mismatched file (including a mismatched
  `evaluations` budget) the same way `load_or_collect_pretrain` already is
  tested; a `main()`-level test that passing both `--pretrain-data` and
  `--search-pretrain-data` produces a correctly concatenated combined
  dataset (length = sum of both, each half's rows recognizable in the
  result).
