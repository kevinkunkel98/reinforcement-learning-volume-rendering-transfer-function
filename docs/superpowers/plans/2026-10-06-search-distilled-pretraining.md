# Search-distilled pretraining — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second supervised-pretraining teacher signal for the one-shot
SAC actor -- `hill_climb` search solutions on real sampled instructions --
alongside the existing hindsight teacher in `rl/pretrain.py`, and let
`rl/oneshot_train.py` mix both into one warm start.

**Architecture:** `OneShotEnv.search_policy_action()` mirrors the existing
`hindsight_policy_action()` exactly, just backed by `hill_climb` instead of
a hindsight-constructed target. `rl.pretrain.collect_search()` mirrors
`collect_hindsight()`'s multiprocessing/caching structure. `rl/oneshot_train.py`
gets its own `--search-pretrain-data`/`load_or_collect_search_pretrain`
(mirroring `--pretrain-data`/`load_or_collect_pretrain`) and a small
`merge_datasets()` helper so `main()` can hand `run_training` one combined
dataset when both sources are requested -- `pretrain_actor`/
`prefill_replay_buffer`/`save_dataset`/`load_dataset` need no changes at all,
since they only ever operate on `{"obs", "actions", "rewards"}` arrays.

**Tech Stack:** Python, Gymnasium, Stable-Baselines3, NumPy, pytest.

Reference: `docs/superpowers/specs/2026-10-06-search-distilled-pretraining-design.md`

---

### Task 1: `OneShotEnv.search_policy_action()`

**Files:**
- Modify: `rl/oneshot_env.py:30` (import), `rl/oneshot_env.py:345-358` (new method after `hindsight_policy_action`)
- Test: `tests/test_oneshot_env.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_oneshot_env.py`, right after `test_hindsight_goal_deltas_match_full_features_path` (ends at line 449, immediately before `test_v8_layer_residual_changes_reachable_opacity_only`):

```python
def test_search_policy_action_is_none_for_hindsight_episodes():
    env = OneShotEnv(["synthetic"], hindsight_ratio=1.0)
    env.reset(seed=0)
    assert env.search_policy_action() is None


def test_search_policy_action_beats_a_random_action_by_a_clear_margin():
    # Mirrors test_hindsight_oracle_beats_a_random_action_by_a_clear_margin:
    # a margin at the median, not a fixed threshold, so the test keeps
    # meaning if the keep penalty (and so absolute attainment) moves.
    # evaluations=50, not the 200 production default, so the suite stays fast.
    env = OneShotEnv(["synthetic"], hindsight_ratio=0.0)
    noise = np.random.default_rng(0)
    search_scores, random_scores = [], []
    for seed in range(10):
        env.reset(seed=seed)
        action = env.search_policy_action(evaluations=50)
        _obs, _reward, _done, _truncated, info = env.step(action)
        search_scores.append(info["attainment"])

        env.reset(seed=seed)   # the same episode, a different action
        _obs, _reward, _done, _truncated, info = env.step(noise.uniform(-1.0, 1.0, ACTION_SIZE))
        random_scores.append(info["attainment"])

    assert np.median(search_scores) > np.median(random_scores) + 0.5
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_oneshot_env.py::test_search_policy_action_is_none_for_hindsight_episodes tests/test_oneshot_env.py::test_search_policy_action_beats_a_random_action_by_a_clear_margin -v`
Expected: FAIL with `AttributeError: 'OneShotEnv' object has no attribute 'search_policy_action'`

- [ ] **Step 3: Read the current code**

Re-read `rl/oneshot_env.py:30` and `rl/oneshot_env.py:340-358` for the exact current text (the `Edit` tool requires an exact match).

- [ ] **Step 4: Import `hill_climb`, in `rl/oneshot_env.py`**

Replace:

```python
from rl.baselines import CONTROLLABLE, apply_controllable
```

with:

```python
from rl.baselines import CONTROLLABLE, apply_controllable, hill_climb
```

- [ ] **Step 5: Add `search_policy_action`, right after `hindsight_policy_action` in `rl/oneshot_env.py`**

Replace:

```python
    def hindsight_policy_action(self):
        """The hindsight oracle in the policy's own action space: the absolute
        controllable values for "absolute" mode, their offset from the start
        state for "residual" mode (layer actions, v8 only, left at 0 -- no
        change). None for an instruction episode. The supervised target for
        `rl.pretrain`."""
        if self._hindsight_action is None:
            return None
        action = self._hindsight_action.copy()
        if self.action_mode == "residual":
            action = action - np.asarray(self._controllable_values(self._start_params))
        if self.policy_version == V8_POLICY_VERSION:
            action = np.concatenate([action, np.zeros(len(LAYER_NAMES))])
        return action.astype(np.float32)
```

with:

```python
    def hindsight_policy_action(self):
        """The hindsight oracle in the policy's own action space: the absolute
        controllable values for "absolute" mode, their offset from the start
        state for "residual" mode (layer actions, v8 only, left at 0 -- no
        change). None for an instruction episode. The supervised target for
        `rl.pretrain`."""
        if self._hindsight_action is None:
            return None
        action = self._hindsight_action.copy()
        if self.action_mode == "residual":
            action = action - np.asarray(self._controllable_values(self._start_params))
        if self.policy_version == V8_POLICY_VERSION:
            action = np.concatenate([action, np.zeros(len(LAYER_NAMES))])
        return action.astype(np.float32)

    def search_policy_action(self, evaluations: int = 200, initial_step: float = 0.2):
        """A `hill_climb` search solution for this episode's real instruction,
        in the policy's own action space -- same encoding as
        `hindsight_policy_action`. None for a hindsight episode (which
        already has a free oracle action; see `hindsight_policy_action`).
        The supervised target for a search-distilled `rl.pretrain` run."""
        if self._hindsight_action is not None:
            return None
        target_params = hill_climb(self._model, self._start_params, self._instruction,
                                   evaluations=evaluations, initial_step=initial_step,
                                   active_layers=getattr(self, "_active_layers", None))
        action = np.asarray(self._controllable_values(target_params))
        if self.action_mode == "residual":
            action = action - np.asarray(self._controllable_values(self._start_params))
        if self.policy_version == V8_POLICY_VERSION:
            action = np.concatenate([action, np.zeros(len(LAYER_NAMES))])
        return action.astype(np.float32)
```

- [ ] **Step 6: Run the tests to confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_oneshot_env.py -v -k "search_policy_action or hindsight"`
Expected: PASS, all hindsight tests and both new search tests.

- [ ] **Step 7: Run the full project test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add rl/oneshot_env.py tests/test_oneshot_env.py
git commit -m "feat(rl): add OneShotEnv.search_policy_action, a hill_climb teacher action"
```

---

### Task 2: `rl.pretrain.collect_search`

**Files:**
- Modify: `rl/pretrain.py` (new seed helper + collection functions, alongside the hindsight ones)
- Test: `tests/test_pretrain.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pretrain.py`, right after `test_collected_reward_is_the_oracle_reward` (ends at line 39, immediately before `test_save_and_load_round_trip`):

```python
def test_collect_search_returns_aligned_samples():
    data = pretrain.collect_search(["synthetic"], V9, count=4, evaluations=20, seed=0)

    assert data["obs"].shape == (4, OBSERVATION_SIZE)
    assert data["actions"].shape == (4, ACTION_SIZE)
    assert data["rewards"].shape == (4,)
    assert np.all(np.abs(data["actions"]) <= 1.0)


def test_collect_search_is_deterministic_and_independent_of_workers():
    # Inherits the same potential full-suite-only flakiness as
    # test_collect_is_deterministic_and_independent_of_workers (see that
    # test and docs/superpowers/specs/2026-10-06-search-distilled-pretraining-design.md
    # -- torch.set_num_threads(1) only reliably constrains intra-op
    # parallelism before the process has done other torch work, which the
    # workers=1 path can't guarantee inside a shared test-suite process).
    # Pre-existing risk, not something to paper over here.
    one = pretrain.collect_search(["synthetic"], V9, count=4, evaluations=20, seed=3, workers=1)
    again = pretrain.collect_search(["synthetic"], V9, count=4, evaluations=20, seed=3, workers=2)

    for key in ("obs", "actions", "rewards"):
        np.testing.assert_allclose(one[key], again[key])


def test_collected_search_reward_is_the_search_reward():
    data = pretrain.collect_search(["synthetic"], V9, count=3, evaluations=20, seed=5)
    env = _env(hindsight_ratio=0.0)
    for index in range(3):
        env.reset(seed=pretrain.search_episode_seed(5, index))
        _obs, reward, *_ = env.step(env.search_policy_action(evaluations=20))
        assert data["rewards"][index] == pytest.approx(reward)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_pretrain.py::test_collect_search_returns_aligned_samples tests/test_pretrain.py::test_collect_search_is_deterministic_and_independent_of_workers tests/test_pretrain.py::test_collected_search_reward_is_the_search_reward -v`
Expected: FAIL with `AttributeError: module 'rl.pretrain' has no attribute 'collect_search'` (and, once that's added, `has no attribute 'search_episode_seed'`)

- [ ] **Step 3: Read the current code**

Re-read `rl/pretrain.py` in full (it's short, ~133 lines) for the exact current text around `episode_seed`, `_collect_chunk`, and `collect_hindsight`.

- [ ] **Step 4: Add the seed helper and collection functions, in `rl/pretrain.py`**

Replace:

```python
# Collection seeds live far from training (SB3's own) and validation
# (`rl.oneshot_train.VALIDATION_SEED_BASE`, 10_000 + i) seeds.
PRETRAIN_SEED_BASE = 1_000_000
SEED_STRIDE = 100_000
# atanh(0.97) ~ 2.1: targets at the edge of [-1, 1] would ask the squashed
# actor for an unbounded pre-tanh mean.
TARGET_CLIP = 0.97


def episode_seed(seed: int, index: int) -> int:
    return PRETRAIN_SEED_BASE + seed * SEED_STRIDE + index


def _collect_chunk(job) -> list:
    volume_ids, contract, seed, indices = job
    torch.set_num_threads(1)
    env = OneShotEnv(list(volume_ids), hindsight_ratio=1.0, **contract)
    rows = []
    for index in indices:
        obs, _ = env.reset(seed=episode_seed(seed, index))
        action = env.hindsight_policy_action()
        _obs, reward, *_ = env.step(action)
        rows.append((index, obs, action, float(reward)))
    return rows


def collect_hindsight(volume_ids, contract: dict, count: int, seed: int = 0,
                      workers: int = 1) -> dict:
    """`count` hindsight episodes on `volume_ids` under the env `contract`
    (policy_version/action_mode/reward_mode): {"obs", "actions", "rewards"}.
    Each episode is seeded by its index alone, so the result does not depend
    on `workers`."""
    indices = list(range(count))
    chunks = [indices[i::workers] for i in range(workers)] if workers > 1 else [indices]
    jobs = [(tuple(volume_ids), dict(contract), seed, chunk) for chunk in chunks if chunk]
    if len(jobs) == 1:
        rows = _collect_chunk(jobs[0])
    else:
        with multiprocessing.get_context("spawn").Pool(len(jobs)) as pool:
            rows = [row for chunk in pool.map(_collect_chunk, jobs) for row in chunk]
    rows.sort(key=lambda row: row[0])
    return {"obs": np.stack([row[1] for row in rows]).astype(np.float32),
            "actions": np.stack([row[2] for row in rows]).astype(np.float32),
            "rewards": np.asarray([row[3] for row in rows], dtype=np.float32)}
```

with:

```python
# Collection seeds live far from training (SB3's own) and validation
# (`rl.oneshot_train.VALIDATION_SEED_BASE`, 10_000 + i) seeds.
PRETRAIN_SEED_BASE = 1_000_000
SEED_STRIDE = 100_000
# Its own seed space, far from PRETRAIN_SEED_BASE: a search-distilled run
# collected under the same --seed as a hindsight run must not reuse the
# exact same rng draws (same volume/start noise) before the two diverge.
SEARCH_SEED_BASE = 2_000_000
# atanh(0.97) ~ 2.1: targets at the edge of [-1, 1] would ask the squashed
# actor for an unbounded pre-tanh mean.
TARGET_CLIP = 0.97


def episode_seed(seed: int, index: int) -> int:
    return PRETRAIN_SEED_BASE + seed * SEED_STRIDE + index


def search_episode_seed(seed: int, index: int) -> int:
    return SEARCH_SEED_BASE + seed * SEED_STRIDE + index


def _collect_chunk(job) -> list:
    volume_ids, contract, seed, indices = job
    torch.set_num_threads(1)
    env = OneShotEnv(list(volume_ids), hindsight_ratio=1.0, **contract)
    rows = []
    for index in indices:
        obs, _ = env.reset(seed=episode_seed(seed, index))
        action = env.hindsight_policy_action()
        _obs, reward, *_ = env.step(action)
        rows.append((index, obs, action, float(reward)))
    return rows


def _collect_search_chunk(job) -> list:
    volume_ids, contract, evaluations, seed, indices = job
    torch.set_num_threads(1)
    env = OneShotEnv(list(volume_ids), hindsight_ratio=0.0, **contract)
    rows = []
    for index in indices:
        obs, _ = env.reset(seed=search_episode_seed(seed, index))
        action = env.search_policy_action(evaluations=evaluations)
        _obs, reward, *_ = env.step(action)
        rows.append((index, obs, action, float(reward)))
    return rows


def collect_hindsight(volume_ids, contract: dict, count: int, seed: int = 0,
                      workers: int = 1) -> dict:
    """`count` hindsight episodes on `volume_ids` under the env `contract`
    (policy_version/action_mode/reward_mode): {"obs", "actions", "rewards"}.
    Each episode is seeded by its index alone, so the result does not depend
    on `workers`."""
    indices = list(range(count))
    chunks = [indices[i::workers] for i in range(workers)] if workers > 1 else [indices]
    jobs = [(tuple(volume_ids), dict(contract), seed, chunk) for chunk in chunks if chunk]
    if len(jobs) == 1:
        rows = _collect_chunk(jobs[0])
    else:
        with multiprocessing.get_context("spawn").Pool(len(jobs)) as pool:
            rows = [row for chunk in pool.map(_collect_chunk, jobs) for row in chunk]
    rows.sort(key=lambda row: row[0])
    return {"obs": np.stack([row[1] for row in rows]).astype(np.float32),
            "actions": np.stack([row[2] for row in rows]).astype(np.float32),
            "rewards": np.asarray([row[3] for row in rows], dtype=np.float32)}


def collect_search(volume_ids, contract: dict, count: int, evaluations: int = 200,
                   seed: int = 0, workers: int = 1) -> dict:
    """Like `collect_hindsight`, but the teacher is a `hill_climb` search
    solution for a real sampled instruction (`hindsight_ratio=0.0`), not a
    reachable hindsight target. `evaluations` is `hill_climb`'s per-episode
    search budget."""
    indices = list(range(count))
    chunks = [indices[i::workers] for i in range(workers)] if workers > 1 else [indices]
    jobs = [(tuple(volume_ids), dict(contract), evaluations, seed, chunk) for chunk in chunks if chunk]
    if len(jobs) == 1:
        rows = _collect_search_chunk(jobs[0])
    else:
        with multiprocessing.get_context("spawn").Pool(len(jobs)) as pool:
            rows = [row for chunk in pool.map(_collect_search_chunk, jobs) for row in chunk]
    rows.sort(key=lambda row: row[0])
    return {"obs": np.stack([row[1] for row in rows]).astype(np.float32),
            "actions": np.stack([row[2] for row in rows]).astype(np.float32),
            "rewards": np.asarray([row[3] for row in rows], dtype=np.float32)}
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_pretrain.py -v`
Expected: PASS, all tests (existing hindsight tests plus the three new ones)

- [ ] **Step 6: Run the full project test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add rl/pretrain.py tests/test_pretrain.py
git commit -m "feat(rl): add pretrain.collect_search, a hill_climb-distilled teacher dataset"
```

---

### Task 3: Wire search-pretrain data into `rl/oneshot_train.py`, with mixing

**Files:**
- Modify: `rl/oneshot_train.py` (new CLI flags, `load_or_collect_search_pretrain`, `merge_datasets`, `main()`)
- Test: `tests/test_oneshot_train.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_oneshot_train.py`, right after `test_cli_rejects_a_pretrain_dataset_for_another_contract` (the file's last test, ends at line 468):

```python
def test_cli_rejects_a_search_pretrain_dataset_for_another_contract(tmp_path):
    from rl import pretrain
    path = tmp_path / "search.npz"
    data = {"obs": np.zeros((1, 97), np.float32), "actions": np.zeros((1, 24), np.float32),
            "rewards": np.zeros(1, np.float32)}
    pretrain.save_dataset(data, str(path), {"policy_version": "oneshot-v7"})
    args = oneshot_train.parse_args(["--policy-version", "oneshot-v9", "--action-mode", "residual",
                                     "--reward-mode", "target", "--search-pretrain-data", str(path)])
    with pytest.raises(ValueError, match="does not match"):
        oneshot_train.load_or_collect_search_pretrain(args)


def test_merge_datasets_concatenates_in_order():
    a = {"obs": np.zeros((2, 3), np.float32), "actions": np.ones((2, 1), np.float32),
         "rewards": np.array([1.0, 2.0], dtype=np.float32)}
    b = {"obs": np.ones((3, 3), np.float32), "actions": np.zeros((3, 1), np.float32),
         "rewards": np.array([3.0, 4.0, 5.0], dtype=np.float32)}

    merged = oneshot_train.merge_datasets(a, b)

    assert merged["obs"].shape == (5, 3)
    assert merged["actions"].shape == (5, 1)
    assert list(merged["rewards"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_oneshot_train.py::test_cli_rejects_a_search_pretrain_dataset_for_another_contract tests/test_oneshot_train.py::test_merge_datasets_concatenates_in_order -v`
Expected: FAIL with `error: unrecognized arguments: --search-pretrain-data ...` (argparse) and, once that's fixed, `AttributeError: module 'rl.oneshot_train' has no attribute 'merge_datasets'`

- [ ] **Step 3: Read the current code**

Re-read `rl/oneshot_train.py:196-278` for the exact current text (the `Edit` tool requires an exact match).

- [ ] **Step 4: Add the three new CLI flags, in `rl/oneshot_train.py`**

Replace:

```python
    parser.add_argument("--pretrain-data", type=str, default=None,
                        help="hindsight dataset (.npz); collected and saved here first if missing")
    parser.add_argument("--pretrain-episodes", type=int, default=4000)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--pretrain-workers", type=int, default=6)
```

with:

```python
    parser.add_argument("--pretrain-data", type=str, default=None,
                        help="hindsight dataset (.npz); collected and saved here first if missing")
    parser.add_argument("--pretrain-episodes", type=int, default=4000)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--pretrain-workers", type=int, default=6)
    parser.add_argument("--search-pretrain-data", type=str, default=None,
                        help="hill-climb search dataset (.npz); collected and saved here first if "
                             "missing, mixed in with --pretrain-data when both are given")
    parser.add_argument("--search-pretrain-episodes", type=int, default=1000)
    parser.add_argument("--search-evaluations", type=int, default=200,
                        help="hill_climb search budget per collected episode")
```

- [ ] **Step 5: Add `load_or_collect_search_pretrain` and `merge_datasets`, in `rl/oneshot_train.py`**

Replace:

```python
def load_or_collect_pretrain(args) -> tuple:
    """(dataset, provenance) for --pretrain-data: loaded when the file exists
    and matches this run's contract, else collected on the training split
    and saved there."""
    contract = {"policy_version": args.policy_version, "action_mode": args.action_mode,
                "reward_mode": args.reward_mode}
    expected = {**observation_metadata(**contract), "episodes": args.pretrain_episodes,
                "seed": args.seed, "split": "train",
                "hindsight_brightness_share": args.hindsight_brightness}
    if os.path.exists(args.pretrain_data):
        data, stored = pretrain.load_dataset(args.pretrain_data)
        # Datasets collected before brightness hindsight existed had none.
        stored.setdefault("hindsight_brightness_share", 0.0)
        mismatched = {key for key in expected if stored.get(key) != expected[key]}
        if mismatched:
            raise ValueError(f"{args.pretrain_data} does not match this run: {sorted(mismatched)}")
    else:
        data = pretrain.collect_hindsight(datasets.volumes_for_split("train"),
                                          {**contract, "hindsight_brightness_share": args.hindsight_brightness},
                                          args.pretrain_episodes, args.seed, args.pretrain_workers)
        pretrain.save_dataset(data, args.pretrain_data, expected)
    return data, {"data": args.pretrain_data, "episodes": args.pretrain_episodes}
```

with:

```python
def load_or_collect_pretrain(args) -> tuple:
    """(dataset, provenance) for --pretrain-data: loaded when the file exists
    and matches this run's contract, else collected on the training split
    and saved there."""
    contract = {"policy_version": args.policy_version, "action_mode": args.action_mode,
                "reward_mode": args.reward_mode}
    expected = {**observation_metadata(**contract), "episodes": args.pretrain_episodes,
                "seed": args.seed, "split": "train",
                "hindsight_brightness_share": args.hindsight_brightness}
    if os.path.exists(args.pretrain_data):
        data, stored = pretrain.load_dataset(args.pretrain_data)
        # Datasets collected before brightness hindsight existed had none.
        stored.setdefault("hindsight_brightness_share", 0.0)
        mismatched = {key for key in expected if stored.get(key) != expected[key]}
        if mismatched:
            raise ValueError(f"{args.pretrain_data} does not match this run: {sorted(mismatched)}")
    else:
        data = pretrain.collect_hindsight(datasets.volumes_for_split("train"),
                                          {**contract, "hindsight_brightness_share": args.hindsight_brightness},
                                          args.pretrain_episodes, args.seed, args.pretrain_workers)
        pretrain.save_dataset(data, args.pretrain_data, expected)
    return data, {"data": args.pretrain_data, "episodes": args.pretrain_episodes}


def load_or_collect_search_pretrain(args) -> tuple:
    """(dataset, provenance) for --search-pretrain-data: loaded when the file
    exists and matches this run's contract, else collected on the training
    split (hill_climb solutions to real instructions) and saved there.
    Reuses --pretrain-workers for parallelism -- the two collection passes
    never run at the same time, so a separate worker-count flag would only
    add CLI surface without adding anything to configure."""
    contract = {"policy_version": args.policy_version, "action_mode": args.action_mode,
                "reward_mode": args.reward_mode}
    expected = {**observation_metadata(**contract), "episodes": args.search_pretrain_episodes,
                "seed": args.seed, "split": "train", "evaluations": args.search_evaluations}
    if os.path.exists(args.search_pretrain_data):
        data, stored = pretrain.load_dataset(args.search_pretrain_data)
        mismatched = {key for key in expected if stored.get(key) != expected[key]}
        if mismatched:
            raise ValueError(f"{args.search_pretrain_data} does not match this run: {sorted(mismatched)}")
    else:
        data = pretrain.collect_search(datasets.volumes_for_split("train"), contract,
                                       args.search_pretrain_episodes, args.search_evaluations,
                                       args.seed, args.pretrain_workers)
        pretrain.save_dataset(data, args.search_pretrain_data, expected)
    return data, {"data": args.search_pretrain_data, "episodes": args.search_pretrain_episodes}


def merge_datasets(*datasets_: dict) -> dict:
    """Concatenate `{"obs", "actions", "rewards"}` dicts into one, in order
    -- e.g. hindsight and search-distilled pretrain data, so `main()` can
    hand `run_training` a single combined `pretrain_data`. `pretrain_actor`/
    `prefill_replay_buffer` don't need to know a dataset came from more than
    one source."""
    return {key: np.concatenate([data[key] for data in datasets_]) for key in ("obs", "actions", "rewards")}
```

(The parameter is named `datasets_` with a trailing underscore to avoid shadowing the `datasets` module already imported at the top of this file.)

- [ ] **Step 6: Wire both sources into `main()`, in `rl/oneshot_train.py`**

Replace:

```python
def main(argv=None):
    args = parse_args(argv)
    out = args.out or DEFAULT_OUT_TEMPLATE.format(seed=args.seed)
    timesteps = SHORT_EXPERIMENT["timesteps"] if args.short else args.timesteps
    eval_interval = SHORT_EXPERIMENT["eval_interval"] if args.short else args.eval_interval
    pretrain_data, pretrain_info = (load_or_collect_pretrain(args)
                                    if args.pretrain_data else (None, None))
    result = run_training(out=out, timesteps=timesteps, seed=args.seed, eval_interval=eval_interval,
                          eval_episode_count=(SHORT_EXPERIMENT["eval_episode_count"] if args.short
                                              else args.eval_episodes or VALIDATION_EPISODES),
                          env_kwargs={"policy_version": args.policy_version,
                                      "action_mode": args.action_mode,
                                      "hindsight_ratio": args.hindsight_ratio,
                                      "balance_classes": args.balance_classes,
                                      "hindsight_brightness_share": args.hindsight_brightness,
                                      "reward_mode": args.reward_mode},
                          pretrain_data=pretrain_data, pretrain_epochs=args.pretrain_epochs,
                          pretrain_info=pretrain_info)
    print(f"\n[oneshot_train] wrote {result['eval_progress_path']} and {result['best_path']}")
```

with:

```python
def main(argv=None):
    args = parse_args(argv)
    out = args.out or DEFAULT_OUT_TEMPLATE.format(seed=args.seed)
    timesteps = SHORT_EXPERIMENT["timesteps"] if args.short else args.timesteps
    eval_interval = SHORT_EXPERIMENT["eval_interval"] if args.short else args.eval_interval
    pretrain_sources = []
    if args.pretrain_data:
        pretrain_sources.append(load_or_collect_pretrain(args))
    if args.search_pretrain_data:
        pretrain_sources.append(load_or_collect_search_pretrain(args))
    if not pretrain_sources:
        pretrain_data, pretrain_info = None, None
    elif len(pretrain_sources) == 1:
        pretrain_data, pretrain_info = pretrain_sources[0]
    else:
        source_data, source_info = zip(*pretrain_sources)
        pretrain_data = merge_datasets(*source_data)
        pretrain_info = {"sources": list(source_info)}
    result = run_training(out=out, timesteps=timesteps, seed=args.seed, eval_interval=eval_interval,
                          eval_episode_count=(SHORT_EXPERIMENT["eval_episode_count"] if args.short
                                              else args.eval_episodes or VALIDATION_EPISODES),
                          env_kwargs={"policy_version": args.policy_version,
                                      "action_mode": args.action_mode,
                                      "hindsight_ratio": args.hindsight_ratio,
                                      "balance_classes": args.balance_classes,
                                      "hindsight_brightness_share": args.hindsight_brightness,
                                      "reward_mode": args.reward_mode},
                          pretrain_data=pretrain_data, pretrain_epochs=args.pretrain_epochs,
                          pretrain_info=pretrain_info)
    print(f"\n[oneshot_train] wrote {result['eval_progress_path']} and {result['best_path']}")
```

Note: the single-source case (`--pretrain-data` alone, today's only usage)
keeps its exact old `pretrain_info` shape (`{"data": ..., "episodes": ...}`)
-- `{"sources": [...]}` only appears when both flags are given, a new
capability nothing old depends on.

- [ ] **Step 7: Run the new tests**

Run: `.venv/bin/python -m pytest tests/test_oneshot_train.py -v`
Expected: PASS, all tests (existing pretrain-dataset tests plus the two new ones)

- [ ] **Step 8: Run the full project test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add rl/oneshot_train.py tests/test_oneshot_train.py
git commit -m "feat(rl): mix hill_climb-distilled data into oneshot_train's warm start"
```

---

### Task 4: Update the training-setup reference doc

**Files:**
- Modify: `docs/rl-training-setup.typ`

- [ ] **Step 1: Add the new flags to the "Supervised warm start" section**

Read `docs/rl-training-setup.typ`'s `== Supervised warm start (`rl/pretrain.py`)` section. After its existing paragraph ending "...only the random warm-up.", add a new paragraph:

```typst
A second teacher source, `--search-pretrain-data`, collects `hill_climb`
solutions on real sampled instructions instead of hindsight targets
(`rl.pretrain.collect_search`, `OneShotEnv.search_policy_action`) --
`docs/STATUS.md`'s "distil search into the policy". When both
`--pretrain-data` and `--search-pretrain-data` are given, the two datasets
are concatenated (`oneshot_train.merge_datasets`) before the single
`pretrain_actor`/`prefill_replay_buffer` pass. `--search-evaluations`
(default 200) sets `hill_climb`'s per-episode budget; at that budget,
collection costs roughly 2.7 s/episode, hence the lower default episode
count (`--search-pretrain-episodes`, default 1000 vs. hindsight's 4000).
```

- [ ] **Step 2: Compile to confirm no syntax errors**

Run: `typst compile docs/rl-training-setup.typ /tmp/rl-training-setup-check.pdf`
Expected: exits 0, no errors. Then `rm /tmp/rl-training-setup-check.pdf`.

- [ ] **Step 3: Commit**

```bash
git add docs/rl-training-setup.typ
git commit -m "docs: record search-distilled pretraining in the training-setup note"
```
