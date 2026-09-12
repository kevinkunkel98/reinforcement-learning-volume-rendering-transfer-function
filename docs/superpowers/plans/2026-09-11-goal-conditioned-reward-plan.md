# Goal-Conditioned Reward Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a goal-conditioned, pretrained, human-fine-tuned reward pipeline with pair extraction, ensemble safeguards, and isolated evaluation.

**Architecture:** `rl/reward_model.py` remains the shared model/loss/serialization core. Data scripts produce canonical pair records and training artifacts; evaluation consumes fixed splits without writing training data. `RewardModelTFEnv` combines ensemble reward, objective anchor, and independent degeneracy penalties while preserving existing environment interfaces.

**Tech Stack:** Python, NumPy, PyTorch, Gymnasium, Pillow, existing VTK renderer, pytest.

---

## File Map

- Modify: `rl/reward_model.py` for 18-feature encoding, 64/32 MLP, weighted Bradley-Terry training, checkpoints, ensembles, and prediction helpers.
- Modify: `rl/reward_model_env.py` for ensemble aggregation, objective anchoring, opacity/coverage penalties, and configurable CLI-facing parameters.
- Create: `rl/extract_pairs.py` for current-log/canonical-client adapters and pair generation.
- Create: `rl/pretrain_reward.py` for synthetic rendering pairs and objective pretraining.
- Create: `rl/finetune_reward.py` for fixed-split human fine-tuning.
- Create: `rl/eval_reward.py` for reward-only accuracy, objective policy sanity checks, and blind head-to-head collection.
- Modify: `tests/test_rl_reward_model.py` for the intentional 18-input/model/loss API change.
- Modify: `tests/test_rl_reward_model_env.py` for anchor, ensemble, and degenerate-state behavior.
- Create: `tests/test_rl_extract_pairs.py` for all pair sources and feature recovery.
- Create: `tests/test_rl_reward_pipeline.py` for synthetic generation, split isolation, and evaluation.
- Modify: `README.md` with commands, artifact names, client-neutral schema, and reported-result workflow.

## Task 1: Replace Reward Core

**Files:** `tests/test_rl_reward_model.py`, `rl/reward_model.py`

- [ ] **Step 1: Write failing representation tests**

```python
def test_featurize_contains_after_before_delta_goal_and_direction():
    before = {"mean": 10., "std": 5., "coverage": .2, "entropy": 1.}
    after = {"mean": 15., "std": 6., "coverage": .3, "entropy": 1.2}
    x = featurize(before, after, "bone", "increase")
    assert x.shape == (18,)
    np.testing.assert_allclose(x[:4], [15., 6., .3, 1.2])
    np.testing.assert_allclose(x[4:8], [10., 5., .2, 1.])
    np.testing.assert_allclose(x[8:12], [5., 1., .1, .2])
    np.testing.assert_allclose(x[12:17], [0., 0., 0., 0., 1.])
    assert x[17] == 1.
```

- [ ] **Step 2: Run test and verify expected failure**

Run: `pytest tests/test_rl_reward_model.py::test_featurize_contains_after_before_delta_goal_and_direction -q`

Expected: FAIL because current input dimension/order is 10.

- [ ] **Step 3: Write failing architecture and weighted-loss tests**

```python
def test_reward_model_has_two_hidden_layers():
    layers = list(RewardModel().net)
    assert [layer.out_features for layer in layers if isinstance(layer, nn.Linear)] == [64, 32, 1]

def test_weighted_bradley_terry_loss_scales_examples():
    preferred = torch.tensor([2.0, 0.0])
    other = torch.tensor([0.0, 0.0])
    low = bradley_terry_loss(preferred, other, torch.tensor([1.0, 1.0]))
    high = bradley_terry_loss(preferred, other, torch.tensor([1.0, 3.0]))
    assert high > low
```

- [ ] **Step 4: Implement minimal core**

Implement `encode_target(cmd)`, 18-value `featurize`, `RewardModel(18, 64, 32)`, `bradley_terry_loss`, and training over records with `weight` values. Keep `predict_reward` returning `2*sigmoid(logit)-1`; accept both canonical nested observations and legacy flat records during migration.

- [ ] **Step 5: Add checkpoint and ensemble tests**

```python
def test_checkpoint_round_trip_preserves_prediction(tmp_path):
    path = tmp_path / "model.pt"
    model = RewardModel()
    save_reward_model(model, path)
    loaded = load_reward_model(path)
    x = torch.zeros(1, INPUT_DIM)
    assert loaded(x).item() == pytest.approx(model(x).item())

def test_ensemble_reward_is_mean_minus_std():
    values = np.array([[.2, .4], [.6, .2]])
    assert ensemble_reward(values) == pytest.approx(values.mean() - values.std())
```

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/test_rl_reward_model.py -q`

Expected: all reward-core tests pass, including updated legacy training tests.

## Task 2: Extract Canonical Preference Pairs

**Files:** `tests/test_rl_extract_pairs.py`, `rl/extract_pairs.py`

- [ ] **Step 1: Write fixture tests for thumbs, episodes, and branches**

Use `tmp_path` JSONL fixtures containing current `log.jsonl`/`feedback.jsonl` rows, canonical records with explicit `parent_step_id`/`carried_forward`, and image paths. Assert output fields, source weights `1.0/.7/.4`, target/direction, labels, and `k-j >= 2` filtering.

- [ ] **Step 2: Run extractor tests and verify expected failure**

Run: `pytest tests/test_rl_extract_pairs.py -q`

Expected: FAIL because `rl.extract_pairs` does not exist.

- [ ] **Step 3: Implement schema adapters and feature recovery**

Read JSONL safely, normalize `command`/`cmd_dict`, resolve target/direction, use supplied feature dictionaries first, and load optional PNG/JPEG paths with Pillow plus `render.features`. Preserve paths in observations. Skip and count rows with neither usable features nor readable images.

- [ ] **Step 4: Implement source generators**

Generate explicit branch pairs only from recoverable branch metadata; generate accepted/ended trajectory pairs only for same session/episode/command and step gaps of at least two; join thumbs by session/step then unique params fallback. Deduplicate using source/session/step pair keys.

- [ ] **Step 5: Implement CLI summary and output**

Write JSONL to `out/pairs.jsonl` by default. Print counts by `source`, `target_tissue`, skipped rows, and total. Empty/missing input produces valid empty output and summary.

- [ ] **Step 6: Run focused and existing ingestion tests**

Run: `pytest tests/test_rl_extract_pairs.py tests/test_rl_ingest_feedback.py -q`

Expected: PASS; existing `ingest_feedback` behavior remains intact.

## Task 3: Synthetic Pretraining

**Files:** `tests/test_rl_reward_pipeline.py`, `rl/pretrain_reward.py`

- [ ] **Step 1: Write generation tests with injected renderer**

```python
def test_generate_pairs_uses_two_deltas_and_objective_label(monkeypatch):
    monkeypatch.setattr(pretrain_reward, "render_features", lambda *args: {
        "mean": float(args[1][0]), "std": 0., "coverage": 1., "entropy": 0.
    })
    rows = generate_synthetic_pairs(4, seed=3, renderer=lambda params: params)
    assert len(rows) == 4
    assert all(row["source"] == "synthetic" for row in rows)
    assert all(row["target_tissue"] in TISSUES for row in rows)
```

- [ ] **Step 2: Run test and verify expected failure**

Run: `pytest tests/test_rl_reward_pipeline.py::test_generate_pairs_uses_two_deltas_and_objective_label -q`

Expected: FAIL because pretraining module/functions do not exist.

- [ ] **Step 3: Implement synthetic generator**

Sample default TF vectors, target/direction, and two bounded deltas with a seeded NumPy generator. Render both through an injectable function, extract scalar features, and label by signed target `mass_fraction`. Include a code `TODO` identifying replacement with visibility when available.

- [ ] **Step 4: Implement pretraining and artifact saving**

Train one model per seed/member, save member checkpoints plus aggregate `reward_pretrained.pt`, and expose CLI flags for pair count, epochs, seed, members, learning rate, and output path. Default pair count is at least 20,000 and default members is 5; tests use small values.

- [ ] **Step 5: Run pipeline tests**

Run: `pytest tests/test_rl_reward_pipeline.py -q`

Expected: synthetic generation and checkpoint artifact tests pass.

## Task 4: Human Fine-Tuning and Fixed Splits

**Files:** `tests/test_rl_reward_pipeline.py`, `rl/finetune_reward.py`

- [ ] **Step 1: Write split/no-leakage tests**

```python
def test_fixed_split_is_seeded_and_disjoint():
    rows = [{"id": i} for i in range(10)]
    first = split_pairs(rows, seed=11, test_fraction=.2)
    second = split_pairs(rows, seed=11, test_fraction=.2)
    assert first == second
    assert {id(row) for row in first[0]}.isdisjoint({id(row) for row in first[1]})
```

- [ ] **Step 2: Run test and verify expected failure**

Run: `pytest tests/test_rl_reward_pipeline.py::test_fixed_split_is_seeded_and_disjoint -q`

Expected: FAIL because split helper does not exist.

- [ ] **Step 3: Implement fixed seeded split and fine-tuning**

Use one deterministic split persisted in fine-tuning/evaluation metadata. Load pretrained member checkpoints, train only on train rows with `lr=pretrain_lr/10`, save distinct fine-tuned member and aggregate artifacts. Never overwrite pretrained paths.

- [ ] **Step 4: Test fine-tuning artifact separation**

Assert both artifact families exist and checkpoint metadata records seed, split seed, train count, test count, and learning rate.

- [ ] **Step 5: Run focused pipeline tests**

Run: `pytest tests/test_rl_reward_pipeline.py -q`

Expected: PASS with no test-row IDs in training-row IDs.

## Task 5: Environment Safeguards

**Files:** `tests/test_rl_reward_model_env.py`, `rl/reward_model_env.py`

- [ ] **Step 1: Write failing safeguard tests**

Add tests asserting `alpha=1` returns ensemble model reward, `alpha=0` returns objective reward, ensemble reward uses mean minus standard deviation, and low coverage/high mean-opacity states receive hard penalty regardless of model output.

- [ ] **Step 2: Run tests and verify expected failure**

Run: `pytest tests/test_rl_reward_model_env.py -q`

Expected: FAIL because constructor has no alpha/ensemble/objective/penalty parameters.

- [ ] **Step 3: Implement configurable reward composition**

Accept either one model or an ensemble, default `alpha=.7`, compute objective anchor from `mass_fraction` signed by command direction, apply model mean-minus-std, and retain `automatic_mass_fraction_reward` in info. Use independent coverage and mean-opacity thresholds and a fixed negative penalty.

- [ ] **Step 4: Preserve existing cache behavior**

Keep one render at reset and one per step; pass cached previous features into model prediction. Update fake predictor tests to the new context while preserving three-render assertion.

- [ ] **Step 5: Run reward environment tests**

Run: `pytest tests/test_rl_reward_model_env.py -q`

Expected: fast tests PASS; slow real-render test remains marked `slow`.

## Task 6: Separate Evaluation Script

**Files:** `tests/test_rl_reward_pipeline.py`, `rl/eval_reward.py`

- [ ] **Step 1: Write evaluation tests**

Test source-stratified accuracy output, identical test rows for pretrained/fine-tuned models, objective/human disagreement records containing image paths, randomized head-to-head assignment, and separate verdict output path.

- [ ] **Step 2: Run tests and verify expected failure**

Run: `pytest tests/test_rl_reward_pipeline.py -q`

Expected: FAIL because evaluation helpers do not exist.

- [ ] **Step 3: Implement reward-only evaluation**

Load fixed split metadata, score both models on same held-out records, report overall and per-source accuracy, and write disagreement JSONL with target/direction, labels, objective verdict, and image paths.

- [ ] **Step 4: Implement policy/objective sanity check**

Reuse `rl.eval` episode builders and policy runners where compatible. Report RLHF policy final objective metric beside hill-climber baseline without feeding results into training.

- [ ] **Step 5: Implement blind head-to-head placeholder**

Render policy and hill-climber pairs, randomize displayed A/B using seeded RNG, collect `better`/`worse`/`tie` verdict, and append only to a separate results file. Do not call extraction or training from this path.

- [ ] **Step 6: Run evaluation tests**

Run: `pytest tests/test_rl_reward_pipeline.py -q`

Expected: PASS, including no-leakage assertions.

## Task 7: Documentation and Full Verification

**Files:** `README.md`, all changed tests/modules

- [ ] **Step 1: Document commands and result reporting**

Add commands for extraction, pretraining, fine-tuning, reward evaluation, alpha ablations, and blind head-to-head. Document web/VR canonical JSONL fields, PNG/JPEG audit-only role, artifact separation, and empty/no-branch behavior.

- [ ] **Step 2: Run targeted suite**

Run: `pytest tests/test_rl_reward_model.py tests/test_rl_reward_model_env.py tests/test_rl_extract_pairs.py tests/test_rl_reward_pipeline.py -q`

Expected: all targeted tests pass.

- [ ] **Step 3: Run full suite**

Run: `pytest -q`

Expected: full repository suite passes; slow tests remain excluded unless explicitly requested.

- [ ] **Step 4: Verify worktree and artifacts**

Run: `git status --short` and `git diff --check`. Confirm only reward/data/evaluation files, tests, docs, and approved spec/plan changes are present; confirm pretrained and fine-tuned paths are distinct in test output.
