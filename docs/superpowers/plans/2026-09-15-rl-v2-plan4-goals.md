# RL v2 Plan 4: Goals and Baselines

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn an instruction into a measurable target — a goal vector, a distance to it, and a sampler that generates the instruction mix — plus the baselines the policy will have to beat.

**Architecture:** `goals.py` owns the goal representation: it aggregates the five measured anatomical classes into the four classes an instruction can target, encodes requested changes, and scores a state by distance to the goal. `rl/baselines.py` implements the non-learned ways to satisfy an instruction (what today's command executor does, an occlusion-aware rule, random, and a hill-climber on the same objective). No environment or training yet — that is Plan 5.

**Tech Stack:** Python 3.14 (`.venv`), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md` (sections "Goals", "Environment", "Evaluation").

**Established facts this plan builds on:**
- `visibility.for_volume(name)` → `features(params)` = `{"vis": {class: float}, "bright": {class: float}, "coverage": float}` over `CLASSES = ("skeleton", "lungs", "organs", "muscle", "vessels")`, plus `solo_max(class)` and `label_source`.
- `totalseg.classes_present(name)` and `totalseg.is_contrast(name)` say what a volume contains.
- **Organs and muscle are not separable by a transfer function** (medians 48 vs 44 HU on a contrast scan, −3 vs 16 on a plain one), so goals target them jointly as `soft`; measurement keeps them apart.
- **Vessels are only separable with contrast** (136 vs 48/44 HU), so vessels are only a goal on contrast scans.
- **RL v2's starting transfer function** has peaks at −800 (lungs), 40 (soft), 300 (vessels), 900 (skeleton); peak centres are not in the action space. Lungs need a narrow peak: at −800 HU width 60 lung visibility is 0.055, at width 100 the image collapses to an opaque wall of air.
- A useless state has two shapes: nothing drawn (coverage < 0.01) and an opaque wall hiding everything (total class visibility < 0.001 at high coverage).

**Repo rules:** run from the repository root with `.venv/bin/python`; plain commit messages with **no** trailers; stage files explicitly by path; leave untracked `docs/prompts_report.pdf` and `docs/vr-integration.md` alone; branch `rl-v2`.

---

## Shared definitions

```python
# goals.py
GOAL_CLASSES = ("skeleton", "lungs", "soft", "vessels")
MEASURED_FOR_GOAL = {"skeleton": ("skeleton",), "lungs": ("lungs",),
                     "soft": ("organs", "muscle"), "vessels": ("vessels",)}

# RL v2 peak order, by the class each peak is seeded for
PEAK_CENTRES_HU = {"lungs": -800.0, "soft": 40.0, "vessels": 300.0, "skeleton": 900.0}
PEAK_INDEX = {"lungs": 0, "soft": 1, "vessels": 2, "skeleton": 3}
PEAK_WIDTHS_HU = {"lungs": 60.0, "soft": 80.0, "vessels": 80.0, "skeleton": 280.0}
PEAK_HEIGHTS = {"lungs": 0.05, "soft": 0.15, "vessels": 0.3, "skeleton": 0.6}
PEAK_COLOURS = {"lungs": (0.55, 0.70, 0.95), "soft": (0.85, 0.35, 0.35),
                "vessels": (0.90, 0.45, 0.40), "skeleton": (0.95, 0.95, 0.90)}

EPSILON = 1e-3           # floor inside log10, so "invisible" is finite
KAPPA = 1.5              # brightness weight: 0.2 brightness ≈ 0.3 log10 visibility
LAMBDA_KEEP = 0.3        # weight of "leave the unmentioned classes alone"
VISIBILITY_STRENGTH = {"slightly": 0.15, "moderately": 0.3, "strongly": 0.6}   # log10 units
BRIGHTNESS_STRENGTH = {"slightly": 0.1, "moderately": 0.2, "strongly": 0.4}
ABSOLUTE_LEVEL = {"low": 0.1, "medium": 0.4, "high": 0.8}                      # share of solo_max
```

---

### Task 1: Goal representation and distance

**Files:** create `goals.py`, `tests/test_goals.py`.

Implement:

```python
def starting_params() -> np.ndarray:
    """RL v2's starting transfer function: one peak per goal class."""
    # Builds a 24-value vector using PEAK_INDEX/PEAK_CENTRES_HU/PEAK_WIDTHS_HU/
    # PEAK_HEIGHTS/PEAK_COLOURS via transfer._from_range/_from_unit.


def aggregate(features: dict) -> dict:
    """Measured classes -> goal classes.

    {"vis": {goal class: float}, "bright": {goal class: float}, "coverage": float}
    `soft` sums the visibility of organs and muscle, because no transfer
    function can separate them; its brightness is their visibility-weighted
    mean (0 when neither is visible).
    """


def goal_vector(targets: dict) -> np.ndarray:
    """A 16-value goal: requested change and a mentioned-flag per goal class.

    targets maps goal class -> {"vis": float, "bright": float}; a missing key
    means that aspect is not part of the instruction. Layout: d[4], m[4],
    e[4], n[4] in GOAL_CLASSES order.
    """


def progress(start: dict, current: dict) -> tuple:
    """(c, b): change so far per goal class.

    c = log10(vis + EPSILON) − log10(vis_start + EPSILON); b = bright − bright_start.
    Both aggregated dicts from aggregate().
    """


def distance(goal: np.ndarray, start: dict, current: dict) -> float:
    """How far a state is from the goal.

    Σ m·|c − d| + KAPPA·Σ n·|b − e|
      + LAMBDA_KEEP·(Σ (1−m)·|c| + KAPPA·Σ (1−n)·|b|)
    """


def attainment(goal, start, final) -> float:
    """1 − distance(final) / distance(start); 1 = goal reached, < 0 = worse."""


def is_useless(features: dict) -> bool:
    """Nothing drawn, or an opaque wall hiding everything."""
    return (features["coverage"] < 0.01
            or sum(features["vis"].values()) < 0.001)
```

- [ ] **Step 1: Write failing tests** in `tests/test_goals.py`:
  - `test_starting_params_places_one_peak_per_goal_class`: decoding the 24-vector with `transfer.peak_internal` gives centres within 1 HU of `PEAK_CENTRES_HU` at the indices in `PEAK_INDEX`.
  - `test_aggregate_sums_soft_and_weights_its_brightness`: organs vis 0.2 bright 0.4 and muscle vis 0.6 bright 0.8 → soft vis 0.8, bright 0.7; skeleton passes through unchanged.
  - `test_aggregate_soft_brightness_is_zero_when_invisible`.
  - `test_goal_vector_layout`: `goal_vector({"skeleton": {"vis": 0.3}})` has `d[skeleton] == 0.3`, `m[skeleton] == 1`, every other entry 0.
  - `test_progress_is_log_change_for_visibility_and_plain_change_for_brightness`: vis 0.001 → 0.011 gives c ≈ log10(0.012/0.002) ≈ 0.778 (assert to 3 decimals); brightness 0.4 → 0.55 gives b = 0.15.
  - `test_distance_is_zero_when_the_goal_is_met_exactly`.
  - `test_distance_penalises_unmentioned_classes_drifting`: same target met, but an unmentioned class moved → larger distance; exactly `LAMBDA_KEEP * |c|` larger.
  - `test_attainment_is_one_when_reached_and_negative_when_worse`.
  - `test_is_useless_detects_empty_and_opaque_states`: coverage 0.0 → True; coverage 1.0 with all visibility 0 → True; a normal state → False.

- [ ] **Step 2:** Run `.venv/bin/python -m pytest -q tests/test_goals.py` — failures (`ModuleNotFoundError`).
- [ ] **Step 3:** Implement `goals.py` as above.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5: Real check.** On `ts_s1379`: `starting_params()` through `visibility.for_volume("ts_s1379").features(...)` then `aggregate(...)` gives non-zero visibility for skeleton, lungs and soft, and `is_useless(...)` is False. Print the four values.
- [ ] **Step 6:** Commit `goals.py tests/test_goals.py` — `feat(rl): goal representation over anatomical classes`.

---

### Task 2: Instruction sampler and text

**Files:** modify `goals.py`, `tests/test_goals.py`.

Implement:

```python
INSTRUCTION_MIX = (("relative", 0.40), ("compound", 0.25), ("show_only", 0.15),
                   ("absolute", 0.10), ("brightness", 0.10))


def goal_classes_for_volume(name: str) -> list:
    """Goal classes this volume supports: those with labels present, with
    `vessels` only on contrast scans (elsewhere vessels share intensities with
    soft tissue and no transfer function can single them out)."""


def sample_instruction(name, model, start_features, rng) -> dict:
    """One instruction: {"kind", "text", "targets", "goal"}.

    `targets` is the goal_vector input; `text` is the spoken form. Absolute
    levels use ABSOLUTE_LEVEL × model.solo_max aggregated for the goal class,
    turned into a requested change from the start state, so "high skeleton"
    means the same thing on every volume.
    """
```

Text templates (one per kind, with synonyms so the same goal has several phrasings): relative "more bone" / "increase opacity for the skeleton strongly" / "a bit less soft tissue"; compound "more bone, a bit less soft tissue"; show only "show only the skeleton" / "show only the lungs and the skeleton"; absolute "high opacity skeleton" / "low opacity soft tissue"; brightness "brighten the skeleton" / "darken the lungs".

- [ ] **Step 1: Write failing tests:**
  - `test_goal_classes_exclude_vessels_without_contrast` and `..._include_vessels_with_contrast` (monkeypatch `totalseg`).
  - `test_goal_classes_exclude_absent_classes` (a volume without lungs).
  - `test_sample_instruction_mix_matches_the_declared_shares`: 400 samples, each kind within ±0.07 of its share.
  - `test_sampled_goals_only_target_supported_classes`: over 200 samples, no goal mentions a class not in `goal_classes_for_volume`.
  - `test_show_only_hides_the_other_classes`: the goal asks for a large negative change on every class except the named one.
  - `test_absolute_level_target_is_relative_to_solo_max`: with a stub model whose `solo_max` is 0.5 and start visibility 0.05, "high" (0.8) requests `log10(0.4 + ε) − log10(0.05 + ε)`.
  - `test_every_sampled_instruction_has_text_and_a_goal_vector`: text non-empty, goal vector shape (16,), at least one mentioned flag set.
  - `test_sampling_is_deterministic_for_a_seed`.

- [ ] **Step 2–4:** Fail, implement, pass.
- [ ] **Step 5: Real check.** Sample 12 instructions on `ts_s1379` and on a non-contrast volume; print kind, text and the non-zero goal entries. Confirm by eye that the texts read naturally and no vessels goal appears on the non-contrast volume.
- [ ] **Step 6:** Commit — `feat(rl): instruction sampler with spoken forms`.

---

### Task 3: Baselines

**Files:** create `rl/baselines.py`, `tests/test_baselines.py`.

All baselines have the signature `(model, start_params, instruction) -> final params`, where `model` is a `VisibilityModel` and `instruction` is what `sample_instruction` returns.

- **B1 `current_executor`** — what today's system does with the instruction: act only on the peak of each mentioned class (`PEAK_INDEX`). Relative: one asymptotic step of `commands._asymptotic_step` with `commands.STRENGTH_WORDS[strength]` on that peak's height. Absolute: set the height to `commands.LEVEL_WORDS[level]`. Show only: named peaks to 0.7, the rest to 0.0. Brightness: the same asymptotic step on the peak's r, g, b.
- **B2 `random_policy`** — 10 random steps of ±0.1 on the 12 controllable values (heights, widths, brightness of all 4 peaks), seeded.
- **B3/B4 `hill_climb`** — coordinate search on `goals.distance`, `evaluations` budget (10 for B3, 200 for B4): repeatedly try ±step on each of the 12 values, keep improvements, halve the step when a full sweep fails.
- **B5 `occlusion_rule`** — B1, and additionally: if any class is being increased or shown, set every *other* peak's height to 0.02 (the "strip everything else" heuristic, which is what a person writes by hand).

- [ ] **Step 1: Write failing tests** with a stub model (a tiny synthetic volume) so they stay fast:
  - `test_current_executor_only_touches_the_mentioned_peak` (heights of other peaks unchanged, target peak's height increased for "more skeleton").
  - `test_current_executor_show_only_zeroes_the_others`.
  - `test_current_executor_absolute_sets_the_level`.
  - `test_current_executor_brightness_changes_colour_not_height`.
  - `test_occlusion_rule_suppresses_the_other_peaks_when_increasing`.
  - `test_occlusion_rule_matches_the_executor_when_only_decreasing` (nothing to strip).
  - `test_random_policy_is_seeded_and_stays_in_bounds` (all 24 values within [−1, 1]; same seed → same result).
  - `test_hill_climb_reduces_the_goal_distance` (distance after ≤ distance before, on a goal that is reachable).
  - `test_hill_climb_respects_its_evaluation_budget` (count calls with a counting stub).
  - `test_every_baseline_returns_valid_params` (shape 24, finite, within bounds).

- [ ] **Step 2–4:** Fail, implement, pass.
- [ ] **Step 5:** Commit `rl/baselines.py tests/test_baselines.py` — `feat(rl): non-learned baselines for instruction following`.

---

### Task 4: First baseline table on real volumes

**Files:** create `tools/baseline_report.py`, `tests/test_baseline_report.py`.

A small CLI that samples N instructions per volume, runs every baseline, and reports mean attainment per baseline and per instruction kind, writing `out/baseline_report.json`.

- [ ] **Step 1:** Write a failing test for the aggregation helper (`summarise(rows)` → mean attainment per baseline and per kind, counts, ignores `None`).
- [ ] **Step 2–4:** Fail, implement, pass.
- [ ] **Step 5: Run it** on three training volumes (one contrast, one thorax, one abdomen), 30 instructions each:
  `.venv/bin/python -m tools.baseline_report --volumes ts_s1379 ts_s1337 ts_s0454 --instructions 30`
  Print the table. **Expected shape of the result, not a gate:** B4 (200 evaluations) should attain the most, B1/B5 clearly less, B2 near zero or negative. If B1 already attains ≈ 1.0 on most instructions, say so loudly in the report — it would mean the instruction set is too easy for a learned policy to add anything, and the controller needs to know before Plan 5 trains anything.
- [ ] **Step 6:** Commit `tools/baseline_report.py tests/test_baseline_report.py` — `feat(rl): baseline attainment report`.
