# RL v2 Plan 7: Preference Collection Page

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A page where a rater sees an instruction and two candidate results, picks the better one, and produces the preference data the reward model will be trained on.

**Architecture:** `rl/candidates.py` builds an item — a volume, a start transfer function, an instruction and two candidates from different sources. `collect.py` is a FastAPI router mounted by `server.py`: it serves the page, hands out items with rendered images, and appends judgments to one canonical JSONL. `static/collect.html` + `static/collect.js` show each candidate as the 6 standard views in a grid, so the rater judges exactly what the reward model will later see.

**Tech Stack:** Python 3.14 (`.venv`), FastAPI, VTK, Pillow, vanilla JS, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md`, section "Collection (`/collect`)" — with two deliberate deviations recorded below.

**Deviations from the spec, decided 2026-09-16:**
- **Images, not linked 3D viewports.** Each candidate is shown as the 6 standard views (`views.cameras_for_volume`) rendered server-side into one grid image. The rater then judges exactly the input the reward model receives, and the page is a half-day rather than a day. Interactive 3D moves to the month.
- **Classes and instructions come from `goals.sample_instruction`** (skeleton, lungs, soft, vessels), not the old intensity tissues.

**Established facts:**
- `goals.py`: `starting_params()`, `sample_instruction(volume, model, start_features, rng)` → `{"kind", "text", "targets", "goal"}`, `aggregate`, `attainment`, `distance`, `is_useless`.
- `visibility.for_volume(name)`, `datasets.volumes_for_split(...)`, `views.cameras_for_volume(volume, spacing)`, `render.render(volume, params, spacing, camera)` (renderer-neutral cameras).
- `rl/baselines.py`: `BASELINES` (B0…B5), `CONTROLLABLE`.
- A trained one-shot policy lives at `out/rl_v2/oneshot_seed0/best.zip` (or a newer run); `rl/oneshot_env.py` defines its observation.
- Rendering one view at 224 px takes ~30–60 ms, so six views per candidate is ~0.3 s; an item (two candidates plus the start state) is ~1 s.

**Repo rules:** run from the repository root with `.venv/bin/python`; plain commit messages with **no** trailers; stage files explicitly by path; nothing under `out/` is committed; leave untracked `docs/prompts_report.pdf` and `docs/vr-integration.md` alone; branch `rl-v2`.

---

### Task 1: Item sampling

**Files:** create `rl/candidates.py`, `tests/test_candidates.py`.

```python
SOURCES = ("policy", "policy", "B1_current_executor", "B3_hill_climb_10", "B5_occlusion_rule", "perturbation")
MIN_VISIBILITY_DIFFERENCE = 0.1     # log10 units, per class
MIN_BRIGHTNESS_DIFFERENCE = 0.05
```

- `sample_item(volume, model, rng, policy=None)` → `{"volume", "start_params", "instruction", "a": {"params", "source"}, "b": {"params", "source"}, "objective_choice", "features": {...}}`.
- Half the items pair two stochastic samples of the policy against each other (`deterministic=False`); the other half pair one policy sample against one of `B1`, `B3`, `B5` or a random perturbation of the start (uniform ±0.3 per controllable group), chosen uniformly.
- Regenerate the pair while the two candidates differ by less than `MIN_VISIBILITY_DIFFERENCE` in every class's `log10(vis + EPSILON)` **and** less than `MIN_BRIGHTNESS_DIFFERENCE` in every class's brightness (up to 10 attempts; then accept and mark `"near_duplicate": True`).
- `objective_choice` is `"a"` or `"b"`, whichever has the lower `goals.distance` — recorded, never shown to the rater.
- When `policy` is `None` (no trained policy available), fall back to pairs drawn from the non-policy sources so the page still works.

- [ ] **Step 1: Write failing tests** with a stub model and a stub policy: both candidates differ from the start; sources are drawn from `SOURCES` and the two sources of one item are never the same object twice unless both are `"policy"`; the near-duplicate filter rejects an identical pair (inject a policy that always returns the same parameters and assert `near_duplicate` after the attempt limit); `objective_choice` matches the lower distance; sampling is deterministic for a seed; with `policy=None` no item has a `"policy"` source.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit — `feat(rl): sample preference items from policy and baselines`.

---

### Task 2: Rendering the views

**Files:** create `collect_images.py`, `tests/test_collect_images.py`.

- `view_grid(volume_id, params, size=224, columns=3)` → a PNG (bytes) of the 6 standard views laid out in a grid, rendered with `render.render` through `views.cameras_for_volume`.
- Cache on disk under `out/cache/collect_images/<sha256 of volume version, params, size>.png`; return the cached bytes when present. Write atomically (`.tmp` then `os.replace`).
- `grid_data_url(...)` returns a `data:image/png;base64,…` string for embedding in the JSON response.

- [ ] **Step 1: Write failing tests** with a stub renderer: the grid has the expected pixel size (3 columns × 2 rows of `size`); the cache is written once and reused on the second call (count renderer calls); a different transfer function produces a different cache key; no `.tmp` file remains.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit — `feat(rl): render the six standard views as a grid`.

---

### Task 3: Routes and storage

**Files:** create `collect.py`, `tests/test_collect.py`; modify `server.py` (mount the router).

- `GET /collect` → `static/collect.html`.
- `POST /api/collect/next` with `{"rater_id"}` → `{"pair_id", "text", "kind", "volume", "a_image", "b_image", "start_image", "repeat_of"}`. The two candidates are shuffled per item so `a` is not always the policy; the mapping from displayed side to source is kept server-side in the pending-item store, never sent to the browser.
- `POST /api/collect/judge` with `{"pair_id", "choice" ("a"|"b"|"equal"|"skip"), "decision_ms"}` → appends one row to `out/vis_preferences.jsonl` and returns the next item.
- **Repeats:** 10% of items are a previously judged item by the same rater, at least 20 items later, with the displayed sides swapped; the row records `repeat_of`.
- Row format (the file training reads directly):

```json
{"pair_id": "...", "timestamp": "...", "rater_id": "kk", "volume": "ts_s0123",
 "volume_version": "sha256:...", "instruction": {"kind": "relative", "text": "more bone",
 "goal": [16 floats], "targets": {}}, "start_params": [24 floats],
 "a": {"params": [24 floats], "source": "policy"}, "b": {"params": [24 floats], "source": "B3_hill_climb_10"},
 "choice": "a", "objective_choice": "b", "near_duplicate": false,
 "features": {"start": {}, "a": {}, "b": {}}, "decision_ms": 4200, "repeat_of": null}
```

- Writes are append-only under a lock (reuse the pattern in `server.py`'s scene log).

- [ ] **Step 1: Write failing tests** driving the router functions directly (not through TestClient — `tests/test_server.py` explains why: VTK must render on the main thread): `next` returns an item whose images are data URLs and whose response contains no source labels; `judge` appends exactly one well-formed row and returns the next item; an unknown `pair_id` raises; `choice` is validated; the repeat scheduler returns a previously seen item with the sides swapped after the gap; rows land in the configured path.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit — `feat(rl): collect preference judgments`.

---

### Task 4: The page

**Files:** create `static/collect.html`, `static/collect.js`; extend `static/style.css`.

- Header: the instruction in large text, the volume name, a progress counter ("judged: 37"), and the rater ID (prompted once, kept in `localStorage`).
- Body: two grids side by side, labelled **A** and **B**, each 3×2 views. A "show start" toggle reveals the start-state grid beneath them.
- Buttons and keys: **A** / **B** / **E** (equally good) / **S** (skip). Keys work without focus; buttons show the same letters.
- After each judgment the next item loads immediately; decision time is measured from when the images finish loading.
- No source labels, no objective hint, nothing that reveals which candidate came from where.

- [ ] **Step 1:** Implement, then verify by hand: start the server, open `/collect`, judge three items, confirm `out/vis_preferences.jsonl` has three well-formed rows with plausible `decision_ms`, and that reloading mid-session does not lose the rater ID.
- [ ] **Step 2:** `node --check static/collect.js`; run the full test suite.
- [ ] **Step 3:** Commit — `feat(ui): preference collection page`.

---

### Task 5: Documentation and a collection run

- [ ] **Step 1:** README section: how to start collecting (`python server.py`, open `/collect`), what a judgment means ("which result better follows the instruction, while still looking useful"), the repeat mechanism and why (it measures the rater's self-consistency, the ceiling for any reward model), and where the data lands.
- [ ] **Step 2:** Commit — `docs: describe preference collection`.
- [ ] **Step 3 (the author's, not an agent's):** collect ≥ 300 decisive judgments.
