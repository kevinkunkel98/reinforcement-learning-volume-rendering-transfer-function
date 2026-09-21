# Three-Mode Comparison Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer one instruction three ways at once — exact application, hill-climb search, and the trained policy — from the same start state, showing each arm's attainment, cost and per-class visibility side by side.

**Architecture:** A new `POST /api/compare` route runs three arms from the session's current step without mutating history, parsing the command and building the goal once so all three answer the same thing. The search arm calls `rl.baselines.hill_climb` — the actual B3 baseline — replacing a legacy coordinate search that was not the one the thesis measured. The panel is built in `static/`'s hand-ported shadcn idiom (vanilla CSS, no build step).

**Tech Stack:** Python 3.14, FastAPI, numpy, pytest; vanilla ES modules + hand-ported shadcn/ui CSS on the front end. Run everything against `.venv`: `.venv/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-09-20-three-mode-comparison-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `server.py` | Modify: extract `run_policy_arm`, add `compare_arms` + `/api/compare`, repoint the search branch to `hill_climb` |
| `tests/test_server.py` | Modify: tests for `run_policy_arm`, `compare_arms`, the search fix |
| `static/index.html` | Modify: the comparison panel markup inside `#viewport` |
| `static/style.css` | Modify: port Table, Progress, Skeleton, Separator; comparison layout; class hues |
| `static/app.js` | Modify: compare action, fetch, render the panel |

`server.py` is 575 lines and already mixes routes with session logic. This plan does **not** restructure it — the new logic goes in as module-level functions beside the existing `_render_step` / `_class_visibility` helpers, matching what is there.

---

## Task 1: Extract `run_policy_arm` from `Session._run_policy`

The policy arm must be callable without a `Session`. Extracting it first means the comparison provably runs the same code the single view does.

**Files:**
- Modify: `server.py:317-342` (`Session._run_policy`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`:

```python
# --- Task 1: run_policy_arm --------------------------------------------------

def test_run_policy_arm_applies_the_action_without_a_session(ts_session):
    """The policy arm is callable with its dependencies passed in, so the
    comparison endpoint can run it without constructing a Session."""
    s = ts_session
    action = np.linspace(-0.4, 0.4, len(CONTROLLABLE))
    model = s.model_for_volume(server._dataset_name)
    params = np.array(s.history[s.cursor]["params"], dtype=np.float64)
    cmd, _ = server.parse_command_with_meta("more bone", parser="rule")

    new_params, goal_text = server.run_policy_arm(model, _StubPolicy(action), cmd, params)

    for group, value in zip(CONTROLLABLE, action):
        assert float(np.mean([new_params[i] for i in group])) == pytest.approx(float(value), abs=1e-6)
    assert isinstance(goal_text, str) and goal_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_server.py::test_run_policy_arm_applies_the_action_without_a_session -v`
Expected: FAIL with `AttributeError: module 'server' has no attribute 'run_policy_arm'`

- [ ] **Step 3: Write minimal implementation**

In `server.py`, add above `class Session` (after `_render_step`):

```python
def run_policy_arm(model, policy, instruction, start_agg, start_params):
    """Run the one-shot policy on an already-built goal.

    The caller owns `instruction` (`goals.goal_from_command`) and `start_agg`,
    so a caller comparing several arms builds each exactly once -- recomputing
    them here would put another `model.features` (~17 ms) inside the policy
    arm's own timing, which is the one number the comparison panel exists to
    show. Builds the same observation layout `rl.candidates` builds for a
    standalone policy query (`rl.oneshot_env.build_observation`).

    Returns the new parameters. Raises `ValueError` when `policy` is None;
    `model.features`/`model.solo_max` may also raise `FileNotFoundError` or
    `KeyError` when the volume has no visibility cache."""
    if policy is None:
        raise ValueError(
            "no trained policy checkpoint found -- applied the command directly instead")

    solo_max_log = [math.log10(sum(model.solo_max(m) for m in goals.MEASURED_FOR_GOAL[c]) + goals.EPSILON)
                     for c in goals.GOAL_CLASSES]
    controllable = [float(np.mean([start_params[i] for i in group])) for group in CONTROLLABLE]
    observation = build_observation(instruction["goal"], model.histogram, start_agg,
                                     solo_max_log, controllable)

    action = _predict_action(policy, observation)
    return apply_controllable(start_params, action)
```

Then replace the body of `Session._run_policy` with a delegation. Mind the
argument order — `run_policy_arm(model, policy, ...)`, model first:

```python
    def _run_policy(self, cmd, current_params):
        """mode="policy": resolve the checkpoint and the volume model, build
        the goal, and hand them to `run_policy_arm`.

        The `policy is None` check stays ahead of `model_for_volume`: that call
        is not total (`visibility.for_volume` raises `FileNotFoundError` when a
        volume has no cache, which is why `_class_visibility` guards it), and
        `command()` catches only `ValueError`. Resolving the model first would
        turn a graceful "no checkpoint, applied directly" fallback into an
        unhandled exception out of the route. The `ValueError` raised here is
        what `command()` catches to fall back to exact application."""
        policy = self.policy_provider()
        if policy is None:
            raise ValueError(
                "no trained policy checkpoint found -- applied the command directly instead")
        model = self.model_for_volume(_dataset_name)
        start_agg = goals.aggregate(model.features(current_params))
        instruction = goals.goal_from_command(cmd, model, start_agg, volume=_dataset_name)
        return run_policy_arm(model, policy, instruction, start_agg, current_params), instruction["text"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server.py -v -k "policy"`
Expected: PASS — the new test plus every existing `test_policy_mode_*` test, which now exercise the delegation.

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "refactor(server): extract run_policy_arm so the policy arm needs no Session"
```

---

## Task 2: Repoint the search branch at the measured hill-climb

`Session._run_objective_search` is not the B3 baseline, and only fires for opacity commands. Both facts go.

**Files:**
- Modify: `server.py:290-315` (`Session._run_objective_search`), `server.py:360-366` (the search branch in `command()`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`:

```python
# --- Task 2: search mode is the measured hill-climb ---------------------------

def test_search_mode_searches_on_a_non_opacity_instruction(ts_session, monkeypatch):
    """The old objective search only fired for opacity increase/decrease, so
    "show only the lungs" silently fell through to exact application while the
    UI still showed search as active. It must now actually search."""
    s = ts_session
    calls = {}

    def _spy(model, start_params, instruction, evaluations=200, initial_step=0.2):
        calls["evaluations"] = evaluations
        return np.asarray(start_params, dtype=np.float64).copy()

    monkeypatch.setattr(server, "hill_climb", _spy)

    state = s.command("show only the lungs", mode="search", steps=10)

    assert calls["evaluations"] == 10
    assert state["current"]["mode"] == "search"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_server.py::test_search_mode_searches_on_a_non_opacity_instruction -v`
Expected: FAIL with `AttributeError: module 'server' has no attribute 'hill_climb'`

- [ ] **Step 3: Write minimal implementation**

In `server.py`, extend the `rl.baselines` import on line 41:

```python
from rl.baselines import CONTROLLABLE, apply_controllable, hill_climb
```

Delete the `Session._run_objective_search` method entirely (`server.py:290-315`).

Replace the search branch in `Session.command()` — the `elif` that reads
`effective_mode == "search" and cmd.get("attribute") == "opacity" and ...` —
with:

```python
        elif effective_mode == "search":
            try:
                model = self.model_for_volume(_dataset_name)
                start_agg = goals.aggregate(model.features(current_params))
                instruction = goals.goal_from_command(cmd, model, start_agg, volume=_dataset_name)
                new_params = hill_climb(model, current_params, instruction, evaluations=steps)
                actual_mode, search_flag = "search", True
            except ValueError as exc:
                message = str(exc)
                new_params = apply_command(cmd, current_params)
                actual_mode, search_flag = "exact", False
```

The `try/except ValueError` mirrors the policy branch directly above it: a
command that is not a goal (or a goal the volume cannot support) falls back to
exact application and says why, rather than 500ing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server.py -v`
Expected: PASS. If a test asserts on the old objective-search behaviour, read it
before changing it — if it pins "search leaves non-opacity commands alone", that
test encoded the defect and should be replaced by the Step 1 test; say so in the
commit message.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "fix(server): search mode now runs the hill-climb the thesis measured"
```

---

## Task 3: `compare_arms` — the three arms from one start

**Files:**
- Modify: `server.py` (add `compare_arms` beside `run_policy_arm`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`:

```python
# --- Task 3: compare_arms -----------------------------------------------------

def test_compare_arms_runs_three_arms_from_the_same_start(ts_session):
    s = ts_session
    model = s.model_for_volume(server._dataset_name)
    params = np.array(s.history[s.cursor]["params"], dtype=np.float64)
    camera = dict(s.history[s.cursor]["camera"])
    cmd, _ = server.parse_command_with_meta("more bone", parser="rule")
    start_agg = goals.aggregate(model.features(params))
    instruction = goals.goal_from_command(cmd, model, start_agg, volume=server._dataset_name)
    policy = _StubPolicy(np.zeros(len(CONTROLLABLE)))

    arms = server.compare_arms(model, policy, cmd, instruction, params, camera,
                                cheap=4, thorough=8)

    assert set(arms) == {"exact", "search_cheap", "search_thorough", "policy"}
    assert arms["exact"]["evaluations"] == 0
    assert arms["search_cheap"]["evaluations"] == 4
    assert arms["search_thorough"]["evaluations"] == 8
    assert arms["policy"]["evaluations"] == 0
    for name, arm in arms.items():
        assert arm["attainment"] is not None, name
        assert set(arm["class_visibility"]) == set(goals.GOAL_CLASSES), name
        assert arm["image_b64"] and not arm["image_b64"].startswith("data:"), name  # bare base64, as _render_step stores it
        assert arm["elapsed_ms"] >= 0, name


def test_compare_arms_marks_the_policy_arm_unavailable_without_a_checkpoint(ts_session):
    """Losing one column must not kill the demo: the other two still render."""
    s = ts_session
    model = s.model_for_volume(server._dataset_name)
    params = np.array(s.history[s.cursor]["params"], dtype=np.float64)
    camera = dict(s.history[s.cursor]["camera"])
    cmd, _ = server.parse_command_with_meta("more bone", parser="rule")
    start_agg = goals.aggregate(model.features(params))
    instruction = goals.goal_from_command(cmd, model, start_agg, volume=server._dataset_name)

    arms = server.compare_arms(model, None, cmd, instruction, params, camera,
                                cheap=4, thorough=8)

    assert arms["policy"]["unavailable"]
    assert arms["exact"]["attainment"] is not None
    assert arms["search_cheap"]["attainment"] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server.py -v -k compare_arms`
Expected: FAIL with `AttributeError: module 'server' has no attribute 'compare_arms'`

- [ ] **Step 3: Write minimal implementation**

In `server.py`, add after `run_policy_arm`:

```python
# The two search budgets the thesis reports: B3 (cheap, 10) and B4 (thorough,
# 200). Showing both is what makes a no-move B3 column legible -- beside a B4
# that did move and a policy that answered instantly, "10 evaluations buys
# little here" reads as the finding rather than as a broken column.
#
# `hill_climb` spends 1 evaluation on the start state and 1 per proposal, and a
# full coordinate sweep is 12 groups x 2 signs = 24, so B3 explores less than
# half a sweep. Measured on ct_chest / "show only the lungs": B3 130 ms, B4
# 2.7 s -- the whole panel is under three seconds.
COMPARE_BUDGET_CHEAP = 10
COMPARE_BUDGET_THOROUGH = 200


def compare_arms(model, policy, cmd, instruction, start_params, camera,
                  cheap=COMPARE_BUDGET_CHEAP, thorough=COMPARE_BUDGET_THOROUGH):
    """Answer one parsed command four ways from the same start state.

    `exact` applies the command directly (0 evaluations), `search_cheap` and
    `search_thorough` run the hill-climb the thesis measures as B3 and B4, and
    `policy` runs the trained one-shot policy (0 evaluations). Every arm is
    scored with `goals.attainment` against the *shared* start aggregate, which
    is what makes the numbers mean what the held-out table means.

    An arm that raises is reported as `unavailable` rather than failing the
    whole comparison: losing one column should not end a live demo.

    `model` and `instruction` must come from a single read of the active
    dataset, not two independent resolutions at different layers -- otherwise
    an arm could be scored against a goal built for a different volume."""
    start_agg = goals.aggregate(model.features(start_params))

    def _finish(params, evaluations, started):
        final_agg = goals.aggregate(model.features(params))
        image_b64, _img, _png = _render_image_b64(params, camera)
        return {
            "params": params.tolist(),
            "image_b64": image_b64,
            "class_visibility": _class_visibility(params),
            "attainment": float(goals.attainment(instruction["goal"], start_agg, final_agg)),
            # An arm that returned its own input is not an arm that agrees with
            # the start -- it is an arm that did not move, and the panel must
            # not let those read the same. At a 10-evaluation budget (B3) the
            # search arm can legitimately land here: one coordinate sweep is
            # 12 groups x 2 signs = 24, so 10 buys less than half a sweep.
            "unchanged": bool(np.array_equal(params, start_params)),
            "evaluations": evaluations,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    def _unavailable(exc, started):
        # `str(exc)` is not user-facing copy. A KeyError stringifies to the bare
        # repr of its key ("'lungs'"), and goal_from_command yields a dumped
        # Python dict. In a single-view toast that is merely scruffy; in a
        # column an examiner is reading closely it is unreadable.
        if isinstance(exc, ValueError):
            reason = f"not applicable -- {exc}"
        else:
            reason = "unavailable -- this volume has no visibility cache"
        return {"unavailable": reason, "attainment": None, "class_visibility": None,
                "image_b64": None, "evaluations": None, "unchanged": None,
                "elapsed_ms": int((time.perf_counter() - started) * 1000)}

    arms = {}

    started = time.perf_counter()
    try:
        arms["exact"] = _finish(apply_command(cmd, start_params), 0, started)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        arms["exact"] = _unavailable(exc, started)

    for name, budget in (("search_cheap", cheap), ("search_thorough", thorough)):
        started = time.perf_counter()
        try:
            arms[name] = _finish(
                run_search_arm(model, start_params, instruction, budget), budget, started)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            arms[name] = _unavailable(exc, started)

    started = time.perf_counter()
    try:
        # instruction and start_agg are the shared ones: recomputing them here
        # would put ~17 ms of model.features inside the policy arm's stopwatch
        # and roughly double the headline the panel exists to show.
        arms["policy"] = _finish(
            run_policy_arm(model, policy, instruction, start_agg, start_params), 0, started)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        arms["policy"] = _unavailable(exc, started)

    return arms
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server.py -v -k compare_arms`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat(server): compare_arms answers one command three ways from one start"
```

---

## Task 4: The `/api/compare` route

**Files:**
- Modify: `server.py` (route beside `@app.post("/api/command")`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`:

```python
# --- Task 4: the /api/compare route -------------------------------------------

def test_compare_route_leaves_session_history_untouched(ts_session, monkeypatch):
    """Comparing is a side quest, not a step."""
    s = ts_session
    s.policy_provider = lambda: _StubPolicy(np.zeros(len(CONTROLLABLE)))
    monkeypatch.setattr(server, "session", s)
    before_len, before_cursor = len(s.history), s.cursor

    result = asyncio.run(server.compare_route(
        server.CompareRequest(text="more bone", parser="rule", cheap=4, thorough=8)))

    assert result["applicable"] is True
    assert set(result["arms"]) == {"exact", "search_cheap", "search_thorough", "policy"}
    assert len(s.history) == before_len
    assert s.cursor == before_cursor


def test_compare_route_reports_a_camera_command_as_not_applicable(ts_session, monkeypatch):
    s = ts_session
    s.policy_provider = lambda: _StubPolicy(np.zeros(len(CONTROLLABLE)))
    monkeypatch.setattr(server, "session", s)

    result = asyncio.run(server.compare_route(
        server.CompareRequest(text="rotate right", parser="rule", cheap=4, thorough=8)))

    assert result["applicable"] is False
    assert result["reason"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server.py -v -k compare_route`
Expected: FAIL with `AttributeError: module 'server' has no attribute 'CompareRequest'`

- [ ] **Step 3: Write minimal implementation**

In `server.py`, beside the other `BaseModel` request classes:

```python
class CompareRequest(BaseModel):
    text: str
    parser: str = "rule"
    model: str = "qwen2.5:7b"
    cheap: int = COMPARE_BUDGET_CHEAP
    thorough: int = COMPARE_BUDGET_THOROUGH
```

And beside `@app.post("/api/command")`:

```python
@app.post("/api/compare")
async def compare_route(req: CompareRequest):
    """Answer one instruction three ways without advancing the session.

    Parses once and builds the goal once, so all three arms answer the
    identical parsed command: a bad parse then makes all three wrong together
    and the panel shows a parsing problem, not a policy problem."""
    try:
        cmd, parser_meta = parse_command_with_meta(req.text, parser=req.parser, model=req.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    start_params = np.array(session.history[session.cursor]["params"], dtype=np.float64)
    camera = dict(session.history[session.cursor].get("camera", DEFAULT_CAMERA))
    volume_model = session.model_for_volume(_dataset_name)

    try:
        start_agg = goals.aggregate(volume_model.features(start_params))
        instruction = goals.goal_from_command(cmd, volume_model, start_agg, volume=_dataset_name)
    except ValueError as exc:
        # Camera, reset, width and centre commands are not goals, and neither
        # is a goal this volume cannot support (vessels on a non-contrast scan).
        return {"applicable": False, "reason": str(exc), "text": req.text, **parser_meta}

    arms = compare_arms(volume_model, session.policy_provider(), cmd, instruction,
                         start_params, camera, cheap=req.cheap, thorough=req.thorough)
    return {"applicable": True, "text": req.text, "goal_text": instruction["text"],
            "budgets": {"cheap": req.cheap, "thorough": req.thorough}, "arms": arms,
            "start": {"class_visibility": _class_visibility(start_params)},
            **parser_meta}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server.py -v -k compare_route`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat(server): POST /api/compare, three arms without advancing history"
```

---

## Task 5: Port the CSS components

**Files:**
- Modify: `static/style.css`

- [ ] **Step 1: Add the components and the class hues**

Append to `static/style.css`. The hues are the dataviz reference palette's dark
steps, validated against this sheet's `--background` (`240 10% 3.9%` = `#09090b`):
lightness band, chroma floor, CVD separation (worst adjacent ΔE 8.4 protan),
normal-vision floor (19.8) and contrast all PASS.

```css
/* ==========================================================================
   Comparison panel
   ========================================================================== */

/* Goal-class hues: categorical identity, assigned in fixed order and never
   cycled. Validated against this sheet's --background (#09090b). Every bar is
   also direct-labelled with its class name, so identity never rests on colour
   alone. */
:root {
  --class-skeleton: #3987e5;
  --class-lungs:    #d95926;
  --class-soft:     #199e70;
  --class-vessels:  #c98500;
}

/* shadcn Separator */
.separator { height: 1px; background: hsl(var(--border)); border: 0; margin: 0; }

/* shadcn Skeleton: the search arm genuinely takes seconds, and this is what
   that gap is for -- it has to read as "working", not "broken". */
.skeleton {
  background: hsl(var(--muted));
  border-radius: calc(var(--radius) - 2px);
  animation: skeleton-pulse 1.6s ease-in-out infinite;
}
@keyframes skeleton-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }
@media (prefers-reduced-motion: reduce) { .skeleton { animation: none; } }

/* shadcn Progress, used as the per-class visibility bar. The tick marks the
   class's value in the start state, so "did the mentioned class move and did
   the others stay put" is one glance. */
.progress {
  position: relative; height: 8px; width: 100%;
  background: hsl(var(--secondary)); border-radius: 9999px; overflow: hidden;
}
.progress-fill {
  height: 100%; border-radius: 9999px;
  background: var(--progress-color, hsl(var(--primary)));
  transition: width 240ms ease;
}
.progress-tick {
  position: absolute; top: -2px; width: 2px; height: 12px;
  background: hsl(var(--muted-foreground)); border-radius: 1px;
}

/* shadcn Table, for the per-class rows */
.table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.table td { padding: 0.3rem 0; vertical-align: middle; }
.table .table-label { color: hsl(var(--muted-foreground)); white-space: nowrap; padding-right: 0.6rem; }
.table .table-value { font-family: var(--mono); text-align: right; padding-left: 0.6rem; white-space: nowrap; }

/* Layout */
#compare-view { display: flex; flex-direction: column; gap: 0.75rem; height: 100%; }
#compare-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
#compare-goal { color: hsl(var(--muted-foreground)); font-size: 0.85rem; }
#compare-columns { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
@media (max-width: 900px) { #compare-columns { grid-template-columns: 1fr; } }

.compare-arm { display: flex; flex-direction: column; gap: 0.6rem; padding: 0.75rem; }
.compare-arm-name { font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; color: hsl(var(--muted-foreground)); }
.compare-arm-image { width: 100%; aspect-ratio: 1; object-fit: contain; background: #000; border-radius: calc(var(--radius) - 2px); }

/* Hero number: attainment is one headline per arm, so it is a number, not a
   chart. The sign is printed, so the colour is never carrying it alone. */
.compare-attainment { font-family: var(--mono); font-size: 1.6rem; line-height: 1.1; }
.compare-attainment[data-sign="positive"] { color: hsl(var(--success)); }
.compare-attainment[data-sign="negative"] { color: hsl(var(--destructive)); }
.compare-attainment-label { font-size: 0.7rem; color: hsl(var(--muted-foreground)); }
.compare-cost { font-size: 0.75rem; color: hsl(var(--muted-foreground)); font-family: var(--mono); }
.compare-unavailable { font-size: 0.8rem; color: hsl(var(--muted-foreground)); padding: 1rem 0; }
```

- [ ] **Step 2: Verify the sheet still parses**

Run: `.venv/bin/python -c "print(open('static/style.css').read().count('{') == open('static/style.css').read().count('}'))"`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add static/style.css
git commit -m "feat(ui): port Table, Progress, Skeleton, Separator for the comparison panel"
```

---

## Task 6: The panel markup

**Files:**
- Modify: `static/index.html:47-70` (inside `#viewport`, as a sibling of `#single-view`)

- [ ] **Step 1: Add the markup**

In `static/index.html`, immediately after the closing `</div>` of `#single-view`
and still inside `#viewport`, add:

```html
          <div id="compare-view" hidden>
            <div id="compare-header">
              <span id="compare-goal">—</span>
              <button id="compare-close" class="btn btn-secondary btn-sm" data-tooltip="Back to the single view">Close</button>
            </div>
            <div id="compare-columns"></div>
          </div>
```

Then add a compare button to the toolbar, after the `#policy-toggle-btn` line:

```html
            <button id="compare-btn" class="btn btn-outline btn-sm" data-tooltip="Answer three ways and compare">compare</button>
```

- [ ] **Step 2: Verify the page still loads**

Run: `.venv/bin/python -c "
import re
html = open('static/index.html').read()
for el in ('compare-view', 'compare-columns', 'compare-close', 'compare-btn'):
    assert f'id=\"{el}\"' in html, el
print('markup ok')
"`
Expected: `markup ok`

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat(ui): comparison panel markup and toolbar action"
```

---

## Task 7: Wire the panel up

**Files:**
- Modify: `static/app.js`

- [ ] **Step 1: Add the client logic**

Append to `static/app.js`:

```javascript
// --- three-mode comparison ---------------------------------------------------
// Answers the instruction three ways from the current step without advancing
// the session, so the amortisation claim -- the policy reaching search's
// neighbourhood at zero evaluations -- is on one screen instead of spread
// across three interactions.

const CLASS_ORDER = ["skeleton", "lungs", "soft", "vessels"];
const CLASS_LABEL = { skeleton: "skeleton", lungs: "lungs", soft: "soft tissue", vessels: "vessels" };
const ARM_ORDER = ["exact", "search", "policy"];

const compareView = el("compare-view");
const compareColumns = el("compare-columns");
const compareGoal = el("compare-goal");

function showCompare(on) {
  el("single-view").hidden = on;
  compareView.hidden = !on;
}

function armSkeleton(name) {
  return `<div class="card compare-arm">
    <span class="compare-arm-name">${name}</span>
    <div class="skeleton compare-arm-image"></div>
    <div class="skeleton" style="height:1.6rem;width:5rem"></div>
    <div class="skeleton" style="height:0.75rem;width:7rem"></div>
  </div>`;
}

function classBars(visibility, start) {
  return `<table class="table">${CLASS_ORDER.map((c) => {
    const value = visibility && visibility[c] != null ? visibility[c] : 0;
    const from = start && start[c] != null ? start[c] : 0;
    const pct = (v) => `${Math.min(100, Math.max(0, v * 100)).toFixed(1)}%`;
    return `<tr>
      <td class="table-label">${CLASS_LABEL[c]}</td>
      <td style="width:100%">
        <div class="progress">
          <div class="progress-fill" style="width:${pct(value)};--progress-color:var(--class-${c})"></div>
          <div class="progress-tick" style="left:${pct(from)}"></div>
        </div>
      </td>
      <td class="table-value">${(value * 100).toFixed(1)}%</td>
    </tr>`;
  }).join("")}</table>`;
}

function armCard(name, arm, start) {
  if (!arm || arm.unavailable) {
    return `<div class="card compare-arm">
      <span class="compare-arm-name">${name}</span>
      <p class="compare-unavailable">${arm ? arm.unavailable : "no result"}</p>
    </div>`;
  }
  const sign = arm.attainment >= 0 ? "positive" : "negative";
  const shown = `${arm.attainment >= 0 ? "+" : "−"}${Math.abs(arm.attainment).toFixed(3)}`;
  return `<div class="card compare-arm">
    <span class="compare-arm-name">${name}</span>
    <img class="compare-arm-image" src="data:image/png;base64,${arm.image_b64}" alt="${name} result" />
    <div>
      <div class="compare-attainment" data-sign="${sign}">${shown}</div>
      <div class="compare-attainment-label">attainment</div>
    </div>
    <div class="compare-cost">${arm.evaluations} eval${arm.evaluations === 1 ? "" : "s"} · ${arm.elapsed_ms} ms</div>
    <hr class="separator" />
    ${classBars(arm.class_visibility, start)}
  </div>`;
}

async function runCompare() {
  const text = textInput.value.trim();
  if (!text) return;
  showCompare(true);
  compareGoal.textContent = `"${text}"`;
  compareColumns.innerHTML = ARM_ORDER.map(armSkeleton).join("");

  let payload;
  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, parser: config.parser, budget: Number(el("steps-input").value) || 10 }),
    });
    payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "compare failed");
  } catch (err) {
    compareColumns.innerHTML = `<p class="compare-unavailable">${err.message}</p>`;
    return;
  }

  if (!payload.applicable) {
    compareGoal.textContent = `"${text}"`;
    compareColumns.innerHTML = `<p class="compare-unavailable">${payload.reason} — the comparison is for visibility instructions.</p>`;
    return;
  }

  compareGoal.textContent = `"${text}" → ${payload.goal_text}`;
  const start = payload.start ? payload.start.class_visibility : null;
  compareColumns.innerHTML = ARM_ORDER.map((name) => armCard(name, payload.arms[name], start)).join("");
}

el("compare-btn").addEventListener("click", runCompare);
el("compare-close").addEventListener("click", () => showCompare(false));
```

- [ ] **Step 2: Check it against the running server by hand**

Start the server and exercise all four paths:

```bash
.venv/bin/python server.py
```

Then at `http://127.0.0.1:8000`, type each instruction and press **compare**:

| Instruction | Expected |
|---|---|
| `show only the lungs` | three columns, search slowest, lungs bar highest in the search/policy columns |
| `rotate right` | "not applicable — the comparison is for visibility instructions" |
| `more bone` | three columns; skeleton bar moves, other ticks stay near their bars |
| (empty input) | nothing happens |

- [ ] **Step 3: Commit**

```bash
git add static/app.js
git commit -m "feat(ui): wire the three-mode comparison panel"
```

---

## Task 8 (vetoable): Delete `evaluate.objective`

After Task 2 it has no caller. **Confirm with the user before doing this task.**

**Files:**
- Modify: `evaluate.py` (delete `objective`), `server.py:36` (drop it from the import), `search.py` if `propose_step`/`resize_step` are now unused too
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Confirm it is genuinely dead**

Run: `grep -rn "objective\|propose_step\|resize_step" --include="*.py" . | grep -v "^./tests/" | grep -v "goals.py"`
Expected: no hits in production code outside `evaluate.py`/`search.py` themselves. If there are hits, stop and report them rather than deleting.

- [ ] **Step 2: Delete and adjust the import**

Delete the `objective` function from `evaluate.py`. In `server.py:36` change:

```python
from evaluate import jsonl_append, objective
```

to:

```python
from evaluate import jsonl_append
```

If Step 1 showed `propose_step`/`resize_step` also have no remaining callers,
delete them from `search.py` and drop `server.py:43` entirely.

- [ ] **Step 3: Delete the tests that covered it**

Remove the `objective` tests from `tests/test_evaluate.py`. Deleting a function
means deleting its tests; leaving them to fail is not a signal.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass, with a lower total than before.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py server.py search.py tests/test_evaluate.py
git commit -m "refactor: delete evaluate.objective, now that search uses the measured hill-climb"
```

---

## Task 9: Update the docs

**Files:**
- Modify: `README.md:210-225` (the answering-modes section), `docs/REPRODUCE.md` (section 6)

- [ ] **Step 1: Describe the comparison in the README**

In `README.md`, after the paragraph describing the three answering modes, add:

```markdown
**compare** answers the same instruction all three ways at once, from the same
start state, and reports each arm's attainment, evaluation count and wall clock
next to its render. The search arm is `rl.baselines.hill_climb` — the same B3
baseline the held-out table reports — and attainment is `goals.attainment`, so
the panel's numbers mean exactly what the thesis's numbers mean.
```

- [ ] **Step 2: Note it in REPRODUCE.md**

In `docs/REPRODUCE.md`'s section 6, after the answering-modes paragraph, add:

```markdown
The **compare** button answers one instruction with all three and shows them
side by side; the search arm's evaluation budget comes from the steps input.
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/REPRODUCE.md
git commit -m "docs: the three-mode comparison panel"
```

---

## Self-review notes

**Spec coverage:** endpoint (Task 4), arms and their reported fields (Task 3),
factoring of `run_policy_arm` (Task 1), the search fix (Task 2), the
`evaluate.objective` knock-on as a vetoable step (Task 8), CSS components and
validated hues (Task 5), forms — hero number, secondary cost stat, per-class
bars with start ticks (Tasks 5 and 7), layout replacing `#single-view` (Tasks 6
and 7), every error-handling row (Tasks 3, 4, 7), and the testing list (Tasks
1–4). The spec's stacking-below-900px note is covered by the media query in
Task 5.

**Known gap, deliberate:** the spec's testing list includes "the session's
history and cursor are unchanged after a compare call" — that is
`test_compare_route_leaves_session_history_untouched` in Task 4, asserted on
both `len(history)` and `cursor`.

**Signature consistency:** `run_policy_arm(model, policy, instruction, start_agg, start_params)` and
`compare_arms(model, policy, cmd, instruction, start_params, camera, budget)`
are used with that argument order in Tasks 1, 3 and 4 and in every test.
