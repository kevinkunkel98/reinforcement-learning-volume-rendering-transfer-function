# RL v2 Phase 0: Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the old RL pipeline, camera RL, the old preference-feedback paths, and junk files, leaving a working chat UI (commands, voice, rule/LLM parsing, objective hill-climb search, history, 3D viewer) as the base for RL v2.

**Architecture:** Pure deletion and simplification. The UI stops using the judge/thumbs/policy features first, so every commit leaves a working app; then the server code and tests for those features are removed; then the now-unreferenced modules, tests, plots, scripts and docs.

**Tech Stack:** Python 3.14 (`.venv`), FastAPI, pytest, vanilla JS.

**Spec:** `docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md`, section "Phase 0: Cleanup".

**Repo rules:**
- Run everything from the repository root with `.venv/bin/python`.
- Commit messages: plain `git commit -m "..."`, **no** `Co-Authored-By` or other Claude trailers.
- Do not touch the user's uncommitted changes in `datasets.py`, `.gitignore`, `.dockerignore`, `docs/prompts_report.pdf`, `docs/vr-integration.md`. Stage files explicitly by path; never `git add -A` or `git add .`.
- Work happens on branch `rl-v2`.

---

## File map

| File | Change |
|---|---|
| `static/index.html` | Remove judge view and evaluator toggle group |
| `static/app.js` | Remove pending/judge logic, evaluator config, verdict tags, thumbs feedback |
| `static/style.css` | Remove judge, verdict-tag, thumbs and small-toggle rules |
| `server.py` | Remove `rl.serve` import, policy + human evaluators, judge state and route, feedback method and route, `PREF_PATH`, `FEEDBACK_PATH`, `verdict`/`feedback` step fields |
| `tests/test_server.py` | Remove tests of removed features; drop `evaluator=` from kept tests |
| `rl/*.py` except `rl/__init__.py` | Delete |
| `tests/test_camera_env.py`, `tests/test_camera_eval.py`, `tests/test_rl_*.py` | Delete |
| `plots/online_eval_curve.py`, `plots/reward_model_eval_curve.py`, `tests/test_plots_online_eval_curve.py` | Delete |
| `mvp.py`, `stats.py`, `.$architecture-highlevel.drawio.bkp`, `.$architecture-mvp.drawio.bkp` | Delete |
| `evaluate.py` | Remove `human()`, update docstring |
| `docs/superpowers/{specs,plans}/…` | Delete docs of removed features |
| `docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md` | Keep `training-curve-plots` docs (plotter stays) |
| `README.md` | Rewrite intro, Quick Start, RL sections, tests, layout |

`plots/training_curves.py`, `plots/style.py`, `plots/read_progress.py` and their tests stay (generic SB3 `progress.csv` plotting, reused by RL v2).

---

### Task 1: UI stops using judge, policy evaluator and thumbs feedback

**Files:**
- Modify: `static/index.html:57-67` (judge view), `static/index.html:94-101` (search options)
- Modify: `static/app.js` (several blocks, listed below)
- Modify: `static/style.css:383-407`, `static/style.css:514-541`, `static/style.css:629-634`

- [ ] **Step 1: Remove the judge view from `static/index.html`**

Delete this block (lines 57–67):

```html
          <div id="judge-view" hidden>
            <div class="pair">
              <div><p>Before</p><img id="before-image" alt="before" /></div>
              <div><p>After</p><img id="after-image" alt="after" /></div>
            </div>
            <p id="judge-progress"></p>
            <div id="judge-buttons">
              <button id="worse-btn" class="btn btn-destructive">Worse</button>
              <button id="better-btn" class="btn btn-success">Better</button>
            </div>
          </div>
```

- [ ] **Step 2: Remove the evaluator toggle group from `static/index.html`**

Replace:

```html
            <div id="search-options" hidden>
              <div id="evaluator-toggle-group" class="toggle-group toggle-group-sm" data-toggle-group="evaluator" role="radiogroup" aria-label="Search evaluator">
                <button type="button" class="toggle-item toggle-item-sm" data-value="objective" data-state="on">objective</button>
                <button type="button" class="toggle-item toggle-item-sm" data-value="human" data-state="off">human</button>
                <button type="button" class="toggle-item toggle-item-sm" data-value="policy" data-state="off">policy</button>
              </div>
              <input id="steps-input" type="number" value="10" min="1" max="50" aria-label="Search steps" />
            </div>
```

with:

```html
            <div id="search-options" hidden>
              <input id="steps-input" type="number" value="10" min="1" max="50" aria-label="Search steps" />
            </div>
```

- [ ] **Step 3: Simplify state, config and element handles in `static/app.js`**

Replace lines 1–9:

```js
const state = { current: null, cursor: 0, total: 1, pending: null, dataset: null };
const config = { parser: "rule", search: false, evaluator: "objective" };

const THUMB_UP_PATH = "M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3";
const THUMB_DOWN_PATH = "M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17";

const el = (id) => document.getElementById(id);
const singleView = el("single-view");
const judgeView = el("judge-view");
```

with:

```js
const state = { current: null, cursor: 0, total: 1, dataset: null };
const config = { parser: "rule", search: false };

const el = (id) => document.getElementById(id);
```

- [ ] **Step 4: Remove the verdict branch from `postSceneTransition` in `static/app.js`**

Delete these lines inside `postSceneTransition`:

```js
  if (metadata.verdict) {
    after.scene_id = `${after.scene_id}:feedback:${metadata.verdict}`;
  }
```

- [ ] **Step 5: Replace `refresh` in `static/app.js`**

Replace the whole `async function refresh(data) { … }` with:

```js
async function refresh(data) {
  state.current = data.current;
  state.cursor = data.cursor;
  state.total = data.total;
  if (data.dataset && data.dataset !== state.dataset) {
    setSelectValue(data.dataset);
  }

  el("step-counter").textContent = `${state.cursor + 1} / ${state.total}`;
  el("back-btn").disabled = state.cursor <= 0;
  el("forward-btn").disabled = state.cursor >= state.total - 1;

  el("current-image").src = `data:image/png;base64,${state.current.image_b64}`;
  if (window.volumeViewer && state.dataset && state.current) {
    await window.volumeViewer.load(state.dataset, state.current.params, state.current.camera);
  }

  updateTelemetry(state.current.masses);
  return data;
}
```

- [ ] **Step 6: Replace `appendMessage` and delete `buildFeedbackRow` in `static/app.js`**

Replace `function appendMessage(step) { … }` and the following `function buildFeedbackRow(step) { … }` (both whole functions) with:

```js
function appendMessage(step) {
  if (emptyState.parentNode === messagesEl) messagesEl.removeChild(emptyState);

  const div = document.createElement("div");
  div.className = "msg";

  const text = document.createElement("div");
  text.className = "msg-text";
  text.textContent = step.cmd_text;
  div.appendChild(text);

  if (step.search) {
    const tags = document.createElement("div");
    tags.className = "msg-tags";
    const t = document.createElement("span");
    t.className = "tag";
    t.textContent = "search";
    tags.appendChild(t);
    div.appendChild(tags);
  }

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
```

- [ ] **Step 7: Replace `sendCommand`, delete `judge` and its listeners in `static/app.js`**

Replace `async function sendCommand(text) { … }` and the following `async function judge(verdict) { … }` with:

```js
async function sendCommand(text) {
  const body = {
    text,
    parser: config.parser,
    search: config.search,
    steps: parseInt(el("steps-input").value, 10),
  };
  const r = await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (r.status !== 200) {
    const err = await r.json();
    showToast(`Could not parse that: ${err.detail}`, "destructive");
    return;
  }
  const data = await r.json();
  await refresh(data);
  await postSceneTransition(data);
  appendMessage(data.current);
}
```

Then delete these two lines:

```js
el("better-btn").addEventListener("click", () => judge("better"));
el("worse-btn").addEventListener("click", () => judge("worse"));
```

- [ ] **Step 8: Remove dead CSS in `static/style.css`**

Delete the `.pair`, `.pair p`, `#judge-progress` and `#judge-buttons` rules (lines 383–407):

```css
.pair {
  display: flex;
  gap: 16px;
}

.pair p {
  text-align: center;
  margin: 0 0 6px;
  font-size: 12px;
  font-weight: 500;
  color: hsl(var(--muted-foreground));
}

#judge-progress {
  font-family: var(--mono);
  font-size: 12px;
  color: hsl(var(--muted-foreground));
  font-variant-numeric: tabular-nums;
}

#judge-buttons {
  display: flex;
  gap: 10px;
  margin-top: 4px;
}
```

Delete the verdict-tag and thumbs rules:

```css
.tag.good { background: transparent; border-color: hsl(var(--success) / 0.4); color: hsl(var(--success)); }
.tag.bad { background: transparent; border-color: hsl(var(--destructive) / 0.4); color: hsl(var(--destructive)); }

.msg-feedback {
  display: flex;
  gap: 4px;
  align-self: flex-end;
}

.feedback-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: var(--radius);
  border: none;
  background: transparent;
  color: hsl(var(--muted-foreground) / 0.7);
  transition: background-color 0.15s ease, color 0.15s ease;
}

.feedback-btn svg { width: 13px; height: 13px; }

.feedback-btn:hover { background: hsl(var(--accent)); color: hsl(var(--foreground)); }

.feedback-btn.active.up { color: hsl(var(--success)); }
.feedback-btn.active.down { color: hsl(var(--destructive)); }
```

Delete the small-toggle rule:

```css
.toggle-group-sm .toggle-item,
.toggle-item-sm {
  height: 26px;
  padding: 0 10px;
  font-size: 11.5px;
}
```

Keep `.btn-success`, `.btn-destructive` and the `--success` tokens (generic button variants, reused by the collect page later).

- [ ] **Step 9: Verify no dangling references and valid JS**

Run:

```bash
grep -nE "judge|pending|evaluator|feedback|THUMB|before-image|after-image|better-btn|worse-btn|singleView|toggle-group-sm|toggle-item-sm" static/app.js static/index.html static/style.css
node --check static/app.js && echo JS_OK
```

Expected: exactly one grep hit, the design-token comment near the top of `static/style.css` ("…needs for "better"/good-feedback states…"), which stays because `--success` is still used by `.btn-success`; then `JS_OK`.

- [ ] **Step 10: Run the static contract tests**

Run: `.venv/bin/python -m pytest -q tests/test_static_viewer_contract.py`
Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add static/index.html static/app.js static/style.css
git commit -m "refactor(ui): remove judge view, policy evaluator and thumbs feedback"
```

---

### Task 2: Server drops the policy evaluator and the human judge flow

**Files:**
- Modify: `server.py`
- Modify: `tests/test_server.py`

- [ ] **Step 1: Update the tests first**

In `tests/test_server.py`, delete these test functions entirely:

- `test_policy_search_raises_when_no_trained_model`
- `test_policy_search_appends_one_final_step`
- `test_human_search_returns_pending_and_judge_advances_it`
- `test_human_search_converges_and_appends_final_step`
- `test_judge_without_pending_raises`
- `test_preferences_logged_with_both_verdicts`
- `test_human_search_pair_images_saved_and_referenced_in_preferences`
- `test_camera_persists_through_human_search_convergence`
- `test_camera_change_during_pending_judgment_does_not_corrupt_final_camera`

Replace `test_objective_search_appends_one_final_step` with:

```python
def test_objective_search_appends_one_final_step():
    s = _fresh_session()
    state = s.command("increase opacity for bone strongly", parser="rule",
                       search=True, steps=5)
    assert state["total"] == 2  # one new step, not one per iteration
    assert state["current"]["search"] is True
    assert state["current"]["masses"]["bone"] > 220.0
```

Replace `test_search_requested_for_non_opacity_attribute_falls_back_to_direct_apply` with:

```python
def test_search_requested_for_non_opacity_attribute_falls_back_to_direct_apply():
    # "sharpen bone" parses to attribute="width", direction="decrease". search.propose_step
    # is hardcoded to mutate the height/opacity parameter, so search must not run for width
    # (or brightness/center) commands even when search=True is requested -- the command
    # should still be applied directly, just without the hill-climbing loop.
    s = _fresh_session()
    state = s.command("sharpen bone", parser="rule", search=True, steps=5)
    assert state["current"]["cmd_dict"]["attribute"] == "width"
    assert state["current"]["search"] is False
```

Add this test after `test_initial_state_has_one_step_at_cursor_zero`:

```python
def test_state_and_steps_have_no_judgment_fields():
    s = _fresh_session()
    state = s.command("increase opacity for bone strongly", parser="rule", search=False)
    assert "pending" not in state
    assert "verdict" not in state["current"]
    assert not hasattr(s, "judge")
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `.venv/bin/python -m pytest -q tests/test_server.py::test_state_and_steps_have_no_judgment_fields`
Expected: FAIL (`"pending"` is still in the state).

- [ ] **Step 3: Remove the `rl.serve` import and `PREF_PATH` in `server.py`**

Delete line 32:

```python
from rl.serve import run_policy
```

Delete line 42:

```python
PREF_PATH = "out/preferences.jsonl"
```

- [ ] **Step 4: Replace `_render_step` and delete `_pending_public` in `server.py`**

Replace `def _render_step(...)` (whole function) and the following `def _pending_public(p): …` with:

```python
def _render_step(params, cmd_text, cmd_dict, search, step_id, session_id, camera):
    image_b64, img, png_bytes = _render_image_b64(params, camera)
    image_path = _save_image_file(session_id, f"step_{step_id}", png_bytes)
    return {
        "id": step_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "cmd_text": cmd_text,
        "cmd_dict": cmd_dict,
        "params": params.tolist(),
        "camera": camera,
        "image_b64": image_b64,
        "image_path": image_path,
        "masses": _masses(params),
        "features": features(img),
        "search": search,
        "feedback": None,
    }
```

(`feedback` is removed in Task 3.)

- [ ] **Step 5: Update `Session.__init__`, `_load_or_init`, `state` and `switch_dataset` in `server.py`**

In `Session.__init__`, delete:

```python
        self.pending = None
```

In `_load_or_init`, replace:

```python
        step = _render_step(default_params(), None, None, None, False, 0, self.session_id, default_camera_for(_dataset_name))
```

with:

```python
        step = _render_step(default_params(), None, None, False, 0, self.session_id, default_camera_for(_dataset_name))
```

In `state`, delete the line:

```python
            "pending": _pending_public(self.pending),
```

Replace `switch_dataset` with:

```python
    def switch_dataset(self, name: str):
        set_dataset(name)  # raises ValueError for an unknown name
        self.session_id = self._new_session_id()
        step = _render_step(default_params(), None, None, False, 0, self.session_id, default_camera_for(name))
        self.history = [step]
        self.cursor = 0
        self.save()
        return self.state()
```

- [ ] **Step 6: Delete the human-search helpers and `judge` in `server.py`**

Delete these `Session` methods entirely: `_next_pending_pair`, `_start_human_search`, `judge`.

- [ ] **Step 7: Replace `Session.command` in `server.py`**

```python
    def command(self, text, parser="rule", model="qwen2.5:7b", search=False, steps=10):
        cmd = parse_command(text, parser=parser, model=model)  # raises ValueError on failure

        current_params = np.array(self.history[self.cursor]["params"], dtype=np.float64)
        current_camera = dict(self.history[self.cursor].get("camera", DEFAULT_CAMERA))

        if "camera" in cmd:
            new_camera = apply_camera_command(cmd["camera"], current_camera)
            step = _render_step(current_params, text, cmd, False,
                                 self.history[-1]["id"] + 1, self.session_id, new_camera)
            self.history = self.history[:self.cursor + 1] + [step]
            self.cursor = len(self.history) - 1
            self.save()
            return self.state()

        if search and cmd.get("attribute") == "opacity" and cmd.get("direction") in ("increase", "decrease"):
            new_params = self._run_objective_search(cmd, current_params, steps)
            step = _render_step(new_params, text, cmd, True,
                                 self.history[-1]["id"] + 1, self.session_id, current_camera)
        else:
            new_params = apply_command(cmd, current_params)
            step = _render_step(new_params, text, cmd, False,
                                 self.history[-1]["id"] + 1, self.session_id, current_camera)
            self._log_command(cmd, current_params, new_params, step)

        self.history = self.history[:self.cursor + 1] + [step]
        self.cursor = len(self.history) - 1
        self.save()
        return self.state()
```

- [ ] **Step 8: Update the command route and delete the judge route in `server.py`**

Replace `CommandRequest` and the `/api/command` route with:

```python
class CommandRequest(BaseModel):
    text: str
    parser: str = "rule"
    model: str = "qwen2.5:7b"
    search: bool = False
    steps: int = 10


@app.post("/api/command")
async def command(req: CommandRequest):
    try:
        return session.command(req.text, req.parser, req.model, req.search, req.steps)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
```

Delete `class JudgeRequest` and the whole `@app.post("/api/judge")` route.

- [ ] **Step 9: Run the server tests**

Run: `.venv/bin/python -m pytest -q tests/test_server.py`
Expected: all pass, including `test_state_and_steps_have_no_judgment_fields`.

- [ ] **Step 10: Verify no dangling references**

Run: `grep -nE "run_policy|PREF_PATH|pending|judge|evaluator|verdict" server.py`
Expected: only the `verdict=payload.get("verdict")` argument inside `scene_transition_route` (scene schema keeps that optional field).

- [ ] **Step 11: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "refactor(server): remove policy evaluator and human judge flow"
```

---

### Task 3: Server drops thumbs feedback

**Files:**
- Modify: `server.py`
- Modify: `tests/test_server.py`

- [ ] **Step 1: Update the tests first**

In `tests/test_server.py`, delete these test functions entirely:

- `test_feedback_sets_step_field_and_logs`
- `test_repeated_feedback_is_idempotent`
- `test_feedback_invalid_rating_raises`
- `test_feedback_unknown_step_id_raises`

Extend `test_state_and_steps_have_no_judgment_fields` to:

```python
def test_state_and_steps_have_no_judgment_fields():
    s = _fresh_session()
    state = s.command("increase opacity for bone strongly", parser="rule", search=False)
    assert "pending" not in state
    assert "verdict" not in state["current"]
    assert "feedback" not in state["current"]
    assert not hasattr(s, "judge")
    assert not hasattr(s, "feedback")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest -q tests/test_server.py::test_state_and_steps_have_no_judgment_fields`
Expected: FAIL (`"feedback"` is still in the step).

- [ ] **Step 3: Remove feedback from `server.py`**

Delete line:

```python
FEEDBACK_PATH = "out/feedback.jsonl"
```

In `_render_step`, delete the line:

```python
        "feedback": None,
```

Delete the `Session.feedback` method entirely, `class FeedbackRequest`, and the whole `@app.post("/api/feedback")` route.

- [ ] **Step 4: Run the server tests**

Run: `.venv/bin/python -m pytest -q tests/test_server.py`
Expected: all pass.

- [ ] **Step 5: Verify no dangling references**

Run: `grep -nE "FEEDBACK_PATH|feedback" server.py`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "refactor(server): remove thumbs feedback"
```

---

### Task 4: Delete the old RL package, camera RL and old-RL plots

**Files:**
- Delete: `rl/camera_env.py`, `rl/camera_eval.py`, `rl/camera_train.py`, `rl/collect_preferences.py`, `rl/env.py`, `rl/eval.py`, `rl/eval_reward.py`, `rl/extract_pairs.py`, `rl/finetune_reward.py`, `rl/ingest_feedback.py`, `rl/online_train.py`, `rl/pretrain_reward.py`, `rl/reward_model.py`, `rl/reward_model_env.py`, `rl/reward_model_train.py`, `rl/serve.py`, `rl/train.py`
- Delete: `tests/test_camera_env.py`, `tests/test_camera_eval.py`, `tests/test_rl_collect_preferences.py`, `tests/test_rl_env.py`, `tests/test_rl_eval.py`, `tests/test_rl_extract_pairs.py`, `tests/test_rl_ingest_feedback.py`, `tests/test_rl_online_train.py`, `tests/test_rl_reward_model.py`, `tests/test_rl_reward_model_env.py`, `tests/test_rl_reward_model_train.py`, `tests/test_rl_reward_pipeline.py`, `tests/test_rl_serve.py`, `tests/test_rl_train_logging.py`
- Delete: `plots/online_eval_curve.py`, `plots/reward_model_eval_curve.py`, `tests/test_plots_online_eval_curve.py`
- Keep: `rl/__init__.py` (empty package for RL v2)

- [ ] **Step 1: Confirm nothing outside the deleted set imports these modules**

Run:

```bash
grep -rnE "^(from|import) rl(\.|[[:space:]]|$)|plots\.(online_eval_curve|reward_model_eval_curve)" --include='*.py' . \
  | grep -vE "^\./(\.venv|rl/|tests/test_rl_|tests/test_camera_|tests/test_plots_online_eval_curve|plots/online_eval_curve|plots/reward_model_eval_curve)"
```

Expected: no output (the only outside importer, `server.py`, was fixed in Task 2).

- [ ] **Step 2: Delete the files**

```bash
git rm -q rl/camera_env.py rl/camera_eval.py rl/camera_train.py rl/collect_preferences.py \
  rl/env.py rl/eval.py rl/eval_reward.py rl/extract_pairs.py rl/finetune_reward.py \
  rl/ingest_feedback.py rl/online_train.py rl/pretrain_reward.py rl/reward_model.py \
  rl/reward_model_env.py rl/reward_model_train.py rl/serve.py rl/train.py
git rm -q tests/test_camera_env.py tests/test_camera_eval.py tests/test_rl_collect_preferences.py \
  tests/test_rl_env.py tests/test_rl_eval.py tests/test_rl_extract_pairs.py \
  tests/test_rl_ingest_feedback.py tests/test_rl_online_train.py tests/test_rl_reward_model.py \
  tests/test_rl_reward_model_env.py tests/test_rl_reward_model_train.py \
  tests/test_rl_reward_pipeline.py tests/test_rl_serve.py tests/test_rl_train_logging.py
git rm -q plots/online_eval_curve.py plots/reward_model_eval_curve.py tests/test_plots_online_eval_curve.py
ls rl
```

Expected: `ls rl` prints only `__init__.py` (and possibly `__pycache__`).

- [ ] **Step 3: Run the fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, no collection errors.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(rl): remove old RL pipeline, camera RL and their plots"
```

---

### Task 5: Delete `mvp.py`, `stats.py`, the console judge and drawio backups

**Files:**
- Delete: `mvp.py`, `stats.py`, `.$architecture-highlevel.drawio.bkp`, `.$architecture-mvp.drawio.bkp`
- Modify: `evaluate.py:1` (docstring), `evaluate.py` (delete `human`)

- [ ] **Step 1: Confirm `human` and the scripts are unused elsewhere**

Run:

```bash
grep -rnE "evaluate import.*human|evaluate\.human|import mvp|from mvp|import stats|from stats" --include='*.py' . | grep -v "^\./\.venv"
```

Expected: only hits inside `mvp.py` itself.

- [ ] **Step 2: Remove `human` from `evaluate.py`**

Replace line 1:

```python
"""+1/-1 verdicts: objective (opacity_mass for opacity commands, always-accept for width/brightness/center) and human (console)."""
```

with:

```python
"""+1/-1 objective verdicts for hill-climb search (opacity_mass for opacity commands, always-accept for width/brightness/center), plus a JSONL append helper."""
```

Delete the whole `def human(before_png: str, after_png: str) -> int:` function.

- [ ] **Step 3: Delete the files**

```bash
git rm -q mvp.py stats.py '.$architecture-highlevel.drawio.bkp' '.$architecture-mvp.drawio.bkp'
```

- [ ] **Step 4: Run the fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (`tests/test_evaluate.py` only covers `objective` and `jsonl_append`).

- [ ] **Step 5: Commit**

```bash
git add evaluate.py
git commit -m "chore: remove mvp CLI, stats script, console judge and drawio backups"
```

---

### Task 6: Delete design docs of removed features

**Files:**
- Delete from `docs/superpowers/specs/`: `2026-09-04-rl-implementation-design.md`, `2026-09-06-camera-viewpoint-rl-design.md`, `2026-09-07-opacity-rl-live-wiring-design.md`, `2026-09-09-rl-online-learning-design.md`, `2026-09-09-rlhf-reward-model-design.md`, `2026-09-11-goal-conditioned-reward-design.md`, `2026-09-11-ingest-feedback-design.md`
- Delete from `docs/superpowers/plans/`: `2026-09-04-rl-implementation.md`, `2026-09-06-camera-viewpoint-rl.md`, `2026-09-07-opacity-rl-live-wiring.md`, `2026-09-09-rl-online-learning.md`, `2026-09-09-rlhf-reward-model.md`, `2026-09-11-goal-conditioned-reward-plan.md`, `2026-09-11-ingest-feedback.md`
- Modify: `docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md` (Phase 0 doc list)

- [ ] **Step 1: Delete the docs**

```bash
cd docs/superpowers
git rm -q specs/2026-09-04-rl-implementation-design.md specs/2026-09-06-camera-viewpoint-rl-design.md \
  specs/2026-09-07-opacity-rl-live-wiring-design.md specs/2026-09-09-rl-online-learning-design.md \
  specs/2026-09-09-rlhf-reward-model-design.md specs/2026-09-11-goal-conditioned-reward-design.md \
  specs/2026-09-11-ingest-feedback-design.md
git rm -q plans/2026-09-04-rl-implementation.md plans/2026-09-06-camera-viewpoint-rl.md \
  plans/2026-09-07-opacity-rl-live-wiring.md plans/2026-09-09-rl-online-learning.md \
  plans/2026-09-09-rlhf-reward-model.md plans/2026-09-11-goal-conditioned-reward-plan.md \
  plans/2026-09-11-ingest-feedback.md
cd ../..
```

- [ ] **Step 2: Correct the spec's doc list**

In `docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md`, replace:

```markdown
- Design docs in `docs/superpowers/{specs,plans}/` for removed features:
  `2026-09-04-rl-implementation`, `2026-09-05-training-curve-plots`,
  `2026-09-06-camera-viewpoint-rl`, `2026-09-07-opacity-rl-live-wiring`,
  `2026-09-09-rl-online-learning`, `2026-09-09-rlhf-reward-model`,
  `2026-09-11-goal-conditioned-reward`, `2026-09-11-ingest-feedback`.
```

with:

```markdown
- Design docs in `docs/superpowers/{specs,plans}/` for removed features:
  `2026-09-04-rl-implementation`, `2026-09-06-camera-viewpoint-rl`,
  `2026-09-07-opacity-rl-live-wiring`, `2026-09-09-rl-online-learning`,
  `2026-09-09-rlhf-reward-model`, `2026-09-11-goal-conditioned-reward`,
  `2026-09-11-ingest-feedback`. (`2026-09-05-training-curve-plots` stays: the
  plotter is generic and kept.)
```

and replace:

```markdown
  `plots/read_progress.py` stay. `plots/training_curves.py` stays only if it is
  a generic SB3 `progress.csv` plotter; the implementation plan checks this.
```

with:

```markdown
  `plots/read_progress.py` stay. `plots/training_curves.py` stays (generic SB3
  `progress.csv` plotter).
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md
git commit -m "docs: remove design docs of deleted RL features"
```

---

### Task 7: README update and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the README intro (lines 3–7)**

Replace:

```markdown
**RLHF for speech-controlled volume rendering.** Say "show only bone" or "make
the skeleton pop" and a real CT/MRI scan re-renders live — parsed by a rule
engine or a local LLM, optimized by hill-climbing or a trained RL policy, and
steered by a goal-conditioned reward model learned from human preferences
instead of a hand-coded metric.
```

with:

```markdown
**Speech-controlled volume rendering with a learned transfer-function agent.**
Say "show only bone" or "increase opacity for bone strongly" and a real CT/MRI
scan re-renders live — parsed by a rule engine or a local LLM and executed by
exact commands or hill-climbing search. RL v2 (in progress) adds a
goal-conditioned policy for perceptual instructions, refined with human
preferences.
```

- [ ] **Step 2: Replace the Quick Start block**

Replace:

```markdown
```bash
python mvp.py
python mvp.py --cmd "increase opacity for bone strongly"
python mvp.py --cmd "show only bone" --learn --steps 15
python mvp.py --listen
python server.py
```

The web UI runs at `http://127.0.0.1:8000`. Start commands from repository
root because the application uses relative paths such as `out/` and `static/`.

The current transfer-function state persists in `out/state.json`. Delete that
file or run `reset` to return to the default state.
```

with:

```markdown
```bash
python server.py
python server.py --dataset ct_chest
```

The web UI runs at `http://127.0.0.1:8000`. Start commands from repository
root because the application uses relative paths such as `out/` and `static/`.

The UI session persists in `out/ui_session.json`. Delete that file or say
`reset` to return to the default transfer function.
```

- [ ] **Step 3: Update "Data And UI"**

Replace:

```markdown
```bash
python mvp.py --dataset ct_chest --cmd "show only bone"
python server.py --dataset ct_chest
```
```

with:

```markdown
```bash
python server.py --dataset ct_chest
```
```

and replace:

```markdown
- Text and voice commands
- Back/forward navigation through session history
- Human Better/Worse judgments during search
- Camera state stored with each history step

VR and web clients can use the same reward-data contract described below.
```

with:

```markdown
- Text and voice commands
- Back/forward navigation through session history
- Objective hill-climb search for opacity commands
- Camera state stored with each history step
```

- [ ] **Step 4: Delete the "Preference collection" subsection**

Delete everything from the line `### Preference collection` up to (not including) the line `## Reinforcement Learning`.

- [ ] **Step 5: Replace the whole "Reinforcement Learning" section**

Replace everything from the line `## Reinforcement Learning` up to (not including) the line `## Tests` with:

```markdown
## Reinforcement Learning (v2, in progress)

The previous RL pipeline (a height-only SAC agent trained on `mass_fraction`,
camera RL, and a reward model on four global image statistics) has been
removed: its reward did not measure what is visible on screen.

RL v2 trains one goal-conditioned policy for perceptual instructions
(relative, compound, absolute, show only, brightness) against a per-tissue
visibility estimate on real CT volumes, then refines it with human A/B
preferences collected on a dedicated page. Design:
[`docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md`](docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md).

```

- [ ] **Step 6: Update "Tests" and "Project Layout"**

In the Tests section, delete:

```markdown
Reward pipeline tests:

```bash
python -m pytest -q \
  tests/test_rl_reward_model.py \
  tests/test_rl_reward_model_env.py \
  tests/test_rl_extract_pairs.py \
  tests/test_rl_reward_pipeline.py
```
```

In Project Layout, replace:

```markdown
- Root files: rendering, transfer functions, parser, CLI, and web UI
- `rl/`: environments, policies, reward model, data pipeline, evaluation
```

with:

```markdown
- Root files: rendering, transfer functions, parser, and web UI
- `rl/`: RL v2 (in progress)
```

- [ ] **Step 7: Verify the README has no stale references**

Run: `grep -nE "mvp\.py|rl\.(train|eval|camera|online|pretrain|finetune|extract|reward)|Better/Worse|state\.json" README.md`
Expected: no output.

- [ ] **Step 8: Repository-wide dangling-reference check**

Run:

```bash
grep -rnE "rl\.(env|eval|serve|train|camera_|online_|reward_|pretrain_|finetune_|extract_|ingest_|collect_)|import mvp|from mvp|/api/judge|/api/feedback|PREF_PATH|FEEDBACK_PATH" \
  --include='*.py' --include='*.js' --include='*.html' --include='*.md' --include='*.yml' --include='Dockerfile' . \
  | grep -vE "^\./(\.venv|docs/superpowers/plans/2026-09-15-rl-v2-phase0-cleanup\.md|docs/superpowers/specs/2026-09-15)"
```

Expected: no output. (Hits in `docs/*.typ`/slides are thesis text, not code, and are left to the author.)

- [ ] **Step 9: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (including `slow`).

- [ ] **Step 10: Manual UI check**

Run `.venv/bin/python server.py --dataset ct_chest` in the background, open `http://127.0.0.1:8000`, then:

1. Send `increase opacity for bone strongly` → a new message appears, the 3D viewer updates, no thumbs buttons.
2. Enable `search`, send `increase opacity for fat` → one new step with a `search` tag; the toolbar shows only the steps input, no evaluator toggle.
3. Back/forward buttons work; `reset` works.
4. Browser console shows no errors.

Stop the server afterwards.

- [ ] **Step 11: Commit**

```bash
git add README.md
git commit -m "docs(readme): describe the app after the RL v2 cleanup"
```

---

## Next plans (written right before execution)

2. Data: TotalSegmentator selection, NIfTI loader, canonical orientation, Slicer mappings.
3. Views + visibility/brightness estimate + validation gate.
4. Goals, sampler, parser extension.
5. Environment, baselines, SAC training, evaluation (tier-1 result).
6. Collection: candidates, routes, linked-viewport page, smoke test.
7. Embeddings + PCA (+ reward model, month).
