# Minimal Shadcn VR-Ready UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans (recommended) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the vanilla viewer into a minimal English-only shadcn-style light/dark interface with voice-first controls, controller fallback, and VR-ready panel hierarchy.

**Architecture:** Preserve backend APIs, existing behavior, and important DOM IDs used by `static/app.js`, while simplifying markup and replacing stylesheet-heavy visual hierarchy with semantic shell regions and stock shadcn zinc tokens. Add theme/About behavior in `static/app.js` using localStorage and native dialog patterns; no framework or build pipeline.

**Tech Stack:** HTML, vanilla CSS, browser JavaScript, native `dialog`, inline SVG, pytest static contract tests.

---

## File Map

- Modify `static/index.html`: minimal semantic shell, English labels, theme/About controls, preserved behavior IDs.
- Modify `static/collect.html`: English-only collector copy and same clean shadcn shell where applicable.
- Modify `static/style.css`: stock shadcn zinc tokens, responsive viewer/console layout, large core hit targets, theme variants, modal/about styles; remove obsolete visual rules.
- Modify `static/app.js`: theme persistence/toggle, About dialog handling, English UI strings, accessible state attributes; preserve command/API logic.
- Modify `static/collect.js`: replace any non-English or stale UI copy and preserve judging behavior.
- Modify `tests/test_static_viewer_contract.py`: assert required IDs, English labels, theme/About links, and no removed German copy.
- Create or modify browser/static tests only where existing repository test style supports behavior checks; no frontend dependency installation.

### Task 1: Establish Static Contract Tests

**Files:**
- Modify: `tests/test_static_viewer_contract.py`
- Test: `tests/test_static_viewer_contract.py`

- [ ] **Step 1: Add failing contract assertions**

Add assertions for:

```python
def test_viewer_has_theme_and_about_controls():
    html = _read("static/index.html")
    assert 'id="theme-toggle"' in html
    assert 'id="about-btn"' in html
    assert 'id="about-modal"' in html
    assert "GitHub repository" in html
    assert "Kevin Kunkel portfolio" in html


def test_viewer_copy_is_english_only():
    html = _read("static/index.html")
    for word in ("LOKALE SITZUNG", "SKELETT", "LUNGE", "WEICHGEW.", "GEFÄSSE"):
        assert word not in html


def test_core_controls_have_accessible_labels():
    html = _read("static/index.html")
    assert 'aria-label="Theme"' in html
    assert 'aria-label="About this project"' in html
    assert 'aria-label="Record voice command"' in html
```

Use existing helper conventions from the test file; do not introduce a JS runtime dependency.

- [ ] **Step 2: Run tests and verify RED**

```bash
.venv/bin/python -m pytest -q tests/test_static_viewer_contract.py
```

Expected: failures for missing new controls and legacy copy.

### Task 2: Replace Viewer Shell Markup

**Files:**
- Modify: `static/index.html`
- Test: `tests/test_static_viewer_contract.py`

- [ ] **Step 1: Preserve JS contract IDs before markup edit**

Retain IDs consumed by `static/app.js`: `app`, `app-bar`, `app-bar-left`, `app-bar-right`, `main`, `image-panel`, `dataset-row`, dataset select IDs, `viewport`, `single-view`, `viewport-frame`, `vtk-viewer`, `viewer-status`, `viewer-error`, `current-image`, `nav`, `back-btn`, `step-counter`, `forward-btn`, `reset-btn`, `telemetry`, all telemetry IDs, `chat-panel`, `messages`, `empty-state`, `composer`, `toolbar`, parser group, search toggle/options/input, policy toggle, compare, sweep, input row, mic, text input, send, command modal IDs, toast viewport.

- [ ] **Step 2: Implement clean semantic markup**

Use `header`, `main`, `section`, `nav`, and `aside` while retaining IDs. Top bar must contain English product name, dataset selector, theme button, About button, and command help. Keep viewer canvas dominant. Keep right command console stable. Visibility readout must use English labels: `Skeleton`, `Lungs`, `Soft tissue`, `Vessels`.

Add About dialog:

```html
<button id="about-btn" class="icon-button" type="button"
        aria-label="About this project" aria-haspopup="dialog">?</button>

<dialog id="about-modal" aria-labelledby="about-modal-title">
  <article class="about-card">
    <header class="about-header">
      <div>
        <p class="eyebrow">Master's thesis project</p>
        <h2 id="about-modal-title">Conversational Transfer Functions</h2>
      </div>
      <button id="about-modal-close" class="icon-button" type="button"
              aria-label="Close About">×</button>
    </header>
    <div class="about-body">
      <p>Explore medical CT volume rendering with typed or spoken instructions.</p>
      <p>The app maps language to semantic transfer-function goals for lungs,
      soft tissue, vessels, and skeleton.</p>
      <p>Exact application, measured search, and a learned one-shot policy answer
      commands. The policy is evaluated with scan-aware visibility and brightness
      measurements.</p>
      <p>Human preference learning and Vrui/VR integration are future directions,
      not completed deployment claims.</p>
    </div>
    <footer class="about-links">
      <a href="https://github.com/kevinkunkel98/reinforcement-learning-volume-rendering-transfer-function"
         target="_blank" rel="noreferrer" aria-label="GitHub repository">[inline GitHub SVG]</a>
      <a href="https://kevin-kunkel.netlify.app" target="_blank" rel="noreferrer"
         aria-label="Kevin Kunkel portfolio">[inline portfolio SVG]</a>
    </footer>
  </article>
</dialog>
```

Replace placeholder SVG text with actual inline SVG paths. No icon dependency.

- [ ] **Step 3: Run static contract tests**

```bash
.venv/bin/python -m pytest -q tests/test_static_viewer_contract.py
```

### Task 3: Implement Stock Shadcn Theme and Layout

**Files:**
- Modify: `static/style.css`

- [ ] **Step 1: Replace token layer**

Keep default shadcn zinc HSL tokens for light mode in `:root` and dark tokens under `[data-theme="dark"]`. Use `color-scheme` accordingly. Do not introduce custom brand palette for shell components. Keep categorical comparison colours only where direct labels remain.

- [ ] **Step 2: Implement stable VR-ready regions**

Use a desktop grid with viewer region and console region. Keep core target sizes at least 40px. Keep viewer black in both themes. Use visible borders, focus rings, spacing, and text labels. No essential state may depend on hover.

Required responsive behavior:

```css
#main { display: grid; grid-template-columns: minmax(0, 1fr) 400px; }
@media (max-width: 900px) {
  #main { display: flex; flex-direction: column; }
  #chat-panel { min-height: 420px; max-width: none; }
}
```

Adapt dimensions to existing viewer behavior, but preserve stable regions and large targets.

- [ ] **Step 3: Add modal, About, theme, focus, reduced-motion styles**

Style native `dialog`, `.about-card`, links, focus-visible states, `aria-pressed` toggles, disabled states, and reduced-motion behavior using stock tokens.

- [ ] **Step 4: Remove obsolete UI styling**

Remove or consolidate rules for German copy, redundant session chrome, decorative controls, and duplicated toolbar visual treatments. Do not remove styles still required by compare/sweep or collector pages.

### Task 4: Add Theme and About Behavior

**Files:**
- Modify: `static/app.js`
- Test: `tests/test_static_viewer_contract.py`

- [ ] **Step 1: Add theme state helpers**

Implement:

```javascript
const THEME_KEY = "tf-rl-theme";

function applyTheme(theme) {
  const resolved = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = resolved;
  const button = el("theme-toggle");
  button?.setAttribute("aria-pressed", String(resolved === "dark"));
  button?.setAttribute("title", `Switch to ${resolved === "dark" ? "light" : "dark"} theme`);
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  applyTheme(saved === "light" ? "light" : "dark");
  el("theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  });
}
```

Initialize before or alongside existing app initialization without changing API requests.

- [ ] **Step 2: Add About dialog behavior**

Open `about-modal` from `about-btn`, close via close button, Escape, and backdrop click. Return focus to trigger. Do not touch viewer state, history, or API.

- [ ] **Step 3: Normalize visible copy to English**

Update all strings in `static/app.js` that reach the viewer: errors, toasts, mode labels, parser labels, telemetry/helper text, compare/sweep copy, and status messages. Keep command semantics and backend payloads unchanged.

- [ ] **Step 4: Run static and syntax checks**

```bash
.venv/bin/python -m pytest -q tests/test_static_viewer_contract.py
node --check static/app.js
```

### Task 5: Normalize Collector Copy and Shared Theme

**Files:**
- Modify: `static/collect.html`
- Modify: `static/collect.js`
- Modify: `static/style.css`

- [ ] **Step 1: Normalize collector labels and actions to English**

Use `Preference collection`, `Candidate A`, `Candidate B`, `Start state`, `Show start`, `Choose A`, `Choose B`, `Equal`, and `Skip`. Preserve blind judging, side randomization, keyboard shortcuts, and data submission behavior.

- [ ] **Step 2: Apply shared theme and accessible target sizing**

Collector should use same stock shadcn tokens, theme behavior where it shares `style.css`, visible focus states, and large judge buttons. Do not add About content if collector flow would bias ratings.

- [ ] **Step 3: Run collector/static tests**

```bash
.venv/bin/python -m pytest -q tests/test_static_viewer_contract.py tests/test_collect.py
node --check static/collect.js
```

### Task 6: Browser Verification

**Files:**
- No source changes unless verification finds a concrete issue.

- [ ] **Step 1: Run application**

```bash
.venv/bin/python server.py
```

- [ ] **Step 2: Verify desktop**

Check:

- Viewer remains dominant.
- Console stays anchored.
- Dataset, history, reset, theme, About, parser, mode, compare, sweep, mic, and send remain usable.
- About links open correct URLs in new tabs.
- Theme toggle updates shell without changing render canvas.
- Compare and sweep preserve layout region.

- [ ] **Step 3: Verify mobile**

Check stacked viewer/console layout, composer reachability, no horizontal overflow, and 40px core targets.

- [ ] **Step 4: Verify VR readiness**

Check visual hierarchy at distance: viewer, command console, visibility readout, and core fallback controls remain distinguishable without hover. Record any decisions for later Vrui shell integration.

- [ ] **Step 5: Run regression suite**

```bash
.venv/bin/python -m pytest -q tests/test_static_viewer_contract.py tests/test_collect.py tests/test_server.py
```

Report unrelated existing data-fixture failures separately; do not alter backend/RL behavior to resolve UI-only test issues.
