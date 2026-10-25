# Scientific Frontend Implementation Plan

> **For agentic workers:** Use executing-plans to implement these tasks inline. Track completion below.

**Goal:** Apply the approved Obsidian × Kaggle design to the working CT explorer, including an equally polished light theme.

**Architecture:** Retain the vanilla frontend and its existing API/renderer contracts. Update the page structure and shared design tokens in place; small JavaScript additions handle starter instructions, method summaries, and theme-aware chart rendering. Keep all real volume and research data sourced from existing endpoints.

**Tech Stack:** HTML, CSS, vanilla JavaScript, vtk.js, Python/FastAPI; existing pytest and Node checks, plus browser verification.

**Spec:** Approved in conversation: charcoal surfaces, sky-blue actions, restrained violet identity, volume-first layout, compact instructions, grouped anatomy and transfer-function readouts, discoverable methods/experiments, consistent icons, responsive layout, keyboard focus, and a persistent light/dark toggle. Reference: `.superpowers/brainstorm/12418-1792894330/content/obsidian-kaggle-study-v2.html`.

## Global Constraints
- Work on `feat/scientific-frontend` in the current checkout, as already approved.
- Preserve existing element IDs, endpoints, history, voice, comparison, and VR hooks.
- Both light and dark themes use the existing `tf-rl-theme` storage preference.
- No new runtime dependencies or font network requests.
- Anatomical values and HU ranges must remain real; unavailable measurements stay explicit.

## Review Focus
- Theme changes must repaint canvas labels and curves, including immediately after a saved light-theme load.
- Narrow and short windows must keep controls reachable and resize the volume without stretching its projection.
- Space on buttons, disclosures, and numeric inputs must operate the focused control rather than start recording.
- Collapsed method controls must still report the selected parser/method accurately.
- Renderer failure must leave the server PNG and error message visible in both themes.

### Task 1: Workspace and themes
**Files:** `static/index.html`, `static/style.css`.
**Interfaces:** Preserve all existing IDs and data attributes. Add `#method-settings`, `#method-summary`, `[data-command]`, `#tf-axis-min`, and `#tf-axis-max`.
- [x] Run existing static contract baseline: `.venv/bin/pytest tests/test_static_viewer_contract.py -q` (40 passed).
- [x] Replace identity and ambiguous icons; add compact context, notebook header, and starter instructions.
- [x] Group real readouts and move research controls into a native details disclosure.
- [x] Replace stock tokens with graphite/blue and coordinated porcelain/blue light tokens; adapt layout and responsive rules in place.
- [x] Verify existing static contracts and inspect both themes in Chrome.

### Task 2: Interaction and visualization integration
**Files:** `static/app.js`, `static/viewer.js`.
**Interfaces:** Consume Task 1 IDs. Preserve `applyTheme(theme)`, config parser/mode values, `drawTfCurve(histogram, curve)`, and `window.volumeViewer` API.
- [x] Wire starter buttons to populate/focus the existing textarea without submitting.
- [x] Sync collapsed method summary with actual parser/mode; align the default parser radio with hybrid configuration.
- [x] Redraw the real chart on theme/size changes; derive axis limits from its data and use theme-readable label colors.
- [x] Keep Space shortcuts from intercepting focused native controls; resize vtk canvas on viewport changes while respecting XR.
- [x] Exercise these behaviors in Chrome, including fallback rendering and light-theme persistence.

### Task 3: Verification and review
**Files:** Above files; this plan records evidence.
- [x] Run `.venv/bin/pytest tests/test_static_viewer_contract.py tests/test_xr_logic.py -q` and `node --test tests/js/*.test.js`.
- [x] Run JavaScript syntax checks and `git diff --check`.
- [x] Browser-check desktop/mobile, light/dark, saved theme, starter commands, methods, dialogs, history, real command response, and chart resize.
- [x] Obtain a fresh review of the branch changes; address substantive findings and record outcomes.

## Execution notes
- Follow-up approved 2026-10-25: remove the bottom transfer-function graph and its frontend drawing code; anatomy readouts now span the available width. Increase general UI text by roughly 1–2px in both themes, with narrow-screen anatomy groups stacked for readability. This supersedes the graph requirements above.
- User approved building directly on the already-created feature branch. No additional worktree is needed for this focused refresh.
- Reversible visual changes are verified in the browser and through existing contracts; no source-text tests are added for CSS/copy decisions.
- Changes remain on the feature branch for user inspection; no merge or push is part of this request.

## Verification evidence
- Static viewer/XR Python checks: 41 passed. Node rendering/VR utilities: 88 passed.
- Browser interaction checks failed before wiring (starter, chart theme redraw, summary, keyboard disclosure), then passed; saved light-theme persistence passed throughout.
- Real vtk.js CT rendering tested at 1440×1000, 1024×768, 390×844, and 320×640 in both themes: no horizontal overflow or duplicate IDs; chart and viewport backing dimensions match their display size; composer and experiments reachable.
- Real API workflows passed: direct rule command, reply/telemetry, back/forward, four-arm comparison, dialogs with focus restoration, dataset selector dismissal. No browser page errors.
- Independent review found a short-desktop clipping regression with expanded experiments. Reproduced Apply below the viewport at 1024×500, then fixed with scrollable chat containment. Wheel-based checks now pass at 1024×500 and 1440×600 with multiline input.
- Physical-headset and microphone/transcription quality checks require hardware/manual validation; existing JS XR tests passed. Cross-browser parity and a full accessibility audit were not part of this browser verification.
