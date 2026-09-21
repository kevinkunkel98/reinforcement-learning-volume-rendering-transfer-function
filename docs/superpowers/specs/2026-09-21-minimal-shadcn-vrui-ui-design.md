# Minimal Shadcn VR-Ready UI

## Goal

Reshape the current vanilla web viewer into a clean, minimal shadcn-style interface that is voice-first, controller-compatible for core actions, and structurally ready for future Vrui/VR integration.

## Design Principles

- Use default shadcn zinc light and dark theme tokens.
- Remove custom visual theming, decorative UI, duplicated status, and nonessential controls.
- Preserve current application behavior, API routes, DOM IDs where practical, and no-build-step architecture.
- Keep viewer canvas dominant and command console stable as a separate right-side region.
- Make interaction state visible without relying on hover, tooltips, or colour alone.
- Keep controls large enough for gaze/controller targeting and readable at VR viewing distance.
- Treat voice as primary interaction; controller fallback exposes core actions only.

## Required Controls

All visible UI copy must be English. This includes labels, buttons, status text,
empty states, errors, tooltips, telemetry labels, command reference copy, and
About content. Dataset names and user-entered commands remain unchanged.

Keep visible in normal single-view mode:

- Dataset selector.
- Previous/next history navigation.
- Reset.
- Light/dark theme toggle.
- Parser selector: rule / LLM.
- Answer mode selector: exact / search / policy.
- Search budget only when search mode is active.
- Compare.
- Sweep 20.
- Voice/text composer with microphone and send controls.
- Compact tissue visibility readout.
- About button opening the thesis/app information modal.

Remove or consolidate:

- German labels and mixed-language status copy.
- Redundant local-session chrome.
- Decorative toolbar/status elements that do not change application state.
- Persistent comparison-specific controls outside compare view.
- Hover-only explanations when visible labels can communicate the state.

## Layout

Desktop:

- Top bar: product mark, dataset selector, theme toggle, command reference.
- Main area: large viewer region on left; command console on right.
- Viewer region: canvas, history/reset controls, compact visibility readout.
- Console: message history, mode controls, composer anchored at bottom.

Comparison mode:

- Replace viewer canvas inside the same viewer region.
- Keep console and composer fixed in place.
- Display four answering arms with clear method labels, attainment, cost, and render.

Mobile:

- Stack viewer above console.
- Keep composer reachable without excessive scrolling.
- Preserve large target sizes and visible state labels.

VR readiness:

- Use stable rectangular regions that can become floating Vrui panels.
- Avoid dense horizontal toolbars that depend on mouse precision.
- Use minimum 40px web hit targets for core controls.
- Keep core controls grouped in predictable order.
- Preserve keyboard and voice operation.
- Do not add web-only hover interactions to essential flows.

## Theme

Use stock shadcn zinc tokens:

- Light: white/zinc surfaces, zinc borders, dark foreground, standard primary/secondary/destructive variants.
- Dark: shadcn zinc dark tokens already present in the project.
- Render canvas remains black in both themes for image consistency.
- Theme preference persists in `localStorage`.
- Theme toggle exposes `aria-pressed` and visible current mode.

No custom tissue colours are removed from comparison data visualizations where categorical identity requires them, but every class remains text-labelled and colour-independent.

## About Modal

Add an accessible About modal opened from the top bar. Use the existing native
`dialog` pattern and shadcn-style card surface; do not add a separate route.

Modal content must describe:

- Project title: conversational transfer-function design for medical volume rendering.
- This is a master's thesis project.
- The app renders CT volumes and accepts typed or spoken instructions.
- The transfer function uses semantic Gaussian peaks for lungs, soft tissue, vessels, and skeleton.
- Exact, search, and policy answering modes.
- The policy is evaluated with scan-aware visibility and brightness measurements.
- Search is a baseline; policy is a learned one-shot proposal, not a claim of universal superiority.
- Human preference collection and Vrui/VR integration are planned directions, not claims of completed deployment.

Keep copy concise and honest. Add two dedicated icon links at the bottom of the
modal:

- GitHub: `https://github.com/kevinkunkel98/reinforcement-learning-volume-rendering-transfer-function`
- Portfolio: `https://kevin-kunkel.netlify.app`

Each icon must have a visible tooltip/accessible label (`GitHub repository` and
`Kevin Kunkel portfolio`), use `target="_blank"`, and include
`rel="noreferrer"`. Use inline SVG icons or existing project-safe assets; do not
add an icon dependency.

Modal requirements:

- Visible `About` trigger with accessible label.
- Heading and labelled dialog relationship.
- Close button, Escape handling, and click-outside behavior consistent with current command modal.
- Keyboard focus enters modal and returns to trigger on close.
- Works in light/dark themes and remains legible at VR panel scale.
- Does not pause, mutate, or reset viewer state.

## Accessibility and Interaction

- Every core control has visible text or an accessible label.
- Toggle groups expose selected state through `aria-checked` or `aria-pressed`.
- Focus-visible styles remain strong in both themes.
- Keyboard navigation works for dataset, parser, answer mode, history, reset, compare, sweep, and composer.
- Reduced-motion preference disables nonessential animations.
- Voice recording state announces through status text and `aria-live`.
- Errors use visible alert/toast copy, not console-only output.

## Scope Limits

- No React migration.
- No component-library build pipeline.
- No Vrui renderer integration in this UI pass.
- No changes to backend APIs or RL behavior.
- No new feature panels such as direct curve editing or DINOv2 controls.
- No separate About route or external documentation dependency.

## Verification

- Existing static viewer contract tests continue passing.
- Add tests for theme persistence/state, required core IDs, and keyboard/accessibility attributes where the current test style supports it.
- Verify desktop and mobile layout in browser.
- Verify both themes manually.
- Verify compare and sweep retain behavior.
- Verify voice recording, parser selection, answer-mode selection, and reset retain behavior.
