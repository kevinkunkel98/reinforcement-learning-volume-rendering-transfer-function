# Three-mode comparison panel — design

Date: 2026-09-20
Status: agreed, not started. Targets the MVP demo on 2026-09-22.

## Why this exists

The prototype is a deliverable in its own right, and on Tuesday it will be
driven live in front of people. The demo has to dramatise the thesis result, not
merely show a working product.

That result is an amortisation claim: the policy reaches the 10-evaluation
hill-climber's median attainment while spending **no** evaluations at all
(+0.275 against +0.263 over three seeds, 200 held-out episodes). Today the
viewer can only show one answering mode at a time, so that comparison has to be
reconstructed by the audience from three separate interactions — which is
exactly what stops it landing.

This design puts all three answering modes on screen at once, from the same
start state and the same parsed command, reporting the same attainment number
the thesis reports.

### The defect this also fixes

`server.py`'s search mode is **not** the search the thesis measured.
`Session._run_objective_search` is a coordinate stepper over
`evaluate.objective` using `propose_step`/`resize_step`; the B3/B4 baselines are
`rl.baselines.hill_climb` on `goals.distance`. They are different searches
against different objectives.

It is also nearly always inert: the branch only fires when
`cmd["attribute"] == "opacity"` and the direction is increase/decrease, so
"show only the lungs" silently falls through to exact application while the UI
still displays the search toggle as active.

A side-by-side panel labelling that column "search" would put a number on screen
that nothing in the thesis backs. Fixing it is a precondition, not a nicety.

## Goal

One instruction, three arms, one screen:

```
  "show only the lungs"                          budget: 10 evaluations

  EXACT                 SEARCH                   POLICY
  [render]              [render]                 [render]
  0 evals · 40 ms       10 evals · 4.1 s         0 evals · 20 ms
  attainment  +0.11     attainment  +0.26        attainment  +0.28

  skeleton ▌            skeleton ▌               skeleton ▌
  lungs    ████         lungs     ██████         lungs    █████
  soft     ██           soft      █              soft     ███
  vessels  ▏            vessels   ▏              vessels  ▏
```

## Non-goals

- **Adopting an arm back into history.** Good demo ergonomics ("compare, pick
  one, carry on") but it is a second endpoint plus history semantics. The
  comparison stands without it.
- **Changing the default answering mode.** Exact stays the default; the
  comparison is an explicit action.
- **Touching the preference collector.**

## Architecture

### `POST /api/compare`

Request: `{text, parser, model, budget}`. Response: the three arms plus the
shared context, or `{applicable: false, reason}`.

The endpoint **never mutates session history**. Comparing is a side quest, not a
step; the session's current step supplies the common start (`params`, `camera`)
and is left exactly as it was.

Sequence:

1. `parse_command_with_meta(text, parser, model)` — **once**. All three arms
   answer the identical parsed command, so a parser failure makes all three
   wrong together and the panel shows a parsing problem rather than a policy
   problem.
2. `goals.goal_from_command(cmd, model, start_agg, volume)` — **once**. Returns
   `{"goal", "text"}`, which is already the shape `hill_climb` accepts as its
   `instruction`.
3. Run the three arms from the same `current_params`.
4. For each arm: render, measure, score.

### The arms

| Arm | Call | Evaluations |
|---|---|---|
| `exact` | `apply_command(cmd, params)` | 0 |
| `search` | `rl.baselines.hill_climb(model, params, instruction, evaluations=budget)` | `budget` |
| `policy` | `run_policy_arm(model, policy, instruction, start_agg, params)` | 0 |

The search arm is the B3 baseline itself, not a lookalike. The policy arm is the
existing code path, which builds its observation exactly as `rl.candidates`
does, so it really is the measured policy.

### What each arm reports

| Field | Source |
|---|---|
| `image_b64` | `_render_image_b64(params, camera)`, as `_render_step` does |
| `class_visibility` | `_class_visibility(params)` — the four goal classes |
| `attainment` | `goals.attainment(instruction["goal"], start_agg, final_agg)` |
| `evaluations` | 0, `budget`, 0 |
| `elapsed_ms` | wall clock around the arm |
| `unavailable` | reason string, or absent |

`attainment` is the same function that produced every number in the paper. That
is what lets the panel claim "+0.28 at 0 evaluations against +0.26 at 10" and
have it mean what the thesis means.

### Factoring

`compare_arms(model, policy, cmd, instruction, start_params, camera, budget)` is
a module-level function returning the arm dicts, with the FastAPI route a thin
wrapper that resolves `model` and `policy` off the session and passes them in.
The route is then trivial, and the logic is testable without spinning up a
server or constructing a `Session`, matching how the rest of `server.py` is
already tested.

`Session._run_policy` currently owns the observation-building and prediction for
the policy arm while reading `self.policy_provider()` and
`self.model_for_volume(...)`. That body is extracted to a module-level
`run_policy_arm(model, policy, instruction, start_agg, params)` taking its
dependencies as arguments -- the goal and the start aggregate included, so a
caller comparing three arms builds each exactly once instead of paying another
`model.features` (~17 ms) inside the policy arm's own timing; `Session._run_policy` becomes a two-line delegation that resolves
them. One implementation, two callers, and the comparison provably runs the same
policy code the single view does.

## The search fix

`Session.command()`'s search branch moves to `hill_climb` as well, so the
single-view toggle and the comparison panel agree, and the `opacity`-only
restriction disappears.

Knock-on: `evaluate.objective` then has no caller. The audit item "decide about
`evaluate.objective`'s unreachable branches" becomes "delete the function". That
deletion is a **separate, vetoable step**, not folded in silently.

Cost note: `_run_objective_search` logs each accepted step to `LOG_PATH`.
`hill_climb` does not, so that per-step log goes away for search commands. The
step itself is still logged by `_render_step`.

## Visual design

The panel is built in the existing idiom: `static/style.css` is a hand-ported
shadcn/ui design system — the stock zinc dark token set plus Button, Select,
Card, Badge, Toggle, ToggleGroup, Dialog, Tooltip, Toast and ScrollArea as
vanilla CSS, no React and no build step. New components are ported the same way.

**To port:** Table (per-class rows), Progress (the visibility bars), Skeleton
(the search column's loading state — it genuinely takes seconds, and a skeleton
is what that gap is for), Separator.

### Forms

Chosen by the job the data does, not by what looks rich:

- **Attainment** is a single headline per arm → a **hero number**, not a chart.
  Three numbers side by side is a stat row; a three-bar chart of three values is
  chart junk.
- **Cost** (evaluations, wall clock) is a secondary stat → small muted text and
  a Badge, under the headline.
- **Per-class visibility** is magnitude with identity → four bars per arm,
  coloured by class, each with a tick marking that class's value in the start
  state, so "did the mentioned class move and did the others stay put" is one
  glance. That question is the `KEEP_TOLERANCE` term in `goals.distance` made
  visible.

### Colour

Four goal classes are **categorical identity**, so hues are assigned in fixed
order and never cycled:

| Class | Slot | Dark hex |
|---|---|---|
| skeleton | 1 blue | `#3987e5` |
| lungs | 2 orange | `#d95926` |
| soft | 3 aqua | `#199e70` |
| vessels | 4 yellow | `#c98500` |

Validated against the zinc dark surface `#09090b`:

```
[PASS] Lightness band      all 4 inside L 0.48–0.67
[PASS] Chroma floor        all 4 >= 0.1
[PASS] CVD separation      worst adjacent #c98500↔#199e70 ΔE 8.4 (protan) · tritan 24.4
[PASS] Normal-vision floor worst adjacent ΔE 19.8 (normal)
[PASS] Contrast vs surface all 4 >= 3:1
```

Every bar is direct-labelled with its class name, so identity never rests on
colour alone. Attainment's sign uses the existing `--success` / `--destructive`
tokens **with the sign printed** (`+0.28`, `−0.04`) — never colour alone. Values
and labels stay in text tokens; the coloured bar beside them carries identity.

### Layout

Three equal columns on a wide screen. The panel **replaces** `#single-view`
inside the existing `#viewport` while active — same footprint, so the chat panel
and composer never move — with a close control that restores the single view.
Below ~900px the columns stack — not a demo requirement, but the
existing CSS is responsive and breaking that would be a regression.

## Error handling

| Case | Behaviour |
|---|---|
| Parse failure | 400 with the parser's message, same as `/api/command` |
| Camera / reset / width / centre command | `{applicable: false, reason}`; the panel says the comparison is for visibility instructions |
| `goal_from_command` raises (goal the volume cannot support, e.g. vessels on a non-contrast scan) | same `applicable: false` path |
| No policy checkpoint loaded | **policy arm alone** reports `unavailable` with its reason; the other two still render |
| An arm raises unexpectedly | that arm reports `unavailable`; the panel degrades rather than 500s |

Losing one column must never kill the demo.

## Testing

Against `compare_arms`, with the stub model the existing tests use:

- all three arms start from the same params (the shared start is the whole point)
- reported `evaluations` are 0 / `budget` / 0
- attainment for each arm is computed against the **shared** start aggregate
- a policy provider returning `None` yields `unavailable` on that arm and leaves
  the other two intact
- a camera command returns `applicable: false`
- the session's history and cursor are unchanged after a compare call

Plus, for the search fix: `Session.command(mode="search")` on a non-opacity
instruction now actually searches, which the current code does not.

## Risk

The search column takes seconds of dead air on stage. That is the point being
demonstrated — it is the cost the policy avoids — but the Skeleton state has to
make it read as "working", not "broken". The budget is a request field so it can
be dialled down live if the room's patience is shorter than expected.
