# New Command Attributes + Command Overview — Design

## Purpose

Today's commands only ever touch a peak's **height** (`attribute: "opacity"`
is the only value ever produced or consumed). This adds three more
attributes users can control by voice/text — **width** (sharpness), **color**
(brightness only), and **center** (HU position) — using the *same* command
schema and grammar shape already established for opacity, plus a command
**overview** (in-app help panel + generated `COMMANDS.md`) so the growing
command surface stays discoverable instead of only living in developers'
heads.

## Grounding in existing code

- `commands.py`'s command dict shape is already generic:
  `{"target", "attribute", "direction", "strength"}` for relative commands,
  `{"target", "attribute", "direction": "set", "level"}` for absolute-level
  commands. Only `attribute: "opacity"` is ever produced today — the field
  already exists and is already threaded through `apply_command`,
  `evaluate.objective`, and the LLM system prompt's schema description. No
  schema change is needed, only new attribute *values*.
- `transfer.py::peak_internal(params, i)` decodes a peak's 6 floats as
  `center, width, height, r, g, b`, each stored in a common pattern: a real
  value normalized into external `[-1, 1]` via either `_to_range`/`_from_range`
  (center, width — arbitrary `[lo, hi]`) or `_unit`/`_from_unit` (height,
  r, g, b — `[0, 1]`). Critically, `_from_unit(u) = u * 2 - 1` is exactly
  `_from_range(u, 0, 1)` — **the external encoding is always a plain
  normalized fraction of *some* real range**, regardless of what that range
  represents. This means the existing asymptotic-step formula used for
  height (`new = current + delta * (1 - current)` for increase, `current -
  delta * current` for decrease, both in `[0,1]` *unit* space) generalizes
  cleanly to *any* attribute once phrased in terms of the external `[-1,1]`
  value: `new_ext = current_ext + delta * (1 - current_ext)` (toward +1) /
  `current_ext - delta * (current_ext + 1)` (toward -1). One helper function
  replaces what was height-only logic.
- `commands.py::LEVEL_WORDS = {"low": 0.15, "medium": 0.5, "high": 0.85}` is
  already a unit fraction, so `_from_unit(level)` already gives the correct
  external value for *any* attribute's absolute "set" — same reasoning as
  above.
- `commands.py::_find_or_create_peak` resolves height-target peaks by
  nearest HU center. Width/brightness/center commands reuse the same
  resolution (a "target tissue" always means "the peak currently
  representing that tissue," regardless of which attribute is being
  adjusted).
- `evaluate.py::objective()` only knows how to score opacity's increase/
  decrease against `opacity_mass` deltas. There is no established exact
  metric for "is this width/brightness/center change better" — inventing
  one would be speculative given this project's stated principle of using
  only exact, non-invented metrics.

## New attribute grammar

Three new attributes, added to `commands.py`:

| Attribute | Peak index/indices | Relative verbs | Absolute (`set`) support |
|---|---|---|---|
| `width` | index 1 | increase/decrease width, sharpen (decrease), soften (increase) | Yes — `low/medium/high width for X` |
| `brightness` | indices 3, 4, 5 (r, g, b together) | increase/decrease brightness, brighten (increase), darken (decrease) | Yes — `low/medium/high brightness for X` |
| `center` | index 0 | increase/decrease center, shift/move up (increase), shift/move down (decrease) | **No** — see Safety below |

`ATTRIBUTE_PARAM_INDEX = {"opacity": 2, "width": 1, "brightness": (3, 4, 5), "center": 0}`
in `commands.py`, used by both `apply_command` and the new asymptotic-step
helper to know which float(s) to touch.

### Regex patterns (rule parser)

Extends `parse_command_rule`, checked in this order (most specific first, so
a dedicated verb like "sharpen" is caught before it could ever fall through
to something generic):

1. **Dedicated verbs** — `sharpen`/`soften`/`brighten`/`darken`:
   ```python
   DIRECT_VERBS = {
       "sharpen": ("width", "decrease"),
       "soften": ("width", "increase"),
       "brighten": ("brightness", "increase"),
       "darken": ("brightness", "decrease"),
   }
   m = re.search(r"\b(sharpen|soften|brighten|darken)\b\s+([\w ]+)", t)
   ```
   tissue resolved via `_find_tissue` on the captured text (same
   strength-word stripping as the existing increase/decrease pattern).

2. **Center shift** — `shift`/`move` + tissue + `up`/`down`/`higher`/`lower`:
   ```python
   m = re.search(r"\b(?:shift|move)\s+([\w ]+?)\s+(up|down|higher|lower)\b", t)
   ```
   `up`/`higher` → `direction: "increase"`; `down`/`lower` → `"decrease"`;
   `attribute: "center"`.

3. **Generalized relative pattern** (replaces the current opacity-only one):
   ```python
   m = re.search(r"(increase|decrease)\s+(opacity|width|sharpness|brightness)\s+for\s+([\w ]+)", t)
   ```
   `"sharpness"` maps to `attribute: "width"` (a synonym, not a new
   attribute — asking to "increase sharpness" and "decrease width" are the
   same operation from opposite ends, but a plain text match on the literal
   word keeps the regex simple: `"sharpness"` written after increase/decrease
   is treated as a width synonym, mapped via a small
   `ATTRIBUTE_WORD_ALIASES = {"sharpness": "width"}` lookup).

4. **Generalized absolute-level pattern** (replaces the current opacity-only
   one; deliberately excludes `center` — see Safety):
   ```python
   m = re.search(r"(low|medium|high)\s+(opacity|width|sharpness|brightness)\s+(?:for\s+)?(\w+)", t)
   ```

The existing `show_only`, compound-level, and `reset` patterns are
untouched.

### LLM parser

`_SYSTEM_PROMPT` gets one additional paragraph documenting the expanded
`attribute` enum (`opacity | width | brightness | center`) and the rule that
`center` never appears with `direction: "set"`. `_validate_cmd` gets one new
check: reject (fall back to rule parser) a `set`/`level` command whose
`attribute` is `"center"`, since that combination is never valid.

## `apply_command` changes

Replace the height-only step logic with:

```python
def _asymptotic_step(current_ext: float, direction: str, delta: float) -> float:
    if direction == "increase":
        return current_ext + delta * (1.0 - current_ext)
    return current_ext - delta * (current_ext + 1.0)
```

used uniformly for `opacity`/`width`/`brightness`/`center` relative
commands, replacing the current height-specific arithmetic inline in
`apply_command`. For `brightness`, the same delta is applied independently
to each of the three RGB indices (each channel steps toward its own +1/-1
asymptote, which preserves relative hue reasonably well without needing an
HSV round-trip). For absolute `set`, `_from_unit(LEVEL_WORDS[level])` is
written directly to the target index/indices (or all three for brightness).

## Safety: `center` band-clamping

A `center` shift is clamped to stay within the **target tissue's own**
`TISSUE_BANDS` range, converted to external units via
`_from_range(TISSUE_BANDS[tissue][0/1], *CENTER_RANGE)`. Without this, a
few repeated "shift bone's center down" commands could walk a peak's center
out of `bone`'s band entirely and into `spongy`'s territory — silently
turning a "bone" peak into something else, with no way for the user to
notice except by looking at the render. Clamping (rather than rejecting the
command once the edge is hit) matches how `propose_step`'s height clamping
already behaves — the command still succeeds, it just stops moving once it
reaches the edge of its own tissue's band. This is also *why* `center` has
no absolute `set`/level support: "high center for bone" has no obvious
mapping to a band-relative fraction that would read naturally to a user
typing/speaking a command, unlike "high opacity" or "high brightness" where
0/1 has an intuitive meaning.

## `evaluate.objective()` change

One new early check, before the existing opacity-specific increase/decrease
branch:

```python
if cmd.get("attribute") in ("width", "brightness", "center"):
    return 1
```

Matches the existing treatment of `compound`/`reset` (always accepted) —
there is no established exact metric for these attributes, and inventing
one now would be speculative. This means hill-climbing search and the RL
agent remain opacity-only; the new attributes are one-shot chat adjustments,
not something searched or trained against. This is an explicit scope
boundary, not an oversight — flagged here so it reads as a decision in the
thesis, not a gap.

## Command overview

### Data source: `commands.py::COMMAND_REFERENCE`

A single list of dicts, the one source of truth for both the UI panel and
the generated doc:

```python
COMMAND_REFERENCE = [
    {
        "category": "Opacity (relative)",
        "examples": ["increase opacity for bone strongly", "decrease opacity for fat slightly"],
        "description": "Nudge a tissue's visibility up or down by a relative amount.",
    },
    {
        "category": "Opacity (absolute)",
        "examples": ["high opacity spongy", "low opacity for bone"],
        "description": "Set a tissue's visibility to a fixed low/medium/high level.",
    },
    {
        "category": "Show only",
        "examples": ["show only bone", "show only bone and spongy"],
        "description": "Isolate one or more tissues, crushing every other peak to zero.",
    },
    {
        "category": "Compound",
        "examples": ["high opacity spongy, low opacity bones"],
        "description": "Set several tissues' absolute levels in one command.",
    },
    {
        "category": "Width / sharpness",
        "examples": ["sharpen the bone peak", "soften soft tissue", "increase width for fat"],
        "description": "Adjust how spread out (blended) or narrow (selective) a tissue's peak is.",
    },
    {
        "category": "Brightness",
        "examples": ["brighten bone", "darken fat", "low brightness for spongy"],
        "description": "Adjust a tissue's color brightness.",
    },
    {
        "category": "Center position",
        "examples": ["shift bone's center up", "move fat down"],
        "description": "Nudge where in Hounsfield space a tissue's peak sits, clamped to that tissue's own band.",
    },
    {
        "category": "Reset",
        "examples": ["reset"],
        "description": "Return the transfer function to its default state.",
    },
]
```

### `/api/commands` endpoint (`server.py`)

```python
@app.get("/api/commands")
async def commands_reference():
    return {"commands": COMMAND_REFERENCE}
```

### UI help panel (`static/`)

- `index.html`: a `?` icon button in `#app-bar-right` (next to the session
  indicator), and a `<dialog id="commands-modal">` (native `<dialog>`
  element — no extra library) containing an empty `#commands-list` div to be
  populated from the endpoint.
- `app.js`: on button click, fetch `/api/commands` (cached in a module-level
  variable after the first successful fetch — the reference data is static
  per server run, no need to refetch every open), render each category as a
  `<div class="cmd-category">` with its description and example chips, call
  `.showModal()`.
- `style.css`: modal styling consistent with the existing "clinical console"
  dark theme (reusing `--panel`, `--raised`, `--accent`, `--mono` tokens
  already defined).

### `COMMANDS.md` (repo doc)

- `tools/gen_commands_doc.py`: a small script that imports
  `commands.COMMAND_REFERENCE` and renders it to Markdown (one `##` heading
  per category, description, examples as a bullet list with inline code
  formatting), writing `COMMANDS.md` at the repo root.
- `tests/test_commands_doc.py`: regenerates the Markdown into a string
  (calling the same rendering function used by the script, not re-invoking
  the script as a subprocess) and asserts it is byte-identical to the
  committed `COMMANDS.md` — catches the reference and the doc drifting apart
  the same way the project already catches other real bugs via tests, not
  via manual discipline.

## Testing (attributes + `apply_command`)

- `tests/test_commands.py` gets new cases: each dedicated verb
  (`sharpen`/`soften`/`brighten`/`darken`) parses to the correct
  `(attribute, direction)`; the center shift pattern parses `up`/`down`/
  `higher`/`lower` correctly; the generalized relative/absolute patterns
  correctly map `"sharpness"` to `attribute: "width"`; a `set`/`level`
  command with `attribute: "center"` is rejected by `_validate_cmd` (LLM
  path) and never produced by the rule parser (no regex path can produce
  it, by construction — verified by asserting no rule-parser input reaches
  it, not by testing a case that "shouldn't parse").
- `tests/test_commands.py` (apply_command): a `width`/`brightness`/`center`
  increase/decrease command actually moves the expected index/indices via
  `_asymptotic_step`; a center shift that repeatedly increases eventually
  clamps at the tissue band's own upper edge (not the global `CENTER_RANGE`
  edge) — this is the one safety property that must have a test, given it's
  the entire justification for center having no absolute-level support.
- `tests/test_evaluate.py`: a `width`/`brightness`/`center` command's
  `objective()` call always returns `1` regardless of what changed between
  `params_before`/`params_after`.

## Explicitly out of scope

- Full hue control (`"make it redder"`) — brightness only, as decided during
  design. Revisit if it turns out to matter.
- Any RL/hill-climbing support for the new attributes (see `objective()`
  change above) — one-shot chat adjustments only.
- Voice/ASR-specific handling for the new verbs — they flow through the
  same `parse_command_rule`/`parse_command_llm` entry points as everything
  else; no changes needed in `mvp.py`'s or `server.py`'s transcription path.
