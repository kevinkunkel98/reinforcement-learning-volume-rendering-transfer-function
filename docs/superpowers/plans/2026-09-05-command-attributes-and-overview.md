# New Command Attributes + Command Overview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `width`, `brightness`, and `center` command attributes (alongside the existing `opacity`), generalizing `apply_command`'s step logic to work for any of them, and add a command overview (in-app help panel + generated `COMMANDS.md`) driven by one shared data source.

**Architecture:** `commands.py` gains a generic `_asymptotic_step` helper (replacing height-only arithmetic) and an `ATTRIBUTE_PARAM_INDEX` map so `apply_command` handles any attribute uniformly; new regex patterns extend the rule parser; the LLM prompt and validator get small additions; `evaluate.objective()` gets one new early-return; a `COMMAND_REFERENCE` list drives both a new `/api/commands` endpoint and a generated `COMMANDS.md`.

**Tech Stack:** Python (existing stack, no new dependencies), FastAPI, vanilla JS/HTML/CSS, pytest.

Spec: `docs/superpowers/specs/2026-09-05-command-attributes-and-overview-design.md`

---

## File Structure

- Modify: `commands.py` — new attribute constants, generic step helper, `apply_command` generalization, new regex patterns, LLM prompt + validator additions, `COMMAND_REFERENCE`.
- Modify: `evaluate.py` — one new early-return in `objective()`.
- Modify: `server.py` — new `/api/commands` endpoint.
- Modify: `static/index.html`, `static/app.js`, `static/style.css` — help panel UI.
- Create: `tools/gen_commands_doc.py` — renders `COMMAND_REFERENCE` to Markdown.
- Create: `COMMANDS.md` — generated output, committed (so it's browsable on GitHub without running anything).
- Modify: `tests/test_commands.py` — new cases for attributes, regex patterns, `apply_command` generalization, clamp safety.
- Modify: `tests/test_evaluate.py` — new case for the new-attribute early-return.
- Create: `tests/test_commands_doc.py` — doc/reference drift check.
- Modify: `tests/test_server.py` — new case for `/api/commands`.

---

### Task 1: Generic attribute step logic in `apply_command`

**Files:**
- Modify: `commands.py`
- Modify: `tests/test_commands.py`

This task is a **regression-safe refactor**: opacity behavior must be
byte-for-byte identical before and after, verified by the full existing
`test_commands.py` suite still passing unchanged, plus new tests for the
three new attributes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_commands.py`:

```python
def test_apply_command_width_increase_widens_peak():
    params = default_params()
    before_w = peak_internal(params, 3)["width"]  # bone is peak index 3
    cmd = {"target": "bone", "attribute": "width", "direction": "increase", "strength": "strongly"}
    after = apply_command(cmd, params)
    after_w = peak_internal(after, 3)["width"]
    assert after_w > before_w


def test_apply_command_width_decrease_narrows_peak():
    params = default_params()
    before_w = peak_internal(params, 3)["width"]
    cmd = {"target": "bone", "attribute": "width", "direction": "decrease", "strength": "strongly"}
    after = apply_command(cmd, params)
    after_w = peak_internal(after, 3)["width"]
    assert after_w < before_w


def test_apply_command_brightness_increase_raises_all_channels():
    params = default_params()
    before_rgb = peak_internal(params, 3)["rgb"]
    cmd = {"target": "bone", "attribute": "brightness", "direction": "increase", "strength": "strongly"}
    after = apply_command(cmd, params)
    after_rgb = peak_internal(after, 3)["rgb"]
    for before_c, after_c in zip(before_rgb, after_rgb):
        assert after_c >= before_c


def test_apply_command_brightness_decrease_lowers_all_channels():
    params = default_params()
    before_rgb = peak_internal(params, 3)["rgb"]
    cmd = {"target": "bone", "attribute": "brightness", "direction": "decrease", "strength": "strongly"}
    after = apply_command(cmd, params)
    after_rgb = peak_internal(after, 3)["rgb"]
    for before_c, after_c in zip(before_rgb, after_rgb):
        assert after_c <= before_c


def test_apply_command_center_increase_shifts_center_up():
    params = default_params()
    before_c = peak_internal(params, 3)["center"]  # bone, defaults near 900 HU
    cmd = {"target": "bone", "attribute": "center", "direction": "increase", "strength": "slightly"}
    after = apply_command(cmd, params)
    after_c = peak_internal(after, 3)["center"]
    assert after_c > before_c


def test_apply_command_center_clamps_to_own_tissue_band():
    from transfer import TISSUE_BANDS
    params = default_params()
    cmd = {"target": "bone", "attribute": "center", "direction": "increase", "strength": "strongly"}
    for _ in range(50):
        params = apply_command(cmd, params)
    final_c = peak_internal(params, 3)["center"]
    lo, hi = TISSUE_BANDS["bone"]
    assert final_c <= hi + 1e-6
    assert final_c >= lo - 1e-6


def test_apply_command_set_level_width():
    params = default_params()
    cmd = {"target": "bone", "attribute": "width", "direction": "set", "level": "high"}
    after = apply_command(cmd, params)
    from transfer import WIDTH_RANGE
    after_w = peak_internal(after, 3)["width"]
    # LEVEL_WORDS["high"] = 0.85 of the way through WIDTH_RANGE
    expected = WIDTH_RANGE[0] + 0.85 * (WIDTH_RANGE[1] - WIDTH_RANGE[0])
    assert abs(after_w - expected) < 1.0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_commands.py -k "width or brightness or center" -v`
Expected: FAIL — `apply_command` doesn't know about these attributes yet
(it always writes to `base + 2`, the height field, regardless of
`cmd["attribute"]`).

- [ ] **Step 3: Implement the generalization**

Edit `commands.py`'s import line (currently):
```python
from transfer import (
    CENTER_RANGE, N_PEAKS, PARAMS_PER_PEAK, TISSUE_HU, WIDTH_RANGE,
    _from_range, _from_unit, _unit, default_params, peak_internal,
)
```
to add `TISSUE_BANDS`:
```python
from transfer import (
    CENTER_RANGE, N_PEAKS, PARAMS_PER_PEAK, TISSUE_BANDS, TISSUE_HU,
    WIDTH_RANGE, _from_range, _from_unit, _unit, default_params, peak_internal,
)
```

Replace the entire `apply_command` function (and add the two new helpers +
`ATTRIBUTE_PARAM_INDEX` immediately before it, right after the existing
`LEVEL_WORDS = {"low": 0.15, "medium": 0.5, "high": 0.85}` line):

```python
LEVEL_WORDS = {"low": 0.15, "medium": 0.5, "high": 0.85}

ATTRIBUTE_PARAM_INDEX = {"opacity": 2, "width": 1, "brightness": (3, 4, 5), "center": 0}


def _asymptotic_step(current_ext: float, direction: str, delta: float) -> float:
    # Move a fraction of the remaining headroom to the +1/-1 bound, not a
    # fixed absolute amount -- generalizes the height-only fix (a fixed step
    # saturated almost any starting value in one command) to any attribute
    # stored in the same normalized external [-1, 1] encoding.
    if direction == "increase":
        return current_ext + delta * (1.0 - current_ext)
    return current_ext - delta * (current_ext + 1.0)


def _center_band_clip(ext_value: float, tissue: str) -> float:
    # A center shift must stay within the target tissue's own HU band --
    # otherwise repeated shifts could walk a peak out of its own tissue
    # entirely and into a neighboring one's territory, silently.
    lo_hu, hi_hu = TISSUE_BANDS[tissue]
    lo_ext = _from_range(lo_hu, *CENTER_RANGE)
    hi_ext = _from_range(hi_hu, *CENTER_RANGE)
    return float(np.clip(ext_value, lo_ext, hi_ext))


def apply_command(cmd: dict, params: np.ndarray) -> np.ndarray:
    if "compound" in cmd:
        # Multiple tissues set to different absolute levels in one utterance
        # ("high opacity spongy, low opacity bone") -- fold each sub-command
        # through this same function in sequence. Not a search target: an
        # absolute multi-tissue assignment isn't something to hill-climb.
        for sub in cmd["compound"]:
            params = apply_command(sub, params)
        return params

    params = params.copy()

    if cmd["direction"] == "reset":
        return default_params()

    if cmd["direction"] == "show_only":
        targets = cmd["target"] if isinstance(cmd["target"], list) else [cmd["target"]]
        target_idxs = set()
        for tissue in targets:
            params, idx = _find_or_create_peak(params, tissue)
            target_idxs.add(idx)
        for i in range(N_PEAKS):
            b = i * PARAMS_PER_PEAK
            params[b + 2] = _from_unit(0.7 if i in target_idxs else 0.0)
        return params

    params, idx = _find_or_create_peak(params, cmd["target"])
    base = idx * PARAMS_PER_PEAK
    indices = ATTRIBUTE_PARAM_INDEX[cmd["attribute"]]
    if isinstance(indices, int):
        indices = (indices,)

    if cmd["direction"] == "set":
        new_ext = _from_unit(LEVEL_WORDS[cmd["level"]])
        for offset in indices:
            params[base + offset] = new_ext
        return params

    delta = STRENGTH_WORDS[cmd["strength"] or "moderately"]
    for offset in indices:
        new_ext = _asymptotic_step(float(params[base + offset]), cmd["direction"], delta)
        if cmd["attribute"] == "center":
            new_ext = _center_band_clip(new_ext, cmd["target"])
        params[base + offset] = float(np.clip(new_ext, -1.0, 1.0))
    return params
```

Note this is a pure refactor for the `opacity` path: previously the code
worked in unit `[0, 1]` space (`current_h = _unit(params[base+2])`, `new_h =
current_h + delta * (1 - current_h)`, then `_from_unit(new_h)` back to
external space). The new `_asymptotic_step` operates directly on the
external `[-1, 1]` value and is algebraically equivalent (substituting
`ext = 2*u - 1` into the old formula yields exactly `ext + delta * (1 -
ext)`) — this is why Step 2's existing opacity tests are the regression
check, not just the new attribute tests.

- [ ] **Step 4: Run the full `test_commands.py` file**

Run: `.venv/bin/python -m pytest tests/test_commands.py -v`
Expected: every existing test still passes (unchanged behavior for
`opacity`), plus all 7 new tests from Step 1 pass.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 7 from before this plan started.

- [ ] **Step 6: Commit**

```bash
git add commands.py tests/test_commands.py
git commit -m "Generalize apply_command's step logic to width/brightness/center attributes"
```

---

### Task 2: Rule-parser grammar for the new attributes

**Files:**
- Modify: `commands.py`
- Modify: `tests/test_commands.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_commands.py`:

```python
def test_parse_sharpen_verb():
    cmd = parse_command_rule("sharpen the bone peak")
    assert cmd["target"] == "bone"
    assert cmd["attribute"] == "width"
    assert cmd["direction"] == "decrease"


def test_parse_soften_verb():
    cmd = parse_command_rule("soften soft tissue")
    assert cmd["target"] == "soft"
    assert cmd["attribute"] == "width"
    assert cmd["direction"] == "increase"


def test_parse_brighten_verb():
    cmd = parse_command_rule("brighten bone strongly")
    assert cmd["target"] == "bone"
    assert cmd["attribute"] == "brightness"
    assert cmd["direction"] == "increase"
    assert cmd["strength"] == "strongly"


def test_parse_darken_verb():
    cmd = parse_command_rule("darken fat")
    assert cmd["target"] == "fat"
    assert cmd["attribute"] == "brightness"
    assert cmd["direction"] == "decrease"


def test_parse_shift_center_up_with_apostrophe():
    cmd = parse_command_rule("shift bone's center up")
    assert cmd["target"] == "bone"
    assert cmd["attribute"] == "center"
    assert cmd["direction"] == "increase"


def test_parse_move_down_without_center_word():
    cmd = parse_command_rule("move fat down")
    assert cmd["target"] == "fat"
    assert cmd["attribute"] == "center"
    assert cmd["direction"] == "decrease"


def test_parse_increase_width_generalized():
    cmd = parse_command_rule("increase width for fat")
    assert cmd["target"] == "fat"
    assert cmd["attribute"] == "width"
    assert cmd["direction"] == "increase"


def test_parse_increase_sharpness_maps_to_width():
    cmd = parse_command_rule("increase sharpness for spongy strongly")
    assert cmd["target"] == "spongy"
    assert cmd["attribute"] == "width"


def test_parse_low_brightness_absolute():
    cmd = parse_command_rule("low brightness for spongy")
    assert cmd == {"target": "spongy", "attribute": "brightness", "direction": "set", "level": "low"}


def test_parse_high_sharpness_absolute_maps_to_width():
    cmd = parse_command_rule("high sharpness bone")
    assert cmd["attribute"] == "width"
    assert cmd["level"] == "high"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_commands.py -k "verb or center_up or without_center or generalized or sharpness or brightness_absolute" -v`
Expected: FAIL — `parse_command_rule` raises `ValueError` for all of these
(none of these patterns exist yet).

- [ ] **Step 3: Implement the new patterns**

Add these two module-level constants right after `STRENGTH_WORDS` /
`NEAR_THRESHOLD_HU` (before `_find_tissue`):

```python
DIRECT_VERBS = {
    "sharpen": ("width", "decrease"),
    "soften": ("width", "increase"),
    "brighten": ("brightness", "increase"),
    "darken": ("brightness", "decrease"),
}

ATTRIBUTE_WORD_ALIASES = {"sharpness": "width"}
```

Then edit `parse_command_rule`, inserting two new checks after the
`show only` block and before the existing absolute-level block, and
generalizing the absolute-level and relative regexes:

```python
def parse_command_rule(text: str) -> dict:
    t = text.lower().strip()

    if "reset" in t:
        return {"target": None, "attribute": None, "direction": "reset", "strength": None}

    m = re.search(r"show only ([\w ,\+]+)", t)
    if m:
        tissues = _split_tissue_list(m.group(1))
        if tissues:
            return {"target": tissues[0] if len(tissues) == 1 else tissues,
                     "attribute": "opacity", "direction": "show_only", "strength": None}

    m = re.search(r"\b(sharpen|soften|brighten|darken)\b\s+([\w ]+)", t)
    if m:
        verb, target_text = m.group(1), m.group(2)
        attribute, direction = DIRECT_VERBS[verb]
        strength = "moderately"
        for word in STRENGTH_WORDS:
            if word in target_text:
                strength = word
                target_text = target_text.replace(word, "")
        tissue = _find_tissue(target_text)
        if tissue:
            return {"target": tissue, "attribute": attribute,
                     "direction": direction, "strength": strength}

    m = re.search(r"\b(?:shift|move)\s+([\w ]+?)(?:'s)?\s+(?:center|position)?\s*(up|down|higher|lower)\b", t)
    if m:
        target_text, word = m.group(1), m.group(2)
        tissue = _find_tissue(target_text)
        if tissue:
            direction = "increase" if word in ("up", "higher") else "decrease"
            return {"target": tissue, "attribute": "center",
                     "direction": direction, "strength": "moderately"}

    # "high opacity spongy", "low opacity for bone", "high sharpness bone" --
    # an absolute level per tissue+attribute, not a relative delta. One or
    # more may appear in the same sentence, each becomes its own
    # sub-command, folded together into one compound command.
    level_matches = list(re.finditer(
        r"(low|medium|high)\s+(opacity|width|sharpness|brightness)\s+(?:for\s+)?(\w+)", t))
    if level_matches:
        subcommands = []
        for lm in level_matches:
            level, attr_word, tissue_text = lm.group(1), lm.group(2), lm.group(3)
            attribute = ATTRIBUTE_WORD_ALIASES.get(attr_word, attr_word)
            tissue = _find_tissue(tissue_text)
            if tissue:
                subcommands.append({"target": tissue, "attribute": attribute,
                                      "direction": "set", "level": level})
        if subcommands:
            return subcommands[0] if len(subcommands) == 1 else {"compound": subcommands}

    m = re.search(r"(increase|decrease)\s+(opacity|width|sharpness|brightness)\s+for\s+([\w ]+)", t)
    if m:
        direction, attr_word, target_text = m.group(1), m.group(2), m.group(3)
        attribute = ATTRIBUTE_WORD_ALIASES.get(attr_word, attr_word)
        strength = "moderately"
        for word in STRENGTH_WORDS:
            if word in target_text:
                strength = word
                target_text = target_text.replace(word, "")
        tissue = _find_tissue(target_text)
        if tissue:
            return {"target": tissue, "attribute": attribute,
                     "direction": direction, "strength": strength}

    raise ValueError(f"rule parser cannot parse: {text!r}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_commands.py -v`
Expected: all previous tests plus all new ones from this task pass (verify
the `shift bone's center up` and `move fat down` cases specifically — the
apostrophe and optional "center"/"position" word are the two trickiest
parts of this regex).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add commands.py tests/test_commands.py
git commit -m "Add rule-parser grammar for width, brightness, and center commands"
```

---

### Task 3: LLM parser prompt + validator update

**Files:**
- Modify: `commands.py`
- Modify: `tests/test_commands.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_commands.py`:

```python
def test_validate_rejects_center_with_set_direction():
    from commands import _validate_cmd
    bad = {"compound": [
        {"target": "bone", "attribute": "center", "direction": "set", "level": "high"},
    ]}
    assert _validate_cmd(bad) is False


def test_validate_accepts_width_set():
    from commands import _validate_cmd
    good = {"compound": [
        {"target": "bone", "attribute": "width", "direction": "set", "level": "high"},
    ]}
    assert _validate_cmd(good) is True
```

- [ ] **Step 2: Run the tests to verify the first one fails**

Run: `.venv/bin/python -m pytest tests/test_commands.py -k validate_rejects_center -v`
Expected: FAIL — `_validate_set_cmd` currently only checks keys/direction/
target/level, not the attribute value, so this currently (incorrectly)
returns `True`.

- [ ] **Step 3: Implement the fix**

In `commands.py`, edit `_validate_set_cmd`:

```python
def _validate_set_cmd(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"target", "attribute", "direction", "level"}:
        return False
    if obj["direction"] != "set":
        return False
    if obj["attribute"] == "center":
        return False
    if not isinstance(obj["target"], str) or obj["target"] not in TISSUE_HU:
        return False
    if obj["level"] not in VALID_LEVELS:
        return False
    return True
```

Then update `_SYSTEM_PROMPT`'s output-schema section to document the wider
attribute enum and the center/set restriction. Edit the existing schema
block:

```python
Output schema -- the usual case is a single command:
{{"target": "<tissue>|[<tissue>, ...]|null", "attribute": "opacity"|null,
 "direction": "increase"|"decrease"|"show_only"|"reset",
 "strength": "slightly"|"moderately"|"strongly"|null}}
```

to:

```python
Output schema -- the usual case is a single command:
{{"target": "<tissue>|[<tissue>, ...]|null", "attribute": "opacity"|null,
 "direction": "increase"|"decrease"|"show_only"|"reset",
 "strength": "slightly"|"moderately"|"strongly"|null}}

"attribute" is usually "opacity", but can also be "width" (how spread out /
sharp a tissue's peak is -- "sharpen"/"soften" mean decrease/increase width)
, "brightness" (how light/dark a tissue's color is -- "brighten"/"darken"
mean increase/decrease), or "center" (where in Hounsfield space the peak
sits -- "shift up"/"shift down" mean increase/decrease). These follow the
same increase/decrease/strength shape as opacity.
```

And add one sentence right after the existing compound-command
documentation (after the `"level" is "low"|"medium"|"high"...` line):

```python
"level" is "low"|"medium"|"high" -- an absolute target, not a relative change.
Use "set"/"level" only inside a compound command, never "strength" there.
"center" never takes an absolute "set"/"level" -- only increase/decrease.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_commands.py -v`
Expected: all pass, including both new tests.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add commands.py tests/test_commands.py
git commit -m "Reject center attribute with absolute set in LLM validator, document new attributes in prompt"
```

---

### Task 4: `evaluate.objective()` treats new attributes as always-accepted

**Files:**
- Modify: `evaluate.py`
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Read `tests/test_evaluate.py`'s existing imports/fixtures first (to match
its style), then append:

```python
def test_objective_always_accepts_width_brightness_center():
    from transfer import default_params
    params = default_params()
    for attribute in ("width", "brightness", "center"):
        cmd = {"target": "bone", "attribute": attribute, "direction": "increase"}
        assert objective(params, params, cmd) == 1  # even a no-op change is accepted
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -k always_accepts -v`
Expected: FAIL — `objective()` currently falls through to the
opacity-specific `TISSUE_BANDS[cmd["target"]]`/`opacity_mass` branch for any
non-`compound`/`reset`/`show_only` command, regardless of `attribute`.

- [ ] **Step 3: Implement the fix**

In `evaluate.py`, edit `objective()` — add one new early check right after
the existing `compound`/`reset` check and before the `show_only` check:

```python
def objective(params_before, params_after, cmd: dict) -> int:
    if "compound" in cmd or cmd["direction"] == "reset":
        return 1

    if cmd.get("attribute") in ("width", "brightness", "center"):
        # No established exact metric for these attributes (opacity_mass is
        # opacity-specific) -- treat as always accepted, same as compound/reset.
        return 1

    if cmd["direction"] == "show_only":
        ...
```
(keep the rest of the function body unchanged below this point).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "Treat width/brightness/center commands as always-accepted in objective()"
```

---

### Task 5: `COMMAND_REFERENCE` data + generated `COMMANDS.md`

**Files:**
- Modify: `commands.py`
- Create: `tools/gen_commands_doc.py`
- Create: `COMMANDS.md`
- Create: `tests/test_commands_doc.py`

- [ ] **Step 1: Add `COMMAND_REFERENCE` to `commands.py`**

Add this near the top of `commands.py`, right after `TISSUE_SYNONYMS`/
`_SYNONYM_LOOKUP` (before `STRENGTH_WORDS`):

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

- [ ] **Step 2: Write the failing test for the doc generator**

Create `tests/test_commands_doc.py`:

```python
"""Regenerates COMMANDS.md from commands.COMMAND_REFERENCE and asserts it
matches the committed file -- catches the reference and the doc drifting
apart, rather than relying on someone remembering to regenerate by hand."""
from tools.gen_commands_doc import render_commands_markdown


def test_committed_commands_md_matches_generated_output():
    generated = render_commands_markdown()
    with open("COMMANDS.md") as f:
        committed = f.read()
    assert generated == committed
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_commands_doc.py -v`
Expected: FAIL — `tools/gen_commands_doc.py` doesn't exist yet.

- [ ] **Step 4: Implement `tools/gen_commands_doc.py`**

```python
"""Generates COMMANDS.md from commands.COMMAND_REFERENCE -- the one source
of truth also served by server.py's /api/commands endpoint for the in-app
help panel. Run this script whenever COMMAND_REFERENCE changes."""
from commands import COMMAND_REFERENCE


def render_commands_markdown() -> str:
    lines = ["# Supported Commands", ""]
    lines.append(
        "Generated from `commands.COMMAND_REFERENCE` -- run "
        "`python -m tools.gen_commands_doc` after changing it."
    )
    lines.append("")
    for entry in COMMAND_REFERENCE:
        lines.append(f"## {entry['category']}")
        lines.append("")
        lines.append(entry["description"])
        lines.append("")
        for example in entry["examples"]:
            lines.append(f"- `{example}`")
        lines.append("")
    return "\n".join(lines)


def main():
    content = render_commands_markdown()
    with open("COMMANDS.md", "w") as f:
        f.write(content)
    print("Wrote COMMANDS.md")


if __name__ == "__main__":
    main()
```

Also create `tools/__init__.py` (empty file) so `tools.gen_commands_doc` is
importable as a module.

- [ ] **Step 5: Generate `COMMANDS.md`**

Run: `.venv/bin/python -m tools.gen_commands_doc`
Expected: prints `Wrote COMMANDS.md`, creates the file at the repo root.

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_commands_doc.py -v`
Expected: 1 passed.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add commands.py tools/__init__.py tools/gen_commands_doc.py COMMANDS.md tests/test_commands_doc.py
git commit -m "Add COMMAND_REFERENCE and generated COMMANDS.md"
```

---

### Task 6: `/api/commands` endpoint

**Files:**
- Modify: `server.py`

**No automated test for this task.** `tests/test_server.py` has a
documented, deliberate constraint (see the file's own top-of-file comment):
FastAPI's `TestClient` runs the ASGI app on a background thread, and VTK's
Cocoa render window can only be created on the true process main thread on
macOS — routing through `TestClient` crashes the interpreter. That file
only ever tests the framework-independent `Session` class directly, never
routes. `/api/commands` is a trivial stateless passthrough (like the
existing `/api/datasets`, which also has no dedicated test in this file for
the same reason) — it doesn't touch `Session` at all, so there is nothing
for that file's testing approach to exercise. This task is verified by a
manual `curl` check instead (Step 3).

- [ ] **Step 1: Add `COMMAND_REFERENCE` to the existing `commands` import**

`server.py` already imports from `commands` (line 28):
```python
from commands import STRENGTH_WORDS, _find_or_create_peak, apply_command, parse_command
```
Change it to:
```python
from commands import COMMAND_REFERENCE, STRENGTH_WORDS, _find_or_create_peak, apply_command, parse_command
```

- [ ] **Step 2: Add the endpoint**

Add this new route in `server.py` immediately after the existing
`/api/datasets` route (around line 358-360):

```python
@app.get("/api/datasets")
async def datasets_list():
    return {"available": list_datasets(), "current": _dataset_name}


@app.get("/api/commands")
async def commands_reference():
    return {"commands": COMMAND_REFERENCE}
```

(Only the new `commands_reference` function is being added — `datasets_list`
is shown above only to mark the exact insertion point immediately after it.)

- [ ] **Step 3: Manually verify**

Restart the running server (see Task 7 Step 4 for how) and run:
```bash
curl -s http://127.0.0.1:8000/api/commands | python3 -m json.tool | head -20
```
Expected: valid JSON, top-level `"commands"` key, a list of 8 objects each
with `category`/`examples`/`description` keys.

- [ ] **Step 4: Run the full suite to confirm nothing broke**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (this task adds no new tests, so the count is unchanged
from Task 5).

- [ ] **Step 5: Commit**

```bash
git add server.py
git commit -m "Add /api/commands endpoint serving the command reference"
```

---

### Task 7: In-app command overview panel

**Files:**
- Modify: `static/index.html`
- Modify: `static/app.js`
- Modify: `static/style.css`

This task has no automated test (it's pure frontend UI) — it is verified by
starting the server and checking the panel manually, per the project's
existing convention for frontend changes.

- [ ] **Step 1: Add the trigger button and modal markup to `index.html`**

In `#app-bar-right` (currently just the session indicator span), add a
button before the session indicator:

```html
<div id="app-bar-right">
  <button id="commands-help-btn" class="icon-btn" aria-label="Command reference" title="Command reference">?</button>
  <span id="session-indicator"><span class="dot"></span>LOKALE SITZUNG</span>
</div>
```

Add the modal markup right before the closing `</div>` of `#app` (after
`#main`, as a sibling, so it isn't clipped by any panel's overflow):

```html
<dialog id="commands-modal">
  <div id="commands-modal-header">
    <h2>Command Reference</h2>
    <button id="commands-modal-close" class="icon-btn" aria-label="Close">×</button>
  </div>
  <div id="commands-list"></div>
</dialog>
```

- [ ] **Step 2: Add the fetch/render/open logic to `app.js`**

Add near the other `el(...)`-based event wiring (find where other buttons
like `reset-btn` get their click handlers, and add this alongside):

```javascript
let commandsCache = null;

async function openCommandsModal() {
  const modal = el("commands-modal");
  if (!commandsCache) {
    const res = await fetch("/api/commands");
    const data = await res.json();
    commandsCache = data.commands;
    const list = el("commands-list");
    list.innerHTML = "";
    for (const entry of commandsCache) {
      const section = document.createElement("div");
      section.className = "cmd-category";
      const title = document.createElement("h3");
      title.textContent = entry.category;
      const desc = document.createElement("p");
      desc.textContent = entry.description;
      const examples = document.createElement("div");
      examples.className = "cmd-examples";
      for (const example of entry.examples) {
        const chip = document.createElement("code");
        chip.className = "cmd-example-chip";
        chip.textContent = example;
        examples.appendChild(chip);
      }
      section.appendChild(title);
      section.appendChild(desc);
      section.appendChild(examples);
      list.appendChild(section);
    }
  }
  modal.showModal();
}

el("commands-help-btn").addEventListener("click", openCommandsModal);
el("commands-modal-close").addEventListener("click", () => el("commands-modal").close());
```

Place this wiring alongside the app's existing startup event-listener
registration code (check where other `addEventListener` calls for buttons
like `reset-btn` are made, and add these two lines in the same place).

- [ ] **Step 3: Add styling to `style.css`**

Add near the other component styles (e.g. after `#app-bar` rules), reusing
the existing dark "clinical console" tokens already defined at the top of
the file (`--panel`, `--raised`, `--accent`, `--mono`, etc. — check their
exact names first):

```css
#commands-modal {
  background: var(--panel);
  color: inherit;
  border: 1px solid var(--raised);
  border-radius: 6px;
  padding: 0;
  max-width: 560px;
  width: 90vw;
  max-height: 80vh;
  overflow-y: auto;
}

#commands-modal::backdrop {
  background: rgba(0, 0, 0, 0.5);
}

#commands-modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1em 1.2em;
  border-bottom: 1px solid var(--raised);
  position: sticky;
  top: 0;
  background: var(--panel);
}

#commands-list {
  padding: 1em 1.2em;
}

.cmd-category {
  margin-bottom: 1.4em;
}

.cmd-category h3 {
  margin: 0 0 0.3em 0;
  color: var(--accent);
  font-size: 0.95em;
}

.cmd-examples {
  display: flex;
  flex-direction: column;
  gap: 0.3em;
  margin-top: 0.5em;
}

.cmd-example-chip {
  font-family: var(--mono);
  font-size: 0.85em;
  background: var(--raised);
  border-radius: 3px;
  padding: 0.3em 0.6em;
  width: fit-content;
}
```

If `--panel`/`--raised`/`--accent`/`--mono` are not the exact token names in
the current `style.css`, use whichever equivalent tokens are actually
defined there — check the top of the file for the `:root` block before
writing this.

- [ ] **Step 4: Manual verification**

Restart the running server (kill the existing process on port 8000 first if
one is running, then start a fresh one) and verify in a browser or via
`curl`:
```bash
curl -s http://127.0.0.1:8000/api/commands | head -c 300
```
Then open the UI in a browser, click the new "?" button in the app bar, and
confirm the modal opens showing all 8 categories with their examples, and
that the close button and clicking outside the modal (native `<dialog>`
backdrop behavior) both work. Take a screenshot if possible and visually
confirm it matches the dark console aesthetic rather than looking like an
unstyled browser default.

- [ ] **Step 5: Commit**

```bash
git add static/index.html static/app.js static/style.css
git commit -m "Add in-app command reference panel"
```

---

## Notes for the implementer

- Do not modify `search.py`, `render.py`, or `rl/` — untouched by this plan.
- Task order matters for testability: Tasks 1-4 (`commands.py`/`evaluate.py`
  logic) have no UI dependency and should land first; Task 5
  (`COMMAND_REFERENCE`) is needed before Task 6 (`/api/commands`), which is
  needed before Task 7 (the panel that fetches it).
- The `shift`/`move` center-command regex (Task 2) is the trickiest part of
  this plan — it must handle both `"shift bone's center up"` (apostrophe +
  explicit "center") and `"move fat down"` (no apostrophe, no "center"
  word). Both are tested explicitly; do not simplify the regex in a way
  that drops either case.
