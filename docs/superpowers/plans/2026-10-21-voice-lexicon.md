# Voice Command Lexicon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Voice commands from visualization people (vis lingo, lay anatomy, German-accent transcription errors) parse correctly within 4 s p90 from trigger release to render.

**Architecture:** One vocabulary file `data/lexicon.json`, loaded by a pure module `lexicon.py`, feeds Whisper's `initial_prompt`, a normalization step, the rule parser's word lists and the LLM prompt. A new `parse_command_hybrid` answers with the rule parser when every content word is accounted for, else asks the LLM (which already falls back to the rule parser). Whisper hallucinations are filtered before parsing. Stage timings and every utterance are logged.

**Tech Stack:** Python 3 (FastAPI server, pytest), Ollama HTTP API, faster-whisper / mlx-whisper, vanilla JS (static/app.js).

**Spec:** `docs/superpowers/specs/2026-10-21-voice-lexicon-design.md`

**Conventions in this repo:**
- Python tests: `.venv/bin/python -m pytest tests/<file> -q`. JS tests: `node --test tests/js/*.test.js`.
- Commit messages: conventional style (`feat(parser): ...`), **no** Co-Authored-By or other attribution trailers.
- In zsh use `command cat` instead of `cat` inside heredocs (cat is aliased to bat).
- `parse_command_llm` logs to `out/llm_requests.jsonl`; tests must `monkeypatch.chdir(tmp_path)` (see `tests/test_commands.py`).

## File map

| File | Responsibility |
|---|---|
| `data/parser_eval_realistic.json` (new) | realistic phrases with expected commands, dev/heldout |
| `data/lexicon.json` (new) | the vocabulary, data only |
| `lexicon.py` (new) | load + validate lexicon; normalize; coverage; cue lookup; strength regexes |
| `commands.py` | class/retired/strength lists from the lexicon; cue phrases; hybrid router; utterance log; LLM schema + prompt section; Ollama check |
| `eval_parsers.py` | `--set`, `--format`, latency columns, per-source rows, `hybrid` parser |
| `asr.py` | backend/model selection, initial_prompt, VAD, hallucination filter |
| `server.py` | hybrid default, ASR prompt + filter, timings, Ollama warm-up |
| `static/app.js` | hybrid default, `[voice-timing]` log, rejection on the VR panel |
| `tools/pair_asr_recordings.py`, `tools/eval_asr.py`, `data/asr_script.txt` (new) | audio evaluation |
| `docs/experiments/2026-10-21-voice-lexicon.md` (new) | baseline and results |
| tests: `tests/test_lexicon.py` (new), `tests/test_asr.py` (new), `tests/test_eval_asr.py` (new), `tests/test_commands.py`, `tests/test_eval_parsers.py` | |

---

### Task 1: Realistic eval set

**Files:**
- Create: `data/parser_eval_realistic.json`
- Modify: `eval_parsers.py` (constants near line 22)
- Test: `tests/test_eval_parsers.py`

The held-out rows are written **before** any lexicon work and never used to tune it. Rows authored by Claude carry `"source": "claude"`; rows from the author or colleagues carry `"author"` / `"colleague"` and are the uncontaminated measure (reported separately in Task 2).

- [ ] **Step 1: Write the failing test** (append to `tests/test_eval_parsers.py`)

```python
def test_realistic_eval_set_is_well_formed():
    with open(eval_parsers.EVAL_SETS["realistic"]) as f:
        rows = json.load(f)
    assert {r["split"] for r in rows} == {"dev", "heldout"}
    assert len({r["text"] for r in rows}) == len(rows)
    assert all(set(r) == {"text", "expected", "split", "source"} for r in rows)
    assert {r["source"] for r in rows} <= {"log-2026-10-21", "claude", "author", "colleague"}
    # both outcomes are represented: commands and refusals
    assert any(r["expected"] is None for r in rows if r["split"] == "heldout")
    assert any(r["expected"] is not None for r in rows if r["split"] == "heldout")
```

- [ ] **Step 2: Run it, expect failure**

Run: `.venv/bin/python -m pytest tests/test_eval_parsers.py -q`
Expected: FAIL, `AttributeError: module 'eval_parsers' has no attribute 'EVAL_SETS'`

- [ ] **Step 3: Add the set registry** in `eval_parsers.py` directly below `EVAL_SET_PATH = "data/parser_eval_set.json"`:

```python
EVAL_SETS = {"base": EVAL_SET_PATH, "realistic": "data/parser_eval_realistic.json"}
```

- [ ] **Step 4: Create `data/parser_eval_realistic.json`**

Dev rows 1–13 are the 2026-10-21 VR session transcripts (`out/audio/20261021-*.txt`); "Revered skeleton." is ambiguous (reveal or revert) and expected to be refused.

```json
[
 {"text": "Show only bones", "expected": {"target": "skeleton", "attribute": "opacity", "direction": "show_only", "strength": null}, "split": "dev", "source": "log-2026-10-21"},
 {"text": "add kidneys", "expected": {"target": "kidneys", "attribute": "opacity", "direction": "increase", "strength": "moderately"}, "split": "dev", "source": "log-2026-10-21"},
 {"text": "Add spleen.", "expected": {"target": "spleen", "attribute": "opacity", "direction": "increase", "strength": "moderately"}, "split": "dev", "source": "log-2026-10-21"},
 {"text": "show only skeleton and show only liver", "expected": {"target": ["skeleton", "liver"], "attribute": "opacity", "direction": "show_only", "strength": null}, "split": "dev", "source": "log-2026-10-21"},
 {"text": "REVERT", "expected": {"target": null, "attribute": null, "direction": "reset", "strength": null}, "split": "dev", "source": "log-2026-10-21"},
 {"text": "Remove soft tissue.", "expected": {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "strongly"}, "split": "dev", "source": "log-2026-10-21"},
 {"text": "Extremely dark and skeleton.", "expected": {"target": "skeleton", "attribute": "brightness", "direction": "decrease", "strength": "strongly"}, "split": "dev", "source": "log-2026-10-21"},
 {"text": "Darkin' Skeleton", "expected": {"target": "skeleton", "attribute": "brightness", "direction": "decrease", "strength": "moderately"}, "split": "dev", "source": "log-2026-10-21"},
 {"text": "Leitenlangs", "expected": {"target": "lungs", "attribute": "brightness", "direction": "increase", "strength": "moderately"}, "split": "dev", "source": "log-2026-10-21"},
 {"text": "Highlight heart.", "expected": {"target": "heart", "attribute": "opacity", "direction": "increase", "strength": "moderately"}, "split": "dev", "source": "log-2026-10-21"},
 {"text": "Revered skeleton.", "expected": null, "split": "dev", "source": "log-2026-10-21"},
 {"text": "You", "expected": null, "split": "dev", "source": "log-2026-10-21"},
 {"text": "We've heard this from Epstein survivors. And... Hard.", "expected": null, "split": "dev", "source": "log-2026-10-21"},
 {"text": "make the liver translucent", "expected": {"target": "liver", "attribute": "opacity", "direction": "decrease", "strength": "moderately"}, "split": "dev", "source": "claude"},
 {"text": "mute the bones", "expected": {"target": "skeleton", "attribute": "opacity", "direction": "decrease", "strength": "strongly"}, "split": "dev", "source": "claude"},
 {"text": "solo vessels", "expected": {"target": "vessels", "attribute": "opacity", "direction": "show_only", "strength": null}, "split": "dev", "source": "claude"},
 {"text": "more opaque heart", "expected": {"target": "heart", "attribute": "opacity", "direction": "increase", "strength": "moderately"}, "split": "dev", "source": "claude"},
 {"text": "tighter peak on the skeleton", "expected": {"target": "skeleton", "attribute": "width", "direction": "decrease", "strength": "moderately"}, "split": "dev", "source": "claude"},
 {"text": "a smidge more spleen", "expected": {"target": "spleen", "attribute": "opacity", "direction": "increase", "strength": "slightly"}, "split": "dev", "source": "claude"},
 {"text": "kill the soft tissue", "expected": {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "strongly"}, "split": "dev", "source": "claude"},
 {"text": "bring back the lungs", "expected": {"target": "lungs", "attribute": "opacity", "direction": "increase", "strength": "moderately"}, "split": "dev", "source": "claude"},
 {"text": "move the liver center down", "expected": {"target": "liver", "attribute": "center", "direction": "decrease", "strength": "moderately"}, "split": "dev", "source": "claude"},
 {"text": "crisper kidneys", "expected": {"target": "kidneys", "attribute": "width", "direction": "decrease", "strength": "moderately"}, "split": "dev", "source": "claude"},
 {"text": "softer edges on the bones", "expected": {"target": "skeleton", "attribute": "width", "direction": "increase", "strength": "moderately"}, "split": "dev", "source": "claude"},
 {"text": "fade out the guts", "expected": {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "strongly"}, "split": "dev", "source": "claude"},
 {"text": "turn the vasculature up a lot", "expected": {"target": "vessels", "attribute": "opacity", "direction": "increase", "strength": "strongly"}, "split": "dev", "source": "claude"},
 {"text": "zoom out", "expected": {"camera": {"action": "zoom", "direction": "out", "strength": "moderately"}}, "split": "dev", "source": "claude"},
 {"text": "thanks for watching", "expected": null, "split": "dev", "source": "claude"},
 {"text": "show me the fat", "expected": null, "split": "dev", "source": "claude"},

 {"text": "make the bones more opaque", "expected": {"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": "moderately"}, "split": "heldout", "source": "claude"},
 {"text": "fade out the soft tissue", "expected": {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "strongly"}, "split": "heldout", "source": "claude"},
 {"text": "ghost the muscles a bit", "expected": {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "slightly"}, "split": "heldout", "source": "claude"},
 {"text": "solo the liver", "expected": {"target": "liver", "attribute": "opacity", "direction": "show_only", "strength": null}, "split": "heldout", "source": "claude"},
 {"text": "isolate kidneys and spleen", "expected": {"target": ["kidneys", "spleen"], "attribute": "opacity", "direction": "show_only", "strength": null}, "split": "heldout", "source": "claude"},
 {"text": "crank up the lungs", "expected": {"target": "lungs", "attribute": "opacity", "direction": "increase", "strength": "strongly"}, "split": "heldout", "source": "claude"},
 {"text": "dial the heart down a touch", "expected": {"target": "heart", "attribute": "opacity", "direction": "decrease", "strength": "slightly"}, "split": "heldout", "source": "claude"},
 {"text": "make the ribcage pop", "expected": {"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": "moderately"}, "split": "heldout", "source": "claude"},
 {"text": "the muscles are hiding everything, get rid of them", "expected": {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "strongly"}, "split": "heldout", "source": "claude"},
 {"text": "alpha up on the vessels", "expected": {"target": "vessels", "attribute": "opacity", "direction": "increase", "strength": "moderately"}, "split": "heldout", "source": "claude"},
 {"text": "make the lungs see-through", "expected": {"target": "lungs", "attribute": "opacity", "direction": "decrease", "strength": "moderately"}, "split": "heldout", "source": "claude"},
 {"text": "narrower peak for the skeleton", "expected": {"target": "skeleton", "attribute": "width", "direction": "decrease", "strength": "moderately"}, "split": "heldout", "source": "claude"},
 {"text": "shift the bone peak up", "expected": {"target": "skeleton", "attribute": "center", "direction": "increase", "strength": "moderately"}, "split": "heldout", "source": "claude"},
 {"text": "brighten the kidneys a lot", "expected": {"target": "kidneys", "attribute": "brightness", "direction": "increase", "strength": "strongly"}, "split": "heldout", "source": "claude"},
 {"text": "the spleen is way too dark", "expected": {"target": "spleen", "attribute": "brightness", "direction": "increase", "strength": "strongly"}, "split": "heldout", "source": "claude"},
 {"text": "back to default", "expected": {"target": null, "attribute": null, "direction": "reset", "strength": null}, "split": "heldout", "source": "claude"},
 {"text": "undo everything", "expected": {"target": null, "attribute": null, "direction": "reset", "strength": null}, "split": "heldout", "source": "claude"},
 {"text": "turn off the soft tissue", "expected": {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "strongly"}, "split": "heldout", "source": "claude"},
 {"text": "can you show me just the heart please", "expected": {"target": "heart", "attribute": "opacity", "direction": "show_only", "strength": null}, "split": "heldout", "source": "claude"},
 {"text": "more aorta, less muscle", "expected": {"compound": [{"target": "vessels", "attribute": "opacity", "direction": "increase", "strength": "moderately"}, {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "moderately"}]}, "split": "heldout", "source": "claude"},
 {"text": "hide the bones and brighten the liver", "expected": {"compound": [{"target": "skeleton", "attribute": "opacity", "direction": "decrease", "strength": "strongly"}, {"target": "liver", "attribute": "brightness", "direction": "increase", "strength": "moderately"}]}, "split": "heldout", "source": "claude"},
 {"text": "darken the sgeleton", "expected": {"target": "skeleton", "attribute": "brightness", "direction": "decrease", "strength": "moderately"}, "split": "heldout", "source": "claude"},
 {"text": "show only the kitneys", "expected": {"target": "kidneys", "attribute": "opacity", "direction": "show_only", "strength": null}, "split": "heldout", "source": "claude"},
 {"text": "make the lever more visible", "expected": {"target": "liver", "attribute": "opacity", "direction": "increase", "strength": "moderately"}, "split": "heldout", "source": "claude"},
 {"text": "rotate left", "expected": {"camera": {"action": "rotate", "direction": "left", "strength": "moderately"}}, "split": "heldout", "source": "claude"},
 {"text": "zoom in a bit", "expected": {"camera": {"action": "zoom", "direction": "in", "strength": "slightly"}}, "split": "heldout", "source": "claude"},
 {"text": "make it look cool", "expected": null, "split": "heldout", "source": "claude"},
 {"text": "you", "expected": null, "split": "heldout", "source": "claude"},
 {"text": "show more fat", "expected": null, "split": "heldout", "source": "claude"},
 {"text": "focus on the vessels", "expected": {"target": "vessels", "attribute": "opacity", "direction": "show_only", "strength": null}, "split": "heldout", "source": "claude"},
 {"text": "the backbone is barely visible", "expected": {"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": ["moderately", "strongly"]}, "split": "heldout", "source": "claude"},
 {"text": "set the heart to high", "expected": {"target": "heart", "attribute": "opacity", "direction": "set", "level": "high"}, "split": "heldout", "source": "claude"},
 {"text": "less guts", "expected": {"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "moderately"}, "split": "heldout", "source": "claude"},
 {"text": "make the vessels a little more solid", "expected": {"target": "vessels", "attribute": "opacity", "direction": "increase", "strength": "slightly"}, "split": "heldout", "source": "claude"},
 {"text": "everything except the skeleton should go away", "expected": {"target": "skeleton", "attribute": "opacity", "direction": "show_only", "strength": null}, "split": "heldout", "source": "claude"},
 {"text": "I can't see the kidneys", "expected": {"target": "kidneys", "attribute": "opacity", "direction": "increase", "strength": "moderately"}, "split": "heldout", "source": "claude"}
]
```

- [ ] **Step 5: Ask the author for their own held-out phrases**

Ask the user (and any colleague available) to write 10–20 commands the way they would say them, without looking at `data/lexicon.json` (it does not exist yet). For each, agree on the expected command (or null) with the user and append rows with `"split": "heldout"` and `"source": "author"` or `"colleague"`. Do not skip: these rows are the uncontaminated measure.

- [ ] **Step 6: Run the test, expect pass**

Run: `.venv/bin/python -m pytest tests/test_eval_parsers.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add data/parser_eval_realistic.json eval_parsers.py tests/test_eval_parsers.py
git commit -m "test(parser): realistic eval set from the 2026-10-21 VR session and vis lingo"
```

---

### Task 2: Eval options: set, latency, per-source rows, LLM format

**Files:**
- Modify: `eval_parsers.py` (`run_parser`, `score`, `main`)
- Test: `tests/test_eval_parsers.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_percentile_is_nearest_rank():
    assert eval_parsers.percentile([30, 10, 20], 50) == 20
    assert eval_parsers.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == 9
    assert eval_parsers.percentile([], 90) is None


def test_score_records_latency_per_row():
    rows = [{"text": "x", "expected": None, "split": "dev", "source": "claude"}]
    results = eval_parsers.score(rows, {"p": lambda t: None})
    row, actual, exact_ok, intent_ok, ms = results["p"][0]
    assert actual is None and exact_ok and intent_ok
    assert ms >= 0
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_eval_parsers.py -q`
Expected: FAIL (`percentile` missing; tuple has 4 items).

- [ ] **Step 3: Implement** in `eval_parsers.py`

Add `import time` at the top. Add below `matches`:

```python
def percentile(values, q):
    """Nearest-rank percentile (q in 0..100), None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-q * len(ordered) // 100))  # ceil
    return ordered[int(rank) - 1]
```

Replace `score` with:

```python
def score(rows, parsers: dict) -> dict:
    """{parser: [(row, actual, exact_ok, intent_ok, ms), ...]}"""
    results = {}
    for name, fn in parsers.items():
        results[name] = []
        for row in rows:
            start = time.perf_counter()
            actual = run_parser(fn, row["text"])
            ms = (time.perf_counter() - start) * 1000
            results[name].append((row, actual, matches(row["expected"], actual, True),
                                  matches(row["expected"], actual, False), ms))
    return results
```

In `main`, add arguments and use them:

```python
    ap.add_argument("--set", default="base", choices=sorted(EVAL_SETS))
    ap.add_argument("--format", default=None, choices=["json", "schema"],
                    help="LLM output format (default: commands.LLM_FORMAT)")
```

After `args = ap.parse_args(argv)`:

```python
    if args.format:
        commands.LLM_FORMAT = args.format
```

Replace `with open(EVAL_SET_PATH) as f:` with `with open(EVAL_SETS[args.set]) as f:`.

Update the failure loop to unpack five values (`for r, a, ok, _, _ in rs`), and replace the summary printing with:

```python
    print(f"\n{'parser':<7} {'split':<8} {'source':<15} {'n':>3}  {'exact':>6}  {'intent':>6}  {'p50ms':>6}  {'p90ms':>6}")
    for name, rs in results.items():
        for split in args.split.split(","):
            part = [x for x in rs if x[0]["split"] == split]
            groups = [("all", part)] + [(s, [x for x in part if x[0]["source"] == s])
                                        for s in sorted({x[0]["source"] for x in part})]
            for source, rows_ in groups:
                if not rows_ or (source != "all" and len(groups) == 2):
                    continue
                n = len(rows_)
                ms = [x[4] for x in rows_]
                print(f"{name:<7} {split:<8} {source:<15} {n:>3}  {sum(x[2] for x in rows_) / n:6.0%}  "
                      f"{sum(x[3] for x in rows_) / n:6.0%}  {percentile(ms, 50):6.0f}  {percentile(ms, 90):6.0f}")
```

(`commands.LLM_FORMAT` is created in Task 9; until then `--format` is simply not passed.)

- [ ] **Step 4: Run tests, expect pass**

Run: `.venv/bin/python -m pytest tests/test_eval_parsers.py -q && .venv/bin/python eval_parsers.py --parsers rule`
Expected: tests pass; the rule parser still prints 100 % on base dev and heldout.

- [ ] **Step 5: Commit**

```bash
git add eval_parsers.py tests/test_eval_parsers.py
git commit -m "feat(eval): parser latency, per-source rows and eval-set choice"
```

---

### Task 3: Baseline measurement

**Files:**
- Create: `docs/experiments/2026-10-21-voice-lexicon.md`

- [ ] **Step 1: Start Ollama and confirm the model**

Run: `ollama serve` in the background (or open the Ollama app), then `ollama list`.
Expected: `qwen2.5:7b` listed.

- [ ] **Step 2: Run both parsers on the realistic set**

Run: `.venv/bin/python eval_parsers.py --set realistic --parsers rule,llm --show-heldout`
Expected: a table with exact/intent and p50/p90 per parser, split and source. Copy it verbatim.

- [ ] **Step 3: Write the experiment doc**

```markdown
# Voice command lexicon — 2026-10-21

Spec: `docs/superpowers/specs/2026-10-21-voice-lexicon-design.md`.

## Baseline (before any lexicon work)

`eval_parsers.py --set realistic --parsers rule,llm` (LLM `qwen2.5:7b`, `format: json`):

<paste the table from Step 2>

Failures on held-out: <paste the failure list>.
```

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/2026-10-21-voice-lexicon.md
git commit -m "docs(experiments): voice parser baseline on the realistic set"
```

---

### Task 4: Lexicon file and loader

**Files:**
- Create: `data/lexicon.json`, `lexicon.py`
- Test: `tests/test_lexicon.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_lexicon.py`)

```python
import copy

import pytest

import anatomy
import lexicon


def test_shipped_lexicon_loads_and_names_every_class():
    lex = lexicon.load()
    assert list(lex["classes"]) == list(anatomy.CANONICAL_CLASSES)
    for word in ["skeleton", "lungs", "heart", "vessels", "liver", "kidneys", "spleen", "soft tissue"]:
        assert word in lex["asr_prompt"]


def test_validate_rejects_a_phrase_in_two_entries():
    lex = copy.deepcopy(lexicon.load())
    lex["phrases"][0]["say"].append("liver")
    with pytest.raises(lexicon.LexiconError, match="'liver'"):
        lexicon.validate(lex)


def test_validate_rejects_unknown_command_fields():
    lex = copy.deepcopy(lexicon.load())
    lex["phrases"][0]["sets"]["direction"] = "explode"
    with pytest.raises(lexicon.LexiconError, match="explode"):
        lexicon.validate(lex)


def test_validate_rejects_missing_or_extra_classes():
    lex = copy.deepcopy(lexicon.load())
    del lex["classes"]["spleen"]
    with pytest.raises(lexicon.LexiconError, match="classes"):
        lexicon.validate(lex)


def test_validate_rejects_phrases_that_are_not_lowercase_and_trimmed():
    lex = copy.deepcopy(lexicon.load())
    lex["strengths"]["slightly"].append(" A Bit More")
    with pytest.raises(lexicon.LexiconError, match="A Bit More"):
        lexicon.validate(lex)
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_lexicon.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'lexicon'`.

- [ ] **Step 3: Create `data/lexicon.json`**

Classes and retired words carry over `commands.CLASS_SYNONYMS` / `RETIRED_CLASS_WORDS` unchanged, plus lay anatomy. Strength words carry over `_SLIGHT_RE` / `_STRONG_RE` unchanged, plus new ones. `asr_fixes` holds only errors seen in real transcripts. `grammar` lists words the rule grammar in `commands.py` handles in code (for the coverage check only).

```json
{
  "version": 1,
  "asr_prompt": "Commands for a CT volume rendering: show only the skeleton, hide the soft tissue, make the lungs translucent, brighten the liver, darken the kidneys, solo the vessels, fade out the heart, more spleen, sharpen the bones, reset.",
  "classes": {
    "skeleton": ["bone", "bones", "skeleton", "ribs", "rib", "spine", "vertebrae", "hip", "femur", "skull",
                 "ribcage", "rib cage", "backbone", "pelvis", "skeletal"],
    "lungs": ["lung", "lungs", "pulmonary", "pulmonary tissue", "lung tissue"],
    "heart": ["heart", "cardiac", "myocardium"],
    "vessels": ["vessel", "vessels", "artery", "arteries", "vein", "veins", "blood vessel", "aorta", "vena cava",
                "contrast", "vasculature", "vascular"],
    "liver": ["liver", "hepatic"],
    "kidneys": ["kidney", "kidneys", "renal"],
    "spleen": ["spleen", "splenic"],
    "soft": ["soft tissue", "soft", "tissue", "organs", "organ", "muscle", "muscles", "musculature",
             "guts", "meat", "flesh"]
  },
  "retired": {
    "fat": ["fatty", "fat", "adipose"],
    "air": ["air", "background"],
    "spongy": ["spongy bone", "spongy", "cancellous", "trabecular"]
  },
  "phrases": [
    {"say": ["fade out", "get rid of", "mute", "turn off", "switch off", "kill", "make invisible"],
     "sets": {"attribute": "opacity", "direction": "decrease", "strength": "strongly"}},
    {"say": ["translucent", "see-through", "ghost", "more transparent", "transparent", "tone down"],
     "sets": {"attribute": "opacity", "direction": "decrease", "strength": "moderately"}},
    {"say": ["more opaque", "alpha up", "opaque", "solid", "bring back", "bring out", "reveal", "unhide",
             "turn on", "switch on"],
     "sets": {"attribute": "opacity", "direction": "increase"}},
    {"say": ["crank up", "max out"],
     "sets": {"attribute": "opacity", "direction": "increase", "strength": "strongly"}},
    {"say": ["alpha down", "less opaque"],
     "sets": {"attribute": "opacity", "direction": "decrease"}},
    {"say": ["narrower", "tighter", "tighten", "crisper", "sharper"],
     "sets": {"attribute": "width", "direction": "decrease"}},
    {"say": ["wider", "broader", "widen", "softer edges", "smoother"],
     "sets": {"attribute": "width", "direction": "increase"}},
    {"say": ["lighten", "lighter", "light up", "more color", "boost the color"],
     "sets": {"attribute": "brightness", "direction": "increase"}},
    {"say": ["dimmer", "dim", "less color"],
     "sets": {"attribute": "brightness", "direction": "decrease"}},
    {"say": ["solo", "focus on", "only show", "just show"],
     "sets": {"direction": "show_only"}},
    {"say": ["undo everything", "back to default", "clear everything"],
     "sets": {"direction": "reset"}}
  ],
  "strengths": {
    "slightly": ["slightly", "a bit", "a little", "a touch", "somewhat", "little", "a smidge", "a tad"],
    "moderately": ["moderately"],
    "strongly": ["strongly", "a lot", "much", "extremely", "completely", "totally", "fully", "really", "way",
                 "far", "very", "massively", "heavily"]
  },
  "asr_fixes": {"darkin": "darken", "leitenlangs": "lighten lungs"},
  "filler": ["please", "can", "could", "would", "you", "the", "a", "an", "now", "again", "even", "just", "me", "i",
             "want", "to", "see", "make", "let's", "let", "us", "okay", "ok", "um", "uh", "it", "them", "they",
             "those", "these", "that", "this", "is", "are", "be", "should", "some", "bit", "lot"],
  "grammar": ["increase", "decrease", "opacity", "width", "sharpness", "brightness", "center", "centre",
              "position", "peak", "window", "edges", "more", "less", "show", "display", "only", "isolate",
              "exclusively", "nothing", "but", "set", "low", "medium", "high", "rotate", "turn", "tilt", "zoom",
              "left", "right", "up", "down", "in", "out", "shift", "move", "higher", "lower", "sharpen", "soften",
              "brighten", "brighter", "darken", "darker", "dark", "bright", "light", "hide", "hiding", "remove",
              "reduce", "add", "boost", "raise", "bring", "stand", "emphasise", "emphasize", "highlight",
              "strengthen", "pop", "punch", "enhance", "better", "can't", "cannot", "barely", "faint", "invisible",
              "washed", "missing", "visible", "stronger", "fade", "get", "rid", "of", "eliminate", "weaken",
              "through", "dominating", "dominate", "drown", "drowning", "obscure", "obscuring", "block",
              "blocking", "cut", "back", "too", "strong", "prominent", "dense", "tone", "dial", "pull", "reset",
              "revert", "start", "over", "from", "scratch", "default", "defaults", "original", "settings", "and",
              "then", "plus", "while", "so", "for", "on", "with", "everything", "view", "level"]
}
```

- [ ] **Step 4: Create `lexicon.py`** (loader and validator only; later tasks add functions)

```python
"""The voice-command vocabulary (data/lexicon.json): load, validate, and
derive what each stage needs -- the Whisper prompt, normalization, the rule
parser's word lists and the coverage check. One file, so speech recognition,
the rule parser and the LLM can never drift onto different vocabularies."""
import functools
import json

import anatomy

LEXICON_PATH = "data/lexicon.json"
COMMAND_FIELDS = {
    "attribute": {"opacity", "width", "brightness", "center"},
    "direction": {"increase", "decrease", "show_only", "reset"},
    "strength": {"slightly", "moderately", "strongly"},
}


class LexiconError(ValueError):
    pass


def _check_phrase(phrase: str, where: str) -> None:
    if not isinstance(phrase, str) or phrase != phrase.strip().lower() or not phrase:
        raise LexiconError(f"{where}: phrase {phrase!r} must be lowercase and trimmed")


def validate(lex: dict) -> dict:
    """Raise LexiconError on anything the stages cannot use; return `lex`."""
    if list(lex.get("classes", {})) != list(anatomy.CANONICAL_CLASSES):
        raise LexiconError(f"classes must be exactly {list(anatomy.CANONICAL_CLASSES)} in that order")
    owner = {}

    def claim(phrase, where):
        _check_phrase(phrase, where)
        if phrase in owner:
            raise LexiconError(f"{phrase!r} is in both {owner[phrase]} and {where}")
        owner[phrase] = where

    for cls, terms in lex["classes"].items():
        for term in terms:
            claim(term, f"classes.{cls}")
    for name, terms in lex["retired"].items():
        for term in terms:
            claim(term, f"retired.{name}")
    for i, entry in enumerate(lex["phrases"]):
        for key, value in entry["sets"].items():
            if key not in COMMAND_FIELDS or value not in COMMAND_FIELDS[key]:
                raise LexiconError(f"phrases[{i}]: {key}={value!r} is not a command field value")
        for phrase in entry["say"]:
            claim(phrase, f"phrases[{i}]")
    for strength, words in lex["strengths"].items():
        if strength not in COMMAND_FIELDS["strength"]:
            raise LexiconError(f"strengths: unknown strength {strength!r}")
        for word in words:
            claim(word, f"strengths.{strength}")
    for wrong in lex["asr_fixes"]:
        claim(wrong, "asr_fixes")
    for key in ("filler", "grammar"):
        for word in lex[key]:
            _check_phrase(word, key)
    if not isinstance(lex.get("asr_prompt"), str) or not lex["asr_prompt"]:
        raise LexiconError("asr_prompt must be a non-empty string")
    return lex


def _cached(lex: dict, name: str, build):
    """Derived tables, built once per lexicon dict and stored on it (under
    "_cache", which validate() ignores)."""
    cache = lex.setdefault("_cache", {})
    if name not in cache:
        cache[name] = build(lex)
    return cache[name]


def load(path: str = LEXICON_PATH) -> dict:
    with open(path) as f:
        return validate(json.load(f))


@functools.cache
def get() -> dict:
    """The shipped lexicon, loaded once per process."""
    return load()
```

- [ ] **Step 5: Run tests, expect pass**

Run: `.venv/bin/python -m pytest tests/test_lexicon.py -q`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add data/lexicon.json lexicon.py tests/test_lexicon.py
git commit -m "feat(lexicon): voice vocabulary file with loader and validation"
```

---

### Task 5: Rule parser takes classes, retired words and strengths from the lexicon

**Files:**
- Modify: `commands.py:20-49` (class/retired tables), `commands.py:191-193` (strength regexes), `commands.py:52-58` (`_check_retired`)
- Modify: `lexicon.py` (add `strength_pattern`)
- Test: `tests/test_lexicon.py`, `tests/test_commands.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lexicon.py`:

```python
def test_strength_pattern_matches_whole_words_and_keeps_very_well_out():
    lex = lexicon.load()
    strong = lexicon.strength_pattern(lex, "strongly")
    assert strong.search("hide it completely")
    assert strong.search("very bright")
    assert not strong.search("works very well")
    assert not strong.search("farther")
    slight = lexicon.strength_pattern(lex, "slightly")
    assert slight.search("a smidge more")
```

Append to `tests/test_commands.py`:

```python
def test_lay_anatomy_words_from_the_lexicon_name_classes():
    assert parse_command_rule("show only the ribcage")["target"] == "skeleton"
    assert parse_command_rule("less guts")["target"] == "soft"
    assert parse_command_rule("add vasculature")["target"] == "vessels"


def test_new_strength_words_from_the_lexicon():
    assert parse_command_rule("a smidge more spleen")["strength"] == "slightly"


def test_retired_class_raises_its_own_error_type():
    with pytest.raises(commands.RetiredClassError, match="no longer a supported class"):
        parse_command_rule("show more fat")
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_lexicon.py tests/test_commands.py -q`
Expected: FAIL (`strength_pattern` missing, "ribcage" unknown, `RetiredClassError` missing).

- [ ] **Step 3: Add `strength_pattern` to `lexicon.py`** (add `import re` at the top)

```python
def strength_pattern(lex: dict, strength: str) -> re.Pattern:
    """Whole-word regex over a strength's words, longest first. "very" is
    kept out of "very well" (as the hand-written regex did)."""
    words = sorted(lex["strengths"][strength], key=len, reverse=True)
    alternatives = [r"very(?! well)" if w == "very" else re.escape(w) for w in words]
    return re.compile(rf"\b({'|'.join(alternatives)})\b")
```

- [ ] **Step 4: Switch `commands.py` to the lexicon**

Add `import lexicon` after `import anatomy`. Replace the literal `CLASS_SYNONYMS = {...}` dict and the following reorder line with:

```python
# Vocabulary lives in data/lexicon.json (lexicon.py), shared with speech
# recognition and the LLM prompt.
_LEX = lexicon.get()
CLASS_SYNONYMS = {name: list(_LEX["classes"][name]) for name in anatomy.CANONICAL_CLASSES}
```

Replace the literal `RETIRED_CLASS_WORDS = {...}` with:

```python
RETIRED_CLASS_WORDS = {name: list(words) for name, words in _LEX["retired"].items()}
```

Add above `_check_retired` and use it there:

```python
class RetiredClassError(ValueError):
    """A retired class (fat/air/spongy) was named: refused, never remapped."""
```

In `_check_retired`, change `raise ValueError(` to `raise RetiredClassError(`.

Replace the two regex definitions `_SLIGHT_RE = ...` and `_STRONG_RE = ...` with:

```python
_SLIGHT_RE = lexicon.strength_pattern(_LEX, "slightly")
_STRONG_RE = lexicon.strength_pattern(_LEX, "strongly")
```

- [ ] **Step 5: Run every parser test and both eval sets**

Run: `.venv/bin/python -m pytest tests/test_lexicon.py tests/test_commands.py tests/test_eval_parsers.py tests/test_commands_doc.py -q && .venv/bin/python eval_parsers.py --parsers rule`
Expected: all pass; base set still 100 % / 100 % for dev and heldout.

- [ ] **Step 6: Commit**

```bash
git add commands.py lexicon.py tests/test_lexicon.py tests/test_commands.py
git commit -m "refactor(parser): class, retired and strength words from the lexicon"
```

---

### Task 6: Cue phrases in the rule parser

**Files:**
- Modify: `lexicon.py` (add `find_cue`), `commands.py` (`_parse_free_clause`, `_parse_show_only`, `parse_command_rule`)
- Test: `tests/test_lexicon.py`, `tests/test_commands.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lexicon.py`:

```python
def test_find_cue_prefers_the_longest_phrase():
    lex = lexicon.load()
    assert lexicon.find_cue("make the bones more opaque", lex) == {"attribute": "opacity", "direction": "increase"}
    assert lexicon.find_cue("softer edges on the bones", lex) == {"attribute": "width", "direction": "increase"}
    assert lexicon.find_cue("make the lungs see-through", lex)["direction"] == "decrease"
    assert lexicon.find_cue("add kidneys", lex) is None
```

Append to `tests/test_commands.py`:

```python
@pytest.mark.parametrize("text, expected", [
    ("mute the bones", ("skeleton", "opacity", "decrease", "strongly")),
    ("make the liver translucent", ("liver", "opacity", "decrease", "moderately")),
    ("tighter peak on the skeleton", ("skeleton", "width", "decrease", "moderately")),
    ("crisper kidneys", ("kidneys", "width", "decrease", "moderately")),
    ("bring back the lungs", ("lungs", "opacity", "increase", "moderately")),
    ("kill the soft tissue", ("soft", "opacity", "decrease", "strongly")),
    ("ghost the muscles a bit", ("soft", "opacity", "decrease", "slightly")),
])
def test_lexicon_cue_phrases(text, expected):
    cmd = parse_command_rule(text)
    assert (cmd["target"], cmd["attribute"], cmd["direction"], cmd["strength"]) == expected


def test_show_only_and_reset_cues():
    assert parse_command_rule("solo vessels")["direction"] == "show_only"
    assert parse_command_rule("focus on the liver")["target"] == "liver"
    assert parse_command_rule("undo everything")["direction"] == "reset"
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_lexicon.py tests/test_commands.py -q`
Expected: FAIL (`find_cue` missing; cue phrases unparsed).

- [ ] **Step 3: Add `find_cue` to `lexicon.py`**

```python
def _cue_patterns(lex: dict):
    pairs = [(phrase, entry["sets"]) for entry in lex["phrases"] for phrase in entry["say"]]
    pairs.sort(key=lambda p: -len(p[0]))
    return [(re.compile(rf"\b{re.escape(p)}(?:s|es|ed|ing)?\b"), sets) for p, sets in pairs]


def find_cue(text: str, lex: dict):
    """The command fields of the longest lexicon phrase in `text` (lowercase),
    or None."""
    for pattern, sets in _cached(lex, "cues", _cue_patterns):
        if pattern.search(text):
            return dict(sets)
    return None
```

- [ ] **Step 4: Use cues in `commands.py`**

In `_parse_free_clause`, directly after the `if level:` block and before `direction = _free_direction(clause, attribute)`:

```python
    cue = lexicon.find_cue(clause, _LEX)
    if cue and cue.get("direction") in ("increase", "decrease"):
        attribute = cue.get("attribute", attribute)
        strength = _strength_in(clause, cue.get("strength", "moderately"))
        return [{"target": c, "attribute": attribute, "direction": cue["direction"], "strength": strength}
                for c in classes]
```

Add a helper below `_parse_show_only`'s definition site (above it is fine):

```python
def _cue_direction(t: str):
    cue = lexicon.find_cue(t, _LEX)
    return cue.get("direction") if cue else None
```

In `_parse_show_only`, change the first condition to:

```python
    if not (_SHOW_ONLY_RE.search(t) or _cue_direction(t) == "show_only") or _RELATIVE_CUE_RE.search(t):
        return None
```

In `parse_command_rule`, change `if _RESET_RE.search(t):` to:

```python
    if _RESET_RE.search(t) or _cue_direction(t) == "reset":
```

- [ ] **Step 5: Run tests and the base eval**

Run: `.venv/bin/python -m pytest tests/test_lexicon.py tests/test_commands.py tests/test_eval_parsers.py -q && .venv/bin/python eval_parsers.py --parsers rule && .venv/bin/python eval_parsers.py --set realistic --parsers rule --split dev`
Expected: tests pass; base set 100 %; realistic dev improved over the Task 3 baseline (record nothing yet).

- [ ] **Step 6: Commit**

```bash
git add commands.py lexicon.py tests/test_lexicon.py tests/test_commands.py
git commit -m "feat(parser): vis-lingo cue phrases from the lexicon"
```

---

### Task 7: Normalization (ASR fixes and fuzzy matching)

**Files:**
- Modify: `lexicon.py` (add `normalize`, `phonetic_key`, `known_words`)
- Test: `tests/test_lexicon.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("raw, fixed", [
    ("Darkin' Skeleton", "darken skeleton"),
    ("Leitenlangs", "lighten lungs"),
    ("darken the sgeleton", "darken the skeleton"),
    ("show only the kitneys", "show only the kidneys"),
    ("make the lever more visible", "make the liver more visible"),
])
def test_normalize_fixes_transcription_errors(raw, fixed):
    assert lexicon.normalize(raw, lexicon.load()).lower() == fixed


@pytest.mark.parametrize("text", [
    "make it look cool",
    "thank you for watching",
    "can you show me just the heart please",
    "the muscles are hiding everything, get rid of them",
    "more aorta, less muscle",
])
def test_normalize_leaves_ordinary_sentences_alone(text):
    assert lexicon.normalize(text, lexicon.load()) == text


def test_phonetic_key_merges_voiced_and_voiceless_consonants():
    assert lexicon.phonetic_key("sgeleton") == lexicon.phonetic_key("skeleton")
    assert lexicon.phonetic_key("kitneys") == lexicon.phonetic_key("kidneys")
    assert lexicon.phonetic_key("lever") == lexicon.phonetic_key("liver")
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_lexicon.py -q`
Expected: FAIL (`normalize` missing).

- [ ] **Step 3: Implement in `lexicon.py`** (add `import difflib`)

```python
_PHONETIC = str.maketrans({"k": "g", "c": "g", "q": "g", "t": "d", "p": "b", "f": "v", "w": "v",
                           "z": "s", "x": "s"})


def phonetic_key(word: str) -> str:
    """Crude sound key: voiced/voiceless pairs merged, vowels after the first
    letter dropped, doubled letters collapsed ("sgeleton" == "skeleton")."""
    w = word.lower().translate(_PHONETIC)
    key = w[:1] + re.sub(r"[aeiouyh']", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", key)


def _words(phrases):
    return {w for p in phrases for w in re.findall(r"[a-z]+(?:'[a-z]+)?", p)}


def _vocab(lex: dict):
    content = _words([t for terms in lex["classes"].values() for t in terms]
                     + [p for e in lex["phrases"] for p in e["say"]]
                     + [w for ws in lex["strengths"].values() for w in ws])
    known = content | _words([t for ts in lex["retired"].values() for t in ts]) \
        | _words(lex["filler"]) | _words(lex["grammar"]) | _words(lex["asr_fixes"].values())
    return sorted(w for w in content if len(w) >= 4), known


def known_words(lex: dict) -> set:
    return _cached(lex, "vocab", _vocab)[1]


def _fix_word(word: str, lex: dict):
    """The lexicon word `word` was most likely meant as, or None to keep it."""
    w = word.lower().strip("'")
    if w in lex["asr_fixes"]:
        return lex["asr_fixes"][w]
    targets, known = _cached(lex, "vocab", _vocab)
    if w in known or len(w) < 4:
        return None
    best = difflib.get_close_matches(w, targets, n=1, cutoff=0.7)
    if not best:
        return None
    ratio = difflib.SequenceMatcher(None, w, best[0]).ratio()
    if ratio >= 0.88 or phonetic_key(w) == phonetic_key(best[0]):
        return best[0]
    return None


def normalize(text: str, lex: dict) -> str:
    """Rewrite known transcription errors and near-misses of lexicon words;
    everything else, including punctuation, stays as it was."""
    def repl(m):
        fixed = _fix_word(m.group(0), lex)
        return fixed if fixed is not None else m.group(0)

    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?'?", repl, text)
```

- [ ] **Step 4: Run tests, expect pass**

Run: `.venv/bin/python -m pytest tests/test_lexicon.py -q`
Expected: all pass. If an ordinary-sentence case is rewritten, raise the 0.88 threshold or require the phonetic match; do not add sentence-specific exceptions.

- [ ] **Step 5: Commit**

```bash
git add lexicon.py tests/test_lexicon.py
git commit -m "feat(lexicon): normalize transcription errors to lexicon words"
```

---

### Task 8: Coverage check, hybrid router, utterance log

**Files:**
- Modify: `lexicon.py` (add `uncovered_words`), `commands.py` (fallback order in `parse_command_llm`, `DEFAULT_LLM_MODEL`, `parse_command_hybrid`, `parse_command_with_meta`), `eval_parsers.py` (hybrid parser), `server.py` (`CommandRequest`, `Session.command` default), `static/app.js:13`
- Test: `tests/test_lexicon.py`, `tests/test_commands.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lexicon.py`:

```python
def test_uncovered_words_lists_content_words_outside_the_lexicon():
    lex = lexicon.load()
    assert lexicon.uncovered_words("show only the bones please", lex) == []
    assert lexicon.uncovered_words("make the lungs see-through", lex) == []
    assert lexicon.uncovered_words("everything except the skeleton should go away", lex) == ["except", "go", "away"]
```

Append to `tests/test_commands.py`:

```python
def _llm_stub(monkeypatch, answer=None, fail=False):
    calls = []

    def fake(text, model=None, host=None, _on_fallback=None):
        calls.append(text)
        if fail:
            if _on_fallback:
                _on_fallback("request failed: <urlopen error refused>")
            return commands.parse_command_rule(text)
        return answer
    monkeypatch.setattr(commands, "parse_command_llm", fake)
    return calls


def test_hybrid_answers_covered_sentences_without_the_llm(monkeypatch):
    calls = _llm_stub(monkeypatch)
    cmd, meta = commands.parse_command_hybrid("Darkin' Skeleton")
    assert cmd["target"] == "skeleton" and cmd["direction"] == "decrease"
    assert meta["path"] == "rule" and meta["normalized"] == "darken Skeleton"
    assert calls == []


def test_hybrid_asks_the_llm_when_words_are_left_over(monkeypatch):
    answer = {"target": "skeleton", "attribute": "opacity", "direction": "show_only", "strength": None}
    calls = _llm_stub(monkeypatch, answer=answer)
    cmd, meta = commands.parse_command_hybrid("everything except the skeleton should go away")
    assert cmd == answer and meta["path"] == "llm"
    assert meta["uncovered"] == ["except", "go", "away"]
    assert len(calls) == 1


def test_hybrid_reports_llm_offline_when_nothing_parses(monkeypatch):
    _llm_stub(monkeypatch, fail=True)
    with pytest.raises(ValueError, match="LLM offline"):
        commands.parse_command_hybrid("everything except the skeleton should go away")


def test_hybrid_refuses_retired_classes_without_the_llm(monkeypatch):
    calls = _llm_stub(monkeypatch)
    with pytest.raises(commands.RetiredClassError):
        commands.parse_command_hybrid("show more fat")
    assert calls == []


def test_hybrid_logs_every_utterance(monkeypatch):
    _llm_stub(monkeypatch)
    commands.parse_command_hybrid("add kidneys")
    with pytest.raises(ValueError):
        commands.parse_command_hybrid("show more fat")
    rows = [json.loads(line) for line in open(commands.UTTERANCE_LOG_PATH)]
    assert [r["text"] for r in rows] == ["add kidneys", "show more fat"]
    assert rows[0]["path"] == "rule" and rows[0]["cmd"]["target"] == "kidneys"
    assert rows[1]["cmd"] is None and "no longer a supported class" in rows[1]["error"]
    assert rows[0]["parse_ms"] >= 0


def test_with_meta_hybrid_carries_the_path(monkeypatch):
    _llm_stub(monkeypatch)
    cmd, meta = commands.parse_command_with_meta("add kidneys", parser="hybrid")
    assert meta["parser_requested"] == "hybrid" and meta["parser_used"] == "rule"
    assert meta["normalized"] == "add kidneys"
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_lexicon.py tests/test_commands.py -q`
Expected: FAIL (`uncovered_words`, `parse_command_hybrid` missing).

- [ ] **Step 3: Add `uncovered_words` to `lexicon.py`**

```python
def uncovered_words(text: str, lex: dict) -> list:
    """Words of `text` the lexicon does not account for, in order. A word
    counts as covered with a plural/verb suffix too ("lungs", "hiding")."""
    known = known_words(lex)
    left = []
    for w in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()):
        stems = {w} | {w[: -len(s)] for s in ("s", "es", "ed", "ing", "ly", "er") if w.endswith(s)}
        if not stems & known:
            left.append(w)
    return left
```

- [ ] **Step 4: Report the LLM failure before the rule fallback** in `parse_command_llm`

Replace the body of `except (URLError, OSError, ValueError, KeyError) as exc:` with:

```python
        print(f"[llm-parser] falling back to rule parser: {exc}")
        reason = f"request failed: {exc}"
        if _on_fallback:
            _on_fallback(reason)
        try:
            final = parse_command_rule(text)
        except ValueError:
            _log_llm_request(text, model, host, None, None, None, reason)
            raise
        _log_llm_request(text, model, host, None, None, final, reason)
        return final
```

Run: `.venv/bin/python -m pytest tests/test_commands.py -q -k llm`
Expected: existing LLM tests still pass.

- [ ] **Step 5: Add the hybrid router to `commands.py`**

Add `import time` at the top. Below `OLLAMA_HOST = ...`:

```python
DEFAULT_LLM_MODEL = "qwen2.5:7b"
UTTERANCE_LOG_PATH = "out/utterances.jsonl"
```

Replace every default argument `model: str = "qwen2.5:7b"` in `commands.py` with `model: str = DEFAULT_LLM_MODEL` (in `parse_command_with_meta` and `parse_command_llm`); move the two constants above the first function that uses them if needed (`parse_command_with_meta` is defined earlier in the file, so put `DEFAULT_LLM_MODEL` and `UTTERANCE_LOG_PATH` next to `STRENGTH_WORDS`).

Add after `parse_command_llm`:

```python
def _log_utterance(text, meta, cmd, error, start):
    meta["parse_ms"] = round((time.perf_counter() - start) * 1000, 1)
    from evaluate import jsonl_append  # same lazy import as _log_llm_request
    os.makedirs(os.path.dirname(UTTERANCE_LOG_PATH) or ".", exist_ok=True)
    jsonl_append(UTTERANCE_LOG_PATH, {
        "timestamp": _dt.datetime.now().isoformat(), "text": text, **meta, "cmd": cmd, "error": error,
    })


def parse_command_hybrid(text: str, model: str = DEFAULT_LLM_MODEL):
    """The lexicon pipeline: normalize, answer with the rule parser when it
    parses and every word is accounted for, else ask the LLM (which falls back
    to the rule parser itself). Returns (cmd, meta) with meta = {"normalized",
    "uncovered", "path" ("rule"|"llm"), "fallback", "parse_ms"}; raises
    ValueError (RetiredClassError for fat/air/spongy) when nothing parses."""
    start = time.perf_counter()
    normalized = lexicon.normalize(text, _LEX)
    meta = {"normalized": normalized, "uncovered": lexicon.uncovered_words(normalized, _LEX),
            "path": None, "fallback": None}
    rule_cmd = None
    try:
        rule_cmd = parse_command_rule(normalized)
    except RetiredClassError as exc:
        meta["path"] = "rejected"
        _log_utterance(text, meta, None, str(exc), start)
        raise
    except ValueError:
        pass
    if rule_cmd is not None and not meta["uncovered"]:
        meta["path"] = "rule"
        _log_utterance(text, meta, rule_cmd, None, start)
        return rule_cmd, meta

    reasons = []
    try:
        cmd = parse_command_llm(normalized, model=model, _on_fallback=reasons.append)
    except ValueError as exc:
        reason = reasons[0] if reasons else str(exc)
        meta["path"], meta["fallback"] = "rejected", reason
        message = (f"LLM offline and the rule parser cannot parse {text!r}"
                   if reason.startswith("request failed") else f"could not parse {text!r}")
        _log_utterance(text, meta, None, message, start)
        raise ValueError(message) from exc
    meta["path"] = "rule" if reasons else "llm"
    meta["fallback"] = reasons[0] if reasons else None
    _log_utterance(text, meta, cmd, None, start)
    return cmd, meta
```

In `parse_command_with_meta`, add before `if parser != "llm":`:

```python
    if parser == "hybrid":
        cmd, hmeta = parse_command_hybrid(text, model=model)
        return cmd, {"parser_requested": "hybrid", "parser_used": hmeta["path"],
                     "parser_fallback": hmeta["fallback"], "normalized": hmeta["normalized"],
                     "uncovered": hmeta["uncovered"], "parse_ms": hmeta["parse_ms"]}
```

- [ ] **Step 6: Run tests, expect pass**

Run: `.venv/bin/python -m pytest tests/test_lexicon.py tests/test_commands.py -q`
Expected: all pass. If `uncovered_words` returns extra words for the three cases, add the missing *grammar* words to `data/lexicon.json` (not sentence-specific hacks).

- [ ] **Step 7: Make hybrid the default everywhere and in the eval**

`server.py`: in `CommandRequest`, set `parser: str = "hybrid"` and `model: str = commands.DEFAULT_LLM_MODEL` (add `import commands` if only names are imported) and update the comment above to say "hybrid is the default: rule parser for fully covered sentences, LLM for the rest". In `Session.command(self, text, parser="rule", model="qwen2.5:7b", ...)` set `model=commands.DEFAULT_LLM_MODEL`.

`static/app.js:13`: `const config = { parser: "hybrid", search: false, mode: "policy" };` and update the comment above it the same way.

`eval_parsers.py` `main`: add to `available`:

```python
                 "hybrid": lambda t: commands.parse_command_hybrid(t, model=args.llm_model)[0],
```

and next to the `LLM_LOG_PATH` redirect:

```python
    commands.UTTERANCE_LOG_PATH = "out/utterances_eval.jsonl"
```

Set `--parsers` default to `"rule,llm,hybrid"` and `--llm-model` default to `commands.DEFAULT_LLM_MODEL`.

- [ ] **Step 8: Run the full Python suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (fix any test that asserted the old `parser="llm"` default by updating it to `"hybrid"`).

- [ ] **Step 9: Commit**

```bash
git add lexicon.py commands.py eval_parsers.py server.py static/app.js tests/test_lexicon.py tests/test_commands.py
git commit -m "feat(parser): hybrid router with coverage check and utterance log"
```

---

### Task 9: LLM prompt from the lexicon, JSON schema, thinking off

**Files:**
- Modify: `commands.py` (`_SYSTEM_PROMPT`, `parse_command_llm` request body)
- Test: `tests/test_commands.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_llm_prompt_lists_lexicon_cue_phrases():
    assert '"fade out"' in commands._SYSTEM_PROMPT
    assert "see-through" in commands._SYSTEM_PROMPT


def test_command_schema_allows_each_command_shape():
    schema = commands.command_schema()
    shapes = schema["anyOf"]
    assert len(shapes) == 4
    single = shapes[0]
    assert single["properties"]["target"]["anyOf"][0]["enum"] == list(commands.CLASS_SYNONYMS)


def _capture_body(monkeypatch):
    sent = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"response": json.dumps({"target": "liver", "attribute": "opacity",
                                                       "direction": "increase", "strength": "moderately"})}).encode()

    def fake_urlopen(req, timeout=None):
        sent.update(json.loads(req.data.decode()))
        return Resp()
    monkeypatch.setattr(commands, "urlopen", fake_urlopen)
    return sent


def test_llm_request_uses_the_schema_and_turns_thinking_off_for_thinking_models(monkeypatch):
    sent = _capture_body(monkeypatch)
    commands.parse_command_llm("add liver", model="qwen3.5:4b")
    assert sent["format"] == commands.command_schema()
    assert sent["think"] is False
    sent.clear()
    commands.parse_command_llm("add liver", model="qwen2.5:7b")
    assert "think" not in sent


def test_llm_format_json_is_still_available(monkeypatch):
    sent = _capture_body(monkeypatch)
    monkeypatch.setattr(commands, "LLM_FORMAT", "json")
    commands.parse_command_llm("add liver")
    assert sent["format"] == "json"
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_commands.py -q -k "schema or prompt or llm_format or thinking"`
Expected: FAIL.

- [ ] **Step 3: Implement**

Below `_class_synonym_lines`, add:

```python
def _cue_lines() -> str:
    lines = []
    for entry in _LEX["phrases"]:
        fields = ", ".join(f"{k} {v}" for k, v in entry["sets"].items())
        lines.append(f"- {', '.join(_json.dumps(p) for p in entry['say'])}: {fields}")
    return "\n".join(lines)
```

In `_SYSTEM_PROMPT`, directly after the paragraph ending `unpromoted organs and muscle. "fat", "air" and "spongy" are not supported.` insert:

```python
Phrases people use, and the command fields they mean (strength "moderately"
unless the phrase or a modifier says otherwise):
{_cue_lines()}
```

Below `VALID_CAMERA_STRENGTHS = ...` add:

```python
LLM_FORMAT = "schema"  # "schema": constrained decoding to command_schema(); "json": any JSON
THINKING_MODEL_PREFIXES = ("qwen3", "deepseek-r1", "magistral", "gpt-oss")


def command_schema() -> dict:
    """JSON schema of every answer shape, for Ollama's constrained decoding.
    Validation after decoding stays (_validate_cmd): the schema only keeps
    the model from producing shapes that can never be right."""
    cls = {"enum": list(CLASS_SYNONYMS)}
    strength = {"enum": ["slightly", "moderately", "strongly", None]}
    single = {"type": "object", "properties": {
        "target": {"anyOf": [cls, {"type": "array", "items": cls}, {"type": "null"}]},
        "attribute": {"enum": ["opacity", "width", "brightness", "center", None]},
        "direction": {"enum": ["increase", "decrease", "show_only", "reset", "set"]},
        "strength": strength,
        "level": {"enum": ["low", "medium", "high"]},
    }, "required": ["target", "attribute", "direction"]}
    compound = {"type": "object", "properties": {"compound": {"type": "array", "items": single, "minItems": 1}},
                "required": ["compound"]}
    camera = {"type": "object", "properties": {"camera": {"type": "object", "properties": {
        "action": {"enum": ["rotate", "tilt", "zoom"]},
        "direction": {"enum": ["left", "right", "up", "down", "in", "out"]},
        "strength": {"enum": ["slightly", "moderately", "strongly"]},
    }, "required": ["action", "direction", "strength"]}}, "required": ["camera"]}
    none = {"type": "object", "properties": {"none": {"type": "boolean"}}, "required": ["none"]}
    return {"anyOf": [single, compound, camera, none]}
```

In `parse_command_llm`, replace the `body = _json.dumps({...}).encode()` construction with a dict built first:

```python
    request = {
        "model": model,
        "system": _SYSTEM_PROMPT,
        "prompt": text,
        "format": command_schema() if LLM_FORMAT == "schema" else "json",
        "stream": False,
        # (keep the existing keep_alive and options comments and values here)
        "keep_alive": "30m",
        "options": {"temperature": 0, "seed": 0},
    }
    if model.startswith(THINKING_MODEL_PREFIXES):
        # Thinking adds seconds of hidden tokens to a one-line answer.
        request["think"] = False
    body = _json.dumps(request).encode()
```

- [ ] **Step 4: Run tests, expect pass**

Run: `.venv/bin/python -m pytest tests/test_commands.py -q`
Expected: all pass.

- [ ] **Step 5: Smoke-test against Ollama**

Run: `.venv/bin/python -c "import commands; print(commands.parse_command_llm('make the lungs see-through'))"`
Expected: a lungs opacity decrease command, and no `invalid schema` fallback line. If Ollama rejects the schema (HTTP 400 in the fallback reason), report it and run the rest of the plan with `LLM_FORMAT = "json"`.

- [ ] **Step 6: Commit**

```bash
git add commands.py tests/test_commands.py
git commit -m "feat(llm-parser): lexicon phrases in the prompt, schema-constrained output"
```

---

### Task 10: ASR: vocabulary prompt, VAD, hallucination filter, backend choice

**Files:**
- Modify: `asr.py`, `server.py` (`/api/transcribe`)
- Test: `tests/test_asr.py` (new)

- [ ] **Step 1: Write the failing tests** (`tests/test_asr.py`)

```python
import pytest

import asr


@pytest.mark.parametrize("text", ["", "You", "you.", "Thank you.", "Thanks for watching!", "Bye",
                                  "We're in a small town and I've just tried to do the right things. Should I go "
                                  "to high school? No, no. Should I go to high school? Yeah, yeah, yeah. It was good."])
def test_hallucinations_are_recognised(text):
    assert asr.is_hallucination(text)


@pytest.mark.parametrize("text", ["Show only bones", "REVERT", "Darkin' Skeleton", "add kidneys"])
def test_commands_are_not_hallucinations(text):
    assert not asr.is_hallucination(text)


def test_faster_whisper_gets_prompt_and_vad(monkeypatch):
    seen = {}

    class Model:
        def transcribe(self, path, **kwargs):
            seen.update(kwargs)
            return [type("S", (), {"text": " show only bones "})()], None
    monkeypatch.setitem(asr._MODEL_CACHE, ("faster-whisper", "small"), ("faster-whisper", Model()))
    text = asr._transcribe_path("x.wav", "en", "small", initial_prompt="skeleton, lungs", backend="faster-whisper")
    assert text == "show only bones"
    assert seen["initial_prompt"] == "skeleton, lungs"
    assert seen["vad_filter"] is True
    assert seen["condition_on_previous_text"] is False


def test_mlx_uses_the_requested_model_size(monkeypatch):
    seen = {}

    class FakeMlx:
        @staticmethod
        def transcribe(path, **kwargs):
            seen.update(kwargs)
            return {"text": " reset "}
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", FakeMlx)
    assert asr._transcribe_path("x.wav", "en", "turbo", initial_prompt="p", backend="mlx") == "reset"
    assert seen["path_or_hf_repo"] == asr.MLX_REPOS["turbo"]
    assert seen["initial_prompt"] == "p"
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_asr.py -q`
Expected: FAIL (`is_hallucination` missing, unexpected keyword `initial_prompt`).

- [ ] **Step 3: Rewrite `asr.py`**

```python
"""Speech recognition for voice commands: audio file -> text.

Audio is captured in the browser and POSTed to `/api/transcribe`. Two
backends: mlx-whisper (Apple GPU) and faster-whisper (CPU, int8); ASR_BACKEND
picks one, otherwise mlx-whisper when installed. ASR_MODEL picks the size.
"""
import os
import re

_MODEL_CACHE = {}
SAMPLE_RATE = 16000
AUDIO_DIR = "out/audio"
ASR_BACKEND = os.environ.get("ASR_BACKEND")  # "mlx" | "faster-whisper" | None
ASR_MODEL = os.environ.get("ASR_MODEL", "small")
MLX_REPOS = {"small": "mlx-community/whisper-small-mlx", "turbo": "mlx-community/whisper-large-v3-turbo"}
FASTER_WHISPER_NAMES = {"small": "small", "turbo": "large-v3-turbo"}

# What Whisper says on silence or noise (2026-10-21 VR session: "You" eleven
# times, invented sentences twice). Commands are short.
HALLUCINATIONS = {"you", "thank you", "thanks", "thank you for watching", "thanks for watching", "bye",
                  "okay", "ok", "so", "um", "uh"}
MAX_COMMAND_WORDS = 20


def is_hallucination(text: str) -> bool:
    t = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    return not t or t in HALLUCINATIONS or len(t.split()) > MAX_COMMAND_WORDS


def _default_backend() -> str:
    if ASR_BACKEND:
        return ASR_BACKEND
    try:
        import mlx_whisper  # noqa: F401  # pyright: ignore[reportMissingImports]
        return "mlx"
    except ImportError:
        return "faster-whisper"


def _load_model(model_size: str, backend: str):
    key = (backend, model_size)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if backend == "mlx":
        _MODEL_CACHE[key] = ("mlx", MLX_REPOS[model_size])
        print(f"[asr] using mlx-whisper ({MLX_REPOS[model_size]})")
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel(FASTER_WHISPER_NAMES[model_size], device="cpu", compute_type="int8")
        _MODEL_CACHE[key] = ("faster-whisper", model)
        print(f"[asr] using faster-whisper ({FASTER_WHISPER_NAMES[model_size]}, cpu, int8)")
    return _MODEL_CACHE[key]


def _transcribe_path(path: str, lang: str, model_size: str = ASR_MODEL, initial_prompt: str = None,
                     backend: str = None) -> str:
    kind, model = _load_model(model_size, backend or _default_backend())
    if kind == "mlx":
        import mlx_whisper  # pyright: ignore[reportMissingImports]
        result = mlx_whisper.transcribe(path, path_or_hf_repo=model, language=lang,
                                        initial_prompt=initial_prompt, condition_on_previous_text=False)
        return result["text"].strip()
    segments, _ = model.transcribe(path, language=lang, initial_prompt=initial_prompt, vad_filter=True,
                                   condition_on_previous_text=False)
    return " ".join(seg.text.strip() for seg in segments).strip()
```

- [ ] **Step 4: Use it in `server.py`**

Change the import to `from asr import ASR_MODEL, _transcribe_path as asr_transcribe_path, is_hallucination` and add `import lexicon`. In `transcribe`, replace the transcription and return lines with:

```python
    start = time.time()
    text = asr_transcribe_path(wav_path, "en", ASR_MODEL, initial_prompt=lexicon.get()["asr_prompt"])
    duration = time.time() - start

    with open(os.path.join(AUDIO_DIR, f"{stamp}.txt"), "w") as f:
        f.write(text)
    if is_hallucination(text):
        # The client shows "No speech recognised" for an empty text.
        return {"text": "", "heard": text, "duration": duration}
    return {"text": text, "duration": duration}
```

- [ ] **Step 5: Run tests, expect pass**

Run: `.venv/bin/python -m pytest tests/test_asr.py tests/test_server.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add asr.py server.py tests/test_asr.py
git commit -m "feat(asr): lexicon prompt, VAD and hallucination filter; backend and size choice"
```

---

### Task 11: Audio evaluation tooling

**Files:**
- Create: `data/asr_script.txt`, `tools/pair_asr_recordings.py`, `tools/eval_asr.py`
- Test: `tests/test_eval_asr.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_eval_asr.py`)

```python
import json
import sys

import pytest

sys.path.insert(0, "tools")
import eval_asr  # noqa: E402
import pair_asr_recordings  # noqa: E402


def test_word_error_rate():
    assert eval_asr.word_error_rate("show only bones", "show only bones") == 0
    assert eval_asr.word_error_rate("show only bones", "show bones") == pytest.approx(1 / 3)
    assert eval_asr.word_error_rate("darken skeleton", "Darkin' Skeleton.") == pytest.approx(1 / 2)


def test_pairing_takes_recordings_after_the_start_stamp_in_order(tmp_path):
    for stamp in ["20261022-100000", "20261022-100500", "20261022-100600"]:
        (tmp_path / f"{stamp}.wav").write_bytes(b"x")
    rows = pair_asr_recordings.pair(str(tmp_path), "20261022-100100", ["add kidneys", "reset"])
    assert [r["reference"] for r in rows] == ["add kidneys", "reset"]
    assert rows[0]["wav"].endswith("20261022-100500.wav")


def test_pairing_refuses_a_count_mismatch(tmp_path):
    (tmp_path / "20261022-100500.wav").write_bytes(b"x")
    with pytest.raises(ValueError, match="1 recordings for 2 script lines"):
        pair_asr_recordings.pair(str(tmp_path), "20261022-100000", ["a", "b"])
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_eval_asr.py -q`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Create `data/asr_script.txt`** (one command per line, read aloud in order)

```
show only bones
add kidneys
make the liver translucent
mute the bones
solo vessels
darken the skeleton
lighten the lungs
highlight the heart
reveal the skeleton
fade out the soft tissue
ghost the muscles a bit
crank up the lungs
dial the heart down a touch
make the ribcage pop
alpha up on the vessels
make the lungs see-through
narrower peak for the skeleton
shift the bone peak up
brighten the kidneys a lot
the spleen is way too dark
back to default
turn off the soft tissue
show me just the heart please
more aorta, less muscle
hide the bones and brighten the liver
show only the kidneys
make the liver more visible
rotate left
zoom in a bit
focus on the vessels
the backbone is barely visible
set the heart to high
make the vessels a little more solid
everything except the skeleton should go away
I can't see the kidneys
reset
```

- [ ] **Step 4: Create `tools/pair_asr_recordings.py`**

```python
"""Pair the recordings of one scripted session with the script lines.

Read data/asr_script.txt aloud in the app, one trigger press per line, after
noting the time. Then:

    .venv/bin/python tools/pair_asr_recordings.py --since 20261022-100000

writes data/asr_eval.json. Refuses when the number of recordings differs from
the number of lines -- delete the extra .wav/.txt pairs (retakes) first.
"""
import argparse
import glob
import json
import os


def pair(audio_dir: str, since: str, lines: list) -> list:
    wavs = sorted(p for p in glob.glob(os.path.join(audio_dir, "*.wav"))
                  if os.path.basename(p)[:-4] >= since)
    if len(wavs) != len(lines):
        raise ValueError(f"{len(wavs)} recordings for {len(lines)} script lines")
    return [{"wav": w, "reference": line} for w, line in zip(wavs, lines)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True, help="first recording's stamp, YYYYMMDD-HHMMSS")
    ap.add_argument("--script", default="data/asr_script.txt")
    ap.add_argument("--audio-dir", default="out/audio")
    ap.add_argument("--out", default="data/asr_eval.json")
    args = ap.parse_args(argv)
    with open(args.script) as f:
        lines = [line.strip() for line in f if line.strip()]
    rows = pair(args.audio_dir, args.since, lines)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {len(rows)} pairs to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `tools/eval_asr.py`**

```python
"""Speech recognition on the scripted recordings (data/asr_eval.json): word
error rate, command accuracy through the hybrid parser, and time per clip,
for each backend x model size x vocabulary prompt.

    .venv/bin/python tools/eval_asr.py --backends faster-whisper,mlx --sizes small,turbo
"""
import argparse
import json
import re
import sys
import time

sys.path.insert(0, ".")
import asr  # noqa: E402
import commands  # noqa: E402
import lexicon  # noqa: E402
from eval_parsers import matches, percentile  # noqa: E402


def _words(text):
    return re.sub(r"[^a-z' ]", " ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = [w.strip("'") for w in _words(reference)], [w.strip("'") for w in _words(hypothesis)]
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def _parse(text):
    try:
        return commands.parse_command_hybrid(text)[0]
    except ValueError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="data/asr_eval.json")
    ap.add_argument("--backends", default="faster-whisper")
    ap.add_argument("--sizes", default="small")
    args = ap.parse_args(argv)
    commands.LLM_LOG_PATH = "out/llm_requests_eval.jsonl"
    commands.UTTERANCE_LOG_PATH = "out/utterances_eval.jsonl"
    with open(args.pairs) as f:
        pairs = json.load(f)
    expected = [_parse(p["reference"]) for p in pairs]
    prompt = lexicon.get()["asr_prompt"]
    print(f"{'backend':<15} {'size':<6} {'prompt':<6} {'WER':>6} {'cmd ok':>7} {'p50 s':>6} {'p90 s':>6}")
    for backend in args.backends.split(","):
        for size in args.sizes.split(","):
            asr._transcribe_path(pairs[0]["wav"], "en", size, backend=backend)  # load + warm up
            for use_prompt in (False, True):
                wers, oks, secs = [], [], []
                for p, exp in zip(pairs, expected):
                    start = time.perf_counter()
                    text = asr._transcribe_path(p["wav"], "en", size, initial_prompt=prompt if use_prompt else None,
                                                backend=backend)
                    secs.append(time.perf_counter() - start)
                    wers.append(word_error_rate(p["reference"], text))
                    got = None if asr.is_hallucination(text) else _parse(text)
                    oks.append(matches(exp, got))
                n = len(pairs)
                print(f"{backend:<15} {size:<6} {'yes' if use_prompt else 'no':<6} {sum(wers) / n:6.1%} "
                      f"{sum(oks) / n:7.0%} {percentile(secs, 50):6.2f} {percentile(secs, 90):6.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests, expect pass**

Run: `.venv/bin/python -m pytest tests/test_eval_asr.py -q`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add data/asr_script.txt tools/pair_asr_recordings.py tools/eval_asr.py tests/test_eval_asr.py
git commit -m "feat(tools): scripted audio evaluation for speech recognition"
```

---

### Task 12: Timings, rejection on the VR panel, Ollama check and warm-up

**Files:**
- Modify: `commands.py` (add `check_llm`, `warm_llm`), `server.py` (`/api/command`, `__main__`), `static/app.js` (`sendCommandImpl`, recorder `onstop`)
- Test: `tests/test_commands.py`, `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_commands.py`:

```python
def test_check_llm_reports_unreachable_and_missing_models(monkeypatch):
    def refused(req, timeout=None):
        raise commands.URLError("refused")
    monkeypatch.setattr(commands, "urlopen", refused)
    assert "not reachable" in commands.check_llm("qwen2.5:7b")

    class Tags:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"models": [{"name": "llama3.2:latest"}]}).encode()
    monkeypatch.setattr(commands, "urlopen", lambda req, timeout=None: Tags())
    assert "not pulled" in commands.check_llm("qwen2.5:7b")
    assert commands.check_llm("llama3.2:latest") is None
```

Append to `tests/test_server.py` (follow the file's existing pattern of calling route functions directly, not through TestClient):

```python
def test_command_route_adds_timings(monkeypatch):
    import asyncio
    import server
    monkeypatch.setattr(server.session, "command",
                        lambda *a, **k: {"current": {"parse_ms": 12.0, "parser_used": "rule"}})
    out = asyncio.run(server.command(server.CommandRequest(text="add kidneys")))
    assert out["timings"]["parse_ms"] == 12.0
    assert out["timings"]["total_ms"] >= 0
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_commands.py tests/test_server.py -q -k "check_llm or timings"`
Expected: FAIL.

- [ ] **Step 3: Add `check_llm` and `warm_llm` to `commands.py`** (after `parse_command_hybrid`)

```python
def check_llm(model: str = DEFAULT_LLM_MODEL, host: str = None):
    """None when Ollama answers and has `model`, else what is wrong."""
    host = host if host is not None else OLLAMA_HOST
    try:
        with urlopen(Request(f"{host}/api/tags"), timeout=3) as resp:
            names = {m["name"] for m in _json.loads(resp.read().decode()).get("models", [])}
    except (URLError, OSError, ValueError) as exc:
        return f"Ollama not reachable at {host} ({exc}) -- start it with `ollama serve`"
    if model not in names and f"{model}:latest" not in names:
        return f"model {model!r} not pulled -- run `ollama pull {model}`"
    return None


def warm_llm(model: str = DEFAULT_LLM_MODEL, host: str = None) -> None:
    """Load the model now so the first voice command has no cold start."""
    host = host if host is not None else OLLAMA_HOST
    body = _json.dumps({"model": model, "prompt": "", "keep_alive": "30m"}).encode()
    with urlopen(Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"}),
                 timeout=60) as resp:
        resp.read()
```

- [ ] **Step 4: Server timings and warm-up** (`server.py`)

Replace the `/api/command` handler with:

```python
@app.post("/api/command")
async def command(req: CommandRequest):
    start = time.perf_counter()
    try:
        result = session.command(req.text, req.parser, req.model, req.search, req.steps, req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    total_ms = round((time.perf_counter() - start) * 1000, 1)
    # The parser meta (parse_ms, parser_used, ...) is spread into the step
    # (_render_step), i.e. result["current"].
    result["timings"] = {"parse_ms": (result.get("current") or {}).get("parse_ms"), "total_ms": total_ms}
    return result
```

In the `if __name__ == "__main__":` block, before `uvicorn.run(...)`, add:

```python
    import threading

    def _llm_ready():
        problem = commands.check_llm()
        if problem:
            print(f"\n[llm-parser] WARNING: {problem}\n[llm-parser] voice commands use the rule parser only\n")
            return
        commands.warm_llm()
        print(f"[llm-parser] {commands.DEFAULT_LLM_MODEL} loaded")
    threading.Thread(target=_llm_ready, daemon=True).start()
```

- [ ] **Step 5: Client: timing log and rejection on the panel** (`static/app.js`)

In `sendCommandImpl`, replace the non-200 branch with:

```javascript
  if (r.status !== 200) {
    const err = await r.json();
    showToast(`Could not parse that: ${err.detail}`, "destructive");
    // The VR panel listens for app:error; without this a rejection was silent there.
    emitAppEvent("error", { message: `Didn't understand "${text}"` });
    return null;
  }
```

and make the function end with `return data;` after the `emitAppEvent("applied", ...)` call.

In the recorder's `onstop` handler, time the stages:

```javascript
    const t0 = performance.now();
    ...
        const r = await fetch("/api/transcribe", { method: "POST", body: form });
        const data = await r.json();
        const t1 = performance.now();
        if (data.text) {
          emitAppEvent("transcript", { text: data.text });
          const reply = await sendCommandImpl(data.text);
          const t2 = performance.now();
          // Same line in the desktop console and the headset's logcat.
          console.info(`[voice-timing] asr=${Math.round(t1 - t0)}ms command+render=${Math.round(t2 - t1)}ms `
            + `total=${Math.round(t2 - t0)}ms parse=${reply?.timings?.parse_ms ?? "-"}ms path=${reply?.current?.parser_used ?? "-"}`);
        } else {
          emitAppEvent("error", { message: "No speech recognised" });
        }
```

(`t0` goes on the first line of `onstop`; the call inside `withLoading` changes from `sendCommand(data.text)` to `sendCommandImpl(data.text)` because `withLoading` is already active there.)

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest -q && node --test tests/js/*.test.js`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add commands.py server.py static/app.js tests/test_commands.py tests/test_server.py
git commit -m "feat(voice): stage timings, rejection on the VR panel, Ollama check and warm-up"
```

---

### Task 13: Choose the LLM

**Files:**
- Modify: `commands.py` (`DEFAULT_LLM_MODEL`), `docs/experiments/2026-10-21-voice-lexicon.md`

- [ ] **Step 1: Pull the candidates**

Check https://ollama.com/library for the current tags of the Qwen 3.5, Gemma 4 and Granite 4 small instruct models (≈3–8B). Then, for example:

```bash
ollama pull qwen3.5:4b
ollama pull gemma4:e4b
ollama pull granite4:3b
ollama list
```

Expected: all three listed. If a tag does not exist, use the nearest small size of that family and note the exact tag in the experiment doc.

- [ ] **Step 2: Evaluate each model, schema and json**

For each `MODEL` in `qwen2.5:7b` and the three candidates:

```bash
.venv/bin/python eval_parsers.py --set realistic --parsers llm,hybrid --llm-model MODEL --format schema --show-heldout
.venv/bin/python eval_parsers.py --set realistic --parsers llm,hybrid --llm-model MODEL --format json
```

Run each model once before measuring (first call loads it). Expected: one table per run; copy all into the experiment doc under "## Model comparison".

- [ ] **Step 3: Choose**

Winner: highest `hybrid` exact accuracy on realistic held-out (all sources) whose `llm` p90 is ≤ 1500 ms; ties go to the smaller model. Set `DEFAULT_LLM_MODEL` in `commands.py` to its tag and `LLM_FORMAT` to the better format. Write the reason in the experiment doc in two sentences.

- [ ] **Step 4: Confirm nothing regressed**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python eval_parsers.py --parsers rule,hybrid`
Expected: tests pass; base set rule 100 %, hybrid ≥ the base-set LLM baseline.

- [ ] **Step 5: Commit**

```bash
git add commands.py docs/experiments/2026-10-21-voice-lexicon.md
git commit -m "feat(llm-parser): switch the default model after the realistic-set comparison"
```

---

### Task 14: Audio session and ASR choice (needs the author)

**Files:**
- Create: `data/asr_eval.json` (generated)
- Modify: `asr.py` (`ASR_MODEL` default, backend), `data/lexicon.json` (`asr_fixes`), `docs/experiments/2026-10-21-voice-lexicon.md`

- [ ] **Step 1: Install mlx-whisper and check ffmpeg**

Run: `.venv/bin/pip install mlx-whisper && which ffmpeg`
Expected: install succeeds; ffmpeg found (mlx-whisper decodes the browser's webm through it). If ffmpeg is missing: `brew install ffmpeg`.

- [ ] **Step 2: Record**

Start the server (`.venv/bin/python server.py --host 0.0.0.0 --ssl-certfile certs/lan.pem --ssl-keyfile certs/lan-key.pem`), note the stamp (`date +%Y%m%d-%H%M%S`), and ask the author to read `data/asr_script.txt` line by line with one trigger press per line, in the headset (the microphone that matters). Retakes: delete the bad `.wav`/`.txt` pair in `out/audio/`.

- [ ] **Step 3: Pair and evaluate**

```bash
.venv/bin/python tools/pair_asr_recordings.py --since <stamp>
.venv/bin/python tools/eval_asr.py --backends faster-whisper,mlx --sizes small,turbo
```

Expected: 8 rows (2 backends × 2 sizes × prompt off/on). Copy the table into the experiment doc under "## Speech recognition".

- [ ] **Step 4: Choose and feed back**

Pick the fastest configuration whose command accuracy is within 3 points of the best and whose p90 ≤ 1.5 s. Set `ASR_MODEL` default (and `ASR_BACKEND` if faster-whisper should win although mlx is installed) in `asr.py`. Add every misrecognition of a lexicon word seen in the chosen configuration's transcripts to `asr_fixes` in `data/lexicon.json` (wrong word → right word), then rerun `tools/eval_asr.py` for that configuration only and record the after-number.

- [ ] **Step 5: Run tests and commit**

```bash
.venv/bin/python -m pytest -q
git add asr.py data/lexicon.json data/asr_eval.json docs/experiments/2026-10-21-voice-lexicon.md
git commit -m "feat(asr): choose backend and size from the scripted recordings"
```

---

### Task 15: End-to-end in the headset and write-up

**Files:**
- Modify: `docs/experiments/2026-10-21-voice-lexicon.md`, `README.md` (voice section), `docs/STATUS.md` (one row in "Since")

- [ ] **Step 1: Measure in the headset**

With the server running and the headset connected (`tools/headset/README.md`), clear logcat (`adb logcat -c`), have the author give 20 varied commands in VR, then:

```bash
adb logcat -d | grep "voice-timing"
```

Expected: 20 lines. Compute p50 and p90 of `total=` and paste the lines' summary into the experiment doc under "## End to end". Success: p90 ≤ 4000 ms.

- [ ] **Step 2: Final realistic-set numbers**

Run: `.venv/bin/python eval_parsers.py --set realistic --parsers rule,llm,hybrid --show-heldout`
Expected: hybrid held-out exact ≥ 90 % (spec success criterion 2) and above both Task 3 baselines. Paste the table under "## Result"; list remaining held-out failures with one line each on why.

- [ ] **Step 3: Update README and STATUS**

In `README.md`'s voice/parser section, describe the pipeline in three sentences (lexicon → normalize → rule if fully covered, else LLM; Whisper with vocabulary prompt and hallucination filter) and point to `data/lexicon.json` for adding words. Add a `| 10-21 | Voice lexicon: ... |` row to the "Since" table in `docs/STATUS.md` linking the experiment doc.

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/2026-10-21-voice-lexicon.md README.md docs/STATUS.md
git commit -m "docs: voice lexicon results and pipeline"
```
