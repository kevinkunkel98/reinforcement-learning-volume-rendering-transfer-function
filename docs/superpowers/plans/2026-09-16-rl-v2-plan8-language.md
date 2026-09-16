# RL v2 Plan 8: Language to Goals

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the loop from spoken or typed words to a transfer function: the parser speaks the anatomical vocabulary, any parsed command becomes a goal vector, and the trained policy answers instructions typed into the chat UI.

**Architecture:** `commands.py` gains the anatomical vocabulary and the phrasings people actually use ("more bone", "a bit less soft tissue", compound instructions). `goals.py` gains `goal_from_command`, turning a parsed command into the same 16-value goal vector the sampler produces. `server.py` gains a `policy` mode that runs the one-shot policy on that goal, so the chat UI demonstrates the whole pipeline end to end.

**Tech Stack:** Python 3.14 (`.venv`), FastAPI, Ollama (optional, for the LLM parser), pytest.

**Why now:** the policy is trained and evaluated but only ever received sampled instructions. Until the parser maps words to the anatomical classes, nothing a user says can reach it, and the speech/VR story has a hole in the middle.

**Established facts:**
- Goal classes: `skeleton`, `lungs`, `soft` (organs + muscle), `vessels` (contrast scans only). Measurement keeps organs and muscle separate; goals do not, because no transfer function separates them.
- `goals.sample_instruction` already produces `{"kind", "text", "targets", "goal"}`; `targets` maps a goal class to `{"vis": float}` and/or `{"bright": float}`, and `goal_vector(targets)` encodes it.
- Strength words map to log₁₀ visibility changes (slightly 0.15, moderately 0.3, strongly 0.6) and brightness changes (0.1 / 0.2 / 0.4); absolute levels map to a share of `solo_max` (low 0.1, medium 0.4, high 0.8).
- `commands.py` currently knows `bone`, `spongy`, `soft`, `fat`, `air` with synonyms, and `parse_command_rule` handles relative, absolute-level, show-only, compound (absolute only), brightness, width, centre, camera and reset.
- Known parser bugs to fix: `"more bone"` does not parse; a sentence with two relative clauses silently returns only the first.
- The one-shot policy lives at `out/rl_v2/oneshot_v2_seed0/best.zip`; `rl/candidates.py` shows how to build its observation outside a Gym env.

**Repo rules:** run from the repository root with `.venv/bin/python`; plain commit messages with **no** trailers; stage files explicitly by path; nothing under `out/` is committed; leave untracked `docs/prompts_report.pdf`, `docs/rl-v2-pipeline.pdf` and `docs/vr-integration.md` alone; branch `main` (the RL work is merged).

---

### Task 1: Anatomical vocabulary in the parser

**Files:** modify `commands.py`, `tests/test_commands.py`, `data/parser_eval_phrases.json`.

The parser keeps its existing command shapes; only the tissue vocabulary and a few phrasings change.

- Vocabulary (synonyms → goal class): **skeleton** ← bone, bones, skeleton, ribs, rib, spine, vertebrae, hip, femur, skull; **lungs** ← lung, lungs, pulmonary; **soft** ← soft tissue, soft, organs, organ, muscle, muscles, liver, kidney, spleen (an organ name maps to `soft` because no transfer function isolates one organ — the parser must not promise what the renderer cannot deliver); **vessels** ← vessel, vessels, artery, arteries, vein, veins, aorta, contrast.
- Keep the old names working as aliases where they are unambiguous (`bone` → skeleton), and **reject** `fat`, `air`, `spongy` with a clear message naming the supported classes — they are no longer classes, and silently mapping them would corrupt collected data.
- New phrasings: `"more <class>"`, `"less <class>"`, `"a bit more <class>"`, `"much less <class>"`, `"show me the <class>"` (→ show only).
- Fix: a sentence with several relative clauses returns a `compound` of all of them, or raises — never silently drops one.

- [ ] **Step 1: Write failing tests** in `tests/test_commands.py`: each synonym group resolves to its class; `"more bone"` → relative increase on skeleton with default strength; `"a bit less soft tissue"` → decrease, strength slightly; `"more bone, a bit less soft tissue"` → a compound with both; `"show me the lungs"` → show-only; `fat`/`air`/`spongy` raise with a message naming the four classes; every existing camera, reset, width and centre command still parses unchanged.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Extend `data/parser_eval_phrases.json` with at least three phrasings per instruction kind, then run `python eval_parsers.py` and report per-kind accuracy for the rule parser.
- [ ] **Step 6:** Commit — `feat(parser): anatomical vocabulary and natural phrasings`.

---

### Task 2: Commands become goals

**Files:** modify `goals.py`, `tests/test_goals.py`.

```python
def goal_from_command(command: dict, model, start_features: dict, volume: str = None) -> dict:
    """Turn a parsed command into the same {"kind", "text", "targets", "goal"}
    shape `sample_instruction` produces.

    Raises ValueError for commands that are not goals (width, centre, camera,
    reset — those are applied exactly by `commands.apply_command`), and for
    goals the volume cannot support (vessels on a plain scan, a class the scan
    does not contain).
    """
```

Mapping: relative → `±VISIBILITY_STRENGTH[strength]`; brightness → `±BRIGHTNESS_STRENGTH[strength]`; absolute level → the `solo_max` share, as a change from the current state; show-only → named classes to a high level and every other class to the hide floor; compound → the union of its sub-commands.

- [ ] **Step 1: Write failing tests:** each command kind maps to the documented entries; a compound merges its parts; a non-goal command raises with a message naming `apply_command`; vessels on a non-contrast volume raises; a class absent from the volume raises; the resulting goal vector matches what `sample_instruction` would produce for the same targets.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit — `feat(rl): turn parsed commands into goal vectors`.

---

### Task 3: The policy answers in the chat UI

**Files:** modify `server.py`, `static/app.js`, `static/index.html`, `tests/test_server.py`.

- `Session.command(..., mode="exact"|"search"|"policy")`. `"policy"` parses the text, builds the goal with `goal_from_command`, runs the one-shot policy on the current volume, and applies the resulting transfer function as one history step. Falls back to `"exact"` with a toast-worthy message when no policy checkpoint exists or the command is not a goal (camera, reset, sharpen…).
- The UI toolbar gains a third mode next to the existing search toggle; the step records which mode produced it.
- Load the policy lazily and cache it, as `collect.py` does.

- [ ] **Step 1: Write failing tests** (driving `Session` directly, as the existing tests do): `mode="policy"` with a stub policy appends one step whose parameters came from the policy; a camera command in policy mode still moves the camera; an unavailable checkpoint falls back to exact application and says so; the step records its mode.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Manual check: start the server, type `more bone` in policy mode on `ts_s1379`, confirm the render changes and the skeleton is more visible (`visibility.for_volume` before/after).
- [ ] **Step 6:** Commit — `feat(ui): answer instructions with the learned policy`.

---

### Task 4: The LLM parser speaks the new vocabulary

**Files:** modify `commands.py` (the LLM prompt and validation), `tests/test_commands.py`.

- Update the prompt's tissue list and examples to the four classes; allow `compound` to contain relative sub-commands with `strength`; keep the JSON shape otherwise unchanged.
- Validation rejects the retired names with the same message the rule parser uses.

- [ ] **Step 1: Write failing tests** against the validator (no Ollama needed): a compound of relative sub-commands validates; a retired tissue name is rejected; the documented shape is unchanged.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** If Ollama is running, `python eval_parsers.py --llm-model qwen2.5:7b` and report accuracy; if not, say so and skip.
- [ ] **Step 6:** Commit — `feat(parser): LLM prompt for the anatomical vocabulary`.

---

### Task 5: Documentation

- [ ] **Step 1:** Regenerate `COMMANDS.md` (`python -m tools.gen_commands_doc`) and update the README's command block to the anatomical vocabulary, including the policy mode.
- [ ] **Step 2:** Commit — `docs: anatomical command vocabulary`.
