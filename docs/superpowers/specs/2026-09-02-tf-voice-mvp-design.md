# Voice-Driven Transfer Function MVP — Design

Local prototype for a master's thesis: a synthetic CT phantom in Hounsfield units,
a 4-Gaussian-peak transfer function, offscreen VTK rendering, voice or text commands
that nudge the transfer function, and hill-climbing search evaluated either by an
exact Hounsfield-range metric or by a human. No RL, no torch, no LLM-generated
numeric parameters, no image-based reward model. Combines the original build prompt
with a follow-up change order (LLM parser alternative, preference-pair logging)
before any of it existed on disk — both are specified here as one build.

## Files

`phantom.py`, `transfer.py`, `render.py`, `asr.py`, `commands.py`, `evaluate.py`,
`stats.py`, `mvp.py`, `eval_parsers.py`, plus `README.md` and `data/parser_eval_phrases.json`.

## 1. Phantom (`phantom.py`)

Synthetic CT volume in Hounsfield units, default 96³, seed hardcoded (`42`) — not a
CLI flag, per "keep it simple." Values: air −1000, fat −100, soft tissue +40,
spongy bone +300, cortical bone +900, soft-edged blobs with mutual occlusion, fixed
Gaussian noise on top.

## 2. Transfer function (`transfer.py`)

24-float vector, 4 peaks × `[center, width, height, r, g, b]`, external range
`[-1, 1]` everywhere, mapped internally as:

- `center` → `[-1050, 2000]` HU
- `width` (Gaussian σ) → `[10, 400]` HU
- `height`, `r`, `g`, `b` → `[0, 1]`

Tissue table (name → HU band), the midpoints between the five reference values,
used both for `opacity_mass` integration bounds and "nearest peak" lookup:

| tissue | band (HU)     |
|--------|---------------|
| air    | < −550        |
| fat    | −550 … −30    |
| soft   | −30 … 170     |
| spongy | 170 … 600     |
| bone   | > 600         |

Default/reset vector: one peak seeded near fat/soft/spongy/bone (air omitted, stays
transparent), ascending heights `0.05 → 0.15 → 0.3 → 0.6`, tissue-plausible colors.

`vector_to_vtk(params) -> (vtkColorTransferFunction, vtkPiecewiseFunction)`
`opacity_mass(params, hu_lo, hu_hi) -> float` — integral of the opacity curve over the range.

## 3. Rendering (`render.py`)

`render(volume, params)` — offscreen, fixed camera, 512×400, `vtkGPUVolumeRayCastMapper`
with automatic fallback to `vtkFixedPointVolumeRayCastMapper` if GPU init fails (prints
which mapper is used). `grab(win) -> np.uint8 (H,W,3)`, flipped `[::-1]` for VTK's
bottom-up convention. `features(rgb) -> {mean, std, coverage, entropy}` (coverage = 
non-background pixel fraction, entropy = Shannon entropy of a 32-bin grayscale histogram).
`ResetCameraClippingRange()` after camera setup. `numpy_to_vtk(..., deep=True)`.

## 4. Speech input (`asr.py`)

`faster-whisper`, `device="cpu"`, `compute_type="int8"`, model size via flag
(default `small`), language via flag (default `en`). Model loaded once, reused.
Prefers `mlx-whisper` if importable, no hard dependency on it. `transcribe_mic()` —
Enter starts recording, Enter stops, no VAD. `transcribe_file(path)`. Every recording
written to `out/audio/<timestamp>.wav` with a matching `<timestamp>.txt` transcript.
Prints recognized text + duration per call.

## 5. Commands — two implementations behind one interface (`commands.py`)

Unified command schema, output of both parsers:

```
{target: <tissue name> | None, attribute: "opacity" | None,
 direction: "increase" | "decrease" | "show_only" | "reset",
 strength: "slightly" | "moderately" | "strongly" | None}
```

(`show_only` and `reset` carry `strength=None`; `reset` carries `target=None`.)

- **`parse_command_rule(text)`** — regex/keyword based, as originally specified.
- **`parse_command_llm(text, model="qwen2.5:7b")`** — calls local Ollama
  (`http://localhost:11434/api/generate`, `format="json"`, stdlib `urllib`, no new
  dependency) with a system prompt listing the five tissues and the schema, asking
  for pure JSON. Response is schema-validated (target ∈ tissue table ∪ None,
  direction ∈ the four values). On connection failure or schema-invalid output:
  fall back to `parse_command_rule(text)` and print one line stating the fallback
  and why. **The LLM never emits numeric values** — only the four categorical fields.
- CLI selects via `--parser rule|llm` (default `rule`); model name via `--llm-model`.

`apply_command(cmd, params) -> params` — unchanged logic: locate the peak nearest
the target tissue's HU band; if none within threshold (300 HU), seed a new peak at
the tissue's default center/width with a small height. Adjust height by the
strength-scaled delta (slightly=0.15, moderately=0.35, strongly=0.6, multiplicative
in normalized space, clamped). `show_only`: crush all other peaks' heights toward
~0.02, push the target peak's height toward 0.7. `reset`: return the default vector
directly (bypasses hill-climbing — a single step, not a search).

### Parser comparison (`eval_parsers.py` + `data/parser_eval_phrases.json`)

~20 free-form English phrases the rule parser is expected to *fail* on ("I can't see
the bones very well", "the soft tissue is hiding everything", "make the skeleton
pop", …), each with a hand-labeled expected `{target, direction}`. The script runs
both parsers over the list and prints a table: phrase, expected, rule-parser result
(✓/✗), llm-parser result (✓/✗), plus a summary accuracy line per parser. This table
is the actual point of the change — making the gap between the two measurable.

## 6. Evaluation (`evaluate.py`)

`objective(params_before, params_after, cmd) -> +1 | -1` — via `opacity_mass`:
`+1` iff the signed mass delta in the target's HU band moves the requested direction
by more than a small epsilon, **and** no other tissue band's `|Δmass|` exceeds the
target band's `|Δmass|`.

`human(params_before, params_after, cmd, before_png, after_png) -> +1 | -1` — writes
both PNGs, asks on the console for `b`/`s` — German *besser*/*schlechter* (better/worse),
mapped to `+1`/`-1`.

**No MLLM judge, no neural reward model.** `opacity_mass` is the reward signal for
every command with a nameable target tissue; there is nothing for an image-based
judge to add here.

### Preference-pair logging (`out/preferences.jsonl`)

Whenever the human evaluator is invoked, **both** verdicts are computed and logged —
the human's, and the objective evaluator's for the same before/after pair — even
though only one of them drives that run's hill-climbing accept/reject decision. One
JSONL line per human judgment:

```
{timestamp, session_id, cmd_text, cmd_dict,
 params_before, params_after,
 features_before, features_after,
 before_png, after_png,
 human_verdict: "better" | "worse",
 objective_verdict: "better" | "worse"}
```

`session_id`: one per `mvp.py` invocation (timestamp-based).

`stats.py` reads `preferences.jsonl` and prints: pair count per target tissue,
agreement rate between human and objective verdicts (overall and per tissue), and
the list of disagreement cases with their PNG paths for manual inspection.

## 7. Hill-climbing (`mvp.py` + `commands.py`)

Search dimension is derived from the command: for `increase|decrease opacity`, the
only searched dimension is that peak's `height` (a 1-D search — this command family
has exactly one relevant dimension). Step size initialized from the strength factor
above. `+1` → keep, step ×1.2. `−1` → revert, step ×0.5. Stop after `--steps` or
step `< 0.01`. `reset` and `show_only` are one-shot, no search loop.

Every step logged to `out/log.jsonl`: timestamp, command, params before/after,
features before/after, verdict — written whether or not it's later read.

## 8. CLI (`mvp.py`)

```
python mvp.py                                            # render start state -> out/start.png
python mvp.py --cmd "increase opacity for bone strongly"  # single apply -> before/after.png
python mvp.py --cmd "..." --learn --steps 15              # objective evaluator
python mvp.py --cmd "..." --learn --human --steps 10      # human evaluator, also logs preferences
python mvp.py --listen                                    # push-to-talk -> parse -> apply
python mvp.py --wav out/audio/xyz.wav                     # replay a prior recording
python mvp.py --cmd "..." --parser llm --llm-model qwen2.5:7b
```

Additional flags: `--model-size` (whisper, default `small`), `--lang` (default `en`).

Per step, stdout shows: command dict, opacity mass per tissue before/after, image
features, verdict.

## Pitfalls carried over

- `numpy_to_vtk(..., deep=True)` — otherwise crashes on freed memory.
- `ResetCameraClippingRange()` after every camera move.
- Deterministic: fixed seed, fixed camera, fixed resolution — same vector must
  yield the same features every time.

## Architecture diagram

`architecture-mvp.drawio` gets updated (via the drawio skill) to: promote the LLM
parser from a dashed "later" box to a solid, flag-selectable alternative behind the
rule parser; drop the "Reward Model / distilled from MLLM judge" box entirely; add
the `preferences.jsonl` output and its `stats.py` consumer; update the scope note to
state explicitly that an MLLM judge was considered and rejected in favor of the
exact `opacity_mass` metric plus logged human/objective preference pairs.

## Closing demonstration (unchanged from the original prompt)

1. `increase opacity for bone strongly` with `--learn --steps 15` — table showing
   whether bone-band opacity mass rises monotonically.
2. Sequential: emphasize bone, emphasize soft tissue, emphasize bone again — table
   showing where it "swims" (the interesting failure case, not the success case).

Plus, for this change order: the `eval_parsers.py` accuracy table for rule vs. LLM
parser over the ~20 free-form phrases.
