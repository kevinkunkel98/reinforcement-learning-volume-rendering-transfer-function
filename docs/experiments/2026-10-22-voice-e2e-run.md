# Voice end to end over the scripted commands — 2026-10-22

All 36 lines of `data/asr_script.txt`, each spoken by two macOS voices
(Eddy, US English; Anna, a German voice speaking English as an accent
stand-in), then the live pipeline: Whisper `small` (faster-whisper, CPU) with
the lexicon prompt and hallucination filter → `parse_command_hybrid`
(`qwen2.5:7b` for uncovered sentences) → the server's policy path (policy +
planning + 200-evaluation refine + keep repair; exact `apply_command` for
non-goals) → render. Every command starts from the task's start TF on
`ts_s0477`, independently (no chaining). Branch `voice-lexicon` at `592be94`
plus the lexicon/STATUS commits after it.

Reproduce: `.venv/bin/python docs/experiments/2026-10-22-scripts/e2e_script.py <out_dir>`
(~25 min). Rows: `2026-10-22-scripts/e2e_results.json`. Renders (English
voice): `2026-10-22-renders/e2e_sheet.png`.

Synthetic speech is cleaner than a person in a headset and its accent is not
a real speaker's; read the ASR rows as a lower bound on errors, not a measure.

## Summary

| | English voice | German voice |
|---|---|---|
| rule parser answered | 27 | 24 |
| LLM answered | 4 | 9 |
| rejected | 5 | 3 |
| right command (by hand) | 30 / 36 | 28 / 36 |

Latency (p50 / p90): ASR 0.90 / 0.96 s; LLM, when used, 1.38 / 1.55 s; rule
parser < 3 ms; **policy path 3.53 / 3.75 s**, of which the 200-evaluation
refine is ~3.0 s (same command: plan + refine 200 = 3.59 s, plan only
0.60 s, policy only 0.48 s). Voice to render is therefore ~4.5 s on the rule
path and ~6 s on the LLM path — over the 4 s budget because of the refine,
not the voice stages.

## Oddities, by stage

### Latency

1. **The refine dominates.** `POLICY_REFINE_EVALUATIONS = 200` costs ~3 s per
   command; everything else together is ~1.5 s. Options: a smaller budget
   (STATUS item 4's render-budget curve answers what 30 or 50 cost in
   attainment), or show the policy + keep answer at once and replace it when
   the refine finishes.

### Speech recognition

2. **"heart" → "art" / "arc"** (English voice, 3 of 3 sentences with
   "heart" in an unstressed position: "Highlight the art", "Dial the arc down
   the touch", "Show me just the art"). All three were rejected. Fuzzy
   normalization skips words under 4 letters, so `asr_fixes` is the fix
   (`art`, `arc` → `heart`; harmless in this vocabulary).
3. **Short words flip meaning** (German voice): "add kidneys" → "Cut
   kidneys" (→ decrease strongly: the opposite), "ghost the muscles" →
   "Post the muscless" (→ LLM guessed *increase*), "rotate left" → "Rotate
   lift" (→ LLM guessed tilt up), "set the heart to high" → "…to ight".
4. **Other misses**: "ribcage" → "ripsage" (English; below the fuzzy
   threshold, rejected), "vessels" → "Vsits" (rejected), "more aorta" →
   "Moriorta" / "More allotter". Good: "see-throoves", "entbreiten the
   liver" (= brighten) were still parsed right by the LLM.

### Parser

5. **The LLM guesses instead of refusing** on garbled input: "Post the
   muscless abit" → soft opacity *increase*; "Rotate lift" → camera tilt up;
   "Set the heart to ight" → heart *brightness* set high, which renders the
   heart white (sheet, 31). The grounding check only verifies class names, not
   direction or attribute. Options: show the parsed command on the panel
   for LLM-path answers ("I understood: …") or ask the model for an
   explicit "unsure".
6. **A compound clause is dropped silently**: "Moriorta, less muscle" → only
   the muscle part (soft decrease). The user is not told half the sentence
   was ignored.
7. "Barrow or peak for the skeleton" → the LLM produced a center `set`
   (not allowed) → rejected. Fine as a rejection; the intent (narrower) was
   recoverable.

### Policy and rendering

8. **Increasing one organ erases soft tissue** ("crank up the lungs",
   "make the liver more visible": soft −5.0 log-vis, plus the blue vertical
   streaks of the unlabelled shell; sheet 11, 26). The goal marks soft, the
   skeleton and the shell as occluders (weight 0), so the keep repair never
   reverts them. Open decision: make weight-0 occluders keep-repair
   candidates (the attainment check keeps the fades the goal needs).
9. **Commands with no visible effect** while attainment is high: "darken
   the skeleton" (0.97), "brighten the kidneys a lot" (0.97), "shift the bone
   peak up", "alpha up on the vessels", "make the vessels a little more
   solid". The targets sit behind other tissue (known weakness for small or
   occluded organs); the user sees nothing happen. A panel hint ("kidneys are
   hidden behind soft tissue") would at least explain it.
10. **Vessels on a scan without contrast are inconsistent**: `ts_s0477` has
   no vessel goal ("no contrast"), so vessel commands go exact. "solo
   vessels" / "focus on the vessels" then show the aorta fine (sheet 4, 29),
   while "alpha up" / "more solid" change nothing visible. Either support
   the class on the exact path consistently or refuse with the reason.
11. **show_only scores low**: attainment 0.59 (skeleton), 0.64 (heart), 0.67
   (kidneys); the renders look right, so this may be the metric's show_only
   weighting rather than the policy.

### Not app issues

- Camera commands (27, 28) render unchanged because the script renders every
  row with one fixed camera; the parse was right.
- "fade out the soft tissue" and "reveal the skeleton" give the same image:
  expected, removing soft tissue reveals the skeleton.

## Next (proposed order)

1. `asr_fixes` for heart/art/arc and ribcage/ripsage (cheap, from this run).
2. Panel shows the parsed command for LLM-path answers; dropped compound
   clauses are reported (5, 6).
3. Refine budget vs latency (1): measure attainment at refine 0 / 30 / 50 /
   200 on the 300 val/test episodes, pick the budget that keeps voice to
   render under 4 s.
4. Occluders as keep-repair candidates (8), measured like the keep repair.
5. Vessel handling on no-contrast scans (10) and a "hidden target" hint (9).
