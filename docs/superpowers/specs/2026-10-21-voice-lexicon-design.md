# Voice command lexicon — design

2026-10-21. Status: approved in conversation, not yet planned.

## Problem

The rule parser scores 100 % on `data/parser_eval_set.json` (57 dev / 50
held-out), yet in the 2026-10-21 VR session it rejected 14 of 24 real voice
commands. The eval set no longer represents how people talk to the app: no
visualization lingo, no everyday anatomy words, no speech-recognition errors
("Darkin' Skeleton" for "darken skeleton"). The LLM parser could not help: Ollama
was not running and nothing warned about it. Spoken text of rejected commands
was not logged anywhere.

## Users and constraints

- Speakers: the author and other computer scientists from visualization.
  Expect transfer-function vocabulary (opacity/alpha, translucent, peak,
  window, narrower, solo/isolate, focus + context) and lay anatomy (ribcage,
  backbone, guts), not clinical terms.
- Language: English only, robust to German accents and the transcription
  errors they cause.
- Latency: p90 at most 4 s from releasing the trigger to the new image in the
  headset. Accuracy matters more than speed within that budget.

## Approach

One lexicon file is the single source of the command vocabulary. It feeds
speech recognition, a normalization step, the rule parser and the LLM. The rule
parser answers when it accounts for every content word of the utterance;
everything else goes to the LLM.

```
voice -> ASR (Whisper, initial_prompt from lexicon)
      -> normalize (lexicon asr_fixes, then conservative fuzzy match)
      -> rule parser (tables from lexicon)
           fully covered -> command
           otherwise     -> LLM (prompt section + JSON schema from lexicon)
                              valid   -> command
                              invalid -> rule result if any, else rejection
```

Rejected alternatives: LLM-first for every command (simpler, but the same
sentence can parse differently and an LLM error has no safety net);
embedding-based intent matching (weak on compound commands and strengths, less
control).

## Components

1. **`data/lexicon.json`**: data only. Replaces the hand-written vocabulary in
   `commands.py` (`CLASS_SYNONYMS`, `RETIRED_CLASS_WORDS`, `DIRECT_VERBS`,
   `ATTRIBUTE_WORD_ALIASES`, the strength and direction word regexes where they
   are pure word lists).
2. **`lexicon.py`**: loads and validates the file; builds the ASR prompt, the
   normalization tables, the rule parser's lookup tables, and the LLM prompt
   section and JSON schema. Pure, unit-tested.
3. **Normalization**: text -> text before parsing. Explicit `asr_fixes` first,
   then fuzzy matching of unknown words against lexicon words (edit distance
   plus a phonetic key) with a conservative threshold. Never rewrites a word
   that is already a known word or filler.
4. **Coverage check**: the rule parser also returns the words it consumed. Any
   leftover content word (not filler) means "not sure", and the LLM gets the
   command.
5. **Stage timings**: ASR, normalize, rule, LLM, apply/render in the
   `/api/command` response and the log.
6. **Utterance log**: every utterance with raw text, normalized text, path
   taken (rule / llm / fallback), result or rejection reason.

## Lexicon format

Every entry maps a phrase to part of a command; the rule parser assembles a
clause from the parts.

```json
{
  "version": 1,
  "classes": {
    "skeleton": ["bone", "bones", "skeleton", "ribcage", "backbone", "spine", "pelvis", "skull"],
    "soft": ["soft tissue", "muscle", "meat", "flesh", "body outline", "skin"]
  },
  "retired": { "fat": ["fat", "adipose"], "air": ["air", "background"] },
  "phrases": [
    { "say": ["fade out", "get rid of", "hide", "mute", "turn off"],
      "sets": { "attribute": "opacity", "direction": "decrease", "strength": "strongly" } },
    { "say": ["translucent", "see-through", "ghost", "more transparent"],
      "sets": { "attribute": "opacity", "direction": "decrease", "strength": "moderately" } },
    { "say": ["more opaque", "alpha up", "solid", "pop"],
      "sets": { "attribute": "opacity", "direction": "increase" } },
    { "say": ["narrower", "tighter", "crisper", "sharpen"],
      "sets": { "attribute": "width", "direction": "decrease" } },
    { "say": ["solo", "isolate", "only show", "focus on"],
      "sets": { "direction": "show_only" } },
    { "say": ["undo everything", "back to default", "revert"],
      "sets": { "direction": "reset" } }
  ],
  "strengths": {
    "slightly": ["a bit", "a touch", "a smidge", "barely"],
    "strongly": ["a lot", "way", "crank", "max", "completely"]
  },
  "asr_fixes": { "darkin": "darken", "sgeleton": "skeleton", "lever": "liver", "kitneys": "kidneys" },
  "filler": ["please", "can you", "could you", "the", "a", "now"]
}
```

(The entries above illustrate the format; the real contents come from the
layers below.)

Contents, in four layers:
1. Today's vocabulary, carried over unchanged (the existing eval set must stay
   at 100 %).
2. Visualization / transfer-function lingo.
3. Everyday anatomy words.
4. Accent and transcription variants taken from real transcripts (the log and
   the audio recording below), not invented.

Rules:
- A phrase belongs to exactly one entry; the loader rejects duplicates.
- Words whose meaning depends on context ("light": brightness or opacity?)
  stay out of the lexicon and go to the LLM.
- `retired` classes keep being rejected explicitly, never remapped.

## Evaluation

**Realistic text set**: a new file next to `data/parser_eval_set.json`, same
schema (expected command, or null for "should be rejected"). Sources: the
2026-10-21 utterances in `out/llm_requests.jsonl`, phrasings by the author and
colleagues, and variants written by Claude and reviewed by the author. Split
`dev` (lexicon work is done against it) and `heldout` (written before any
lexicon work, reporting only).

**Audio set**: `/api/transcribe` already saves each recording with its
transcript to `out/audio/`. The author records 30–40 commands once; reference text per clip.
Measured: word error rate and end-to-end command accuracy for faster-whisper
vs mlx-whisper, small vs turbo, with and without the lexicon `initial_prompt`.

**LLM choice**: `eval_parsers.py` gains per-phrase latency (median, p90) and a
JSON-schema option. Two or three current small models (verify tags in the
Ollama library) against `qwen2.5:7b`. Winner: highest exact accuracy on
realistic held-out with LLM p90 within its budget.

**Budget (p90)**: ASR ≤ 1.5 s, LLM ≤ 1.5 s, everything else ≤ 1 s; total
≤ 4 s, measured in the headset.

## Error handling

- Whisper hallucinations: the 2026-10-21 recordings contain "You" eleven times
  and two invented sentences from silence or noise. Voice-activity filtering
  (faster-whisper `vad_filter`) plus a filter for known hallucination phrases
  and over-long transcripts; a filtered recording is reported as "No speech
  recognised" and never reaches the parser.

- Server start: check Ollama is reachable and the model is present; warn
  loudly if not; preload the model (`keep_alive`) so the first command has no
  cold start.
- VR panel and flat UI: on rejection show what was heard and why ("Didn't
  understand: 'make it look cool'"); when the LLM is down, say "LLM offline,
  rule parser only".

## Success criteria

1. `data/parser_eval_set.json` stays at 100 % for the rule parser.
2. Realistic held-out: at least 90 % exact accuracy for the full pipeline,
   and above both baselines (today's rule parser, today's LLM parser), which
   are measured before any lexicon work.
3. p90 voice-to-render ≤ 4 s in the headset.

## Out of scope

German-language commands; clinical vocabulary; new command types (colour
changes, clipping); fine-tuning a model.
