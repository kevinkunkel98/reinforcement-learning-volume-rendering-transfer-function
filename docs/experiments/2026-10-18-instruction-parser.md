# 2026-10-18: instruction parser on everyday phrasings

The policy is only as good as the command it is handed. This measures both
parsers -- the rule grammar (`commands.parse_command_rule`) and the local LLM
(`parse_command_llm`, qwen2.5:7b, temperature 0) -- on full commands, then
fixes what fails.

## Evaluation set (`data/parser_eval_set.json`, `eval_parsers.py`)

107 phrases with the full expected command (targets, attribute, direction,
strength, level), or null where the parser should refuse:

- **dev** (57): 36 real commands from the viewer's LLM log
  (`out/llm_requests.jsonl`, 522 requests, 174 distinct texts, 2026-09-03 to
  10-15) -- including speech-recognition noise ("thank you.", "bye.", "you")
  and retired classes ("show only fat.") -- plus the earlier 21-phrase
  free-form set.
- **heldout** (50): new phrasings across every instruction kind (relative,
  absolute, show only, compound, brightness, reset, camera, refusals),
  written before any parser change and only scored, never tuned against.

"exact" compares everything; "intent" ignores strength. Where a phrase does
not pin a strength down ("the lungs are barely visible"), several are
accepted.

## Results

| parser | dev exact | dev intent | heldout exact | heldout intent |
|---|---|---|---|---|
| rule, before | 40 % | 40 % | 18 % | 26 % |
| **rule, after** | **100 %** | **100 %** | **100 %** | **100 %** |
| LLM, before | 81 % | 88 % | 66 % | 74 % |
| **LLM, after** | **96 %** | **98 %** | **88 %** | **94 %** |

Caveat: the held-out phrases and the new rule parser were written by the
same author in the same session, so the held-out score is optimistic for a
hand-written grammar. Phrases from other people (the planned user check) are
the honest test.

## What was wrong

**LLM.** Most of its "failures" in real use were right answers the validator
threw away: "show only bones" fell back to the rule parser 33 times because
the model wrapped it in a one-element compound without a `strength` key.
Others: a bare `"set"` command without the compound wrapper, `"strength":
"extremely"`, `"attribute": null` on a relative change, and `slightly` as the
default strength. Worse, it turned speech noise into commands: "bye." and
"oh" became **reset**, "you" became "more skeleton".

**Rule parser.** No "add", "increase X", "opacity *of* X", "X opacity",
trailing "a lot" / "extremely", "revert", typos ("langs"), descriptions
("barely visible", "dominating the view"). And three silent misparses:
"darken bone extremely and decrease opacity of soft tissue strongly" was one
darken of soft tissue; "more bone and a little less soft tissue" lost the
bone; "show a lot less bones" became *show only* bones.

## Changes

- **Rule parser**: the sentence is split into clauses (commas, "and", "but");
  each clause tries the fixed grammar first (every grammar text parses as
  before -- checked on 574 grammar, reference, log and test texts: 0 newly
  refused, 17 changed, all corrections), then an everyday-words parser:
  a class (whole words, longest phrase first, fuzzy match for typos, "it" /
  "them" for the previous clause's class), plus a level ("set the spleen to
  high", "high spleen"), a brightness word, or an opacity verb or
  description. Modifier-only clauses ("…, extremely") join their neighbour; a
  clause that names nothing ("more bone, make the weather nicer") still makes
  the whole command fail rather than being dropped. "only" / "just" /
  "isolate" / "nothing but" mean show only; "revert" / "start over" /
  "default" mean reset; "tissue" means soft tissue, "pulmonary tissue" lungs.
- **LLM**: near-miss answers are repaired before validation (missing
  strength, null attribute, off-schema strength words, bare `set`). The
  prompt now defines the default strength and an explicit `{"none": true}`
  for non-instructions. A **grounding check** refuses any command whose target
  class is not named in the text, and any reset without a reset word; those,
  like every other refusal, fall back to the rule parser.

## Next

- With these numbers the rule parser is the better first choice: instant,
  deterministic, and at least as accurate on everything measured here; the
  LLM is most useful for phrasings the grammar has never seen. Trying the
  rule parser first and the LLM only when it refuses is a one-line change in
  `parse_command_with_meta`.
- Collect phrases from other people (a short user check) as an independent
  test set.
