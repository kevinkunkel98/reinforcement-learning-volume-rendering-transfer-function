# Ingest Chat-UI Feedback into RLHF Preference Data — Design

## Purpose

`rl/collect_preferences.py` (already built) requires a dedicated interactive
session where a human judges 50 freshly-rendered before/after pairs.
`server.py`'s chat UI already collects a similar signal passively — every
applied command is logged with rendered image features, and every
👍/👎 rating on a step is logged separately — as a byproduct of ordinary use.
This adds `rl/ingest_feedback.py`, which builds `out/rlhf_preferences.jsonl`
(the same file, same schema, `rl/reward_model.py` already trains on) from
that already-logged data instead of running a new labeling session. This
scales with ordinary app usage (including by other users of the app, not
just one dedicated rater) rather than requiring a standalone sitting.

`rl/reward_model.py`, `rl/reward_model_env.py`, and `rl/reward_model_train.py`
need **zero changes** — this only adds a second way to produce their input
file.

## Grounding in existing code

Verified directly against the real logs produced by using the chat UI
(`out/feedback.jsonl`, 16 rows; `out/log.jsonl`, 15 rows), not assumed:

- `server.py`'s `_log_command` (`server.py:244-250`) writes, for **every**
  applied command: `command` (the parsed cmd dict), `params_before`,
  `params_after`, and — critically — `features_before`/`features_after`,
  which are `render.features(img)` dicts, i.e. exactly
  `{"mean", "std", "coverage", "entropy"}`, the same shape
  `rl/reward_model.py`'s `featurize()` already expects.
- `Session.feedback()` (`server.py:195-203`) writes, for a rated step:
  `cmd_dict`, `params` (the step's resulting/"after" params), and
  `rating` (`"up"` or `"down"` — confirmed, only these two values appear).
- **The join key**: `feedback.jsonl`'s `params` and the originating
  `log.jsonl` row's `params_after` are the same array serialized
  (`.tolist()`) from the same in-memory step twice, so exact list equality
  reliably identifies the matching log row. Verified: of 16 real feedback
  rows, 15 matched a `log.jsonl` row on `params_after == params` (1 was a
  camera command, never logged via `_log_command` the same way — moot,
  since it's filtered out below anyway); 0 false negatives, 0 ambiguous
  matches.
- **Filter**: of those 16 rows, 15 have `cmd_dict.attribute == "opacity"`,
  `cmd_dict.direction in ("increase", "decrease")`, and a single string
  `target` — exactly the shape `TFEnv`'s MDP models (@`rl-write-test.typ`
  §2). The 1 exception was the camera command. `show_only`, compound,
  width/brightness/center commands would also be filtered if present (none
  were, in this real data) — the reward model's input space has no slot for
  them.
- `evaluate.jsonl_append` is NOT reused here — that function only appends;
  this script needs to overwrite (see below), so it writes the file directly.

## New module: `rl/ingest_feedback.py`

```python
FEEDBACK_PATH = "out/feedback.jsonl"
LOG_PATH = "out/log.jsonl"
OUT_PATH = "out/rlhf_preferences.jsonl"   # same constant value as
                                            # rl.collect_preferences.PREFERENCES_PATH
RATING_TO_LABEL = {"up": 1, "down": -1}
```

`ingest_feedback(feedback_path=FEEDBACK_PATH, log_path=LOG_PATH, out_path=OUT_PATH) -> int`:

1. Read `log_path`, build `by_params_after = {tuple(row["params_after"]): row for row in ...}`.
2. Read `feedback_path` line by line. For each row:
   - Skip if `rating` not in `RATING_TO_LABEL` (defensive — only `up`/`down`
     are ever written today, but this is external log data, not a value this
     script controls).
   - Skip if `cmd_dict` doesn't have `attribute == "opacity"`,
     `direction in ("increase", "decrease")`, and a string `target`.
   - Skip if `tuple(row["params"])` isn't in `by_params_after` (no
     originating log row found — external data may legitimately have gaps,
     e.g. a truncated log file).
   - Otherwise emit `{"timestamp": row["timestamp"], "rater_id":
     row.get("session_id", "unknown"), "target_tissue": cmd["target"],
     "direction": cmd["direction"], "delta": null, "before_features":
     log_row["features_before"], "after_features": log_row["features_after"],
     "label": RATING_TO_LABEL[row["rating"]]}` — identical field set to what
     `rl/collect_preferences.py` already writes, `delta` aside (not
     recoverable from logged data, and not read by `featurize()` either way).
3. **Rebuild from scratch**: open `out_path` in `"w"` mode (not append) and
   write every emitted row — a full rebuild each run, matching the explicit
   choice made for this task. Re-running never duplicates rows; it always
   reflects the current state of `feedback.jsonl`/`log.jsonl`.
4. Print a one-line honest summary — `n` written vs. total feedback rows
   seen, same "report the real number" ethos as the rest of this RL work —
   and return `n`.

`main()`: `argparse` with `--feedback`, `--log`, `--out` overrides, default
to the constants above.

## Data flow

```
(ordinary chat-UI usage) → out/feedback.jsonl, out/log.jsonl
python -m rl.ingest_feedback → out/rlhf_preferences.jsonl  (full rebuild)
python -m rl.reward_model --preferences out/rlhf_preferences.jsonl → out/rl_models/reward_model.pt
python -m rl.reward_model_train → ... (unchanged from the existing plan)
```

## Testing

`tests/test_rl_ingest_feedback.py` — fast, synthetic `feedback.jsonl`/
`log.jsonl` fixtures in `tmp_path`, no real rendering:

- A matched, opacity-shaped, `up`-rated row → written with `label: 1` and
  the correct `before_features`/`after_features` pulled from the matching
  log row.
- A `down`-rated row → `label: -1`.
- A camera-command row (no `attribute`/single-tissue `target`) → filtered
  out.
- A feedback row whose `params` matches no log row → filtered out (no
  crash).
- Calling `ingest_feedback()` twice in a row → the output file has the same
  row count both times (rebuild, not append/duplicate).

## New dependencies

None.

## Explicitly out of scope

- Changes to `rl/reward_model.py`, `rl/reward_model_env.py`,
  `rl/reward_model_train.py`, or `rl/collect_preferences.py` — none needed.
- Merging/deduplicating `collect_preferences.py`-produced rows with
  ingested ones in the same run — not built; if both are used, the later
  one to run simply determines what's in `out/rlhf_preferences.jsonl`
  (`collect_preferences.py` appends, `ingest_feedback.py` overwrites).
- Any attempt to recover a reward-model-usable signal from `show_only`,
  compound, width/brightness/center, or camera commands.
