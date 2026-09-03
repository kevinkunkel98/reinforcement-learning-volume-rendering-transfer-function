# Chat/Voice UI with Step Navigation — Design

Adds a local web UI on top of the existing CLI prototype: a chat panel (text + mic,
via the existing Whisper wrapper) driving the transfer function, a rendered-image
view, and back/forward navigation through the session's history — replacing manual
terminal invocations for interactive testing. `mvp.py` and the pipeline modules
(`phantom.py`, `transfer.py`, `render.py`, `commands.py`, `evaluate.py`, `asr.py`)
are unchanged and reused as-is; this is a new orchestrator alongside `mvp.py`, not a
replacement.

## Files

`server.py` (FastAPI + uvicorn), `static/index.html`, `static/app.js`,
`static/style.css`.

## Backend (`server.py`)

In-memory session: a list of **steps** plus a cursor.

```
Step = {
  id: int, timestamp: str,
  cmd_text: str | None, cmd_dict: dict | None,
  params: list[float] (24),
  image_b64: str (PNG),
  masses: {tissue: float}, features: {...},
  verdict: 1 | -1 | None,
  search: bool,
}
```

Step 0 is always `default_params()` with `cmd_text=None`. The phantom is built once
at startup and reused (module-level cache, same pattern as `mvp.py`'s
`get_phantom()`). After every mutation, `{history, cursor}` is written to
`out/ui_session.json`; on startup, loaded if present, so a page reload or server
restart resumes the same session instead of losing it.

### Endpoints

- `GET /` — serves `static/index.html`.
- `GET /api/state` — `{cursor, total, current: <Step>, pending: <PendingJudgment | null>}`.
- `POST /api/command` `{text, parser, model, search, evaluator, steps}`:
  - Parses `text` via `commands.parse_command` (existing rule/LLM parsers, unchanged).
  - Parse failure (`ValueError`) → `400` with the error message, history untouched.
  - `search=false` → one `apply_command` call, rendered, appended as one new step.
  - `search=true, evaluator="objective"` → runs the existing hill-climbing loop
    (extracted from `mvp.py`'s `hill_climb` into a shared helper both `mvp.py` and
    `server.py` import, so the search logic isn't duplicated) synchronously
    server-side; only the **final** converged params become one new history step
    (intermediate iterations still go to `out/log.jsonl` as before, just not to the
    UI timeline — the timeline tracks chat turns, not search internals).
  - `search=true, evaluator="human"` → does **not** append a step yet. Computes the
    first proposed params, renders before/after, stores them as the session's
    pending judgment (current params, proposed params, step size, sign, cmd,
    cmd_text, session_id), and returns them for the client to judge.
  - Any new command truncates `history` to `cursor+1` before appending (going back
    and sending a new message discards the abandoned branch — browser-history
    semantics, not a tree).
- `POST /api/judge` `{verdict: "better"|"worse"}` — only valid while a pending
  judgment exists. Applies the same accept/revert/step-resize rule as `mvp.py`'s
  hill-climbing, logs the step to `out/log.jsonl` **and** the pair to
  `out/preferences.jsonl` (both human and objective verdict for that pair, exactly
  as today's `--learn --human` path does). If converged (step count or step-size
  threshold reached): finalizes, appends one new history step, clears the pending
  judgment. Otherwise: renders the next proposed pair and returns it as the new
  pending judgment.
- `POST /api/back`, `POST /api/forward` — move the cursor by one (clamped to
  `[0, len(history)-1]`), return the resulting `/api/state`.
- `POST /api/transcribe` — multipart WAV upload from the browser's
  `MediaRecorder`. Saved to `out/audio/<timestamp>.wav` + `.txt` exactly like
  `asr.transcribe_mic` does today (reusing `asr._save_wav`/`asr._transcribe_path`
  rather than the mic-capture half of `asr.py`, since the browser already captured
  the audio). Returns `{text, duration}`.

### Refactor note

`mvp.py`'s `hill_climb` function currently mixes the search loop with PNG-writing
side effects and console printing. It gets split so `server.py` can reuse the pure
search step (given current params + cmd + step size + sign, propose next params)
without duplicating the accept/revert/resize math — `mvp.py`'s CLI behavior is
unchanged after the split.

## Frontend (`static/`)

Plain HTML/CSS/JS, no build step, no framework — one `<script>` tag, `fetch()` for
all API calls. Layout:

- **Image panel** (top/left, large): current step's rendered PNG, a step counter
  ("3 / 7"), Back/Forward buttons (disabled at the ends of history).
- **Chat panel** (bottom/right): scrollable message list (each entry: command text,
  a small thumbnail of the resulting image, verdict badge if any), a text input +
  Send button, a mic button (🎤, toggles `MediaRecorder` start/stop, POSTs the blob
  to `/api/transcribe`, then auto-sends the returned text as a chat message — per
  your choice, no edit step).
- **Controls row**: parser dropdown (rule/llm, default rule), a "search" toggle
  (off = single apply, on = hill-climb) with an evaluator dropdown
  (objective/human, only shown when search is on) and a steps number input.
- **Pending-judgment overlay**: when `/api/state` reports a `pending` judgment (on
  load or after `/api/command`), the image panel swaps to a side-by-side
  before/after view with **Better** / **Worse** buttons in place of the normal
  single-image view; clicking one POSTs `/api/judge` and re-renders from the
  response (either the next pending pair, or back to the normal single-image view
  once converged).

On page load: `GET /api/state` restores whatever session is already on disk (or
creates a fresh one with just step 0) — refreshing the browser never loses history.

## VR note (not built now)

The frontend only talks to a JSON API, and rendering happens server-side and is
handed over as a flat image. A later VR client (WebXR page, or a native app) can
call the same `/api/command` / `/api/state` / `/api/judge` endpoints unchanged —
only how the image is displayed differs client-side. No VR-specific code is added
in this change.

## Dependencies

Add to `requirements.txt`: `fastapi`, `uvicorn`, `python-multipart` (needed for
the WAV upload endpoint).
