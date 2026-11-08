"""The web app: the viewer's routes around one `session.Session`.

    .venv/bin/python server.py                       # http://127.0.0.1:8000
    .venv/bin/python server.py --host 0.0.0.0 --ssl-certfile certs/lan.pem --ssl-keyfile certs/lan-key.pem

Route handlers are `async def` so Starlette runs them on the event-loop
thread: on macOS VTK may only render on the main thread, and `uvicorn.run`
drives the loop from it. Only the warm search of a two-stage policy answer
runs in a worker thread (it does not render).
"""
import asyncio
import datetime
import os
import sys
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import answers
import collect
import commands
import goals
import lexicon
import scans
from asr import ASR_MODEL, _transcribe_path as asr_transcribe_path, is_hallucination
from camera import DEFAULT_CAMERA
from session import Session, scan_label, scan_list

AUDIO_DIR = "out/audio"
MIN_AUDIO_BYTES = 1000
# Two-stage policy answers: the fast stage is the command's response; the warm
# search then runs in a worker thread and the client long-polls /api/refined.
# POLICY_TWO_STAGE=0 answers in one synchronous stage.
TWO_STAGE = os.environ.get("POLICY_TWO_STAGE", "1") != "0"
REFINE_WAIT_SECONDS = 20.0
# SCANS=all lists every downloaded patient, not only the demo set.
SHOW_ALL_SCANS = os.environ.get("SCANS", "demo") == "all"

_session = None
_refine_events = {}   # step id -> asyncio.Event, set when its refine settles
_refine_tasks = set()  # strong references, so pending tasks are not collected


def start_scan() -> str:
    """--dataset, else UI_DATASET, else the first demo patient."""
    if "--dataset" in sys.argv[:-1]:
        return sys.argv[sys.argv.index("--dataset") + 1]
    if os.environ.get("UI_DATASET"):
        return os.environ["UI_DATASET"]
    available = scan_list(SHOW_ALL_SCANS)
    if not available:
        raise SystemExit("no scans downloaded -- run `make scans` (python -m tools.fetch_scans --demo --splits)")
    return available[0]


def session() -> Session:
    global _session
    if _session is None:
        _session = Session(start_scan())
    return _session


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(collect.router)


@app.middleware("http")
async def revalidate_static(request, call_next):
    """The headset browser kept stale scripts across reloads: `no-cache` makes
    every load revalidate (an unchanged file is still a 304)."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/state")
async def get_state():
    return session().state()


@app.post("/api/back")
async def back():
    return session().back()


@app.post("/api/forward")
async def forward():
    return session().forward()


@app.get("/api/commands")
async def commands_reference():
    return {"commands": commands.COMMAND_REFERENCE}


# --- scans ----------------------------------------------------------------------

@app.get("/api/datasets")
async def datasets_list():
    current = session().scan
    names = scan_list(SHOW_ALL_SCANS)
    if current not in names:
        names = [current] + names
    return {"available": names, "current": current, "labels": {n: scan_label(n) for n in names}}


def _scan_or_404(name: str):
    if name not in scans.available_names():
        raise HTTPException(status_code=404, detail=f"unknown scan {name!r}")


def _chunk_index(index: str) -> int:
    try:
        return int(index)
    except ValueError:
        raise HTTPException(status_code=422, detail="chunk index must be an integer")


@app.get("/api/datasets/{name}/metadata")
async def volume_metadata(name: str):
    _scan_or_404(name)
    return scans.volume_metadata(name)


@app.get("/api/datasets/{name}/chunks/{index}")
async def volume_chunk(name: str, index: str):
    _scan_or_404(name)
    try:
        chunk = scans.volume_chunk(name, _chunk_index(index))
    except IndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return Response(content=chunk, media_type="application/octet-stream",
                    headers={"X-Dataset-Version": scans.version(name)})


@app.get("/api/datasets/{name}/labels/metadata")
async def label_metadata(name: str):
    _scan_or_404(name)
    return scans.label_metadata(name)


@app.get("/api/datasets/{name}/labels/chunks/{index}")
async def label_chunk(name: str, index: str):
    _scan_or_404(name)
    try:
        chunk = scans.label_chunk(name, _chunk_index(index))
    except IndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return Response(content=chunk, media_type="application/octet-stream",
                    headers={"X-Dataset-Version": scans.version(name)})


class DatasetRequest(BaseModel):
    name: str


@app.post("/api/dataset")
async def dataset_switch(req: DatasetRequest):
    if req.name not in scans.available_names():
        raise HTTPException(status_code=400, detail=f"unknown scan {req.name!r}")
    return session().switch_dataset(req.name)


# --- commands ---------------------------------------------------------------------

class CommandRequest(BaseModel):
    text: str
    # Rule parser for fully covered sentences, the local LLM for the rest; the
    # LLM falls back to the rule parser when Ollama is down.
    parser: str = "hybrid"
    model: str = commands.DEFAULT_LLM_MODEL
    mode: str | None = None   # "exact" | "search" | "policy"
    search: bool = False      # older clients: search=True means mode="search"
    steps: int = Field(10, ge=1, le=500)


async def _finish_refine(sess, step_id, job, event):
    try:
        params = await asyncio.get_running_loop().run_in_executor(None, answers.full_answer, job)
        sess.apply_refined(step_id, params, job)   # back on the loop (main) thread: renders
    except Exception as exc:   # a failed refine must not leave the step "refining"
        print(f"[refine] step {step_id} failed: {exc!r}", file=sys.stderr)
        sess.discard_refine(step_id, job)
    finally:
        event.set()
        if _refine_events.get(step_id) is event:
            del _refine_events[step_id]


@app.post("/api/command")
async def command(req: CommandRequest):
    started = time.perf_counter()
    sess = session()
    mode = req.mode or ("search" if req.search else "exact")
    try:
        result = sess.command(req.text, req.parser, req.model, mode=mode, steps=req.steps, defer_refine=TWO_STAGE)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    current = result["current"]
    job = sess.pending_refines.get(current["id"]) if current.get("refining") else None
    if job is not None:
        event = asyncio.Event()
        _refine_events[current["id"]] = event
        task = asyncio.create_task(_finish_refine(sess, current["id"], job, event))
        _refine_tasks.add(task)
        task.add_done_callback(_refine_tasks.discard)
    result["timings"] = {"parse_ms": current.get("parse_ms"),
                         "total_ms": round((time.perf_counter() - started) * 1000, 1)}
    return result


@app.get("/api/refined/{step_id}")
async def refined(step_id: int):
    """Long-poll for a two-stage step's final answer: waits while its refine
    is pending, then returns the state and whether the cursor step is that
    step, refined."""
    event = _refine_events.get(step_id)
    if event is not None:
        try:
            await asyncio.wait_for(event.wait(), REFINE_WAIT_SECONDS)
        except asyncio.TimeoutError:
            pass
    state = session().state()
    return {**state, "refined": state["current"]["id"] == step_id and bool(state["current"].get("refined"))}


# --- comparison panel -------------------------------------------------------------

class CompareRequest(BaseModel):
    text: str
    parser: str = "hybrid"
    model: str = commands.DEFAULT_LLM_MODEL
    # Bounded: these handlers run on the event loop, a huge search would freeze the UI.
    cheap: int = Field(answers.COMPARE_CHEAP, ge=1, le=500)
    thorough: int = Field(answers.COMPARE_THOROUGH, ge=1, le=500)


@app.post("/api/compare")
async def compare(req: CompareRequest):
    """Answer one instruction several ways without advancing the session."""
    sess = session()
    try:
        cmd, parser_meta = commands.parse_command_with_meta(req.text, parser=req.parser, model=req.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    step = sess.history[sess.cursor]
    start_params = np.array(step["params"], dtype=np.float64)
    camera = dict(step.get("camera", DEFAULT_CAMERA))
    model = sess.model()
    try:
        start_agg, instruction = sess.instruction_for(cmd, model, start_params)
    except ValueError as exc:
        return {"applicable": False, "detail": str(exc), "text": req.text, **parser_meta,
                "reason": "this instruction is not a visibility goal, so there is nothing to score"}
    arms = answers.compare(model, sess.policy_provider(), cmd, instruction, start_agg, start_params,
                           lambda params: sess.view(params, camera), cheap=req.cheap, thorough=req.thorough)
    n = len(goals.GOAL_CLASSES)
    channels = {name: [c for i, c in enumerate(goals.GOAL_CLASSES) if instruction["goal"][offset + i]]
                for name, offset in (("vis", n), ("bright", 3 * n))}
    return {"applicable": True, "text": req.text, "goal_text": instruction["text"],
            "budgets": {"cheap": req.cheap, "thorough": req.thorough}, "arms": arms, "channels": channels,
            "start": sess.readout(start_params), **parser_meta}


class SweepRequest(BaseModel):
    episodes: int = Field(answers.SWEEP_EPISODES, ge=1, le=200)
    seed: int = 0
    thorough: int | None = Field(None, ge=1, le=500)


@app.post("/api/compare/sweep")
async def sweep(req: SweepRequest):
    """What the arms do across many instructions, not just the one typed."""
    sess = session()
    start_params = np.array(sess.history[sess.cursor]["params"], dtype=np.float64)
    return answers.sweep(sess.model(), sess.policy_provider(), sess.scan, start_params,
                         episodes=req.episodes, seed=req.seed, thorough=req.thorough)


# --- speech -----------------------------------------------------------------------

try:
    import av
    # faster-whisper decodes with PyAV; mlx-whisper runs ffmpeg and raises RuntimeError.
    _AUDIO_DECODE_ERRORS = (av.error.FFmpegError, RuntimeError)
except ImportError:
    _AUDIO_DECODE_ERRORS = (RuntimeError,)


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wav_path = os.path.join(AUDIO_DIR, f"{stamp}.wav")
    data = await audio.read()
    with open(wav_path, "wb") as stream:
        stream.write(data)
    started = time.time()
    # A trigger tap uploads a near-empty clip the decoders fail on; the client
    # shows "No speech recognised" for an empty text.
    if len(data) < MIN_AUDIO_BYTES:
        return {"text": "", "heard": "", "duration": 0.0}
    try:
        text = asr_transcribe_path(wav_path, "en", ASR_MODEL, initial_prompt=lexicon.get()["asr_prompt"])
    except _AUDIO_DECODE_ERRORS as exc:
        print(f"[asr] could not decode {wav_path}: {type(exc).__name__}: {exc}")
        return {"text": "", "heard": "", "duration": time.time() - started}
    duration = time.time() - started
    with open(os.path.join(AUDIO_DIR, f"{stamp}.txt"), "w", encoding="utf-8") as stream:
        stream.write(text)
    if is_hallucination(text):
        return {"text": "", "heard": text, "duration": duration}
    return {"text": text, "duration": duration}


def main():
    import argparse
    import threading

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="scan to open, e.g. ts_s0591 (default: the first demo patient)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from a headset on the LAN")
    # Mic capture and WebXR need a secure context, which off-localhost means HTTPS.
    parser.add_argument("--ssl-certfile", help="TLS certificate (e.g. from mkcert)")
    parser.add_argument("--ssl-keyfile", help="TLS private key matching --ssl-certfile")
    args = parser.parse_args()
    print(f"[server] scan: {session().scan}")

    def warm_llm():
        problem = commands.check_llm()
        if problem:
            print(f"\n[llm-parser] WARNING: {problem}\n[llm-parser] voice commands use the rule parser only\n")
            return
        commands.warm_llm()
        print(f"[llm-parser] {commands.DEFAULT_LLM_MODEL} loaded")

    threading.Thread(target=warm_llm, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)


if __name__ == "__main__":
    main()
