"""FastAPI backend for the chat/voice UI.

Route handlers are thin `async def` wrappers around `Session` — kept async (not
plain `def`) so Starlette runs them directly on the event-loop thread instead of
dispatching to a worker-thread pool. On macOS, VTK's Cocoa render window can only
be created on the true main thread; since `uvicorn.run()` in `__main__` drives the
event loop from the main thread, async handlers keep every render() call there too.
`Session` itself has no FastAPI/threading dependency, so it's also directly unit
testable in-process without going anywhere near that constraint.
"""
import base64
import datetime
import io
import json
import os
import sys
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from asr import _transcribe_path as asr_transcribe_path
from commands import COMMAND_REFERENCE, STRENGTH_WORDS, _find_or_create_peak, apply_command, parse_command
from datasets import list_datasets, load_dataset
from evaluate import jsonl_append, objective
import render as render_module
from render import features, grab, render
from search import propose_step, resize_step
from transfer import TISSUE_BANDS, default_params, opacity_mass

LOG_PATH = "out/log.jsonl"
PREF_PATH = "out/preferences.jsonl"
FEEDBACK_PATH = "out/feedback.jsonl"
AUDIO_DIR = "out/audio"


def _resolve_dataset_name():
    # Read --dataset from argv directly (not argparse) so this resolves before
    # any module-level code below -- notably `session = Session(...)` -- runs
    # and potentially triggers the first render with the wrong dataset.
    if "--dataset" in sys.argv:
        idx = sys.argv.index("--dataset")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return os.environ.get("UI_DATASET", "synthetic")


_dataset_name = _resolve_dataset_name()
_volume = None
_spacing = None


def get_volume():
    global _volume, _spacing
    if _volume is None:
        _volume, _spacing = load_dataset(_dataset_name)
    return _volume, _spacing


def set_dataset(name: str):
    """Switch the active dataset. Raises ValueError for an unknown name."""
    global _dataset_name, _volume, _spacing
    volume, spacing = load_dataset(name)  # validates name, raises ValueError if unknown
    _dataset_name = name
    _volume, _spacing = volume, spacing


def _masses(params):
    return {t: opacity_mass(params, lo, hi) for t, (lo, hi) in TISSUE_BANDS.items()}


def _render_image_b64(params):
    volume, spacing = get_volume()
    img = grab(render(volume, params, spacing=spacing))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), img, buf.getvalue()


IMAGE_DIR = "out/ui_images"


def _save_image_file(session_id, name, png_bytes):
    # Namespaced by session_id: step ids restart from 0 after switch_dataset,
    # so a bare <name>.png would silently overwrite an earlier session's
    # image with the same name.
    session_dir = os.path.join(IMAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, f"{name}.png")
    with open(path, "wb") as f:
        f.write(png_bytes)
    return path


def _render_step(params, cmd_text, cmd_dict, verdict, search, step_id, session_id):
    image_b64, img, png_bytes = _render_image_b64(params)
    image_path = _save_image_file(session_id, f"step_{step_id}", png_bytes)
    return {
        "id": step_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "cmd_text": cmd_text,
        "cmd_dict": cmd_dict,
        "params": params.tolist(),
        "image_b64": image_b64,
        "image_path": image_path,
        "masses": _masses(params),
        "features": features(img),
        "verdict": verdict,
        "search": search,
        "feedback": None,
    }


def _pending_public(p):
    if p is None:
        return None
    return {
        "cmd_text": p["cmd_text"],
        "cmd_dict": p["cmd"],
        "iteration": p["iteration"],
        "max_steps": p["max_steps"],
        "before_image_b64": p["before_image_b64"],
        "after_image_b64": p["after_image_b64"],
    }


class Session:
    """All command/history/pending-judgment logic, independent of FastAPI."""

    def __init__(self, path: str):
        self.path = path
        self.history = []
        self.cursor = 0
        self.pending = None
        self.session_id = None
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.history, self.cursor = data["history"], data["cursor"]
            self.session_id = data.get("session_id") or self._new_session_id()
            return
        self.session_id = self._new_session_id()
        step = _render_step(default_params(), None, None, None, False, 0, self.session_id)
        self.history = [step]
        self.cursor = 0
        self.save()

    @staticmethod
    def _new_session_id():
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"history": self.history, "cursor": self.cursor, "session_id": self.session_id}, f)

    def state(self):
        volume, spacing = get_volume()
        return {
            "cursor": self.cursor,
            "total": len(self.history),
            "current": self.history[self.cursor],
            "pending": _pending_public(self.pending),
            "dataset": _dataset_name,
            "render_info": {
                "width": render_module.WIDTH,
                "height": render_module.HEIGHT,
                "mapper": render_module.MAPPER_NAME,
                "volume_shape": list(volume.shape),
                "spacing": list(spacing),
            },
        }

    def switch_dataset(self, name: str):
        set_dataset(name)  # raises ValueError for an unknown name
        self.session_id = self._new_session_id()
        step = _render_step(default_params(), None, None, None, False, 0, self.session_id)
        self.history = [step]
        self.cursor = 0
        self.pending = None
        self.save()
        return self.state()

    def feedback(self, step_id: int, rating: str):
        if rating not in ("up", "down"):
            raise ValueError(f"invalid rating: {rating!r}")
        step = next((s for s in self.history if s["id"] == step_id), None)
        if step is None:
            raise ValueError(f"no step with id {step_id}")
        step["feedback"] = rating
        jsonl_append(FEEDBACK_PATH, {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.session_id,
            "step_id": step_id,
            "cmd_text": step["cmd_text"],
            "cmd_dict": step["cmd_dict"],
            "params": step["params"],
            "rating": rating,
        })
        self.save()
        return self.state()

    def back(self):
        self.cursor = max(0, self.cursor - 1)
        self.save()
        return self.state()

    def forward(self):
        self.cursor = min(len(self.history) - 1, self.cursor + 1)
        self.save()
        return self.state()

    def _next_pending_pair(self, cmd_text, cmd, current, idx, sign, step_size, iteration, max_steps, session_id):
        proposed = propose_step(current, idx, sign, step_size)
        before_b64, _, before_bytes = _render_image_b64(current)
        after_b64, _, after_bytes = _render_image_b64(proposed)
        before_png = _save_image_file(session_id, f"judge{iteration}_before", before_bytes)
        after_png = _save_image_file(session_id, f"judge{iteration}_after", after_bytes)
        return {
            "cmd_text": cmd_text, "cmd": cmd, "current": current, "proposed": proposed,
            "idx": idx, "sign": sign, "step_size": step_size, "iteration": iteration,
            "max_steps": max_steps, "session_id": session_id,
            "before_image_b64": before_b64, "after_image_b64": after_b64,
            "before_png": before_png, "after_png": after_png,
        }

    def _start_human_search(self, cmd_text, cmd, params, steps):
        sign = 1.0 if cmd["direction"] == "increase" else -1.0
        step_size = STRENGTH_WORDS[cmd["strength"] or "moderately"]
        _, idx = _find_or_create_peak(params, cmd["target"])
        return self._next_pending_pair(cmd_text, cmd, params, idx, sign, step_size, 0, steps, self.session_id)

    def _run_objective_search(self, cmd, params, steps):
        sign = 1.0 if cmd["direction"] == "increase" else -1.0
        step = STRENGTH_WORDS[cmd["strength"] or "moderately"]
        current = params.copy()
        _, idx = _find_or_create_peak(current, cmd["target"])
        for _ in range(steps):
            proposed = propose_step(current, idx, sign, step)
            verdict = objective(current, proposed, cmd)
            jsonl_append(LOG_PATH, {
                "timestamp": datetime.datetime.now().isoformat(), "command": cmd,
                "params_before": current.tolist(), "params_after": proposed.tolist(),
                "features_before": None, "features_after": None, "verdict": verdict,
            })
            if verdict == 1:
                current = proposed
            step = resize_step(step, accepted=(verdict == 1))
            if step < 0.01:
                break
        return current

    def command(self, text, parser="rule", model="qwen2.5:7b", search=False, evaluator="objective", steps=10):
        cmd = parse_command(text, parser=parser, model=model)  # raises ValueError on failure

        current_params = np.array(self.history[self.cursor]["params"], dtype=np.float64)

        if search and cmd.get("attribute") == "opacity" and cmd.get("direction") in ("increase", "decrease"):
            if evaluator == "human":
                self.pending = self._start_human_search(text, cmd, current_params, steps)
                return self.state()
            new_params = self._run_objective_search(cmd, current_params, steps)
            step = _render_step(new_params, text, cmd, None, True,
                                 self.history[-1]["id"] + 1, self.session_id)
        else:
            new_params = apply_command(cmd, current_params)
            step = _render_step(new_params, text, cmd, None, False,
                                 self.history[-1]["id"] + 1, self.session_id)
            jsonl_append(LOG_PATH, {
                "timestamp": step["timestamp"], "command": cmd,
                "params_before": current_params.tolist(), "params_after": new_params.tolist(),
                "features_before": self.history[self.cursor]["features"], "features_after": step["features"],
                "verdict": None,
            })

        self.history = self.history[:self.cursor + 1] + [step]
        self.cursor = len(self.history) - 1
        self.save()
        return self.state()

    def judge(self, verdict: str):
        if self.pending is None:
            raise ValueError("no pending judgment")

        p = self.pending
        human_verdict = 1 if verdict == "better" else -1
        obj_verdict = objective(p["current"], p["proposed"], p["cmd"])

        jsonl_append(LOG_PATH, {
            "timestamp": datetime.datetime.now().isoformat(), "command": p["cmd"],
            "params_before": p["current"].tolist(), "params_after": p["proposed"].tolist(),
            "features_before": None, "features_after": None, "verdict": human_verdict,
        })
        jsonl_append(PREF_PATH, {
            "timestamp": datetime.datetime.now().isoformat(), "session_id": p["session_id"],
            "cmd_text": p["cmd_text"], "cmd_dict": p["cmd"],
            "params_before": p["current"].tolist(), "params_after": p["proposed"].tolist(),
            "features_before": None, "features_after": None,
            "before_png": p["before_png"], "after_png": p["after_png"],
            "human_verdict": verdict,
            "objective_verdict": "better" if obj_verdict == 1 else "worse",
        })

        current = p["proposed"] if human_verdict == 1 else p["current"]
        step_size = resize_step(p["step_size"], accepted=(human_verdict == 1))
        iteration = p["iteration"] + 1

        if iteration >= p["max_steps"] or step_size < 0.01:
            step = _render_step(current, p["cmd_text"], p["cmd"], human_verdict, True,
                                 self.history[-1]["id"] + 1, self.session_id)
            self.history = self.history[:self.cursor + 1] + [step]
            self.cursor = len(self.history) - 1
            self.pending = None
            self.save()
            return self.state()

        self.pending = self._next_pending_pair(p["cmd_text"], p["cmd"], current, p["idx"], p["sign"],
                                                step_size, iteration, p["max_steps"], p["session_id"])
        return self.state()


UI_SESSION_PATH = os.environ.get("UI_SESSION_PATH", "out/ui_session.json")
session = Session(UI_SESSION_PATH)

app = FastAPI()
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/state")
async def get_state():
    return session.state()


@app.post("/api/back")
async def back():
    return session.back()


@app.post("/api/forward")
async def forward():
    return session.forward()


@app.get("/api/datasets")
async def datasets_list():
    return {"available": list_datasets(), "current": _dataset_name}


@app.get("/api/commands")
async def commands_reference():
    return {"commands": COMMAND_REFERENCE}


class DatasetRequest(BaseModel):
    name: str


@app.post("/api/dataset")
async def dataset_switch(req: DatasetRequest):
    try:
        return session.switch_dataset(req.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class FeedbackRequest(BaseModel):
    step_id: int
    rating: str  # "up" | "down"


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    try:
        return session.feedback(req.step_id, req.rating)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class CommandRequest(BaseModel):
    text: str
    parser: str = "rule"
    model: str = "qwen2.5:7b"
    search: bool = False
    evaluator: str = "objective"
    steps: int = 10


@app.post("/api/command")
async def command(req: CommandRequest):
    try:
        return session.command(req.text, req.parser, req.model, req.search, req.evaluator, req.steps)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class JudgeRequest(BaseModel):
    verdict: str  # "better" | "worse"


@app.post("/api/judge")
async def judge(req: JudgeRequest):
    try:
        return session.judge(req.verdict)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wav_path = os.path.join(AUDIO_DIR, f"{stamp}.wav")
    with open(wav_path, "wb") as f:
        f.write(await audio.read())

    start = time.time()
    text = asr_transcribe_path(wav_path, "en", "small")
    duration = time.time() - start

    with open(os.path.join(AUDIO_DIR, f"{stamp}.txt"), "w") as f:
        f.write(text)
    return {"text": text, "duration": duration}


if __name__ == "__main__":
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=_dataset_name,
                     help="synthetic, or a real CT dataset name (see datasets.DATASETS) "
                          "-- already applied above; listed here only for --help")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"[server] dataset: {_dataset_name}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
