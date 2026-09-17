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
import hashlib
import io
import json
import math
import os
import sys
import threading
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from asr import _transcribe_path as asr_transcribe_path
from camera import DEFAULT_CAMERA, apply_camera_command
import collect
from commands import COMMAND_REFERENCE, STRENGTH_WORDS, _find_or_create_peak, apply_command, parse_command
from datasets import _dataset_version, dataset_metadata, default_camera_for, get_volume_chunk, list_datasets, load_dataset
from evaluate import jsonl_append, objective
import goals
import render as render_module
from render import features, grab, render
from rl.baselines import CONTROLLABLE, apply_controllable
from rl.oneshot_env import build_observation
from search import propose_step, resize_step
from scene_schema import normalize_scene, scene_transition as normalize_scene_transition
from transfer import TISSUE_BANDS, default_params, opacity_mass
import visibility

LOG_PATH = "out/log.jsonl"
SCENE_TRANSITIONS_PATH = "out/scene_transitions.jsonl"
_SCENE_WRITE_LOCK = threading.Lock()
AUDIO_DIR = "out/audio"

# --- Task 3: the one-shot policy, for mode="policy" --------------------------
# Loaded lazily and cached, the same pattern collect.py uses for the same
# checkpoint: importing this module (and starting the server) must not
# require a finished training run, and every command after the first pays no
# reload cost.
POLICY_PATH = "out/rl_v2/oneshot_v2_seed0/best.zip"
_policy_state = {"loaded": False, "policy": None}


def _load_policy():
    if not _policy_state["loaded"]:
        _policy_state["loaded"] = True
        if os.path.exists(POLICY_PATH):
            from stable_baselines3 import SAC
            _policy_state["policy"] = SAC.load(POLICY_PATH)
    return _policy_state["policy"]


def _predict_action(policy, observation: np.ndarray) -> np.ndarray:
    """One action from `policy`, an SB3-style model (`.predict(observation,
    deterministic=...) -> (action, state)`) or a plain callable -- mirrors
    `rl.candidates._predict`, deterministic here since the chat UI wants one
    consistent answer, not the sampling diversity preference collection
    wants."""
    if hasattr(policy, "predict"):
        action, _ = policy.predict(observation, deterministic=True)
    else:
        action = policy(observation)
    return np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)


def _resolve_dataset_name():
    # Read --dataset from argv directly (not argparse) so this resolves before
    # any module-level code below -- notably `session = Session(...)` -- runs
    # and potentially triggers the first render with the wrong dataset.
    if "--dataset" in sys.argv:
        idx = sys.argv.index("--dataset")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    # ct_chest, not mri_head: MRI has no calibrated Hounsfield scale (see
    # datasets.py's module docstring) and is excluded from RL v2 for exactly
    # that reason, so the chat UI's default should be a calibrated CT where
    # every mode -- including policy -- works.
    return os.environ.get("UI_DATASET", "ct_chest")


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


def _render_image_b64(params, camera):
    volume, spacing = get_volume()
    img = grab(render(volume, params, spacing=spacing, camera=camera))
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


def _scene_event_id(before, after, event_id=None):
    if event_id:
        return event_id
    payload = json.dumps({"before": before, "after": after}, sort_keys=True, separators=(",", ":"))
    return "scene-event:" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def _render_step(params, cmd_text, cmd_dict, search, step_id, session_id, camera,
                  mode="exact", message=None):
    image_b64, img, png_bytes = _render_image_b64(params, camera)
    image_path = _save_image_file(session_id, f"step_{step_id}", png_bytes)
    return {
        "id": step_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "cmd_text": cmd_text,
        "cmd_dict": cmd_dict,
        "params": params.tolist(),
        "camera": camera,
        "image_b64": image_b64,
        "image_path": image_path,
        "masses": _masses(params),
        "features": features(img),
        "search": search,
        "mode": mode,
        "message": message,
    }


class Session:
    """All command/history logic, independent of FastAPI."""

    def __init__(self, path: str, policy_provider=None, model_for_volume=visibility.for_volume):
        self.path = path
        self.history = []
        self.cursor = 0
        self.session_id = None
        # Injectable, like collect.Collector's seams: policy_provider defaults
        # to the lazily-loaded/cached checkpoint above, model_for_volume to
        # the real visibility model -- tests substitute cheap stand-ins for
        # both without touching VTK or a real checkpoint.
        self.policy_provider = policy_provider or _load_policy
        self.model_for_volume = model_for_volume
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.history, self.cursor = data["history"], data["cursor"]
            self.session_id = data.get("session_id") or self._new_session_id()
            return
        self.session_id = self._new_session_id()
        step = _render_step(default_params(), None, None, False, 0, self.session_id, default_camera_for(_dataset_name))
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
            "dataset": _dataset_name,
            "session_id": self.session_id,
            "render_info": {
                "width": render_module.WIDTH,
                "height": render_module.HEIGHT,
                "mapper": render_module.MAPPER_NAME,
                "volume_shape": list(volume.shape),
                "spacing": list(spacing),
                "dataset_version": _dataset_version(_dataset_name),
            },
        }

    def switch_dataset(self, name: str):
        set_dataset(name)  # raises ValueError for an unknown name
        self.session_id = self._new_session_id()
        step = _render_step(default_params(), None, None, False, 0, self.session_id, default_camera_for(name))
        self.history = [step]
        self.cursor = 0
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

    def _log_command(self, cmd, current_params, new_params, step):
        jsonl_append(LOG_PATH, {
            "timestamp": step["timestamp"], "command": cmd,
            "params_before": current_params.tolist(), "params_after": new_params.tolist(),
            "features_before": self.history[self.cursor]["features"], "features_after": step["features"],
            "verdict": None,
        })

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

    def _run_policy(self, cmd, current_params):
        """mode="policy": build the goal `cmd` asks for (`goals.
        goal_from_command`) and run the cached one-shot policy on it, the
        same observation layout `rl.candidates` builds for a standalone
        policy query (`rl.oneshot_env.build_observation`). Returns
        `(new_params, goal_text)`. Raises `ValueError` -- caught by
        `command()`, which falls back to exact application -- when no
        checkpoint is loaded, or `goal_from_command`/the volume's model
        raises for a non-goal command (camera, reset, width, centre) or a
        goal this volume can't support."""
        policy = self.policy_provider()
        if policy is None:
            raise ValueError(
                "no trained policy checkpoint found -- applied the command directly instead")

        model = self.model_for_volume(_dataset_name)
        start_agg = goals.aggregate(model.features(current_params))
        goal = goals.goal_from_command(cmd, model, start_agg, volume=_dataset_name)

        solo_max_log = [math.log10(sum(model.solo_max(m) for m in goals.MEASURED_FOR_GOAL[c]) + goals.EPSILON)
                         for c in goals.GOAL_CLASSES]
        controllable = [float(np.mean([current_params[i] for i in group])) for group in CONTROLLABLE]
        observation = build_observation(goal["goal"], model.histogram, start_agg, solo_max_log, controllable)

        action = _predict_action(policy, observation)
        new_params = apply_controllable(current_params, action)
        return new_params, goal["text"]

    def command(self, text, parser="rule", model="qwen2.5:7b", search=False, steps=10, mode=None):
        cmd = parse_command(text, parser=parser, model=model)  # raises ValueError on failure

        current_params = np.array(self.history[self.cursor]["params"], dtype=np.float64)
        current_camera = dict(self.history[self.cursor].get("camera", DEFAULT_CAMERA))

        if "camera" in cmd:
            new_camera = apply_camera_command(cmd["camera"], current_camera)
            step = _render_step(current_params, text, cmd, False,
                                 self.history[-1]["id"] + 1, self.session_id, new_camera,
                                 mode="camera")
            self.history = self.history[:self.cursor + 1] + [step]
            self.cursor = len(self.history) - 1
            self.save()
            return self.state()

        # mode wins over the legacy `search` bool when given; `search=True`
        # alone still means mode="search", so existing callers (and the UI's
        # search toggle) keep working unchanged.
        effective_mode = mode or ("search" if search else "exact")

        message = None
        if effective_mode == "policy":
            try:
                new_params, _goal_text = self._run_policy(cmd, current_params)
                actual_mode, search_flag = "policy", False
            except ValueError as exc:
                message = str(exc)
                new_params = apply_command(cmd, current_params)
                actual_mode, search_flag = "exact", False
        elif effective_mode == "search" and cmd.get("attribute") == "opacity" and cmd.get("direction") in ("increase", "decrease"):
            new_params = self._run_objective_search(cmd, current_params, steps)
            actual_mode, search_flag = "search", True
        else:
            new_params = apply_command(cmd, current_params)
            actual_mode, search_flag = "exact", False

        step = _render_step(new_params, text, cmd, search_flag,
                             self.history[-1]["id"] + 1, self.session_id, current_camera,
                             mode=actual_mode, message=message)
        if actual_mode == "exact":
            self._log_command(cmd, current_params, new_params, step)

        self.history = self.history[:self.cursor + 1] + [step]
        self.cursor = len(self.history) - 1
        self.save()
        return self.state()


UI_SESSION_PATH = os.environ.get("UI_SESSION_PATH", "out/ui_session.json")
session = Session(UI_SESSION_PATH)

app = FastAPI()
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(collect.router)


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


@app.get("/api/datasets/{name}/metadata")
async def dataset_metadata_route(name: str):
    if name not in list_datasets():
        raise HTTPException(status_code=404, detail=f"unknown dataset {name!r}")
    try:
        return dataset_metadata(name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/datasets/{name}/chunks/{index}")
async def dataset_chunk(name: str, index: str):
    try:
        index = int(index)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="chunk index must be an integer")
    if name not in list_datasets():
        raise HTTPException(status_code=404, detail=f"unknown dataset {name!r}")
    try:
        chunk = get_volume_chunk(name, index)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except IndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return Response(
        content=chunk,
        media_type="application/octet-stream",
        headers={"X-Dataset-Version": _dataset_version(name)},
    )


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


class CommandRequest(BaseModel):
    text: str
    parser: str = "rule"
    model: str = "qwen2.5:7b"
    search: bool = False
    steps: int = 10
    mode: str | None = None  # "exact" | "search" | "policy"; overrides `search` when given


@app.post("/api/command")
async def command(req: CommandRequest):
    try:
        return session.command(req.text, req.parser, req.model, req.search, req.steps, req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/scenes/transition")
async def scene_transition_route(payload: dict):
    try:
        for field in ("event_id", "dedupe_key"):
            if field in payload and (not isinstance(payload[field], str) or not payload[field].strip()):
                raise ValueError(f"{field} must be a non-empty string")
        after = payload["after"]
        if payload.get("boundary"):
            transition = normalize_scene(after)
            before = None
            if transition["parent_scene_id"] is not None or transition["command"] != {"attribute": "neutral", "kind": "dataset_boundary"}:
                raise ValueError("boundary event must be a root dataset_boundary scene")
        else:
            before = normalize_scene(payload["before"])
            transition = normalize_scene_transition(
                before,
                after,
                verdict=payload.get("verdict"),
                accepted=payload.get("accepted"),
                ended=payload.get("ended"),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    event_id = _scene_event_id(before, transition, payload.get("event_id"))
    dedupe_key = payload.get("dedupe_key") or event_id
    event = {"event_id": event_id, "dedupe_key": dedupe_key,
             "before_scene": before, "after_scene": transition}
    os.makedirs(os.path.dirname(SCENE_TRANSITIONS_PATH) or ".", exist_ok=True)
    with _SCENE_WRITE_LOCK:
        if os.path.exists(SCENE_TRANSITIONS_PATH):
            with open(SCENE_TRANSITIONS_PATH) as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    existing = json.loads(line)
                    if existing.get("event_id") == event_id or existing.get("dedupe_key") == dedupe_key:
                        if existing.get("after_scene") != transition or existing.get("before_scene") != before:
                            raise HTTPException(status_code=409, detail="event_id conflicts with existing scene transition")
                        return existing["after_scene"]
        jsonl_append(SCENE_TRANSITIONS_PATH, event)
    return transition


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
                     help="a real CT/MRI dataset name (see datasets.DATASETS), or 'synthetic' "
                          "-- already applied above; listed here only for --help")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"[server] dataset: {_dataset_name}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
