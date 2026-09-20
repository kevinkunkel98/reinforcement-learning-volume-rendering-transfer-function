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
from commands import COMMAND_REFERENCE, apply_command, parse_command_with_meta
from datasets import _dataset_version, dataset_metadata, default_camera_for, get_volume_chunk, list_datasets, load_dataset
from evaluate import jsonl_append
import goals
import policy as policy_module
import render as render_module
from render import features, grab, render
from rl.baselines import CONTROLLABLE, apply_controllable, hill_climb
from rl.oneshot_env import build_observation
from scene_schema import normalize_scene, scene_transition as normalize_scene_transition
from transfer import TISSUE_BANDS, default_params, opacity_mass
import visibility

LOG_PATH = "out/log.jsonl"
SCENE_TRANSITIONS_PATH = "out/scene_transitions.jsonl"
_SCENE_WRITE_LOCK = threading.Lock()
AUDIO_DIR = "out/audio"
NO_POLICY_CHECKPOINT_MESSAGE = (
    "no trained policy checkpoint found -- applied the command directly instead")

# --- Task 3: the one-shot policy, for mode="policy" --------------------------
# Loaded lazily and cached, the same pattern collect.py uses for the same
# checkpoint: importing this module (and starting the server) must not
# require a finished training run, and every command after the first pays no
# reload cost.
# v3, not v2: the v2 seeds were trained before the observation fixes (the
# reachable-ceiling probe sat on the retired band layout's fat peak, and the
# colour action wrote one scalar to r, g and b). Retraining on the corrected
# observation does move held-out attainment: v3 +0.286 against v2 +0.201 on the
# seed-averaged per-episode median, paired Wilcoxon p = 0.0064. The earlier
# reading -- indistinguishable, +0.1732 vs +0.1743, p = 0.85 -- came from the
# stale evaluation batch of 2026-09-17 and is withdrawn. The viewer should
# demonstrate the pipeline the thesis describes in any case.
# `policy.py` owns the checkpoint path; these are re-exports so the existing
# call sites and tests keep working.
POLICY_PATH = policy_module.POLICY_PATH
_policy_state = policy_module._state
_load_policy = policy_module.load_policy


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


def _class_visibility(params, model_for_volume=visibility.for_volume):
    """Share of the rendered image each goal class contributes, for the
    viewer's readout.

    `_masses` measures opacity over the retired fat/air/spongy bands -- peak
    geometry, not anatomy, and in a vocabulary no instruction can target any
    more. This is what the policy and the preference collector are scored on
    instead. Classes this volume cannot support (`vessels` without contrast)
    report None rather than a 0.0 the viewer would draw as "hidden"."""
    try:
        model = model_for_volume(_dataset_name)
        vis = goals.aggregate(model.features(np.asarray(params, dtype=np.float64)))["vis"]
        supported = set(goals.goal_classes_for_volume(_dataset_name))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"[telemetry] no visibility model for {_dataset_name}: {exc}")
        return {goal_class: None for goal_class in goals.GOAL_CLASSES}
    return {goal_class: (float(vis[goal_class]) if goal_class in supported else None)
            for goal_class in goals.GOAL_CLASSES}


_frame_bounds_cache = {"dataset": None, "bounds": None}


def _frame_bounds():
    """The framing every step of this dataset's conversation is rendered in.

    Measured once, from the dataset's *default* transfer function, so a
    command that hides a tissue doesn't also re-zoom the picture -- the whole
    point of a before/after pair is that only the tissue changed."""
    volume, spacing = get_volume()
    if _frame_bounds_cache["dataset"] != _dataset_name:
        _frame_bounds_cache["dataset"] = _dataset_name
        _frame_bounds_cache["bounds"] = render_module.frame_bounds(volume, default_params(), spacing)
    return _frame_bounds_cache["bounds"]


def _render_image_b64(params, camera):
    volume, spacing = get_volume()
    img = grab(render(volume, params, spacing=spacing, camera=camera, frame_bounds=_frame_bounds()))
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
                  mode="exact", message=None, parser_meta=None):
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
        "class_visibility": _class_visibility(params),
        "features": features(img),
        "search": search,
        "mode": mode,
        "message": message,
        **(parser_meta or {"parser_requested": None, "parser_used": None, "parser_fallback": None}),
    }


def run_policy_arm(model, policy, instruction, start_agg, start_params):
    """Run the one-shot policy on an already-built goal.

    The caller owns `instruction` (`goals.goal_from_command`) and `start_agg`,
    so a caller comparing several arms builds each exactly once -- recomputing
    them here would put another `model.features` (~17 ms) inside the policy
    arm's own timing, which is the one number the comparison panel exists to
    show. Builds the same observation layout `rl.candidates` builds for a
    standalone policy query (`rl.oneshot_env.build_observation`).

    Returns the new parameters. Raises `ValueError` when `policy` is None;
    `model.features`/`model.solo_max` may also raise `FileNotFoundError` or
    `KeyError` when the volume has no visibility cache."""
    if policy is None:
        raise ValueError(NO_POLICY_CHECKPOINT_MESSAGE)

    solo_max_log = [math.log10(sum(model.solo_max(m) for m in goals.MEASURED_FOR_GOAL[c]) + goals.EPSILON)
                     for c in goals.GOAL_CLASSES]
    controllable = [float(np.mean([start_params[i] for i in group])) for group in CONTROLLABLE]
    observation = build_observation(instruction["goal"], model.histogram, start_agg,
                                     solo_max_log, controllable)

    action = _predict_action(policy, observation)
    return apply_controllable(start_params, action)


def run_search_arm(model, start_params, instruction, evaluations):
    """Run the hill-climb baseline (`rl.baselines.hill_climb`) on an
    already-built goal -- the search twin of `run_policy_arm`.

    The caller owns `instruction` (`goals.goal_from_command`), the same
    reason `run_policy_arm` takes a pre-built `instruction`/`start_agg`
    rather than recomputing them: a caller comparing several arms builds the
    goal exactly once instead of putting another `model.features` call
    inside the arm's own timing.

    `evaluations` is a budget on `model.features` calls inside `hill_climb`,
    not a count of accepted moves: one evaluation scores the start state,
    and each remaining one tries a single +-step on one of the 12
    controllable groups. A full sweep (12 groups x 2 signs) costs ~24 --
    below that, `hill_climb` can run out of budget before trying every
    direction once and return the start state completely untouched. This
    function does not detect that case; the caller (`Session._run_search`)
    does, because only it knows the params it started from.

    Returns the new parameters. `model.features`/`model.solo_max` may raise
    `FileNotFoundError` or `KeyError` when the volume has no visibility
    cache."""
    return hill_climb(model, start_params, instruction, evaluations=evaluations)


# The two search budgets the thesis reports: B3 (cheap, 10) and B4 (thorough,
# 200). Showing both is what makes a no-move B3 column legible -- beside a B4
# that did move and a policy that answered instantly, "10 evaluations buys
# little here" reads as the finding rather than as a broken column.
#
# `hill_climb` spends 1 evaluation on the start state and 1 per proposal, and a
# full coordinate sweep is 12 groups x 2 signs = 24, so B3 explores less than
# half a sweep. Measured on ct_chest / "show only the lungs": B3 130 ms,
# B4 2.7 s -- the whole panel is under three seconds.
COMPARE_BUDGET_CHEAP = 10
COMPARE_BUDGET_THOROUGH = 200


def compare_arms(model, policy, cmd, instruction, start_params, camera,
                  cheap=COMPARE_BUDGET_CHEAP, thorough=COMPARE_BUDGET_THOROUGH):
    """Answer one parsed command four ways from the same start state.

    `exact` applies the command directly (0 evaluations), `search_cheap` and
    `search_thorough` run the hill-climb the thesis measures as B3 and B4, and
    `policy` runs the trained one-shot policy (0 evaluations). Every arm is
    scored with `goals.attainment` against the *shared* start aggregate, which
    is what makes these numbers mean what the held-out table means.

    `model` and `instruction` must come from a single read of the active
    dataset, not two independent resolutions at different layers -- otherwise
    an arm could be scored against a goal built for a different volume.

    Does not touch session history: comparing is a side quest, not a step.

    An arm that raises is reported as `unavailable` rather than failing the
    whole comparison -- losing one column should not end a live demo."""
    start_agg = goals.aggregate(model.features(start_params))

    def _answer(fn):
        """Run one arm and time *only* the answering.

        The timing column exists to show that the policy answers for free
        while thorough search costs seconds. Scoring and rendering cost the
        same for every arm, so folding them in inflates the cheap arms towards
        the expensive one and the policy's number stops being the policy's
        number -- the same mistake `run_policy_arm`'s seam was shaped to
        avoid, one layer up."""
        started = time.perf_counter()
        params = fn()
        return params, int((time.perf_counter() - started) * 1000)

    def _finish(params, evaluations, elapsed_ms):
        final_agg = goals.aggregate(model.features(params))
        image_b64, _img, _png = _render_image_b64(params, camera)
        return {
            "params": params.tolist(),
            "image_b64": image_b64,
            "class_visibility": _class_visibility(params),
            "attainment": float(goals.attainment(instruction["goal"], start_agg, final_agg)),
            # An arm that returned its own input is not an arm that *agrees*
            # with the start -- it is an arm that did not move, and rendered
            # side by side those two read identically (same image, same
            # attainment, a confident evaluation count) while meaning opposite
            # things. At B3's budget the search arm can legitimately land here.
            "unchanged": bool(np.array_equal(params, start_params)),
            "evaluations": evaluations,
            "elapsed_ms": elapsed_ms,
        }

    def _unavailable(exc, elapsed_ms):
        # `str(exc)` is not user-facing copy. A KeyError stringifies to the
        # bare repr of its key ("'lungs'"), and goal_from_command yields a
        # dumped Python dict. In a single-view toast that is merely scruffy;
        # in a column someone is reading closely it is unreadable.
        if isinstance(exc, ValueError):
            reason = f"not applicable -- {exc}"
        else:
            reason = "unavailable -- this volume has no visibility cache"
        # Same keys as a successful arm, so a consumer iterating the arms
        # never has to special-case a failed one before reading a field.
        return {"params": None, "unavailable": reason, "attainment": None,
                "class_visibility": None, "image_b64": None, "evaluations": None,
                "unchanged": None, "elapsed_ms": elapsed_ms}

    def _arm(fn, evaluations):
        started = time.perf_counter()
        try:
            params, elapsed_ms = _answer(fn)
            return _finish(params, evaluations, elapsed_ms)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            return _unavailable(exc, int((time.perf_counter() - started) * 1000))

    arms = {"exact": _arm(lambda: apply_command(cmd, start_params), 0)}
    for name, budget in (("search_cheap", cheap), ("search_thorough", thorough)):
        arms[name] = _arm(
            lambda budget=budget: run_search_arm(model, start_params, instruction, budget),
            budget)
    arms["policy"] = _arm(
        lambda: run_policy_arm(model, policy, instruction, start_agg, start_params), 0)
    return arms


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

    def _run_policy(self, cmd, current_params):
        """mode="policy": resolve the checkpoint and the volume model, build
        the goal, and hand them to `run_policy_arm`.

        The `policy is None` check stays ahead of `model_for_volume`: that call
        is not total (`visibility.for_volume` raises `FileNotFoundError` when a
        volume has no cache, which is why `_class_visibility` guards it), and
        `command()` catches only `ValueError`. Resolving the model first would
        turn a graceful "no checkpoint, applied directly" fallback into an
        unhandled exception out of the route. The `ValueError` raised here is
        what `command()` catches to fall back to exact application."""
        policy = self.policy_provider()
        if policy is None:
            raise ValueError(NO_POLICY_CHECKPOINT_MESSAGE)
        model = self.model_for_volume(_dataset_name)
        start_agg = goals.aggregate(model.features(current_params))
        instruction = goals.goal_from_command(cmd, model, start_agg, volume=_dataset_name)
        return run_policy_arm(model, policy, instruction, start_agg, current_params), instruction["text"]

    def _run_search(self, cmd, current_params, steps):
        """mode="search": resolve the volume model, build the goal, and hand
        them to `run_search_arm` -- the search twin of `_run_policy`.

        Unlike `_run_policy` there is no checkpoint to check for ahead of
        `model_for_volume`, so this calls it directly; `command()` widens its
        `except` to `(ValueError, FileNotFoundError, KeyError)` instead
        (`visibility.for_volume` raises `FileNotFoundError` for a volume with
        no cache, same as it does for the policy branch)."""
        model = self.model_for_volume(_dataset_name)
        start_agg = goals.aggregate(model.features(current_params))
        instruction = goals.goal_from_command(cmd, model, start_agg, volume=_dataset_name)
        return run_search_arm(model, current_params, instruction, steps), instruction["text"]

    def command(self, text, parser="rule", model="qwen2.5:7b", search=False, steps=10, mode=None):
        cmd, parser_meta = parse_command_with_meta(text, parser=parser, model=model)  # raises ValueError on failure

        current_params = np.array(self.history[self.cursor]["params"], dtype=np.float64)
        current_camera = dict(self.history[self.cursor].get("camera", DEFAULT_CAMERA))

        if "camera" in cmd:
            new_camera = apply_camera_command(cmd["camera"], current_camera)
            step = _render_step(current_params, text, cmd, False,
                                 self.history[-1]["id"] + 1, self.session_id, new_camera,
                                 mode="camera", parser_meta=parser_meta)
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
        elif effective_mode == "search":
            try:
                new_params, _goal_text = self._run_search(cmd, current_params, steps)
                actual_mode, search_flag = "search", True
                if np.array_equal(new_params, current_params):
                    # hill_climb ran -- search_flag stays True -- but a
                    # budget short of a full sweep (12 groups x 2 signs, plus
                    # 1 to score the start = ~25 evaluations) can spend its
                    # whole budget without finding a single improving move
                    # and return the start state untouched. Silence here is
                    # exactly the "search toggle active, nothing happened"
                    # failure this task exists to eliminate, so say so.
                    full_sweep = 2 * len(CONTROLLABLE) + 1
                    message = (f"search spent {steps} evaluations without improving on the "
                               f"start; a full sweep needs about {full_sweep}")
            except (ValueError, FileNotFoundError, KeyError) as exc:
                message = str(exc)
                new_params = apply_command(cmd, current_params)
                actual_mode, search_flag = "exact", False
        else:
            new_params = apply_command(cmd, current_params)
            actual_mode, search_flag = "exact", False

        if parser_meta["parser_fallback"] and not message:
            # The viewer asked for the LLM parser and got a rule parse; say so
            # in the same channel policy-mode fallbacks already use.
            message = f"LLM parse rejected, used the rule parser -- {parser_meta['parser_fallback']}"

        step = _render_step(new_params, text, cmd, search_flag,
                             self.history[-1]["id"] + 1, self.session_id, current_camera,
                             mode=actual_mode, message=message, parser_meta=parser_meta)
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


class CompareRequest(BaseModel):
    text: str
    parser: str = "rule"
    model: str = "qwen2.5:7b"
    cheap: int = COMPARE_BUDGET_CHEAP
    thorough: int = COMPARE_BUDGET_THOROUGH


@app.post("/api/compare")
async def compare_route(req: CompareRequest):
    """Answer one instruction four ways without advancing the session.

    Parses once and builds the goal once, so all four arms answer the
    identical parsed command: a bad parse then makes all four wrong together
    and the panel shows a parsing problem, not a policy problem. `model` and
    `instruction` come from a single read of the active dataset for the same
    reason -- an arm scored against a goal built for a different volume would
    be quietly meaningless."""
    try:
        cmd, parser_meta = parse_command_with_meta(req.text, parser=req.parser, model=req.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    start_params = np.array(session.history[session.cursor]["params"], dtype=np.float64)
    camera = dict(session.history[session.cursor].get("camera", DEFAULT_CAMERA))

    try:
        volume_model = session.model_for_volume(_dataset_name)
        start_agg = goals.aggregate(volume_model.features(start_params))
        instruction = goals.goal_from_command(cmd, volume_model, start_agg, volume=_dataset_name)
    except ValueError as exc:
        # Camera, reset, width and centre commands are not goals, and neither
        # is a goal this volume cannot support (lungs on an abdominal scan).
        return {"applicable": False, "reason": str(exc), "text": req.text, **parser_meta}
    except (FileNotFoundError, KeyError):
        return {"applicable": False,
                "reason": "this volume has no visibility cache, so it cannot be scored",
                "text": req.text, **parser_meta}

    arms = compare_arms(volume_model, session.policy_provider(), cmd, instruction,
                         start_params, camera, cheap=req.cheap, thorough=req.thorough)
    return {"applicable": True, "text": req.text, "goal_text": instruction["text"],
            "budgets": {"cheap": req.cheap, "thorough": req.thorough}, "arms": arms,
            "start": {"class_visibility": _class_visibility(start_params)},
            **parser_meta}


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
