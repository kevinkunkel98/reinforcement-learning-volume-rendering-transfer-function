"""A viewer session: the active scan, the history of rendered steps, and how a
command becomes the next step.

A command is parsed (`commands`), turned into a goal on this scan
(`instruction_for`) and answered in the session's mode -- "exact" applies it
directly, "search" hill-climbs, "policy" asks the trained policy
(`answers`). Each step is rendered (`render`), saved to out/ui_images and
kept in out/ui_session.json, so a restart resumes where it was.

Policy answers come in two stages when `defer_refine` is set: the fast stage
(policy + planning + repairs, ~0.7 s) becomes the step at once; the warm
search runs later (`answers.full_answer`) and `apply_refined` swaps its render
into the same step.
"""
import base64
import datetime
import io
import json
import os
import uuid

import numpy as np
from PIL import Image

import answers
import commands
import goals
import policy as policy_module
import render as render_module
import scans
import transfer
import visibility
from camera import DEFAULT_CAMERA, apply_camera_command
from commands import apply_command, parse_command_with_meta
from rl.search import CONTROLLABLE

IMAGE_DIR = "out/ui_images"
SESSION_PATH = os.environ.get("UI_SESSION_PATH", "out/ui_session.json")

# A command that targets a class but leaves the picture unchanged reads as
# "nothing happened" -- the class sits behind other tissue. Measured on 31
# renders of one camera: no visible change 0.00-0.02 % of the pixels differ by
# more than VISIBLE_PIXEL_DELTA (of 255), barely visible 0.22 %, visible from
# 0.57 %. NO_VISIBLE_CHANGE_SHARE separates the first two from the third.
VISIBLE_PIXEL_DELTA = 16
NO_VISIBLE_CHANGE_SHARE = 0.003

# "add X" of a class a command hid restores it to its start: a relative
# increase of an invisible class stays invisible ("add liver" after "show only
# skeleton" left the liver at 0.0 % of the image). Hidden is below
# goals.EPSILON of the image, or below RESTORE_BELOW_FRACTION of the class's
# share at the task's start -- 0.35, the geometric middle of one moderate (0.5)
# and one strong (0.25) decrease, so "add X" after "less X" stays "more X".
RESTORE_BELOW_SHARE = 0.001
RESTORE_BELOW_FRACTION = 0.35


def scan_label(name: str) -> str:
    """"s0477 · neck–pelvis" from the study type "ct neck-thorax-abdomen-pelvis"."""
    entry = scans.subject(name)
    study = entry["study_type"].lower().removeprefix("ct ")
    prefix = "angio " if study.startswith("angiography ") else ""
    parts = study.removeprefix("angiography ").split("-")
    region = parts[0] if len(parts) == 1 else f"{parts[0]}–{parts[-1]}"
    return f"{entry['id']} · {prefix}{region}"


def scan_list(show_all: bool = False) -> list:
    """The viewer's scan list: the downloaded demo patients (data/demo_scans.json)
    in their order -- every downloaded patient after them with `show_all`, or
    when none of them is downloaded."""
    available = set(scans.available_names())
    demo = [n for n in scans.demo_names() if n in available]
    if show_all or not demo:
        return demo + sorted(available - set(demo))
    return demo


# --- hints and restores --------------------------------------------------------------

def _subs(cmd):
    return cmd["compound"] if isinstance(cmd.get("compound"), list) else [cmd]


def target_classes(cmd) -> list:
    """Goal classes a command singles out, in order (show_only and resets name none)."""
    found = []
    for sub in _subs(cmd):
        if sub.get("direction") in ("show_only", "reset"):
            continue
        target = sub.get("target")
        for name in (target if isinstance(target, list) else [target]):
            if name in goals.GOAL_CLASSES and name not in found:
                found.append(name)
    return found


def _pixels(step):
    try:
        if step.get("image_b64"):
            raw = io.BytesIO(base64.b64decode(step["image_b64"]))
        elif step.get("image_path"):
            raw = step["image_path"]
        else:
            return None
        return np.asarray(Image.open(raw).convert("RGB"), dtype=np.int16)
    except (OSError, ValueError):
        return None


def changed_pixel_share(before_step, after_step):
    """Fraction of pixels visibly different between two steps' renders, or None."""
    a, b = _pixels(before_step), _pixels(after_step)
    if a is None or b is None or a.shape != b.shape:
        return None
    return float((np.abs(a - b).max(axis=-1) > VISIBLE_PIXEL_DELTA).mean())


def no_visible_change_hint(cmd, before_step, after_step):
    """A note for a targeted command whose render barely changed, else None."""
    targets = target_classes(cmd)
    if not targets:
        return None
    share = changed_pixel_share(before_step, after_step)
    if share is None or share >= NO_VISIBLE_CHANGE_SHARE:
        return None
    labels = [commands.CLASS_LABEL.get(name, name) for name in targets]
    return (f"no visible change -- {' and '.join(labels)} mostly hidden behind other "
            f"tissue; try 'show only {labels[0]}'")


def _is_restore(sub, visibility_now, visibility_start):
    """An opacity increase of a class that is hidden now (see RESTORE_BELOW_*)."""
    if sub.get("attribute") != "opacity" or sub.get("direction") != "increase":
        return False
    target = sub.get("target")
    if not isinstance(target, str):
        return False
    share, start_share = visibility_now.get(target), (visibility_start or {}).get(target)
    if share is None:
        return False
    return share < RESTORE_BELOW_SHARE or (start_share is not None and share < RESTORE_BELOW_FRACTION * start_share)


def _class_reset(target):
    return {"target": target, "attribute": None, "direction": "reset", "strength": None}


def restore_hidden(cmd, visibility_now, visibility_start=None):
    """(`cmd` with every opacity increase of a hidden class replaced by a reset
    of that class, the classes so restored). The step stores the rewritten
    command; `split_restores` is how it is applied."""
    if not visibility_now or not isinstance(cmd, dict) or "camera" in cmd:
        return cmd, []
    subs = _subs(cmd)
    hidden = [_is_restore(sub, visibility_now, visibility_start) for sub in subs]
    restored = [sub["target"] for sub, h in zip(subs, hidden) if h]
    if not restored:
        return cmd, []
    new = [_class_reset(sub["target"]) if h else sub for sub, h in zip(subs, hidden)]
    return ({**cmd, "compound": new} if "compound" in cmd else new[0]), restored


def split_restores(cmd, visibility_now, visibility_start=None):
    """(restored classes, the rest of `cmd` or None): the restores are applied
    exactly first, the rest then takes the session's mode from that state."""
    _new, restored = restore_hidden(cmd, visibility_now, visibility_start)
    if not restored:
        return [], cmd
    rest = [sub for sub in _subs(cmd) if not _is_restore(sub, visibility_now, visibility_start)]
    if not rest or "compound" not in cmd:
        return restored, None
    return restored, (rest[0] if len(rest) == 1 else {**cmd, "compound": rest})


def restore_message(classes) -> str:
    labels = " and ".join(commands.CLASS_LABEL.get(c, c) for c in classes)
    # "lungs", "kidneys", "vessels" are plural nouns even as a single class.
    if len(classes) == 1 and not commands.CLASS_LABEL.get(classes[0], classes[0]).endswith("s"):
        return f"{labels} was hidden -- restored to its start"
    return f"{labels} were hidden -- restored to their start"


def _join(message, note):
    return f"{message} · {note}" if message else note


class Session:
    """The active scan and its history of steps, independent of FastAPI.

    `policy_provider` and `model_for_volume` are seams for tests."""

    def __init__(self, scan: str, path: str = SESSION_PATH, policy_provider=None,
                 model_for_volume=visibility.for_volume):
        self.path = path
        self.policy_provider = policy_provider or policy_module.load_policy
        self.model_for_volume = model_for_volume
        self.refine_evaluations = answers.REFINE_EVALUATIONS
        self.pending_refines = {}   # step id -> the job `apply_refined` finishes
        self._load_scan(scan)
        self._start_visibility = None
        self.history, self.cursor, self.session_id = [], 0, None
        if not self._resume():
            self._new_history()

    # --- the active scan ---------------------------------------------------------

    def _load_scan(self, name: str):
        self.volume, self.spacing = scans.load_volume(name)   # raises ValueError if unknown
        self.scan = name
        self.labels = scans.load_labels(name)
        self._frame_bounds = render_module.frame_bounds(self.volume, transfer.default_params(), self.spacing)

    def model(self):
        return self.model_for_volume(self.scan)

    def render_png(self, params, camera) -> bytes:
        img = render_module.grab(render_module.render(
            self.volume, params, spacing=self.spacing, camera=camera, frame_bounds=self._frame_bounds,
            labels=self.labels))
        buffer = io.BytesIO()
        Image.fromarray(img).save(buffer, format="PNG")
        return buffer.getvalue()

    def readout(self, params) -> dict:
        """Share of the image and brightness per goal class; None for a class
        this scan cannot support (vessels without contrast)."""
        agg = goals.aggregate(self.model().features(np.asarray(params, dtype=np.float64)))
        supported = set(goals.goal_classes_for_volume(self.scan))
        return {key: {c: (float(agg[channel][c]) if c in supported else None) for c in goals.GOAL_CLASSES}
                for key, channel in (("class_visibility", "vis"), ("class_brightness", "bright"))}

    def view(self, params, camera) -> dict:
        """What the comparison panel shows for one answer."""
        png = self.render_png(params, camera)
        return {"image_b64": base64.b64encode(png).decode(), **self.readout(params)}

    def start_visibility(self) -> dict:
        """Per-class share at the task's start TF: what "hidden" is measured against."""
        if self._start_visibility is None:
            self._start_visibility = self.readout(goals.starting_params())["class_visibility"]
        return self._start_visibility

    # --- history -----------------------------------------------------------------

    def _new_session_id(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def _resume(self) -> bool:
        """Pick up the saved session when it belongs to this scan."""
        if not os.path.exists(self.path):
            return False
        with open(self.path, encoding="utf-8") as stream:
            data = json.load(stream)
        history = data.get("history", [])
        if (data.get("dataset") != self.scan or not history
                or any(len(step.get("params", [])) != transfer.TOTAL_PARAMS for step in history)):
            return False
        for step in history:
            if step.get("refining"):   # its refine died with the process that started it
                step["refining"], step["refine_discarded"] = False, True
        self.history, self.cursor, self.session_id = history, data["cursor"], data["session_id"]
        return True

    def _new_history(self):
        self.session_id = self._new_session_id()
        self.history = [self.render_step(goals.starting_params(), None, None, 0, dict(scans.DEFAULT_CAMERA))]
        self.cursor = 0
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as stream:
            json.dump({"dataset": self.scan, "history": self.history, "cursor": self.cursor,
                       "session_id": self.session_id}, stream)

    def render_step(self, params, cmd_text, cmd_dict, step_id, camera, mode="exact", message=None,
                    parser_meta=None):
        png = self.render_png(params, camera)
        image_dir = os.path.join(IMAGE_DIR, self.session_id)
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"step_{step_id}.png")
        with open(image_path, "wb") as stream:
            stream.write(png)
        return {
            "id": step_id, "timestamp": datetime.datetime.now().isoformat(),
            "cmd_text": cmd_text, "cmd_dict": cmd_dict, "params": np.asarray(params).tolist(),
            "camera": camera, "image_b64": base64.b64encode(png).decode(), "image_path": image_path,
            "class_visibility": self.readout(params)["class_visibility"],
            "search": mode == "search", "mode": mode, "message": message,
            **(parser_meta or {"parser_requested": None, "parser_used": None, "parser_fallback": None}),
        }

    def state(self) -> dict:
        metadata = scans.label_metadata(self.scan)
        return {
            "cursor": self.cursor, "total": len(self.history), "current": self.history[self.cursor],
            "dataset": self.scan, "session_id": self.session_id,
            "render_info": {"width": render_module.WIDTH, "height": render_module.HEIGHT,
                            "mapper": render_module.MAPPER_NAME, "volume_shape": list(self.volume.shape),
                            "spacing": list(self.spacing), "dataset_version": scans.version(self.scan)},
            "anatomy": {"label_available": True, "available_classes": sorted(metadata["classes"]),
                        "label_layout_version": metadata["label_layout_version"]},
            # The browser viewer shades the unlabelled row with this too (tf_table.js).
            "unlabelled_opacity": transfer.UNLABELLED_OPACITY,
        }

    def switch_dataset(self, name: str) -> dict:
        self._load_scan(name)
        self._start_visibility = None
        self.pending_refines.clear()
        self._new_history()
        return self.state()

    def back(self) -> dict:
        self.cursor = max(0, self.cursor - 1)
        self.save()
        return self.state()

    def forward(self) -> dict:
        self.cursor = min(len(self.history) - 1, self.cursor + 1)
        self.save()
        return self.state()

    def _append(self, step) -> dict:
        self.history = self.history[:self.cursor + 1] + [step]
        self.cursor = len(self.history) - 1
        self.save()
        return self.state()

    # --- answering ---------------------------------------------------------------

    def instruction_for(self, cmd, model, params):
        """(start aggregate, goal) that `cmd` sets from `params` on this scan."""
        start_agg = goals.aggregate(model.features(params))
        instruction = goals.goal_from_command(cmd, model, start_agg, volume=self.scan)
        return start_agg, goals.well_posed(instruction, model, params, start_agg)

    def _policy_job(self, cmd, params):
        policy = self.policy_provider()
        if policy is None:   # before resolving the model: the fallback is a ValueError
            raise ValueError(answers.NO_POLICY_MESSAGE)
        model = self.model()
        start_agg, instruction = self.instruction_for(cmd, model, params)
        return answers.policy_job(model, policy, instruction, start_agg, params,
                                  evaluations=self.refine_evaluations)

    def _answer(self, cmd, params, mode, steps, defer_refine):
        """(new params, mode actually used, message, refine job or None)."""
        if mode == "policy":
            try:
                job = self._policy_job(cmd, params)
            except ValueError as exc:
                return apply_command(cmd, params, reset_params=goals.starting_params()), "exact", str(exc), None
            if defer_refine and job["evaluations"] > 0:
                return answers.fast_answer(job), "policy", None, job
            return answers.full_answer(job), "policy", None, None
        if mode == "search":
            try:
                model = self.model()
                _start_agg, instruction = self.instruction_for(cmd, model, params)
                new_params = answers.search_answer(model, params, instruction, steps)
            except (ValueError, FileNotFoundError, KeyError) as exc:
                return apply_command(cmd, params, reset_params=goals.starting_params()), "exact", str(exc), None
            message = None
            if np.array_equal(new_params, params):
                message = (f"search spent {steps} evaluations without improving on the start; "
                           f"a full sweep needs about {2 * len(CONTROLLABLE) + 1}")
            return new_params, "search", message, None
        return apply_command(cmd, params, reset_params=goals.starting_params()), "exact", None, None

    def command(self, text, parser="rule", model=commands.DEFAULT_LLM_MODEL, mode="exact", steps=10,
                defer_refine=False) -> dict:
        """Parse `text`, answer it in `mode` ("exact", "search" or "policy") and append the step."""
        cmd, parser_meta = parse_command_with_meta(text, parser=parser, model=model)   # ValueError on failure
        current = self.history[self.cursor]
        params = np.array(current["params"], dtype=np.float64)
        camera = dict(current.get("camera", DEFAULT_CAMERA))
        next_id = self.history[-1]["id"] + 1

        if "camera" in cmd:
            step = self.render_step(params, text, cmd, next_id, apply_camera_command(cmd["camera"], camera),
                                    mode="camera", parser_meta=parser_meta)
            return self._append(step)

        # "add X" of a hidden class restores X first; the rest takes the mode's path.
        visibility_now = current.get("class_visibility")
        increases = any(s.get("attribute") == "opacity" and s.get("direction") == "increase" for s in _subs(cmd))
        visibility_start = self.start_visibility() if increases else None
        stored_cmd, restored = restore_hidden(cmd, visibility_now, visibility_start)
        if restored:
            _restored, rest = split_restores(cmd, visibility_now, visibility_start)
            params = apply_command(_class_reset(restored if len(restored) > 1 else restored[0]), params,
                                   reset_params=goals.starting_params())
            if "understood" in parser_meta:
                said = f"restore the {commands._and_list(commands._labels(restored))}"
                parser_meta = {**parser_meta, "understood": " + ".join(
                    [said] + ([commands.describe_command(rest)] if rest is not None else []))}
        else:
            rest = cmd

        job = None
        if rest is None:
            new_params, used, message = params, "exact", None
        elif all(sub.get("direction") == "reset" for sub in _subs(rest)):
            # A reset is an intended exact command, never a policy fallback.
            new_params, used, message = apply_command(rest, params, reset_params=goals.starting_params()), "exact", None
        else:
            new_params, used, message, job = self._answer(rest, params, mode, steps, defer_refine)

        if parser_meta["parser_fallback"] and not message:
            message = f"LLM parse rejected, used the rule parser -- {parser_meta['parser_fallback']}"
        step = self.render_step(new_params, text, stored_cmd, next_id, camera, mode=used, message=message,
                                parser_meta=parser_meta)
        hint = no_visible_change_hint(stored_cmd, current, step)
        if hint:
            step["message"] = _join(step["message"], hint)
        if restored:
            step["message"] = _join(step["message"], restore_message(restored))
        if job is not None:
            step["refining"] = True
            self.pending_refines[step["id"]] = {**job, "session_id": self.session_id,
                                                "fast_params": step["params"], "message": message,
                                                "parser_meta": parser_meta}
        return self._append(step)

    # --- two-stage policy answers --------------------------------------------------

    def _step_index(self, step_id):
        return next((i for i, step in enumerate(self.history) if step["id"] == step_id), None)

    def discard_refine(self, step_id, job=None):
        """Drop a pending refine (e.g. its search raised); the step stops saying it is refining."""
        if job is not None and self.pending_refines.get(step_id) is not job:
            return
        popped = self.pending_refines.pop(step_id, None)
        if popped is not None and popped["session_id"] != self.session_id:
            return
        index = self._step_index(step_id)
        if index is not None and self.history[index].get("refining"):
            self.history[index]["refining"], self.history[index]["refine_discarded"] = False, True
            self.save()

    def apply_refined(self, step_id, params, job=None) -> bool:
        """Render the refined `params` into step `step_id` in place, if that step
        is still the cursor step of this session with the fast params it was
        given; otherwise discard them. Renders, so main thread only."""
        pending = self.pending_refines.get(step_id)
        if pending is None or (job is not None and pending is not job):
            return False
        job = self.pending_refines.pop(step_id)
        if job["session_id"] != self.session_id:
            return False
        index = self._step_index(step_id)
        fast = self.history[index] if index is not None else None
        if index != self.cursor or not fast.get("refining") or fast["params"] != job["fast_params"]:
            if fast is not None and fast.get("refining"):
                fast["refining"], fast["refine_discarded"] = False, True
                self.save()
            return False
        step = self.render_step(np.asarray(params, dtype=np.float64), fast["cmd_text"], fast["cmd_dict"],
                                step_id, fast["camera"], mode=fast["mode"], message=job["message"],
                                parser_meta=job["parser_meta"])
        if index > 0:
            hint = no_visible_change_hint(fast["cmd_dict"], self.history[index - 1], step)
            if hint:
                step["message"] = _join(step["message"], hint)
        step["refining"], step["refined"] = False, True
        self.history[index] = step
        self.save()
        return True
