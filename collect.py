"""FastAPI router for the preference-collection page (`/collect`).

Route handlers are `async def`, for the same reason `server.py`'s are (see
its module docstring, and `tests/test_server.py`'s): `next_route` renders
images (`collect_images.view_grid`, through `Collector.image_fn`), and VTK's
Cocoa render window can only be created on the true process main thread on
macOS. Keeping the handlers `async def` keeps every render call on the
event-loop thread, which `uvicorn.run()` in `server.py`'s `__main__` drives
from the process main thread. Tests call `Collector` methods and the route
coroutines directly (`asyncio.run(...)`) rather than through `TestClient`,
which runs the app on a background thread and would violate that constraint.

`Collector` holds all item-generation, pending-item and repeat-scheduling
state, independent of FastAPI, so it is directly unit-testable and every
external seam (volume list, model lookup, image rendering, dataset version,
item sampling, the policy) is injectable.
"""
import collections
import datetime
import os
import threading
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import visibility
from collect_images import grid_data_url
from datasets import _dataset_version, volumes_for_split
from evaluate import jsonl_append
from rl.candidates import sample_item

PREF_PATH = "out/vis_preferences.jsonl"
_WRITE_LOCK = threading.Lock()

VALID_CHOICES = ("a", "b", "equal", "skip")
VALID_ROLES = ("radiologist", "clinician", "researcher", "other")
REPEAT_RATE = 0.10          # share of items that are a repeat of an earlier one
REPEAT_MIN_GAP = 20         # a repeat is shown only >= this many items after the original

POLICY_PATH = "out/rl_v2/oneshot_v2_seed0/best.zip"
_policy_state = {"loaded": False, "policy": None}


def _load_policy():
    """The trained one-shot policy, loaded from `POLICY_PATH` on first use
    and cached after -- lazily, so the page still works (falling back to the
    non-policy sources, see `rl.candidates.sample_item`) before or without a
    finished training run."""
    if not _policy_state["loaded"]:
        _policy_state["loaded"] = True
        if os.path.exists(POLICY_PATH):
            from stable_baselines3 import SAC
            _policy_state["policy"] = SAC.load(POLICY_PATH)
    return _policy_state["policy"]


def _to_displayed(raw: dict) -> dict:
    """`rl.candidates.sample_item`'s output, narrowed to the fields the
    displayed item needs (drops `"volume"`'s duplication handled elsewhere,
    keeps everything `_swap_sides`/`_response`/the stored row use)."""
    return {
        "volume": raw["volume"],
        "start_params": raw["start_params"],
        "instruction": raw["instruction"],
        "a": raw["a"],
        "b": raw["b"],
        "objective_choice": raw["objective_choice"],
        "near_duplicate": raw["near_duplicate"],
        "features": raw["features"],
    }


def _swap_sides(item: dict) -> dict:
    """The same item with displayed `a`/`b` exchanged -- used both to
    randomize which side a freshly sampled candidate lands on, and to show a
    repeat with sides swapped from the original."""
    return {
        **item,
        "a": item["b"],
        "b": item["a"],
        "objective_choice": {"a": "b", "b": "a"}[item["objective_choice"]],
        "features": {"start": item["features"]["start"], "a": item["features"]["b"], "b": item["features"]["a"]},
    }


def _instruction_json(instruction: dict) -> dict:
    return {
        "kind": instruction["kind"],
        "text": instruction["text"],
        "targets": instruction["targets"],
        "goal": np.asarray(instruction["goal"], dtype=np.float64).tolist(),
    }


class Collector:
    """Item generation, pending-item state, repeat scheduling and the
    append-only judgment log for `/collect`.

    Every external seam is injectable so tests never touch VTK, real
    datasets or a real policy checkpoint: `model_for_volume` (default
    `visibility.for_volume`), `image_fn` (default `collect_images.
    grid_data_url`), `dataset_version_fn` (default `datasets._dataset_
    version`), `sample_item_fn` (default `rl.candidates.sample_item`) and
    `policy_provider` (default `_load_policy`, called -- lazily -- on every
    sampled item).
    """

    def __init__(self, pref_path: str = PREF_PATH, volumes=None, rng: np.random.Generator = None,
                 policy_provider=None, model_for_volume=visibility.for_volume,
                 image_fn=grid_data_url, dataset_version_fn=_dataset_version,
                 sample_item_fn=sample_item):
        self.pref_path = pref_path
        self._volumes = list(volumes) if volumes is not None else None
        self.rng = rng if rng is not None else np.random.default_rng()
        self.policy_provider = policy_provider or _load_policy
        self.model_for_volume = model_for_volume
        self.image_fn = image_fn
        self.dataset_version_fn = dataset_version_fn
        self.sample_item_fn = sample_item_fn

        self._model_cache = {}
        self._items = {}                                  # pair_id -> displayed item, kept for repeats
        self._pending = {}                                 # pair_id -> rater_id, not yet judged
        self._repeat_of = {}                                # pair_id -> original pair_id or None
        self._rater_history = collections.defaultdict(list)  # rater_id -> judged pair_id, in order
        self._rater_meta = {}                               # rater_id -> {"role", "experience"}

    def _get_volumes(self) -> list:
        # Resolved lazily (not at construction) so importing this module, and
        # building the default module-level Collector, never depends on the
        # TotalSegmentator data being present on disk.
        if self._volumes is None:
            self._volumes = volumes_for_split("train") + volumes_for_split("val")
        return self._volumes

    def _get_model(self, volume: str):
        model = self._model_cache.get(volume)
        if model is None:
            model = self.model_for_volume(volume)
            self._model_cache[volume] = model
        return model

    def _should_repeat(self, rater_id: str) -> bool:
        history = self._rater_history[rater_id]
        return len(history) >= REPEAT_MIN_GAP and self.rng.random() < REPEAT_RATE

    def _serve(self, rater_id: str) -> tuple:
        """Generate (or repeat) one item for `rater_id`, register it as
        pending judgment, and return `(pair_id, item, repeat_of)`."""
        repeat_of = None
        if self._should_repeat(rater_id):
            history = self._rater_history[rater_id]
            eligible = history[:len(history) - REPEAT_MIN_GAP + 1]
            repeat_of = str(self.rng.choice(eligible))
            item = _swap_sides(self._items[repeat_of])
        else:
            volume = str(self.rng.choice(self._get_volumes()))
            model = self._get_model(volume)
            raw = self.sample_item_fn(volume, model, self.rng, policy=self.policy_provider())
            item = _to_displayed(raw)
            if self.rng.random() < 0.5:
                item = _swap_sides(item)

        pair_id = str(uuid.uuid4())
        self._items[pair_id] = item
        self._pending[pair_id] = rater_id
        self._repeat_of[pair_id] = repeat_of
        return pair_id, item, repeat_of

    def _response(self, pair_id: str, item: dict, repeat_of) -> dict:
        volume = item["volume"]
        return {
            "pair_id": pair_id,
            "text": item["instruction"]["text"],
            "kind": item["instruction"]["kind"],
            "volume": volume,
            "a_image": self.image_fn(volume, item["a"]["params"]),
            "b_image": self.image_fn(volume, item["b"]["params"]),
            "start_image": self.image_fn(volume, item["start_params"]),
            "repeat_of": repeat_of,
        }

    def next_item(self, rater_id: str, rater_role: str = None, rater_experience: str = None) -> dict:
        """`rater_role`/`rater_experience` are asked once by the frontend
        (alongside the rater ID) and resent on every `/next` call, same as
        `rater_id` itself; when given here they (re)record this rater's
        metadata, used on every row they judge from now on (see `judge`).
        Omitting them (as `judge`'s own internal call does) keeps whatever
        was recorded earlier for this `rater_id`. Raises `ValueError` for a
        `rater_role` outside `VALID_ROLES`."""
        if rater_role is not None:
            if rater_role not in VALID_ROLES:
                raise ValueError(f"invalid role: {rater_role!r}, expected one of {VALID_ROLES}")
            self._rater_meta[rater_id] = {"role": rater_role, "experience": rater_experience}
        pair_id, item, repeat_of = self._serve(rater_id)
        return self._response(pair_id, item, repeat_of)

    def judge(self, pair_id: str, choice: str, decision_ms) -> dict:
        """Append one row for `pair_id` and return the next item. Raises
        `ValueError` for an invalid `choice`, `KeyError` for an unknown or
        already-judged `pair_id`."""
        if choice not in VALID_CHOICES:
            raise ValueError(f"invalid choice: {choice!r}, expected one of {VALID_CHOICES}")
        if pair_id not in self._pending:
            raise KeyError(f"unknown pair_id: {pair_id!r}")

        rater_id = self._pending.pop(pair_id)
        item = self._items[pair_id]
        volume = item["volume"]
        rater_meta = self._rater_meta.get(rater_id, {"role": None, "experience": None})
        row = {
            "pair_id": pair_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "rater_id": rater_id,                    # kept for backwards compatibility with earlier rows
            "rater": {"id": rater_id, "role": rater_meta["role"], "experience": rater_meta["experience"]},
            "volume": volume,
            "volume_version": self.dataset_version_fn(volume),
            "instruction": _instruction_json(item["instruction"]),
            "start_params": np.asarray(item["start_params"], dtype=np.float64).tolist(),
            "a": {"params": np.asarray(item["a"]["params"], dtype=np.float64).tolist(),
                  "source": item["a"]["source"]},
            "b": {"params": np.asarray(item["b"]["params"], dtype=np.float64).tolist(),
                  "source": item["b"]["source"]},
            "choice": choice,
            "objective_choice": item["objective_choice"],
            "near_duplicate": bool(item["near_duplicate"]),
            "features": item["features"],
            "decision_ms": decision_ms,
            "repeat_of": self._repeat_of.get(pair_id),
        }
        with _WRITE_LOCK:
            os.makedirs(os.path.dirname(self.pref_path) or ".", exist_ok=True)
            jsonl_append(self.pref_path, row)
        self._rater_history[rater_id].append(pair_id)

        return self.next_item(rater_id)


router = APIRouter()
collector = Collector()


class NextRequest(BaseModel):
    rater_id: str
    rater_role: str | None = None
    rater_experience: str | None = None


class JudgeRequest(BaseModel):
    pair_id: str
    choice: str
    decision_ms: float | None = None


@router.get("/collect")
async def collect_page():
    return FileResponse("static/collect.html")


@router.post("/api/collect/next")
async def next_route(req: NextRequest):
    try:
        return collector.next_item(req.rater_id, req.rater_role, req.rater_experience)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/collect/judge")
async def judge_route(req: JudgeRequest):
    try:
        return collector.judge(req.pair_id, req.choice, req.decision_ms)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
