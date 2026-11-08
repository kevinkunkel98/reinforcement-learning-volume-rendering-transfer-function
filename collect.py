"""The preference-collection page (`/collect`): raters see a start render, an
instruction and two candidate answers (`rl.candidates`), judged blind (A, B,
equal, skip). Rows go to out/vis_preferences_v3.jsonl with each candidate's
source and attainment; tools/merge_preferences.py merges raters' files.

Some items repeat an earlier one with sides swapped (rater consistency), and
some come from a fixed anchor pool every rater judges (agreement between
raters). Handlers are `async def` so the renders stay on the main thread (see
server.py). `Collector`'s seams (volumes, model lookup, images, item sampling,
the policy) are injectable for tests.
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

import json

import policy as policy_module
import scans
import visibility
from collect_images import grid_data_url
from rl.candidates import anchor_items, sample_item

# Judgments on task-v3 renders (label-scoped, faded shell) with the
# model-based candidates (rl.candidates.TASK_SOURCES), from 2026-10-19 on. The
# 59 earlier rows in out/vis_preferences.jsonl compare v1-era renders and
# SAC-era sources -- a pilot, kept in its own file and never mixed in.
PREF_PATH = "out/vis_preferences_v3.jsonl"
_WRITE_LOCK = threading.Lock()

VALID_CHOICES = ("a", "b", "equal", "skip")
VALID_ROLES = ("radiologist", "clinician", "researcher", "other")
REPEAT_RATE = 0.10          # share of items that are a repeat of an earlier one
REPEAT_MIN_GAP = 20         # a repeat is shown only >= this many items after the original
ANCHOR_RATE = 0.20          # share of items drawn from the shared anchor pool (roughly one in five)



def _append(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


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
    features = {**item["features"], "a": item["features"]["b"], "b": item["features"]["a"]}
    if "attainment" in features:
        features["attainment"] = {"a": features["attainment"]["b"], "b": features["attainment"]["a"]}
    return {
        **item,
        "a": item["b"],
        "b": item["a"],
        "objective_choice": {"a": "b", "b": "a"}[item["objective_choice"]],
        "features": features,
    }


def _instruction_json(instruction: dict) -> dict:
    return {
        "kind": instruction["kind"],
        "text": None if instruction["text"] is None else str(instruction["text"]),
        "targets": instruction["targets"],
        "goal": np.asarray(instruction["goal"], dtype=np.float64).tolist(),
        **({"weights": {c: float(w) for c, w in instruction["weights"].items()}}
           if instruction.get("weights") else {}),
    }


class Collector:
    """Item generation, pending items, repeat scheduling and the append-only
    judgment log for `/collect`, independent of FastAPI."""

    def __init__(self, pref_path: str = PREF_PATH, volumes=None, rng: np.random.Generator = None,
                 policy_provider=None, model_for_volume=visibility.for_volume,
                 image_fn=grid_data_url, sample_item_fn=sample_item, anchor_items_fn=anchor_items):
        self.pref_path = pref_path
        self._volumes = list(volumes) if volumes is not None else None
        self.rng = rng if rng is not None else np.random.default_rng()
        self.policy_provider = policy_provider or policy_module.load_policy
        self.model_for_volume = model_for_volume
        self.image_fn = image_fn
        self.sample_item_fn = sample_item_fn
        self.anchor_items_fn = anchor_items_fn

        self._model_cache = {}
        self._items = {}                                    # pair_id -> displayed item, kept for repeats
        self._pending = {}                                  # pair_id -> rater_id, not yet judged
        self._repeat_of = {}                                # pair_id -> original pair_id or None
        self._anchor_id = {}                                # pair_id -> anchor pool index or None
        self._rater_history = collections.defaultdict(list)  # rater_id -> judged pair_id, in order
        self._rater_meta = {}                               # rater_id -> {"role", "experience"}
        self._anchor_pool = None                            # generated on first use, then cached
        self._rater_anchor_seen = collections.defaultdict(set)  # rater_id -> anchor indices served

    def _get_volumes(self) -> list:
        # Resolved on first use, so importing this module never needs the scans.
        if self._volumes is None:
            self._volumes = scans.split_names("train") + scans.split_names("val")
        return self._volumes

    def _get_model(self, volume: str):
        if volume not in self._model_cache:
            self._model_cache[volume] = self.model_for_volume(volume)
        return self._model_cache[volume]

    def _should_repeat(self, rater_id: str) -> bool:
        history = self._rater_history[rater_id]
        return len(history) >= REPEAT_MIN_GAP and self.rng.random() < REPEAT_RATE

    def _get_anchor_pool(self) -> list:
        # Generated at most once per process (the module-level `collector`
        # lives for the life of the server) -- `anchor_items_fn` itself
        # caches to disk (see `rl.candidates.anchor_items`), so a restart is
        # cheap too.
        if self._anchor_pool is None:
            # The anchors include the policy's candidates, so the pool is tied
            # to the checkpoint (`policy_id`).
            self._anchor_pool = self.anchor_items_fn(volumes=self._get_volumes(),
                                                     model_for_volume=self.model_for_volume,
                                                     policy=self.policy_provider(),
                                                     policy_id=policy_module.POLICY_PATH)
        return self._anchor_pool

    def _next_anchor_index(self, rater_id: str):
        """The lowest anchor-pool index `rater_id` has not yet been served,
        in the same deterministic order (0, 1, 2, ...) for every rater --
        or `None` once the pool is exhausted for them."""
        pool = self._get_anchor_pool()
        seen = self._rater_anchor_seen[rater_id]
        for i in range(len(pool)):
            if i not in seen:
                return i
        return None

    def _serve(self, rater_id: str) -> tuple:
        """Generate (or repeat, or draw from the anchor pool) one item for
        `rater_id`, register it as pending judgment, and return `(pair_id,
        item, repeat_of, anchor_id)`."""
        repeat_of = None
        anchor_id = None
        if self._should_repeat(rater_id):
            history = self._rater_history[rater_id]
            eligible = history[:len(history) - REPEAT_MIN_GAP + 1]
            repeat_of = str(self.rng.choice(eligible))
            item = _swap_sides(self._items[repeat_of])
        elif (candidate := self._next_anchor_index(rater_id)) is not None and self.rng.random() < ANCHOR_RATE:
            anchor_id = candidate
            self._rater_anchor_seen[rater_id].add(anchor_id)
            item = _to_displayed(self._get_anchor_pool()[anchor_id])
            if self.rng.random() < 0.5:
                item = _swap_sides(item)
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
        self._anchor_id[pair_id] = anchor_id
        return pair_id, item, repeat_of, anchor_id

    def _response(self, pair_id: str, item: dict, repeat_of) -> dict:
        volume = item["volume"]
        return {
            "pair_id": pair_id,
            "text": None if item["instruction"]["text"] is None else str(item["instruction"]["text"]),
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
        pair_id, item, repeat_of, _anchor_id = self._serve(rater_id)
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
            "volume_version": scans.version(volume),
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
            "anchor_id": self._anchor_id.get(pair_id),
            "task_version": "v3",
            "policy_path": policy_module.POLICY_PATH,
        }
        with _WRITE_LOCK:
            os.makedirs(os.path.dirname(self.pref_path) or ".", exist_ok=True)
            _append(self.pref_path, row)
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
