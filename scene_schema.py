"""Validation and normalization for renderer-neutral scene records."""

from __future__ import annotations

import copy
import math
from numbers import Real
from typing import Any, Mapping


_REQUIRED_SCENE_FIELDS = (
    "scene_id",
    "parent_scene_id",
    "session_id",
    "client",
    "dataset",
    "dataset_version",
    "volume",
    "transfer_function",
    "camera",
    "goal",
    "command",
)
_OPTIONAL_TRANSITION_FIELDS = ("verdict", "accepted", "ended")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _identifier(value: Any, field: str, *, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _finite_number(value: Any, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return int(value) if value.is_integer() else value


def _vector(
    value: Any,
    field: str,
    length: int,
    *,
    positive: bool = False,
    integers: bool = False,
) -> list[int | float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain {length} values")
    result = [_finite_number(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if integers and any(not isinstance(item, int) for item in result):
        raise ValueError(f"{field} values must be integers")
    if positive and any(item <= 0 for item in result):
        raise ValueError(f"{field} values must be positive")
    return result


def _text_field(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _text_value(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def normalize_scene(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a scene and return an independent JSON-serializable copy."""
    record = _mapping(record, "scene")
    missing = [field for field in _REQUIRED_SCENE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"scene missing fields: {', '.join(missing)}")

    scene: dict[str, Any] = {
        "scene_id": _identifier(record["scene_id"], "scene_id"),
        "parent_scene_id": _identifier(record["parent_scene_id"], "parent_scene_id", allow_none=True),
        "session_id": _identifier(record["session_id"], "session_id"),
        "client": record["client"],
        "dataset": _text_field(record, "dataset"),
        "dataset_version": _text_field(record, "dataset_version"),
    }
    if scene["client"] not in ("web", "vrui"):
        raise ValueError("client must be 'web' or 'vrui'")

    volume = _mapping(record["volume"], "volume")
    for field in ("dimensions", "spacing", "scalar_type", "orientation"):
        if field not in volume:
            raise ValueError(f"volume missing field: {field}")
    scalar_type = volume["scalar_type"]
    if scalar_type != "float32":
        raise ValueError("volume scalar_type must be 'float32'")
    scene["volume"] = {
        "dimensions": _vector(volume["dimensions"], "volume.dimensions", 3, positive=True, integers=True),
        "spacing": _vector(volume["spacing"], "volume.spacing", 3, positive=True),
        "scalar_type": scalar_type,
        "orientation": _text_field(volume, "orientation"),
    }

    transfer_function = record["transfer_function"]
    scene["transfer_function"] = _vector(transfer_function, "transfer_function", 24)

    camera = _mapping(record["camera"], "camera")
    for field in ("position", "focal_point", "view_up", "zoom"):
        if field not in camera:
            raise ValueError(f"camera missing field: {field}")
    zoom = _finite_number(camera["zoom"], "camera.zoom")
    if zoom <= 0:
        raise ValueError("camera.zoom must be positive")
    scene["camera"] = {
        "position": _vector(camera["position"], "camera.position", 3),
        "focal_point": _vector(camera["focal_point"], "camera.focal_point", 3),
        "view_up": _vector(camera["view_up"], "camera.view_up", 3),
        "zoom": zoom,
    }

    goal = _mapping(record["goal"], "goal")
    scene["goal"] = {
        "target": _text_value(goal.get("target"), "goal.target"),
        "direction": _text_value(goal.get("direction"), "goal.direction"),
    }

    command = _mapping(record["command"], "command")
    if command.get("attribute") != "opacity":
        raise ValueError("command.attribute must be 'opacity'")
    scene["command"] = {
        "attribute": "opacity",
        "target": _text_value(command.get("target"), "command.target"),
        "direction": _text_value(command.get("direction"), "command.direction"),
    }

    for field in _OPTIONAL_TRANSITION_FIELDS:
        if field in record:
            value = record[field]
            if field == "verdict" and value is not None and not isinstance(value, str):
                raise ValueError("verdict must be a string or null")
            if field in ("accepted", "ended") and value is not None and not isinstance(value, bool):
                raise ValueError(f"{field} must be boolean or null")
            scene[field] = value

    for field, value in record.items():
        if field not in scene:
            scene[field] = copy.deepcopy(value)
    return scene


def scene_transition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    verdict: str | None = None,
    accepted: bool | None = None,
    ended: bool | None = None,
) -> dict[str, Any]:
    """Normalize a transition and require an explicit parent scene link."""
    before_scene = normalize_scene(before)
    after_scene = normalize_scene(after)
    if after_scene["parent_scene_id"] != before_scene["scene_id"]:
        raise ValueError("after.parent_scene_id must equal before.scene_id")
    transition = after_scene
    if verdict is not None:
        transition["verdict"] = verdict
    if accepted is not None:
        transition["accepted"] = accepted
    if ended is not None:
        transition["ended"] = ended
    return normalize_scene(transition)
