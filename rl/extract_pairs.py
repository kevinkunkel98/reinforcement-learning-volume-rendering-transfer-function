"""Extract canonical goal-conditioned preference pairs from JSONL artifacts."""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from commands import TISSUE_HU, _find_tissue
import render


DEFAULT_LOG = "out/log.jsonl"
DEFAULT_FEEDBACK = "out/feedback.jsonl"
DEFAULT_PREFERENCES = "out/rlhf_preferences.jsonl"
DEFAULT_OUT = "out/pairs.jsonl"
WEIGHTS = {"branch": 1.0, "trajectory": 0.7, "thumbs": 0.4}
RATING_LABELS = {"up": 1, "down": -1, "positive": 1, "negative": -1}
FEATURE_KEYS = ("mean", "std", "coverage", "entropy")


def _read_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path) as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _command(row):
    command = row.get("command") or row.get("cmd_dict") or {}
    if not isinstance(command, dict):
        return None
    target = command.get("target_tissue", command.get("target"))
    direction = command.get("direction")
    attribute = command.get("attribute", "opacity")
    if attribute != "opacity":
        return None
    if isinstance(target, str) and target not in TISSUE_HU:
        target = _find_tissue(target)
    if not target or target not in TISSUE_HU or direction not in ("increase", "decrease"):
        return None
    return {"attribute": attribute, "target": target, "direction": direction}


def _image_features(path):
    if not path:
        return None
    try:
        with Image.open(path) as image:
            return render.features(np.asarray(image.convert("RGB")))
    except (OSError, ValueError):
        return None


def _valid_features(features):
    if not isinstance(features, dict) or not all(key in features for key in FEATURE_KEYS):
        return False
    try:
        return all(np.isfinite(float(features[key])) for key in FEATURE_KEYS)
    except (TypeError, ValueError, OverflowError):
        return False


def _observation(row):
    observation = row.get("observation")
    if isinstance(observation, dict):
        result = dict(observation)
    else:
        result = {
            "before_features": row.get("before_features", row.get("features_before")),
            "after_features": row.get("after_features", row.get("features_after")),
            "before_image": row.get("before_image", row.get("before_png", row.get("image_before"))),
            "after_image": row.get("after_image", row.get("after_png", row.get("image_after"))),
        }
    result.setdefault("before_image", result.get("before_png"))
    result.setdefault("after_image", result.get("after_png"))
    for side in ("before", "after"):
        feature_key = f"{side}_features"
        image_key = f"{side}_image"
        result[feature_key] = result.get(feature_key) or _image_features(result.get(image_key))
    if not all(_valid_features(result.get(f"{side}_features")) for side in ("before", "after")):
        return None
    result.setdefault("before_image", None)
    result.setdefault("after_image", None)
    return result


def _state_observation(observation, side, step_id=None):
    features = observation[f"{side}_features"]
    image = observation.get(f"{side}_image")
    return {
        "before_features": dict(features),
        "after_features": dict(features),
        "before_image": image,
        "after_image": image,
        "step_id": step_id,
    }


def _metadata(row):
    return (row.get("session_id", "unknown"), row.get("episode_id", row.get("session_id", "unknown")),
            row.get("step_id"))


def _pair(a, b, command, source, label, metadata):
    session, episode, a_step = metadata
    return {
        "observation_a": a,
        "observation_b": b,
        "command": command,
        "target_tissue": command["target"],
        "direction": command["direction"],
        "label": int(label),
        "source": source,
        "weight": WEIGHTS[source],
        "session_id": session,
        "episode_id": episode,
        "step_id": a_step,
    }


def _pair_key(source, a, b, command, metadata):
    session, episode, _ = metadata
    return (source, session, episode, a.get("step_id"), b.get("step_id"),
            command["attribute"], command["target"], command["direction"])


def _canonical_pair(row):
    if not isinstance(row.get("observation_a"), dict) or not isinstance(row.get("observation_b"), dict):
        return None
    command = _command(row)
    if command is None or row.get("label") not in (-1, 1):
        return None
    source = row.get("source")
    if source not in WEIGHTS:
        return None
    observation_a = _observation({"observation": row["observation_a"]})
    observation_b = _observation({"observation": row["observation_b"]})
    if observation_a is None or observation_b is None:
        return None
    pair = dict(row)
    pair["observation_a"] = observation_a
    pair["observation_b"] = observation_b
    pair["command"] = command
    pair["target_tissue"] = command["target"]
    pair["direction"] = command["direction"]
    pair["label"] = int(row["label"])
    pair["weight"] = float(row.get("weight", WEIGHTS[source]))
    return pair


def _with_step(observation, row):
    result = dict(observation)
    result["step_id"] = row.get("step_id")
    return result


def extract_pairs(log_path=DEFAULT_LOG, feedback_path=DEFAULT_FEEDBACK,
                  out_path=DEFAULT_OUT, preferences_path=None):
    if preferences_path is None:
        preferences_path = DEFAULT_PREFERENCES
    output = Path(out_path).resolve()
    inputs = [path for path in (log_path, feedback_path, preferences_path) if path]
    if any(output == Path(path).resolve() for path in inputs):
        raise ValueError("input and output paths collide")
    log_rows = _read_jsonl(log_path)
    feedback_rows = _read_jsonl(feedback_path)
    preference_rows = _read_jsonl(preferences_path)
    canonical_pairs = []
    for row in [*log_rows, *feedback_rows, *preference_rows]:
        if "observation_a" in row or "observation_b" in row:
            pair = _canonical_pair(row)
            if pair is not None:
                canonical_pairs.append(pair)
    raw_logs = [row for row in [*log_rows, *preference_rows]
                if "observation_a" not in row and "observation_b" not in row]
    feedback = [row for row in [*feedback_rows, *preference_rows]
                if "observation_a" not in row and "observation_b" not in row]
    logs = []
    skipped = 0
    for row in raw_logs:
        command = _command(row)
        observation = _observation(row)
        if command is None or observation is None:
            skipped += 1
            continue
        logs.append({"row": row, "command": command, "observation": observation})

    pairs = []
    seen = set()
    stats = Counter({"branch": 0, "trajectory": 0, "thumbs": 0, "skipped": skipped})

    def add(source, first, second, command, label, metadata):
        key = _pair_key(source, first, second, command, metadata)
        if key in seen:
            return
        seen.add(key)
        pairs.append(_pair(first, second, command, source, label, metadata))
        stats[source] += 1

    for pair in canonical_pairs:
        key = ("canonical", json.dumps(pair, sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            pairs.append(pair)
            stats[pair["source"]] += 1

    by_episode = defaultdict(list)
    for item in logs:
        row = item["row"]
        session, episode, step_id = _metadata(row)
        command = item["command"]
        by_episode[(session, episode, command["attribute"], command["target"], command["direction"])].append(item)
        # Branches require explicit parent and carried-forward metadata. No inference.
        if "parent_step_id" in row and row.get("carried_forward") is True:
            parent = next((candidate for candidate in logs
                           if _metadata(candidate["row"])[0] == session and
                           _metadata(candidate["row"])[1] == episode and
                           candidate["command"] == item["command"] and
                           candidate["row"].get("step_id") == row["parent_step_id"]), None)
            if parent and (row.get("accepted") is True or row.get("ended") is True):
                add("branch", _with_step(item["observation"], row),
                    _with_step(parent["observation"], parent["row"]), item["command"], 1,
                    (session, episode, step_id))

    for items in by_episode.values():
        items.sort(key=lambda item: item["row"].get("step_id", 0))
        for current_index, current in enumerate(items):
            current_row = current["row"]
            if current_row.get("accepted") is not True and current_row.get("ended") is not True:
                continue
            current_step = current_row.get("step_id")
            for earlier in items[:current_index]:
                earlier_step = earlier["row"].get("step_id")
                if isinstance(current_step, int) and isinstance(earlier_step, int) and current_step - earlier_step >= 2:
                    add("trajectory", _with_step(current["observation"], current_row),
                        _with_step(earlier["observation"], earlier["row"]), current["command"], 1,
                        _metadata(current_row))

    by_step = {(row["row"].get("session_id"), row["row"].get("step_id")): row for row in logs}
    by_params = defaultdict(list)
    for item in logs:
        params = item["row"].get("params_after", item["row"].get("params"))
        if params is not None:
            by_params[tuple(params)].append(item)
    for fb in feedback:
        verdict = fb.get("human_verdict")
        label = RATING_LABELS.get(fb.get("rating", fb.get("label")))
        if label is None:
            label = {"better": 1, "worse": -1}.get(verdict)
        if label is None:
            continue
        item = by_step.get((fb.get("session_id"), fb.get("step_id")))
        if item is None and fb.get("params") is not None:
            candidates = by_params.get(tuple(fb["params"]), [])
            item = candidates[0] if len(candidates) == 1 else None
        if item is None:
            skipped += 1
            continue
        if "command" in fb or "cmd_dict" in fb:
            command = _command(fb)
            if command is None:
                skipped += 1
                continue
        else:
            command = item["command"]
        row = item["row"]
        step_id = row.get("step_id")
        add("thumbs", _state_observation(item["observation"], "after", step_id),
            _state_observation(item["observation"], "before", step_id), command, label,
            _metadata(row))

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair) + "\n")
    stats["skipped"] = skipped
    stats["total"] = len(pairs)
    return pairs, dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Extract canonical preference pairs from JSONL logs.")
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--feedback", default=DEFAULT_FEEDBACK)
    parser.add_argument("--preferences", default=DEFAULT_PREFERENCES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    rows, stats = extract_pairs(args.log, args.feedback, args.out, args.preferences)
    sources = ",".join(f"{name}={stats.get(name, 0)}" for name in WEIGHTS)
    targets = Counter(row["target_tissue"] for row in rows)
    target_summary = ",".join(f"{name}={count}" for name, count in sorted(targets.items())) or "none=0"
    print(f"source={sources} target={target_summary} skipped={stats['skipped']} total={stats['total']}")


if __name__ == "__main__":
    main()
