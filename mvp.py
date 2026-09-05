"""CLI: render, apply a command, hill-climb, or listen on the mic."""
import argparse
import datetime
import json
import os
import uuid

import numpy as np
from PIL import Image

from datasets import load_dataset
from transfer import default_params, opacity_mass, TISSUE_BANDS
from render import render, grab, features
from commands import parse_command, apply_command, STRENGTH_WORDS, _find_or_create_peak
from evaluate import objective, human, jsonl_append
from search import propose_step, resize_step

STATE_PATH = "out/state.json"
LOG_PATH = "out/log.jsonl"
PREF_PATH = "out/preferences.jsonl"

_VOLUME = None
_SPACING = None
_DATASET_NAME = "synthetic"


def get_volume():
    global _VOLUME, _SPACING
    if _VOLUME is None:
        _VOLUME, _SPACING = load_dataset(_DATASET_NAME)
    return _VOLUME, _SPACING


def load_state() -> np.ndarray:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return np.array(json.load(f), dtype=np.float64)
    return default_params()


def save_state(params: np.ndarray) -> None:
    os.makedirs("out", exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(params.tolist(), f)


def save_png(params: np.ndarray, path: str) -> np.ndarray:
    volume, spacing = get_volume()
    img = grab(render(volume, params, spacing=spacing))
    Image.fromarray(img).save(path)
    return img


def masses(params: np.ndarray) -> dict:
    return {t: opacity_mass(params, lo, hi) for t, (lo, hi) in TISSUE_BANDS.items()}


def print_step(cmd, before, after, feats_before, feats_after, verdict):
    print("command:", cmd)
    print("opacity mass before:", {k: round(v, 4) for k, v in masses(before).items()})
    print("opacity mass after: ", {k: round(v, 4) for k, v in masses(after).items()})
    print("features before:", {k: round(v, 3) for k, v in feats_before.items()})
    print("features after: ", {k: round(v, 3) for k, v in feats_after.items()})
    print("verdict:", verdict)


def log_step(cmd, before, after, feats_before, feats_after, verdict):
    jsonl_append(LOG_PATH, {
        "timestamp": datetime.datetime.now().isoformat(),
        "command": cmd,
        "params_before": before.tolist(),
        "params_after": after.tolist(),
        "features_before": feats_before,
        "features_after": feats_after,
        "verdict": verdict,
    })


def log_preference(session_id, cmd_text, cmd, before, after, feats_before, feats_after,
                    before_png, after_png, human_verdict, objective_verdict):
    jsonl_append(PREF_PATH, {
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "cmd_text": cmd_text,
        "cmd_dict": cmd,
        "params_before": before.tolist(),
        "params_after": after.tolist(),
        "features_before": feats_before,
        "features_after": feats_after,
        "before_png": before_png,
        "after_png": after_png,
        "human_verdict": "better" if human_verdict == 1 else "worse",
        "objective_verdict": "better" if objective_verdict == 1 else "worse",
    })


def hill_climb(cmd_text, cmd, params, steps, use_human, session_id, out_dir="out"):
    os.makedirs(out_dir, exist_ok=True)
    sign = 1.0 if cmd["direction"] == "increase" else -1.0
    step = STRENGTH_WORDS[cmd["strength"] or "moderately"]

    current = params.copy()
    _, current_idx = _find_or_create_peak(current, cmd["target"])

    for i in range(steps):
        proposed = propose_step(current, current_idx, sign, step)

        before_png = os.path.join(out_dir, f"step_{i:03d}_before.png")
        after_png = os.path.join(out_dir, f"step_{i:03d}_after.png")
        img_before = save_png(current, before_png)
        img_after = save_png(proposed, after_png)
        feats_before, feats_after = features(img_before), features(img_after)

        obj_verdict = objective(current, proposed, cmd)
        if use_human:
            human_verdict = human(before_png, after_png)
            log_preference(session_id, cmd_text, cmd, current, proposed,
                            feats_before, feats_after, before_png, after_png,
                            human_verdict, obj_verdict)
            verdict = human_verdict
        else:
            verdict = obj_verdict

        print_step(cmd, current, proposed, feats_before, feats_after, verdict)
        log_step(cmd, current, proposed, feats_before, feats_after, verdict)

        if verdict == 1:
            current = proposed
        step = resize_step(step, accepted=(verdict == 1))
        if step < 0.01:
            print(f"[mvp] step size below threshold after {i + 1} steps, stopping")
            break

    return current


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd")
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--wav")
    ap.add_argument("--parser", choices=["rule", "llm"], default="rule")
    ap.add_argument("--llm-model", default="qwen2.5:7b")
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--dataset", default="synthetic",
                     help="synthetic, or a real CT dataset name (see datasets.DATASETS)")
    args = ap.parse_args()

    global _DATASET_NAME
    _DATASET_NAME = args.dataset

    os.makedirs("out", exist_ok=True)
    params = load_state()
    session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    cmd_text = args.cmd
    if args.listen:
        import asr
        cmd_text = asr.transcribe_mic(lang=args.lang, model_size=args.model_size)
    elif args.wav:
        import asr
        cmd_text = asr.transcribe_file(args.wav, lang=args.lang, model_size=args.model_size)

    if not cmd_text:
        save_png(params, "out/start.png")
        print("[mvp] rendered current state -> out/start.png")
        return

    try:
        cmd = parse_command(cmd_text, parser=args.parser, model=args.llm_model)
    except ValueError as exc:
        print(f"[mvp] could not parse command: {exc}")
        raise SystemExit(1)
    print("parsed command:", cmd)

    if args.learn and cmd.get("attribute") == "opacity" and cmd.get("direction") in ("increase", "decrease"):
        new_params = hill_climb(cmd_text, cmd, params, args.steps, args.human, session_id)
    else:
        before = params
        new_params = apply_command(cmd, params)
        img_before = save_png(before, "out/before.png")
        img_after = save_png(new_params, "out/after.png")
        feats_before, feats_after = features(img_before), features(img_after)
        verdict = objective(before, new_params, cmd)
        print_step(cmd, before, new_params, feats_before, feats_after, verdict)
        log_step(cmd, before, new_params, feats_before, feats_after, verdict)

    save_state(new_params)


if __name__ == "__main__":
    main()
