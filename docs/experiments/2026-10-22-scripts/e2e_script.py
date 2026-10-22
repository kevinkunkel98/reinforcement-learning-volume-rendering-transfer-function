"""End to end over data/asr_script.txt: TTS (two voices) -> ASR (lexicon prompt,
hallucination filter) -> hybrid parser -> server's command path (policy +
planning + keep repair, exact for non-goals) -> render. Every command starts
from the task's start TF on ts_s0477, independently.

Run from the repo root:  .venv/bin/python <this file> <out_dir>
"""
import base64
import json
import os
import subprocess
import sys
import time

OUT = sys.argv[1]
os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
os.makedirs(os.path.join(OUT, "wav"), exist_ok=True)
os.environ["UI_SESSION_PATH"] = os.path.join(OUT, "e2e_session.json")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

import asr  # noqa: E402
import commands  # noqa: E402
import goals  # noqa: E402
import lexicon  # noqa: E402
import server  # noqa: E402
from anatomy_layers import default_layers  # noqa: E402

commands.LLM_LOG_PATH = os.path.join(OUT, "llm_requests.jsonl")
commands.UTTERANCE_LOG_PATH = os.path.join(OUT, "utterances.jsonl")

VOICES = {"en": "Eddy (Englisch (USA))", "de": "Anna (Deutsch (Deutschland))"}
VOLUME = "ts_s0477"

server.set_dataset(VOLUME)
session = server.session
model = session.model_for_volume(VOLUME)
layers = default_layers()
start = server.initial_params(session.task_version)
camera = json.load(open("out/ui_session.json"))["history"][-1].get("camera", dict(server.DEFAULT_CAMERA))
start_agg = goals.aggregate(goals.features(model, start, layers), layers)
prompt = lexicon.get()["asr_prompt"]
lines = [l.strip() for l in open("data/asr_script.txt") if l.strip()]


def tts(text, voice, path):
    aiff = path[:-4] + ".aiff"
    subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, path], check=True)
    os.remove(aiff)


def drift(agg, mentioned):
    out = {}
    for c in goals.GOAL_CLASSES:
        if c in mentioned or c not in agg["vis"] or agg["vis"][c] is None:
            continue
        dv = np.log10(max(agg["vis"][c], 1e-6)) - np.log10(max(start_agg["vis"][c], 1e-6))
        db = agg["bright"].get(c, 0.0) - start_agg["bright"].get(c, 0.0)
        out[c] = (round(float(dv), 2), round(float(db), 2))
    out["other"] = (round(float(np.log10(max(agg["vis"]["other"], 1e-6))
                                - np.log10(max(start_agg["vis"]["other"], 1e-6))), 2), None)
    return out


# Warm models (ASR, LLM, policy) so the first row's timings are not cold starts.
tts("show only bones", VOICES["en"], os.path.join(OUT, "wav", "warm.wav"))
asr._transcribe_path(os.path.join(OUT, "wav", "warm.wav"), "en", asr.ASR_MODEL, initial_prompt=prompt)
commands.parse_command_hybrid("everything except the bones should vanish")
session._run_policy(commands.parse_command_rule("add liver"), start)

rows = []
for i, line in enumerate(lines):
    for voice_key, voice in VOICES.items():
        row = {"i": i, "line": line, "voice": voice_key}
        wav = os.path.join(OUT, "wav", f"{i:02d}_{voice_key}.wav")
        tts(line, voice, wav)
        t = time.perf_counter()
        heard = asr._transcribe_path(wav, "en", asr.ASR_MODEL, initial_prompt=prompt)
        row["asr_ms"] = round((time.perf_counter() - t) * 1000)
        row["heard"] = heard
        if asr.is_hallucination(heard):
            row["outcome"] = "filtered as hallucination"
            rows.append(row)
            print(json.dumps(row))
            continue
        try:
            cmd, meta = commands.parse_command_hybrid(heard)
        except ValueError as exc:
            row["outcome"] = f"rejected: {exc}"
            rows.append(row)
            print(json.dumps(row))
            continue
        row.update(cmd=cmd, path=meta["path"], parse_ms=meta["parse_ms"], uncovered=meta["uncovered"],
                   fallback=meta["fallback"])
        t = time.perf_counter()
        params, mode, goal_text = start, None, None
        if "camera" in cmd:
            mode = "camera"
        else:
            try:
                params, goal_text = session._run_policy(cmd, start)
                mode = "policy+keep"
            except ValueError as exc:
                params = commands.apply_command(cmd, start, reset_params=start)
                mode = f"exact ({str(exc)[:60]})"
        row["apply_ms"] = round((time.perf_counter() - t) * 1000)
        row["mode"], row["goal"] = mode, goal_text
        params = np.asarray(params, dtype=np.float64)
        agg = goals.aggregate(goals.features(model, params, layers), layers)
        if goal_text is not None:
            _a, instruction = session.instruction_for(cmd, model, start, layers)
            row["attainment"] = round(float(goals.attainment(
                instruction["goal"], start_agg, agg,
                **({} if instruction.get("weights") is None else {"weights": instruction["weights"]}))), 2)
        subs = cmd.get("compound", [cmd])
        mentioned = set()
        for s in subs:
            tg = s.get("target")
            mentioned |= set(tg if isinstance(tg, list) else [tg] if tg else [])
        row["drift"] = drift(agg, mentioned)
        if voice_key == "en":
            b64, _img, png = server._render_image_b64(params, camera, layers=layers)
            open(os.path.join(OUT, "img", f"{i:02d}.png"), "wb").write(png)
        rows.append(row)
        print(json.dumps(row))

json.dump(rows, open(os.path.join(OUT, "e2e.json"), "w"), indent=1)
_b, _i, png = server._render_image_b64(start, camera, layers=layers)
open(os.path.join(OUT, "img", "start.png"), "wb").write(png)
