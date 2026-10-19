"""Render vis_eval episodes as a contact sheet: start | policy | policy + N search.

    render_episodes.py CKPT SPLIT OUT.png i [i ...] [--refine N]

Each row is labelled with the instruction, attainment and visible pp, and the
largest per-class visibility changes, so a score can be checked against what
the render shows.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

torch.set_num_threads(2)
sys.path.insert(0, ".")
import goals  # noqa: E402
import server  # noqa: E402
from rl import vis_eval as ve  # noqa: E402
from rl.baselines import hill_climb  # noqa: E402

args = [a for a in sys.argv[1:]]
refine = 200
if "--refine" in args:
    k = args.index("--refine")
    refine = int(args[k + 1])
    del args[k:k + 2]
ckpt, split, out, idx = args[0], args[1], args[2], [int(a) for a in args[3:]]
meta = ve.load_policy_metadata(ckpt)
episodes = ve.fixed_episodes(split, 300, seed=0, formulation="one_shot", policy_metadata=meta)
policy = ve._load_sac(ckpt)
scratch = os.path.join(os.path.dirname(out) or ".", "_render_session")
os.makedirs(scratch, exist_ok=True)
server.IMAGE_DIR = os.path.join(scratch, "ui_images")
session = server.Session(os.path.join(scratch, "session.json"))
current = None
X0, X1, SCALE = 256, 768, 0.5


def image(params):
    arr = server._render_image_b64(params, session.history[0].get("camera", server.DEFAULT_CAMERA))[1]
    cell = Image.fromarray(arr[..., :3]).crop((X0, 0, X1, arr.shape[0]))
    return cell.resize((int(cell.width * SCALE), int(cell.height * SCALE)))


def changes(start, final, n=4):
    delta = {c: final[c] - start[c] for c in start}
    top = sorted(delta, key=lambda c: -abs(delta[c]))[:n]
    return " ".join(f"{c} {100 * start[c]:.1f}>{100 * final[c]:.1f}" for c in top)


rows = []
for i in idx:
    ep = episodes[i]
    if ep["volume"] != current:
        session.switch_dataset(ep["volume"])
        current = ep["volume"]
    env, obs, _ = ve._frozen_one_shot_env(ep["volume"], ep["start_params"], ep["instruction"],
                                          policy_metadata=ve._episode_policy_metadata(ep, meta))
    action, _ = policy.predict(obs, deterministic=True)
    env.step(action)
    proposal, start = env._params.copy(), env._start_params.copy()
    inst = ep["instruction"]

    def scored(params):
        final = goals.aggregate(ve._features(env._model, params))
        pp = goals.visible_change(inst["goal"], env._start_agg, final, inst.get("weights"))
        return env._attainment(final), pp, final["vis"]

    att_p, pp_p, vis_p = scored(proposal)
    warm = hill_climb(env._model, proposal, inst, evaluations=refine,
                      initial_step=ve.REFINE_INITIAL_STEP, reference_params=start)
    att_w, pp_w, vis_w = scored(warm)
    if att_w <= att_p:
        warm, att_w, pp_w, vis_w = proposal, att_p, pp_p, vis_p
    s_vis = env._start_agg["vis"]
    fmt_pp = lambda pp: "n/a" if pp is None else f"{pp:+.1f}pp"
    label = [f"#{i} {ep['volume']}  \"{inst.get('text', inst['kind'])}\" ({inst['kind']})",
             f"policy {att_p:+.2f} {fmt_pp(pp_p)}: {changes(s_vis, vis_p)}",
             f"+{refine} {att_w:+.2f} {fmt_pp(pp_w)}: {changes(s_vis, vis_w)}"]
    rows.append((label, [image(start), image(proposal), image(warm)]))
    print(" | ".join(label), flush=True)

w, h = rows[0][1][0].size
head = 46
sheet = Image.new("RGB", (3 * w, len(rows) * (h + head)), "white")
draw = ImageDraw.Draw(sheet)
for r, (label, cells) in enumerate(rows):
    y = r * (h + head)
    for line, text in enumerate(label):
        draw.text((4, y + 2 + 14 * line), text, fill="black")
    for c, cell in enumerate(cells):
        sheet.paste(cell, (c * w, y + head))
sheet.save(out)
