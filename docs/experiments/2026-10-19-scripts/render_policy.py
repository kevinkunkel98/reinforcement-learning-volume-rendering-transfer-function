"""Render instructions answered by the viewer's policy on one volume: start, pure policy, policy + search."""
import io, os, sys, json
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, ".")
import server, goals
volume, out = sys.argv[1], sys.argv[2]
instructions = sys.argv[3].split("|")
server.IMAGE_DIR = os.path.join(out, "ui_images")
s = server.Session(os.path.join(out, "session.json"))
s.switch_dataset(volume)
model = s.model_for_volume(volume)
start = np.array(s.history[s.cursor]["params"])
start_agg, _ = s.instruction_for(server.parse_command_with_meta("more bone", parser="rule")[0], model, start, None)

def img(params):
    return server._render_image_b64(params, s.history[0].get("camera", server.DEFAULT_CAMERA))[1]

def score(text, params):
    cmd, _ = server.parse_command_with_meta(text, parser="rule")
    _, inst = s.instruction_for(cmd, model, start, None)
    final = goals.aggregate(model.features(params))
    return goals.attainment(inst["goal"], start_agg, final, weights=inst.get("weights")), inst

start_img = img(start)
rows, report = [], []
for text in instructions:
    results = {}
    for label, refine in (("policy", 0), ("policy+200", 200)):
        s.policy_refine_evaluations = refine
        params, _ = s._run_policy(server.parse_command_with_meta(text, parser="rule")[0], start)
        att, inst = score(text, params)
        results[label] = (params, att)
    vis = lambda p: {c: round(v, 3) for c, v in goals.aggregate(model.features(p))["vis"].items()}
    report.append({"text": text, "kind": inst["kind"], "start_vis": vis(start),
                   **{k: {"attainment": round(a, 2), "vis": vis(p)} for k, (p, a) in results.items()}})
    rows.append((text, [("start", start_img)] + [(f"{k}  att {a:+.2f}", img(p)) for k, (p, a) in results.items()]))

w, h = start_img.shape[1], start_img.shape[0]
sheet = Image.new("RGB", (3 * w, len(rows) * (h + 24)), "white")
d = ImageDraw.Draw(sheet)
for r, (text, cells) in enumerate(rows):
    y = r * (h + 24)
    for c, (label, arr) in enumerate(cells):
        sheet.paste(Image.fromarray(arr[..., :3]), (c * w, y + 24))
        d.text((c * w + 4, y + 4), (text + "  |  " if c == 0 else "") + label, fill="black")
sheet.save(os.path.join(out, f"{volume}.png"))
json.dump(report, open(os.path.join(out, f"{volume}.json"), "w"), indent=1)
for row in report: print(row["text"], "|", row["kind"], "|", row["policy"]["attainment"], row["policy+200"]["attainment"])
print(w, h)
