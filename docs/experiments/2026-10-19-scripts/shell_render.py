"""Start view + policy+200 answers with the unlabelled row's opacity scaled (no retrain)."""
import os, sys, json; sys.path.insert(0, ".")
import numpy as np
from PIL import Image, ImageDraw
import transfer, render as R
orig = transfer.label_scoped_tables
SCALE = [1.0]
def patched(params, hu, classes=transfer.CANONICAL_CLASSES, mode=transfer.DEFAULT_TRANSFER_MODE):
    rows = orig(params, hu, classes, mode)
    if mode == "label_scoped":
        rows[0] = (rows[0][0] * SCALE[0], rows[0][1])
    return rows
transfer.label_scoped_tables = patched; R.label_scoped_tables = patched
import server, goals
volume, out = sys.argv[1], sys.argv[2]
texts = sys.argv[3].split("|")
scales = [float(x) for x in sys.argv[4].split(",")]
server.IMAGE_DIR = os.path.join(out, "ui_images")
s = server.Session(os.path.join(out, "session.json"))
s.switch_dataset(volume)
model = s.model_for_volume(volume)
start = np.array(s.history[s.cursor]["params"])
cam = s.history[0].get("camera", server.DEFAULT_CAMERA)
img = lambda p: server._render_image_b64(p, cam)[1]
rows, report = [], []
for sc in scales:
    SCALE[0] = sc
    cells = [(f"shell x{sc}: start", img(start))]
    for text in texts:
        cmd = server.parse_command_with_meta(text, parser="rule")[0]
        start_agg, inst = s.instruction_for(cmd, model, start, None)
        params, _ = s._run_policy(cmd, start)
        fin = goals.aggregate(model.features(params))
        att = goals.attainment(inst["goal"], start_agg, fin, weights=inst.get("weights"))
        cells.append((f"{text} att {att:+.2f}", img(params)))
        report.append({"scale": sc, "text": text, "att": round(att, 2),
                       "start": {k: round(v, 3) for k, v in start_agg["vis"].items()},
                       "final": {k: round(v, 3) for k, v in fin["vis"].items()}})
        print(sc, text, round(att, 2), flush=True)
    rows.append(cells)
h, w = rows[0][0][1].shape[:2]
sheet = Image.new("RGB", (len(rows[0]) * w, len(rows) * (h + 20)), "white")
d = ImageDraw.Draw(sheet)
for r, cells in enumerate(rows):
    for c, (label, arr) in enumerate(cells):
        sheet.paste(Image.fromarray(arr[..., :3]), (c * w, r * (h + 20) + 20))
        d.text((c * w + 4, r * (h + 20) + 4), label, fill="black")
sheet.save(os.path.join(out, f"{volume}.png"))
json.dump(report, open(os.path.join(out, f"{volume}.json"), "w"), indent=1)
