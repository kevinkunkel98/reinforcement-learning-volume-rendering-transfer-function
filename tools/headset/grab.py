"""Usage: grab.py <url-substring> <js-expression-returning-dataURL> <out.png>"""
import base64, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rdp import RDP
from page import target_for, evaluate

r = RDP()
tab, frame = target_for(r, sys.argv[1])
res = evaluate(r, frame["consoleActor"], sys.argv[2])
v = res.get("result")
if isinstance(v, dict) and v.get("type") == "longString":
    r.send({"to": v["actor"], "type": "substring", "start": 0, "end": v["length"]})
    while True:
        m = r.recv()
        if m.get("from") == v["actor"]:
            v = m["substring"]; break
if not isinstance(v, str):
    sys.exit(f"unexpected result: {str(res)[:300]}")
open(sys.argv[3], "wb").write(base64.b64decode(v.split(",", 1)[1]))
print("saved", sys.argv[3])
