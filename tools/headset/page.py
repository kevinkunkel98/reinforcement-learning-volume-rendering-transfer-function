"""Usage: page.py <url-substring> <js-expression>  -- evaluates JS in the matching headset tab."""
import json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rdp import RDP

def target_for(r, needle):
    r.send({"to": "root", "type": "listTabs"})
    tabs = r.recv()["tabs"]
    tab = next(t for t in tabs if needle in t["url"])
    r.send({"to": tab["actor"], "type": "getTarget"})
    while True:
        m = r.recv()
        if m.get("from") == tab["actor"] and "frame" in m:
            return tab, m["frame"]

def evaluate(r, console, text):
    r.send({"to": console, "type": "evaluateJSAsync", "text": text})
    while True:
        m = r.recv()
        if m.get("type") == "evaluationResult" and m.get("from") == console:
            return m

if __name__ == "__main__":
    r = RDP()
    tab, frame = target_for(r, sys.argv[1])
    res = evaluate(r, frame["consoleActor"], sys.argv[2])
    if res.get("exception"):
        print("EXCEPTION:", res.get("exceptionMessage"))
    else:
        v = res.get("result")
        print(v if not isinstance(v, dict) else json.dumps(v)[:2000])
