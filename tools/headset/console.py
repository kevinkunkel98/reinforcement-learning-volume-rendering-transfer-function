"""Usage: console.py <url-substring> [n] -- prints the last n cached console messages/errors."""
import json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rdp import RDP
from page import target_for

r = RDP()
tab, frame = target_for(r, sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
r.send({"to": frame["consoleActor"], "type": "getCachedMessages", "messageTypes": ["PageError", "ConsoleAPI"]})
while True:
    m = r.recv()
    if m.get("from") == frame["consoleActor"] and "messages" in m:
        break
for msg in m["messages"][-n:]:
    if "pageError" in msg or msg.get("type") == "pageError":
        e = msg.get("pageError", msg)
        print("ERROR", e.get("errorMessage"), e.get("sourceName", "").rsplit("/", 1)[-1], e.get("lineNumber"))
    else:
        c = msg.get("message", msg)
        args = c.get("arguments", [])
        print(c.get("level"), " ".join(a if isinstance(a, str) else json.dumps(a)[:80] for a in args)[:200])
