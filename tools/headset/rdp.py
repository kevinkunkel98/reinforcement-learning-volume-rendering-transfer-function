"""Minimal Firefox Remote Debugging Protocol client (length-prefixed JSON)."""
import json, socket, sys

class RDP:
    def __init__(self, port=6000, timeout=15):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.buf = b""
        self.hello = self.recv()

    def recv(self):
        while b":" not in self.buf:
            chunk = self.s.recv(65536)
            if not chunk: raise EOFError("closed")
            self.buf += chunk
        n, rest = self.buf.split(b":", 1)
        n = int(n)
        while len(rest) < n:
            chunk = self.s.recv(65536)
            if not chunk: raise EOFError("closed")
            rest += chunk
        self.buf = rest[n:]
        return json.loads(rest[:n])

    def send(self, msg):
        data = json.dumps(msg).encode()
        self.s.sendall(str(len(data)).encode() + b":" + data)

    def request(self, to, type_, **kw):
        self.send({"to": to, "type": type_, **kw})
        while True:
            m = self.recv()
            if m.get("from") == to and "type" not in m or m.get("from") == to and m.get("error"):
                return m
            if m.get("from") == to and m.get("type") not in (None,) and type_ in ("listTabs", "getTarget"):
                return m

if __name__ == "__main__":
    r = RDP()
    print("hello:", json.dumps(r.hello)[:300])
    tabs = r.request("root", "listTabs")
    print(json.dumps(tabs)[:1500])
