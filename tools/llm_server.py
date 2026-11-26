"""Run the LLM parser's model in a llama-server with n-gram speculative decoding.

Ollama starts llama-server itself and passes no speculative-decoding flags.
This starts a second llama-server -- the binary Ollama ships, on the model
file Ollama downloaded -- with `--spec-type ngram-simple`: drafted tokens come
from n-grams of the context (the prompt's JSON examples), and greedy decoding
verifies every one, so the answers are those of plain decoding. On qwen2.5:7b
(RTX 5080, 2026-11-26) the median parse went from ~185 to ~103 ms with
byte-identical answers on all 185 parser-eval phrases. `commands.py` uses it
whenever it serves the requested model (LLAMA_SERVER_URL), Ollama otherwise.

Linux only by default (Ollama's CUDA layout, tested); elsewhere set
LLAMA_SERVER=1 to try. LLAMA_SERVER=0 never starts it.

    python -m tools.llm_server ensure   # start unless running or unavailable (make llm, make run)
    python -m tools.llm_server status
    python -m tools.llm_server stop
"""
import argparse
import glob
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import commands  # noqa: E402

PID_PATH = "out/llama_server.pid"
LOG_PATH = "out/llama_server.log"
START_TIMEOUT_S = 180
# The context Ollama gives the model, and its batch sizes (Ollama's runner
# command line); the system prompt is ~1850 tokens.
SERVER_ARGS = ["-c", "4096", "-np", "1", "-ngl", "99", "--flash-attn", "auto", "-b", "512", "-ub", "512",
               "--chat-template", "chatml", "--no-jinja", "--no-webui"]
SPECULATIVE_ARGS = ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "3",
                    "--spec-ngram-simple-size-m", "16"]


def _port() -> int:
    match = re.search(r":(\d+)", commands.LLAMA_SERVER_URL.split("//", 1)[-1])
    return int(match.group(1)) if match else 80


def _enabled():
    """None when this machine should run it, else why not."""
    flag = os.environ.get("LLAMA_SERVER")
    if flag == "0":
        return "LLAMA_SERVER=0"
    if flag != "1" and platform.system() != "Linux":
        return f"not on {platform.system()} unless LLAMA_SERVER=1 (tested on Linux only); Ollama answers"
    if commands.LLM_BACKEND == "ollama":
        return "LLM_BACKEND=ollama"
    return None


def _ollama_lib_dirs() -> list:
    """Where Ollama installs keep llama-server and the ggml backends."""
    dirs = []
    ollama = shutil.which("ollama")
    if ollama:
        prefix = os.path.dirname(os.path.dirname(os.path.realpath(ollama)))
        dirs.append(os.path.join(prefix, "lib", "ollama"))
    dirs += [os.path.expanduser("~/.local/lib/ollama"), "/usr/lib/ollama", "/usr/local/lib/ollama"]
    return list(dict.fromkeys(d for d in dirs if os.path.isdir(d)))


def find_llama_server():
    """(llama-server path, Ollama lib dir or None) -- LLAMA_SERVER_BIN, an
    Ollama install's, or one on PATH (llama.cpp's own)."""
    explicit = os.environ.get("LLAMA_SERVER_BIN")
    if explicit:
        return explicit, None
    for directory in _ollama_lib_dirs():
        candidate = os.path.join(directory, "llama-server")
        if os.access(candidate, os.X_OK):
            return candidate, directory
    found = shutil.which("llama-server")
    return (found, None) if found else (None, None)


def model_path(model: str):
    """The GGUF file Ollama keeps for `model` (`ollama show --modelfile`)."""
    try:
        out = subprocess.run(["ollama", "show", "--modelfile", model], capture_output=True, text=True,
                             timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if line.startswith("FROM ") and os.path.isfile(line[5:].strip()):
            return line[5:].strip()
    return None


def _environment(lib_dir):
    """Ollama's llama-server finds CUDA through GGML_BACKEND_PATH and
    LD_LIBRARY_PATH, which Ollama sets when it starts one (read from its
    process, 2026-11-26); without them it silently runs on the CPU."""
    env = dict(os.environ)
    if not lib_dir:
        return env
    # Ollama's own builds: cuda_v12/, cuda_v13/ (newest last); distribution
    # builds against the system CUDA (e.g. Arch's ollama-cuda) may keep it flat.
    backends = sorted(glob.glob(os.path.join(lib_dir, "cuda_v*", "libggml-cuda.so")),
                      key=lambda p: int(re.search(r"cuda_v(\d+)", p).group(1)))
    if not backends:
        backends = glob.glob(os.path.join(lib_dir, "libggml-cuda.so"))
    if backends and shutil.which("nvidia-smi"):
        cuda_dir = os.path.dirname(backends[-1])
        env["GGML_BACKEND_PATH"] = backends[-1]
        env["LD_LIBRARY_PATH"] = ":".join(p for p in (lib_dir, cuda_dir, env.get("LD_LIBRARY_PATH")) if p)
    else:
        env["LD_LIBRARY_PATH"] = ":".join(p for p in (lib_dir, env.get("LD_LIBRARY_PATH")) if p)
    return env


def _supports_speculation(binary, env) -> bool:
    try:
        help_text = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30, env=env)
    except (OSError, subprocess.SubprocessError):
        return False
    return "ngram-simple" in help_text.stdout + help_text.stderr


def _running_pid():
    try:
        with open(PID_PATH) as stream:
            pid = int(stream.read().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def _healthy() -> bool:
    try:
        with urlopen(Request(f"{commands.LLAMA_SERVER_URL}/health"), timeout=1) as resp:
            return resp.status == 200
    except (URLError, OSError):
        return False


def _unload_from_ollama(model: str) -> None:
    """Ollama keeps its own copy of the model on the GPU for 30 min after the
    last request; with llama-server answering it only takes memory."""
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    try:
        with urlopen(Request(f"{commands.OLLAMA_HOST}/api/generate", data=body,
                             headers={"Content-Type": "application/json"}), timeout=10) as resp:
            resp.read()
    except (URLError, OSError):
        pass


def ensure(model: str) -> int:
    reason = _enabled()
    if reason:
        print(f"[llm-server] not started: {reason}")
        return 0
    if commands.llama_server_model(force=True) == model:
        print(f"[llm-server] running at {commands.LLAMA_SERVER_URL} ({model})")
        return 0
    if _healthy():
        print(f"[llm-server] something else answers at {commands.LLAMA_SERVER_URL}; set LLAMA_SERVER_URL")
        return 0
    binary, lib_dir = find_llama_server()
    if not binary:
        print("[llm-server] no llama-server found (Ollama's or llama.cpp's); Ollama answers")
        return 0
    path = model_path(model)
    if not path:
        print(f"[llm-server] Ollama has no file for {model} (make llm pulls it); Ollama answers")
        return 0
    env = _environment(lib_dir)
    args = [binary, "--model", path, "--alias", model, "--host", "127.0.0.1", "--port", str(_port()), *SERVER_ARGS]
    if _supports_speculation(binary, env):
        args += SPECULATIVE_ARGS
    else:
        print("[llm-server] this llama-server has no ngram-simple speculation; starting without it")
    os.makedirs(os.path.dirname(PID_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)   # outlives `make run`'s Ctrl+C
    with open(PID_PATH, "w") as stream:
        stream.write(str(process.pid))
    print(f"[llm-server] starting {model} on {commands.LLAMA_SERVER_URL} (log: {LOG_PATH}) ...")
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(f"[llm-server] exited with code {process.returncode}; see {LOG_PATH}. Ollama answers")
            return 0
        if _healthy():
            with open(LOG_PATH) as stream:
                on_cpu = "no usable GPU found" in stream.read()
            if on_cpu and shutil.which("nvidia-smi"):
                print("[llm-server] WARNING: running on the CPU (slower than Ollama); "
                      f"see {LOG_PATH}, or `python -m tools.llm_server stop`")
            speculative = SPECULATIVE_ARGS[0] in args
            print(f"[llm-server] ready: {model}" + (", n-gram speculative decoding" if speculative else ""))
            _unload_from_ollama(model)
            return 0
        time.sleep(0.5)
    print(f"[llm-server] not ready after {START_TIMEOUT_S} s; see {LOG_PATH}")
    return 0


def stop() -> int:
    pid = _running_pid()
    if pid is None:
        print("[llm-server] not running")
        return 0
    os.kill(pid, signal.SIGTERM)
    print(f"[llm-server] stopped (pid {pid})")
    return 0


def status(model: str) -> int:
    served = commands.llama_server_model(force=True)
    print(f"[llm-server] {'serving ' + served if served else 'not running'} at {commands.LLAMA_SERVER_URL}")
    return 0 if served == model else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["ensure", "status", "stop"])
    ap.add_argument("--model", default=commands.DEFAULT_LLM_MODEL)
    args = ap.parse_args(argv)
    if args.action == "ensure":
        return ensure(args.model)
    if args.action == "stop":
        return stop()
    return status(args.model)


if __name__ == "__main__":
    sys.exit(main())
