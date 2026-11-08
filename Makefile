# Clone, then:  make system-deps && make setup && make scans && make run
# (macOS, Ubuntu/Debian, Fedora, Arch). `make help` lists every target.

# The first interpreter in the supported range (3.12-3.14): torch, vtk and
# matplotlib publish no wheels for 3.15 yet, which Fedora 45 ships as python3.
PYTHON ?= $(shell for p in python3.14 python3.13 python3.12 python3; do \
	command -v $$p >/dev/null 2>&1 && $$p -c 'import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 14))' 2>/dev/null \
	&& { echo $$p; break; }; done)
VENV   := .venv
PY     := $(VENV)/bin/python
PORT   ?= 8000
LLM    ?= qwen2.5:7b
# Linux only: make setup CPU=1 installs the CPU build of torch (~200 MB instead
# of ~3 GB of CUDA libraries) on a machine without an NVIDIA GPU.
CPU    ?=
TORCH  := $(shell sed -n 's/^torch==//p' requirements.txt)

# This machine's LAN address, for the headset (override: make certs IP=192.168.1.20)
IP ?= $(shell { ipconfig getifaddr en0 || ipconfig getifaddr en1 \
	|| ip -4 route get 1.1.1.1 | awk '{for (i = 1; i < NF; i++) if ($$i == "src") print $$(i + 1)}'; } 2>/dev/null | head -n1)

.PHONY: help system-deps setup scans scans-candidates scans-all llm run certs run-vr test

help:
	@echo "make system-deps  system packages: Python 3.12-3.14, OpenGL, mkcert (apt, dnf, pacman or brew; uses sudo)"
	@echo "make setup        virtualenv in .venv + pinned dependencies (Python: $(or $(PYTHON),none found))"
	@echo "                  (Linux without an NVIDIA GPU: make setup CPU=1)"
	@echo "make scans        download the 7 demo patients (data/demo_scans.json) from Zenodo"
	@echo "make scans-candidates  the 24 patients the demo set was picked from (~1 GB); view them: make run SCANS=all"
	@echo "make scans-all    all 100 scans, for retraining (~3 GB)"
	@echo "make llm          pull the local LLM ($(LLM)) for free-form voice commands (needs Ollama)"
	@echo "make run          viewer on http://127.0.0.1:$(PORT)"
	@echo "make certs        HTTPS certificate for this machine's LAN address ($(IP)) with mkcert"
	@echo "make run-vr       viewer on https://$(IP):$(PORT) for a headset on the same network"
	@echo "make test         the tests (~30 s, synthetic scans, no download needed)"

system-deps:
	@if command -v apt-get >/dev/null 2>&1; then \
		set -x; sudo apt-get update; sudo apt-get install -y python3 python3-venv make curl libgl1 libegl1 mkcert; \
	elif command -v dnf >/dev/null 2>&1; then \
		set -x; sudo dnf install -y python3.14 make curl mesa-libGL mesa-libEGL mesa-dri-drivers mkcert; \
	elif command -v pacman >/dev/null 2>&1; then \
		set -x; sudo pacman -S --needed --noconfirm python make curl mesa libglvnd mkcert; \
	elif command -v brew >/dev/null 2>&1; then \
		set -x; brew install python@3.13 mkcert; \
	else \
		echo "unknown package manager: install Python 3.12-3.14, make, OpenGL/EGL (Mesa) and mkcert yourself"; exit 1; \
	fi

$(PY):
	@test -n "$(PYTHON)" || { echo "no Python 3.12-3.14 found (python3.14, python3.13, python3.12 or python3): make system-deps"; exit 1; }
	@echo "using $(PYTHON) ($$($(PYTHON) --version))"
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

setup: $(PY)
ifneq ($(CPU),)
	$(PY) -m pip install torch==$(TORCH) --index-url https://download.pytorch.org/whl/cpu
endif
	$(PY) -m pip install -r requirements.txt
	@echo; echo "next: make scans && make run"

scans: $(PY)
	$(PY) -u -m tools.fetch_scans --demo --splits

scans-candidates: $(PY)
	$(PY) -u -m tools.fetch_scans --candidates --splits

scans-all: $(PY)
	$(PY) -u -m tools.fetch_scans --splits all

llm:
	@command -v ollama >/dev/null || { echo "Ollama is not installed: https://ollama.com/download (Arch: sudo pacman -S ollama)"; exit 1; }
	ollama pull $(LLM)

# SCANS=all lists every downloaded patient, not only the demo set
run: $(PY)
	SCANS=$(or $(SCANS),demo) $(PY) server.py --port $(PORT)

certs:
	@command -v mkcert >/dev/null || { echo "mkcert is not installed: make system-deps"; exit 1; }
	@test -n "$(IP)" || { echo "no LAN address found; pass one: make certs IP=..."; exit 1; }
	mkdir -p certs
	mkcert -cert-file certs/lan.pem -key-file certs/lan-key.pem $(IP) localhost 127.0.0.1

run-vr: $(PY)
	@test -f certs/lan.pem || { echo "no certificate yet: make certs"; exit 1; }
	@echo "headset browser: https://$(IP):$(PORT)/"
	SCANS=$(or $(SCANS),demo) $(PY) server.py --host 0.0.0.0 --port $(PORT) --ssl-certfile certs/lan.pem --ssl-keyfile certs/lan-key.pem

test: $(PY)
	$(PY) -m pytest -q
