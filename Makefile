# Clone, then:  make setup && make scans && make run   (macOS and Linux)
# `make help` lists every target.

PYTHON ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python
PORT   ?= 8000
SPLITS ?= test
LLM    ?= qwen2.5:7b
# Linux only: make setup CPU=1 installs the CPU build of torch (~200 MB instead
# of ~3 GB of CUDA libraries) on a machine without an NVIDIA GPU.
CPU    ?=
TORCH  := $(shell sed -n 's/^torch==//p' requirements.txt)

# This machine's LAN address, for the headset (override: make certs IP=192.168.1.20)
IP ?= $(shell (ipconfig getifaddr en0 || ipconfig getifaddr en1 || hostname -I | awk '{print $$1}') 2>/dev/null | head -n1)

.PHONY: help setup scans scans-all llm run certs run-vr test

help:
	@echo "make setup      virtualenv in .venv + pinned dependencies ($(PYTHON), needs 3.12-3.14)"
	@echo "                (Linux without an NVIDIA GPU: make setup CPU=1)"
	@echo "make scans      download the 6 test patients from Zenodo (~230 MB; SPLITS='test val train' for more)"
	@echo "make scans-all  all 100 scans, for retraining (~3 GB)"
	@echo "make llm        pull the local LLM ($(LLM)) for free-form voice commands (needs Ollama)"
	@echo "make run        viewer on http://127.0.0.1:$(PORT)"
	@echo "make certs      HTTPS certificate for this machine's LAN address ($(IP)) with mkcert"
	@echo "make run-vr     viewer on https://$(IP):$(PORT) for a headset on the same network"
	@echo "make test       fast test suite"

$(PY):
	$(PYTHON) -c 'import sys; v = sys.version_info[:2]; sys.exit(0 if (3, 12) <= v <= (3, 14) else f"Python 3.12-3.14 needed, {sys.executable} is {v[0]}.{v[1]} -- try make setup PYTHON=python3.12")'
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

setup: $(PY)
ifneq ($(CPU),)
	$(PY) -m pip install torch==$(TORCH) --index-url https://download.pytorch.org/whl/cpu
endif
	$(PY) -m pip install -r requirements.txt
	@echo; echo "next: make scans && make run"

scans: $(PY)
	$(PY) -u -m tools.fetch_scans --splits $(SPLITS)

scans-all: $(PY)
	$(PY) -u -m tools.fetch_scans --splits all

llm:
	@command -v ollama >/dev/null || { echo "Ollama is not installed: https://ollama.com/download"; exit 1; }
	ollama pull $(LLM)

run: $(PY)
	$(PY) server.py --port $(PORT)

certs:
	@command -v mkcert >/dev/null || { echo "mkcert is not installed: brew install mkcert / sudo apt install mkcert"; exit 1; }
	@test -n "$(IP)" || { echo "no LAN address found; pass one: make certs IP=..."; exit 1; }
	mkdir -p certs
	mkcert -cert-file certs/lan.pem -key-file certs/lan-key.pem $(IP) localhost 127.0.0.1

run-vr: $(PY)
	@test -f certs/lan.pem || { echo "no certificate yet: make certs"; exit 1; }
	@echo "headset browser: https://$(IP):$(PORT)/"
	$(PY) server.py --host 0.0.0.0 --port $(PORT) --ssl-certfile certs/lan.pem --ssl-keyfile certs/lan-key.pem

test: $(PY)
	$(PY) -m pytest -q -m "not slow"
