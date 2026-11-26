# Clone, then:  make system-deps && make setup && make scans && make llm && make run
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
# Linux + NVIDIA GPU only (see setup): the cuBLAS faster-whisper's CTranslate2 build links.
CUBLAS12 := 12.9.3.13

# This machine's LAN address, for the headset (override: make certs IP=192.168.1.20)
IP ?= $(shell { ipconfig getifaddr en0 || ipconfig getifaddr en1 \
	|| ip -4 route get 1.1.1.1 | awk '{for (i = 1; i < NF; i++) if ($$i == "src") print $$(i + 1)}'; } 2>/dev/null | head -n1)

.PHONY: help system-deps setup scans scans-candidates scans-all scans-eval llm run certs public-cert run-vr test

# run-vr certificate. public (default): local-ip.sh's Let's Encrypt certificate
# for *.local-ip.sh, whose names resolve to the LAN address they spell
# (172-20-10-5.local-ip.sh -> 172.20.10.5). Every headset browser trusts it,
# and the Vive Focus Vision browser refuses mkcert's. Its key is public, so the
# connection is only as private as the network; the name lookup needs internet.
# mkcert: this machine's own CA (make certs), for a network without internet.
CERT ?= public

help:
	@echo "make system-deps  system packages: Python 3.12-3.14, OpenGL, mkcert (apt, dnf, pacman or brew; uses sudo)"
	@echo "make setup        virtualenv in .venv + pinned dependencies (Python: $(or $(PYTHON),none found))"
	@echo "                  (Linux without an NVIDIA GPU: make setup CPU=1)"
	@echo "make scans        download the 7 demo patients (data/demo_scans.json) from Zenodo"
	@echo "make scans-candidates  the 24 patients the demo set was picked from (~1 GB); they show up in make run"
	@echo "make scans-all    all 100 scans, for retraining (~3 GB)"
	@echo "make scans-eval   the larger held-out splits val_extra + test_extra (full dataset, ~2 GB)"
	@echo "make llm          install and start Ollama if needed, then pull the local LLM ($(LLM)) for free-form voice commands"
	@echo "make run          viewer on http://127.0.0.1:$(PORT)"
	@echo "make certs        mkcert certificate for this machine's LAN address ($(IP)), for run-vr CERT=mkcert"
	@echo "make run-vr       viewer on https://$(subst .,-,$(IP)).local-ip.sh:$(PORT) for a headset on the same network"
	@echo "                  (CERT=mkcert: https://$(IP):$(PORT) with the make certs certificate)"
	@echo "make test         the tests (~30 s, synthetic scans, no download needed)"

system-deps:
	@if [ "$$(uname -s)" = Linux ] && v=$$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $$2}') && [ -n "$$v" ] \
		&& [ "$$(printf '%s\n' 2.28 "$$v" | sort -V | head -n1)" != 2.28 ]; then \
		echo "glibc $$v is older than 2.28 (e.g. Ubuntu 18.04): torch and vtk have no wheels for it. Use Docker instead (README, Docker)"; exit 1; \
	fi
	@if command -v apt-get >/dev/null 2>&1; then \
		set -x; sudo apt-get update; sudo apt-get install -y python3 python3-venv make curl libgl1 libegl1 mkcert; \
		if ! python3 -c 'import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 14))'; then \
			: "python3 is outside 3.12-3.14 (Ubuntu 22.04 has 3.10): Python 3.12 from the deadsnakes PPA"; \
			sudo apt-get install -y software-properties-common && sudo add-apt-repository -y ppa:deadsnakes/ppa \
			&& sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv; \
		fi; \
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
	@if [ "$$(uname -s)" = Linux ] && v=$$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $$2}') && [ -n "$$v" ] \
		&& [ "$$(printf '%s\n' 2.28 "$$v" | sort -V | head -n1)" != 2.28 ]; then \
		echo "glibc $$v is older than 2.28 (e.g. Ubuntu 18.04): torch and vtk have no wheels for it. Use Docker instead (README, Docker)"; exit 1; \
	fi
	@test -n "$(PYTHON)" || { echo "no Python 3.12-3.14 found (python3.14, python3.13, python3.12 or python3): make system-deps"; exit 1; }
	@echo "using $(PYTHON) ($$($(PYTHON) --version))"
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

setup: $(PY)
ifneq ($(CPU),)
	$(PY) -m pip install torch==$(TORCH) --index-url https://download.pytorch.org/whl/cpu
endif
	$(PY) -m pip install -r requirements.txt
ifeq ($(CPU),)
	@# Speech recognition on the GPU (asr.py): CTranslate2 needs cuBLAS 12, torch's CUDA 13 wheels bring only 13.
	@if [ "$$(uname -s)" = Linux ] && command -v nvidia-smi >/dev/null 2>&1; then \
		set -x; $(PY) -m pip install nvidia-cublas-cu12==$(CUBLAS12); fi
endif
	@echo; echo "next: make scans && make llm && make run"

scans: $(PY)
	$(PY) -u -m tools.fetch_scans --demo --splits

scans-candidates: $(PY)
	$(PY) -u -m tools.fetch_scans --candidates --splits

scans-all: $(PY)
	$(PY) -u -m tools.fetch_scans --splits all

scans-eval: $(PY)
	$(PY) -u -m tools.fetch_scans --splits val_extra test_extra

# Installs Ollama when it is missing (pacman, Homebrew, or the official script
# on other Linux distributions), starts its server when nothing answers, then
# pulls the model. The packages and the script also set up a service that
# starts Ollama on boot.
llm:
	@if command -v ollama >/dev/null 2>&1; then \
		echo "Ollama is installed: $$(ollama --version 2>/dev/null | tail -n1)"; \
	elif command -v pacman >/dev/null 2>&1; then \
		pkg=ollama; command -v nvidia-smi >/dev/null 2>&1 && pkg=ollama-cuda; \
		set -x; sudo pacman -S --needed --noconfirm $$pkg && sudo systemctl enable --now ollama; \
	elif [ "$$(uname -s)" = Darwin ]; then \
		command -v brew >/dev/null 2>&1 || { echo "Homebrew is missing: https://brew.sh, or install Ollama from https://ollama.com/download"; exit 1; }; \
		set -x; brew install ollama && brew services start ollama; \
	elif [ "$$(uname -s)" = Linux ]; then \
		set -x; curl -fsSL https://ollama.com/install.sh | sh; \
	else \
		echo "install Ollama yourself: https://ollama.com/download"; exit 1; \
	fi
	@for i in 1 2 3 4 5; do ollama list >/dev/null 2>&1 && exit 0; sleep 1; done; \
	echo "Ollama is not running: starting ollama serve in the background (log: out/ollama.log)"; \
	mkdir -p out; nohup ollama serve >out/ollama.log 2>&1 & \
	for i in $$(seq 30); do ollama list >/dev/null 2>&1 && exit 0; sleep 1; done; \
	echo "Ollama did not start: see out/ollama.log"; exit 1
	ollama pull $(LLM)
	@# Faster parsing: the same model in a llama-server with speculative decoding (tools/llm_server.py; Linux)
	@if [ -x $(PY) ]; then $(PY) -m tools.llm_server ensure --model $(LLM); fi

# Lists every downloaded patient, the demo set first; SCANS=demo lists only the demo set
run: $(PY)
	@$(PY) -m tools.llm_server ensure
	SCANS=$(or $(SCANS),all) $(PY) server.py --port $(PORT)

certs:
	@command -v mkcert >/dev/null || { echo "mkcert is not installed: make system-deps"; exit 1; }
	@test -n "$(IP)" || { echo "no LAN address found; pass one: make certs IP=..."; exit 1; }
	mkdir -p certs
	mkcert -cert-file certs/lan.pem -key-file certs/lan-key.pem $(IP) localhost 127.0.0.1

# Refreshed on every run-vr (local-ip.sh renews it monthly); the saved copy is
# used when the site is unreachable.
public-cert:
	@mkdir -p certs
	@if curl -fsS -m 20 -o certs/public.pem.tmp https://local-ip.sh/server.pem \
		&& curl -fsS -m 20 -o certs/public-key.pem.tmp https://local-ip.sh/server.key; then \
		mv certs/public.pem.tmp certs/public.pem; mv certs/public-key.pem.tmp certs/public-key.pem; \
		chmod 600 certs/public-key.pem; \
	else \
		rm -f certs/public.pem.tmp certs/public-key.pem.tmp; \
		test -f certs/public.pem || { echo "local-ip.sh unreachable and no saved certificate: make run-vr CERT=mkcert"; exit 1; }; \
		echo "local-ip.sh unreachable; using the saved certificate"; \
	fi

run-vr: $(PY)
ifeq ($(CERT),mkcert)
	@test -f certs/lan.pem || { echo "no certificate yet: make certs"; exit 1; }
	@echo "headset browser: https://$(IP):$(PORT)/"
	@$(PY) -m tools.llm_server ensure
	SCANS=$(or $(SCANS),all) $(PY) server.py --host 0.0.0.0 --port $(PORT) --ssl-certfile certs/lan.pem --ssl-keyfile certs/lan-key.pem
else
	@test -n "$(IP)" || { echo "no LAN address found; pass one: make run-vr IP=..."; exit 1; }
	@$(MAKE) --no-print-directory public-cert
	@echo "headset browser: https://$(subst .,-,$(IP)).local-ip.sh:$(PORT)/"
	@$(PY) -m tools.llm_server ensure
	SCANS=$(or $(SCANS),all) $(PY) server.py --host 0.0.0.0 --port $(PORT) --ssl-certfile certs/public.pem --ssl-keyfile certs/public-key.pem
endif

test: $(PY)
	$(PY) -m pytest -q
