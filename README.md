# Say what you want to see

Speak to a CT scan and it renders what you asked for: "show only the
skeleton", "a bit more kidneys", "make the lungs darker". A reinforcement
learning policy turns the instruction into a transfer function, which works on
patients it has never seen. It also works in VR.

<p align="center">
  <img src="docs/screenshots/ui.png" width="85%" alt="The viewer: a render, the instruction history with each command's effect, and the transfer function panel" />
</p>

## How it works

1. **Understand.** Whisper transcribes speech. A rule parser handles known
   phrasings, and a local LLM (Ollama) handles everything else.
2. **Propose.** A policy trained with model-based RL suggests a transfer
   function in one step.
3. **Refine.** Gradients through the policy's learned model of the renderer
   improve the suggestion in about 0.2 s, without a single render.
4. **Polish.** A short search on the real objective finishes the job.

On held-out patients, the policy reaches about half of the goal (median
attainment 0.48). Planning raises that to 0.64. With 30 search steps on top it
matches what a search from scratch reaches with 200 (0.77 vs. 0.70), and with
200 it reaches 0.90. Model-free RL (SAC) stayed below 0.1.

<p align="center">
  <img src="docs/figures/budget_curve.png" width="65%" alt="Median attainment against search budget: starting from the policy and planning needs far fewer renders than searching from scratch" />
</p>

## Quick start

macOS, Ubuntu/Debian, Fedora or Arch. You need `git` and `make`.

```bash
git clone https://github.com/kevinkunkel98/reinforcement-learning-volume-rendering-transfer-function.git
cd reinforcement-learning-volume-rendering-transfer-function
make system-deps   # Python 3.12-3.14, OpenGL, mkcert (asks for sudo)
make setup         # Linux without an NVIDIA GPU: make setup CPU=1
make scans         # 6 test patients from Zenodo, ~250 MB
make run           # http://127.0.0.1:8000
```

Pick a `ts_…` scan in the scan list. These patients have the organ labels the
policy was trained on. Type a command, or hold Space to talk.

**Free-form voice commands:** install [Ollama](https://ollama.com/download),
then run `make llm`. Without it, only the rule parser answers.

**VR headset** (tested on a Vive Focus 3) on the same network:

```bash
make certs     # HTTPS certificate for this machine's address
make run-vr    # prints the address to open in the headset browser
```

The headset browser warns once about the certificate. Accept the warning.
Hold the trigger to talk and use the thumbstick to turn the volume.

`make help` lists every target, and `make test` runs the tests (~13 min,
~8 GB RAM).

## More

- [`docs/STATUS.md`](docs/STATUS.md): the current state, results and open problems
- [`docs/experiments/`](docs/experiments/): one dated write-up per experiment,
  with full tables and the scripts to reproduce them. Retraining the policy
  starts at `docs/experiments/2026-10-19-scripts/pipeline_v3.sh` and needs
  `make scans-all`.
- [`tools/headset/README.md`](tools/headset/README.md): VR URL flags and
  headset debugging

## Data

The CT scans and organ labels come from
[TotalSegmentator](https://zenodo.org/records/10047263) (CC-BY-4.0): 90
training, 4 validation and 6 test patients. The four out-of-distribution CTs
come from the 3D Slicer sample data.
