"""Latency per stage of one voice command, before and after the 2026-11-26
speed-ups (commits c58e033..562190a, merged as PR #1). Warm medians on the
RTX 5080 / Ryzen 9 9950X3D workstation, from that session's measurements:

- speech recognition: faster-whisper small on the real recordings, CPU int8
  vs CUDA float16 in the server's import order (four commands);
- LLM parse: qwen2.5:7b through Ollama vs llama-server with n-gram
  speculative decoding (eval_parsers.py, 185 phrases, median);
- goal ... render: the viewer pipeline benchmark (6 commands x 3), before the
  label-cache fix and with visibility on the CPU vs after both;
- refine search: 200 evaluations, CPU vs GPU visibility;
- browser: the transfer-function update in vtk.js, ~1.13 s per image before
  the setNodes fix ([viewer-timing]); not yet re-measured after it.

    .venv/bin/python docs/thesis/scripts/make_latency.py
"""
import thesis_style as ts
import matplotlib.pyplot as plt
import numpy as np

# (stage, before ms, after ms or None when not measured)
STAGES = [
    ("speech recognition", 550, 65),
    ("LLM parse (when used)", 185, 103),
    ("goal (incl. occluders)", 121, 9),
    ("planning", 91, 89),
    ("repair", 45, 4),
    ("render + readout", 102, 49),
    ("refine search (200)", 1262, 115),
    ("browser TF update, per image", 1130, None),
]


def main():
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    y = np.arange(len(STAGES))[::-1]
    before = np.array([s[1] for s in STAGES], float)
    after = np.array([np.nan if s[2] is None else s[2] for s in STAGES], float)
    ax.barh(y + 0.19, before, height=0.36, color="#d0d7de", edgecolor=ts.MUTED, linewidth=0.5, label="before")
    ax.barh(y - 0.19, np.nan_to_num(after), height=0.36, color=ts.BLUE, label="after (2026-11-26)")
    for yi, b, a in zip(y, before, after):
        ax.text(b * 1.08, yi + 0.19, f"{b:.0f}", va="center", fontsize=7, color=ts.MUTED)
        ax.text((a if a == a else 1.2) * 1.08, yi - 0.19, f"{a:.0f}" if a == a else "not re-measured",
                va="center", fontsize=7, color=ts.BLUE if a == a else ts.MUTED)
    ax.set_yticks(y, [s[0] for s in STAGES], fontsize=7.8)
    ax.set_xscale("log")
    ax.set_xlim(1, 5000)
    ax.set_xlabel("milliseconds (log scale), warm median, RTX 5080")
    ax.grid(axis="y", visible=False)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.6)
    ts.save(fig, "latency.png")


if __name__ == "__main__":
    main()
