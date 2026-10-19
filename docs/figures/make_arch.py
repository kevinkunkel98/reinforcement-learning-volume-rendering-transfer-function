"""Architecture overview figure (thesis abstract) in the OpenMoji
paper-figure style.

    python docs/figures/make_arch.py docs/figures/openmoji docs/figures/arch.svg
    rsvg-convert -f pdf docs/figures/arch.svg -o docs/figures/arch.pdf
    rsvg-convert -w 2200 docs/figures/arch.svg -o docs/figures/arch.png

Icons: OpenMoji 15.0 (https://openmoji.org, CC BY-SA 4.0), copied into
docs/figures/openmoji/ -- credit them in the caption. The VR headset and the
transfer-function cards are drawn here. The numbers in the panel are an
illustration, not one episode.
"""
import re
import sys
from pathlib import Path

ICONS = Path(sys.argv[1])          # folder with OpenMoji colour SVGs
OUT = Path(sys.argv[2])

W, H = 2200, 760
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "Menlo, DejaVu Sans Mono, monospace"
GREY = "#a3a3a3"
parts = []


def icon(code, x, y, size):
    raw = (ICONS / f"{code}.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", raw, re.S).group(1)
    inner = re.sub(r'\sid="[^"]*"', "", inner)
    parts.append(f'<svg x="{x}" y="{y}" width="{size}" height="{size}" viewBox="0 0 72 72">{inner}</svg>')


def text(x, y, s, size=30, weight=400, anchor="start", fill="#111", family=FONT, italic=False):
    style = ' font-style="italic"' if italic else ""
    parts.append(f'<text xml:space="preserve" x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
                 f'text-anchor="{anchor}" fill="{fill}"{style}>{s}</text>')


def lines(x, y, rows, size=30, gap=34, anchor="start", **kw):
    for i, row in enumerate(rows):
        text(x, y + i * gap, row, size=size, anchor=anchor, **kw)


def rbox(x, y, w, h, r=36, stroke=4, fill="white", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="#111" stroke-width="{stroke}"{d}/>')


def arrow(x1, y1, x2, y2, color=GREY, width=7):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2 - 14}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
    parts.append(f'<polygon points="{x2},{y2} {x2 - 18},{y2 - 10} {x2 - 18},{y2 + 10}" fill="{color}"/>')


def dashed(x1, x2, y):
    parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#444" stroke-width="3" stroke-dasharray="14 10"/>')


def tf_glyph(x, y, size, variant):
    """A transfer-function 'document': rounded card with opacity peaks.
    variant: plain | planned (green fill) | searched (green fill + tick)."""
    s = size / 72
    parts.append(f'<g transform="translate({x},{y}) scale({s})">')
    parts.append('<rect x="10" y="6" width="52" height="60" rx="8" fill="white" stroke="#000" stroke-width="4.5"/>')
    if variant != "plain":
        parts.append('<path d="M12,50 L60,50 L60,58 Q60,64 54,64 L18,64 Q12,64 12,58 Z" fill="#5cb85c"/>')
    parts.append('<polyline points="16,46 24,46 29,22 34,46 40,46 45,32 50,46 56,46" fill="none" stroke="#000" '
                 'stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')
    if variant == "searched":
        parts.append('<polyline points="44,12 49,17 58,8" fill="none" stroke="#5cb85c" stroke-width="5" '
                     'stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append("</g>")


def headset(x, y, size):
    """VR headset drawn in OpenMoji's line style (no headset emoji exists)."""
    s = size / 72
    parts.append(f'<g transform="translate({x},{y}) scale({s})" stroke="#000" stroke-width="2.6" stroke-linejoin="round">')
    parts.append('<path d="M6,34 Q6,26 12,26" fill="none"/><path d="M66,34 Q66,26 60,26" fill="none"/>')
    parts.append('<rect x="9" y="22" width="54" height="30" rx="9" fill="#d0cfce"/>')
    parts.append('<rect x="14" y="27" width="44" height="14" rx="5" fill="#3f3f3f"/>')
    parts.append('<path d="M30,52 Q36,44 42,52" fill="#9b9b9a"/>')
    parts.append('<circle cx="24" cy="34" r="3" fill="#61b2e4" stroke="none"/><circle cx="48" cy="34" r="3" fill="#61b2e4" stroke="none"/>')
    parts.append("</g>")


# ---- left: the data ---------------------------------------------------------
rbox(55, 85, 330, 485)
icon("1FA7B", 80, 105, 70)
text(165, 155, "CT volume", 38, 500)
for i, (code, name) in enumerate([("1FAC1", "lungs"), ("1FAC0", "heart"), ("1F9B4", "skeleton")]):
    icon(code, 92, 200 + i * 78, 56)
    text(170, 240 + i * 78, name, 30, fill="#333")
text(170, 462, "+ 5 organ labels", 26, fill="#666", italic=True)
tf_glyph(88, 482, 60, "plain")
text(170, 522, "transfer function", 28, fill="#333")

rbox(55, 595, 720, 82, r=41)
icon("1F3A4", 76, 607, 58)
text(150, 650, "Instruction", 38, 500)
text(352, 648, "“more bone, less lungs”", 30, fill="#333", italic=True)

# ---- middle: three ways to answer (the compared conditions) -----------------
rows = [
    (185, "1F916", "Policy", ["One-shot", "proposal"], "plain"),
    (350, "1F4AD", "Planning", ["Planned in the", "learned model"], "planned"),
    (512, "1F50D", "Search", ["Refined with", "30–200 renders"], "searched"),
]
for cy, code, actor, label, variant in rows:
    arrow(392, cy, 488, cy)
    icon(code, 403, cy - 112, 74)
    text(440, cy + 40, actor, 28, anchor="middle")
    tf_glyph(500, cy - 50, 100, variant)
    lines(618, cy - 8, label, 29, gap=34, fill="#333")
    rbox(880, cy - 52, 250, 104, r=26)
    icon("1FA7B", 897, cy - 34, 66)
    icon("1F3A4", 973, cy - 34, 66)
    tf_glyph(1047, cy - 36, 72, variant)
dashed(500, 1130, 268)
dashed(500, 1130, 432)

# ---- centre: the agent -------------------------------------------------------
arrow(1170, 190, 1290, 190)
arrow(1170, 350, 1290, 350, color="#dcdcdc", width=5)
arrow(1170, 520, 1290, 520, color="#dcdcdc", width=5)
icon("1F916", 1174, 222, 112)
lines(1226, 400, ["Model-based", "RL agent"], 31, gap=38, anchor="middle")

# ---- the 'trace' panel: what one command does --------------------------------
px, pw = 1340, 400
parts.append(f'<rect x="{px}" y="72" width="{pw}" height="372" fill="#ececec"/>')
lines(px + 22, 118, ["parse:", "  skeleton ↑  lungs ↓", "measure visibility", "propose Δθ", "plan (ensemble)", "search: 30 renders"],
      27, gap=55, family=MONO, fill="#222")
parts.append(f'<rect x="{px}" y="458" width="{pw}" height="92" fill="#d9ead3"/>')
lines(px + 18, 496, ["+ skeleton  10.9 → 15.2 %", "+ heart      3.0 →  5.6 %"], 24, gap=36, family=MONO, fill="#222")
parts.append(f'<rect x="{px}" y="550" width="{pw}" height="92" fill="#f2d0d0"/>')
lines(px + 18, 588, ["− lungs     13.2 →  5.1 %", "− other      8.4 →  0.4 %"], 24, gap=36, family=MONO, fill="#222")

# ---- right: what is measured ----------------------------------------------------
rx = 1770
icon("1F50D", rx + 22, 120, 78)
arrow(rx, 232, rx + 120, 232)
lines(rx + 60, 280, ["Visibility", "metric"], 28, gap=34, anchor="middle")
icon("1F4CA", rx + 148, 150, 120)
lines(rx + 208, 312, ["Attainment"], 28, anchor="middle")

headset(rx + 22, 362, 78)
arrow(rx, 452, rx + 120, 452)
lines(rx + 60, 498, ["VR viewer"], 28, anchor="middle")
text(rx + 208, 452, "58 fps", 44, 600, anchor="middle")
lines(rx + 208, 498, ["Frame rate"], 28, anchor="middle")

icon("1F9D1", rx + 24, 552, 74)
arrow(rx, 648, rx + 120, 648)
lines(rx + 60, 694, ["User", "study"], 28, gap=34, anchor="middle")
for i, code in enumerate(["2714", "274C", "2714"]):
    icon(code, rx + 140 + i * 52, 594, 44)
lines(rx + 208, 694, ["Human rating"], 28, anchor="middle")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       f'<rect width="{W}" height="{H}" fill="white"/>' + "".join(parts) + "</svg>")
OUT.write_text(svg)
print("wrote", OUT)
