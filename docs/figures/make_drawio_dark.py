"""Dark variant of a draw.io diagram, for the README's dark theme: maps the
light palette onto GitHub's dark one and gives every text a light colour.

    python docs/figures/make_drawio_dark.py docs/figures/pipeline_rl_warmstart.drawio

writes pipeline_rl_warmstart.dark.drawio next to it. Export it to PNG with
draw.io (or the diagrams.net viewer) on a #0d1117 background.
"""
import re
import sys

PAGE = "#0d1117"
INK = "#e6edf3"
# light -> dark, applied to styles and to the inline HTML in labels
PALETTE = {
    "#f5f5f5": "#151b23",   # training panel
    "#fcfcfb": "#12171e",   # use panel
    "#ffffff": "#1f2630",   # plain boxes
    "#e8f1fc": "#13294a",   # RL boxes
    "#2a78d6": "#58a6ff",   # RL accent (strokes, numbers)
    "#6b6a63": "#9198a1",   # grey strokes and notes
    "#bdbdbd": "#3d444d",   # panel borders
    "#1baf7a": "#3fcf8e",   # baseline
}
# edge labels sit on a panel or on the page; draw.io's default label
# background is white
PANEL_OF_LABEL = {"labelBackgroundColor=#ffffff": f"labelBackgroundColor={PAGE}"}


def darken(xml: str) -> str:
    for light, dark in PANEL_OF_LABEL.items():
        xml = xml.replace(light, dark)
    for light, dark in PALETTE.items():
        xml = re.sub(re.escape(light), dark, xml, flags=re.IGNORECASE)

    def style(match):
        value = match.group(1)
        if "fontColor=" not in value:
            value += f";fontColor={INK}"
        if "edgeStyle=" in value and "labelBackgroundColor=" not in value:
            value += f";labelBackgroundColor={PALETTE['#f5f5f5']}"
        return f'style="{value}"'

    return re.sub(r'style="([^"]*)"', style, xml)


if __name__ == "__main__":
    path = sys.argv[1]
    out = path.replace(".drawio", ".dark.drawio")
    with open(path, encoding="utf-8") as stream:
        dark = darken(stream.read())
    with open(out, "w", encoding="utf-8") as stream:
        stream.write(dark)
    print(f"wrote {out}")
