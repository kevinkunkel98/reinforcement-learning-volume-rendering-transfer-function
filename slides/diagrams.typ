#import "@preview/cetz:0.3.4": canvas, draw
#import "helpers.typ": navy, blue, sage, amber

// ── Pipeline overview diagram ────────────────────────────────────────────────
#let pipeline-diagram = canvas(length: 1cm, {
  import draw: *

  let box-h = 1.7
  let box-w = 3.0
  let gap = 0.55
  let y-top = 1.4
  let y-bot = -1.4

  let node(pos, title, sub, fill, text-col: white) = {
    rect(
      (pos.at(0) - box-w / 2, pos.at(1) - box-h / 2),
      (pos.at(0) + box-w / 2, pos.at(1) + box-h / 2),
      fill: fill, stroke: none, radius: 0.12,
    )
    content((pos.at(0), pos.at(1) + 0.32), text(fill: text-col, size: 0.34cm, weight: "bold")[#title])
    content((pos.at(0), pos.at(1) - 0.28), text(fill: text-col, size: 0.24cm)[#sub])
  }

  let arrow(from, to, col: luma(140)) = {
    line(from, to, stroke: (paint: col, thickness: 1.3pt), mark: (end: ">", size: 0.22))
  }

  // top row
  let x1 = -8.6
  let x2 = -5.2
  let x3 = -1.8
  let x4 = 1.6
  let x5 = 5.0

  node((x1, y-top), [Voice / Text], [Whisper + Chat-UI], luma(90))
  node((x2, y-top), [Parser], [Regel · LLM], rgb("#6b3fa0"))
  node((x3, y-top), [Kommando], [\{target, direction,\ strength\}], blue)
  node((x4, y-top), [TF-Vektor], [4 Gaussians · 24 Werte], navy)
  node((x5, y-top), [VTK-Render], [GPU Raycasting], blue)

  arrow((x1 + box-w / 2, y-top), (x2 - box-w / 2, y-top))
  arrow((x2 + box-w / 2, y-top), (x3 - box-w / 2, y-top))
  arrow((x3 + box-w / 2, y-top), (x4 - box-w / 2, y-top))
  arrow((x4 + box-w / 2, y-top), (x5 - box-w / 2, y-top))

  // bottom row
  node((x5, y-bot), [Bewertung], [opacity\_mass · Mensch], amber)
  node((x4, y-bot), [Hill-Climbing], [+1 Schritt ×1.2 / −1 halbieren], sage)
  node((x3, y-bot), [JSONL-Log], [log · preferences · feedback], rgb("#8a2e2e"))

  arrow((x5, y-top - box-h / 2), (x5, y-bot + box-h / 2))
  arrow((x5 - box-w / 2, y-bot), (x4 + box-w / 2, y-bot))
  arrow((x4 - box-w / 2, y-bot), (x3 + box-w / 2, y-bot))

  line(
    (x4, y-bot + box-h / 2), (x4, 0), (x4 - 2.3, 0), (x4 - 2.3, y-top - box-h / 2),
    stroke: (paint: sage, thickness: 1.3pt), mark: (end: ">", size: 0.22),
  )
  content((x4 - 2.3, 0.15), text(fill: sage, size: 0.22cm, style: "italic")[nächster Schritt])
})

// ── Transfer-function opacity-curve diagram ──────────────────────────────────
#let tf-curve-diagram = canvas(length: 1cm, {
  import draw: *
  let col-ax = luma(100)

  let bands = (
    (label: [Luft], x: -8.5, col: luma(160)),
    (label: [Fett], x: -3.5, col: rgb("#c9a227")),
    (label: [Weich-\ gewebe], x: -0.5, col: rgb("#b04a4a")),
    (label: [Spongiosa], x: 2.5, col: rgb("#b58a4a")),
    (label: [Kortikalis], x: 6.5, col: navy),
  )

  line((-9.5, 0), (8.5, 0), stroke: (paint: col-ax, thickness: 0.9pt), mark: (end: ">", size: 0.2))
  content((8.8, 0), text(size: 0.28cm, fill: col-ax)[HU])
  line((-9.5, 0), (-9.5, 3.6), stroke: (paint: col-ax, thickness: 0.9pt), mark: (end: ">", size: 0.2))
  content((-9.7, 3.85), text(size: 0.26cm, fill: col-ax)[$alpha$])

  let gauss(center, width, height, col) = {
    let pts = ()
    let x = center - 3 * width
    while x <= center + 3 * width {
      let y = height * calc.exp(-0.5 * calc.pow((x - center) / width, 2))
      pts.push((x, y))
      x += width / 12
    }
    line(..pts, stroke: (paint: col, thickness: 2.2pt))
  }

  gauss(-3.5, 1.3, 1.0, rgb("#c9a227"))
  gauss(-0.5, 1.0, 1.6, rgb("#b04a4a"))
  gauss(2.5, 1.6, 2.3, rgb("#b58a4a"))
  gauss(6.5, 2.2, 3.2, navy)

  for b in bands {
    content((b.x, -0.45), text(size: 0.22cm, fill: b.col, weight: "bold")[#b.label])
  }
  content((-6, 3.3), text(size: 0.24cm, fill: luma(90), style: "italic")[
    4 × [Zentrum, Breite, Höhe, R, G, B] = 24 Werte, normiert auf $[-1, 1]$
  ])
})
