#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Gliederung
// ══════════════════════════════════════════════════════════════════════════════
== Gliederung

#v(0.3em)
#grid(
  columns: (1fr, auto),
  gutter: 0.6em,
  row-gutter: 0.55em,
  align: (left, right),
  [*1 — Motivation:* Warum Transferfunktions-Design so mühsam ist], [~2 Min.],
  [*2 — Verwandte Arbeiten & Fragestellung*], [~2 Min.],
  [*3 — Architektur:* Sprache/Text → Kommando → Render → Bewertung], [~3 Min.],
  [*4 — Die Transferfunktion:* 24 Werte, echte Hounsfield-Einheiten], [~2 Min.],
  [*5 — Der Prototyp:* Live-Demo der Chat-Oberfläche], [~2 Min.],
  [*6 — Kommandogrammatik & Parser-Auswertung*], [~2 Min.],
  [*7 — Ergebnisse an echten CT-Daten*], [~1 Min.],
  [*8 — Erkenntnisse aus dem Testen*], [~1 Min.],
  [*9 — Von der Baseline zum RL-Agenten:* SAC, Online-Lernen, RLHF], [~4 Min.],
)
