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
  [*1 — Motivation:* Warum Transferfunktions-Design so mühsam ist], [*2 — Verwandte Arbeiten & Fragestellung*],
  [*3 — Architektur:* Sprache/Text → Kommando → Render → Bewertung],
  [*4 — Die Transferfunktion:* 24 Werte, echte Hounsfield-Einheiten],

  [*5 — Der Prototyp:* Live-Demo der Chat-Oberfläche], [*6 — Kommandogrammatik & Parser-Auswertung*],
  [*7 — Ergebnisse an echten CT-Daten*], [*8 — Erkenntnisse aus dem Testen*],
  [*9 — Von der Baseline zum RL-Agenten:* SAC, Online-Lernen, RLHF],
)
