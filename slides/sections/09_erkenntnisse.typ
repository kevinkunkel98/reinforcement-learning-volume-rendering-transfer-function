#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Erkenntnisse aus dem Testen
// ══════════════════════════════════════════════════════════════════════════════
== Erkenntnisse aus dem Testen

#v(0.15em)
Drei Bugs — gefunden durch echtes Testen, nicht durch Unit-Tests:

#v(0.35em)
#thm-box([Render-Crash], [
  - Annahme: würfelförmiger Datensatz $arrow.r$ Absturz bei echten CT ($512 times 512 times 139$)
  - *Fix:* Dimensionen pro Achse statt einer Kantenlänge
], fill: rgb("#fdeaea"), stroke-color: rgb("#a83232"))

#v(0.3em)
#thm-box([Sättigungs-Bug], [
  - "increase opacity strongly" $arrow.r$ Maximum in *einem* Schritt, danach stiller No-Op
  - *Fix:* Schrittweite proportional zur Restdistanz, nicht fixer Betrag
], fill: rgb("#fff3d6"), stroke-color: amber)

#v(0.3em)
#thm-box([Spracherkennung], [
  - Whisper verhört "spongy" reproduzierbar: "sponges", "spudgy", "spore G"
  - 24 echte Sprachkommandos: 19 (79 %) scheitern am Parser, nicht am System
], fill: mint, stroke-color: sage)

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Der teuerste Fehler war kein Bug
// ══════════════════════════════════════════════════════════════════════════════
== Der teuerste Fehler war kein Bug

#v(0.15em)
Eine Evaluation startete abends, der Fix an der Messung landete um 22:48, die
Ergebnisdateien entstanden um 03:26 — mit dem Code vom Prozessstart.

#v(0.4em)
#table(
  columns: (auto, auto, auto),
  align: (left, center, center),
  stroke: (x, y) => if y == 0 { (bottom: 0.8pt + navy) } else { (bottom: 0.3pt + luma(200)) },
  inset: (x: 6pt, y: 4pt),
  table.header[][*veröffentlicht*][*korrekt gemessen*],
  [Policy, Median], [+0.169], [*+0.275*],
  [Nachtrainieren hilft?], [nein ($p = 0.85$)], [*ja ($p = 0.0064$)*],
)

#v(0.4em)
#thm-box([Konsequenz], [
  Jede Ergebnisdatei speichert jetzt Git-Commit und Fingerabdruck der
  *importierten* Scoring-Module — erfasst beim Import, nicht beim Schreiben.
], fill: mint, stroke-color: sage)
