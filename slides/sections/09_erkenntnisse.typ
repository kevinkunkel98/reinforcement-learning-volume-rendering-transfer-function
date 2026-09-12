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
