#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Die Zielgröße ist selbst nur ein Stellvertreter
// ══════════════════════════════════════════════════════════════════════════════
== Die Zielgröße ist selbst nur ein Stellvertreter

#v(0.1em)
`visibility.py` misst, *wie viel* einer Klasse im Bild ankommt — nicht, ob ein
Mensch das Bild für gelungen hält.

#v(0.3em)
#table(
  columns: (auto, 1fr),
  stroke: none,
  row-gutter: 0.45em,
  column-gutter: 0.8em,
  inset: (x: 0pt, y: 2pt),
  [*Erhebung*], [Blindvergleich zweier Kandidaten zu derselben Instruktion, je sechs Ansichten],
  [*Tasten*], [A · B · E (gleich) · S (überspringen)],
  [*Kontrolle*], [Jedes zehnte Paar wiederholt ein früheres seitenvertauscht],
  [*Pilot*], [54 Urteile, als Pilot markiert und von jeder Auswertung ausgeschlossen],
)

#v(0.3em)
#thm-box([Erster Befund, noch ohne Aussagekraft], [
  Im Pilot stimmte die Metrik in *21 von 38* entschiedenen Paaren mit dem
  Urteil überein (55 %, 95 %-KI 0.40–0.70) — kaum über Zufall. Genau diese
  Lücke rechtfertigt ein gelerntes Reward-Modell.
], fill: rgb("#fff3d6"), stroke-color: amber)

#v(0.2em)
#remark[
  Nach Instruktionsart: relative Änderungen 79 %, "show only" und
  zusammengesetzte Kommandos um 43 %.
]
