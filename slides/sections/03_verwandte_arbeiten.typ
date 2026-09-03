#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Verwandte Arbeiten & Fragestellung
// ══════════════════════════════════════════════════════════════════════════════
== Verwandte Arbeiten & Fragestellung

#v(0.05em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.3em,
  [
    *TF-Design: sprachbasiert & nicht*
    - T2TF: Text $arrow$ TF, differenzierbar @jeong2024textbased
    - IntuiTF: MLLM als Explorer *und* Bewerter @wang2025intuitf
    - NLI4VolVis: Chat-Agenten für Volume-Viz @ai2025nli4volvis
    - Segmentierung via Vision-Transformer @engel2024leveraging
    - TF-Optimierung über Bildvergleich @neuhauser2023transfer
  ],
  [
    *RL & Evaluation für Parameterexploration*
    - Deep RL mit Menschen im Loop @scurto2021designing
    - RL für sprachgesteuerte Viewpoint-Navigation @zhao2025natural
    - Harte, reproduzierbare Metriken statt MLLM-Urteil @ai2025evaluation
  ],
)

#v(0.6em)
#thm-box([Fragestellung], [
  Lässt sich Transferfunktions-Design durch Sprache *und* eine exakte,
  kostenlose Metrik (statt eines MLLM-Richters) so weit automatisieren, dass
  daraus eine trainierbare RL-Baseline entsteht?
], fill: sky, stroke-color: blue)
