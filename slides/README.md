# Pitchvortrag: Transferfunktionen im Volume Rendering mit RL

Masterseminar-Pitch an der Universität Leipzig (SS 2026).

Betreuer: Prof. Gerik Scheuermann, Dr. Baldwin Nsonga, Dr. Thomas Wischgoll
Autor: Kevin Kunkel

## Thema

Sprach-/text-gesteuertes Transferfunktions-Design für Volume Rendering,
bewertet über eine exakte Hounsfield-Metrik statt eines MLLM-Judges, als
Baseline für einen späteren RL-Agenten. Inhalte, Screenshots und Ergebnisse
stammen aus dem Prototyp im Repo-Root (siehe `../README.md`).

## Files

| File | Beschreibung |
|------|--------------|
| `slides.typ` | Die Präsentation (Typst + Touying, Metropolis-Theme) |
| `sections/` | Ein `.typ` pro Folienblock, `01_gliederung.typ` … `10_rl_ausblick.typ` |
| `helpers.typ` | Farbpalette + Box-Helfer (Definition/Theorem/Anmerkung/Beispiel) |
| `diagrams.typ` | cetz-Diagramme: Architektur-Pipeline, TF-Gaussian-Kurven |
| `refs.bib` | Literatur (siehe `literature/` im Repo-Root für die PDFs) |
| `screenshot_ui.png`, `skull_before.png`, `skull_after.png` | Aus dem laufenden Prototyp erzeugte Bilder |
| `leipziglogo.png` | Uni-Logo für die Titelfolie |

## Build

```bash
typst compile slides.typ    # -> slides.pdf
typst watch slides.typ      # Live-Neukompilierung
```

Präsentieren mit `zathura slides.pdf` → `F5` für Vollbild.

## Assets neu erzeugen

Screenshot und Vorher/Nachher-Bilder stammen direkt aus dem Prototyp und lassen
sich bei Bedarf neu erzeugen (siehe `../README.md` für den Server-Start).
