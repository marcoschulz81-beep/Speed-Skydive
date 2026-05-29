# Speed-Skydive Analyzer

Webanwendung zur automatischen FlySight-Auswertung fuer Speed-Skydiving mit Fokus auf Techniktraining und regelnahe 3s-Wertung.

## Enthaltene Funktionen

- CSV-Upload (FlySight 1 kompatibel) inkl. Pflichtspalten-Validierung
- Automatische `t0`-Erkennung (Absprungzeitpunkt) mit Plausibilitaetspruefung
- Berechnung pro Sample:
  - vertikale/horizontale/gesamte Geschwindigkeit
  - Tauchwinkel
  - vertikale Beschleunigung
  - relative Zeitachse
- Fixpunkte `+10/+15/+20/+24/+28s` per linearer Interpolation
- Bestes zusammenhaengendes 3s-Fenster (zeitkontinuierlich) aus `t0`-Bezug
- Regelnaher 3s-Score im Performance Window (`velD >= 10m/s`, Hoehenverlust/Breakoff)
- Phasenmodell (Start, Beschleunigung, Max-Speed, Ende)
- Hot-Zone-Erkennung und Negativ/Kippen-Heuristik
- Qualitaetsflags + Qualitaetsscore
- Automatische Scorecard + konkrete Technik-Tipps
- Speicherung pro Springer in SQLite (jumps/samples/metrics)
- HTML-Report mit Kurven + PDF-Export
- Vergleichsansicht je Springer

## Tech-Stack

- Python 3.11+
- FastAPI + Jinja2
- Pandas/Numpy (Analyse)
- Plotly.js (Kurven in UI)
- SQLite (Persistenz)
- fpdf2 (PDF-Report)

## Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Danach: `http://127.0.0.1:8000`

## Tests

```powershell
pytest
```

## Datenmodell

- `jumps`: Metadaten pro Sprung, t0, Exit-Hoehe, Gueltigkeit, Qualitaet
- `samples`: abgeleitete Samplewerte (t_rel, vVert, vHor, Winkel, Flags ...)
- `metrics`: 3s-Score, Window, Hot-Zone, Risiko, Fixpunkte, Phasen, Tipps

## GitHub-Setup

```powershell
git init
git add .
git commit -m "Initial commit: Speed-Skydive analyzer"
```

Optional mit GitHub CLI:

```powershell
gh repo create speed-skydive-analyzer --public --source . --remote origin --push
```

