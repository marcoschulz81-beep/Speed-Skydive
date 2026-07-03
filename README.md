# Speed-Skydive Analyzer

Webanwendung zur automatischen FlySight-Auswertung für Speed-Skydiving mit Fokus auf Techniktraining und regelnahe 3s-Wertung.

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
- Regelnaher 3s-Score im Performance Window (`velD >= 10m/s`, Höhenverlust/Breakoff)
- Phasenmodell (Start, Beschleunigung, Max-Speed, Ende)
- Hot-Zone-Erkennung und Negativ/Kippen-Heuristik
- Qualitätsflags + Qualitätsscore
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

## Neue Version: Kontext und Leistungsprofil

- Beim Upload kann jeder Sprung als `Training` oder `Wettkampf` markiert werden.
- Falls die Markierung beim Upload vergessen wurde, kann sie spaeter im Sprungreport geaendert werden.
- Die Markierung aendert keine Speed-Berechnung. Sie dient fuer Verlauf, Filter, Profil und spaetere Auswertungen.
- Pro Springer wird automatisch ein Leistungsprofil berechnet:
  - primaer aus dem regelnahen 3s-Score, falls vorhanden
  - zusaetzlich aus dem Training-3s-Max als Vergleichswert
  - mit Top-1/Top-3/Top-5/Top-10-Durchschnitten
  - mit Profil-Vertrauen je nach Anzahl gueltiger Spruenge
- Die Coaching-Tipps nutzen das Profil:
  - Basis/Aufbau fokussiert staerker Exit und Aufbau
  - Schnell/Elite fokussiert staerker Hot-Zone, Stabilitaet und Korrekturen

## Datenmodell

- `jumps`: Metadaten pro Sprung, t0, Exit-Höhe, Gültigkeit, Qualität
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
