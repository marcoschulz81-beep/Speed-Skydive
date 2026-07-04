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

## Optionale OpenAI KI-Coaching-Texte

Die technische Analyse bleibt lokal und deterministisch. OpenAI wird nur genutzt, um aus den bereits berechneten Fakten bessere Coaching-Texte zu formulieren.

Aktivierung:

```powershell
$env:AI_COACHING_ENABLED="true"
$env:AI_COACHING_MODEL="gpt-5.5"
$env:OPENAI_API_KEY="sk-..."
uvicorn app.main:app --reload
```

Alternativ kann `.env.example` als Vorlage fuer eine lokale `.env` genutzt werden. Die App liest `.env` beim Start ein; bereits gesetzte Umgebungsvariablen haben Vorrang. `.env` ist absichtlich nicht versioniert.

Hinweise:

- Benoetigt wird ein OpenAI API-Key von `https://platform.openai.com/api-keys`.
- Ein ChatGPT Pro Account ist nicht automatisch ein API-Key. Fuer API-Nutzung muss der Key im OpenAI-Platform-Account erzeugt und ggf. API-Billing aktiv sein.
- Wenn `AI_COACHING_ENABLED=false` ist oder kein Key gesetzt wurde, funktioniert die normale Analyse unveraendert.
- Die KI bekommt keine Rohkurven, sondern nur kompakte Fakten: Scorecard, Hauptprobleme, strukturierte Coaching-Ziele, Follow-up und Qualitaetshinweise.
- Die KI darf keine Messwerte erfinden und ersetzt weder Speed-Berechnung noch Ranking.
- Im Expertenmodus wird ein Hinweis angezeigt, falls KI aktiviert ist, aber nicht erzeugt werden konnte.

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

## Strukturierte Coaching-Ziele

- Die bisherigen Tipptexte bleiben erhalten, werden aber zusaetzlich als strukturierte Coaching-Ziele gespeichert.
- Jedes Ziel enthaelt Phase, Prioritaet, Originaltext und messbare Zielmetriken.
- Beispiele fuer Zielmetriken:
  - `angle_10` / `angle_15`: frueher Tauchwinkel soll bei zu schnellem Steilwerden sinken
  - `gain_10_20`: Speed-Aufbau zwischen +10s und +20s soll steigen
  - `vhor_min_20_25`: Vorwaertsreserve in der Hot-Zone soll stabiler bleiben
  - `angle_turns_20_25`: Korrekturen in der Hot-Zone sollen sinken
- Beim naechsten Sprung prueft die Expertenansicht zuerst diese konkreten Ziele.
- Falls ein alter Datensatz noch keine strukturierten Ziele hat, nutzt die Software weiterhin den bisherigen Phasen-Fallback.

## KI-Vorbereitung fuer Coaching-Texte

- Die technische Bewertung bleibt deterministisch und nachvollziehbar.
- Die OpenAI-Erweiterung verbessert nur die Formulierung der Coaching-Texte.
- Als Eingabe fuer KI eignen sich die strukturierten Fakten: Phase, Ziel, Messwerte vorher/nachher, Status und Leistungsprofil.
- Die KI sollte keine Speed-Berechnung ersetzen und keine neuen Messwerte erfinden.
- Ablauf:
  - Analyse erzeugt Fakten und Zielmetriken lokal.
  - KI formuliert daraus einfache oder Experten-Coaching-Texte per OpenAI Responses API.
  - UI zeigt die KI-Texte als zusaetzlichen Coaching-Block.
  - Ohne KI-Schluessel funktioniert die bestehende lokale Analyse unveraendert weiter.

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
