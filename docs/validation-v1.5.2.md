# Validierung App-Version 1.5.2

Datum: 21. Juli 2026

## Umfang

Version 1.5.2 animiert das lokale SVG des KI-Analyse-Platzhalters. Springer,
Geschwindigkeitslinien, Außenring und Ladepunkte zeigen während einer laufenden
Auswertung kontinuierliche Aktivität. KI-Polling, Cache, Analyse und gespeicherte
Sprungdaten bleiben unverändert.

## Automatisierte Prüfungen

- `python -m ruff check .`: bestanden
- `python -m mypy app/analysis/evaluation.py app/analysis/potential.py scripts`: bestanden
- `python -m pytest -q`: 165 Tests bestanden
- `node --check app/static/ai-loading.js`: bestanden
- Python-Kompilierung und XML-Validierung des lokalen SVGs: bestanden

Die Regressionstests prüfen insbesondere die drei SVG-Animationen, den weiterhin
aktiven Außenring und die Ladepunkte sowie den statischen Fallback über
`prefers-reduced-motion`.

## Browser- und Integrationsprüfung

- Zwei Headless-Chrome-Aufnahmen bei 100 und 900 Millisekunden unterscheiden sich
  im normalen Modus um 2.857 Bildpunkte: Die SVG-Animation läuft tatsächlich.
- Dieselben Aufnahmen mit emuliertem `prefers-reduced-motion: reduce` unterscheiden
  sich um null Bildpunkte: Der barrierearme statische Fallback greift vollständig.
- Die laufende App liefert Version `1.5.2` und das SVG mit HTTP 200 sowie
  `image/svg+xml` aus.
- Eine zuvor fehlende Expertenauswertung zeigte den Ladezustand ohne Score,
  Coaching-Bewertung oder Plotly-Inhalte. Nach 26 Statusabfragen wechselte der
  Endpunkt von `pending` zu `ready`; danach verschwand der Platzhalter und die
  fertige KI-Bewertung einschließlich Fokus wurde dargestellt.
- Die gemeinsame Loader-Komponente und ihr Pending-Verhalten sind zusätzlich für
  einfache und Expertenansicht sowie das Springerprofil durch Regressionstests
  abgedeckt.

## Datenintegrität

- `PRAGMA integrity_check`: `ok`
- `PRAGMA foreign_key_check`: keine Fehler
- Anzahl gespeicherter Sprünge: 363

## Kompatibilität

- Keine Änderungen an öffentlichen Endpunkten oder Statusantworten.
- Keine Datenbankmigration und keine Änderung an Analyseversion `1.1.0`.
