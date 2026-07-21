# Validierung App-Version 1.5.1

Datum: 21. Juli 2026

## Umfang

Version 1.5.1 ergänzt einen bewertungsfreien KI-Ladezustand und behebt das abgebrochene automatische Nachladen beim Wechsel zwischen einfacher und Expertenansicht. Score-Regeln, Analyseversion, Cache-Schlüssel und gespeicherte Sprungdaten bleiben unverändert.

## Automatisierte Prüfungen

- `python -m ruff check .`: bestanden
- `python -m mypy app/analysis/evaluation.py app/analysis/potential.py scripts`: bestanden
- `python -m pytest -q`: 164 Tests bestanden

Die neuen Regressionstests prüfen:

- Pending-HTML enthält in einfacher und Expertenansicht keine Scores, Coaching-Bewertung, Scorecard, Kurven oder Diagnose;
- Springerprofile verbergen während Pending Profilwerte, Vergleiche und Trenddiagramme;
- das lokale SVG ist valides XML und ohne externe Bildabhängigkeit eingebunden;
- `pending`, `ready` und `error` bleiben über additive Statusfelder unterscheidbar;
- das gemeinsame Polling besitzt keine 20-/45-Sekunden-Abbruchgrenze, schaltet nach 30 Sekunden auf ein langsameres Intervall und unterstützt den Wiederherstellungs-Reload;
- identische Refreshs erzeugen weiterhin nur einen Hintergrundauftrag.

## Reale Integrationsprüfung

Die laufende Anwendung wurde mit einem vorhandenen Sprung geprüft, dessen einfache
KI-Auswertung bereits im Cache lag, während die Expertenauswertung noch fehlte:

- App-Version `1.5.1` wurde über die laufende API bestätigt;
- die einfache Ansicht zeigte weiterhin ihren vorhandenen KI-Text;
- der Wechsel zur Expertenansicht zeigte ausschließlich Metadaten und den neuen
  SVG-Ladezustand, aber keine Scores, Coachings, Diagramme oder Diagnose;
- der Status-Endpunkt meldete zunächst
  `{"state":"pending","complete":false,"available":false}`;
- nach 18 Statusabfragen meldete er `ready`, anschließend enthielt die automatisch
  neu geladene Expertenansicht den passenden KI-Text und keinen Ladezustand mehr;
- beim Rückwechsel blieb der KI-Text der einfachen Ansicht erhalten;
- das lokale SVG wurde mit HTTP 200 und `image/svg+xml` ausgeliefert.

## Datenintegrität

- `PRAGMA integrity_check`: `ok`
- `PRAGMA foreign_key_check`: keine Fehler
- Anzahl gespeicherter Sprünge vor dem Release: 363
- keine Datenbankmigration und keine Änderung an bestehenden Sprungdaten

## Fallback und Kompatibilität

- Bei `ready` erscheint die vollständige Bewertung einschließlich der zur Ansicht passenden KI-Texte.
- Bei `error`, deaktivierter KI oder fehlendem API-Key erscheint die vollständige Regelbewertung mit Hinweis.
- Bereits persistierte KI-Ergebnisse werden ohne Ladeansicht verwendet.
- Die bisherigen Statusfelder `complete` und `available` bleiben erhalten; `state` wird additiv ergänzt.
- Keine Änderung an Analyseversion `1.1.0`.
