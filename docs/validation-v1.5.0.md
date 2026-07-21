# Validierung App-Version 1.5.0

Datum: 21. Juli 2026

## Umfang

Version 1.5.0 ergänzt den profilweiten KI-Coach, ohne Berechnung, Regel-Score, Analyseversion oder gespeicherte Rohdaten zu verändern. Geprüft wurden Springerprofil, Profilvergleich und Einzelreport jeweils in einfacher und Expertenansicht sowie Cache-, Hintergrund- und Fehlerpfade.

## Automatisierte Prüfungen

- `python -m ruff check .`: bestanden
- `python -m mypy app/analysis/evaluation.py app/analysis/potential.py scripts`: bestanden
- `python -m pytest -q`: bestanden
- Gezielte KI-, Profil-, Template- und Persistenztests: bestanden

Die Regressionstests prüfen insbesondere:

- profilweiten, kompakten KI-Payload aus allen auswertbaren Sprüngen;
- standardmäßigen Ausschluss von Springername, Dateiname und Zeitstempel aus dem KI-Payload;
- getrennte einfache und technische Formulierungen;
- persistente Trennung nach Profilstand, Modell, Promptversion und Ansicht;
- sichtbaren Fallback auf die vollständige Regelanalyse;
- automatisches Status-Polling und Schutz vor doppelten Hintergrundaufträgen bei Refresh;
- vollständige Satzgrenzen bei begrenzten KI-Texten.

## Reale HTTP- und OpenAI-Integration

Der lokale Uvicorn-Server wurde mit dem neuen Stand auf `127.0.0.1:8000` neu gestartet. Für das größte vorhandene Springerprofil mit 184 Sprüngen wurden echte Hintergrundläufe ausgeführt:

| Seite | Ergebnis |
|---|---|
| Springerprofil, Experte | HTTP 200, zunächst Pending, anschließend `ready`; Profilfazit, Hebel, KI-Erklärung und Fokus sichtbar |
| Springerprofil, einfach | HTTP 200, zunächst Pending, anschließend `ready`; Kurzfassung, Erklärung und nächster Fokus sichtbar |
| Profilvergleich, Experte | HTTP 200; Vergleich und profilweiter Experten-Coach gleichzeitig sichtbar |
| Profilvergleich, einfach | HTTP 200; Vergleich und einfacher Profil-Coach gleichzeitig sichtbar |
| Einzelreport, Experte | HTTP 200, Hintergrundlauf `ready`; Coaching-Erklärung und KI-Fokus sichtbar |
| Einzelreport, einfach | HTTP 200, Hintergrundlauf `ready`; einfache Coaching-Erklärung und Trainingsfokus sichtbar |

Die finalen Profilantworten wurden zusätzlich auf vollständige Sätze, Profilbezug, verständliche Rundung in der einfachen Ansicht und vorhandene Vertrauenshinweise kontrolliert. Beide Profilvarianten sind persistent mit Status `ready` gespeichert.

## Unveränderte Fachlogik

- App-Version: `1.5.0`
- Analyseversion: weiterhin `1.1.0`
- Keine Änderung an Score-Fenstern, Metriken, Ranking, Referenzlogik oder Uploadanalyse
- Keine Migration oder Löschung bestehender Sprünge erforderlich
- Ohne OpenAI-Schlüssel oder bei API-Fehlern bleibt die deterministische Auswertung vollständig sichtbar
