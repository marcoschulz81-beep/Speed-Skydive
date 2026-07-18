# Validierung v1.3.0

Stand: 2026-07-18

## Ziel

Version 1.3.0 ergänzt eine ausschließlich lesende Dropzone-Katalogoberfläche. Die bestehende
Upload-Erkennung, Katalogpflege, Score-Berechnung, Auswertung und Lernlogik werden nicht verändert.

## Abnahmekriterien

- `/dropzones` zeigt den vollständigen Katalog und filtert nach Freitext, Land und Status.
- Freitext findet Namen, Orte, Regionen, ICAO-/Flugplatzkennungen und Betreiber.
- `/dropzones/{dropzone_id}` zeigt Stammdaten, Höhen, Zonen, Betreiber, Quellen und aggregierte Nutzung.
- Unbekannte Dropzone-IDs liefern HTTP 404.
- Externe Links werden nur für HTTP(S) ausgegeben.
- Springernamen, Dateinamen und einzelne Sprung-IDs erscheinen nicht in Dropzone-Detailseiten.
- Sprungreports verlinken eine zugeordnete Dropzone zur Detailseite.
- Katalog- und Datenbank-Audits bleiben fehlerfrei; Regel-Score-Verteilung und gespeicherte Zuordnungen
  bleiben gegenüber v1.2.0 unverändert.

## Prüfbefehle

```powershell
python -m pytest
python -m ruff check .
python -m mypy app/analysis/evaluation.py app/analysis/potential.py app/services/dropzone_matching.py app/services/dropzone_directory.py scripts
python -m scripts.dropzone_catalog --audit --json
python -m scripts.dropzone_catalog --audit-matches --json
python -m scripts.audit_database --json
```

## Ergebnis

- Pytest: `148 passed`
- Ruff: keine Befunde
- Mypy: keine Befunde in den geprüften Analyse-, Dropzone- und Skriptmodulen
- Katalogaudit: 89 Dropzones, 91 Zonen, 81 Betreiber, 346 Quellen, keine Probleme
- Match-Audit: 363 Versuche, 349 akzeptiert, 14 mit unzureichenden Daten, keine offenen Fälle
- Datenbankaudit: 363 Sprünge und Metriken, 769.438 Samples, keine Fremdschlüsselverletzung,
  keine Integritätsprobleme
- Regel-Score-Status unverändert: 330 `valid`, 2 `estimated`, 31 `invalid`

Die HTTP-Stichprobe umfasste die vollständige Übersicht, Deutschland-/Statusfilter, Freitextsuche,
vier stark genutzte Dropzone-Detailseiten, vier zugeordnete Sprungreports und zwei nicht zugeordnete
Sprungreports. Alle Seiten lieferten HTTP 200; ein unbekannter Detailpfad lieferte erwartungsgemäß 404.
Listenanzahlen, Detailquellen, Zonen, Betreiber und Zuordnungszahlen entsprachen jeweils direkt den
Datenbankabfragen. Alle extern ausgegebenen Links waren HTTP(S), und die Katalogdetailseiten enthielten
keine Springernamen oder einzelnen Sprung-IDs.
