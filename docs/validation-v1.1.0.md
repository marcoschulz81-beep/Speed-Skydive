# Validierungsprotokoll 1.1.0

Datum: 2026-07-18

## Umfang

Geprüft wurden Berechnung, Wertungs- und Validierungsfenster, Qualitätsstatus, Persistenz, Referenzauswahl, persönliche Lernlogik, Potenzialmodell, Coaching-Follow-up, HTML-Reporte und die bestehende SQLite-Datenbank.

Vor der Migration wurde eine bytegenaue Datenbankkopie außerhalb des Repositories angelegt und per SHA-256 gegen das Original geprüft.

## Automatisierte Prüfung

- `python -m pytest -q`: 133 Tests bestanden.
- `python -m ruff check .`: keine Befunde.
- `python -m mypy app/analysis/evaluation.py app/analysis/potential.py scripts`: keine Befunde.
- Alle Jinja-Templates werden in einem Test vollständig geparst.
- Regressionstests decken unter anderem exakt 3,0s lange Fenster, das 0,1s-Raster, Regel-Score-Status, Potenzialstart bei zehn historischen Sprüngen, Ausschluss des aktuellen Sprungs, sichere Inline-JSON-Ausgabe und Exit unterhalb Breakoff ab.

## Migration des realen Bestands

- Dry-Run: 363/363 Sprünge erfolgreich, 0 Fehler.
- Reanalyse: 363/363 Sprünge erfolgreich, 0 Fehler.
- Repariert: 2.534 verwaiste Sample-Zeilen und eine verwaiste Metrik.
- Final: 363 Sprünge, 363 Metriken, 769.438 Samples.
- Foreign-Key-Verletzungen final: 0.
- Alle Sprung- und Metrikdatensätze tragen Analyseversion `1.1.0`.
- Regel-Score-Status: 320 `valid`, 11 `estimated`, 32 `invalid`.
- Datenbank-Audit: 0 Befunde.

Das erste Audit fand fünf Aufzeichnungen, deren erkannter Exit bereits unter der Breakoff-Höhe lag. Die Pipeline wurde daraufhin korrigiert: Für diese Fälle werden Performance- und Validierungsende nicht mehr künstlich auf den Startzeitpunkt gesetzt. Sie erhalten jetzt `PERFORMANCE_WINDOW_INCOMPLETE` und keinen Regel-Score. Nach erneuter Vollreanalyse war das Audit fehlerfrei.

## Vollständige Report-Stichprobe

| Fall | Regel-Score | Status | Performance | Validierung | Ergebnis |
|---|---:|---|---|---|---|
| Marco Hepp, `09-42-21.CSV` | 516,40 | valid | 0,00–22,72s | 15,52–22,72s | referenzfähig, HTTP 200 |
| Marc Weinert, `26-05-30/12-39-30/TRACK.CSV` | 412,92 | valid | 0,12–26,58s | 17,62–26,58s | referenzfähig, HTTP 200 |
| Anna Nordin, `10-36-03.CSV` | 324,59 | estimated | 0,00–29,92s | 18,58–29,92s | wegen geschätzter Bodenhöhe ausgeschlossen, HTTP 200 |
| Dominik Plotz, `09-08-39.CSV` | 438,15 | invalid | 0,00–25,17s | 16,78–25,17s | wegen Zeitlücken ausgeschlossen, HTTP 200 |
| Dominik Plotz, `07-03-19.CSV` | – | invalid | Start 0,03s, kein Ende | – | Exit unter Breakoff, sichtbar ausgeschlossen, HTTP 200 |

Für alle fünf Fälle wurden aus den gespeicherten Samples erneut berechnet und mit der Persistenz verglichen:

- Training-3s und Regel-Score stimmen auf 0,01 km/h.
- vHor- und Winkelmittel des 3s-Fensters stimmen auf die gespeicherte Rundung.
- Jedes berechenbare Fenster ist exakt 3,0s lang; Regel-Fensterstarts liegen auf dem globalen 0,1s-Raster.
- Alle geprüften Fixpunkte haben 0,0 km/h Interpolationsabweichung.
- Die erneut berechnete Validierungsqualität stimmt exakt.
- Experten- und einfache Reports, Springerprofil und Vergleichsseite rendern mit HTTP 200.
- Ungültige und geschätzte Werte werden sichtbar als nicht bestwert-, referenz- und lernfähig gekennzeichnet.

## Lern- und Modellprüfung

Das größte geprüfte Profil, Marco Hepp, enthält 99 Sprünge. Davon gehen 94 saubere gültige Sprünge in Lernen und Profil ein; fünf werden ausgeschlossen. Der persönliche Bestwert ist der gültige Regel-Score von 516,40 km/h. Stabile und instabile Gruppen sind disjunkt und werden nur bei echter Evidenz gebildet.

Das Potenzialmodell für den schnellsten Sprung nutzte 93 historische saubere Sprünge; der aktuelle Sprung war ausgeschlossen. Leave-one-out ergab RMSE 13,71 km/h und MAE 6,70 km/h. Deshalb bleibt die angezeigte Sicherheit trotz großer Datenmenge bewusst `niedrig`. Das Modell ist als experimentell gekennzeichnet und ersetzt weder Regel-Score noch Coaching-Regeln.

## Bewusste Übergangsentscheidungen

- Coaching-Snapshots bleiben bis zum Go-live aktualisierbar. Eine Reanalyse entfernt den alten Snapshot; das nächste Öffnen des Expertenreports schreibt ihn mit der aktuellen Logik neu.
- Horizontale und seitliche GPS-Kennwerte sind bodenrelativ und nicht windkorrigiert. Report und Vergleich kennzeichnen das ausdrücklich.
- Authentifizierung und CSRF-Härtung bleiben Teil des Go-live-Schritts. Das Upload-Limit und die sichere Inline-JSON-Serialisierung sind bereits aktiv.
