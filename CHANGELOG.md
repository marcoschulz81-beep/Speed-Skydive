# Changelog

## 1.2.0 - 2026-07-18

### Dropzone-Katalog

- Versionierten Seed mit 89 physischen Sprungplätzen, 91 Landezonen, 81 deutschen Betreibern und 346 feldbezogenen Quellenbelegen ergänzt.
- Eigenständiges Datenmodell für Dropzones, Zonen, Betreiber, Zuordnungen, Quellen, historische GNSS-Beobachtungen und Katalog-Metadaten eingeführt.
- Reproduzierbaren Validierungs-, Import- und Audit-Befehl ergänzt; Seed-Import ist idempotent und schützt manuell verifizierte Bodenhöhen.
- Strenge lokale Upload-Erkennung auf Basis stabiler Boden-GNSS-Sequenzen ergänzt; automatische Nutzung ist auf eindeutige `trusted`/`verified`-Treffer bis 750 m und mindestens 0,75 Konfidenz begrenzt.
- Match-Versuche mit Algorithmus- und Katalogversion, Position, Höhe, Streuung, Entfernung, Konfidenz und Status revisionssicher gespeichert; unbekannte, mehrdeutige und unsichere Orte bleiben in einer lokalen Prüfliste.
- Manuelle Bodenhöhen behalten Vorrang. Dropzones können im Expertenreport manuell bestätigt oder erneut automatisch erkannt werden.
- 349 von 363 Altsprüngen sicher zugeordnet; 338 manuelle Bodenhöhen unverändert bewahrt, elf alte Schätzwerte durch Kataloghöhen ersetzt und 14 unzureichende Sequenzen unverändert belassen.
- Regel-Score-Zahlen blieben unverändert; durch gesicherte Bodenhöhen wechselten neun Status von `estimated` zu `valid` und einer von `invalid` zu `valid`.
- 84 Einträge als `trusted` und fünf bewusst als `candidate` klassifiziert; alle Integritäts- und Quellenabdeckungsprüfungen sind fehlerfrei.

### Versionierung

- App-Version auf `1.2.0` erhöht. Die mathematisch unveränderte Score-Engine bleibt korrekt als Analyseversion `1.1.0` gekennzeichnet; Dropzone, Katalogrevision und Zuordnungsart sind separat Teil der Analysesignatur.

## 1.1.0 - 2026-07-18

### Geändert

- Regel-Score auf ein global verankertes 0,1s-Raster, exakt 3,0 Sekunden und einheitliche zeitgewichtete Mittelwerte umgestellt.
- Performance- und 1006m-Validierungsfenster mit exakten interpolierten Grenzzeiten eingeführt; Standard-Breakoff ist 1707m AGL.
- Status `valid`, `estimated` und `invalid` samt maschinenlesbaren Gründen für jeden Regel-Score ergänzt.
- Rankings, Referenzen, persönliche Bestwerte und Lernprofile auf saubere gültige Regel-Scores begrenzt.
- Potenzialmodell auf mindestens zehn historische saubere Sprünge, Ausschluss des aktuellen Sprungs, Regel-Score-Ziel, Ridge-Regularisierung und Leave-one-out-Validierung umgestellt.
- Persönliche Stabilitätsreferenz verlangt mindestens fünf saubere Sprünge und erfindet keine stabilen/instabilen Gruppen mehr.
- Follow-up-Status auf `Verbessert`, `Verschlechtert`, `Unverändert`, `Uneindeutig` und `Nicht messbar` umgestellt; gruppierte Ziele behalten alle Messwerte.
- Gemeinsame technische Auswertungsgrenze für Report, Review, Vergleiche und Lernlogik eingeführt.
- Reanalyse bewahrt Bodenhöhenquelle, Breakoff und manuelle t0-Einstellung; Analyseversion und Signatur werden gespeichert.
- Reportkennzeichnung für Analyseversion, Score-Status, Validierungsfenster sowie bodenrelative, nicht windkorrigierte GPS-Werte ergänzt.

### Sicherheit und Betrieb

- Konfigurierbares Upload-Limit mit 25 MiB Standard ergänzt.
- JSON in Inline-Scripts gegen HTML-End-Tags escaped.
- FastAPI-Startup auf Lifespan umgestellt.
- Reanalyse- und Datenbank-Auditwerkzeuge ergänzt.
- CI für Tests, Ruff und Mypy ergänzt.

### Bewusste Übergangsentscheidung

- Coaching-Snapshots bleiben bis zum Go-live überschreibbar und werden nach einer Reanalyse mit der neuen Logik neu aufgebaut.
