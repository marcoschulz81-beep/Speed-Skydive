# Changelog

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
