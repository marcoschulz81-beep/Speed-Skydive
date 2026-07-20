# Architektur und Ausbaupfad ab App-Version 1.4.0

## Zielbild

Uploads, Reports und persönliche Lernprofile sollen mit wachsendem Datenbestand schnell bleiben, ohne Score-Regeln, historische Zusammenhänge oder Rohdaten zu verlieren. Version 1.4.0 setzt deshalb zuerst die gemessenen Engpässe um und hält die Datenmigration additiv und rückrollbar.

```text
CSV/FS2-Log
   │
   ├─ einmal parsen, normalisieren, validieren
   │             │
   │             ├─ stabile Bodenbeobachtung und Dropzone-Match
   │             └─ vektorisierte deterministische Analyse
   │
   ├─ Originaldatei + normalisierte Samples (Wiederherstellungsquelle)
   ├─ kompakte technische Serie (schneller Reportpfad)
   ├─ versionierte Analyse-Features (schnelle Lernlogik)
   └─ Springerprofil-Snapshot
                    │
                    └─ Report sofort; KI-Text asynchron und persistent gecacht
```

## Ausgangsmessung

- 363 Sprünge, 769.438 normalisierte Samples und 361 erreichbare eindeutige Quelldateien.
- SQLite-Integrität vor der Änderung: `quick_check=ok`, keine Foreign-Key-Verletzung.
- Die Sample-Persistenz war nicht der Upload-Engpass: rund 0,013 Sekunden bei 1.679 Samples und 0,257 Sekunden bei 72.620 Samples in einer Testdatenbank.
- Ein akzeptierter Dropzone-Match führte bisher zu zwei vollständigen Analyseläufen.
- Die alte 3s-Suche prüfte jedes 0,1s-Fenster mit drei separaten Interpolationen.
- Ein kalter Report für einen Springer mit 184 Sprüngen lud und berechnete hunderte Reports erneut und benötigte rund 24 Sekunden.

## In v1.4.0 umgesetzt

### Einmalige Upload-Vorbereitung

`prepare_flysight_csv` übernimmt Einlesen, Einheitenkorrektur, Plausibilitätsprüfung, Geräteerkennung, Qualitätsbasis und automatische t0-Erkennung einmalig. Die Dropzone-Erkennung nutzt daraus nur die benötigten Bodenfelder. Die finale Bewertung arbeitet mit demselben vorbereiteten Objekt.

### Vektorisierte 3s-Suche

Alle Kandidaten bleiben auf dem bisherigen globalen 0,1s-Raster. Interpolierte Reihen und trapezförmige Segmentflächen werden einmal aufgebaut; Präfixsummen liefern danach alle 3s-Mittelwerte. Zeitlücken, das früheste Fenster bei Gleichstand und die bisherigen Grenzen bleiben identisch.

Die Score-Engine bleibt deshalb auf Analyseversion `1.1.0`. Drei reale Golden-Master-Fälle mit 1.679, 3.000 und 72.620 Samples ergaben über Jump-Metadaten, Metrics, Fixpunkte, Phasen, Scorecard, Tipps und Notes exakt dieselben SHA-256-Hashes wie v1.3.0.

### Additive Speicherebenen

| Ebene | Zweck | Verlust-/Rollback-Schutz |
|---|---|---|
| Original-CSV | unveränderte Quelle für Reanalyse | bleibt erhalten |
| `samples` | vollständiges normalisiertes Datenmodell | bleibt in v1.4.0 erhalten |
| `jump_series` | komprimierte, reportrelevante Float64-Reihe | Fallback auf `samples` bei Fehlen/Fehler |
| `jump_analysis_features` | kompakte Eingaben und fertige Lernrecords | versions- und Referenzsignatur |
| `jumper_profile_snapshots` | fertige Trends, Referenzen und Profile | History- und Referenzsignatur |
| `ai_coaching_results` | KI-Ergebnis über Neustarts hinweg | Analyse-, Modell-, Prompt- und Ansichtsbindung |

Die kompakte Serie endet nach allen relevanten Analyse-, Kurven-, Brems- und Referenzfenstern, mindestens aber nach 30 Sekunden. Vollständige Samples und Originaldatei bleiben verfügbar; v1.4.0 löscht keine Altdaten.

### Konsistenz und Invalidierung

- Kontext- und Feedbackänderungen entfernen betroffene Analyse-Features, KI-Ergebnisse und Springerprofil-Snapshots.
- Eine Reanalyse ersetzt den Sprung unter derselben Sprung-ID und baut abhängige Daten neu auf.
- Änderungen an der Marco-Top-15-Referenz erzeugen eine neue Referenzsignatur; Records mit alter Signatur werden nicht verwendet.
- Neue oder gelöschte Sprünge ändern die History-Signatur des Springers.
- Alle neuen Tabellen referenzieren den Sprung mit Foreign Keys und `ON DELETE CASCADE`, soweit die Daten einem einzelnen Sprung gehören.

### Migration und Betrieb

`schema_migrations` protokolliert nummerierte, wiederholbare Migrationen. Der v1.4-Backfill ist standardmäßig ein Dry-Run und erzeugt mit `--apply` zuerst ein konsistentes SQLite-Backup. Danach folgen kompakte Serien, Features, Profile, `quick_check` und `foreign_key_check`.

SQLite läuft im lokalen Einzelserverbetrieb mit WAL, `busy_timeout=5000`, `synchronous=NORMAL` und regelmäßigem WAL-Autocheckpoint. WAL darf nicht auf einem Netzwerk-Dateisystem betrieben werden.

## Warum derzeit kein Datenbankwechsel

Die Messung zeigt CPU- und Wiederholungsarbeit in Python als Engpass, nicht SQLite. Ein Wechsel auf PostgreSQL würde die doppelte Analyse, die alte Fensterschleife oder die hunderten Report-Neuberechnungen nicht beseitigen. Für einen einzelnen App-Server mit wenigen Schreibvorgängen ist SQLite weiterhin einfacher, schneller zu sichern und betrieblich risikoärmer.

PostgreSQL wird sinnvoll, sobald mindestens einer dieser Trigger eintritt:

- mehr als eine App-Instanz oder mehrere Worker schreiben gleichzeitig;
- wiederkehrende `database is locked`-Fehler oder messbare Schreibwartezeiten trotz WAL;
- zentrale Datenhaltung über mehrere Hosts wird erforderlich;
- Benutzerkonten, Mandanten, Rechte oder umfangreiche serverseitige Abfragen benötigen stärkere Nebenläufigkeit;
- Analysejobs werden auf verteilte Worker ausgelagert.

Der spätere Wechsel erfolgt über die bestehende Storage-Funktionsgrenze: zuerst PostgreSQL-Implementierung und Dual-Read-Test, dann vollständiger Export mit Zeilen- und Hash-Abgleich, anschließend kurzer schreibgeschützter Umschaltzeitraum. Rohdateien gehören bei mehreren Hosts in versionierten Object Storage; die Datenbank speichert Hash, Größe, Schlüssel und Importstatus.

## Nächste Ausbaustufen

### 1. Beobachtbarkeit vor weiterer Technik

Zu messen sind Upload-Gesamtzeit, Parse-/Analyse-/Persistenzzeit, Reportzeit kalt/warm, Cache-Treffer, KI-Laufzeit, SQLite-Busy-Zeit, Backfill-Fehler und Größenwachstum. Sinnvolle Startziele sind p95 unter 3 Sekunden für deterministische Uploadanalyse und p95 unter 1 Sekunde für einen gecachten Report auf der aktuellen Hardware.

### 2. Deterministische Job-Queue nur bei Bedarf

Wenn große Dateien trotz der Optimierung das Uploadziel überschreiten oder mehrere Nutzer parallel analysieren, wird der deterministische Lauf als idempotenter Job gespeichert: `queued`, `running`, `ready`, `failed`. Der Dateihash plus Analysesignatur bildet den Idempotenzschlüssel. Der Browser kann dann wie beim KI-Coaching pollen. Für einen einzelnen Prozess genügt zunächst eine SQLite-Jobtabelle; verteilte Worker sind gleichzeitig ein PostgreSQL-Trigger.

### 3. Stabile Springeridentität

Vor Benutzerkonten wird eine additive `jumpers`-Tabelle mit stabiler `jumper_id` eingeführt. Bestehende Namen werden kontrolliert zugeordnet; unklare Dubletten werden nicht automatisch zusammengeführt. `jumper_name` bleibt während einer Übergangsrelease lesbar. Damit überstehen Profile Namensänderungen und spätere Mandantenmodelle.

### 4. Kontrollierte Datenbereinigung

Normalisierte `samples` werden frühestens in einer späteren Release entfernt, wenn mindestens eine vollständige Betriebsperiode, Restore-Test, Serienaudit und Reanalyse aus Originaldateien bestanden sind. Bis dahin ist die neue Serie ein zusätzlicher schneller Lesepfad, kein Ersatz ohne Rückweg.

### 5. PostgreSQL und Object Storage

Nach Eintritt eines Triggers werden zuerst Repository-Verträge und Integrationstests gegen SQLite und PostgreSQL parallel ausgeführt. Erst nach identischen fachlichen Resultaten, vollständigem Foreign-Key-Audit, Quellhash-Abgleich und erfolgreichem Restore wird die Produktion umgeschaltet.

## Pflichtprüfungen pro künftiger Release

1. Externes konsistentes DB- und Rohdatei-Backup.
2. `pytest`, Ruff, Mypy und `compileall` vor und nach der Migration.
3. Golden Master aus kleinen, mittleren, großen, lückenhaften, geschätzten und manuell korrigierten Sprüngen.
4. `quick_check`, `foreign_key_check`, Tabellen- und Sprunganzahlen vor/nachher.
5. Hash-/Zeilenabgleich für migrierte Features und Restore-Probe.
6. Kalt-/Warmmessung von Upload, Einzelreport, Springerprofil und Vergleich.
7. Erst danach Commit, Push und grüne GitHub-CI.
