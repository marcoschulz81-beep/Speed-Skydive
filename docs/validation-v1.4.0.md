# Validierung App-Version 1.4.0

Datum: 20.07.2026

## Umfang

Geprüft wurden der unveränderte fachliche Output der Analyse, der einmalige Uploadpfad, die vektorisierte 3s-Suche, additive SQLite-Migrationen, kompakte Messreihen, Feature-/Profil-Snapshots, deren Invalidierung, das asynchrone KI-Coaching, der vollständige Datenbackfill und die Reportlaufzeit.

## Sicherungen vor Änderungen

- Konsistentes Gesamtbackup: `C:\Users\marco\Speed-Skydive-backups\pre-v1.4.0-20260720-1415`
- SQLite SHA-256: `07ecdcc692d1e3f01dd6168735752793034cfa3c53ee155de64eddffcbc998b6`
- 361 SHA-256-deduplizierte Quelldateien, zusammen 448.562.412 Bytes
- Der schreibende Backfill legte zusätzlich unmittelbar vor der finalen Feature-Regeneration `speed_skydive-pre-v1.4.0-backfill-20260720-182615.db` an.

## Vorabprüfung v1.3.0

- 149 Tests bestanden.
- Ruff, Mypy im dokumentierten CI-Umfang und `compileall` bestanden.
- `PRAGMA quick_check`: `ok`
- Foreign-Key-Verletzungen: 0
- Bestand: 363 Sprünge, 769.438 Samples, 363 Metrics

## Golden Master der Score-Analyse

Version 1.3.0 und v1.4.0 analysierten dieselben realen Quelldateien mit identischen Eingaben. Gehasht wurden Jump-Metadaten ohne zufällige ID/Signatur, Metrics, Fixpunkte, Phasen, Scorecard, Tipps und Notes.

| Datei | Samples | SHA-256 v1.3.0/v1.4.0 | v1.3.0 | v1.4.0 |
|---|---:|---|---:|---:|
| `10-31-44.CSV` | 1.679 | `f417f48057b419a0e8f057a46725093bec0aed187439f30d74f6f67ac6be5c47` | 0,1765 s | 0,0843 s |
| `14-18-11.CSV` | 3.000 | `ab8886f347fa6ff3188107916574c2b337d0d10a9c8022872a3e874239116423` | 0,2177 s | 0,1367 s |
| `12-01-09.CSV` | 72.620 | `6279bc3fa98c7919c74236efb7af8bbe22fb9d272d9e07e0f11adce8c693c426` | 2,6517 s | 0,7900 s |

Alle drei Hashes, Regel-Scores und Status stimmen exakt überein. Die Analyseversion bleibt `1.1.0`.

Der größte Datensatz wurde außerdem über den vollständigen Dropzone-Wrapper mit manueller Katalogzuordnung geprüft:

- v1.3.0: 5,525 Sekunden
- v1.4.0: 1,048 Sekunden
- identischer Ergebnis-Hash: `ff059ff713da3731cea01e6013ccfcac60c1df87880d663575b76c9be538d921`
- identischer Regel-Score: 225,36 km/h

## Migrationsprobe auf Datenbankkopie

Vor der echten Migration wurde der vollständige Ablauf auf einer Kopie der Produktionsdatenbank ausgeführt:

- Laufzeit: 15,695 Sekunden
- 363/363 kompakte Serien erstellt
- 363/363 Feature-Records erstellt
- sieben Springerprofile erstellt
- `quick_check=ok`
- Foreign-Key-Verletzungen: 0
- fehlende oder beschädigte Serien: 0

Eine absichtlich durch das Testwerkzeug unterbrochene erste Produktionsausführung ließ sich idempotent fortsetzen. Die abschließende Regeneration lief nach einem neuen Backup vollständig durch.

## Produktionsbestand nach Backfill

| Tabelle/Prüfung | Ergebnis |
|---|---:|
| `jumps` | 363 |
| `samples` | 769.438 |
| `metrics` | 363 |
| `jump_series` | 363 |
| Punkte in `jump_series` | 106.031 (13,78 % der vollständigen Samples) |
| komprimierte Serien-Payloads | 4.691.680 Bytes |
| `jump_analysis_features` | 363 |
| `jumper_profile_snapshots` | 7 |
| fehlende/beschädigte Serien | 0/0 |
| SQLite-Modus | WAL |
| `quick_check` | `ok` |
| Foreign-Key-Verletzungen | 0 |

Die vollständigen 769.438 Sample-Zeilen wurden nicht gelöscht. Reports können bei jeder fehlenden, unbekannten oder beschädigten Serie auf sie zurückfallen.

Für alle 363 historischen Lernrecords wurde zusätzlich ein Golden-Master-Hash über die vollständigen Record-JSONs gebildet. Der Hash aus v1.3.0 und der final gespeicherte v1.4.0-Hash stimmen exakt überein:

`b95c39611224f34ff7f5c01150ef0b1c315377f0468de9401419683bfe418a7d`

## Report- und KI-Integration

Getestet wurde ein Expertenreport für den größten Springerbestand mit 184 Sprüngen und deaktiviertem externem KI-Aufruf:

- alter Kaltstart: rund 24,0 Sekunden
- v1.4.0 Kaltstart in einem neuen Prozess: 0,173 Sekunden
- v1.4.0 Warmaufruf: 0,119 Sekunden
- HTTP-Status jeweils 200

Der asynchrone KI-Pfad wurde mit einem kontrollierten Test-Client geprüft:

1. Erster Report liefert sofort HTTP 200 und einen Pending-Status.
2. Hintergrundtask speichert das Ergebnis.
3. Status-Endpoint meldet `complete=true`, `available=true`.
4. Der nächste Report liefert den persistent gespeicherten KI-Text.

## Abschließende lokale Tests

- 154 Tests bestanden.
- Ruff: bestanden.
- Mypy im dokumentierten Umfang: bestanden.
- `compileall` für `app`, `scripts` und `tests`: bestanden.
- Erster GitHub-Actions-Lauf `29759957553`: alle Schritte in 41 Sekunden bestanden. Die dabei gemeldete Node-20-Abkündigung wurde durch das Upgrade der offiziellen Actions auf ihre Node-24-Versionen behoben; der aktualisierte Lauf ist die abschließende externe Release-Prüfung.
