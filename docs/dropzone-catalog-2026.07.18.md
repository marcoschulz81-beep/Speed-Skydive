# Dropzone-Katalog 2026.07.18-1

## Ergebnis

Der erste versionierte Dropzone-Katalog ist aufgebaut, in die lokale SQLite-Datenbank importiert
und vollständig auditiert. Dieser Schritt schafft ausschließlich die Datenbasis. Die automatische
Zuordnung eines neuen Uploads ist noch nicht aktiviert; bestehende Sprünge, Scores, Lernprofile,
Coaching-Snapshots und Reports wurden nicht verändert oder neu berechnet.

| Objekt | Anzahl |
|---|---:|
| Physische Dropzones | 89 |
| Räumliche Zonen | 91 |
| Betreiber | 81 |
| Betreiber-Zuordnungen | 81 |
| Feldbezogene Quellenbelege | 346 |
| Akzeptierte historische GNSS-Beobachtungen | 349 |
| Status `trusted` | 84 |
| Status `candidate` | 5 |

Die 81 deutschen Betreiber wurden nach dem zugehörigen physischen Flugplatz gruppiert. Dadurch
entstehen 79 deutsche Dropzones. Magdeburg City und Rottweil-Zepfenhan besitzen jeweils zwei
Betreiber. Zehn aus dem lokalen Sprungbestand belegte Plätze in weiteren Ländern ergänzen den
Katalog auf insgesamt 89 physische Orte.

## Quellen und zulässige Verwendung

Der Seed speichert nur einzelne Fakten und deren Herkunft, keine Beschreibungstexte, Bilder oder
Kontaktlisten fremder Websites.

| Quelle | Verwendung | Vertrauensstufe | Belege |
|---|---|---|---:|
| [Sky-Junkies Sprungplatzkarte](https://www.sky-junkies.de/sprungplatzkarte/) | Auffinden deutscher Betreiber und Verzeichnispositionen | `discovery` | 81 |
| Verlinkte Betreiber-Websites | Primärbeleg für die Betreiber-Website | `primary` | 81 |
| [OurAirports](https://ourairports.com/data/) | Flugplatzkennung, veröffentlichte Platzhöhe, Region und Ort | `supporting` | 85 |
| [OpenTopoData](https://www.opentopodata.org/api/) | Unabhängige DEM-Geländehöhe an der Zone | `supporting` | 89 |
| Lokaler FlySight-Bestand | Historisch beobachtete Position und Nutzungshäufigkeit | `supporting` | 10 zusammengefasste Quellen sowie 349 Einzelbeobachtungen |

Sky-Junkies dient bewusst nur als Ausgangsverzeichnis. Die dort veröffentlichten Koordinaten werden
nicht als amtliche Vermessung behandelt. OurAirports stellt seinen Datensatz als Public Domain bereit.
OpenTopoData liefert interpolierte Geländehöhen aus dem jeweils verfügbaren Höhenmodell. Diese Werte
sind belastbare technische Anhaltspunkte, aber keine vermessene Höhe einer konkreten Landezone.

Der Katalog wurde am 18. Juli 2026 recherchiert. Jeder Quellenbeleg enthält Quelle, URL oder Referenz,
Abrufdatum, Vertrauensstufe und die konkret unterstützten Felder. Der importierte Seed wird zusätzlich
mit SHA-256 `ae52242a361ca1e5dc1c55c9fddf180ed00fb369723591758daf5629e19c517e`
identifiziert.

## Vertrauenslogik

Die Zustände sind absichtlich konservativ:

- `candidate`: Ort ist plausibel, aber noch nicht stark genug für eine automatische produktive Nutzung belegt.
- `trusted`: mehrere unabhängige technische Quellen stimmen innerhalb der definierten Toleranz überein.
- `verified`: für spätere manuelle oder amtliche Freigabe reserviert; solche Werte werden vom Seed-Import geschützt.
- `inactive`: historischer oder nicht mehr aktiver Platz; wird bei der Erkennung ausgeschlossen.

Für deutsche Verzeichniseinträge wird `trusted` nur vergeben, wenn ein passender Flugplatzdatensatz,
eine veröffentlichte Flugplatzhöhe und eine DEM-Höhe vorhanden sind und die beiden Höhen höchstens
20 Meter voneinander abweichen. Historisch belegte ausländische Plätze benötigen eine DEM-Höhe und
zusätzlich entweder einen passenden Flugplatzdatensatz oder mindestens drei lokale Beobachtungen.

Fünf Einträge bleiben Kandidaten:

| Dropzone | Grund |
|---|---|
| Fallschirmsportclub Calw | Kein ausreichend naher, eindeutig passender Flugplatzdatensatz mit veröffentlichter Höhe |
| Fallschirmsportspringerclub Oberhausen e. V. | Kein ausreichend naher, eindeutig passender Flugplatzdatensatz mit veröffentlichter Höhe |
| Skydive Bad Lippspringe | Kein ausreichend naher, eindeutig passender Flugplatzdatensatz mit veröffentlichter Höhe |
| Sonderlandeplatz Meißendorf-Brunsiek | Flugplatzdatensatz gefunden, aber keine veröffentlichte Höhe im Register |
| Gelnhausen Airfield | DEM-Höhe 167,6 m und veröffentlichte Platzhöhe 135,9 m weichen um 31,7 m ab |

Kandidaten besitzen weiterhin Koordinaten, Geländehöhe und Quellen, werden aber in einer späteren
automatischen Zuordnung nicht wie freigegebene Plätze behandelt.

## Historische GNSS-Nachweise

Der Historienimport prüfte alle 363 vorhandenen Sprünge. 357 Sprünge enthielten grundsätzlich
geeignete langsame GNSS-Samples. 349 davon bildeten eine stabile Bodensequenz innerhalb einer
bekannten Katalogzone; acht wurden wegen fehlender Stabilität nicht übernommen. Bei keinem
akzeptierten Nachweis lag die Position außerhalb der zugehörigen Zone. Sechs weitere Sprünge
enthielten bereits keine grundsätzlich geeignete Sequenz.

Eine Bodensequenz muss mindestens fünf Sekunden dauern und folgende Grenzen einhalten:

- mindestens 3D-GPS-Fix und sechs Satelliten,
- Gesamtgeschwindigkeit höchstens 10 km/h,
- Betrag der vertikalen Geschwindigkeit höchstens 1 m/s,
- `sAcc` höchstens 3 m/s und `vAcc` höchstens 10 m, sofern vorhanden,
- höchstens 0,35 Sekunden Abstand zwischen aufeinanderfolgenden gültigen Samples,
- Höhen-MAD höchstens 5 m,
- 95-Prozent-Radius der Positionspunkte höchstens 100 m.

Die 349 Nachweise verteilen sich auf 18 Plätze:

| Dropzone | Nachweise |
|---|---:|
| Schlierstadt-Seligenberg | 110 |
| Fano | 51 |
| Günzburg-Donauried | 49 |
| Bad Saulgau | 29 |
| Hohenems-Dornbirn | 24 |
| Skydive Dubai Desert Campus | 18 |
| Portimão / Alvor | 11 |
| Gransee | 10 |
| Netheravon | 10 |
| Neustadt-Glewe | 9 |
| Michael J. Smith Field | 6 |
| Stadtlohn-Vreden | 5 |
| Johannisbergs / Västerås | 5 |
| Locarno | 4 |
| Illertissen | 4 |
| Abu Dhabi Sports Aviation Club | 2 |
| Soest-Bad Sassendorf | 1 |
| Kägiswil | 1 |

Die mittlere Höhen-MAD beträgt 0,45 m, die höchste akzeptierte Höhen-MAD 4,977 m. Jede Beobachtung
speichert Sprungreferenz, Zeitpunkt, Medianposition, Medianhöhe, Streuung, Sampleanzahl, Dauer,
Zonendistanz, Algorithmusversion und verwendete Grenzwerte. Sie ändert weder die gespeicherte
Bodenhöhe des Sprungs noch die Kataloghöhe automatisch.

## Datenmodell

- `dropzones`: physischer Platz, Referenzkoordinate, Erkennungsradius, Höhenwerte, Status und Revision.
- `dropzone_zones`: eine oder mehrere konkrete Lande-, Alternativ- oder historische Zonen pro Platz.
- `dropzone_operators`: organisatorische Betreiber getrennt vom physischen Ort.
- `dropzone_operator_assignments`: n:m-Verknüpfung zwischen Platz und Betreiber.
- `dropzone_sources`: feldbezogene Herkunft und Vertrauensstufe jedes Internet- oder lokalen Belegs.
- `dropzone_observations`: reproduzierbare Einzelbeobachtungen aus stabilen Boden-GNSS-Sequenzen.
- `dropzone_catalog_metadata`: Katalogversion, Seed-Hash, Importzeitpunkt, Sollzahlen und letzter Audit.

Die Tabelle `jumps` besitzt nullable Felder für eine spätere Dropzone-, Zonen- und Revisionszuordnung.
Sie sind derzeit bei allen 363 Sprüngen leer. Damit ist das Schema vorbereitet, ohne die bestehende
Upload- oder Analysefunktion vorwegzunehmen.

## Reproduzierbarer Betrieb

Nur validieren:

```powershell
python -m scripts.dropzone_catalog --json
```

Katalog importieren und lokale Beobachtungen neu aufbauen:

```powershell
python -m scripts.dropzone_catalog --apply --observe-history --json
```

Aktuellen Datenbankbestand auditieren:

```powershell
python -m scripts.dropzone_catalog --audit --json
```

Der Seed-Import ist wiederholbar. Manuell auf `verified` gesetzte Bodenhöhe, Höhenunsicherheit und
Quelle bleiben erhalten. Bestätigte oder deaktivierte Betreiberzustände werden ebenfalls nicht durch
den Seed zurückgestuft. Der Historienimport baut nur seine eigenen, deterministisch benannten
Beobachtungen neu auf und lässt manuelle Nachweise unberührt.

## Audit-Ergebnis

- SQLite `integrity_check`: `ok`
- Foreign-Key-Verstöße: 0
- Dropzones ohne Zone: 0
- Dropzones ohne Quelle: 0
- Betreiber ohne Dropzone: 0
- Vertrauenswürdige Dropzones ohne Bodenhöhe: 0
- Doppelte physische Koordinaten auf fünf Dezimalstellen: 0
- Ungültig formatierte Quellen-URLs: 0
- Automatische Sprungzuordnung in diesem initialen Katalogschritt: 0; der kontrollierte Rollout erfolgte anschließend mit App-Version 1.2.0.

## Bewusste Grenzen und Umsetzung in 1.2.0

Der Katalog ist eine belastbare Grundlage, aber noch keine amtliche Dropzone-Liste. Betreiber können
umziehen, Plätze schließen, Landezonen wechseln und Websites veralten. Kandidaten brauchen eine
zusätzliche manuelle oder primäre Bestätigung. DEM-Höhen dürfen nicht blind als exakte Landehöhe
behandelt werden.

Diese getrennte Ausbaustufe wurde mit App-Version 1.2.0 umgesetzt. Sie bestimmt eine stabile
Bodenposition, verwendet nur eindeutige `trusted`/`verified`-Treffer, speichert Distanz und Konfidenz
und führt unbekannte oder mehrdeutige Orte einer lokalen Prüfliste zu. Der Upload benötigt keinen
Internetzugriff. Details und realer Rollout: `validation-v1.2.0.md`.
