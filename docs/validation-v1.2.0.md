# Validierungsprotokoll App-Version 1.2.0

## Ergebnis

Version 1.2.0 aktiviert die lokale Dropzone-Erkennung für Upload, Reanalyse und Report. Die
mathematische Score-Engine wurde nicht geändert und bleibt deshalb korrekt als Analyseversion 1.1.0
gekennzeichnet. Dropzone-ID, Zone, Katalogrevision und Zuordnungsart fließen zusätzlich in die
Analysesignatur ein.

Der reale Rollout auf 363 gespeicherte Sprünge endete ohne Fehler:

| Ergebnis | Anzahl |
|---|---:|
| Sicher erkannt und zugeordnet | 349 |
| Keine ausreichend stabile Bodensequenz | 14 |
| Automatische Zuordnung zu Kandidaten | 0 |
| Mehrdeutige automatische Zuordnung | 0 |
| Außerhalb des sicheren Radius | 0 |
| Match-Versuche vollständig protokolliert | 363 |

Die 349 Konfidenzen liegen zwischen 0,757 und 0,996, im Mittel bei 0,934. Die größte akzeptierte
Entfernung zur gespeicherten Zone beträgt 426,5 m.

## Datenfluss und Sicherheitsregeln

1. Die CSV wird zunächst mit der bestehenden Analyse-Engine normalisiert und auf einen plausiblen Sprung geprüft.
2. Aus den normalisierten Samples wird frühestens 45 Sekunden nach `t0` eine stabile Bodensequenz gesucht.
3. Diese Beobachtung wird ausschließlich gegen den lokalen, versionierten Katalog geprüft. Der Upload führt keinen Internetzugriff aus.
4. Nur eindeutige `trusted`- oder `verified`-Zonen werden automatisch übernommen.
5. Bei sicherem Treffer wird die Analyse mit der Kataloghöhe erneut deterministisch berechnet, sofern keine manuelle Bodenhöhe vorliegt.
6. Manuelle Bodenhöhen haben immer Vorrang. Eine manuelle Dropzone-Auswahl ist im Expertenreport möglich und wird als solche gespeichert.
7. Kandidaten, Mehrdeutigkeiten, geringe Konfidenz und unbekannte Orte verändern die Analyse nicht; sie bleiben als Prüffall erhalten.

Eine automatisch nutzbare Bodensequenz benötigt:

- mindestens fünf Sekunden Dauer,
- 3D-GPS-Fix und mindestens sechs Satelliten,
- Gesamtgeschwindigkeit höchstens 10 km/h,
- vertikale Geschwindigkeit höchstens 1 m/s im Betrag,
- `sAcc` höchstens 3 m/s und `vAcc` höchstens 10 m, sofern vorhanden,
- maximal 0,35 Sekunden Lücke zwischen gültigen Samples,
- Höhen-MAD höchstens 5 m,
- 95-Prozent-Positionsradius höchstens 100 m.

Für den automatischen Platztreffer gelten zusätzlich:

- effektive Distanz höchstens 750 m, auch wenn der allgemeine Katalogradius größer ist,
- Konfidenz mindestens 0,75,
- keine zweite physische Dropzone mit weniger als 500 m Abstandsvorsprung oder Distanzverhältnis unter 2,0,
- vorhandene Katalog-Bodenhöhe und Dropzone-Status `trusted` oder `verified`.

## Persistenz und Bedienung

Jeder Sprung speichert bei sicherer Zuordnung Dropzone-ID, konkrete Zone, Katalogrevision,
Zuordnungsquelle, Konfidenz und Entfernung. `dropzone_match_attempts` enthält darüber hinaus für jeden
Upload oder Shadow-Lauf Algorithmusversion, Katalogversion, beobachtete Position und Höhe, Streuung,
Sampleanzahl, Status und Ablehnungsgrund. Dadurch bleiben auch unbekannte oder unsichere Orte ohne
Internetabhängigkeit prüfbar.

Der Report zeigt die erkannte Dropzone in einfacher und erweiterter Ansicht. Die Expertenansicht
zeigt zusätzlich Zone, Entfernung, Konfidenz und Zuordnungsart. Dort kann der Nutzer die automatische
Erkennung erneut ausführen oder einen Platz einschließlich eines Kandidaten ausdrücklich manuell
bestätigen. Eine vorhandene manuelle Bodenhöhe wird dabei nicht überschrieben.

## Rollout der Altdaten

Vor der Migration wurde folgende Sicherung erstellt:

`tmp/speed_skydive.before-v1.2.0-dropzone-rollout-20260718.db`

Der vollständige Shadow-Dry-Run analysierte 363 Original-CSV-Dateien und meldete null Fehler. Danach
wurden 349 sichere Treffer reproduzierbar aus den Originaldateien neu gespeichert:

- 338 manuelle Bodenhöhen blieben numerisch unverändert.
- Elf alte automatische Schätzwerte wurden durch Kataloghöhen ersetzt.
- Zwei geschätzte und zwölf manuelle Fälle ohne stabile Bodensequenz blieben vollständig unverändert.
- Sprunganzahl, Sampleanzahl, Sprung-IDs, `t0`, Kontext, Referenzstatus und Quelldatei-Hashes blieben gegenüber der Sicherung identisch.
- Kein numerischer Regel-Score änderte sich.

Die gesicherte Bodenhöhe änderte jedoch erwartungsgemäß die Verwendbarkeit einzelner Scores:

| Regel-Score-Status | Vorher | Nachher |
|---|---:|---:|
| `valid` | 320 | 330 |
| `estimated` | 11 | 2 |
| `invalid` | 32 | 31 |

Neun Sprünge wechselten von `estimated` zu `valid`. Ein Sprung wechselte von `invalid` zu `valid`,
weil die alte 2-Prozent-Schätzung in Fano eine offensichtlich falsche Bodenhöhe von −251,61 m und
damit eine falsche Exit-Höhenbewertung erzeugt hatte. Die Kataloghöhe beträgt dort 8,5 m.

## Vollständig geprüfte Reports

Vier reale Reportpfade wurden einschließlich Coaching-, Vergleichs-, Profil- und Templateaufbau über
FastAPI gerendert. Alle lieferten HTTP 200, den erwarteten Dropzone-Status und keinen Traceback:

- Fano, Expertenansicht: 89,4 m Distanz, Konfidenz 0,951, Bodenhöhe −251,61 m → 8,5 m, Status `invalid` → `valid`, Regel-Score unverändert 434,86 km/h.
- Hohenems, Expertenansicht: 41,4 m, Konfidenz 0,963, Bodenhöhe 404,14 m → 409,4 m, Status `estimated` → `valid`, Regel-Score unverändert 428,28 km/h.
- Locarno, einfache Ansicht: 48,2 m, Konfidenz 0,861, Bodenhöhe 203,01 m → 197,7 m, Status `estimated` → `valid`, Regel-Score unverändert 423,07 km/h.
- Nicht zuordenbarer Locarno-Datensatz, Expertenansicht: keine stabile Sequenz; Bodenhöhe 198,45 m und Status `estimated` blieben erhalten.

## Abschlussaudit

- SQLite `integrity_check`: `ok`
- Foreign-Key-Verletzungen: 0
- Sprünge/Metriken: 363/363
- Samples: 769.438
- Dropzone-Zuordnungen: 349
- Match-Protokollabdeckung: 363/363
- Automatische Zuordnungen zu nicht freigegebenen Dropzones: 0
- Zugeordnete Sprünge ohne passenden akzeptierten Match-Versuch: 0
- Manuelle Bodenhöhen gegenüber der Sicherung verändert: 0
- Numerische Regel-Scores gegenüber der Sicherung verändert: 0
- `t0`, Kontext, Referenzstatus oder Quelldatei-Hash verändert: 0

Reproduzierbare Betriebsbefehle:

```powershell
python -m scripts.dropzone_catalog --apply --json
python -m scripts.reprocess_analysis --dry-run --dropzone-rollout --json
python -m scripts.reprocess_analysis --dropzone-rollout --json
python -m scripts.audit_database
python -m scripts.dropzone_catalog --audit --json
python -m scripts.dropzone_catalog --audit-matches --json
```
