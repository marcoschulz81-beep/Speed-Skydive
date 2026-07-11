# Speed-Skydive Bewertungslogik Memory

Stand: 2026-07-11

Diese Datei ist die technische Merkhilfe zur aktuellen Auswertungs-, Bewertungs- und Coachinglogik. Quelle der Wahrheit bleibt der Code, aber diese Notizen sollen schnell zeigen, wo welche Entscheidung entsteht.

## Hauptdatenfluss

1. Upload/Import laeuft ueber `app.analysis.pipeline.analyze_flysight_csv`.
2. Das Ergebnis wird in SQLite gespeichert und spaeter ueber `app.services.storage.get_jump_report` als Report geladen.
3. Die Reportseite `app.main.jump_detail` baut daraus nacheinander:
   - historische Referenzen und Vergleiche (`build_jump_comparison`)
   - Springerprofil/Stabilitaetsreferenz (`_build_jumper_summary`)
   - Potentialvorschau (`build_speed_potential_preview`)
   - lokale Review/Bewertung (`build_jump_review`)
   - Feedback-Kontext (`build_feedback_coaching_context`)
   - Scorecard-Zeilen (`_build_scorecard_rows`)
   - Phasen-Tabelle mit Referenzwerten (`_build_phase_rows_with_reference`)
   - Expert-Briefing (`_build_jump_brief_summary`)
   - Simple-Briefing (`_build_jump_brief_simple`)
   - optional KI-Coaching (`_build_ai_coaching` -> `generate_ai_coaching_texts`)
4. Templates rendern diese Daten in `app/templates/jump_detail.html`.

## CSV- und Basisanalyse

Zentrale Datei: `app/analysis/pipeline.py`

- Unterstuetzt klassische FlySight CSVs und FlySight-2 TRACK.CSV.
- Pflichtspalten stehen in `app/config.py`: `time`, `hMSL`, `velN`, `velE`, `velD`, `sAcc`, `hAcc`, `vAcc`, `gpsFix`, `numSV`.
- Fuer FlySight-2 oder reduzierte Logs werden fehlende GPS/Accuracy-Spalten teilweise mit Defaults ergaenzt.
- Zeit wird UTC-normalisiert, Daten werden sortiert, Duplikate entfernt, numerische Spalten gecastet.
- `t0` ist der erkannte Absprungzeitpunkt. Unsichere Erkennung setzt `NO_CLEAR_EXIT`.
- T0-Erkennung ist auf den dominanten Freefall-Peak geankert. Wenn die erste Erkennung erst mitten in einer steigenden Rampe landet, wird sie auf den plausiblen Rampenbeginn zurueckgeschoben. Das greift auch bei mittelspaeten Kandidaten um ca. `35-45 m/s` vertikal, sofern davor eine monotone Rampe ab ca. `10 m/s` und ein klarer Peak-Gain liegen.
- Wenn keine Bodenhoehe gesetzt ist, wird Boden aus niedrigem `hMSL`-Quantil geschaetzt und `NO_GROUND_LEVEL` gesetzt.
- Abgeleitete Samples:
  - `t_rel_s`
  - `vVert_mps`, `vVert_kmh` aus `velD`
  - `vHor_mps`, `vHor_kmh` aus `velN/velE`
  - `vTotal_*`
  - `angle_deg = atan2(abs(velD), vHor)`
  - `accVert_mps2` per Gradient
  - `hAGL_m`, wenn Bodenhoehe verfuegbar
- Fixpunkte werden bei `+10/+15/+20/+24/+28s` linear interpoliert.

## Bewertungsfenster und Speed

- `best_3s_vVert_kmh` ist das beste zusammenhaengende 3s-Trainingsfenster ab `t0`.
- `rule_based_3s_score` ist das regelnahe 3s-Fenster innerhalb des Performance Windows.
- Performance Window:
  - Start ab `velD >= 10 m/s` nach `t0`.
  - Ende ueber maximalen Hoehenverlust `PERFORMANCE_WINDOW_VERTICAL_DROP_M = 2256.0` oder Breakoff-Hoehe.
- Coaching/3s-Qualitaet trennt spaete Ausstiegs-/Decel-Drops von echten Technikproblemen. Ein Speed-Drop nach dem Fenster wird nur als Problem gewertet, wenn er vor `decel_start` und in belastbarem Folgezeitraum liegt.
- `curve_window` begrenzt den technisch sinnvollen Bereich; zu kurze/inkonsistente Tracks koennen `analysis_blocked` setzen.

## Qualitaetsflags

Wichtige Flags:

- `NO_CLEAR_EXIT`: t0 unsicher.
- `NO_GROUND_LEVEL`: Bodenhoehe geschaetzt.
- `INVALID_EXIT_ALTITUDE`: unplausible Exit-Hoehe.
- `LOW_GPS_FIX`, `HIGH_SPEED_ACCURACY_ERROR`, `TIME_GAPS`, `SPEED_SPIKE`.
- `EARLY_JUMP_END`: Sprung/Track endet zu frueh fuer Speed-Auswertung.

Geblockte Analysen liefern keine normale Technikbewertung, sondern Hinweise zur Datei/T0/Trackqualitaet.

## Zentrales 5-Phasen-Modell

Zentrale Quelle: `app/config.py`, `TECHNICAL_PHASE_SPECS`.

1. `Exit / Stabilisierung`: `+0.0s` bis `+3.0s`, Zielwinkel `0-40 Grad`.
2. `Dive-Aufbau`: `+3.0s` bis `+8.0s`, Zielwinkel `60-70 Grad`.
3. `Hauptbeschleunigung`: `+8.0s` bis `+15.0s`, Zielwinkel `75-83 Grad`.
4. `Hot-Zone Aufbau`: `+15.0s` bis `+22.0s`, Zielwinkel `82-86 Grad`.
5. `Max-Speed Fenster`: `+20.0s` bis `+28.0s`, Zielwinkel `83-87 Grad`.

Die Phasen werden dynamisch bis zum effektiven Auswertungsende gekappt. Wenn weniger als ca. `0.75s` nutzbar sind oder die Coverage unter `0.75` liegt, wird die Phase als `nicht belastbar` markiert.

## Phasenstatus

Statuslogik in `pipeline.py` und gespiegelt in `main.py`:

- `zu flach`: Avg Winkel < Ziel-Minimum minus `1.5 Grad`.
- `zu steil`: Avg Winkel > Ziel-Maximum plus `1.0 Grad`.
- `kurz zu steil`: Durchschnitt passt, aber Max Winkel > Ziel-Maximum plus `2.0 Grad`.
- `eher flach`: Durchschnitt knapp, aber Min Winkel < Ziel-Minimum minus `3.0 Grad` und Avg nahe am unteren Rand.
- `im Zielbereich`: keiner der obigen Faelle.
- `nicht belastbar`: zu wenig verwertbare Daten.

Kommentare sind phasenspezifisch. Beispiel: `Max-Speed Fenster` mit `eher flach` sagt nicht mehr pauschal "liegt im Zielwinkel", sondern weist darauf hin, dass einzelne Abschnitte darunter fallen.

## Scorecard

Zentrale Funktionen: `app/main.py`

- `_build_legacy_scorecard_rows` baut die alte 4er-Logik weiter intern auf.
- `_build_scorecard_rows` gibt bevorzugt die neue 5-Phasen-Scorecard aus.
- Die Legacy-Scores bleiben als Teilgewicht/Fallback in der 5-Phasen-Bewertung erhalten, werden aber nicht mehr als eigene "Referenzscore"-Zeilen unter jeder Phase ausgegeben.

Statusscore:

- `im Zielbereich`: 88
- `eher flach` oder `kurz zu steil`: 70
- `zu flach` oder `zu steil`: 50
- `nicht belastbar`: 58
- unbekannt: kein Winkel-Teilscore

Gewichte:

- `Exit / Stabilisierung`: Winkel 35%, Legacy-Exit 65%.
- `Dive-Aufbau`: Winkel 45%, Legacy-Aufbau 25%, Phasen-vVert-Zuwachs 30%.
- `Hauptbeschleunigung`: Winkel 45%, Legacy-Aufbau 35%, Phasen-vVert-Zuwachs 20%.
- `Hot-Zone Aufbau`: Winkel 35%, Legacy-Hot 30%, vHor-Min 20%, Phasen-vVert-Zuwachs 15%.
- `Max-Speed Fenster`: Winkel 25%, Legacy-Hot 20%, Legacy-Stabilitaet 20%, vHor-Min 15%, Winkel-Korrekturen 10%, >400-km/h-Dauer 10%, Kipp-Risiko 10%.

Hilfsscores:

- vVert-Zuwachs: je Phase andere Mindestwerte (`Dive 45`, `Haupt 55`, `Hot 20`, `Max 0` km/h).
- vHor-Min: >=40 sehr gut, >=30 gut, >=25 mittel, darunter schwach.
- Winkel-Korrekturen: <=1 sehr gut, <=3 gut, <=5 mittel, mehr schwach.
- >400-km/h-Dauer: >=3s sehr gut, >=1.5s gut, >0 mittel, sonst schwach.

Scorecard-Reihenfolge je Phase:

1. Zeitfenster und Zielwinkel.
2. Istwerte: Avg Winkel, Avg vVert, Avg vHor.
3. Bewertung mit Begruendung.
4. Technischer Kommentar.
5. Phasenmetriken wie vVert-Zuwachs, vHor-Min, >400-Dauer, Winkel-Korrekturen.
6. Referenzvergleiche des besten gespeicherten Sprungs, falls vorhanden.

Referenzvergleich:

- vVert ist `higher_is_better=True`.
- vHor in Phasen wird inzwischen neutral als Delta kommuniziert, nicht als "bestes Ergebnis bisher".
- Winkel wird neutral als Delta kommuniziert.

## Review und Coachinglogik

Zentrale Datei: `app/analysis/review.py`

`build_jump_review` erzeugt:

- `happened`
- `good`
- `not_good`
- `improve`
- `coaching_goals`
- `technical_assessment`
- `primary_diagnosis`

Die Review nutzt Fixpunkte, Scorecard, Phasenstatus, Chartdaten, laterale Analyse, Feedbackprofil, Springer-Stabilitaetsprofil und Referenzvergleiche.

Persoenliche Korridore:

- Kommen aus `jumper_stability_reference`.
- Wichtige Thresholds: Winkel +20s, vHor-Floor +20/+25, Gain +10/+20, vVert +10s, Phasen-Gains +10/+15 und +15/+20.
- Bei niedrigem +10/+20-Zuwachs wird der Zielwert als naechster Schritt begrenzt: `max(90, min(profile_target, aktueller_gain + 25))`. Dadurch entstehen realistischere Ziele wie `+99 km/h` statt pauschal `+180 km/h`.

Technikmodell:

- `_technical_coaching_assessment` prueft 5 Phasen, 3s-Fensterqualitaet und Beschleunigungsruhe/Jerk.
- `steep_without_gain`: Aufbauphase wird steiler, aber vVert/Acceleration profitieren kaum.
- `oversteep_vhor_cost`: spaete Phase zu steil bei knapper vHor-Reserve.
- 3s-Fensterlabel:
  - `stabil`
  - `unruhig`
  - `kritisch`
- Faktoren fuer 3s-Fenster: vVert-Std, Winkel-Std, vHor-Min, mittlere Acceleration, verwertbarer Speed-Drop danach.

Primary Diagnosis:

- Dient dazu, klare Hauptmuster gegen generische Tipps zu priorisieren.
- Wichtige Muster:
  - `late_hard_steepening_vhor_collapse`
  - `too_fast_steep_not_hold`
  - `angle_rollback_with_instability`
- Wenn Primary Diagnosis verfuegbar ist, dominiert sie Summary, Hauptproblem und naechsten Fokus.

## Expert-Briefing

Zentrale Funktion: `_build_jump_brief_summary`.

- Nutzt zuerst Primary Diagnosis, falls vorhanden.
- Sonst werden schwache Scorecard-Phasen und Review-Issues priorisiert.
- Hauptthemen werden aus echten Issues abgeleitet, nicht mehr aus den Actions ueberschrieben.
- Reine Kontextzeilen wie `+3.0s bis +8.0s, Zielwinkel ...` werden aus Hauptproblemen gefiltert.
- Actions werden mit Issues abgeglichen, damit Empfehlungen nicht voellig andere Phasen priorisieren.
- Finale Actions werden vor der Ausgabe harmonisiert:
  - Redundante Korrektur-/Lenkimpuls-/Beschleunigungsruhe-Tipps in der schnellen Phase werden zu einem Tipp zusammengefuehrt.
  - Aufbau-Druck, Winkelkontrolle, horizontale Reserve, Exit und >400-Halten bleiben getrennte Handlungsfamilien.
  - Danach werden die sichtbaren Tipps nach Sprungverlauf sortiert; bei Primary Diagnosis bleibt der Hauptfokus auf Position 1, die weiteren Tipps folgen chronologisch.
  - Diese Harmonisierung aendert keine Scores, keine Review-Rohsignale und keine Primary Diagnosis.
- In der Expert-Ansicht mit KI-Coaching wird `next_jump_focus` als `KI-Coaching-Fokus` genutzt:
  - KI darf die regelbasierten Tipps aus `jump_brief.actions` als Zusammenhang in Fliesstext erklaeren.
  - KI darf Dopplungen sprachlich glaetten, aber keine neuen Ziele oder Messwerte erfinden.
  - Die harmonisierte regelbasierte Liste `Konkrete Trainings-Tipps` bleibt darunter sichtbar und pruefbar.
  - Wenn der KI-Fokus unbrauchbar ist (z. B. reines Ergebnisziel oder erfundene Zeitpraezision), baut der Code einen kurzen Fokus aus den vorhandenen Tipps.
- Bei Feedback kann ein Coaching-Hinweis in Issues/Actions einfliessen, ohne Messwerte zu veraendern.

## Simple-Briefing

Zentrale Funktion: `_build_jump_brief_simple`.

- Vermeidet Fachjargon und technische Payload-Begriffe.
- Bei Primary Diagnosis werden Titel, Zusammenfassung und naechster Fokus direkt uebernommen.
- Ohne Primary Diagnosis:
  - Summary nennt max. zwei groesste Hebel.
  - Main Issues bevorzugen verstaendliche Briefing-Issues, danach Scorecard-Phasen.
  - Kaputte technische Kontextreste wie `bis ,`, `Zielwinkel 60...` oder `Geschwindigkeit nach unten-Zuwachs` werden gefiltert.
  - Simple Issues uebersetzen typische Briefingtexte:
    - +10/+15 Zuwachs zu niedrig -> "zu wenig zusaetzlicher Speed"
    - Hot-Zone ineffizient/Hoehenverlust -> "viel Hoehe, wenig zusaetzlicher Speed"
    - Hot-Zone vHor verloren -> "waagerechte Geschwindigkeit faellt zu stark ab"
  - Staerken werden nicht aus Scorecard-Zeilen mit Winkelproblem gebildet.
  - `Hot-Zone Aufbau` bekommt als Action die schnelle Phase, nicht die generische Aufbauphase.

## Vergleich, Profil und Potential

`app/analysis/comparison.py`:

- Vergleicht immer gegen den schnelleren Sprung als Referenz.
- Deltas sind `comparison - reference`.
- Summary vergleicht Training-3s, Rule Score und Negativ-Risiko.
- Fixpunktvergleich nutzt +10/+15/+20/+24/+28s.
- Phasenvergleiche folgen den 5 technischen Phasen.

`_build_jumper_summary` in `main.py`:

- Baut Leistungsprofil, Stabilitaetsreferenz, Feedback-Training-Profil und Tip-Effect-Profil.
- Stabilitaetsreferenz liefert persoenliche Zielkorridore fuer Review und Coaching.
- Baut zusaetzlich `timing_reference` als persoenliche Timing-Referenz.

Persoenliche Timing-Referenz:

- Standard-Phasen bleiben fix und vergleichbar; sie werden nicht pro Springer verschoben.
- `timing_reference` ist additiver Kontext, damit schnelle und langsamere Springer trotzdem mit ihrem eigenen stabilen Timing verglichen werden koennen.
- Pro Sprung werden unter anderem gespeichert:
  - Start/Ende bestes 3s-Fenster.
  - erste Aufwaerts-Crossings fuer 60/75/82/85 Grad.
  - vVert-Peak-Zeit.
  - Decel/Recovery-Start.
- Basis sind zuerst stabile/kontrollierte Spruenge (`stability_score`, `hot_score`, `build_score`, stabile Geometrie). Wenn davon zu wenig vorhanden ist, werden die besten verfuegbaren kontrollierten Spruenge genommen.
- Aus der Basis entstehen IQR-Korridore (`low`, `median`, `high`) und kurze Summary-Zeilen.
- Reife-Gate:
  - `off`: weniger als 5 verwertbare Spruenge; keine Timing-Referenz anzeigen, nicht an KI geben, nicht fuer Review-Hinweise nutzen.
  - `soft`: mindestens 5 verwertbare Spruenge, aber noch nicht belastbar. Gruende sind z. B. beste Referenz unter 430 km/h, Top-3-Schnitt unter 415 km/h, weniger als 3 kontrollierte Basis-Spruenge oder Timing-IQR fuer 82 Grad / Start bestes 3s-Fenster groesser als 2.0s.
  - `active`: mindestens 5 verwertbare Spruenge, beste Referenz >= 430 km/h, Top-3-Schnitt >= 415 km/h, mindestens 3 kontrollierte Basis-Spruenge und stabile Timing-Anker.
- Expert-Ansicht zeigt diese Referenz nur bei `active` in einem eigenen Block `Persoenliches Timing` vor der Phasentabelle.
- KI-Payload bekommt die kompakte Referenz als `personal_timing` nur bei `active`; sonst nur `available: false` mit Reifegrund.
- Rule-Review nutzt die Referenz nur bei deutlichen Abweichungen:
  - 85 Grad deutlich frueher als persoenlich stabil kann einen fruehen horizontalen Reserveverlust erklaeren.
  - 82 Grad plus bestes 3s-Fenster deutlich spaeter als persoenlich stabil erzeugt einen schlanken Timing-Hinweis.
- Die Timing-Referenz aendert keine Scorecard-Gewichte und keine Phasenfenster.

`app/analysis/potential.py`:

- Nutzt historische saubere Spruenge desselben Springers.
- Mindestmenge: 5 verwertbare historische Reports.
- Features: vVert +10s, Gain +10/+20, Winkelabweichung spaete Phase, Risiko.
- Erwartetes Potential kommt aus linearer Regression gegen Historie und zeigt grobe Hebel, keine harte Prognose.

## KI-Coaching

Zentrale Datei: `app/services/ai_coach.py`.

- KI ist optional. Lokale Analyse bleibt immer Fallback.
- Config in `app/config.py`:
  - `AI_COACHING_ENABLED`
  - `AI_COACHING_MODEL`
  - `AI_COACHING_TIMEOUT_S`
  - `AI_COACHING_MAX_REQUESTS_PER_DAY`
  - `AI_COACHING_INCLUDE_IDENTIFIERS`
  - `OPENAI_API_KEY`
- Payload wird in `_build_ai_coaching_payload` gebaut und enthaelt nur kompakte Fakten:
  - Kennzahlen
  - Scorecard
  - technische Phasen
  - Review
  - Primary Diagnosis
  - technische Bewertung
  - Jump Brief
  - Tip Follow-up
  - Feedback-Kontext
  - Qualitaet
- Standardmaessig keine Namen, Dateinamen oder Zeitstempel in der KI-Payload.
- Response-Schema verlangt:
  - `summary`
  - `main_issue`
  - `coaching_text`
  - `next_jump_focus`
  - `confidence_note`
- `coaching_text` darf bis 1800 Zeichen lang sein und wird sauber an Satz-/Wortgrenzen gekappt.
- Cache-Key basiert auf kompaktem Payload, View Mode und Modell.
- Tageslimit zaehlt neue API-Requests im Prozess.

KI-Fehler:

- Wenn deaktiviert: `KI-Coaching ist deaktiviert.`
- Wenn Key fehlt: `OPENAI_API_KEY ist nicht gesetzt.`
- Wenn OpenAI-Paket fehlt: entsprechender Installationshinweis.
- Wenn Request fehlschlaegt: Grund enthaelt jetzt Kategorie, aber keine rohen Secrets:
  - `Timeout`
  - `Rate Limit`
  - `Auth/API-Key`
  - `Netzwerk`
  - `Clientfehler`
- Wenn JSON/Schema ungueltig ist: Fallback auf normale Analyse.

Wichtig fuer Debugging: Wenn Simple KI liefert, Expert aber nicht, ist die lokale Analyse nicht kaputt. Dann ist sehr wahrscheinlich genau der jeweilige Expert-KI-Request fehlgeschlagen oder gecached/fallbacked. Die UI bekommt jetzt eine Kategorie statt nur generischem Ausfall.

## Tests

Relevante Testdateien:

- `tests/test_main_summary.py`: Scorecard, Phasen, Briefing, Simple/Expert-Kommunikation, KI-Payload, Follow-up.
- `tests/test_review.py`: lokale Review, Prioritaeten, Primary Diagnosis, technische Bewertung.
- `tests/test_ai_coach.py`: KI-Schema, Laengenlimits, Fehler-Fallbacks, Simple-Sanitizing.
- `tests/test_potential.py`: Potentialmodell.

Nuetzliche Befehle:

```powershell
python -m pytest -q
python -m pytest tests/test_main_summary.py tests/test_review.py tests/test_ai_coach.py -q
python -m compileall app tests
```

## Aktuelle Kommunikationsregeln

- Scorecard soll erst Ziel/Ist/Bewertung erklaeren und dann Zusatzmetriken zeigen.
- Widersprueche vermeiden: Status und Kommentar muessen aus derselben Phasenlogik kommen.
- Simple View soll nie rohe technische Kontextzeilen oder kaputte Regex-Reste anzeigen.
- Hauptproblem darf nicht durch spaeter sortierte Actions ersetzt werden.
- Max-Speed darf nicht als groesste Baustelle erscheinen, wenn die eigentlichen Issues Hauptbeschleunigung/Hot-Zone sind und Max-Speed nur als Folge-Action auftaucht.
- Ziele sollen als naechster sinnvoller Schritt formuliert werden, nicht als maximaler Benchmark-Sprung.

## Horizontale Reserve / vHor-Timing

Neue Bewertungsregel: Ein Sprung kann ein stabiles spaeteres 3s-Fenster haben und trotzdem technisch kritisch sein, wenn die horizontale Reserve vorher zusammenbricht.

- Muster: vHor-Minimum liegt klar vor dem besten 3s-Fenster, danach steigt vHor wieder deutlich an.
- Zusatzevidenz: Winkel-Peak nahe am vHor-Minimum und/oder Forward-Track erreicht dort sein Maximum und laeuft danach zurueck.
- Interpretation: Der spaetere vHor-Anstieg ist dann kein Stabilitaetsplus, sondern Recovery/Nachkorrektur nach verlorener Linie.
- Review-Primary-Diagnosis: `early_horizontal_reserve_collapse`.
- Scorecard: Hot-Zone Aufbau und Max-Speed Fenster werden bei klarer Evidenz gedeckelt, auch wenn der Avg Winkel im Zielbereich liegt.
- Technische Bewertung: Ergaenzt Issue, dass die horizontale Reserve vor dem 3s-Fenster zu frueh verbraucht wurde.
- Pipeline: `_negative_risk` prueft zusaetzlich das Hot-Zone-vHor-Minimum mit Rebound, damit Re-Analysen dieses Muster nicht als niedriges Kipp-Risiko speichern.
- Coaching-Prioritaet: Zuerst horizontale Reserve laenger tragen und Winkel-Peak nicht ueberziehen; danach erst allgemeine Build-Speed- oder Referenz-Gap-Tipps.
