from __future__ import annotations

from app.main import (
    _build_fs2_quality_issue_lines,
    _build_jump_brief_summary,
    _build_jump_brief_simple,
    _build_jumper_stability_reference,
    _build_jumper_trend_rows,
    _jumper_overview_simple_status,
    _normalize_view_mode,
    _render_simple_glossary,
    _tip_focus_from_previous,
    _tip_follow_status,
)


def test_jump_brief_summary_uses_compact_sections_and_strips_priority_prefix():
    report = {
        "jump": {"file_name": "test.csv"},
        "metrics": {
            "best_3s_vVert_kmh": 398.4,
            "best_3s_start_s": 22.6,
            "best_3s_end_s": 25.6,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 36.0,
        },
    }
    review = {
        "happened": [],
        "good": ["Startphase war ruhig."],
        "not_good": ["In der schnellen Phase gab es spaete Gegenkorrekturen."],
        "improve": ["Prioritaet 1: In der Hot-Zone kleinere, fruehere Korrekturen setzen."],
    }
    scorecard_rows = [
        {"name": "Exit", "score": 82, "reason": "vVert@10s und Carry sind stabil."},
        {"name": "Aufbau 10-20s", "score": 86, "reason": "Der Aufbau ist gleichmaessig."},
        {"name": "Hot-Zone", "score": 42, "reason": "Die >400 km/h Zone wird zu kurz gehalten."},
        {"name": "Stabilitaet / Kipp-Risiko", "score": 45, "reason": "Mehrere Korrekturen im Schlussteil."},
    ]

    summary = _build_jump_brief_summary(
        report=report,
        review=review,
        scorecard_rows=scorecard_rows,
        best_reference=None,
        top_reference_jumps=[],
    )

    assert "Hot-Zone" in summary["summary"]
    assert summary["main_issues"][0].startswith("Hot-Zone:")
    assert summary["actions"][0] == "In der Hot-Zone kleinere, fruehere Korrekturen setzen."
    assert summary["key_facts"][0].startswith("Bestes 3s-Fenster:")


def test_jump_brief_summary_actions_follow_timeline_and_focus_main_issues():
    report = {
        "jump": {"file_name": "timeline.csv"},
        "metrics": {
            "best_3s_vVert_kmh": 401.0,
            "best_3s_start_s": 21.0,
            "best_3s_end_s": 24.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 34.0,
        },
    }
    review = {
        "happened": [],
        "good": [],
        "not_good": [
            "In der Hot-Zone bricht vHor zu stark ein.",
            "Im Aufbau bis +15s fehlt vertikaler Speed.",
        ],
        "improve": [
            "Prioritaet 1: In der Hot-Zone kleine, fruehe Korrekturen setzen und vHor stabil halten.",
            "Prioritaet 2: Bis +15s frueher Druck aufbauen, damit der Zuwachs im Aufbau wieder passt.",
            "Prioritaet 3: Ablauf stabil wiederholen und nur kleine Korrekturen setzen.",
        ],
    }
    scorecard_rows = [
        {"name": "Exit", "score": 80, "reason": "Exit ist stabil."},
        {"name": "Aufbau 10-20s", "score": 58, "reason": "Im Aufbau fehlt im fruehen Segment Druck."},
        {"name": "Hot-Zone", "score": 49, "reason": "In der Hot-Zone geht zu viel vHor verloren."},
        {"name": "Stabilitaet / Kipp-Risiko", "score": 73, "reason": "Stabilitaet ist okay."},
    ]

    summary = _build_jump_brief_summary(
        report=report,
        review=review,
        scorecard_rows=scorecard_rows,
        best_reference=None,
        top_reference_jumps=[],
    )

    assert any("Aufbau" in item for item in summary["main_issues"])
    assert any("Hot" in item for item in summary["main_issues"])
    assert any("Bis +15s frueher Druck aufbauen" in item for item in summary["actions"])
    assert any("Hot-Zone" in item for item in summary["actions"])
    assert not any("Ablauf stabil wiederholen" in item for item in summary["actions"])

    main_build_idx = next(i for i, item in enumerate(summary["main_issues"]) if "Aufbau" in item)
    main_hot_idx = next(i for i, item in enumerate(summary["main_issues"]) if "Hot" in item)
    assert main_build_idx < main_hot_idx

    build_idx = next(i for i, item in enumerate(summary["actions"]) if "Bis +15s frueher Druck aufbauen" in item)
    hot_idx = next(i for i, item in enumerate(summary["actions"]) if "Hot-Zone" in item)
    assert build_idx < hot_idx


def test_jump_brief_summary_strengths_remove_semantic_duplicates():
    report = {
        "jump": {"file_name": "dedupe.csv"},
        "metrics": {
            "best_3s_vVert_kmh": 401.5,
            "best_3s_start_s": 22.0,
            "best_3s_end_s": 25.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 34.0,
        },
    }
    review = {
        "happened": [],
        "good": [
            "Der Start in den Sprung wirkt sauber und kontrolliert.",
            "Exit-Dynamik wird gut mitgenommen (stabiler Uebergang von 0-2s auf 2-6s).",
        ],
        "not_good": [],
        "improve": [],
    }
    scorecard_rows = [
        {
            "name": "Exit",
            "score": 84,
            "reason": "Der Start ist sauber und kontrolliert. Du nimmst den Druck nach dem Exit gut mit.",
        },
        {"name": "Aufbau 10-20s", "score": 81, "reason": "Der Aufbau ist stabil."},
        {"name": "Hot-Zone", "score": 58, "reason": "In der Hot-Zone geht zu viel vHor verloren."},
        {"name": "Stabilitaet / Kipp-Risiko", "score": 52, "reason": "Mehrere Korrekturen im Schlussteil."},
    ]

    summary = _build_jump_brief_summary(
        report=report,
        review=review,
        scorecard_rows=scorecard_rows,
        best_reference=None,
        top_reference_jumps=[],
    )

    start_lines = [
        item
        for item in summary["strengths"]
        if "start" in item.lower() and "sauber" in item.lower() and "kontrolliert" in item.lower()
    ]
    assert len(start_lines) == 1
    assert any("Exit-Dynamik" in item for item in summary["strengths"])


def test_fs2_quality_issue_lines_only_for_non_stable_labels():
    stable_notes = {
        "fs2_track_summary": {
            "available": True,
            "quality_label": "stabil",
            "window_start_s": 20.0,
            "window_end_s": 25.0,
            "sAcc_p95": 0.6,
            "hAcc_p95": 1.6,
            "numSV_p10": 27.0,
        }
    }
    crit_notes = {
        "fs2_track_summary": {
            "available": True,
            "quality_label": "kritisch",
            "window_start_s": 20.0,
            "window_end_s": 25.0,
            "sAcc_p95": 2.4,
            "hAcc_p95": 15.1,
            "numSV_p10": 9.0,
        }
    }

    assert _build_fs2_quality_issue_lines(stable_notes) == []
    crit_lines = _build_fs2_quality_issue_lines(crit_notes)
    assert len(crit_lines) == 1
    assert "Messqualitaet (FS2)" in crit_lines[0]
    assert "kritisch" in crit_lines[0]


def test_jumper_trend_rows_detect_better_and_worse_developments():
    records = [
        {
            "best_3s_kmh": 440.0,
            "exit_score": 82,
            "build_score": 79,
            "hot_score": 50,
            "stability_score": 61,
            "angle_turns_20_25": 6.0,
        },
        {
            "best_3s_kmh": 438.0,
            "exit_score": 81,
            "build_score": 78,
            "hot_score": 52,
            "stability_score": 60,
            "angle_turns_20_25": 7.0,
        },
        {
            "best_3s_kmh": 430.0,
            "exit_score": 75,
            "build_score": 79,
            "hot_score": 66,
            "stability_score": 63,
            "angle_turns_20_25": 3.5,
        },
        {
            "best_3s_kmh": 428.0,
            "exit_score": 74,
            "build_score": 80,
            "hot_score": 68,
            "stability_score": 64,
            "angle_turns_20_25": 3.0,
        },
    ]

    rows = _build_jumper_trend_rows(records)
    by_name = {row["name"]: row for row in rows}

    assert by_name["Top-Speed"]["status"] == "schlechter"
    assert by_name["Hot-Zone"]["status"] == "besser"
    assert by_name["Korrekturen 20-25s"]["status"] == "besser"
    assert by_name["Top-Speed"]["earlier_better_text"].startswith("Top-Speed war frueher besser")


def test_jumper_stability_reference_builds_stable_and_unstable_lines():
    records = [
        {
            "analysis_blocked": False,
            "stability_score": 82,
            "hot_score": 78,
            "build_score": 80,
            "vvert_10s": 248.0,
            "vvert_15s": 341.0,
            "vvert_20s": 419.0,
            "angle_10s": 65.0,
            "angle_15s": 75.5,
            "angle_20s": 83.2,
            "gain_10_20": 171.0,
            "gain_10_15": 93.0,
            "gain_15_20": 78.0,
            "vhor_min_20_25": 31.5,
            "angle_turns_20_25": 2.0,
        },
        {
            "analysis_blocked": False,
            "stability_score": 76,
            "hot_score": 72,
            "build_score": 74,
            "vvert_10s": 252.0,
            "vvert_15s": 346.0,
            "vvert_20s": 423.0,
            "angle_10s": 66.2,
            "angle_15s": 76.8,
            "angle_20s": 84.0,
            "gain_10_20": 171.0,
            "gain_10_15": 94.0,
            "gain_15_20": 77.0,
            "vhor_min_20_25": 30.4,
            "angle_turns_20_25": 2.0,
        },
        {
            "analysis_blocked": False,
            "stability_score": 34,
            "hot_score": 42,
            "build_score": 58,
            "vvert_10s": 234.0,
            "vvert_15s": 320.0,
            "vvert_20s": 401.0,
            "angle_10s": 69.0,
            "angle_15s": 79.5,
            "angle_20s": 86.8,
            "gain_10_20": 167.0,
            "gain_10_15": 86.0,
            "gain_15_20": 81.0,
            "vhor_min_20_25": 21.0,
            "angle_turns_20_25": 7.0,
        },
        {
            "analysis_blocked": False,
            "stability_score": 46,
            "hot_score": 53,
            "build_score": 60,
            "vvert_10s": 236.0,
            "vvert_15s": 323.0,
            "vvert_20s": 404.0,
            "angle_10s": 68.5,
            "angle_15s": 79.1,
            "angle_20s": 86.4,
            "gain_10_20": 168.0,
            "gain_10_15": 87.0,
            "gain_15_20": 81.0,
            "vhor_min_20_25": 23.0,
            "angle_turns_20_25": 6.0,
        },
    ]

    ref = _build_jumper_stability_reference(records)

    assert ref["available"] is True
    assert ref["stable_count"] == 2
    assert ref["unstable_count"] == 2
    assert any(line.startswith("+10s:") for line in ref["stable_lines"])
    assert any("Winkel ueber etwa" in line for line in ref["unstable_lines"])
    assert ref["thresholds"]["angle_20_target_low"] is not None
    assert ref["bands"]["stable"]["angle_20s"] is not None
    assert ref["thresholds"]["phase_10_15_gain_low"] is not None
    assert ref["thresholds"]["phase_15_20_gain_low"] is not None
    assert ref["capability_profile"]["mode"] in {"basis", "safe", "build", "push"}


def test_tip_focus_from_previous_uses_weak_scores_and_tip_keywords():
    previous_rows = [
        {"name": "Exit", "score": 82},
        {"name": "Aufbau 10-20s", "score": 64},
        {"name": "Hot-Zone", "score": 58},
        {"name": "Stabilitaet / Kipp-Risiko", "score": 75},
    ]
    previous_tips = [
        "In der Hot-Zone kleinere Korrekturen setzen.",
        "Beim Exit Druck besser mitnehmen.",
    ]
    phases = _tip_focus_from_previous(previous_score_rows=previous_rows, previous_tips=previous_tips)
    assert phases == ["Exit", "Aufbau 10-20s", "Hot-Zone", "Stabilitaet / Kipp-Risiko"]


def test_tip_follow_status_classifies_implemented_partial_open():
    assert _tip_follow_status(score_delta=8, positive_hits=2, negative_hits=0)[0] == "umgesetzt"
    assert _tip_follow_status(score_delta=2, positive_hits=1, negative_hits=0)[0] == "teilweise"
    assert _tip_follow_status(score_delta=-7, positive_hits=0, negative_hits=2)[0] == "offen"


def test_view_mode_normalization_and_simple_status():
    assert _normalize_view_mode("simple") == "simple"
    assert _normalize_view_mode("SIMPLE") == "simple"
    assert _normalize_view_mode("expert") == "expert"
    assert _normalize_view_mode("unknown") == "expert"
    assert _normalize_view_mode(None) == "expert"

    assert _jumper_overview_simple_status(stable_count=0, unstable_count=0) == "Noch offen"
    assert _jumper_overview_simple_status(stable_count=7, unstable_count=2) == "Stabil"
    assert _jumper_overview_simple_status(stable_count=5, unstable_count=5) == "Wechselhaft"
    assert _jumper_overview_simple_status(stable_count=2, unstable_count=6) == "Unruhig"


def test_jump_brief_simple_uses_plain_language_and_limits_numeric_jargon():
    report = {
        "notes": {"analysis_blocked": False},
    }
    jump_brief = {
        "basis_lines": ["Zusatz-Benchmark: Top-5 schnellste plausible Spruenge aller Springer."],
        "main_issues": [
            "Bis +15s fehlen im Top-5 Vergleich im Schnitt 95.0 km/h bei +10s.",
        ],
        "strengths": [
            "Die Fluglinie bleibt im relevanten Bereich vorwaertsgerichtet (kein relevanter Rueckdrift).",
        ],
        "actions": [
            "Zwischen +10s und +20s mehr Druck aufbauen.",
        ],
    }
    scorecard_rows = [
        {"name": "Exit", "score": 62},
        {"name": "Aufbau 10-20s", "score": 49},
        {"name": "Hot-Zone", "score": 54},
        {"name": "Stabilitaet / Kipp-Risiko", "score": 73},
    ]

    simple = _build_jump_brief_simple(
        report=report,
        jump_brief=jump_brief,
        scorecard_rows=scorecard_rows,
    )

    text_blob = " ".join(
        [simple["summary"], simple.get("basis_line", "")]
        + simple.get("main_issues", [])
        + simple.get("strengths", [])
        + simple.get("actions", [])
    ).lower()
    assert "top-5" not in text_blob
    assert "vvert" not in text_blob
    assert "bei +20s fehlen" not in text_blob
    assert len(simple.get("main_issues", [])) <= 2
    assert len(simple.get("actions", [])) <= 3


def test_render_simple_glossary_wraps_known_terms_with_tooltip():
    html = _render_simple_glossary(
        "Startphase: Nach dem Absprung ruhig Druck halten und kleine Korrekturen machen."
    )
    assert 'class="glossary-term"' in html
    assert 'title="' in html
    assert "Druck halten" in html
    assert "kleine Korrekturen" in html
