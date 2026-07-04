from __future__ import annotations

from app.main import (
    _build_ai_coaching_payload,
    _build_fs2_quality_issue_lines,
    _build_jump_brief_summary,
    _build_jump_brief_simple,
    _build_jumper_stability_reference,
    _build_tip_follow_up,
    _build_jumper_trend_rows,
    _jumper_overview_simple_status,
    _normalize_view_mode,
    _render_simple_glossary,
    _tip_focus_from_previous,
    _tip_follow_status,
)


def _minimal_goal_follow_report(
    *,
    jump_id: str,
    file_name: str,
    angle_10: float,
    angle_15: float,
    risk_score: float = 35.0,
) -> dict:
    time_s = [float(i) for i in range(0, 31)]
    angle = [68.0 + (0.45 * t) for t in time_s]
    vvert = [230.0 + (7.0 * t) for t in time_s]
    vhor = [96.0 - (2.0 * t) for t in time_s]
    return {
        "jump": {"jump_id": jump_id, "file_name": file_name, "t0_utc": "2026-01-01T12:00:00Z"},
        "metrics": {
            "best_3s_vVert_kmh": 410.0,
            "best_3s_start_s": 20.0,
            "best_3s_end_s": 23.0,
            "negative_risk_score": risk_score,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 30.0,
            "exit_profile": {"carry_ratio": 0.75},
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "optimal",
            "hot_zone": "stabil",
            "kipp_risiko": "mittel",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 292.0, "vHor_kmh": 70.0, "angle_deg": angle_10},
            {"t_rel_s": 15.0, "vVert_kmh": 340.0, "vHor_kmh": 58.0, "angle_deg": angle_15},
            {"t_rel_s": 20.0, "vVert_kmh": 390.0, "vHor_kmh": 42.0, "angle_deg": 83.0},
            {"t_rel_s": 24.0, "vVert_kmh": 405.0, "vHor_kmh": 32.0, "angle_deg": 84.0},
            {"t_rel_s": 28.0, "vVert_kmh": 407.0, "vHor_kmh": 31.0, "angle_deg": 84.2},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "hAGL_m": [3600.0 - (55.0 * t) for t in time_s],
            "accVert_mps2": [2.2 for _ in time_s],
        },
    }


_TEST_MARCO_PROFILE = {
    "v10_ref": 300.0,
    "carry_ref": 0.85,
    "gain_10_20_ref": 120.0,
    "angle_20_low": 82.0,
    "angle_20_high": 85.0,
    "dur_400_ref": 2.0,
    "vhor_min_ref": 30.0,
    "vvert_gain_20_25_ref": 10.0,
    "turns_ref": 1.0,
}


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
        "not_good": ["In der schnellen Phase gab es späte Gegenkorrekturen."],
        "improve": ["Priorität 1: In der Hot-Zone kleinere, frühere Korrekturen setzen."],
    }
    scorecard_rows = [
        {"name": "Exit", "score": 82, "reason": "vVert@10s und Carry sind stabil."},
        {"name": "Aufbau 10-20s", "score": 86, "reason": "Der Aufbau ist gleichmäßig."},
        {"name": "Hot-Zone", "score": 42, "reason": "Die >400 km/h Zone wird zu kurz gehalten."},
        {"name": "Stabilität / Kipp-Risiko", "score": 45, "reason": "Mehrere Korrekturen im Schlussteil."},
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
    assert summary["actions"][0] == "In der Hot-Zone kleinere, frühere Korrekturen setzen."
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
            "Priorität 1: In der Hot-Zone kleine, frühe Korrekturen setzen und vHor stabil halten.",
            "Priorität 2: Bis +15s früher Druck aufbauen, damit der Zuwachs im Aufbau wieder passt.",
            "Priorität 3: Ablauf stabil wiederholen und nur kleine Korrekturen setzen.",
        ],
    }
    scorecard_rows = [
        {"name": "Exit", "score": 80, "reason": "Exit ist stabil."},
        {"name": "Aufbau 10-20s", "score": 58, "reason": "Im Aufbau fehlt im frühen Segment Druck."},
        {"name": "Hot-Zone", "score": 49, "reason": "In der Hot-Zone geht zu viel vHor verloren."},
        {"name": "Stabilität / Kipp-Risiko", "score": 73, "reason": "Stabilität ist okay."},
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
    assert any("Bis +15s früher Druck aufbauen" in item for item in summary["actions"])
    assert any("Hot-Zone" in item for item in summary["actions"])
    assert not any("Ablauf stabil wiederholen" in item for item in summary["actions"])

    main_build_idx = next(i for i, item in enumerate(summary["main_issues"]) if "Aufbau" in item)
    main_hot_idx = next(i for i, item in enumerate(summary["main_issues"]) if "Hot" in item)
    assert main_build_idx < main_hot_idx

    build_idx = next(i for i, item in enumerate(summary["actions"]) if "Bis +15s früher Druck aufbauen" in item)
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
            "Exit-Dynamik wird gut mitgenommen (stabiler Übergang von 0-2s auf 2-6s).",
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
        {"name": "Stabilität / Kipp-Risiko", "score": 52, "reason": "Mehrere Korrekturen im Schlussteil."},
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


def test_ai_coaching_payload_uses_compact_facts_without_raw_chart_data():
    report = _minimal_goal_follow_report(
        jump_id="ai",
        file_name="ai.csv",
        angle_10=74.0,
        angle_15=83.0,
    )
    report["jump"]["jumper_name"] = "Test Jumper"
    report["jump"]["jump_context"] = "training"
    review = {
        "happened": ["Bestes 3-Sekunden-Fenster: +20.0s bis +23.0s mit 410.0 km/h."],
        "good": ["Exit war ruhig."],
        "not_good": ["Im Aufbau wird der Winkel zu schnell steil."],
        "improve": ["Prioritaet 1: Im Aufbau ruhiger steigern."],
        "coaching_goals": [
            {
                "id": "phase_10_15_too_steep_not_hold",
                "priority": 1,
                "phase": "Aufbau 10-20s",
                "text": "Im Aufbau nicht zu schnell maximal steil werden.",
                "target_metrics": [
                    {
                        "metric": "angle_10",
                        "label": "Winkel +10s",
                        "direction": "decrease",
                        "min_delta": 1.0,
                        "unit": " Grad",
                    }
                ],
            }
        ],
    }
    jump_brief = {
        "summary": "Groesster Hebel im Aufbau.",
        "main_issues": ["Aufbau: Winkel zu frueh steil."],
        "strengths": ["Exit ruhig."],
        "actions": ["Im Aufbau ruhiger steigern."],
    }
    jump_brief_simple = {
        "summary": "Arbeite am Aufbau.",
        "main_issues": ["Du wirst zu frueh zu steil."],
        "strengths": ["Der Start war ruhig."],
        "actions": ["Bleib am Anfang ruhiger."],
    }
    scorecard_rows = [
        {"name": "Exit", "score": 82, "status": "gut", "reason": "Exit stabil."},
        {"name": "Aufbau 10-20s", "score": 55, "status": "kritisch", "reason": "Winkel steigt zu schnell."},
    ]

    payload = _build_ai_coaching_payload(
        report=report,
        review=review,
        jump_brief=jump_brief,
        jump_brief_simple=jump_brief_simple,
        scorecard_rows=scorecard_rows,
        tip_follow_up={"available": False, "reason": "Kein vorheriger Sprung."},
        jumper_summary={
            "performance_profile": {
                "available": True,
                "summary": "Schnell: Top-3 Training 440.0 km/h.",
                "performance_band": "schnell",
                "performance_band_label": "Schnell",
                "confidence": "medium",
                "confidence_label": "mittel",
                "valid_jump_count": 3,
                "top_available_avg_kmh": 440.0,
            }
        },
        quality_issue_lines=[],
        view_mode="expert",
    )

    serialized = str(payload)
    assert "chart_data" not in serialized
    assert "time_s" not in serialized
    assert "Test Jumper" not in serialized
    assert "ai.csv" not in serialized
    assert payload["jump"]["jumper_name"] == ""
    assert payload["jump"]["file_name"] == ""
    assert payload["review"]["coaching_goals"][0]["target_metrics"][0]["metric"] == "angle_10"
    assert payload["jump"]["jump_context"] == "training"
    assert payload["performance_profile"]["performance_band"] == "schnell"
    assert payload["primary_diagnosis"]["available"] is False


def test_ai_payload_includes_primary_diagnosis_when_available():
    report = {
        "jump": {"jumper_name": "Test Jumper", "file_name": "ai.csv", "jump_context": "training"},
        "metrics": {"best_3s_vVert_kmh": 324.6, "best_3s_start_s": 22.8, "best_3s_end_s": 25.8},
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 31.4},
        "quality_flags": [],
    }
    review = {
        "happened": [],
        "good": [],
        "not_good": [],
        "coaching_goals": [],
        "primary_diagnosis": {
            "available": True,
            "pattern": "late_hard_steepening_vhor_collapse",
            "severity": "high",
            "phase": "transition_to_peak",
            "title": "Zu harter Uebergang in den Steilflug",
            "summary": "Der Winkel wird spaet sehr steil und vHor bricht ein.",
            "main_issue": "Der Uebergang in den Steilflug ist zu hart.",
            "next_focus": "Winkel gleichmaessiger aufbauen und vHor halten.",
            "evidence": {
                "angle_20": 78.4,
                "angle_peak": 86.4,
                "angle_peak_s": 23.4,
                "angle_gain_20_peak": 8.0,
                "vhor_20": 66.7,
                "vhor_min_after_20": 18.5,
                "vhor_drop_after_20_pct": 72.0,
            },
        },
    }
    jump_brief = {
        "summary": "Hauptdiagnose: Zu harter Uebergang.",
        "main_issues": ["Der Uebergang in den Steilflug ist zu hart."],
        "strengths": ["Start nutzbar."],
        "actions": ["Winkel gleichmaessiger aufbauen."],
    }

    payload = _build_ai_coaching_payload(
        report=report,
        review=review,
        jump_brief=jump_brief,
        jump_brief_simple=jump_brief,
        scorecard_rows=[],
        tip_follow_up={"available": False},
        jumper_summary={"performance_profile": {"available": False}},
        quality_issue_lines=[],
        view_mode="expert",
    )

    assert payload["primary_diagnosis"]["available"] is True
    assert payload["primary_diagnosis"]["pattern"] == "late_hard_steepening_vhor_collapse"
    assert payload["primary_diagnosis"]["evidence"]["angle_peak"] == 86.4


def test_simple_brief_prioritizes_primary_diagnosis():
    jump_brief = {
        "summary": "Die groessten Baustellen liegen bei Aufbau und Hot-Zone.",
        "primary_diagnosis": {
            "available": True,
            "title": "Zu harter Uebergang in den Steilflug",
            "summary": "Der Aufbau bleibt moderat, danach wird der Winkel schnell steil und vHor bricht ein.",
            "main_issue": "Der spaete harte Steilflug kostet vHor und Stabilitaet.",
            "next_focus": "Ab +15s gleichmaessiger Richtung 83 bis 85 Grad aufbauen.",
        },
        "main_issues": ["Aufbau: Winkel zu flach.", "Hot-Zone kritisch."],
        "strengths": ["Der Start war kontrolliert."],
        "actions": ["Mehr Druck aufbauen.", "Hot-Zone beruhigen."],
    }

    out = _build_jump_brief_simple(
        report={"notes": {}},
        jump_brief=jump_brief,
        scorecard_rows=[
            {"name": "Aufbau 10-20s", "score": 23},
            {"name": "Hot-Zone", "score": 21},
        ],
    )

    assert out["summary"] == "Zu harter Uebergang in den Steilflug"
    assert out["main_issues"] == [
        "Der Aufbau bleibt moderat, danach wird der Winkel schnell steil und vHor bricht ein."
    ]
    assert out["actions"] == ["Ab +15s gleichmaessiger Richtung 83 bis 85 Grad aufbauen."]


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
    assert "Messqualität (FS2)" in crit_lines[0]
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
    assert by_name["Top-Speed"]["earlier_better_text"].startswith("Top-Speed war früher besser")


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
    assert any("Winkel über etwa" in line for line in ref["unstable_lines"])
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
        {"name": "Stabilität / Kipp-Risiko", "score": 75},
    ]
    previous_tips = [
        "In der Hot-Zone kleinere Korrekturen setzen.",
        "Beim Exit Druck besser mitnehmen.",
    ]
    phases = _tip_focus_from_previous(previous_score_rows=previous_rows, previous_tips=previous_tips)
    assert phases == ["Exit", "Aufbau 10-20s", "Hot-Zone", "Stabilität / Kipp-Risiko"]


def test_tip_follow_status_classifies_implemented_partial_open():
    assert _tip_follow_status(score_delta=8, positive_hits=2, negative_hits=0)[0] == "umgesetzt"
    assert _tip_follow_status(score_delta=2, positive_hits=1, negative_hits=0)[0] == "teilweise"
    assert _tip_follow_status(score_delta=-7, positive_hits=0, negative_hits=2)[0] == "offen"


def test_tip_follow_up_uses_structured_goal_metrics_before_text_fallback():
    previous_report = _minimal_goal_follow_report(
        jump_id="prev",
        file_name="previous.csv",
        angle_10=75.0,
        angle_15=85.0,
    )
    current_report = _minimal_goal_follow_report(
        jump_id="current",
        file_name="current.csv",
        angle_10=72.5,
        angle_15=82.0,
    )
    previous_goals = [
        {
            "id": "phase_10_15_too_steep_not_hold",
            "phase": "Aufbau 10-20s",
            "text": "Im Aufbau nicht zu schnell maximal steil werden.",
            "target_metrics": [
                {
                    "metric": "angle_10",
                    "label": "Winkel +10s",
                    "direction": "decrease",
                    "min_delta": 1.0,
                    "unit": " Grad",
                    "decimals": 1,
                },
                {
                    "metric": "angle_15",
                    "label": "Winkel +15s",
                    "direction": "decrease",
                    "min_delta": 1.0,
                    "unit": " Grad",
                    "decimals": 1,
                },
            ],
        }
    ]

    follow_up = _build_tip_follow_up(
        current_report=current_report,
        previous_report=previous_report,
        marco_profile=_TEST_MARCO_PROFILE,
        previous_coaching_goals=previous_goals,
    )

    assert follow_up["available"] is True
    assert follow_up["entries"][0]["status_key"] == "umgesetzt"
    assert follow_up["entries"][0]["goal_text"] == "Im Aufbau nicht zu schnell maximal steil werden."
    assert "Winkel +10s" in follow_up["entries"][0]["detail"]
    assert "Winkel +15s" in follow_up["entries"][0]["detail"]


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
        "basis_lines": ["Zusatz-Benchmark: Top-5 schnellste plausible Sprünge aller Springer."],
        "main_issues": [
            "Bis +15s fehlen im Top-5 Vergleich im Schnitt 95.0 km/h bei +10s.",
        ],
        "strengths": [
            "Die Fluglinie bleibt im relevanten Bereich vorwärtsgerichtet (kein relevanter Rückdrift).",
        ],
        "actions": [
            "Zwischen +10s und +20s mehr Druck aufbauen.",
        ],
    }
    scorecard_rows = [
        {"name": "Exit", "score": 62},
        {"name": "Aufbau 10-20s", "score": 49},
        {"name": "Hot-Zone", "score": 54},
        {"name": "Stabilität / Kipp-Risiko", "score": 73},
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
