from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from fastapi import BackgroundTasks
from jinja2 import Environment, FileSystemLoader

from app.config import TECHNICAL_PHASE_SPECS
from app.main import (
    _AI_COACHING_INFLIGHT,
    _ai_coaching_status_payload,
    _annotate_scorecard_with_reference,
    _build_ai_coaching_payload,
    _build_coaching_snapshot,
    _build_fs2_quality_issue_lines,
    _build_jump_brief_simple,
    _build_jump_brief_summary,
    _build_jumper_profile_ai_payload,
    _build_jumper_stability_reference,
    _build_jumper_timing_reference,
    _build_jumper_trend_rows,
    _build_phase_rows_with_reference,
    _build_scorecard_rows,
    _build_tip_follow_up,
    _harmonize_action_texts,
    _jumper_overview_simple_status,
    _normalize_view_mode,
    _phase_rows_for_report,
    _queue_ai_coaching,
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


def test_jumper_profile_ai_payload_uses_all_profile_signals_without_identifier():
    payload = _build_jumper_profile_ai_payload(
        jumper_name="Test Jumper",
        view_mode="simple",
        jumper_summary={
            "jump_count": 12,
            "learning_jump_count": 10,
            "excluded_learning_jump_count": 2,
            "best_speed_kmh": 421.25,
            "trend_summary": "Positiver Verlauf.",
            "improved_points": ["Aufbau wurde besser."],
            "worse_points": ["Das schnelle Fenster schwankt."],
            "earlier_better_points": [],
            "focus_actions": ["Die gute Linie stabil wiederholen."],
            "trend_rows": [
                {
                    "name": "Regel-Score",
                    "early_value": 390.0,
                    "recent_value": 410.0,
                    "delta": 20.0,
                    "unit": "km/h",
                    "status": "besser",
                }
            ],
            "performance_profile": {"available": False},
            "feedback_training_profile": {"available": False},
            "stability_reference": {"available": False},
            "timing_reference": {"available": False},
            "tip_effect_profile": {"available": False},
        },
    )

    assert payload["report_kind"] == "jumper_profile"
    assert payload["view_mode"] == "simple"
    assert payload["jump"]["jumper_name"] == ""
    assert payload["profile"]["jump_count"] == 12
    assert payload["profile"]["trend_rows"][0]["delta"] == 20.0
    assert payload["jump_brief"]["actions"] == ["Die gute Linie stabil wiederholen."]
    assert payload["quality"]["quality_issue_lines"]


def test_jumper_template_exposes_profile_ai_in_simple_and_expert_views():
    template = Path("app/templates/jumper_detail.html").read_text(encoding="utf-8")

    assert "KI-Erklärung &amp; Profil-Coach" in template
    assert "Was dahintersteckt" in template
    assert "KI-Coaching-Erklärung" in template
    assert "KI-Coaching-Fokus" in template
    assert "Profilbewertung in Analyse" in template
    assert '_ai_analysis_loading.html' in template
    assert "/ai-coaching-status?cache_key=" in template


def test_pending_jump_report_renders_only_metadata_and_analysis_loader():
    env = Environment(loader=FileSystemLoader("app/templates"), autoescape=True)
    template = env.get_template("jump_detail.html")
    report = {
        "jump": {
            "jumper_name": "Test Jumper",
            "file_name": "test.csv",
            "jump_context": "training",
            "t0_utc": "2026-07-21T10:00:00Z",
            "sample_rate_hz": 10,
            "is_valid_altitude": True,
        },
        "metrics": {"analysis_version": "1.1.0"},
        "notes": {"analysis_blocked": False},
        "feedback": {"available": False},
        "dropzone": None,
        "dropzone_match": None,
    }

    for view_mode in ("simple", "expert"):
        rendered = template.render(
            view_mode=view_mode,
            jump_id="jump-pending",
            report=report,
            ai_coaching={"pending": True, "cache_key": f"cache-{view_mode}"},
            coach_view_enabled=False,
            needs_plotly=False,
            quality_issue_lines=[],
            jump_context_options=[],
            dropzone_options=[],
            message=None,
            error=None,
        )

        assert "Sprungbewertung in Analyse" in rendered
        assert "speed-skydive-loader.svg" in rendered
        assert "Test Jumper" in rendered
        assert "Speed Score" not in rendered
        assert "Speed Übersicht" not in rendered
        assert "Coaching-Bewertung" not in rendered
        assert "Scorecard" not in rendered
        assert "Kurvenanalyse" not in rendered
        assert "Diagnose" not in rendered
        assert "plotly.min.js" not in rendered


def test_pending_jumper_profile_hides_profile_scores_comparisons_and_charts():
    env = Environment(loader=FileSystemLoader("app/templates"), autoescape=True)
    template = env.get_template("jumper_detail.html")

    rendered = template.render(
        view_mode="expert",
        jumper_name="Test Jumper",
        jumper_ai_coaching={"pending": True, "cache_key": "profile-cache"},
        coach_view_enabled=False,
        needs_plotly=False,
    )

    assert "Profilbewertung in Analyse" in rendered
    assert "speed-skydive-loader.svg" in rendered
    assert "Gesamtbild" not in rendered
    assert "Regel-Score" not in rendered
    assert "Sprünge vergleichen" not in rendered
    assert "Trend 3s-Max" not in rendered
    assert "plotly.min.js" not in rendered


def test_ai_status_payload_is_additive_and_distinguishes_ready_error_pending():
    assert _ai_coaching_status_payload(None) == {
        "state": "pending",
        "complete": False,
        "available": False,
    }
    assert _ai_coaching_status_payload(
        {"_cache_status": "ready", "available": True}
    ) == {"state": "ready", "complete": True, "available": True}
    assert _ai_coaching_status_payload(
        {"_cache_status": "error", "available": False}
    ) == {"state": "error", "complete": True, "available": False}


def test_shared_ai_loader_keeps_polling_after_old_twenty_second_limit():
    script = Path("app/static/ai-loading.js").read_text(encoding="utf-8")
    base = Path("app/templates/base.html").read_text(encoding="utf-8")

    assert 'src="/static/ai-loading.js" defer' in base
    assert "attempts < 20" not in script
    assert "attempts < 45" not in script
    assert "elapsedMs < 30000 ? 1000 : 3000" in script
    assert 'state === "ready" || state === "error"' in script
    assert "elapsedMs >= 60000" in script
    assert "window.location.reload()" in script


def test_ai_loader_svg_is_local_and_well_formed():
    path = Path("app/static/speed-skydive-loader.svg")
    root = ElementTree.parse(path).getroot()

    assert root.tag.endswith("svg")
    assert root.attrib["viewBox"] == "0 0 96 96"
    assert len([node for node in root.iter() if node.tag.endswith("path")]) >= 3


def test_ai_background_queue_deduplicates_repeated_refreshes():
    cache_key = "profile-refresh-test"
    background_tasks = BackgroundTasks()
    _AI_COACHING_INFLIGHT.discard(cache_key)
    try:
        first = _queue_ai_coaching(
            background_tasks,
            payload={"report_kind": "jumper_profile"},
            cache_key=cache_key,
            jump_id=None,
            analysis_signature="profile-signature",
            view_mode="expert",
            prompt_version="profile-test",
        )
        second = _queue_ai_coaching(
            background_tasks,
            payload={"report_kind": "jumper_profile"},
            cache_key=cache_key,
            jump_id=None,
            analysis_signature="profile-signature",
            view_mode="expert",
            prompt_version="profile-test",
        )

        assert first is True
        assert second is False
        assert len(background_tasks.tasks) == 1
    finally:
        _AI_COACHING_INFLIGHT.discard(cache_key)


def test_ai_payload_includes_compact_feedback_context():
    report = _minimal_goal_follow_report(
        jump_id="feedback-payload",
        file_name="feedback.csv",
        angle_10=74.0,
        angle_15=80.0,
    )
    feedback_context = {
        "available": True,
        "text_excerpt": "Ich wollte Arme enger fuehren und am Ende hat es gewackelt.",
        "intents": [{"key": "arms_closer", "label": "Arme/Haende enger"}],
        "felt_issues": [{"key": "end_instability", "label": "spaete Unruhe"}],
        "match_lines": ["Dein Gefuehl von spaeter Unruhe passt zu den Messdaten."],
        "coaching_hint": "Dein Gefuehl von spaeter Unruhe passt zu den Messdaten.",
        "next_focus_hint": "Kompaktere Haltung nur kleiner testen.",
        "confidence": "medium",
        "caution": "Feedback ist subjektiver Kontext; Messdaten bleiben die Bewertungsgrundlage.",
        "evidence": {
            "objective_signals": ["Hot-Zone kritisch"],
            "end_unstable": True,
            "lateral": {"hot_pattern": "schlangenlinie", "hot_vlat_abs_mean_kmh": 9.2},
        },
    }

    payload = _build_ai_coaching_payload(
        report=report,
        review={"happened": [], "good": [], "not_good": [], "coaching_goals": []},
        jump_brief={"summary": "Test", "main_issues": [], "strengths": [], "actions": []},
        jump_brief_simple={"summary": "Test", "main_issues": [], "strengths": [], "actions": []},
        scorecard_rows=[],
        tip_follow_up={"available": False},
        jumper_summary={
            "performance_profile": {"available": False},
            "feedback_training_profile": {
                "available": True,
                "feedback_count": 2,
                "recent_feedback_count": 2,
                "summary": "Aktueller Trainingskontext aus Feedback: Arme/Haende enger.",
                "lines": ["Aktueller Trainingskontext aus Feedback: Arme/Haende enger."],
                "top_focus_keys": ["arms_closer"],
            },
        },
        quality_issue_lines=[],
        view_mode="expert",
        feedback_context=feedback_context,
    )

    assert payload["jump_feedback"]["available"] is True
    assert payload["jump_feedback"]["intents"][0]["key"] == "arms_closer"
    assert payload["jump_feedback"]["evidence"]["end_unstable"] is True
    assert payload["feedback_training_profile"]["available"] is True


def test_ai_payload_includes_compact_personal_timing_reference():
    report = _minimal_goal_follow_report(
        jump_id="timing-payload",
        file_name="timing.csv",
        angle_10=74.0,
        angle_15=80.0,
    )

    payload = _build_ai_coaching_payload(
        report=report,
        review={"happened": [], "good": [], "not_good": [], "coaching_goals": []},
        jump_brief={"summary": "Test", "main_issues": [], "strengths": [], "actions": []},
        jump_brief_simple={"summary": "Test", "main_issues": [], "strengths": [], "actions": []},
        scorecard_rows=[],
        tip_follow_up={"available": False},
        jumper_summary={
            "performance_profile": {"available": False},
            "timing_reference": {
                "available": True,
                "maturity": "active",
                "confidence": "medium",
                "basis_count": 5,
                "usable_count": 5,
                "source": "stable_control",
                "best_reference_kmh": 434.0,
                "top3_avg_kmh": 433.0,
                "max_timing_spread_s": 0.9,
                "summary_lines": ["Persoenliches Timing: 82 Grad meist bei +13.0 bis +13.6s."],
                "anchors": {
                    "angle_82_time_s": {"label": "82 Grad", "low": 13.0, "high": 13.6, "median": 13.3},
                    "best_3s_start_s": {
                        "label": "Start bestes 3s-Fenster",
                        "low": 20.0,
                        "high": 20.9,
                        "median": 20.45,
                    },
                },
            },
        },
        quality_issue_lines=[],
        view_mode="expert",
        feedback_context=None,
    )

    assert payload["personal_timing"]["available"] is True
    assert payload["personal_timing"]["basis_count"] == 5
    assert payload["personal_timing"]["maturity"] == "active"
    assert payload["personal_timing"]["top3_avg_kmh"] == 433.0
    assert payload["personal_timing"]["anchors"]["angle_82_time_s"]["median_s"] == 13.3


def test_scorecard_keeps_five_phase_structure_when_max_speed_not_evaluable():
    report = _minimal_goal_follow_report(
        jump_id="short-window-scorecard",
        file_name="short.csv",
        angle_10=74.0,
        angle_15=80.0,
    )
    report["notes"]["decel_start_s"] = 19.8

    rows = _build_scorecard_rows(report)

    assert [row["name"] for row in rows] == [phase["name"] for phase in TECHNICAL_PHASE_SPECS]
    assert rows[-1]["name"] == "Max-Speed Fenster"
    assert rows[-1]["target_status"] == "nicht belastbar"
    assert "Bewertungsfenster endet" in rows[-1]["reason"]


def test_scorecard_does_not_load_reference_profile_implicitly(monkeypatch):
    report = _minimal_goal_follow_report(
        jump_id="scorecard-without-database",
        file_name="scorecard-without-database.csv",
        angle_10=74.0,
        angle_15=80.0,
    )

    def fail_on_database_profile_lookup(*, limit: int = 15):
        raise AssertionError(f"unexpected database profile lookup with limit={limit}")

    monkeypatch.setattr("app.main._get_marco_top15_profile", fail_on_database_profile_lookup)

    rows = _build_scorecard_rows(report)

    assert [row["name"] for row in rows] == [phase["name"] for phase in TECHNICAL_PHASE_SPECS]


def test_scorecard_penalizes_early_horizontal_reserve_collapse():
    time_s = [i * 0.5 for i in range(0, 61)]
    vvert = []
    vhor = []
    angle = []
    forward_m = []
    for t in time_s:
        vvert.append(230.0 + min(t, 25.0) * 8.2)
        if t <= 18.0:
            vhor.append(92.0 - t * 3.0)
        elif t <= 21.0:
            vhor.append(38.0 - (t - 18.0) * 5.2)
        elif t <= 25.0:
            vhor.append(22.4 + (t - 21.0) * 5.5)
        else:
            vhor.append(44.4 + min(3.0, (t - 25.0) * 0.8))
        if t <= 21.0:
            angle.append(70.0 + t * 0.82)
        elif t <= 25.0:
            angle.append(87.2 - (t - 21.0) * 0.75)
        else:
            angle.append(84.2)
        if t <= 21.0:
            forward_m.append(t * 28.0)
        else:
            forward_m.append(588.0 - (t - 21.0) * 12.0)

    report = {
        "jump": {"jump_id": "early-reserve-scorecard", "file_name": "early.csv"},
        "metrics": {
            "best_3s_vVert_kmh": 432.0,
            "best_3s_start_s": 24.5,
            "best_3s_end_s": 27.5,
            "negative_risk_score": 0.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 30.0, "decel_start_s": 31.0},
        "scorecard": {"exit": "sauber", "phase_10_20": "optimal", "hot_zone": "stabil", "kipp_risiko": "niedrig"},
        "quality_flags": [],
        "fixpoints": [],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "hAGL_m": [3600.0 - 55.0 * t for t in time_s],
            "accVert_mps2": [2.0 for _ in time_s],
            "forward_m": forward_m,
        },
    }

    rows = _build_scorecard_rows(report)
    hot_zone = next(row for row in rows if row["name"] == "Hot-Zone Aufbau")
    max_speed = next(row for row in rows if row["name"] == "Max-Speed Fenster")

    assert hot_zone["score"] <= 62
    assert max_speed["score"] <= 64
    assert any("Horizontale Reserve zu frueh verbraucht" in line for line in hot_zone["reason_lines"])
    assert any("vor dem 3s-Fenster" in line for line in max_speed["reason_lines"])


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


def test_jump_brief_summary_hides_global_top5_basis_below_elite_speed():
    report = {
        "jump": {"file_name": "sub-elite.csv"},
        "metrics": {
            "best_3s_vVert_kmh": 401.0,
            "best_3s_start_s": 22.0,
            "best_3s_end_s": 25.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 34.0},
    }

    summary = _build_jump_brief_summary(
        report=report,
        review={"happened": [], "good": [], "not_good": [], "improve": []},
        scorecard_rows=[],
        best_reference=None,
        top_reference_jumps=[{"jump_id": "elite", "best_3s_vVert_kmh": 520.0}],
    )

    assert not any("Top-5" in line for line in summary["basis_lines"])


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


def test_jump_brief_summary_suppresses_pressure_tip_when_build_is_too_steep():
    report = {
        "jump": {"file_name": "too-steep.csv"},
        "metrics": {
            "best_3s_vVert_kmh": 416.0,
            "best_3s_start_s": 21.0,
            "best_3s_end_s": 24.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 34.0},
    }
    review = {
        "happened": [],
        "good": [],
        "not_good": [
            "Bei +10s ist der Tauchwinkel fuer dein aktuelles stabiles Niveau zu steil.",
            "In der Hot-Zone wechselst du seitlich mehrfach die Richtung.",
        ],
        "improve": [
            "Prioritaet 1: Zwischen +10s und +20s mehr Druck aufbauen. Ziel: in diesem Abschnitt mindestens +180 km/h Zuwachs.",
            "Prioritaet 2: Bis +10s zuerst stabil im Bereich 72.3 bis 76.5 Grad bleiben. Erst wenn die Linie bis +15s ruhig bleibt, schrittweise weiter steiler werden.",
            "Prioritaet 3: Hot-Zone ruhiger halten.",
        ],
    }
    scorecard_rows = [
        {"name": "Aufbau 10-20s", "score": 45, "reason": "Der Tauchwinkel ist zu steil."},
        {"name": "Hot-Zone", "score": 51, "reason": "In der Hot-Zone gibt es Nachkorrekturen."},
    ]

    summary = _build_jump_brief_summary(
        report=report,
        review=review,
        scorecard_rows=scorecard_rows,
        best_reference=None,
        top_reference_jumps=[],
    )

    assert any("zuerst stabil" in item for item in summary["actions"])
    assert not any("mehr Druck aufbauen" in item for item in summary["actions"])


def test_jump_brief_summary_keeps_personal_stability_corridor_in_build_focus():
    report = {
        "jump": {"file_name": "stability-corridor.csv"},
        "metrics": {
            "best_3s_vVert_kmh": 438.6,
            "best_3s_start_s": 23.9,
            "best_3s_end_s": 26.9,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 31.0},
    }
    review = {
        "happened": [],
        "good": [],
        "not_good": ["Bei +20s wird der Winkel zu hart in die schnelle Phase geschoben."],
        "improve": [
            "Prioritaet 1: Bei +20s nicht ueber deinen persoenlichen Stabilitaetsbereich schieben: 77.3 bis 80.4 Grad.",
            "Prioritaet 2: Die letzte schnelle Phase mit weniger Lenkimpulsen fliegen.",
        ],
    }

    summary = _build_jump_brief_summary(
        report=report,
        review=review,
        scorecard_rows=[
            {"name": "Aufbau 10-20s", "score": 45, "reason": "Winkel wird zu steil."},
            {"name": "Stabilitaet / Kipp-Risiko", "score": 50, "reason": "Linie wird unruhig."},
        ],
        best_reference=None,
        top_reference_jumps=[],
    )

    assert "Hauptbeschleunigung" in summary["summary"]


def test_harmonize_action_texts_merges_redundant_fast_correction_tips():
    actions = [
        "Die letzte schnelle Phase mit weniger Lenkimpulsen fliegen: kleine fruehe Korrekturen statt spaeter grosser Gegenkorrektur.",
        "Korrekturen frueher und kleiner setzen, damit die Beschleunigungskurve ruhiger wird und der Speed nicht durch harte Gegenbewegungen verloren geht.",
    ]

    merged = _harmonize_action_texts(actions, max_items=5)

    assert merged == [
        "In der letzten schnellen Phase Korrekturen frueher und kleiner setzen, damit Lenkimpulse und Beschleunigungskurve ruhiger werden."
    ]


def test_harmonize_action_texts_keeps_build_gain_and_angle_control_separate():
    actions = [
        "Bei +20s nicht ueber deinen persoenlichen Stabilitaetsbereich schieben: 81.5 bis 84.4 Grad.",
        "Im Segment +10 bis +15s frueher Druck aufbauen, aber den Winkel nicht erzwingen und die Linie ruhiger halten, damit der Zuwachs in Richtung +94 km/h geht.",
    ]

    merged = _harmonize_action_texts(actions, max_items=5)

    assert merged == [actions[1], actions[0]]


def test_expert_ai_template_keeps_rule_based_training_tips_visible():
    template = Path("app/templates/jump_detail.html").read_text(encoding="utf-8")
    ai_branch_start = template.index("{% if ai_coaching and ai_coaching.available %}")
    fallback_start = template.index("{% elif vm == \"simple\" %}", ai_branch_start)
    ai_branch = template[ai_branch_start:fallback_start]

    assert "KI-Coaching-Fokus" in ai_branch
    assert "Konkrete Trainings-Tipps" in ai_branch
    assert "{% for item in jump_brief.actions %}" in ai_branch


def test_report_template_exposes_dropzone_evidence_and_manual_correction():
    template = Path("app/templates/jump_detail.html").read_text(encoding="utf-8")

    assert "report.dropzone" in template
    assert "dropzone_assignment_confidence" in template
    assert 'action="/jumps/{{ jump_id }}/dropzone?view={{ vm }}"' in template
    assert "Automatisch neu erkennen" in template


def test_jump_brief_summary_harmonizes_redundant_training_tips():
    report = {
        "jump": {"file_name": "redundant-tips.csv"},
        "metrics": {
            "best_3s_vVert_kmh": 430.0,
            "best_3s_start_s": 22.0,
            "best_3s_end_s": 25.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 31.0},
    }
    review = {
        "happened": [],
        "good": [],
        "not_good": [
            "Bei +20s wird der Winkel zu hart in die schnelle Phase geschoben.",
            "Sobald du schnell wirst, wird die Linie unruhig.",
        ],
        "improve": [
            "Prioritaet 1: Bei +20s nicht ueber deinen persoenlichen Stabilitaetsbereich schieben: 81.5 bis 84.4 Grad.",
            "Prioritaet 2: Im Segment +10 bis +15s frueher Druck aufbauen, aber den Winkel nicht erzwingen und die Linie ruhiger halten, damit der Zuwachs in Richtung +94 km/h geht.",
            "Prioritaet 3: Die letzte schnelle Phase mit weniger Lenkimpulsen fliegen: kleine fruehe Korrekturen statt spaeter grosser Gegenkorrektur.",
            "Prioritaet 4: Korrekturen frueher und kleiner setzen, damit die Beschleunigungskurve ruhiger wird und der Speed nicht durch harte Gegenbewegungen verloren geht.",
        ],
    }

    summary = _build_jump_brief_summary(
        report=report,
        review=review,
        scorecard_rows=[
            {"name": "Hauptbeschleunigung", "score": 62, "reason": "Winkel wird zu hart."},
            {"name": "Max-Speed Fenster", "score": 58, "reason": "Linie wird unruhig."},
        ],
        best_reference=None,
        top_reference_jumps=[],
    )

    assert any("Stabilitaetsbereich" in item for item in summary["actions"])
    assert any("+10 bis +15s" in item for item in summary["actions"])
    assert sum("Lenkimpulse" in item or "Beschleunigungskurve" in item for item in summary["actions"]) == 1
    assert len(summary["actions"]) == 3
    assert "+10 bis +15s" in summary["actions"][0]
    assert "+20s" in summary["actions"][1]


def test_jump_brief_summary_uses_actual_issue_topics_for_focus_line():
    report = {
        "jump": {"file_name": "focus.csv"},
        "metrics": {
            "best_3s_vVert_kmh": 401.6,
            "best_3s_start_s": 22.0,
            "best_3s_end_s": 25.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 34.0},
    }
    review = {
        "happened": [],
        "good": [],
        "not_good": [
            "Sobald du schnell wirst, wird die Linie unruhig mit Seitbewegung und Winkelschwankung.",
            "Hot-Zone: Du kommst in die sehr schnelle Phase, haeltst sie aber nur kurz.",
            "Stabilitaet / Kipp-Risiko: In der schnellen Phase verlierst du zu stark Vorwaertsbewegung.",
        ],
        "improve": [
            "Prioritaet 1: Hot-Zone ruhiger halten.",
            "Prioritaet 2: Die letzte schnelle Phase mit weniger Lenkimpulsen fliegen.",
        ],
    }
    scorecard_rows = [
        {
            "name": "Aufbau 10-20s",
            "score": 33,
            "reason": "Der Aufbau ist sehr stark und dabei kontrolliert. Der Tauchwinkel ist dabei eher zu flach.",
            "reason_lines": [
                "Der Aufbau ist sehr stark und dabei kontrolliert.",
                "Der Tauchwinkel ist dabei eher zu flach.",
            ],
        },
        {"name": "Hot-Zone", "score": 48, "reason": "Du haeltst die schnelle Phase nur kurz."},
        {"name": "Stabilitaet / Kipp-Risiko", "score": 43, "reason": "In der schnellen Phase verlierst du vHor."},
    ]

    summary = _build_jump_brief_summary(
        report=report,
        review=review,
        scorecard_rows=scorecard_rows,
        best_reference=None,
        top_reference_jumps=[],
    )

    assert "Hot-Zone" in summary["summary"]
    assert "Max-Speed" in summary["summary"]
    assert "Aufbau 10-20s" not in summary["summary"]
    assert not any("Tauchwinkel ist dabei" in item for item in summary["main_issues"])


def test_jump_brief_summary_does_not_replace_issue_focus_with_action_focus():
    report = {
        "jump": {"file_name": "issue-focus.csv"},
        "metrics": {
            "best_3s_vVert_kmh": 416.2,
            "best_3s_start_s": 22.8,
            "best_3s_end_s": 25.8,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 28.0},
    }
    review = {
        "happened": [],
        "good": ["Das Max-Speed-Fenster war nutzbar."],
        "not_good": [
            "Phase +10 bis +15s: Zuwachs +58.8 km/h (dein stabiler Korridor: +67.9 bis +70.7).",
            "Die Hot-Zone ist ineffizient: viel Hoehenverlust bei zu wenig zusaetzlichem Speed.",
        ],
        "improve": [
            "Prioritaet 1: Im Max-Speed-Fenster den Winkel halten und Korrekturen klein halten.",
            "Prioritaet 2: Zwischen +10s und +20s mehr Druck aufbauen.",
        ],
    }

    summary = _build_jump_brief_summary(
        report=report,
        review=review,
        scorecard_rows=[
            {"name": "Hauptbeschleunigung", "score": 63, "reason": "Zuwachs ist zu niedrig."},
            {"name": "Hot-Zone Aufbau", "score": 50, "reason": "Hot-Zone ist ineffizient."},
            {"name": "Max-Speed Fenster", "score": 75, "reason": "Fenster ist nutzbar."},
        ],
        best_reference=None,
        top_reference_jumps=[],
    )

    assert "Hauptbeschleunigung" in summary["summary"]
    assert "Hot-Zone" in summary["summary"]
    assert "Max-Speed" not in summary["summary"]


def test_jump_brief_summary_removes_duplicate_primary_transition_action():
    report = {
        "jump": {"file_name": "primary.csv"},
        "metrics": {
            "best_3s_vVert_kmh": 324.6,
            "best_3s_start_s": 22.0,
            "best_3s_end_s": 25.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 31.0},
    }
    review = {
        "happened": [],
        "good": [],
        "not_good": ["Der Winkel wird nicht gleichmaessig in die schnelle Phase gefuehrt."],
        "improve": [
            "Prioritaet 1: Den Uebergang in den Steilflug frueher und gleichmaessiger fahren.",
            "Prioritaet 2: Nach dem Exit den Druck laenger tragen.",
        ],
        "primary_diagnosis": {
            "available": True,
            "pattern": "late_hard_steepening_vhor_collapse",
            "title": "Zu harter Uebergang in den Steilflug",
            "main_issue": "Der spaete harte Steilflug kostet vHor und Stabilitaet.",
            "next_focus": "Ab +15s schrittweise Richtung 83 bis 85 Grad aufbauen und vHor halten.",
        },
    }

    summary = _build_jump_brief_summary(
        report=report,
        review=review,
        scorecard_rows=[{"name": "Hot-Zone", "score": 42, "reason": "vHor bricht ein."}],
        best_reference=None,
        top_reference_jumps=[],
    )

    assert summary["actions"][0].startswith("Ab +15s")
    assert not any("Uebergang in den Steilflug" in item for item in summary["actions"][1:])


def test_scorecard_filters_legacy_reference_score_context_from_phase_reasons():
    report = _minimal_goal_follow_report(
        jump_id="ref-score",
        file_name="ref-score.csv",
        angle_10=70.0,
        angle_15=77.0,
    )
    marco_profile = {
        "v10_ref": 430.0,
        "carry_ref": 0.94,
        "gain_10_20_ref": 324.0,
        "angle_20_low": 83.8,
        "angle_20_high": 86.0,
        "dur_400_ref": 5.5,
        "vhor_min_ref": 31.8,
        "vvert_gain_20_25_ref": 36.6,
        "turns_ref": 0.0,
    }

    rows = _build_scorecard_rows(report, marco_profile=marco_profile)
    build_row = next(row for row in rows if row["name"] == "Hauptbeschleunigung")

    assert build_row["score"] < 70
    assert "Referenzscore" not in build_row["reason"]
    assert "Referenzband" not in build_row["reason"]
    assert "Aufbaufenster" not in build_row["reason"]


def test_scorecard_phase_reason_order_puts_measurements_before_rating():
    report = _minimal_goal_follow_report(
        jump_id="scorecard-order",
        file_name="scorecard-order.csv",
        angle_10=74.0,
        angle_15=80.0,
    )

    rows = _build_scorecard_rows(report)
    hot_zone = next(row for row in rows if row["name"] == "Hot-Zone Aufbau")
    reason_lines = hot_zone["reason_lines"]

    assert reason_lines[0] == "+15.0s bis +22.0s, Zielwinkel 82-86 Grad."
    assert reason_lines[1].startswith("Istwerte: Avg Winkel")
    assert "Avg vVert" in reason_lines[1]
    assert reason_lines[2].startswith("Bewertung: zu flach")
    assert reason_lines.index(next(line for line in reason_lines if line.startswith("Bewertung:"))) < reason_lines.index(
        next(line for line in reason_lines if line.startswith("vVert-Zuwachs"))
    )


def test_scorecard_reference_phase_average_lines_are_neutral_deltas():
    report = _minimal_goal_follow_report(
        jump_id="scorecard-current",
        file_name="scorecard-current.csv",
        angle_10=74.0,
        angle_15=80.0,
    )
    reference = _minimal_goal_follow_report(
        jump_id="scorecard-reference",
        file_name="scorecard-reference.csv",
        angle_10=72.0,
        angle_15=78.0,
    )
    reference["chart_data"]["vVert_kmh"] = [value - 20.0 for value in reference["chart_data"]["vVert_kmh"]]

    rows = _annotate_scorecard_with_reference(
        rows=_build_scorecard_rows(report),
        report=report,
        reference_report=reference,
    )
    hot_zone = next(row for row in rows if row["name"] == "Hot-Zone Aufbau")
    avg_vvert_line = next(line for line in hot_zone["benchmark_lines"] if line.startswith("Avg vVert"))

    assert "Delta" in avg_vvert_line
    assert "bestes Ergebnis bisher" not in avg_vvert_line


def test_max_speed_eher_flach_comment_matches_status():
    report = _minimal_goal_follow_report(
        jump_id="max-speed-eher-flach",
        file_name="max-speed-eher-flach.csv",
        angle_10=77.0,
        angle_15=82.0,
    )
    time_s = [float(i) for i in range(0, 31)]
    angle = []
    for t in time_s:
        if t < 20.0:
            angle.append(78.0 + (0.25 * t))
        elif t == 20.0:
            angle.append(79.5)
        else:
            angle.append(84.0)
    report["chart_data"]["time_s"] = time_s
    report["chart_data"]["angle_deg"] = angle

    phase_rows = _phase_rows_for_report(report)
    max_speed = next(row for row in phase_rows if row["name"] == "Max-Speed Fenster")

    assert max_speed["target_status"] == "eher flach"
    assert "einzelne Abschnitte fallen darunter" in max_speed["comment"]


def test_short_too_steep_phase_explains_avg_and_max_angle():
    report = _minimal_goal_follow_report(
        jump_id="short-too-steep",
        file_name="short-too-steep.csv",
        angle_10=74.0,
        angle_15=80.0,
    )
    time_s = report["chart_data"]["time_s"]
    angle = [78.0 for _ in time_s]
    for i, t in enumerate(time_s):
        if 3.0 <= t < 8.0:
            angle[i] = 62.0
        elif t == 8.0:
            angle[i] = 72.3
    report["chart_data"]["angle_deg"] = angle

    phase_rows = _phase_rows_for_report(report)
    dive_phase = next(row for row in phase_rows if row["name"] == "Dive-Aufbau")

    assert dive_phase["target_status"] == "kurz zu steil"
    assert dive_phase["angle_start_deg"] == 62.0
    assert dive_phase["angle_end_deg"] == 72.3
    assert dive_phase["angle_delta_deg"] == 10.3
    assert "Durchschnitt liegt im Zielbereich" in dive_phase["comment"]
    assert "Max Winkel 72.3 Grad" in dive_phase["comment"]
    assert "Winkel liegt ueber dem Technikmodell" not in dive_phase["comment"]

    display_rows = _build_phase_rows_with_reference(report=report, reference_report=None)
    display_dive = next(row for row in display_rows if row["name"] == "Dive-Aufbau")

    assert display_dive["angle_progression_display"] == "62.0 -> 72.3 Grad (+10.3)"

    score_rows = _build_scorecard_rows(report)
    dive_score = next(row for row in score_rows if row["name"] == "Dive-Aufbau")
    rating_line = next(line for line in dive_score["reason_lines"] if line.startswith("Bewertung:"))

    assert "Avg Winkel 63.7 Grad zum Zielband 60-70 Grad passt" in rating_line
    assert "Max Winkel 72.3 Grad aber kurz darueber liegt" in rating_line
    assert "Winkelverlauf: 62.0 -> 72.3 Grad (+10.3)." in dive_score["reason_lines"]
    assert sum("Max Winkel" in line for line in dive_score["reason_lines"]) == 1


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
    long_action = (
        "Im Segment +10 bis +15s frueher Druck aufbauen, aber den Winkel nicht erzwingen "
        "und die Linie ruhiger halten, damit der Zuwachs in Richtung stabiler Referenz geht. "
        "Erst Stabilitaet sichern und nicht sofort steiler werden, sondern die aktuelle Linie ruhig halten."
    )
    jump_brief["actions"] = [long_action]
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
    assert payload["jump_brief"]["actions"][0] == long_action


def test_ai_coaching_payload_simple_uses_expert_brief_for_focus_source():
    report = {
        "jump": {"jumper_name": "Test Jumper", "file_name": "ai.csv", "jump_context": "training"},
        "metrics": {"best_3s_vVert_kmh": 416.2, "best_3s_start_s": 23.0, "best_3s_end_s": 26.0},
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 36.0},
        "quality_flags": [],
    }
    expert_action = (
        "Bis +10s zuerst stabil im Bereich 72.3 bis 76.5 Grad bleiben. "
        "Erst wenn die Linie bis +15s ruhig bleibt, den Winkel in kleinen Schritten weiter aufbauen."
    )
    simple_action = "Aufbauphase: Zwischen +10s und +20s gleichmaessig weiter beschleunigen."

    payload = _build_ai_coaching_payload(
        report=report,
        review={"happened": [], "good": [], "not_good": [], "coaching_goals": []},
        jump_brief={
            "summary": "Die groessten Baustellen liegen bei Aufbau 10-20s.",
            "main_issues": ["Bei +10s ist der Tauchwinkel fuer dein stabiles Niveau zu steil."],
            "strengths": ["Exit ruhig."],
            "actions": [expert_action],
        },
        jump_brief_simple={
            "summary": "Arbeite am Aufbau.",
            "main_issues": ["Du wirst zu frueh zu steil."],
            "strengths": ["Der Start war ruhig."],
            "actions": [simple_action],
        },
        scorecard_rows=[],
        tip_follow_up={"available": False},
        jumper_summary={"performance_profile": {"available": False}},
        quality_issue_lines=[],
        view_mode="simple",
    )

    assert payload["view_mode"] == "simple"
    assert payload["jump_brief"]["actions"] == [expert_action]
    assert simple_action not in payload["jump_brief"]["actions"]


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
    assert "pattern" not in payload["primary_diagnosis"]
    assert payload["primary_diagnosis"]["evidence"]["angle_peak"] == 86.4


def test_ai_payload_includes_compact_technical_assessment():
    report = {
        "jump": {"jumper_name": "Test Jumper", "file_name": "ai.csv", "jump_context": "training"},
        "metrics": {"best_3s_vVert_kmh": 430.0, "best_3s_start_s": 20.0, "best_3s_end_s": 23.0},
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 28.0},
        "quality_flags": [],
    }
    review = {
        "happened": [],
        "good": [],
        "not_good": [],
        "coaching_goals": [],
        "technical_assessment": {
            "available": True,
            "summary_line": "Technikmodell: 3s-Fenster kritisch.",
            "best_window_quality": {
                "available": True,
                "label": "kritisch",
                "vvert_std_kmh": 18.2,
                "angle_std_deg": 2.4,
                "vhor_min_kmh": 23.1,
                "vvert_drop_after_kmh": 28.0,
            },
            "jerk_quality": {
                "available": True,
                "label": "unruhig",
                "jerk_rms_mps3": 5.3,
            },
            "phases": [
                {
                    "name": "Hauptbeschleunigung",
                    "start_s": 8.0,
                    "end_s": 15.0,
                    "angle_status": "too_steep",
                    "avg_angle_deg": 84.5,
                    "max_angle_deg": 87.2,
                    "vvert_gain_kmh": 21.0,
                    "min_vhor_kmh": 48.0,
                    "acc_mean_mps2": 0.5,
                    "steep_without_gain": True,
                    "oversteep_vhor_cost": False,
                }
            ],
            "issues": ["Das beste 3s-Fenster wirkt eher wie ein kurzer Peak."],
            "actions": [{"text": "Im Peak nicht weiter nachdruecken."}],
        },
    }

    payload = _build_ai_coaching_payload(
        report=report,
        review=review,
        jump_brief={"summary": "Kurz.", "main_issues": [], "strengths": [], "actions": []},
        jump_brief_simple={"summary": "Kurz.", "main_issues": [], "strengths": [], "actions": []},
        scorecard_rows=[],
        tip_follow_up={"available": False},
        jumper_summary={"performance_profile": {"available": False}},
        quality_issue_lines=[],
        view_mode="expert",
    )

    assert payload["technical_assessment"]["available"] is True
    assert payload["technical_assessment"]["best_window_quality"]["label"] == "kritisch"
    assert payload["technical_assessment"]["phase_issues"][0]["name"] == "Hauptbeschleunigung"
    assert "chart_data" not in str(payload["technical_assessment"])


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
                "rule_score_kmh": 438.0,
            "exit_score": 82,
            "dive_score": 79,
            "main_accel_score": 79,
            "build_score": 79,
            "hot_build_score": 50,
            "max_speed_score": 61,
            "hot_score": 50,
            "stability_score": 61,
            "angle_turns_20_25": 6.0,
        },
            {
                "best_3s_kmh": 438.0,
                "rule_score_kmh": 436.0,
            "exit_score": 81,
            "dive_score": 78,
            "main_accel_score": 78,
            "build_score": 78,
            "hot_build_score": 52,
            "max_speed_score": 60,
            "hot_score": 52,
            "stability_score": 60,
            "angle_turns_20_25": 7.0,
        },
            {
                "best_3s_kmh": 430.0,
                "rule_score_kmh": 428.0,
            "exit_score": 75,
            "dive_score": 79,
            "main_accel_score": 79,
            "build_score": 79,
            "hot_build_score": 66,
            "max_speed_score": 63,
            "hot_score": 66,
            "stability_score": 63,
            "angle_turns_20_25": 3.5,
        },
            {
                "best_3s_kmh": 428.0,
                "rule_score_kmh": 426.0,
            "exit_score": 74,
            "dive_score": 80,
            "main_accel_score": 80,
            "build_score": 80,
            "hot_build_score": 68,
            "max_speed_score": 64,
            "hot_score": 68,
            "stability_score": 64,
            "angle_turns_20_25": 3.0,
        },
    ]

    rows = _build_jumper_trend_rows(records)
    by_name = {row["name"]: row for row in rows}

    assert by_name["Regel-Score"]["status"] == "schlechter"
    assert by_name["Hot-Zone Aufbau"]["status"] == "besser"
    assert by_name["Korrekturen 20-25s"]["status"] == "besser"
    assert by_name["Regel-Score"]["earlier_better_text"].startswith("Regel-Score war früher besser")


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
        {
            "analysis_blocked": False,
            "stability_score": 64,
            "hot_score": 63,
            "build_score": 63,
            "vvert_10s": 242.0,
            "vvert_15s": 330.0,
            "vvert_20s": 410.0,
            "angle_10s": 67.0,
            "angle_15s": 78.0,
            "angle_20s": 85.0,
            "gain_10_20": 168.0,
            "gain_10_15": 88.0,
            "gain_15_20": 80.0,
            "vhor_min_20_25": 26.0,
            "angle_turns_20_25": 4.0,
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


def test_jumper_timing_reference_uses_stable_personal_anchors():
    records = [
        {
            "analysis_blocked": False,
            "best_3s_kmh": 430.0 + idx,
            "rule_score_kmh": 428.0 + idx,
            "best_3s_start_s": 20.0 + idx * 0.3,
            "best_3s_end_s": 23.0 + idx * 0.3,
            "angle_75_time_s": 8.0 + idx * 0.2,
            "angle_82_time_s": 13.0 + idx * 0.2,
            "angle_85_time_s": 17.0 + idx * 0.2,
            "vvert_peak_time_s": 24.0 + idx * 0.2,
            "decel_start_s": 25.0 + idx * 0.2,
            "stability_score": 74,
            "hot_score": 70,
            "build_score": 72,
            "vhor_min_20_25": 34.0,
            "angle_turns_20_25": 2.0,
            "build_coverage": 0.95,
            "hot_coverage": 0.95,
        }
        for idx in range(5)
    ]

    ref = _build_jumper_timing_reference(records)

    assert ref["available"] is True
    assert ref["maturity"] == "active"
    assert ref["usable_count"] == 5
    assert ref["basis_count"] == 5
    assert ref["confidence"] == "medium"
    assert ref["source"] == "stable_control"
    assert ref["best_reference_kmh"] == 432.0
    assert ref["top3_avg_kmh"] == 431.0
    assert ref["anchors"]["angle_82_time_s"]["median"] == 13.4
    assert any("82 Grad" in line for line in ref["summary_lines"])
    assert any("Decel/Recovery" in line for line in ref["summary_lines"])


def test_jumper_timing_reference_requires_enough_usable_jumps():
    records = [
        {
            "analysis_blocked": False,
            "best_3s_kmh": 440.0 + idx,
            "best_3s_start_s": 20.0 + idx * 0.2,
            "best_3s_end_s": 23.0 + idx * 0.2,
            "angle_82_time_s": 13.0 + idx * 0.2,
            "angle_85_time_s": 17.0 + idx * 0.2,
            "stability_score": 74,
            "hot_score": 70,
            "build_score": 72,
            "vhor_min_20_25": 34.0,
            "angle_turns_20_25": 2.0,
            "build_coverage": 0.95,
            "hot_coverage": 0.95,
        }
        for idx in range(4)
    ]

    ref = _build_jumper_timing_reference(records)

    assert ref["available"] is False
    assert ref["maturity"] == "off"
    assert ref["usable_count"] == 4
    assert "mindestens 5" in ref["reason"]


def test_jumper_timing_reference_stays_soft_when_level_or_timing_is_not_stable():
    low_top3_records = [
        {
            "analysis_blocked": False,
            "best_3s_kmh": speed,
            "best_3s_start_s": 20.0 + idx * 0.2,
            "best_3s_end_s": 23.0 + idx * 0.2,
            "angle_82_time_s": 13.0 + idx * 0.2,
            "angle_85_time_s": 17.0 + idx * 0.2,
            "stability_score": 74,
            "hot_score": 70,
            "build_score": 72,
            "vhor_min_20_25": 34.0,
            "angle_turns_20_25": 2.0,
            "build_coverage": 0.95,
            "hot_coverage": 0.95,
        }
        for idx, speed in enumerate([431.0, 405.0, 404.0, 403.0, 402.0])
    ]
    variable_timing_records = [
        {
            "analysis_blocked": False,
            "best_3s_kmh": 440.0 + idx,
            "best_3s_start_s": 20.0 + idx * 0.2,
            "best_3s_end_s": 23.0 + idx * 0.2,
            "angle_82_time_s": 11.0 + idx * 3.0,
            "angle_85_time_s": 17.0 + idx * 0.2,
            "stability_score": 74,
            "hot_score": 70,
            "build_score": 72,
            "vhor_min_20_25": 34.0,
            "angle_turns_20_25": 2.0,
            "build_coverage": 0.95,
            "hot_coverage": 0.95,
        }
        for idx in range(5)
    ]

    low_top3_ref = _build_jumper_timing_reference(low_top3_records)
    variable_ref = _build_jumper_timing_reference(variable_timing_records)

    assert low_top3_ref["available"] is False
    assert low_top3_ref["maturity"] == "soft"
    assert any("Top-3" in item for item in low_top3_ref["maturity_reasons"])
    assert variable_ref["available"] is False
    assert variable_ref["maturity"] == "soft"
    assert any("82-Grad-Timing streut" in item for item in variable_ref["maturity_reasons"])


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
    assert phases == [
        "Exit / Stabilisierung",
        "Dive-Aufbau",
        "Hauptbeschleunigung",
        "Hot-Zone Aufbau",
        "Max-Speed Fenster",
    ]


def test_tip_follow_status_uses_evidence_states_without_forced_partial():
    assert _tip_follow_status(score_delta=8, positive_hits=2, negative_hits=0)[0] == "verbessert"
    assert _tip_follow_status(score_delta=0, positive_hits=0, negative_hits=0)[0] == "unveraendert"
    assert _tip_follow_status(score_delta=-7, positive_hits=0, negative_hits=2)[0] == "verschlechtert"
    assert _tip_follow_status(score_delta=1, positive_hits=1, negative_hits=1)[0] == "uneindeutig"


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
    assert follow_up["entries"][0]["status_key"] == "verbessert"
    assert follow_up["entries"][0]["goal_text"] == "Im Aufbau nicht zu schnell maximal steil werden."
    assert "Winkel +10s" in follow_up["entries"][0]["detail"]
    assert "Winkel +15s" in follow_up["entries"][0]["detail"]


def test_coaching_snapshot_preserves_displayed_feedback_focus_for_follow_up():
    report = _minimal_goal_follow_report(
        jump_id="snapshot",
        file_name="snapshot.csv",
        angle_10=74.0,
        angle_15=80.0,
    )
    review = {
        "coaching_goals": [
            {
                "id": "review_goal",
                "phase": "Aufbau 10-20s",
                "text": "Zuwachs im Segment +10-20s erhoehen.",
                "target_metrics": [
                    {
                        "metric": "gain_10_20",
                        "label": "Zuwachs +10 bis +20s",
                        "direction": "increase",
                        "min_delta": 12.0,
                        "unit": " km/h",
                        "decimals": 1,
                    }
                ],
            }
        ]
    }
    feedback_context = {
        "available": True,
        "next_focus_hint": "Kompaktere Haltung nur kleiner testen und Linie ruhig halten.",
        "evidence": {"end_unstable": True},
    }

    snapshot = _build_coaching_snapshot(
        report=report,
        review=review,
        jump_brief={"actions": ["Regel-Fokus"]},
        jump_brief_simple={"actions": ["Einfacher Fokus"]},
        ai_coaching={
            "available": True,
            "next_jump_focus": "Kompaktere Haltung nur kleiner testen und Linie ruhig halten.",
        },
        feedback_context=feedback_context,
        view_mode="expert",
    )

    assert snapshot["available"] is True
    assert snapshot["ai_used"] is True
    assert snapshot["feedback_used"] is True
    assert snapshot["source"] == "ai+feedback"
    assert snapshot["goals"][0]["text"] == "Kompaktere Haltung nur kleiner testen und Linie ruhig halten."
    assert snapshot["goals"][0]["target_metrics"][1]["metric"] == "angle_turns_20_25"
    assert snapshot["goals"][0]["target_metrics"][1]["decimals"] == 0


def test_coaching_snapshot_matches_ai_focus_to_corresponding_review_goal_metrics():
    report = _minimal_goal_follow_report(
        jump_id="snapshot-match",
        file_name="snapshot-match.csv",
        angle_10=74.0,
        angle_15=80.0,
    )
    review = {
        "coaching_goals": [
            {
                "id": "build_gain",
                "phase": "Aufbau 10-20s",
                "text": "Zwischen +10s und +20s mehr Druck aufbauen.",
                "target_metrics": [
                    {
                        "metric": "gain_10_20",
                        "label": "Zuwachs +10 bis +20s",
                        "direction": "increase",
                        "min_delta": 12.0,
                        "unit": " km/h",
                        "decimals": 1,
                    }
                ],
            },
            {
                "id": "hot_efficiency",
                "phase": "Hot-Zone",
                "text": "Hot-Zone frueher klein korrigieren und die Linie ruhiger halten.",
                "target_metrics": [
                    {
                        "metric": "vhor_min_20_25",
                        "label": "vHor-Min 20-25s",
                        "direction": "increase",
                        "min_delta": 2.0,
                        "unit": " km/h",
                        "decimals": 1,
                    }
                ],
            },
        ]
    }

    snapshot = _build_coaching_snapshot(
        report=report,
        review=review,
        jump_brief={"actions": []},
        jump_brief_simple={"actions": []},
        ai_coaching={
            "available": True,
            "next_jump_focus": "Hot-Zone frueher klein korrigieren und die Linie ruhiger halten.",
        },
        feedback_context={"available": False},
        view_mode="expert",
    )

    assert snapshot["available"] is True
    assert snapshot["goals"][0]["text"] == "Hot-Zone frueher klein korrigieren und die Linie ruhiger halten."
    assert snapshot["goals"][0]["phase"] == "Hot-Zone"
    assert snapshot["goals"][0]["target_metrics"][0]["metric"] == "vhor_min_20_25"
    assert snapshot["goals"][0]["target_metrics"][0]["metric"] != "gain_10_20"


def test_tip_follow_up_prefers_saved_coaching_snapshot_goal():
    previous_report = _minimal_goal_follow_report(
        jump_id="prev",
        file_name="previous.csv",
        angle_10=75.0,
        angle_15=85.0,
        risk_score=35.0,
    )
    previous_report["coaching_snapshot"] = {
        "available": True,
        "source": "ai+feedback",
        "focus_text": "Kompaktere Haltung nur kleiner testen.",
        "goals": [
            {
                "id": "display_focus",
                "phase": "Stabilitaet / Kipp-Risiko",
                "text": "Kompaktere Haltung nur kleiner testen.",
                "target_metrics": [
                    {
                        "metric": "negative_risk_score",
                        "label": "Risiko-Score",
                        "direction": "decrease",
                        "min_delta": 5.0,
                        "unit": "",
                        "decimals": 1,
                    }
                ],
            }
        ],
    }
    current_report = _minimal_goal_follow_report(
        jump_id="current",
        file_name="current.csv",
        angle_10=80.0,
        angle_15=88.0,
        risk_score=20.0,
    )
    previous_goals = [
        {
            "id": "old_dynamic_goal",
            "phase": "Aufbau 10-20s",
            "text": "Im Aufbau flacher bleiben.",
            "target_metrics": [
                {
                    "metric": "angle_10",
                    "label": "Winkel +10s",
                    "direction": "decrease",
                    "min_delta": 1.0,
                    "unit": " Grad",
                    "decimals": 1,
                }
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
    assert follow_up["snapshot_used"] is True
    assert follow_up["snapshot_source"] == "ai+feedback"
    assert follow_up["entries"][0]["status_key"] == "verbessert"
    assert follow_up["entries"][0]["goal_text"] == "Kompaktere Haltung nur kleiner testen."
    assert "Risiko-Score" in follow_up["entries"][0]["detail"]
    assert "Im Aufbau flacher bleiben" not in follow_up["entries"][0]["goal_text"]


def test_tip_follow_up_falls_back_when_saved_snapshot_is_not_evaluable():
    previous_report = _minimal_goal_follow_report(
        jump_id="prev",
        file_name="previous.csv",
        angle_10=75.0,
        angle_15=85.0,
    )
    previous_report["coaching_snapshot"] = {
        "available": True,
        "source": "ai",
        "focus_text": "Nicht auswertbares altes Ziel.",
        "goals": [
            {
                "id": "broken_snapshot_goal",
                "phase": "Coaching-Fokus",
                "text": "Nicht auswertbares altes Ziel.",
                "target_metrics": [
                    {
                        "metric": "unsupported_metric",
                        "label": "Unbekannt",
                        "direction": "increase",
                        "min_delta": 1.0,
                    }
                ],
            }
        ],
    }
    current_report = _minimal_goal_follow_report(
        jump_id="current",
        file_name="current.csv",
        angle_10=72.5,
        angle_15=82.0,
    )
    previous_goals = [
        {
            "id": "fallback_goal",
            "phase": "Aufbau 10-20s",
            "text": "Im Aufbau flacher bleiben.",
            "target_metrics": [
                {
                    "metric": "angle_10",
                    "label": "Winkel +10s",
                    "direction": "decrease",
                    "min_delta": 1.0,
                    "unit": " Grad",
                    "decimals": 1,
                }
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
    assert follow_up["snapshot_used"] is False
    assert follow_up["entries"][0]["goal_text"] == "Im Aufbau flacher bleiben."
    assert "Winkel +10s" in follow_up["entries"][0]["detail"]


def test_tip_follow_up_compacts_duplicate_structured_goals_and_flags_quality():
    previous_report = _minimal_goal_follow_report(
        jump_id="prev",
        file_name="previous.csv",
        angle_10=75.0,
        angle_15=83.0,
    )
    current_report = _minimal_goal_follow_report(
        jump_id="current",
        file_name="current.csv",
        angle_10=73.5,
        angle_15=84.5,
    )
    current_report["quality_flags"] = ["TIME_GAPS"]
    previous_goals = [
        {
            "id": "angle_flatter",
            "phase": "Aufbau 10-20s",
            "text": "Im Aufbau flacher und ruhiger bleiben.",
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
        },
        {
            "id": "efficiency_20_25_low",
            "phase": "Hot-Zone",
            "text": "Hot-Zone ruhiger halten.",
            "target_metrics": [
                {
                    "metric": "vhor_min_20_25",
                    "label": "vHor-Min 20-25s",
                    "direction": "increase",
                    "min_delta": 2.0,
                    "unit": " km/h",
                    "decimals": 1,
                },
            ],
        },
        {
            "id": "vhor_price_20_25",
            "phase": "Hot-Zone",
            "text": "Ab +20s sauberer arbeiten.",
            "target_metrics": [
                {
                    "metric": "vhor_min_20_25",
                    "label": "vHor-Min 20-25s",
                    "direction": "increase",
                    "min_delta": 2.0,
                    "unit": " km/h",
                    "decimals": 1,
                },
            ],
        },
    ]

    follow_up = _build_tip_follow_up(
        current_report=current_report,
        previous_report=previous_report,
        marco_profile=_TEST_MARCO_PROFILE,
        previous_coaching_goals=previous_goals,
    )

    assert follow_up["available"] is True
    assert len(follow_up["entries"]) == 2
    assert follow_up["entries"][0]["status_key"] == "uneindeutig"
    assert follow_up["entries"][1]["phase"] == "Hot-Zone Aufbau"
    assert "ähnliche" in follow_up["entries"][1]["detail"]
    assert "Zeitlücken" in follow_up["quality_note"]


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


def test_jump_brief_simple_prefers_actionable_brief_issues_over_phase_context():
    jump_brief = {
        "summary": "Die groessten Baustellen liegen bei Hauptbeschleunigung und Hot-Zone.",
        "basis_lines": [],
        "main_issues": [
            "Dive-Aufbau: +3.0s bis +8.0s, Zielwinkel 60-70 Grad: zu steil.",
            "Dive-Aufbau: Istwerte: Avg Winkel 49.9 Grad, Avg vVert 165.8 km/h, Avg vHor 135.9 km/h.",
            "Phase +10 bis +15s: Zuwachs +58.8 km/h (dein stabiler Korridor: +67.9 bis +70.7).",
            "Die Hot-Zone ist ineffizient: viel Hoehenverlust bei zu wenig zusaetzlichem Speed.",
        ],
        "strengths": ["Das Max-Speed-Fenster war gut nutzbar."],
        "actions": ["Zwischen +10s und +20s mehr Druck aufbauen."],
    }

    simple = _build_jump_brief_simple(
        report={"notes": {}},
        jump_brief=jump_brief,
        scorecard_rows=[
            {
                "name": "Dive-Aufbau",
                "score": 56,
                "target_status": "zu steil",
                "target_angle_label": "60-70 Grad",
                "avg_angle_deg": 71.7,
            },
            {
                "name": "Hot-Zone Aufbau",
                "score": 50,
                "target_status": "zu flach",
                "target_angle_label": "82-86 Grad",
                "avg_angle_deg": 77.0,
            },
            {
                "name": "Max-Speed Fenster",
                "score": 75,
                "target_status": "im Zielbereich",
            },
        ],
    )

    assert simple["main_issues"] == [
        "Zwischen +10s und +15s kommt noch zu wenig zusaetzlicher Speed.",
        "In der Hot-Zone geht viel Hoehe verloren, aber es kommt nur wenig zusaetzlicher Speed dazu.",
    ]
    combined = " ".join(simple["main_issues"])
    assert "Zielwinkel 60" not in combined
    assert "bis ," not in combined
    assert "Istwerte" not in combined
    assert "Avg" not in combined
    assert "Das Max-Speed-Fenster war gut nutzbar." in simple["strengths"]
    assert any(item.startswith("Schnelle Phase:") for item in simple["actions"])


def test_jump_brief_simple_maps_hot_zone_build_action_to_fast_phase():
    simple = _build_jump_brief_simple(
        report={"notes": {}},
        jump_brief={
            "summary": "Hot-Zone kritisch.",
            "basis_lines": [],
            "main_issues": ["Die Hot-Zone ist ineffizient."],
            "strengths": [],
            "actions": [],
        },
        scorecard_rows=[
            {"name": "Hot-Zone Aufbau", "score": 50, "target_status": "zu flach"},
        ],
    )

    assert simple["actions"][0].startswith("Schnelle Phase:")
    assert not simple["actions"][0].startswith("Aufbauphase:")


def test_jump_brief_simple_translates_scorecard_angle_reason_without_avg_jargon():
    simple = _build_jump_brief_simple(
        report={"notes": {}},
        jump_brief={
            "summary": "Dive-Aufbau kritisch.",
            "basis_lines": [],
            "main_issues": [
                "Dive-Aufbau: Bewertung: zu flach, weil der Avg Winkel 49.9 Grad unter dem Ziel 60-70 Grad liegt."
            ],
            "strengths": [],
            "actions": [],
        },
        scorecard_rows=[
            {
                "name": "Dive-Aufbau",
                "score": 60,
                "target_status": "zu flach",
                "target_angle_label": "60-70 Grad",
                "avg_angle_deg": 49.9,
            },
        ],
    )

    assert simple["main_issues"][0] == "Dive-Aufbau: Der Winkel liegt im Durchschnitt unter dem Zielbereich."
    assert "Avg" not in simple["main_issues"][0]


def test_render_simple_glossary_wraps_known_terms_with_tooltip():
    html = _render_simple_glossary(
        "Startphase: Nach dem Absprung ruhig Druck halten und kleine Korrekturen machen."
    )
    assert 'class="glossary-term"' in html
    assert 'title="' in html
    assert "Druck halten" in html
    assert "kleine Korrekturen" in html
