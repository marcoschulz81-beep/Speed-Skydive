from __future__ import annotations

from app.analysis.feedback import build_feedback_coaching_context, build_feedback_training_profile


def _report_with_feedback(text: str) -> dict:
    return {
        "feedback": {"available": True, "text": text},
        "metrics": {
            "best_3s_vVert_kmh": 438.0,
            "performance_window_end_s": 26.0,
        },
        "notes": {"curve_window_end_s": 26.0},
        "fixpoints": [
            {"t_rel_s": 10.0, "angle_deg": 74.0},
            {"t_rel_s": 15.0, "angle_deg": 80.0},
            {"t_rel_s": 20.0, "angle_deg": 86.0},
        ],
        "scorecard": {
            "phase_10_20": "optimal",
            "hot_zone": "kritisch",
            "kipp_risiko": "hoch",
        },
        "chart_data": {},
    }


def test_feedback_context_matches_subjective_end_unrest_to_objective_signals():
    report = _report_with_feedback(
        "Ich habe versucht die Arme naeher am Koerper zu fuehren und laenger zu ziehen. "
        "Am Ende hat es gewackelt und es hat mich zerrissen."
    )
    review = {
        "primary_diagnosis": {"available": True, "pattern": "late_hard_steepening_vhor_collapse"},
        "technical_assessment": {
            "available": True,
            "best_window_quality": {"available": True, "label": "unruhig"},
            "jerk_quality": {"available": True, "label": "kritisch"},
        },
    }

    context = build_feedback_coaching_context(report, review=review)

    assert context["available"] is True
    assert any(item["key"] == "arms_closer" for item in context["intents"])
    assert any(item["key"] == "longer_hold" for item in context["intents"])
    assert any(item["key"] == "end_instability" for item in context["felt_issues"])
    assert context["evidence"]["end_unstable"] is True
    assert "passt zu den Messdaten" in context["coaching_hint"]
    assert "Kompaktere Haltung" in context["next_focus_hint"]
    assert context["confidence"] in {"medium", "high"}


def test_feedback_context_normalizes_review_phase_statuses_before_evidence():
    report = _report_with_feedback("Ich wollte spaeter steil werden, aber es fuehlte sich zu flach an.")
    review = {
        "technical_assessment": {
            "available": True,
            "phases": [
                {"name": "Hauptbeschleunigung", "angle_status": "in_band"},
                {"name": "Hot-Zone Aufbau", "angle_status": "too_flat"},
            ],
        },
    }

    context = build_feedback_coaching_context(report, review=review)

    assert context["evidence"]["technical_phase_statuses"]["Hauptbeschleunigung"] == "im Zielbereich"
    assert context["evidence"]["technical_phase_statuses"]["Hot-Zone Aufbau"] == "zu flach"
    assert context["evidence"]["build_too_flat"] is True
    assert "Speed-Aufbau bleibt dadurch noch zu schwach" in context["coaching_hint"]


def test_feedback_training_profile_detects_repeated_compact_unrest_context():
    records = [
        {
            "feedback_context": build_feedback_coaching_context(
                _report_with_feedback("Arme enger getestet, am Ende unruhig."),
                review={"technical_assessment": {"jerk_quality": {"available": True, "label": "kritisch"}}},
            )
        },
        {
            "feedback_context": build_feedback_coaching_context(
                _report_with_feedback("Haende naeher an die Beine, spaeter hat es gewackelt."),
                review={"technical_assessment": {"best_window_quality": {"available": True, "label": "unruhig"}}},
            )
        },
    ]

    profile = build_feedback_training_profile(records)

    assert profile["available"] is True
    assert profile["feedback_count"] == 2
    assert any("Kompaktere Koerperlinie" in line for line in profile["lines"])
