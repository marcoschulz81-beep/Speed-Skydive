from __future__ import annotations

import json

import app.database as database
from app.database import init_db
from app.main import _build_performance_profile, _build_tip_effect_profile
from app.services.storage import (
    get_coaching_snapshot,
    get_jump_feedback,
    get_jump_report,
    get_jump_summary,
    replace_analysis_result,
    save_analysis_result,
    update_jump_context,
    upsert_coaching_snapshot,
    upsert_jump_feedback,
)


def _record(
    *,
    rule_score: float | None,
    training_score: float,
    jump_context: str = "training",
    flags: list[str] | None = None,
    hot_score: int = 74,
    stability_score: int = 73,
) -> dict[str, object]:
    return {
        "jump_context": jump_context,
        "rule_score_kmh": rule_score,
        "best_3s_kmh": training_score,
        "quality_flags": flags or [],
        "analysis_blocked": False,
        "t0_review_required": False,
        "exit_score": 82,
        "build_score": 80,
        "hot_score": hot_score,
        "stability_score": stability_score,
    }


def _minimal_result(*, jump_id: str, jumper_name: str = "Test Jumper", rule_score: float = 420.0) -> dict:
    return {
        "jump_record": {
            "jump_id": jump_id,
            "jumper_name": jumper_name,
            "file_name": f"{jump_id}.csv",
            "device_type": "FlySight",
            "raw_start_time_utc": "2026-01-01T10:00:00+00:00",
            "t0_utc": "2026-01-01T10:00:03+00:00",
            "exit_altitude_msl_m": 4000.0,
            "exit_altitude_agl_m": 3800.0,
            "ground_elevation_m": 200.0,
            "is_valid_altitude": 1,
            "sample_rate_hz": 5.0,
            "quality_score": 95.0,
            "quality_flags": json.dumps([]),
        },
        "metrics_record": {
            "jump_id": jump_id,
            "best_3s_start_s": 20.0,
            "best_3s_end_s": 23.0,
            "best_3s_vVert_mps": 116.667,
            "best_3s_vVert_kmh": 430.0,
            "best_3s_vHor_kmh": 31.0,
            "best_3s_angle_deg": 84.0,
            "training_3s_max_from_t0": 430.0,
            "rule_based_3s_score": rule_score,
            "rule_based_3s_score_mps": rule_score / 3.6,
            "performance_window_start_s": 4.0,
            "performance_window_end_s": 28.0,
            "validation_window_quality": 1.0,
            "hot_zone_start_s": 20.0,
            "hot_zone_end_s": 25.0,
            "negative_risk_score": 12.0,
            "notes": json.dumps({}),
            "fixpoints_json": json.dumps([]),
            "phases_json": json.dumps([]),
            "scorecard_json": json.dumps({}),
            "tips_json": json.dumps([]),
            "quality_flags": json.dumps([]),
        },
        "sample_records": [],
    }


def test_performance_profile_uses_rule_score_and_keeps_context_counts():
    records = [
        _record(rule_score=505.0, training_score=512.0, jump_context="competition"),
        _record(rule_score=490.0, training_score=498.0, jump_context="training"),
        _record(rule_score=450.0, training_score=470.0, jump_context="unknown"),
        _record(rule_score=530.0, training_score=540.0, flags=["SPEED_SPIKE"]),
    ]

    profile = _build_performance_profile(records)

    assert profile["available"] is True
    assert profile["score_source"] == "rule"
    assert profile["valid_jump_count"] == 3
    assert profile["excluded_count"] == 1
    assert profile["top_available_avg_kmh"] == 481.67
    assert profile["performance_band"] == "schnell"
    assert profile["confidence"] == "medium"
    assert profile["competition_count"] == 1
    assert profile["training_count"] == 2
    assert profile["unknown_count"] == 1


def test_tip_effect_profile_uses_performance_band_for_focus_boosts():
    records = [
        _record(rule_score=500.0, training_score=508.0, hot_score=82, stability_score=81),
        _record(rule_score=494.0, training_score=501.0, hot_score=80, stability_score=79),
        _record(rule_score=488.0, training_score=497.0, hot_score=71, stability_score=72),
        _record(rule_score=482.0, training_score=491.0, hot_score=70, stability_score=70),
    ]
    performance_profile = {
        "available": True,
        "performance_band": "schnell",
        "confidence": "stable",
        "summary": "Schnell: Top-4 regelnah 491.0 km/h, Profil stabil.",
    }

    profile = _build_tip_effect_profile(records, performance_profile=performance_profile)

    assert profile["available"] is True
    assert profile["phase_boosts"].get(3, 0) >= 1
    assert profile["phase_boosts"].get(3, 0) >= 1
    assert "Leistungsprofil" in profile["summary_line"]


def test_jump_context_storage_update_and_replace_preserves_context(tmp_path, monkeypatch):
    db_path = tmp_path / "speed_skydive.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    init_db()

    jump_id, duplicate = save_analysis_result(
        _minimal_result(jump_id="jump-a"),
        jump_context="competition",
        source_file_sha256="hash-a",
        source_file_path=str(tmp_path / "jump-a.csv"),
    )
    assert jump_id == "jump-a"
    assert duplicate is False
    assert get_jump_report("jump-a")["jump"]["jump_context"] == "competition"

    assert update_jump_context("jump-a", "training") is True
    assert get_jump_summary("jump-a")["jump_context"] == "training"

    replace_analysis_result(
        jump_id="jump-a",
        result=_minimal_result(jump_id="replacement", rule_score=440.0),
        source_file_sha256="hash-b",
        source_file_path=str(tmp_path / "jump-a-reprocessed.csv"),
    )

    report = get_jump_report("jump-a")
    assert report["jump"]["jump_context"] == "training"
    assert report["metrics"]["rule_based_3s_score"] == 440.0

    assert update_jump_context("jump-a", "invalid") is False
    assert get_jump_summary("jump-a")["jump_context"] == "training"


def test_jump_feedback_storage_update_remove_and_replace_preserves_feedback(tmp_path, monkeypatch):
    db_path = tmp_path / "speed_skydive.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    init_db()

    jump_id, duplicate = save_analysis_result(
        _minimal_result(jump_id="jump-feedback"),
        jump_context="training",
        source_file_sha256="hash-feedback",
        source_file_path=str(tmp_path / "jump-feedback.csv"),
    )
    assert jump_id == "jump-feedback"
    assert duplicate is False

    assert upsert_jump_feedback(
        "jump-feedback",
        "Ich wollte die Arme enger fuehren. Am Ende wurde es unruhig.",
    )
    report = get_jump_report("jump-feedback")
    assert report["feedback"]["available"] is True
    assert "Arme enger" in report["feedback"]["text"]
    assert get_jump_feedback("jump-feedback")["available"] is True

    replace_analysis_result(
        jump_id="jump-feedback",
        result=_minimal_result(jump_id="replacement", rule_score=445.0),
        source_file_sha256="hash-feedback-reprocessed",
        source_file_path=str(tmp_path / "jump-feedback-reprocessed.csv"),
    )
    report = get_jump_report("jump-feedback")
    assert report["metrics"]["rule_based_3s_score"] == 445.0
    assert report["feedback"]["available"] is True
    assert "Am Ende wurde es unruhig" in report["feedback"]["text"]

    assert upsert_jump_feedback("jump-feedback", "")
    assert get_jump_report("jump-feedback")["feedback"]["available"] is False


def test_coaching_snapshot_storage_and_replace_drops_stale_snapshot(tmp_path, monkeypatch):
    db_path = tmp_path / "speed_skydive.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    init_db()

    jump_id, duplicate = save_analysis_result(
        _minimal_result(jump_id="jump-snapshot"),
        jump_context="training",
        source_file_sha256="hash-snapshot",
        source_file_path=str(tmp_path / "jump-snapshot.csv"),
    )
    assert jump_id == "jump-snapshot"
    assert duplicate is False

    snapshot = {
        "available": True,
        "schema_version": 1,
        "jump_id": "jump-snapshot",
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

    assert upsert_coaching_snapshot("jump-snapshot", snapshot)
    stored = get_coaching_snapshot("jump-snapshot")
    assert stored["available"] is True
    assert stored["focus_text"] == "Kompaktere Haltung nur kleiner testen."
    assert stored["goals"][0]["target_metrics"][0]["metric"] == "negative_risk_score"

    report = get_jump_report("jump-snapshot")
    assert report["coaching_snapshot"]["available"] is True
    assert report["coaching_snapshot"]["source"] == "ai+feedback"

    replace_analysis_result(
        jump_id="jump-snapshot",
        result=_minimal_result(jump_id="replacement", rule_score=445.0),
        source_file_sha256="hash-snapshot-reprocessed",
        source_file_path=str(tmp_path / "jump-snapshot-reprocessed.csv"),
    )
    report = get_jump_report("jump-snapshot")
    assert report["metrics"]["rule_based_3s_score"] == 445.0
    assert report["coaching_snapshot"]["available"] is False
