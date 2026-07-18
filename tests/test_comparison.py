from __future__ import annotations

from app.analysis.comparison import build_jump_comparison


def _report(
    *,
    jump_id: str,
    file_name: str,
    t0_utc: str,
    three_s: float,
    rule_score: float,
    risk: float,
    quality: float,
    rule_status: str = "valid",
) -> dict:
    return {
        "jump": {
            "jump_id": jump_id,
            "file_name": file_name,
            "t0_utc": t0_utc,
            "quality_score": quality,
        },
        "metrics": {
            "best_3s_vVert_kmh": three_s,
            "rule_based_3s_score": rule_score,
            "rule_score_status": rule_status,
            "negative_risk_score": risk,
        },
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 250.0, "vHor_kmh": 80.0, "angle_deg": 75.0},
            {"t_rel_s": 20.0, "vVert_kmh": three_s - 15.0, "vHor_kmh": 40.0, "angle_deg": 84.0},
        ],
        "chart_data": {
            "time_s": [0.0, 10.0, 20.0, 30.0],
            "vVert_kmh": [100.0, 250.0, three_s - 15.0, 160.0],
            "vHor_kmh": [120.0, 80.0, 40.0, 20.0],
            "angle_deg": [55.0, 75.0, 84.0, 82.0],
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 30.0,
        },
    }


def test_build_jump_comparison_computes_deltas():
    left = _report(
        jump_id="A",
        file_name="left.csv",
        t0_utc="2024-01-01T10:00:00Z",
        three_s=380.0,
        rule_score=375.0,
        risk=55.0,
        quality=72.0,
    )
    right = _report(
        jump_id="B",
        file_name="right.csv",
        t0_utc="2024-01-02T10:00:00Z",
        three_s=392.0,
        rule_score=389.0,
        risk=41.0,
        quality=81.0,
    )

    cmp_data = build_jump_comparison(left_report=left, right_report=right)
    by_label = {row["label"]: row for row in cmp_data["summary"]}

    assert cmp_data["reference"]["jump_id"] == "B"
    assert cmp_data["comparison"]["jump_id"] == "A"
    assert by_label["3s Max (Training)"]["delta"] == -12.0
    assert by_label["3s Max (Training)"]["trend"] == "referenz besser"
    assert by_label["Negativ-Risiko"]["delta"] == 14.0
    assert by_label["Negativ-Risiko"]["trend"] == "referenz besser"
    assert len(cmp_data["fixpoint_rows"]) >= 2
    assert len(cmp_data["insights"]) >= 1
    assert "brief" in cmp_data
    assert "summary" in cmp_data["brief"]
    assert len(cmp_data["brief"]["main_issues"]) >= 1
    assert len(cmp_data["brief"]["actions"]) >= 1


def test_build_jump_comparison_limits_chart_to_curve_window():
    left = _report(
        jump_id="A",
        file_name="left.csv",
        t0_utc="2024-01-01T10:00:00Z",
        three_s=380.0,
        rule_score=375.0,
        risk=55.0,
        quality=72.0,
    )
    right = _report(
        jump_id="B",
        file_name="right.csv",
        t0_utc="2024-01-02T10:00:00Z",
        three_s=392.0,
        rule_score=389.0,
        risk=41.0,
        quality=81.0,
    )

    # Add out-of-window tail samples that should not appear in comparison charts.
    left["chart_data"]["time_s"].extend([120.0, 140.0])
    left["chart_data"]["vVert_kmh"].extend([20.0, 10.0])
    left["chart_data"]["vHor_kmh"].extend([5.0, 3.0])
    left["chart_data"]["angle_deg"].extend([12.0, 8.0])

    right["chart_data"]["time_s"].extend([150.0, 170.0])
    right["chart_data"]["vVert_kmh"].extend([18.0, 9.0])
    right["chart_data"]["vHor_kmh"].extend([4.0, 2.0])
    right["chart_data"]["angle_deg"].extend([10.0, 7.0])

    cmp_data = build_jump_comparison(left_report=left, right_report=right)
    charts = cmp_data["charts"]

    assert max(charts["left"]["time_s"]) <= 30.0
    assert max(charts["right"]["time_s"]) <= 30.0
    assert charts["x_axis_end_s"] == 30.0


def test_valid_rule_score_outranks_invalid_higher_training_peak() -> None:
    valid = _report(
        jump_id="valid",
        file_name="valid.csv",
        t0_utc="2024-01-01T10:00:00Z",
        three_s=390.0,
        rule_score=385.0,
        risk=20.0,
        quality=90.0,
        rule_status="valid",
    )
    invalid = _report(
        jump_id="invalid",
        file_name="invalid.csv",
        t0_utc="2024-01-02T10:00:00Z",
        three_s=520.0,
        rule_score=515.0,
        risk=50.0,
        quality=50.0,
        rule_status="invalid",
    )

    comparison = build_jump_comparison(left_report=valid, right_report=invalid)

    assert comparison["reference"]["jump_id"] == "valid"
    assert "einziger gültiger Regel-Score" in comparison["brief"]["basis_lines"][0]
