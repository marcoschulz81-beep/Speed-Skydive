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

    assert by_label["3s Max (Training)"]["delta"] == 12.0
    assert by_label["3s Max (Training)"]["trend"] == "improved"
    assert by_label["Negativ-Risiko"]["delta"] == -14.0
    assert by_label["Negativ-Risiko"]["trend"] == "improved"
    assert len(cmp_data["fixpoint_rows"]) >= 2
    assert len(cmp_data["insights"]) >= 1

