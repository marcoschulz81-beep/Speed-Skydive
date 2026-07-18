from __future__ import annotations

from app.analysis.potential import build_speed_potential_preview


def _report(
    *,
    jump_id: str,
    three_s: float,
    v10: float,
    v20: float,
    angle20: float,
    risk: float,
    flags: list[str] | None = None,
) -> dict:
    flags = flags or []
    return {
        "jump": {
            "jump_id": jump_id,
        },
        "metrics": {
            "best_3s_vVert_kmh": three_s,
            "rule_based_3s_score": three_s - 3.0,
            "rule_score_status": "valid",
            "negative_risk_score": risk,
        },
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": v10, "angle_deg": 78.0},
            {"t_rel_s": 20.0, "vVert_kmh": v20, "angle_deg": angle20},
        ],
        "quality_flags": flags,
    }


def test_speed_potential_available_with_enough_history():
    history = [
        _report(
            jump_id=f"h-{idx}",
            three_s=405.0 + idx * 5.5,
            v10=245.0 + idx * 7.0,
            v20=345.0 + idx * 8.5,
            angle20=84.5 - idx * 0.18,
            risk=38.0 - idx * 1.1,
        )
        for idx in range(10)
    ]
    current = _report(jump_id="current", three_s=410.0, v10=250.0, v20=352.0, angle20=84.2, risk=36.0)

    out = build_speed_potential_preview(current_report=current, historical_reports=history)

    assert out["available"] is True
    assert out["sample_size"] == 10
    assert out["minimum_sample_size"] == 10
    assert out["validation_method"] == "leave-one-out"
    assert out["confidence"] == "niedrig"
    assert out["expected_gain_kmh"] >= 0
    assert out["potential_kmh"] >= out["baseline_kmh"]


def test_speed_potential_excludes_current_jump_before_minimum_check():
    current = _report(jump_id="current", three_s=420.0, v10=270.0, v20=380.0, angle20=83.0, risk=30.0)
    history = [current] + [
        _report(
            jump_id=f"h-{idx}",
            three_s=410.0 + idx * 4.0,
            v10=250.0 + idx * 5.0,
            v20=355.0 + idx * 6.0,
            angle20=84.0 - idx * 0.1,
            risk=35.0 - idx,
        )
        for idx in range(9)
    ]

    out = build_speed_potential_preview(current_report=current, historical_reports=history)

    assert out["available"] is False
    assert out["sample_size"] == 9
    assert out["minimum_sample_size"] == 10


def test_speed_potential_unavailable_for_blocked_jump():
    history = [
        _report(jump_id="a", three_s=410.0, v10=255.0, v20=355.0, angle20=84.0, risk=34.0),
        _report(jump_id="b", three_s=412.0, v10=258.0, v20=360.0, angle20=83.6, risk=33.0),
        _report(jump_id="c", three_s=416.0, v10=266.0, v20=372.0, angle20=83.2, risk=31.0),
        _report(jump_id="d", three_s=421.0, v10=274.0, v20=381.0, angle20=82.9, risk=29.0),
        _report(jump_id="e", three_s=427.0, v10=286.0, v20=393.0, angle20=82.6, risk=27.0),
    ]
    current = _report(
        jump_id="x",
        three_s=440.0,
        v10=340.0,
        v20=70.0,
        angle20=12.0,
        risk=80.0,
        flags=["EARLY_JUMP_END"],
    )

    out = build_speed_potential_preview(current_report=current, historical_reports=history)

    assert out["available"] is False
    assert "endet zu früh" in out["reason"]
