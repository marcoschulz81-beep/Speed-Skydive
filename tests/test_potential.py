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
        _report(jump_id="a", three_s=408.0, v10=250.0, v20=350.0, angle20=84.0, risk=35.0),
        _report(jump_id="b", three_s=414.0, v10=260.0, v20=365.0, angle20=83.5, risk=32.0),
        _report(jump_id="c", three_s=420.0, v10=270.0, v20=378.0, angle20=83.0, risk=30.0),
        _report(jump_id="d", three_s=426.0, v10=285.0, v20=390.0, angle20=82.8, risk=28.0),
        _report(jump_id="e", three_s=432.0, v10=295.0, v20=402.0, angle20=82.5, risk=26.0),
        _report(jump_id="f", three_s=438.0, v10=305.0, v20=412.0, angle20=82.2, risk=24.0),
    ]
    current = _report(jump_id="a", three_s=408.0, v10=250.0, v20=350.0, angle20=84.0, risk=35.0)

    out = build_speed_potential_preview(current_report=current, historical_reports=history)

    assert out["available"] is True
    assert out["sample_size"] >= 5
    assert out["expected_gain_kmh"] >= 0
    assert out["potential_kmh"] >= out["baseline_kmh"]


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
    assert "endet zu frueh" in out["reason"]
