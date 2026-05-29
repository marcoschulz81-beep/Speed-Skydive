from __future__ import annotations

from typing import Any


def build_jump_comparison(
    *,
    left_report: dict[str, Any],
    right_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare two jump reports. Delta is always right - left.
    """
    left_jump = left_report["jump"]
    right_jump = right_report["jump"]
    left_metrics = left_report["metrics"]
    right_metrics = right_report["metrics"]

    summary = [
        _summary_row(
            label="3s Max (Training)",
            key="best_3s_vVert_kmh",
            unit="km/h",
            better_when="higher",
            left=left_metrics,
            right=right_metrics,
        ),
        _summary_row(
            label="Rule Score",
            key="rule_based_3s_score",
            unit="km/h",
            better_when="higher",
            left=left_metrics,
            right=right_metrics,
        ),
        _summary_row(
            label="Negativ-Risiko",
            key="negative_risk_score",
            unit="score",
            better_when="lower",
            left=left_metrics,
            right=right_metrics,
        ),
        _summary_row(
            label="Datenqualitaet",
            key="quality_score",
            unit="score",
            better_when="higher",
            left=left_jump,
            right=right_jump,
        ),
    ]

    left_fix = {float(item["t_rel_s"]): item for item in left_report["fixpoints"]}
    right_fix = {float(item["t_rel_s"]): item for item in right_report["fixpoints"]}
    all_fix_times = sorted(set(left_fix.keys()) | set(right_fix.keys()))

    fixpoint_rows: list[dict[str, Any]] = []
    for t_sec in all_fix_times:
        l = left_fix.get(t_sec, {})
        r = right_fix.get(t_sec, {})
        fixpoint_rows.append(
            {
                "t_rel_s": t_sec,
                "left_vVert_kmh": _num(l.get("vVert_kmh")),
                "right_vVert_kmh": _num(r.get("vVert_kmh")),
                "delta_vVert_kmh": _delta(l.get("vVert_kmh"), r.get("vVert_kmh")),
                "left_vHor_kmh": _num(l.get("vHor_kmh")),
                "right_vHor_kmh": _num(r.get("vHor_kmh")),
                "delta_vHor_kmh": _delta(l.get("vHor_kmh"), r.get("vHor_kmh")),
                "left_angle_deg": _num(l.get("angle_deg")),
                "right_angle_deg": _num(r.get("angle_deg")),
                "delta_angle_deg": _delta(l.get("angle_deg"), r.get("angle_deg")),
            }
        )

    insights = _build_insights(summary=summary, fixpoint_rows=fixpoint_rows)
    comparison_chart = {
        "left": {
            "label": f"{left_jump['file_name']} ({left_jump['t0_utc']})",
            "time_s": left_report["chart_data"]["time_s"],
            "vVert_kmh": left_report["chart_data"]["vVert_kmh"],
            "vHor_kmh": left_report["chart_data"]["vHor_kmh"],
            "angle_deg": left_report["chart_data"]["angle_deg"],
        },
        "right": {
            "label": f"{right_jump['file_name']} ({right_jump['t0_utc']})",
            "time_s": right_report["chart_data"]["time_s"],
            "vVert_kmh": right_report["chart_data"]["vVert_kmh"],
            "vHor_kmh": right_report["chart_data"]["vHor_kmh"],
            "angle_deg": right_report["chart_data"]["angle_deg"],
        },
    }

    return {
        "left": {"jump_id": left_jump["jump_id"], "file_name": left_jump["file_name"], "t0_utc": left_jump["t0_utc"]},
        "right": {
            "jump_id": right_jump["jump_id"],
            "file_name": right_jump["file_name"],
            "t0_utc": right_jump["t0_utc"],
        },
        "summary": summary,
        "fixpoint_rows": fixpoint_rows,
        "insights": insights,
        "charts": comparison_chart,
    }


def _summary_row(
    *,
    label: str,
    key: str,
    unit: str,
    better_when: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_v = _num(left.get(key))
    right_v = _num(right.get(key))
    delta = _delta(left_v, right_v)
    trend = "neutral"
    if delta is not None:
        if abs(delta) < 1e-9:
            trend = "neutral"
        elif better_when == "higher":
            trend = "improved" if delta > 0 else "worse"
        else:
            trend = "improved" if delta < 0 else "worse"
    return {
        "label": label,
        "left": left_v,
        "right": right_v,
        "delta": delta,
        "unit": unit,
        "trend": trend,
    }


def _build_insights(summary: list[dict[str, Any]], fixpoint_rows: list[dict[str, Any]]) -> list[str]:
    insights: list[str] = []

    score_row = next((row for row in summary if row["label"] == "3s Max (Training)"), None)
    if score_row and score_row["delta"] is not None:
        if score_row["delta"] > 0:
            insights.append(f"Rechter Sprung ist beim 3s-Max um {score_row['delta']:.2f} km/h schneller.")
        elif score_row["delta"] < 0:
            insights.append(f"Rechter Sprung ist beim 3s-Max um {abs(score_row['delta']):.2f} km/h langsamer.")
        else:
            insights.append("3s-Max ist in beiden Spruengen gleich.")

    risk_row = next((row for row in summary if row["label"] == "Negativ-Risiko"), None)
    if risk_row and risk_row["delta"] is not None:
        if risk_row["delta"] < 0:
            insights.append("Negativ/Kipp-Risiko ist im rechten Sprung geringer.")
        elif risk_row["delta"] > 0:
            insights.append("Negativ/Kipp-Risiko ist im rechten Sprung hoeher.")

    fp20 = next((row for row in fixpoint_rows if abs(row["t_rel_s"] - 20.0) < 1e-6), None)
    if fp20 and fp20["delta_vVert_kmh"] is not None:
        if fp20["delta_vVert_kmh"] > 0:
            insights.append(f"Bei +20s liegt vVert rechts um {fp20['delta_vVert_kmh']:.2f} km/h hoeher.")
        elif fp20["delta_vVert_kmh"] < 0:
            insights.append(f"Bei +20s liegt vVert rechts um {abs(fp20['delta_vVert_kmh']):.2f} km/h niedriger.")

    if not insights:
        insights.append("Keine belastbaren Unterschiede gefunden (fehlende oder identische Werte).")
    return insights


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: Any, right: Any) -> float | None:
    l = _num(left)
    r = _num(right)
    if l is None or r is None:
        return None
    return round(r - l, 3)

