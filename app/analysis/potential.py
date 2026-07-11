from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _FeatureRow:
    jump_id: str
    best_3s_kmh: float
    v10_kmh: float
    gain10_20_kmh: float
    angle_dev20_deg: float
    risk_score: float
    has_time_gaps: bool


def build_speed_potential_preview(
    *,
    current_report: dict[str, Any],
    historical_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    current_flags = set(current_report.get("quality_flags", []))
    if "EARLY_JUMP_END" in current_flags:
        return {
            "available": False,
            "reason": "Keine Prognose: Sprungdaten nicht belastbar (endet zu früh).",
        }

    rows: list[_FeatureRow] = []
    for report in historical_reports:
        row = _extract_feature_row(report)
        if row is not None:
            rows.append(row)

    if len(rows) < 5:
        return {
            "available": False,
            "reason": "Keine Prognose: zu wenige saubere Vergleichssprünge für diesen Springer.",
            "sample_size": len(rows),
        }

    current_row = _extract_feature_row(current_report)
    if current_row is None:
        return {
            "available": False,
            "reason": "Keine Prognose: notwendige Kennwerte (+10s/+20s) fehlen im aktuellen Sprung.",
            "sample_size": len(rows),
        }

    # Build linear baseline model on same-jumper history.
    X = np.array(
        [
            [r.v10_kmh, r.gain10_20_kmh, r.angle_dev20_deg, r.risk_score]
            for r in rows
        ],
        dtype=float,
    )
    y = np.array([r.best_3s_kmh for r in rows], dtype=float)
    Xs = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    fitted = Xs @ beta
    rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))

    base_features = np.array(
        [current_row.v10_kmh, current_row.gain10_20_kmh, current_row.angle_dev20_deg, current_row.risk_score],
        dtype=float,
    )
    improved_features = _improved_feature_target(base_features=base_features, rows=rows)

    pred_base = float(np.dot(np.append([1.0], base_features), beta))
    pred_improved = float(np.dot(np.append([1.0], improved_features), beta))
    expected_gain = max(0.0, pred_improved - pred_base)

    baseline_actual = float(current_row.best_3s_kmh)
    potential = baseline_actual + expected_gain
    spread = rmse * 0.7
    low = max(baseline_actual, potential - spread)
    high = max(low, potential + spread)

    contributions = _feature_contributions(
        beta=beta,
        base_features=base_features,
        improved_features=improved_features,
    )

    positive_contrib = [item for item in contributions if item["delta_kmh"] > 0.2]
    top_levers = [item["label"] for item in positive_contrib[:3]]

    confidence = _confidence_label(
        n=len(rows),
        rmse_kmh=rmse,
        has_time_gaps_ratio=(sum(1 for r in rows if r.has_time_gaps) / len(rows)),
    )
    return {
        "available": True,
        "sample_size": len(rows),
        "confidence": confidence,
        "baseline_kmh": round(baseline_actual, 2),
        "potential_kmh": round(potential, 2),
        "potential_low_kmh": round(low, 2),
        "potential_high_kmh": round(high, 2),
        "expected_gain_kmh": round(expected_gain, 2),
        "rmse_kmh": round(rmse, 2),
        "top_levers": top_levers,
        "lever_contributions": positive_contrib[:4],
    }


def _extract_feature_row(report: dict[str, Any]) -> _FeatureRow | None:
    flags = set(report.get("quality_flags", []))
    if "EARLY_JUMP_END" in flags or "SPEED_SPIKE" in flags:
        return None

    jump = report.get("jump", {})
    metrics = report.get("metrics", {})
    fixpoints = report.get("fixpoints", [])

    fp10 = _fixpoint_at(fixpoints, 10.0)
    fp20 = _fixpoint_at(fixpoints, 20.0)
    if fp10 is None or fp20 is None:
        return None

    v10 = _num(fp10.get("vVert_kmh"))
    v20 = _num(fp20.get("vVert_kmh"))
    a20 = _num(fp20.get("angle_deg"))
    phase_gain = _phase_build_gain(report)
    phase_angle_dev = _phase_late_angle_deviation(report)
    target = _num(metrics.get("best_3s_vVert_kmh"))
    risk = _num(metrics.get("negative_risk_score"))
    jump_id = str(jump.get("jump_id", ""))
    if v10 is None or v20 is None or a20 is None or target is None or risk is None or not jump_id:
        return None

    gain_feature = float(phase_gain) if phase_gain is not None else float(v20 - v10)
    angle_dev_feature = float(phase_angle_dev) if phase_angle_dev is not None else float(abs(a20 - 83.0))

    return _FeatureRow(
        jump_id=jump_id,
        best_3s_kmh=float(target),
        v10_kmh=float(v10),
        gain10_20_kmh=gain_feature,
        angle_dev20_deg=angle_dev_feature,
        risk_score=float(risk),
        has_time_gaps=("TIME_GAPS" in flags),
    )


def _improved_feature_target(*, base_features: np.ndarray, rows: list[_FeatureRow]) -> np.ndarray:
    v10_values = np.array([r.v10_kmh for r in rows], dtype=float)
    gain_values = np.array([r.gain10_20_kmh for r in rows], dtype=float)
    angle_dev_values = np.array([r.angle_dev20_deg for r in rows], dtype=float)
    risk_values = np.array([r.risk_score for r in rows], dtype=float)

    target_v10 = float(np.quantile(v10_values, 0.75))
    target_gain = float(np.quantile(gain_values, 0.75))
    target_angle_dev = float(np.quantile(angle_dev_values, 0.25))
    target_risk = float(np.quantile(risk_values, 0.25))

    v10_new = min(base_features[0] + 90.0, max(base_features[0], target_v10))
    gain_new = min(base_features[1] + 120.0, max(base_features[1], target_gain))
    angle_dev_new = max(0.0, min(base_features[2], target_angle_dev))
    risk_new = max(0.0, min(base_features[3], target_risk))
    return np.array([v10_new, gain_new, angle_dev_new, risk_new], dtype=float)


def _feature_contributions(
    *,
    beta: np.ndarray,
    base_features: np.ndarray,
    improved_features: np.ndarray,
) -> list[dict[str, Any]]:
    labels = ["Exit-Speed (+10s)", "Speed-Aufbau Phasen", "Winkelpraezision Hot-Zone/Max-Speed", "Stabilitaet (Risiko)"]
    out: list[dict[str, Any]] = []
    for idx, label in enumerate(labels, start=1):
        delta_feature = float(improved_features[idx - 1] - base_features[idx - 1])
        delta_kmh = float(beta[idx] * delta_feature)
        out.append(
            {
                "label": label,
                "delta_kmh": round(delta_kmh, 2),
            }
        )
    out.sort(key=lambda item: float(item["delta_kmh"]), reverse=True)
    return out


def _confidence_label(*, n: int, rmse_kmh: float, has_time_gaps_ratio: float) -> str:
    if n >= 12 and rmse_kmh <= 4.5 and has_time_gaps_ratio < 0.35:
        return "hoch"
    if n >= 8 and rmse_kmh <= 7.0:
        return "mittel"
    return "niedrig"


def _fixpoint_at(fixpoints: list[dict[str, Any]], t_rel_s: float) -> dict[str, Any] | None:
    return next((item for item in fixpoints if abs(float(item.get("t_rel_s", -1)) - t_rel_s) < 1e-6), None)


def _phase_build_gain(report: dict[str, Any]) -> float | None:
    phases = report.get("phases") if isinstance(report.get("phases"), list) else []
    gains: list[float] = []
    for name in ["Dive-Aufbau", "Hauptbeschleunigung", "Hot-Zone Aufbau"]:
        row = _phase_by_name(phases, name)
        if row is None:
            continue
        direct_gain = _num(row.get("vVert_gain_kmh", row.get("vvert_gain_kmh")))
        if direct_gain is not None:
            gains.append(float(direct_gain))
            continue
        start_v = _num(row.get("start_vVert_kmh"))
        end_v = _num(row.get("end_vVert_kmh"))
        if start_v is not None and end_v is not None:
            gains.append(float(end_v - start_v))
            continue
        avg_v = _num(row.get("avg_vVert_kmh"))
        if avg_v is not None:
            gains.append(float(avg_v))
    if not gains:
        return None
    if len(gains) >= 2 and all(value > 120.0 for value in gains):
        return max(0.0, max(gains) - min(gains))
    return float(sum(gains))


def _phase_late_angle_deviation(report: dict[str, Any]) -> float | None:
    phases = report.get("phases") if isinstance(report.get("phases"), list) else []
    deviations: list[float] = []
    for name, target in [("Hot-Zone Aufbau", 84.0), ("Max-Speed Fenster", 85.0)]:
        row = _phase_by_name(phases, name)
        if row is None:
            continue
        angle = _num(row.get("avg_angle_deg"))
        if angle is not None:
            deviations.append(abs(float(angle) - target))
    if not deviations:
        return None
    return float(sum(deviations) / len(deviations))


def _phase_by_name(phases: list[Any], name: str) -> dict[str, Any] | None:
    for row in phases:
        if isinstance(row, dict) and str(row.get("name") or "") == name:
            return row
    return None


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
