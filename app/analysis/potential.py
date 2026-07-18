from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.analysis.evaluation import REFERENCE_HARD_FLAGS, RULE_SCORE_VALID

MIN_CLEAN_HISTORY = 10
RIDGE_ALPHA = 1.0


@dataclass
class _FeatureRow:
    jump_id: str
    rule_score_kmh: float
    v10_kmh: float
    gain10_20_kmh: float
    angle_dev20_deg: float
    risk_score: float


@dataclass
class _RidgeModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    coefficients: np.ndarray


def build_speed_potential_preview(
    *,
    current_report: dict[str, Any],
    historical_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    current_flags = {str(item) for item in current_report.get("quality_flags", [])}
    if "EARLY_JUMP_END" in current_flags:
        return {
            "available": False,
            "reason": "Keine Prognose: Sprungdaten nicht belastbar (endet zu früh).",
        }

    current_row = _extract_feature_row(current_report)
    if current_row is None:
        return {
            "available": False,
            "reason": "Keine Prognose: aktueller Regel-Score oder notwendige Kennwerte sind nicht gültig.",
            "sample_size": 0,
        }

    unique_rows: dict[str, _FeatureRow] = {}
    for report in historical_reports:
        row = _extract_feature_row(report)
        if row is None or row.jump_id == current_row.jump_id:
            continue
        unique_rows[row.jump_id] = row
    rows = list(unique_rows.values())

    if len(rows) < MIN_CLEAN_HISTORY:
        return {
            "available": False,
            "reason": (
                "Keine Prognose: mindestens zehn saubere historische Sprünge "
                "ohne den aktuellen Sprung sind erforderlich."
            ),
            "sample_size": len(rows),
            "minimum_sample_size": MIN_CLEAN_HISTORY,
        }

    X = np.array(
        [[r.v10_kmh, r.gain10_20_kmh, r.angle_dev20_deg, r.risk_score] for r in rows],
        dtype=float,
    )
    y = np.array([r.rule_score_kmh for r in rows], dtype=float)
    loo_predictions: np.ndarray = np.empty(len(rows), dtype=float)
    for idx in range(len(rows)):
        keep = np.arange(len(rows)) != idx
        fold_model = _fit_ridge(X[keep], y[keep])
        loo_predictions[idx] = _predict_ridge(fold_model, X[idx])

    loo_errors = loo_predictions - y
    loo_rmse = float(np.sqrt(np.mean(loo_errors**2)))
    loo_mae = float(np.mean(np.abs(loo_errors)))
    allowed_rmse = max(15.0, float(np.std(y)) * 2.0)
    if not np.isfinite(loo_rmse) or loo_rmse > allowed_rmse:
        return {
            "available": False,
            "reason": "Keine Prognose: historische Sprünge liefern keine belastbare Out-of-sample-Güte.",
            "sample_size": len(rows),
            "loo_rmse_kmh": None if not np.isfinite(loo_rmse) else round(loo_rmse, 2),
            "loo_mae_kmh": None if not np.isfinite(loo_mae) else round(loo_mae, 2),
        }

    model = _fit_ridge(X, y)
    base_features = np.array(
        [current_row.v10_kmh, current_row.gain10_20_kmh, current_row.angle_dev20_deg, current_row.risk_score],
        dtype=float,
    )
    improved_features = _improved_feature_target(base_features=base_features, rows=rows)
    pred_base = _predict_ridge(model, base_features)
    pred_improved = _predict_ridge(model, improved_features)
    expected_gain = max(0.0, float(pred_improved - pred_base))

    baseline_actual = float(current_row.rule_score_kmh)
    potential = baseline_actual + expected_gain
    spread = max(loo_mae, loo_rmse * 0.7)
    low = max(baseline_actual, potential - spread)
    high = max(low, potential + spread)
    contributions = _feature_contributions(
        model=model,
        base_features=base_features,
        improved_features=improved_features,
    )
    positive_contrib = [item for item in contributions if item["delta_kmh"] > 0.2]

    return {
        "available": True,
        "experimental": True,
        "sample_size": len(rows),
        "minimum_sample_size": MIN_CLEAN_HISTORY,
        "confidence": _confidence_label(n=len(rows), loo_rmse_kmh=loo_rmse),
        "validation_method": "leave-one-out",
        "baseline_kmh": round(baseline_actual, 2),
        "potential_kmh": round(potential, 2),
        "potential_low_kmh": round(low, 2),
        "potential_high_kmh": round(high, 2),
        "expected_gain_kmh": round(expected_gain, 2),
        "loo_rmse_kmh": round(loo_rmse, 2),
        "loo_mae_kmh": round(loo_mae, 2),
        "top_levers": [item["label"] for item in positive_contrib[:3]],
        "lever_contributions": positive_contrib[:4],
    }


def _extract_feature_row(report: dict[str, Any]) -> _FeatureRow | None:
    flags = {str(item) for item in report.get("quality_flags", [])}
    if flags & REFERENCE_HARD_FLAGS:
        return None
    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    if bool(notes.get("analysis_blocked")) or bool(notes.get("t0_review_required")):
        return None

    jump = report.get("jump", {})
    metrics = report.get("metrics", {})
    if metrics.get("rule_score_status") != RULE_SCORE_VALID:
        return None
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
    target = _num(metrics.get("rule_based_3s_score"))
    risk = _num(metrics.get("negative_risk_score"))
    jump_id = str(jump.get("jump_id", ""))
    if v10 is None or v20 is None or a20 is None or target is None or risk is None or not jump_id:
        return None

    return _FeatureRow(
        jump_id=jump_id,
        rule_score_kmh=float(target),
        v10_kmh=float(v10),
        gain10_20_kmh=float(phase_gain) if phase_gain is not None else float(v20 - v10),
        angle_dev20_deg=(
            float(phase_angle_dev) if phase_angle_dev is not None else float(abs(a20 - 83.0))
        ),
        risk_score=float(risk),
    )


def _fit_ridge(X: np.ndarray, y: np.ndarray) -> _RidgeModel:
    feature_mean = np.mean(X, axis=0)
    feature_scale = np.std(X, axis=0)
    feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)
    X_normalized = (X - feature_mean) / feature_scale
    target_mean = float(np.mean(y))
    centered_target = y - target_mean
    penalty = np.eye(X_normalized.shape[1], dtype=float) * RIDGE_ALPHA
    coefficients = np.linalg.solve(
        X_normalized.T @ X_normalized + penalty,
        X_normalized.T @ centered_target,
    )
    return _RidgeModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        coefficients=coefficients,
    )


def _predict_ridge(model: _RidgeModel, features: np.ndarray) -> float:
    normalized = (np.asarray(features, dtype=float) - model.feature_mean) / model.feature_scale
    return float(model.target_mean + normalized @ model.coefficients)


def _improved_feature_target(*, base_features: np.ndarray, rows: list[_FeatureRow]) -> np.ndarray:
    targets = np.array(
        [
            [
                np.quantile([r.v10_kmh for r in rows], 0.75),
                np.quantile([r.gain10_20_kmh for r in rows], 0.75),
                np.quantile([r.angle_dev20_deg for r in rows], 0.25),
                np.quantile([r.risk_score for r in rows], 0.25),
            ]
        ],
        dtype=float,
    )[0]
    return np.array(
        [
            min(base_features[0] + 90.0, max(base_features[0], targets[0])),
            min(base_features[1] + 120.0, max(base_features[1], targets[1])),
            max(0.0, min(base_features[2], targets[2])),
            max(0.0, min(base_features[3], targets[3])),
        ],
        dtype=float,
    )


def _feature_contributions(
    *,
    model: _RidgeModel,
    base_features: np.ndarray,
    improved_features: np.ndarray,
) -> list[dict[str, Any]]:
    labels = [
        "Exit-Speed (+10s)",
        "Speed-Aufbau Phasen",
        "Winkelpräzision Hot-Zone/Max-Speed",
        "Stabilität (Risiko)",
    ]
    raw_coefficients = model.coefficients / model.feature_scale
    out = []
    for idx, label in enumerate(labels):
        delta_feature = float(improved_features[idx] - base_features[idx])
        out.append(
            {
                "label": label,
                "delta_kmh": round(float(raw_coefficients[idx] * delta_feature), 2),
            }
        )
    return sorted(out, key=lambda item: float(item["delta_kmh"]), reverse=True)


def _confidence_label(*, n: int, loo_rmse_kmh: float) -> str:
    if n >= 40 and loo_rmse_kmh <= 7.0:
        return "hoch"
    if n >= 20 and loo_rmse_kmh <= 12.0:
        return "mittel"
    return "niedrig"


def _fixpoint_at(fixpoints: list[dict[str, Any]], t_rel_s: float) -> dict[str, Any] | None:
    return next(
        (item for item in fixpoints if abs(float(item.get("t_rel_s", -1)) - t_rel_s) < 1e-6),
        None,
    )


def _phase_build_gain(report: dict[str, Any]) -> float | None:
    raw_phases = report.get("phases")
    phases: list[Any] = raw_phases if isinstance(raw_phases, list) else []
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
    return None if not gains else float(sum(gains))


def _phase_late_angle_deviation(report: dict[str, Any]) -> float | None:
    raw_phases = report.get("phases")
    phases: list[Any] = raw_phases if isinstance(raw_phases, list) else []
    deviations: list[float] = []
    for name, target in [("Hot-Zone Aufbau", 84.0), ("Max-Speed Fenster", 85.0)]:
        row = _phase_by_name(phases, name)
        if row is None:
            continue
        angle = _num(row.get("avg_angle_deg"))
        if angle is not None:
            deviations.append(abs(float(angle) - target))
    return None if not deviations else float(sum(deviations) / len(deviations))


def _phase_by_name(phases: list[Any], name: str) -> dict[str, Any] | None:
    return next(
        (row for row in phases if isinstance(row, dict) and str(row.get("name") or "") == name),
        None,
    )


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
