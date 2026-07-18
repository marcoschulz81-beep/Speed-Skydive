from __future__ import annotations

from typing import Any

from app.config import TARGET_ANGLE_BANDS

RULE_SCORE_VALID = "valid"
RULE_SCORE_ESTIMATED = "estimated"
RULE_SCORE_INVALID = "invalid"

REFERENCE_HARD_FLAGS = frozenset(
    {
        "EARLY_JUMP_END",
        "SPEED_SPIKE",
        "TIME_GAPS",
        "NO_CLEAR_EXIT",
        "INVALID_EXIT_ALTITUDE",
        "HIGH_SPEED_ACCURACY_ERROR",
        "LOW_GPS_FIX",
    }
)


def effective_eval_window_end_s(
    *,
    metrics: dict[str, Any] | None,
    notes: dict[str, Any] | None,
    chart_data: dict[str, Any] | None,
    minimum_s: float = 8.0,
) -> float | None:
    """Return the common technical evaluation end used by every subsystem."""
    metrics = metrics or {}
    notes = notes or {}
    chart_data = chart_data or {}
    candidates: list[float] = []

    for value in (
        metrics.get("performance_window_end_s"),
        notes.get("decel_start_s"),
        notes.get("canopy_open_s"),
        notes.get("curve_window_end_s"),
    ):
        number = _num(value)
        if number is not None and number >= minimum_s:
            candidates.append(float(number))

    max_time = _max_finite(chart_data.get("time_s"))
    if max_time is not None and max_time >= minimum_s:
        if not candidates:
            candidates.append(max_time)
        else:
            candidates.append(max_time)

    if not candidates:
        return None
    end_s = min(candidates)
    return float(end_s) if end_s >= minimum_s else None


def is_reference_eligible(
    *,
    metrics: dict[str, Any] | None,
    notes: dict[str, Any] | None,
    quality_flags: set[str] | list[str] | tuple[str, ...] | None,
) -> bool:
    metrics = metrics or {}
    notes = notes or {}
    flags = {str(item) for item in (quality_flags or [])}
    score = _num(metrics.get("rule_based_3s_score"))
    status = metrics.get("rule_score_status")
    return bool(
        (status == RULE_SCORE_VALID or (status is None and score is not None))
        and score is not None
        and not bool(notes.get("analysis_blocked"))
        and not bool(notes.get("t0_review_required"))
        and not bool(flags & REFERENCE_HARD_FLAGS)
    )


def primary_speed_kmh(metrics: dict[str, Any] | None, *, require_valid: bool = True) -> float | None:
    metrics = metrics or {}
    if require_valid and metrics.get("rule_score_status") != RULE_SCORE_VALID:
        return None
    return _num(metrics.get("rule_based_3s_score"))


def target_angle_band_at(t_rel_s: float) -> tuple[float, float] | None:
    t_value = float(t_rel_s)
    matches: list[tuple[dict[str, object], float, float]] = []
    for item in TARGET_ANGLE_BANDS:
        start_s = _num(item.get("start_s"))
        end_s = _num(item.get("end_s"))
        if start_s is not None and end_s is not None and start_s <= t_value <= end_s:
            matches.append((item, start_s, end_s))
    if not matches:
        return None
    # Overlapping phase definitions are intentional. Prefer the most specific/latest phase.
    match, _, _ = max(matches, key=lambda row: (row[1], -row[2]))
    min_deg = _num(match.get("min_deg"))
    max_deg = _num(match.get("max_deg"))
    if min_deg is None or max_deg is None:
        return None
    return min_deg, max_deg


def angle_distance_to_target(angle_deg: Any, t_rel_s: float) -> float | None:
    angle = _num(angle_deg)
    band = target_angle_band_at(t_rel_s)
    if angle is None or band is None:
        return None
    low, high = band
    if low <= angle <= high:
        return 0.0
    return min(abs(angle - low), abs(angle - high))


def _max_finite(values: Any) -> float | None:
    if not isinstance(values, (list, tuple)):
        return None
    numbers = [_num(value) for value in values]
    finite = [float(value) for value in numbers if value is not None]
    return max(finite) if finite else None


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number
