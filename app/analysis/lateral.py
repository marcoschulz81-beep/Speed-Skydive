from __future__ import annotations

import math
from typing import Any

import numpy as np


def analyze_lateral_dynamics(
    chart_data: dict[str, Any],
    *,
    eval_end_s: float | None = None,
) -> dict[str, Any]:
    t_raw = chart_data.get("time_s", []) or []
    vn_raw = chart_data.get("velN_mps", []) or []
    ve_raw = chart_data.get("velE_mps", []) or []
    vv_raw = chart_data.get("vVert_kmh", []) or []
    av_raw = chart_data.get("accVert_mps2", []) or []
    angle_raw = chart_data.get("angle_deg", []) or []

    if not t_raw or not vn_raw or not ve_raw:
        return {"available": False, "reason": "MISSING_VELOCITY_COMPONENTS"}
    if not (len(t_raw) == len(vn_raw) == len(ve_raw)):
        return {"available": False, "reason": "MISMATCHED_LENGTHS"}

    t_list: list[float] = []
    vn_list: list[float] = []
    ve_list: list[float] = []
    vv_list: list[float | None] = []
    av_list: list[float | None] = []
    angle_list: list[float | None] = []
    has_vv = len(vv_raw) == len(t_raw)
    has_av = len(av_raw) == len(t_raw)
    has_angle = len(angle_raw) == len(t_raw)

    for i in range(len(t_raw)):
        t = _num(t_raw[i])
        vn = _num(vn_raw[i])
        ve = _num(ve_raw[i])
        if t is None or vn is None or ve is None:
            continue
        if not np.isfinite(t) or not np.isfinite(vn) or not np.isfinite(ve):
            continue
        if t < 0.0:
            continue
        t_list.append(float(t))
        vn_list.append(float(vn))
        ve_list.append(float(ve))
        vv_list.append(_num(vv_raw[i]) if has_vv else None)
        av_list.append(_num(av_raw[i]) if has_av else None)
        angle_list.append(_num(angle_raw[i]) if has_angle else None)

    if len(t_list) < 8:
        return {"available": False, "reason": "TOO_FEW_POINTS"}

    t = np.asarray(t_list, dtype=float)
    vn = np.asarray(vn_list, dtype=float)
    ve = np.asarray(ve_list, dtype=float)
    order = np.argsort(t)
    t = t[order]
    vn = vn[order]
    ve = ve[order]
    vv = np.asarray([vv_list[i] for i in order], dtype=object)
    av = np.asarray([av_list[i] for i in order], dtype=object)
    angle = np.asarray([angle_list[i] for i in order], dtype=object)

    end_s = float(eval_end_s) if eval_end_s is not None else float(t[-1])
    end_s = max(8.0, min(float(t[-1]), end_s))
    eval_mask = (t >= 0.0) & (t <= end_s)
    if np.sum(eval_mask) < 8:
        return {"available": False, "reason": "TOO_FEW_POINTS_IN_WINDOW"}

    axis_n, axis_e, axis_source = _reference_axis(
        t=t[eval_mask],
        vn=vn[eval_mask],
        ve=ve[eval_mask],
        end_s=end_s,
    )
    axis_norm = math.hypot(axis_n, axis_e)
    if axis_norm < 1e-9:
        return {"available": False, "reason": "NO_REFERENCE_AXIS"}

    u_forward = np.array([axis_n / axis_norm, axis_e / axis_norm], dtype=float)
    # Positive lateral means drift to the right relative to forward direction.
    u_lateral = np.array([-u_forward[1], u_forward[0]], dtype=float)

    v_forward = vn * u_forward[0] + ve * u_forward[1]
    v_lat = vn * u_lateral[0] + ve * u_lateral[1]
    v_hor = np.hypot(vn, ve)
    heading_rad = np.unwrap(np.arctan2(ve, vn))
    heading_rate_dps = _safe_gradient(heading_rad, t) * (180.0 / math.pi)
    a_lat = _safe_gradient(v_lat, t)

    dt = np.diff(t)
    cross_track = np.zeros_like(v_lat)
    if len(t) >= 2:
        cross_track[1:] = np.cumsum(0.5 * (v_lat[:-1] + v_lat[1:]) * dt)

    hot_start = 0.70 * end_s
    hot_end = 0.90 * end_s
    hot_mask = (t >= hot_start) & (t <= hot_end)
    if np.sum(hot_mask) < 4:
        hot_mask = (t >= max(0.0, end_s - 6.0)) & (t <= end_s)

    median_dt = _median_dt(t)
    smooth_win = _window_samples(median_dt, seconds=0.8, minimum=3)
    v_lat_s = _moving_average(v_lat, smooth_win)
    heading_rate_s = _moving_average(heading_rate_dps, smooth_win)
    a_lat_s = _moving_average(a_lat, smooth_win)

    hot_abs_mean_kmh = _safe_mean(np.abs(v_lat_s[hot_mask])) * 3.6
    hot_signed_mean_kmh = _safe_mean(v_lat_s[hot_mask]) * 3.6
    hot_sign_changes = _sign_changes(v_lat_s[hot_mask], eps=0.9)
    hot_heading_rms = _safe_rms(heading_rate_s[hot_mask])
    hot_alat_peak = _safe_max(np.abs(a_lat_s[hot_mask]))
    hot_cross_span_m = _safe_span(cross_track[hot_mask])
    hot_direction = _direction_label(signed_kmh=hot_signed_mean_kmh, abs_mean_kmh=hot_abs_mean_kmh)
    hot_pattern = _hot_pattern(
        abs_mean_kmh=hot_abs_mean_kmh,
        sign_changes=hot_sign_changes,
        heading_rate_rms_dps=hot_heading_rms,
    )

    speed_cost_event = _lateral_speed_cost_event(
        t=t,
        hot_mask=hot_mask,
        v_lat=v_lat_s,
        a_lat=a_lat_s,
        v_vert_kmh=vv,
        a_vert_mps2=av,
        angle_deg=angle,
    )

    high_speed = _high_speed_stability(
        t=t,
        end_s=end_s,
        v_vert_kmh=vv,
        angle_deg=angle,
        v_lat=v_lat_s,
        heading_rate_dps=heading_rate_s,
    )

    return {
        "available": True,
        "axis_source": axis_source,
        "axis_unit_n": float(u_forward[0]),
        "axis_unit_e": float(u_forward[1]),
        "hot_start_s": float(hot_start),
        "hot_end_s": float(hot_end),
        "hot_vlat_abs_mean_kmh": hot_abs_mean_kmh,
        "hot_vlat_signed_mean_kmh": hot_signed_mean_kmh,
        "hot_direction": hot_direction,
        "hot_sign_changes": hot_sign_changes,
        "hot_heading_rate_rms_dps": hot_heading_rms,
        "hot_alat_peak_mps2": hot_alat_peak,
        "hot_cross_track_span_m": hot_cross_span_m,
        "hot_pattern": hot_pattern,
        "speed_cost_event": speed_cost_event,
        "high_speed": high_speed,
        "no_wind_correction": True,
        "metrics": {
            "v_forward_mean_kmh": _safe_mean(v_forward[eval_mask]) * 3.6,
            "v_hor_mean_kmh": _safe_mean(v_hor[eval_mask]) * 3.6,
        },
    }


def _reference_axis(*, t: np.ndarray, vn: np.ndarray, ve: np.ndarray, end_s: float) -> tuple[float, float, str]:
    ref_mask = (t >= 3.0) & (t <= min(8.0, end_s))
    source = "3-8s"
    if np.sum(ref_mask) < 3:
        ref_mask = (t >= 0.5) & (t <= min(max(4.0, 0.35 * end_s), 12.0))
        source = "0.5-12s"
    if np.sum(ref_mask) < 3:
        speeds = np.hypot(vn, ve)
        top_idx = np.argsort(speeds)[-min(12, len(speeds)) :]
        ref_mask = np.zeros_like(t, dtype=bool)
        ref_mask[top_idx] = True
        source = "top-speed"

    axis_n = float(np.median(vn[ref_mask])) if np.sum(ref_mask) > 0 else 0.0
    axis_e = float(np.median(ve[ref_mask])) if np.sum(ref_mask) > 0 else 0.0
    if math.hypot(axis_n, axis_e) < 1e-9:
        speeds = np.hypot(vn, ve)
        best_idx = int(np.argmax(speeds))
        axis_n = float(vn[best_idx])
        axis_e = float(ve[best_idx])
        source = "max-hor-speed"
    return axis_n, axis_e, source


def _lateral_speed_cost_event(
    *,
    t: np.ndarray,
    hot_mask: np.ndarray,
    v_lat: np.ndarray,
    a_lat: np.ndarray,
    v_vert_kmh: np.ndarray,
    a_vert_mps2: np.ndarray,
    angle_deg: np.ndarray,
) -> dict[str, Any]:
    if np.sum(hot_mask) < 5:
        return {"available": False}

    abs_alat_hot = np.abs(a_lat[hot_mask])
    if len(abs_alat_hot) < 3:
        return {"available": False}
    a_lat_threshold = max(1.5, float(np.percentile(abs_alat_hot, 90)))

    hot_indices = np.where(hot_mask)[0]
    peak_idx = int(hot_indices[np.argmax(np.abs(a_lat[hot_indices]))])
    if abs(float(a_lat[peak_idx])) < a_lat_threshold:
        return {"available": False}

    t_peak = float(t[peak_idx])
    pre_mask = (t >= max(t[0], t_peak - 0.8)) & (t <= t_peak - 0.1)
    post_mask = (t >= t_peak + 0.5) & (t <= t_peak + 2.0)
    if np.sum(pre_mask) < 2 or np.sum(post_mask) < 2:
        return {"available": False}

    pre_a = _safe_mean_optional(a_vert_mps2[pre_mask])
    post_a = _safe_mean_optional(a_vert_mps2[post_mask])
    pre_v = _safe_mean_optional(v_vert_kmh[pre_mask])
    post_v = _safe_mean_optional(v_vert_kmh[post_mask])
    angle_at_peak = _safe_mean_optional(angle_deg[(t >= t_peak - 0.3) & (t <= t_peak + 0.3)])

    drop_a = None if pre_a is None or post_a is None else float(pre_a - post_a)
    drop_v = None if pre_v is None or post_v is None else float(pre_v - post_v)
    vlat_at_peak_kmh = float(v_lat[peak_idx] * 3.6)

    likely = False
    if drop_a is not None and drop_v is not None:
        if drop_a >= 0.8 and drop_v >= 6.0:
            likely = True
    if not likely and drop_v is not None and drop_v >= 10.0:
        likely = True
    if abs(vlat_at_peak_kmh) < 6.0:
        likely = False

    direction = "rechts" if vlat_at_peak_kmh > 0 else "links"
    return {
        "available": True,
        "likely_speed_cost": bool(likely),
        "t_peak_s": t_peak,
        "direction": direction,
        "vlat_at_peak_kmh": vlat_at_peak_kmh,
        "alat_peak_mps2": float(a_lat[peak_idx]),
        "avert_drop_mps2": drop_a,
        "vvert_drop_kmh": drop_v,
        "angle_at_peak_deg": angle_at_peak,
    }


def _high_speed_stability(
    *,
    t: np.ndarray,
    end_s: float,
    v_vert_kmh: np.ndarray,
    angle_deg: np.ndarray,
    v_lat: np.ndarray,
    heading_rate_dps: np.ndarray,
) -> dict[str, Any]:
    v_vert = _to_float_array(v_vert_kmh)
    if v_vert is None or len(v_vert) != len(t):
        return {"available": False}

    eval_mask = (t >= 0.0) & (t <= end_s)
    if np.sum(eval_mask) < 5:
        return {"available": False}

    vmax = float(np.nanmax(v_vert[eval_mask]))
    if not np.isfinite(vmax) or vmax < 220.0:
        return {"available": False}
    threshold = 0.8 * vmax
    hs_mask = eval_mask & (v_vert >= threshold)
    if np.sum(hs_mask) < 5:
        return {"available": False}

    angle_f = _to_float_array(angle_deg)
    theta_std = None
    if angle_f is not None and len(angle_f) == len(t):
        vals = angle_f[hs_mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 4:
            theta_std = float(np.std(vals))

    vlat_abs_mean = _safe_mean(np.abs(v_lat[hs_mask])) * 3.6
    heading_rms = _safe_rms(heading_rate_dps[hs_mask])
    unstable = False
    if theta_std is not None and theta_std >= 2.2:
        unstable = True
    if vlat_abs_mean >= 8.0 or heading_rms >= 10.0:
        unstable = True

    return {
        "available": True,
        "threshold_kmh": threshold,
        "theta_std_deg": theta_std,
        "vlat_abs_mean_kmh": vlat_abs_mean,
        "heading_rate_rms_dps": heading_rms,
        "unstable": unstable,
    }


def _hot_pattern(*, abs_mean_kmh: float, sign_changes: int, heading_rate_rms_dps: float) -> str:
    if sign_changes >= 4 or heading_rate_rms_dps >= 10.0:
        return "schlangenlinie"
    if abs_mean_kmh >= 7.0 and sign_changes <= 2:
        return "gerichtete_drift"
    if abs_mean_kmh <= 4.0 and sign_changes <= 2 and heading_rate_rms_dps <= 5.0:
        return "ruhig"
    return "gemischt"


def _direction_label(*, signed_kmh: float, abs_mean_kmh: float) -> str:
    if abs_mean_kmh < 4.0 or abs(signed_kmh) < 1.5:
        return "neutral"
    return "rechts" if signed_kmh > 0 else "links"


def _to_float_array(values: np.ndarray) -> np.ndarray | None:
    out: list[float] = []
    has_any = False
    for item in values.tolist():
        n = _num(item)
        if n is None:
            out.append(float("nan"))
        else:
            has_any = True
            out.append(float(n))
    if not has_any:
        return None
    return np.asarray(out, dtype=float)


def _safe_gradient(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(values) < 3 or len(t) < 3:
        return np.zeros_like(values, dtype=float)
    try:
        return np.gradient(values, t)
    except Exception:
        return np.zeros_like(values, dtype=float)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.astype(float)
    w = int(max(1, window))
    if w % 2 == 0:
        w += 1
    if w >= len(values):
        w = max(3, len(values) // 2 * 2 + 1)
    if w <= 1:
        return values.astype(float)
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(values, kernel, mode="same")


def _window_samples(dt: float, *, seconds: float, minimum: int) -> int:
    if not np.isfinite(dt) or dt <= 0.0:
        return minimum
    return max(minimum, int(round(seconds / dt)))


def _median_dt(t: np.ndarray) -> float:
    if len(t) < 2:
        return 0.2
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(dt) == 0:
        return 0.2
    return float(np.median(dt))


def _safe_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


def _safe_mean_optional(values: np.ndarray) -> float | None:
    arr = _to_float_array(values)
    if arr is None:
        return None
    vals = arr[np.isfinite(arr)]
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def _safe_rms(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return 0.0
    return float(math.sqrt(np.mean(np.square(vals))))


def _safe_max(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return 0.0
    return float(np.max(vals))


def _safe_span(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return 0.0
    return float(np.max(vals) - np.min(vals))


def _sign_changes(values: np.ndarray, *, eps: float) -> int:
    if len(values) < 3:
        return 0
    states: list[int] = []
    for value in values.tolist():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        v = float(value)
        if abs(v) <= eps:
            continue
        states.append(1 if v > 0 else -1)
    if len(states) < 2:
        return 0
    changes = 0
    prev = states[0]
    for cur in states[1:]:
        if cur != prev:
            changes += 1
            prev = cur
    return changes


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
