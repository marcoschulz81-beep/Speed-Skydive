from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_curve_window(
    samples: pd.DataFrame,
    *,
    sample_rate_hz: float | None = None,
) -> dict[str, Any]:
    """
    Determine a practical chart display range:
    from t0 (0s) to min(deceleration_start + 10s, canopy_open_estimate).
    """
    post = samples[samples["t_rel_s"] >= 0].sort_values("t_rel_s").copy()
    if post.empty:
        return {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 0.0,
            "decel_start_s": None,
            "canopy_open_s": None,
            "peak_s": None,
            "curve_window_reason": "Keine Post-Exit-Daten.",
        }

    t = post["t_rel_s"].to_numpy(dtype=float)
    vvert = post["vVert_kmh"].to_numpy(dtype=float)
    t_max = float(t[-1])

    if len(t) < 8:
        return {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": round(t_max, 2),
            "decel_start_s": None,
            "canopy_open_s": None,
            "peak_s": None,
            "curve_window_reason": "Zu wenige Samples, volles Fenster verwendet.",
        }

    if sample_rate_hz is None or sample_rate_hz <= 0:
        dt = np.diff(t)
        dt = dt[dt > 0]
        sample_rate_hz = float(1.0 / np.median(dt)) if len(dt) else 5.0
    sample_rate_hz = float(max(sample_rate_hz, 1.0))

    smooth_window = max(3, int(round(sample_rate_hz * 1.0)))
    vvert_smooth = pd.Series(vvert).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
    dv = np.gradient(vvert_smooth, t)

    peak_idx = int(np.argmax(vvert_smooth))
    peak_t = float(t[peak_idx])
    peak_v = float(vvert_smooth[peak_idx])

    sustained_n = max(4, int(round(sample_rate_hz * 0.8)))
    decel_idx = _first_sustained_index(
        condition=((dv < -8.0) & ((peak_v - vvert_smooth) > 20.0)),
        start_idx=min(peak_idx + 1, len(t) - 1),
        min_run=sustained_n,
    )
    decel_t = None if decel_idx is None else float(t[decel_idx])

    canopy_idx = _detect_canopy_open_idx(
        t=t,
        vvert_smooth=vvert_smooth,
        dv=dv,
        peak_v=peak_v,
        sample_rate_hz=sample_rate_hz,
        decel_t=decel_t,
    )
    canopy_t = None if canopy_idx is None else float(t[canopy_idx])

    if decel_t is not None and canopy_t is not None:
        curve_end = min(decel_t + 10.0, canopy_t)
    elif decel_t is not None:
        curve_end = decel_t + 10.0
    elif canopy_t is not None:
        curve_end = canopy_t
    else:
        curve_end = max(peak_t + 10.0, min(35.0, t_max))

    curve_end = float(min(curve_end, t_max))
    curve_end = float(max(curve_end, min(t_max, 8.0)))

    reason_parts = [f"Peak bei +{peak_t:.1f}s"]
    if decel_t is not None:
        reason_parts.append(f"Abbremsen ab +{decel_t:.1f}s")
    if canopy_t is not None:
        reason_parts.append(f"Schirmindikator bei +{canopy_t:.1f}s")
    reason_parts.append(f"Anzeige bis +{curve_end:.1f}s")

    return {
        "curve_window_start_s": 0.0,
        "curve_window_end_s": round(curve_end, 2),
        "decel_start_s": None if decel_t is None else round(decel_t, 2),
        "canopy_open_s": None if canopy_t is None else round(canopy_t, 2),
        "peak_s": round(peak_t, 2),
        "curve_window_reason": ", ".join(reason_parts) + ".",
    }


def _first_sustained_index(condition: np.ndarray, start_idx: int, min_run: int) -> int | None:
    run_start = None
    run_len = 0
    for idx in range(start_idx, len(condition)):
        if bool(condition[idx]):
            if run_start is None:
                run_start = idx
            run_len += 1
            if run_len >= min_run:
                return run_start
        else:
            run_start = None
            run_len = 0
    return None


def _detect_canopy_open_idx(
    *,
    t: np.ndarray,
    vvert_smooth: np.ndarray,
    dv: np.ndarray,
    peak_v: float,
    sample_rate_hz: float,
    decel_t: float | None,
) -> int | None:
    if len(t) < 10:
        return None

    start_t = (decel_t + 4.0) if decel_t is not None else (float(t[np.argmax(vvert_smooth)]) + 6.0)
    start_idx = int(np.searchsorted(t, start_t))
    if start_idx >= len(t) - 2:
        return None

    forward_n = max(2, int(round(sample_rate_hz * 1.2)))
    threshold_v = max(130.0, peak_v * 0.45)

    for idx in range(start_idx, len(t) - forward_n):
        steep_drop = dv[idx] < -25.0
        low_enough = vvert_smooth[idx] <= threshold_v
        delta_future = vvert_smooth[idx] - vvert_smooth[idx + forward_n]
        if steep_drop and low_enough and delta_future > 18.0:
            return idx

    tail = slice(start_idx, len(t))
    local_min_idx = start_idx + int(np.argmin(dv[tail]))
    if dv[local_min_idx] < -40.0 and vvert_smooth[local_min_idx] < (peak_v * 0.6):
        return local_min_idx

    return None

