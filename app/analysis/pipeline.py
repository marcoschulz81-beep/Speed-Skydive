from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.curve_window import detect_curve_window
from app.analysis.evaluation import (
    REFERENCE_HARD_FLAGS,
    RULE_SCORE_ESTIMATED,
    RULE_SCORE_INVALID,
    RULE_SCORE_VALID,
)
from app.config import (
    ANALYSIS_VERSION,
    DEFAULT_BREAKOFF_ALTITUDE_AGL_M,
    FIXPOINT_SECONDS,
    MAX_SACC_MPS,
    MAX_VALID_EXIT_ALTITUDE_AGL_M,
    MIN_NUM_SV,
    MIN_SAMPLE_RATE_HZ,
    NEGATIVE_RISK_LOOKBACK_S,
    PERFORMANCE_WINDOW_VERTICAL_DROP_M,
    REQUIRED_COLUMNS,
    SCORING_GRID_STEP_S,
    SCORING_WINDOW_DURATION_S,
    TARGET_ANGLE_BANDS,
    TECHNICAL_PHASE_SPECS,
    VALIDATION_WINDOW_VERTICAL_DROP_M,
)


class AnalysisError(Exception):
    pass


@dataclass
class WindowResult:
    start_s: float
    end_s: float
    avg_vvert_mps: float
    avg_vvert_kmh: float
    avg_vhor_kmh: float
    avg_angle_deg: float


@dataclass
class PreparedFlySightTrack:
    df: pd.DataFrame
    unit_normalization: dict[str, Any] | None
    leading_gap_repair: dict[str, Any] | None
    device_type: str
    raw_start_time_utc: str
    t_abs_s: np.ndarray
    dt: np.ndarray
    sample_rate_hz: float
    quality_flags: tuple[str, ...]
    quality_score: float
    auto_t0_idx: int
    auto_t0_confidence: float
    auto_t0_uncertainty_s: float
    auto_t0_reason: str
    auto_t0_utc: str


def _read_csv(content: bytes) -> pd.DataFrame:
    raw = content.decode("utf-8", errors="replace")
    if _looks_like_flysight2_log(raw):
        df = _read_flysight2_track(raw)
    else:
        try:
            df = pd.read_csv(StringIO(raw), low_memory=False)
        except Exception as exc:  # pragma: no cover - pandas errors are noisy
            raise AnalysisError(f"CSV konnte nicht gelesen werden: {exc}") from exc

    if df.empty:
        raise AnalysisError("CSV enthält keine Datenzeilen.")

    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        missing_set = set(missing)
        if missing_set.issubset({"gpsFix", "sAcc", "hAcc", "vAcc", "numSV"}):
            if "gpsFix" in missing_set:
                df["gpsFix"] = 3
            if "numSV" in missing_set:
                df["numSV"] = np.nan
            if "sAcc" in missing_set:
                df["sAcc"] = np.nan
            if "hAcc" in missing_set:
                df["hAcc"] = np.nan
            if "vAcc" in missing_set:
                df["vAcc"] = np.nan
            missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise AnalysisError(f"Pflichtspalten fehlen: {', '.join(missing)}")

    if "lat" not in df.columns:
        df["lat"] = np.nan
    if "lon" not in df.columns:
        df["lon"] = np.nan

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="ISO8601")
    if df["time"].isna().all():
        raise AnalysisError("Keine gültigen Zeitstempel in Spalte 'time' gefunden.")

    numeric_cols = [
        "lat",
        "lon",
        "hMSL",
        "velN",
        "velE",
        "velD",
        "sAcc",
        "hAcc",
        "vAcc",
        "gpsFix",
        "numSV",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time", "hMSL", "velN", "velE", "velD"])
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
    if len(df) < 20:
        raise AnalysisError("Zu wenige gültige Samples für eine robuste Analyse.")

    return df


def _looks_like_flysight2_log(raw: str) -> bool:
    head = raw[:4096]
    return "$FLYS" in head and "$COL,GNSS" in head


def _read_flysight2_track(raw: str) -> pd.DataFrame:
    gnss_columns: list[str] | None = None
    rows: list[list[str]] = []
    in_data = False

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("$COL,GNSS,"):
            gnss_columns = [part.strip() for part in line.split(",")[2:]]
            continue
        if line == "$DATA":
            in_data = True
            continue
        if not in_data or not line.startswith("$GNSS,"):
            continue

        parts = [part.strip() for part in line.split(",")]
        row_values = parts[1:]
        if gnss_columns is not None:
            if len(row_values) < len(gnss_columns):
                row_values += [""] * (len(gnss_columns) - len(row_values))
            elif len(row_values) > len(gnss_columns):
                row_values = row_values[: len(gnss_columns)]
        rows.append(row_values)

    if gnss_columns is None:
        raise AnalysisError(
            "FlySight-2 Datei erkannt, aber keine GNSS-Spaltendefinition gefunden. Bitte TRACK.CSV verwenden."
        )
    if not rows:
        raise AnalysisError("FlySight-2 TRACK.CSV enthält keine GNSS-Daten.")

    df = pd.DataFrame(rows, columns=gnss_columns)
    # Keep later logic unchanged by exposing classic FlySight columns.
    if "gpsFix" not in df.columns:
        df["gpsFix"] = 3
    if "sAcc" not in df.columns:
        df["sAcc"] = np.nan
    if "hAcc" not in df.columns:
        df["hAcc"] = np.nan
    if "vAcc" not in df.columns:
        df["vAcc"] = np.nan
    if "numSV" not in df.columns:
        df["numSV"] = np.nan

    df.attrs["device_type_hint"] = "FlySight 2"
    df.attrs["source_format"] = "FLYSIGHT2_TRACK"
    return df


def _detect_device_type(df: pd.DataFrame) -> str:
    hint = str(df.attrs.get("device_type_hint") or "").strip()
    if hint:
        return hint
    imu_markers = {"ax", "ay", "az", "gx", "gy", "gz", "accX", "accY", "accZ"}
    columns = {str(c).strip() for c in df.columns}
    if columns.intersection(imu_markers):
        return "FlySight 2"
    return "FlySight 1"


def _validate_plausible_jump_profile(df: pd.DataFrame) -> None:
    vel_d = df["velD"].to_numpy(dtype=float)
    valid = vel_d[~np.isnan(vel_d)]
    if len(valid) == 0:
        raise AnalysisError("Keine gültigen vertikalen Geschwindigkeitswerte gefunden.")

    if float(np.max(valid)) < 10.0:
        raise AnalysisError(
            "Kein plausibler Sprung erkannt: vertikale Geschwindigkeit bleibt durchgehend unter 10 m/s."
        )


def _repair_leading_orphan_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """
    Repair pattern: one or few early samples, then a large gap, then normal 5 Hz track.
    In that case, trim the orphan prefix so exit phase is not distorted by missing 0..N seconds.
    """
    if len(df) < 30:
        return df, None

    t_abs_s = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    if len(t_abs_s) < 8:
        return df, None
    dt = np.diff(t_abs_s)
    positive_dt = dt[dt > 0]
    if len(positive_dt) < 6:
        return df, None

    median_dt = float(np.median(positive_dt))
    gap_threshold_s = max(0.6, median_dt * 2.5)
    cut_idx: int | None = None

    max_probe_idx = min(4, len(dt) - 2)
    for i in range(max_probe_idx):
        gap = float(dt[i])
        if gap <= gap_threshold_s:
            continue

        prefix_duration = float(t_abs_s[i] - t_abs_s[0])
        if prefix_duration > 1.2:
            continue

        post_dt = dt[i + 1 : i + 1 + 12]
        if len(post_dt) < 5:
            continue
        post_regular_ratio = float(np.mean(post_dt <= max(gap_threshold_s, median_dt * 1.8)))
        if post_regular_ratio < 0.8:
            continue

        # keep enough samples for robust analysis
        candidate_cut = i + 1
        if (len(df) - candidate_cut) < 20:
            continue

        cut_idx = candidate_cut
        break

    if cut_idx is None:
        return df, None

    gap_removed_s = float(t_abs_s[cut_idx] - t_abs_s[0])
    repaired = df.iloc[cut_idx:].reset_index(drop=True)
    note = {
        "applied": True,
        "trimmed_samples": int(cut_idx),
        "gap_removed_s": round(gap_removed_s, 3),
        "reason": "leading_orphan_samples",
    }
    return repaired, note


def _normalize_import_units(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """
    Normalize common non-standard exports:
    - velocity columns in km/h instead of m/s
    - altitude column hMSL in feet instead of meters
    """
    if df.empty:
        return df, None

    out = df.copy()
    note: dict[str, Any] = {}

    vel_total = np.sqrt(out["velN"].to_numpy(dtype=float) ** 2 + out["velE"].to_numpy(dtype=float) ** 2 + out["velD"].to_numpy(dtype=float) ** 2)
    finite_vel = vel_total[np.isfinite(vel_total)]
    if len(finite_vel) >= 20:
        v_max = float(np.nanmax(finite_vel))
        p99 = float(np.nanpercentile(finite_vel, 99))
        p99_converted = p99 / 3.6
        v_max_converted = v_max / 3.6
        looks_like_kmh = v_max > 220.0 and 40.0 <= v_max_converted <= 170.0
        if looks_like_kmh:
            out["velN"] = out["velN"] / 3.6
            out["velE"] = out["velE"] / 3.6
            out["velD"] = out["velD"] / 3.6
            note["velocity_conversion"] = {
                "applied": True,
                "from": "kmh",
                "to": "mps",
                "max_before": round(v_max, 2),
                "max_after": round(v_max_converted, 2),
                "p99_before": round(p99, 2),
                "p99_after": round(p99_converted, 2),
            }

    h_values = out["hMSL"].to_numpy(dtype=float)
    finite_h = h_values[np.isfinite(h_values)]
    if len(finite_h) >= 20:
        h_max = float(np.nanmax(finite_h))
        h_p99 = float(np.nanpercentile(finite_h, 99))
        looks_like_feet = 7000.0 <= h_max <= 40000.0 and (h_p99 > 2200.0 or h_max > 9000.0)
        if looks_like_feet:
            out["hMSL"] = out["hMSL"] / 3.28084
            note["altitude_conversion"] = {
                "applied": True,
                "from": "ft",
                "to": "m",
                "max_before": round(h_max, 2),
                "max_after": round(h_max / 3.28084, 2),
                "p99_before": round(h_p99, 2),
                "p99_after": round(h_p99 / 3.28084, 2),
            }

    return out, (note if note else None)


def _safe_interp(x: np.ndarray, y: np.ndarray, x_target: float) -> float | None:
    if len(x) < 2 or x_target < x[0] or x_target > x[-1]:
        return None
    return float(np.interp(x_target, x, y))


def _manual_t0_from_utc(
    *,
    manual_t0_utc: str,
    raw_start_time: pd.Timestamp,
    t_abs_s: np.ndarray,
) -> tuple[float, str]:
    try:
        manual_ts = pd.Timestamp(manual_t0_utc)
    except Exception as exc:
        raise AnalysisError("Manueller Absprungzeitpunkt ist kein gueltiger UTC-Zeitstempel.") from exc

    if manual_ts.tzinfo is None:
        manual_ts = manual_ts.tz_localize("UTC")
    else:
        manual_ts = manual_ts.tz_convert("UTC")

    raw_start = pd.Timestamp(raw_start_time)
    if raw_start.tzinfo is None:
        raw_start = raw_start.tz_localize("UTC")
    else:
        raw_start = raw_start.tz_convert("UTC")

    t0_abs_s = float((manual_ts - raw_start).total_seconds())
    if len(t_abs_s) == 0 or t0_abs_s < float(t_abs_s[0]) or t0_abs_s > float(t_abs_s[-1]):
        raise AnalysisError("Manueller Absprungzeitpunkt liegt ausserhalb der Original-CSV.")
    return t0_abs_s, manual_ts.isoformat()


def _build_fs2_track_summary(
    *,
    post: pd.DataFrame,
    curve_window: dict[str, Any],
    device_type: str,
) -> dict[str, Any] | None:
    if device_type != "FlySight 2" or post.empty:
        return None

    curve_end_raw = curve_window.get("curve_window_end_s")
    try:
        curve_end = float(curve_end_raw) if curve_end_raw is not None else float(post["t_rel_s"].max())
    except Exception:
        curve_end = float(post["t_rel_s"].max())
    decel_raw = curve_window.get("decel_start_s")
    try:
        decel = float(decel_raw) if decel_raw is not None else None
    except Exception:
        decel = None

    end_s = min(25.0, curve_end)
    if decel is not None and decel >= 8.0:
        end_s = min(end_s, decel)
    if end_s <= 8.0:
        return {
            "available": False,
            "reason": "window_too_short",
        }
    start_s = max(0.0, end_s - 5.0)

    window = post[(post["t_rel_s"] >= start_s) & (post["t_rel_s"] <= end_s)].copy()
    if len(window) < 12:
        return {
            "available": False,
            "reason": "not_enough_samples",
            "window_start_s": round(start_s, 2),
            "window_end_s": round(end_s, 2),
            "sample_count": int(len(window)),
        }

    def _metric(col: str, fn: str) -> float | None:
        if col not in window.columns:
            return None
        vals = pd.to_numeric(window[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 6:
            return None
        if fn == "mean":
            return float(np.mean(vals))
        if fn == "p95":
            return float(np.nanpercentile(vals, 95))
        if fn == "p10":
            return float(np.nanpercentile(vals, 10))
        return None

    sacc_mean = _metric("sAcc", "mean")
    sacc_p95 = _metric("sAcc", "p95")
    hacc_p95 = _metric("hAcc", "p95")
    vacc_p95 = _metric("vAcc", "p95")
    numsv_p10 = _metric("numSV", "p10")

    risk_points = 0
    if sacc_p95 is not None:
        if sacc_p95 >= 2.2:
            risk_points += 2
        elif sacc_p95 >= 1.6:
            risk_points += 1
    if hacc_p95 is not None:
        if hacc_p95 >= 14.0:
            risk_points += 2
        elif hacc_p95 >= 10.0:
            risk_points += 1
    if numsv_p10 is not None:
        if numsv_p10 < 10.0:
            risk_points += 2
        elif numsv_p10 < 12.0:
            risk_points += 1

    quality_label = "stabil"
    if risk_points >= 4:
        quality_label = "kritisch"
    elif risk_points >= 2:
        quality_label = "grenzwertig"

    return {
        "available": True,
        "window_start_s": round(start_s, 2),
        "window_end_s": round(end_s, 2),
        "sample_count": int(len(window)),
        "sAcc_mean": None if sacc_mean is None else round(sacc_mean, 3),
        "sAcc_p95": None if sacc_p95 is None else round(sacc_p95, 3),
        "hAcc_p95": None if hacc_p95 is None else round(hacc_p95, 3),
        "vAcc_p95": None if vacc_p95 is None else round(vacc_p95, 3),
        "numSV_p10": None if numsv_p10 is None else round(numsv_p10, 3),
        "quality_label": quality_label,
        "risk_points": int(risk_points),
    }


def _compute_quality_flags(df: pd.DataFrame, sample_rate_hz: float, dt: np.ndarray) -> tuple[list[str], float]:
    flags: list[str] = []
    if sample_rate_hz < (MIN_SAMPLE_RATE_HZ - 0.25):
        flags.append("LOW_SAMPLE_RATE")

    if (df["gpsFix"] != 3).mean() > 0.05:
        flags.append("LOW_GPS_FIX")

    if (df["sAcc"] >= MAX_SACC_MPS).mean() > 0.05:
        flags.append("HIGH_SPEED_ACCURACY_ERROR")

    # Compare altitude delta against expected delta from vertical speed.
    # This avoids false ALTITUDE_SPIKE flags during normal fast freefall.
    altitude_step = np.abs(np.diff(df["hMSL"].to_numpy(dtype=float)))
    if len(altitude_step) and len(dt):
        expected_step = np.abs(df["velD"].to_numpy(dtype=float)[1:] * dt)
        step_residual = np.abs(altitude_step - expected_step)
        spike_threshold = np.maximum(45.0, expected_step * 2.2 + 10.0)
        spike_mask = step_residual > spike_threshold
        if float(np.mean(spike_mask)) > 0.01:
            flags.append("ALTITUDE_SPIKE")

    v_total_mps = np.sqrt(df["velN"] ** 2 + df["velE"] ** 2 + df["velD"] ** 2)
    v_step = np.diff(v_total_mps.to_numpy())
    if np.any(np.abs(v_step) > 30.0) or float(v_total_mps.max()) > 180.0:
        flags.append("SPEED_SPIKE")

    if (df["numSV"] < MIN_NUM_SV).mean() > 0.15:
        flags.append("LOW_NUM_SV")

    score = 100.0
    penalties = {
        "LOW_SAMPLE_RATE": 18,
        "LOW_GPS_FIX": 20,
        "HIGH_SPEED_ACCURACY_ERROR": 20,
        "TIME_GAPS": 12,
        "ALTITUDE_SPIKE": 10,
        "SPEED_SPIKE": 10,
        "LOW_NUM_SV": 8,
    }
    for flag in flags:
        score -= penalties.get(flag, 7)
    return flags, max(score, 0.0)


def _has_time_gaps_in_window(df: pd.DataFrame, *, time_col: str) -> bool:
    if df.empty or time_col not in df.columns:
        return False
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 3:
        return False
    dt = np.diff(t)
    if len(dt) == 0:
        return False
    median_dt = float(np.nanmedian(dt))
    threshold = max(median_dt * 2.5, 0.6)
    return bool(np.any(dt > threshold))


def _has_speed_spike_in_window(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    if not {"velN", "velE", "velD"}.issubset(df.columns):
        return False
    v_n = pd.to_numeric(df["velN"], errors="coerce").to_numpy(dtype=float)
    v_e = pd.to_numeric(df["velE"], errors="coerce").to_numpy(dtype=float)
    v_d = pd.to_numeric(df["velD"], errors="coerce").to_numpy(dtype=float)
    v_total = np.sqrt(v_n ** 2 + v_e ** 2 + v_d ** 2)
    v_total = v_total[np.isfinite(v_total)]
    if len(v_total) < 3:
        return False
    v_step = np.diff(v_total)
    if np.any(np.abs(v_step) > 30.0):
        return True
    if float(np.max(v_total)) > 180.0:
        return True
    return False


def _first_sustained_index(
    mask: np.ndarray,
    *,
    start_idx: int,
    min_run: int,
    end_idx: int | None = None,
) -> int | None:
    run_start: int | None = None
    run_len = 0
    stop = len(mask) if end_idx is None else min(end_idx, len(mask))
    for i in range(start_idx, stop):
        if bool(mask[i]):
            if run_start is None:
                run_start = i
            run_len += 1
            if run_len >= min_run:
                return run_start
        else:
            run_start = None
            run_len = 0
    return None


def _detect_t0(df: pd.DataFrame, t_abs_s: np.ndarray, sample_rate_hz: float) -> tuple[int, float, float, str]:
    """
    Detect real exit timing (t_exit), not high-speed anchor timing.
    Returns: index, confidence(0..1), uncertainty_seconds, reason
    """
    smooth_window = max(3, int(round(sample_rate_hz * 0.7)))
    pre_window = max(5, int(round(sample_rate_hz * 2.0)))
    future_window = max(4, int(round(sample_rate_hz * 0.9)))
    min_run = max(4, int(round(sample_rate_hz * 0.6)))

    smooth = pd.DataFrame(index=df.index)
    smooth["hMSL"] = df["hMSL"].rolling(smooth_window, center=True, min_periods=1).mean()
    smooth["velD"] = df["velD"].rolling(smooth_window, center=True, min_periods=1).mean()
    smooth["vHor"] = np.sqrt(df["velN"] ** 2 + df["velE"] ** 2).rolling(
        smooth_window, center=True, min_periods=1
    ).mean()
    smooth["accVert"] = np.gradient(smooth["velD"].to_numpy(), t_abs_s)
    smooth["hDropRate"] = -np.gradient(smooth["hMSL"].to_numpy(), t_abs_s)

    # Anchor to the primary freefall event: search exit BEFORE the dominant, sustained peak.
    peak_idx = _select_primary_peak_index(
        vel_d_smooth=smooth["velD"].to_numpy(dtype=float),
        t_abs_s=t_abs_s,
        sample_rate_hz=sample_rate_hz,
    )
    peak_vel = float(smooth["velD"].iloc[peak_idx])

    # Build a search window that starts before the main rise and ends before/around peak.
    rise_threshold = max(10.0, peak_vel * 0.22)
    vel_arr = smooth["velD"].to_numpy(dtype=float)
    above_rise = vel_arr >= rise_threshold
    if bool(above_rise[peak_idx]):
        rise_run_start = peak_idx
        while rise_run_start > 0 and bool(above_rise[rise_run_start - 1]):
            rise_run_start -= 1
        search_start = max(pre_window, int(rise_run_start - round(sample_rate_hz * 6.0)))
    else:
        search_start = max(pre_window, peak_idx - int(round(sample_rate_hz * 45.0)))
    search_end = max(search_start + min_run + future_window + 1, peak_idx + int(round(sample_rate_hz * 0.4)))
    search_end = min(search_end, len(df) - future_window - 1)

    strict_mask: np.ndarray = np.zeros(len(df), dtype=bool)
    soft_mask: np.ndarray = np.zeros(len(df), dtype=bool)
    candidate_details: dict[int, tuple[float, float, float, float, float, float]] = {}
    max_i = max(search_start + 1, search_end)
    for i in range(search_start, max_i):
        pre_slice = smooth.iloc[i - pre_window : i]
        fut_slice = smooth.iloc[i : i + future_window]
        if pre_slice.empty or fut_slice.empty:
            continue

        pre_vel = float(pre_slice["velD"].median())
        fut_vel = float(fut_slice["velD"].median())
        vel_gain = fut_vel - pre_vel

        pre_hor = float(pre_slice["vHor"].median())
        fut_hor = float(fut_slice["vHor"].median())
        vhor_drop = pre_hor - fut_hor

        pre_drop = float(pre_slice["hDropRate"].median())
        fut_drop = float(fut_slice["hDropRate"].median())
        drop_gain = fut_drop - pre_drop

        acc_value: Any = smooth.at[i, "accVert"]
        acc_now = float(acc_value)

        cond_speed = fut_vel >= max(10.0, pre_vel + 8.0) and vel_gain >= 8.0
        cond_acc = acc_now >= 2.2
        cond_drop = fut_drop >= max(4.0, pre_drop + 2.0) and drop_gain >= 2.0
        cond_vhor = vhor_drop >= max(4.0, pre_hor * 0.08)

        if cond_speed and cond_acc and cond_drop:
            soft_mask[i] = True
            candidate_details[i] = (fut_vel, vel_gain, acc_now, fut_drop, vhor_drop, pre_hor)
            if cond_vhor:
                strict_mask[i] = True

    first_idx = _first_sustained_index(
        strict_mask,
        start_idx=search_start,
        min_run=min_run,
        end_idx=search_end + 1,
    )
    if first_idx is not None:
        refined_idx = _refine_exit_to_ramp_start(
            vel_d_smooth=smooth["velD"].to_numpy(dtype=float),
            t_abs_s=t_abs_s,
            peak_idx=peak_idx,
            detected_idx=first_idx,
            search_start_idx=search_start,
            sample_rate_hz=sample_rate_hz,
        )
        fut_vel, vel_gain, acc_now, fut_drop, vhor_drop, pre_hor = candidate_details[first_idx]
        conf_components = [
            min(max((vel_gain - 8.0) / 12.0, 0.0), 1.0),
            min(max((acc_now - 2.2) / 4.0, 0.0), 1.0),
            min(max((fut_drop - 4.0) / 10.0, 0.0), 1.0),
            min(max((vhor_drop - max(4.0, pre_hor * 0.08)) / 8.0, 0.0), 1.0),
        ]
        confidence = 0.55 + 0.45 * float(np.mean(conf_components))
        uncertainty_s = max(0.2, min(1.2, (min_run / sample_rate_hz) * 0.6))
        reason = (
            f"exit before peak at +{t_abs_s[peak_idx]:.1f}s: velD={fut_vel:.1f}m/s "
            f"(gain {vel_gain:.1f}), accVert={acc_now:.2f}m/s2, "
            f"hDropRate={fut_drop:.1f}m/s, vHorDrop={vhor_drop:.1f}m/s"
        )
        if refined_idx < first_idx:
            dt_shift = float(t_abs_s[first_idx] - t_abs_s[refined_idx])
            reason = f"{reason} | Rampenbeginn auf +{t_abs_s[refined_idx]:.1f}s korrigiert (Shift {dt_shift:.1f}s)."
            confidence = max(0.52, confidence - 0.03)
            uncertainty_s = min(1.5, uncertainty_s + 0.2)
        return refined_idx, min(confidence, 1.0), uncertainty_s, reason

    # Plausibility fallback: same transition criteria but without mandatory vHor drop.
    soft_idx = _first_sustained_index(
        soft_mask,
        start_idx=search_start,
        min_run=min_run,
        end_idx=search_end + 1,
    )
    if soft_idx is not None:
        refined_idx = _refine_exit_to_ramp_start(
            vel_d_smooth=smooth["velD"].to_numpy(dtype=float),
            t_abs_s=t_abs_s,
            peak_idx=peak_idx,
            detected_idx=soft_idx,
            search_start_idx=search_start,
            sample_rate_hz=sample_rate_hz,
        )
        fut_vel, vel_gain, acc_now, fut_drop, vhor_drop, pre_hor = candidate_details[soft_idx]
        dt_to_peak = max(0.0, float(t_abs_s[peak_idx] - t_abs_s[soft_idx]))
        conf_components = [
            min(max((vel_gain - 8.0) / 12.0, 0.0), 1.0),
            min(max((acc_now - 2.2) / 4.0, 0.0), 1.0),
            min(max((fut_drop - 4.0) / 10.0, 0.0), 1.0),
            min(max((peak_vel - 90.0) / 35.0, 0.0), 1.0),
        ]
        confidence = 0.52 + 0.40 * float(np.mean(conf_components))
        if 6.0 <= dt_to_peak <= 45.0:
            confidence += 0.08
        uncertainty_s = max(0.3, min(1.0, (min_run / sample_rate_hz) * 0.8))
        reason = (
            f"exit (plausibility) before peak at +{t_abs_s[peak_idx]:.1f}s: velD={fut_vel:.1f}m/s "
            f"(gain {vel_gain:.1f}), accVert={acc_now:.2f}m/s2, "
            f"hDropRate={fut_drop:.1f}m/s, vHorDrop={vhor_drop:.1f}m/s"
        )
        if refined_idx < soft_idx:
            dt_shift = float(t_abs_s[soft_idx] - t_abs_s[refined_idx])
            reason = f"{reason} | Rampenbeginn auf +{t_abs_s[refined_idx]:.1f}s korrigiert (Shift {dt_shift:.1f}s)."
            confidence = max(0.5, confidence - 0.04)
            uncertainty_s = min(1.6, uncertainty_s + 0.25)
        return refined_idx, min(confidence, 1.0), uncertainty_s, reason

    # Fallback anchored to primary peak window: first velD>=10 before peak.
    above_10_pre_peak = np.where(df["velD"].to_numpy()[: peak_idx + 1] >= 10.0)[0]
    if len(above_10_pre_peak) > 0:
        idx = int(above_10_pre_peak[0])
        vel_at_idx = float(df["velD"].iloc[idx])
        dt_to_peak = max(0.0, float(t_abs_s[peak_idx] - t_abs_s[idx]))
        peak_gain = max(0.0, float(peak_vel - vel_at_idx))
        ramp_confident = peak_vel >= 95.0 and 6.0 <= dt_to_peak <= 50.0 and peak_gain >= 45.0
        if ramp_confident:
            reason = (
                "Fallback: erster velD>=10m/s vor Haupt-Peak mit plausibler Rampenphase "
                f"(peak={peak_vel:.1f}m/s, dt={dt_to_peak:.1f}s, gain={peak_gain:.1f}m/s)."
            )
            return idx, 0.62, 1.0, reason
        return idx, 0.45, 1.5, "Fallback: erster velD>=10m/s vor Haupt-Peak als t_exit."

    above_10 = np.where(df["velD"].to_numpy() >= 10.0)[0]
    if len(above_10) > 0:
        idx = int(above_10[0])
        return idx, 0.35, 1.8, "Fallback: erster velD>=10m/s im Track als t_exit."

    return 0, 0.25, 2.0, "Fallback: kein klarer Exit, erster Sample verwendet."


def _refine_exit_to_ramp_start(
    *,
    vel_d_smooth: np.ndarray,
    t_abs_s: np.ndarray,
    peak_idx: int,
    detected_idx: int,
    search_start_idx: int,
    sample_rate_hz: float,
) -> int:
    """
    Shift a late-but-valid exit detection back to the beginning of the same acceleration ramp.
    This preserves robust peak anchoring while making t=0 closer to real exit.
    """
    if detected_idx <= search_start_idx or detected_idx <= 1:
        return detected_idx

    if peak_idx <= detected_idx:
        return detected_idx

    v_detect = float(vel_d_smooth[detected_idx])
    v_peak = float(vel_d_smooth[peak_idx])
    if not np.isfinite(v_detect) or not np.isfinite(v_peak):
        return detected_idx
    if v_detect < 35.0 or v_peak < 90.0:
        return detected_idx

    seg = vel_d_smooth[search_start_idx : detected_idx + 1]
    if len(seg) < 8:
        return detected_idx

    # Earliest plausible ramp start in the current event window.
    rel_candidates = np.where((seg >= 10.0) & (seg <= 30.0))[0]
    if len(rel_candidates) == 0:
        return detected_idx

    for rel_idx in rel_candidates:
        idx = search_start_idx + int(rel_idx)
        dt_to_detect = float(t_abs_s[detected_idx] - t_abs_s[idx])
        if dt_to_detect < 4.0 or dt_to_detect > 45.0:
            continue

        v_start = float(vel_d_smooth[idx])
        gain_to_detect = v_detect - v_start
        gain_to_peak = v_peak - v_start
        if gain_to_detect < 24.0 or gain_to_peak < 45.0:
            continue

        # Require clear early acceleration in the first ~2s after ramp start.
        t_limit = float(t_abs_s[idx] + 2.0)
        j2 = int(np.searchsorted(t_abs_s, t_limit, side="left"))
        if j2 <= idx or j2 >= len(vel_d_smooth):
            continue
        gain_2s = float(vel_d_smooth[j2] - vel_d_smooth[idx])
        if gain_2s < 8.0:
            continue

        # Mostly rising in the ramp segment (allow tiny local noise).
        ramp_diff = np.diff(vel_d_smooth[idx : detected_idx + 1])
        if len(ramp_diff) < max(4, int(round(sample_rate_hz))):
            continue
        non_decreasing_ratio = float(np.mean(ramp_diff > -0.15))
        if non_decreasing_ratio < 0.75:
            continue

        return idx

    return detected_idx


def _select_primary_peak_index(
    *,
    vel_d_smooth: np.ndarray,
    t_abs_s: np.ndarray,
    sample_rate_hz: float,
) -> int:
    """
    Select the primary freefall peak while ignoring short transient spikes.
    """
    if len(vel_d_smooth) == 0:
        return 0
    if len(vel_d_smooth) != len(t_abs_s):
        return int(np.argmax(vel_d_smooth))

    valid = np.isfinite(vel_d_smooth) & np.isfinite(t_abs_s)
    if not np.any(valid):
        return int(np.argmax(vel_d_smooth))

    vel = vel_d_smooth.copy()
    vel[~np.isfinite(vel)] = 0.0

    p85 = float(np.nanpercentile(vel[valid], 85))
    threshold = max(22.0, min(35.0, p85 * 0.55))
    min_duration_s = max(3.0, 8.0 / max(sample_rate_hz, 1.0))

    dt = np.diff(t_abs_s[valid])
    median_dt = float(np.nanmedian(dt)) if len(dt) else 0.2
    gap_threshold_s = max(1.0, median_dt * 3.0)

    best_score = -1.0
    best_peak_idx = int(np.argmax(vel))

    def _score_segment(seg_start: int, seg_end: int) -> tuple[float, int]:
        if seg_end <= seg_start:
            return -1.0, seg_start
        seg_t = t_abs_s[seg_start : seg_end + 1]
        seg_v = vel[seg_start : seg_end + 1]
        if len(seg_t) < 2:
            return -1.0, seg_start
        seg_dt = np.diff(seg_t)
        effective_duration = float(np.sum(np.minimum(seg_dt, gap_threshold_s)))
        if effective_duration < min_duration_s:
            return -1.0, seg_start
        local_peak_offset = int(np.argmax(seg_v))
        local_peak_idx = seg_start + local_peak_offset
        local_peak = float(seg_v[local_peak_offset])
        score = effective_duration * min(local_peak, 130.0)
        return score, local_peak_idx

    start_idx: int | None = None
    for i in range(len(vel)):
        above = bool(vel[i] >= threshold)

        if start_idx is not None and i > start_idx:
            gap = float(t_abs_s[i] - t_abs_s[i - 1])
            if gap > gap_threshold_s:
                score, peak_idx = _score_segment(start_idx, i - 1)
                if score > best_score:
                    best_score = score
                    best_peak_idx = peak_idx
                start_idx = None

        if above and start_idx is None:
            start_idx = i
            continue
        if above:
            continue
        if start_idx is None:
            continue
        score, peak_idx = _score_segment(start_idx, i - 1)
        if score > best_score:
            best_score = score
            best_peak_idx = peak_idx
        start_idx = None

    if start_idx is not None:
        score, peak_idx = _score_segment(start_idx, len(vel) - 1)
        if score > best_score:
            best_peak_idx = peak_idx

    return int(best_peak_idx)


def _calc_derived(df: pd.DataFrame, t_abs_s: np.ndarray, t0_abs_s: float, ground_elevation_m: float | None) -> pd.DataFrame:
    out = df.copy()
    out["t_rel_s"] = t_abs_s - t0_abs_s
    out["vVert_mps"] = out["velD"]
    out["vVert_kmh"] = out["vVert_mps"] * 3.6
    out["vHor_mps"] = np.sqrt(out["velN"] ** 2 + out["velE"] ** 2)
    out["vHor_kmh"] = out["vHor_mps"] * 3.6
    out["vTotal_mps"] = np.sqrt(out["velN"] ** 2 + out["velE"] ** 2 + out["velD"] ** 2)
    out["vTotal_kmh"] = out["vTotal_mps"] * 3.6
    out["angle_deg"] = np.degrees(np.arctan2(np.abs(out["velD"]), np.maximum(out["vHor_mps"], 1e-6)))
    out["accVert_mps2"] = np.gradient(out["vVert_mps"].to_numpy(), t_abs_s)
    out["hAGL_m"] = np.nan if ground_elevation_m is None else out["hMSL"] - ground_elevation_m
    return out


def _time_weighted_window_mean(
    t_rel: np.ndarray,
    values: np.ndarray,
    start_s: float,
    end_s: float,
    step_s: float = SCORING_GRID_STEP_S,
) -> float:
    if end_s <= start_s or len(t_rel) < 2:
        return float("nan")
    interval_count = max(1, int(round((end_s - start_s) / step_s)))
    grid = np.linspace(start_s, end_s, interval_count + 1, dtype=float)
    series = np.interp(grid, t_rel, values)
    return float(np.trapezoid(series, grid) / (end_s - start_s))


def _gap_intervals(t_rel: np.ndarray) -> list[tuple[float, float]]:
    finite = np.asarray(t_rel, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        return []
    dt = np.diff(finite)
    median_dt = float(np.nanmedian(dt))
    threshold = max(median_dt * 2.5, 0.6)
    return [
        (float(finite[idx]), float(finite[idx + 1]))
        for idx, delta in enumerate(dt)
        if float(delta) > threshold
    ]


def _window_crosses_gap(*, start_s: float, end_s: float, gaps: list[tuple[float, float]]) -> bool:
    return any(gap_start < end_s and gap_end > start_s for gap_start, gap_end in gaps)


def _best_3s_window(df: pd.DataFrame, start_limit: float, end_limit: float | None = None) -> WindowResult | None:
    if df.empty:
        return None
    ordered = df.sort_values("t_rel_s")
    t_rel = ordered["t_rel_s"].to_numpy(dtype=float)
    if len(t_rel) < 2:
        return None
    available_start = max(float(t_rel[0]), float(start_limit))
    available_end = float(t_rel[-1]) if end_limit is None else min(float(t_rel[-1]), float(end_limit))
    duration = float(SCORING_WINDOW_DURATION_S)
    step = float(SCORING_GRID_STEP_S)
    first_start = float(np.ceil((available_start - 1e-9) / step) * step)
    last_start = float(np.floor((available_end - duration + 1e-9) / step) * step)
    if last_start < first_start:
        return None
    vvert = ordered["vVert_mps"].to_numpy(dtype=float)
    vhor = ordered["vHor_kmh"].to_numpy(dtype=float)
    angle = ordered["angle_deg"].to_numpy(dtype=float)
    gaps = _gap_intervals(t_rel)
    candidate_count = int(round((last_start - first_start) / step)) + 1
    interval_count = max(1, int(round(duration / step)))
    grid = np.linspace(
        first_start,
        last_start + duration,
        candidate_count + interval_count,
        dtype=float,
    )
    grid_step = duration / interval_count

    def _all_window_means(values: np.ndarray) -> np.ndarray:
        interpolated = np.interp(grid, t_rel, values)
        finite_segments = np.isfinite(interpolated[:-1]) & np.isfinite(interpolated[1:])
        segment_areas = np.where(
            finite_segments,
            (interpolated[:-1] + interpolated[1:]) * 0.5 * grid_step,
            0.0,
        )
        area_prefix = np.concatenate(([0.0], np.cumsum(segment_areas)))
        invalid_prefix = np.concatenate(([0], np.cumsum(~finite_segments)))
        means = (
            area_prefix[interval_count : interval_count + candidate_count]
            - area_prefix[:candidate_count]
        ) / duration
        invalid_counts = (
            invalid_prefix[interval_count : interval_count + candidate_count]
            - invalid_prefix[:candidate_count]
        )
        means[invalid_counts > 0] = np.nan
        return means

    starts = np.round(first_start + np.arange(candidate_count, dtype=float) * step, 10)
    valid = np.ones(candidate_count, dtype=bool)
    for gap_start, gap_end in gaps:
        valid &= ~((gap_start < starts + duration) & (gap_end > starts))

    mean_vvert = _all_window_means(vvert)
    valid &= np.isfinite(mean_vvert)
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return None

    # np.argmax returns the first maximum and therefore preserves the existing
    # deterministic tie-breaking rule (earliest scoring window).
    best_idx = int(valid_indices[int(np.argmax(mean_vvert[valid_indices]))])
    mean_vhor = _all_window_means(vhor)
    mean_angle = _all_window_means(angle)
    start = float(starts[best_idx])
    avg_vvert_mps = float(mean_vvert[best_idx])
    return WindowResult(
        start_s=start,
        end_s=float(round(start + duration, 10)),
        avg_vvert_mps=avg_vvert_mps,
        avg_vvert_kmh=float(avg_vvert_mps * 3.6),
        avg_vhor_kmh=float(mean_vhor[best_idx]),
        avg_angle_deg=float(mean_angle[best_idx]),
    )


def _slice_analysis_window(
    post: pd.DataFrame,
    *,
    start_s: float,
    end_s: float,
) -> pd.DataFrame:
    if post.empty:
        return post
    start = float(max(0.0, start_s))
    end = float(max(start, end_s))
    window = post[(post["t_rel_s"] >= start) & (post["t_rel_s"] <= end)].copy()
    return window if not window.empty else post


def _fixpoints(df: pd.DataFrame) -> list[dict[str, Any]]:
    t = df["t_rel_s"].to_numpy()
    vvert = df["vVert_kmh"].to_numpy()
    vhor = df["vHor_kmh"].to_numpy()
    angle = df["angle_deg"].to_numpy()
    alt = df["hAGL_m"].to_numpy()

    points: list[dict[str, Any]] = []
    for sec in FIXPOINT_SECONDS:
        item = {
            "t_rel_s": sec,
            "vVert_kmh": _safe_interp(t, vvert, sec),
            "vHor_kmh": _safe_interp(t, vhor, sec),
            "angle_deg": _safe_interp(t, angle, sec),
            "hAGL_m": _safe_interp(t, alt, sec) if not np.isnan(alt).all() else None,
        }
        points.append(item)
    return points


def _detect_early_end_issue(
    *,
    post: pd.DataFrame,
    curve_window: dict[str, Any],
) -> str | None:
    curve_end = _to_float(curve_window.get("curve_window_end_s"))
    if curve_end is None:
        return None

    t = post["t_rel_s"].to_numpy(dtype=float)
    vvert = post["vVert_kmh"].to_numpy(dtype=float)
    track_end = float(t[-1]) if len(t) else 0.0
    vvert_20 = _safe_interp(t, vvert, 20.0)

    # For speed-jump analysis we require enough usable timeline after exit.
    # Short windows around ~15s are typically malformed tracks or wrong event anchors.
    ends_too_early = curve_end < 18.0
    too_slow_at_20 = vvert_20 is not None and vvert_20 < 120.0

    if not ends_too_early and not too_slow_at_20:
        return None

    parts = [f"Sprung endet zu früh (Kurvenfenster bis +{curve_end:.1f}s)"]
    if vvert_20 is not None:
        parts.append(f"vVert bei +20s nur {vvert_20:.1f} km/h")
    parts.append("Daten für Speed-Analyse nicht belastbar.")
    if track_end < 18.0:
        parts.append("Track selbst endet vor +18s.")
    return " ".join(parts)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _phase_stats(df: pd.DataFrame, spec: dict[str, Any], end_limit_s: float | None = None) -> dict[str, Any]:
    name = str(spec["name"])
    start_s = float(spec["start_s"])
    end_s = float(spec["end_s"])
    if end_limit_s is not None:
        end_s = min(end_s, float(end_limit_s))
    target_low = float(spec["angle_min"])
    target_high = float(spec["angle_max"])
    target_label = f"{target_low:.0f}-{target_high:.0f} Grad"
    if end_s <= start_s + 0.75:
        return {
            "name": name,
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "duration_s": round(max(0.0, end_s - start_s), 2),
            "avg_vVert_kmh": None,
            "avg_vHor_kmh": None,
            "avg_angle_deg": None,
            "angle_start_deg": None,
            "angle_end_deg": None,
            "angle_delta_deg": None,
            "max_vVert_kmh": None,
            "target_angle_label": target_label,
            "target_status": "nicht belastbar",
            "comment": "Nicht genug Daten in dieser Phase.",
        }

    seg = df[(df["t_rel_s"] >= start_s) & (df["t_rel_s"] <= end_s)]
    if seg.empty:
        return {
            "name": name,
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "duration_s": round(max(0.0, end_s - start_s), 2),
            "avg_vVert_kmh": None,
            "avg_vHor_kmh": None,
            "avg_angle_deg": None,
            "angle_start_deg": None,
            "angle_end_deg": None,
            "angle_delta_deg": None,
            "max_vVert_kmh": None,
            "target_angle_label": target_label,
            "target_status": "nicht belastbar",
            "comment": "Nicht genug Daten in dieser Phase.",
        }
    t_values = df["t_rel_s"].to_numpy(dtype=float)
    avg_vvert = _time_weighted_window_mean(
        t_values,
        df["vVert_kmh"].to_numpy(dtype=float),
        start_s,
        end_s,
    )
    avg_vhor = _time_weighted_window_mean(
        t_values,
        df["vHor_kmh"].to_numpy(dtype=float),
        start_s,
        end_s,
    )
    avg_angle = _time_weighted_window_mean(
        t_values,
        df["angle_deg"].to_numpy(dtype=float),
        start_s,
        end_s,
    )
    max_vvert = float(seg["vVert_kmh"].max())
    start_vvert = float(np.interp(start_s, t_values, df["vVert_kmh"].to_numpy(dtype=float)))
    end_vvert = float(np.interp(end_s, t_values, df["vVert_kmh"].to_numpy(dtype=float)))
    start_angle = float(np.interp(start_s, t_values, df["angle_deg"].to_numpy(dtype=float)))
    end_angle = float(np.interp(end_s, t_values, df["angle_deg"].to_numpy(dtype=float)))
    min_angle = float(seg["angle_deg"].min())
    max_angle = float(seg["angle_deg"].max())
    return {
        "name": name,
        "start_s": round(start_s, 2),
        "end_s": round(end_s, 2),
        "duration_s": round(max(0.0, end_s - start_s), 2),
        "avg_vVert_kmh": round(avg_vvert, 2),
        "avg_vHor_kmh": round(avg_vhor, 2),
        "avg_angle_deg": round(avg_angle, 2),
        "angle_start_deg": round(start_angle, 2),
        "angle_end_deg": round(end_angle, 2),
        "angle_delta_deg": round(end_angle - start_angle, 2),
        "start_vVert_kmh": round(start_vvert, 2),
        "end_vVert_kmh": round(end_vvert, 2),
        "vVert_gain_kmh": round(end_vvert - start_vvert, 2),
        "min_angle_deg": round(min_angle, 2),
        "max_angle_deg": round(max_angle, 2),
        "max_vVert_kmh": round(max_vvert, 2),
        "target_angle_label": target_label,
        "target_status": _technical_phase_status(
            avg_angle=avg_angle,
            min_angle=min_angle,
            max_angle=max_angle,
            target_low=target_low,
            target_high=target_high,
        ),
        "comment": _technical_phase_comment(
            name=name,
            avg_vvert=avg_vvert,
            avg_vhor=avg_vhor,
            avg_angle=avg_angle,
            min_angle=min_angle,
            max_angle=max_angle,
            target_low=target_low,
            target_high=target_high,
        ),
    }


def _technical_phase_status(
    *,
    avg_angle: float | None,
    min_angle: float | None,
    max_angle: float | None,
    target_low: float,
    target_high: float,
) -> str:
    if avg_angle is None:
        return "unbekannt"
    avg = float(avg_angle)
    low = float(target_low)
    high = float(target_high)
    max_val = None if max_angle is None else float(max_angle)
    min_val = None if min_angle is None else float(min_angle)
    if avg < low - 1.5:
        return "zu flach"
    if avg > high + 1.0:
        return "zu steil"
    if max_val is not None and max_val > high + 2.0:
        return "kurz zu steil"
    if min_val is not None and min_val < low - 3.0 and avg < low + 0.8:
        return "eher flach"
    return "im Zielbereich"


def _technical_phase_comment(
    *,
    name: str,
    avg_vvert: float | None,
    avg_vhor: float | None,
    avg_angle: float | None,
    min_angle: float | None,
    max_angle: float | None,
    target_low: float,
    target_high: float,
) -> str:
    if avg_angle is None:
        return "Phase nicht belastbar."
    status = _technical_phase_status(
        avg_angle=avg_angle,
        min_angle=min_angle,
        max_angle=max_angle,
        target_low=target_low,
        target_high=target_high,
    )
    vhor = None if avg_vhor is None else float(avg_vhor)
    vvert = None if avg_vvert is None else float(avg_vvert)

    if status == "kurz zu steil":
        max_val = None if max_angle is None else float(max_angle)
        target_text = f"{float(target_low):.0f}-{float(target_high):.0f} Grad"
        if max_val is not None and vhor is not None and vhor < 28.0:
            return (
                f"Durchschnitt liegt im Zielbereich; Max Winkel {max_val:.1f} Grad liegt bei knapper "
                f"horizontaler Reserve kurz ueber dem Ziel {target_text}."
            )
        if max_val is not None:
            return (
                f"Durchschnitt liegt im Zielbereich; Max Winkel {max_val:.1f} Grad liegt kurz "
                f"ueber dem Ziel {target_text}."
            )
        return "Durchschnitt liegt im Zielbereich; einzelne Messpunkte liegen kurz ueber dem Zielwinkel."
    if status == "zu steil":
        if vhor is not None and vhor < 28.0:
            return "Zu steil bei knapper horizontaler Reserve; Risiko fuer Nachkorrekturen."
        return "Winkel liegt ueber dem Technikmodell; nur sinnvoll, wenn die Linie ruhig bleibt."
    if status in {"zu flach", "eher flach"}:
        if name == "Max-Speed Fenster" and status == "eher flach":
            return "Durchschnitt liegt im Zielwinkel, aber einzelne Abschnitte fallen darunter; Stabilitaet des 3s-Fensters separat pruefen."
        if name == "Dive-Aufbau":
            return "Aufbau bleibt flach; Druck kommt wahrscheinlich spaeter."
        return "Winkel liegt unter dem Technikmodell; Speed-Aufbau kann spaeter fehlen."
    if name == "Max-Speed Fenster":
        if vvert is not None and vhor is not None and vvert >= 390.0 and vhor >= 25.0:
            return "Max-Speed-Fenster ist technisch nutzbar: hoch und noch mit horizontaler Reserve."
        return "Max-Speed-Fenster liegt im Zielwinkel; Stabilitaet des 3s-Fensters separat pruefen."
    return "Phase liegt im technischen Zielbereich."


def _detect_hot_zone(df: pd.DataFrame, best_window: WindowResult | None) -> tuple[float | None, float | None, str, str]:
    post = df[df["t_rel_s"] >= 0]
    if post.empty:
        return None, None, "kritisch", "Keine Post-Exit-Daten."

    peak = float(post["vVert_kmh"].max())
    candidates = post[(post["vVert_kmh"] >= peak * 0.85) & (post["angle_deg"] > 83)]
    if candidates.empty:
        if best_window is None:
            return 18.0, 22.0, "kritisch", "Hot-Zone nur statisch (+18s bis +22s) bestimmt."
        return (
            round(best_window.start_s - 2.0, 2),
            round(best_window.start_s + 2.0, 2),
            "stabil",
            "Hot-Zone um bestes 3s-Fenster gelegt.",
        )

    start = float(candidates["t_rel_s"].min())
    end = float(candidates["t_rel_s"].max())
    segment = post[(post["t_rel_s"] >= start) & (post["t_rel_s"] <= end)]
    if segment.empty:
        return start, end, "stabil", "Hot-Zone gefunden."

    vhor = segment["vHor_kmh"]
    drop_pct = 0.0
    if len(vhor) > 2 and float(vhor.max()) > 1e-6:
        drop_pct = float((vhor.max() - vhor.min()) / vhor.max())
    max_angle = float(segment["angle_deg"].max())

    if drop_pct > 0.4 or (float(vhor.min()) < 25 and max_angle > 87):
        return (
            round(start, 2),
            round(end, 2),
            "kritisch",
            f"vHor sinkt um {drop_pct*100:.0f}% bis {float(vhor.min()):.1f} km/h, Winkel bis {max_angle:.1f} deg.",
        )
    if drop_pct < 0.2 and max_angle <= 86:
        return (
            round(start, 2),
            round(end, 2),
            "sehr gut",
            f"stabile horizontale Komponente, Winkel max {max_angle:.1f} deg.",
        )
    return (
        round(start, 2),
        round(end, 2),
        "stabil",
        f"moderater vHor-Drop {drop_pct*100:.0f}% bei Winkel bis {max_angle:.1f} deg.",
    )


def _negative_risk(df: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    post = df[df["t_rel_s"] >= 0]
    if len(post) < 10:
        return 0.0, {"label": "niedrig", "details": "Zu wenige Daten für Heuristik."}

    vhor = post["vHor_kmh"].to_numpy()
    angle = post["angle_deg"].to_numpy()
    vvert = post["vVert_kmh"].to_numpy()
    time_s = post["t_rel_s"].to_numpy()

    rolling_max = np.empty_like(vhor, dtype=float)
    for idx, current_t in enumerate(time_s):
        lookback_mask = (time_s >= current_t - NEGATIVE_RISK_LOOKBACK_S) & (time_s <= current_t)
        rolling_max[idx] = float(np.max(vhor[lookback_mask]))
    dip_ratio = np.where(rolling_max > 1e-6, (rolling_max - vhor) / rolling_max, 0.0)
    dip_idx = int(np.argmax(dip_ratio))
    dip_value = float(dip_ratio[dip_idx])

    rebound = 0.0
    if dip_idx < len(vhor) - 1:
        future_mask = (time_s > time_s[dip_idx]) & (
            time_s <= time_s[dip_idx] + NEGATIVE_RISK_LOOKBACK_S
        )
        future_max = float(np.max(vhor[future_mask])) if bool(np.any(future_mask)) else float(vhor[dip_idx])
        if vhor[dip_idx] > 1e-6:
            rebound = (future_max - float(vhor[dip_idx])) / float(vhor[dip_idx])

    local_mask = (time_s >= time_s[dip_idx] - 0.4) & (time_s <= time_s[dip_idx] + 0.8)
    angle_near_vertical = float(np.max(angle[local_mask]))
    local_vvert = vvert[local_mask]
    vvert_unrest = float(np.std(local_vvert)) if len(local_vvert) else 0.0

    score = 0.0
    if dip_value > 0.35:
        score += 35
    if float(vhor[dip_idx]) < 30:
        score += 20
    if rebound > 0.25:
        score += 20
    if angle_near_vertical > 87:
        score += 20
    if vvert_unrest > 12:
        score += 10

    details = (
        f"vHor-Dip {dip_value*100:.0f}%, vHor-Min {float(vhor[dip_idx]):.1f} km/h, "
        f"Rebound {rebound*100:.0f}%, Winkel-Max {angle_near_vertical:.1f} deg."
    )

    late_mask = (time_s >= 18.0) & (time_s <= 28.0)
    if bool(np.any(late_mask)):
        late_indices = np.where(late_mask)[0]
        late_min_idx = int(late_indices[int(np.argmin(vhor[late_mask]))])
        min_t = float(time_s[late_min_idx])
        min_vhor = float(vhor[late_min_idx])
        pre_mask = (time_s >= max(8.0, min_t - 8.0)) & (time_s <= min_t)
        pre_max = float(np.max(vhor[pre_mask])) if bool(np.any(pre_mask)) else min_vhor
        future_mask = (time_s > min_t) & (time_s <= min(28.0, min_t + 7.0))
        future_max = float(np.max(vhor[future_mask])) if bool(np.any(future_mask)) else min_vhor
        late_dip = (pre_max - min_vhor) / pre_max if pre_max > 1e-6 else 0.0
        late_rebound = (future_max - min_vhor) / min_vhor if min_vhor > 1e-6 else 0.0
        near_mask = (time_s >= min_t - 1.5) & (time_s <= min_t + 1.5)
        late_angle_max = float(np.max(angle[near_mask])) if bool(np.any(near_mask)) else float(angle[late_min_idx])
        late_score = 0.0
        if late_dip > 0.35:
            late_score += 30
        if min_vhor < 30.0:
            late_score += 25
        if late_rebound > 0.25:
            late_score += 25
        if late_angle_max > 86.0:
            late_score += 20
        if late_score >= min(score, 100.0):
            score = late_score
            details = (
                f"vHor-Dip {late_dip*100:.0f}%, vHor-Min {min_vhor:.1f} km/h bei +{min_t:.1f}s, "
                f"Rebound {late_rebound*100:.0f}%, Winkel-Max {late_angle_max:.1f} deg."
            )

    score = min(score, 100.0)
    label = "niedrig"
    if score >= 65:
        label = "hoch"
    elif score >= 35:
        label = "mittel"

    return score, {"label": label, "details": details}


def _score_band(angle: float | None, t_sec: float) -> tuple[str, str]:
    if angle is None:
        return "unbekannt", "keine Daten"
    for band in TARGET_ANGLE_BANDS:
        if band["start_s"] <= t_sec < band["end_s"]:
            if angle < band["min_deg"]:
                return "zu flach", band["label"]
            if angle > band["max_deg"]:
                return "zu steil", band["label"]
            return "optimal", band["label"]
    return "unbekannt", "keine Zielzone"


def _compute_exit_profile(
    *,
    post: pd.DataFrame,
    sample_rate_hz: float,
    fixpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    fp10 = next((p for p in fixpoints if p["t_rel_s"] == 10.0), None)
    vvert10 = None if fp10 is None else fp10.get("vVert_kmh")

    early_acc = post[(post["t_rel_s"] >= 0.0) & (post["t_rel_s"] <= 5.0)]["accVert_mps2"].to_numpy(dtype=float)
    early_acc = early_acc[np.isfinite(early_acc)]

    acc_mean = None
    acc_peak = None
    acc_p95_abs = None
    if len(early_acc) > 0:
        acc_mean = float(np.mean(early_acc))
        acc_peak = float(np.max(np.abs(early_acc)))
        acc_p95_abs = float(np.percentile(np.abs(early_acc), 95))

    early_track = post[(post["t_rel_s"] >= 0.0) & (post["t_rel_s"] <= 8.0)].copy()
    early_turns_10s = 0.0
    if len(early_track) >= 8:
        vvert = early_track["vVert_kmh"].to_numpy(dtype=float)
        smooth_window = max(3, int(round(max(sample_rate_hz, 1.0) * 0.5)))
        vvert_smooth = pd.Series(vvert).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
        turns = _turn_count(vvert_smooth, eps=4.0)
        duration = max(1.0, float(early_track["t_rel_s"].iloc[-1] - early_track["t_rel_s"].iloc[0]))
        early_turns_10s = float(turns / duration * 10.0)

    unsteady = False
    if early_turns_10s > 3.0:
        unsteady = True
    elif acc_p95_abs is not None and acc_p95_abs > 18.0:
        unsteady = True
    elif acc_p95_abs is not None and acc_p95_abs > 14.0 and early_turns_10s > 1.5:
        unsteady = True

    label = "neutral"
    if vvert10 is not None and vvert10 < 230:
        label = "zu_langsam"
    elif vvert10 is not None and vvert10 > 300 and not unsteady:
        label = "dynamisch_stabil"
    elif vvert10 is not None and vvert10 > 300 and unsteady:
        label = "dynamisch_unruhig"
    elif unsteady:
        label = "unruhig"
    else:
        label = "sauber"

    details = (
        f"vVert@10={vvert10 if vvert10 is None else round(float(vvert10), 1)} km/h, "
        f"accMean0_5={acc_mean if acc_mean is None else round(acc_mean, 2)} m/s2, "
        f"accP95Abs0_5={acc_p95_abs if acc_p95_abs is None else round(acc_p95_abs, 2)} m/s2, "
        f"turnsVvert10s={round(early_turns_10s, 2)}"
    )
    return {
        "label": label,
        "unsteady": unsteady,
        "vvert10_kmh": None if vvert10 is None else float(vvert10),
        "early_acc_mean_0_5": acc_mean,
        "early_acc_peak_abs_0_5": acc_peak,
        "early_acc_p95_abs_0_5": acc_p95_abs,
        "early_vvert_turns_10s": early_turns_10s,
        "details": details,
    }


def _turn_count(values: np.ndarray, *, eps: float) -> int:
    if len(values) < 3:
        return 0
    turns = 0
    prev_sign = 0
    for i in range(1, len(values)):
        d = float(values[i] - values[i - 1])
        sign = 1 if d > eps else -1 if d < -eps else 0
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            turns += 1
        prev_sign = sign
    return turns


def _build_scorecard(
    fixpoints: list[dict[str, Any]],
    hot_zone_label: str,
    negative_label: str,
    best_window: WindowResult,
    exit_profile: dict[str, Any],
    phases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fp10 = next((p for p in fixpoints if p["t_rel_s"] == 10.0), None)
    fp20 = next((p for p in fixpoints if p["t_rel_s"] == 20.0), None)

    vvert10 = None if not fp10 else fp10.get("vVert_kmh")
    exit_state = "sauber"
    if vvert10 is None:
        exit_state = "verzoegert"
    elif vvert10 < 230:
        exit_state = "zu langsam"
    else:
        dynamic_fast = bool(vvert10 > 300)
        unsteady = bool(exit_profile.get("unsteady"))
        if dynamic_fast and unsteady:
            exit_state = "aggressiv"
        else:
            exit_state = "sauber"

    build_state = "gut"
    if fp10 and fp20 and fp10["vVert_kmh"] and fp20["vVert_kmh"]:
        gain = fp20["vVert_kmh"] - fp10["vVert_kmh"]
        if gain < 90:
            build_state = "zu langsam"
        elif gain > 200:
            build_state = "zu aggressiv"

    angle_state = "optimal"
    if fp20:
        angle_state, _ = _score_band(fp20.get("angle_deg"), 20.0)

    return {
        "exit": exit_state,
        "exit_dynamik": exit_profile.get("label"),
        "aufbau_0_10": build_state,
        "phase_10_20": angle_state,
        "hot_zone": hot_zone_label,
        "three_second_speed_kmh": round(best_window.avg_vvert_kmh, 2),
        "winkel": angle_state,
        "kipp_risiko": negative_label,
        "technical_phase_statuses": {
            str(row.get("name") or ""): str(row.get("target_status") or "")
            for row in (phases or [])
            if isinstance(row, dict) and str(row.get("name") or "").strip()
        },
    }


def _generate_tips(
    fixpoints: list[dict[str, Any]],
    hot_zone_label: str,
    neg_details: dict[str, Any],
    scorecard: dict[str, Any],
) -> list[str]:
    tips: list[str] = []
    fp10 = next((p for p in fixpoints if p["t_rel_s"] == 10.0), None)
    fp20 = next((p for p in fixpoints if p["t_rel_s"] == 20.0), None)
    fp24 = next((p for p in fixpoints if p["t_rel_s"] == 24.0), None)

    if fp10 and fp10["vVert_kmh"] is not None and fp10["vVert_kmh"] < 230:
        tips.append("Bis +10s früher in die stabile Position gehen. Der Speed-Aufbau startet zu spät.")
    if fp20 and fp20["angle_deg"] is not None and fp20["angle_deg"] < 80:
        tips.append("Zwischen +10s und +20s etwas steiler fliegen. Zielbereich: 80 bis 85 Grad.")
    if fp20 and fp20["angle_deg"] is not None and fp20["angle_deg"] > 87:
        tips.append("Zwischen +10s und +20s etwas flacher bleiben. Zu steil kostet oft Stabilität.")
    if hot_zone_label == "kritisch":
        tips.append("In der schnellen Phase ruhiger Druck halten, damit vHor nicht so stark einbricht.")
    if neg_details["label"] in {"mittel", "hoch"}:
        tips.append("Wenn vHor einbricht, Körperspannung in Schulter und Hüfte früher stabilisieren.")
    if fp24 and fp24["vHor_kmh"] is not None and fp24["vHor_kmh"] < 25:
        tips.append("Ab +22s leicht gegensteuern, damit vHor über 25 km/h bleibt.")
    if not tips:
        tips = [
            "Das Profil ist stabil. Ziel: diesen Ablauf so wiederholbar wie möglich machen.",
            "Den Aufbau zwischen +10s und +20s weiter ruhig und konstant halten.",
        ]

    return tips[:5]


def _first_upward_threshold_crossing_time(
    *,
    time_s: np.ndarray,
    values: np.ndarray,
    threshold: float,
    minimum_time_s: float = 0.0,
) -> float | None:
    candidates = np.where((time_s >= minimum_time_s) & (values >= threshold))[0]
    if not len(candidates):
        return None
    idx = int(candidates[0])
    if idx <= 0 or values[idx - 1] >= threshold or time_s[idx - 1] < minimum_time_s:
        return float(time_s[idx])
    value_delta = float(values[idx] - values[idx - 1])
    if abs(value_delta) < 1e-9:
        return float(time_s[idx])
    ratio = (threshold - float(values[idx - 1])) / value_delta
    return float(time_s[idx - 1] + ratio * (time_s[idx] - time_s[idx - 1]))


def _first_altitude_crossing_time(
    *,
    time_s: np.ndarray,
    altitude_m: np.ndarray,
    start_s: float,
    target_altitude_m: float,
) -> float | None:
    if len(time_s) < 2:
        return None
    start_altitude = float(np.interp(start_s, time_s, altitude_m))
    if start_altitude <= target_altitude_m:
        return float(start_s)
    candidates = np.where((time_s >= start_s) & (altitude_m <= target_altitude_m))[0]
    if not len(candidates):
        return None
    idx = int(candidates[0])
    previous_time = float(start_s) if idx <= 0 or time_s[idx - 1] < start_s else float(time_s[idx - 1])
    previous_altitude = (
        start_altitude
        if idx <= 0 or time_s[idx - 1] < start_s
        else float(altitude_m[idx - 1])
    )
    current_time = float(time_s[idx])
    current_altitude = float(altitude_m[idx])
    altitude_delta = current_altitude - previous_altitude
    if abs(altitude_delta) < 1e-9:
        return current_time
    ratio = (target_altitude_m - previous_altitude) / altitude_delta
    return float(previous_time + ratio * (current_time - previous_time))


def _analysis_signature(
    *,
    ground_elevation_m: float,
    ground_elevation_source: str,
    breakoff_altitude_agl_m: float,
    manual_t0_utc: str | None,
) -> str:
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "ground_elevation_m": round(float(ground_elevation_m), 3),
        "ground_elevation_source": str(ground_elevation_source),
        "breakoff_altitude_agl_m": round(float(breakoff_altitude_agl_m), 3),
        "manual_t0_utc": str(manual_t0_utc or ""),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rule_score_assessment(
    *,
    rule_best: WindowResult | None,
    ground_estimated: bool,
    performance_window_complete: bool,
    validation_quality: float | None,
    validation_missing_accuracy: bool,
    validation_failed: bool,
    quality_flags: list[str],
) -> tuple[str, list[str]]:
    invalid_reasons: list[str] = []
    estimated_reasons: list[str] = []
    if rule_best is None:
        invalid_reasons.append("NO_RULE_WINDOW")
    if not performance_window_complete:
        invalid_reasons.append("PERFORMANCE_WINDOW_INCOMPLETE")
    if validation_failed:
        invalid_reasons.append("VALIDATION_ACCURACY_FAILED")
    if ground_estimated:
        estimated_reasons.append("GROUND_LEVEL_ESTIMATED")
    if validation_quality is None or validation_missing_accuracy:
        estimated_reasons.append("VALIDATION_ACCURACY_UNKNOWN")

    invalid_flag_set = set(quality_flags) & (set(REFERENCE_HARD_FLAGS) | {"LOW_SAMPLE_RATE"})
    invalid_reasons.extend(sorted(invalid_flag_set))
    if invalid_reasons:
        return RULE_SCORE_INVALID, sorted(set(invalid_reasons + estimated_reasons))
    if estimated_reasons:
        return RULE_SCORE_ESTIMATED, sorted(set(estimated_reasons))
    return RULE_SCORE_VALID, []


def prepare_flysight_csv(content: bytes) -> PreparedFlySightTrack:
    """Parse, normalize and validate a FlySight track exactly once per upload."""
    df = _read_csv(content)
    df, unit_normalization = _normalize_import_units(df)
    df, leading_gap_repair = _repair_leading_orphan_samples(df)
    _validate_plausible_jump_profile(df)
    device_type = _detect_device_type(df)
    raw_start_time_utc = df["time"].iloc[0].isoformat()
    t_abs_s = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy()
    dt = np.diff(t_abs_s)
    sample_rate_hz = float(1.0 / np.median(dt)) if len(dt) else 0.0
    quality_flags, quality_score = _compute_quality_flags(df, sample_rate_hz, dt)
    auto_t0_idx, auto_t0_confidence, auto_t0_uncertainty_s, auto_t0_reason = _detect_t0(
        df,
        t_abs_s,
        sample_rate_hz,
    )
    return PreparedFlySightTrack(
        df=df,
        unit_normalization=unit_normalization,
        leading_gap_repair=leading_gap_repair,
        device_type=device_type,
        raw_start_time_utc=raw_start_time_utc,
        t_abs_s=t_abs_s,
        dt=dt,
        sample_rate_hz=sample_rate_hz,
        quality_flags=tuple(quality_flags),
        quality_score=quality_score,
        auto_t0_idx=auto_t0_idx,
        auto_t0_confidence=auto_t0_confidence,
        auto_t0_uncertainty_s=auto_t0_uncertainty_s,
        auto_t0_reason=auto_t0_reason,
        auto_t0_utc=df["time"].iloc[auto_t0_idx].isoformat(),
    )


def _resolve_prepared_t0(
    prepared: PreparedFlySightTrack,
    manual_t0_utc: str | None,
) -> tuple[int, float, str, bool, float, float, str]:
    if manual_t0_utc is not None and str(manual_t0_utc).strip():
        t0_abs_s, t0_utc = _manual_t0_from_utc(
            manual_t0_utc=str(manual_t0_utc),
            raw_start_time=prepared.df["time"].iloc[0],
            t_abs_s=prepared.t_abs_s,
        )
        t0_idx = int(np.searchsorted(prepared.t_abs_s, t0_abs_s, side="left"))
        if t0_idx >= len(prepared.t_abs_s):
            t0_idx = len(prepared.t_abs_s) - 1
        return (
            t0_idx,
            t0_abs_s,
            t0_utc,
            True,
            1.0,
            0.0,
            "Manuell gesetzter Absprungzeitpunkt; automatische Erkennung wurde nur als Referenz gespeichert.",
        )
    return (
        prepared.auto_t0_idx,
        float(prepared.t_abs_s[prepared.auto_t0_idx]),
        prepared.auto_t0_utc,
        False,
        prepared.auto_t0_confidence,
        prepared.auto_t0_uncertainty_s,
        prepared.auto_t0_reason,
    )


def build_ground_observation_samples(
    prepared: PreparedFlySightTrack,
    *,
    manual_t0_utc: str | None = None,
) -> list[dict[str, Any]]:
    """Build only the fields needed for dropzone matching from a prepared track."""
    _, t0_abs_s, _, _, _, _, _ = _resolve_prepared_t0(prepared, manual_t0_utc)
    df = prepared.df
    time_rel = prepared.t_abs_s - t0_abs_s
    vel_n = df["velN"].to_numpy(dtype=float)
    vel_e = df["velE"].to_numpy(dtype=float)
    vel_d = df["velD"].to_numpy(dtype=float)
    total_speed_kmh = np.sqrt(vel_n**2 + vel_e**2 + vel_d**2) * 3.6
    time_values = df["time"].tolist()
    lat_values = df["lat"].to_numpy()
    lon_values = df["lon"].to_numpy()
    altitude_values = df["hMSL"].to_numpy()
    vacc_values = df["vAcc"].to_numpy()
    sacc_values = df["sAcc"].to_numpy()
    gps_fix_values = df["gpsFix"].to_numpy()
    satellite_values = df["numSV"].to_numpy()
    rows: list[dict[str, Any]] = []
    for index in range(len(df)):
        rows.append(
            {
                "time_utc": time_values[index].isoformat(),
                "t_rel_s": float(time_rel[index]),
                "lat": lat_values[index],
                "lon": lon_values[index],
                "hMSL_m": altitude_values[index],
                "velD_mps": float(vel_d[index]),
                "vTotal_kmh": float(total_speed_kmh[index]),
                "vAcc": vacc_values[index],
                "sAcc": sacc_values[index],
                "gpsFix": gps_fix_values[index],
                "numSV": satellite_values[index],
            }
        )
    return rows


def analyze_flysight_csv(
    *,
    content: bytes,
    file_name: str,
    jumper_name: str,
    ground_elevation_m: float | None,
    breakoff_altitude_agl_m: float | None,
    manual_t0_utc: str | None = None,
    ground_elevation_source_override: str | None = None,
    prepared_track: PreparedFlySightTrack | None = None,
) -> dict[str, Any]:
    resolved_breakoff_altitude_agl_m = (
        float(DEFAULT_BREAKOFF_ALTITUDE_AGL_M)
        if breakoff_altitude_agl_m is None
        else float(breakoff_altitude_agl_m)
    )
    if not np.isfinite(resolved_breakoff_altitude_agl_m) or resolved_breakoff_altitude_agl_m <= 0.0:
        raise AnalysisError("Breakoff-Höhe muss eine positive Zahl sein.")
    prepared = prepared_track or prepare_flysight_csv(content)
    df = prepared.df
    unit_normalization = prepared.unit_normalization
    leading_gap_repair = prepared.leading_gap_repair
    device_type = prepared.device_type
    raw_start_time_utc = prepared.raw_start_time_utc
    t_abs_s = prepared.t_abs_s
    sample_rate_hz = prepared.sample_rate_hz
    quality_flags = list(prepared.quality_flags)
    quality_score = prepared.quality_score
    auto_t0_confidence = prepared.auto_t0_confidence
    auto_t0_uncertainty_s = prepared.auto_t0_uncertainty_s
    auto_t0_reason = prepared.auto_t0_reason
    auto_t0_utc = prepared.auto_t0_utc
    (
        _t0_idx,
        t0_abs_s,
        t0_utc,
        manual_t0_applied,
        t0_confidence,
        t0_uncertainty_s,
        t0_reason,
    ) = _resolve_prepared_t0(prepared, manual_t0_utc)
    if not manual_t0_applied and (t0_confidence < 0.55 or t0_uncertainty_s > 1.2):
        quality_flags.append("NO_CLEAR_EXIT")

    ground_estimated = False
    ground_elevation_source = str(ground_elevation_source_override or "manual")
    if ground_elevation_m is None:
        ground_elevation_m = float(df["hMSL"].quantile(0.02))
        ground_estimated = True
        ground_elevation_source = "estimated"
        quality_flags.append("NO_GROUND_LEVEL")

    data = _calc_derived(df, t_abs_s, t0_abs_s, ground_elevation_m)
    post = data[data["t_rel_s"] >= 0].copy()
    if post.empty:
        raise AnalysisError("Nach t0 sind keine Samples vorhanden.")

    exit_altitude_msl = float(np.interp(t0_abs_s, t_abs_s, data["hMSL"].to_numpy(dtype=float)))
    hagl_series = data["hAGL_m"].to_numpy(dtype=float)
    exit_altitude_agl = (
        None
        if np.isnan(hagl_series).all()
        else float(np.interp(t0_abs_s, t_abs_s, hagl_series))
    )
    is_valid_altitude = True if exit_altitude_agl is None else exit_altitude_agl <= MAX_VALID_EXIT_ALTITUDE_AGL_M

    if exit_altitude_agl is not None and not is_valid_altitude:
        quality_flags.append("INVALID_EXIT_ALTITUDE")

    best_training = _best_3s_window(post, start_limit=0.0)
    if best_training is None:
        raise AnalysisError("3-Sekunden-Fenster konnte nicht bestimmt werden.")

    relative_time_s = t_abs_s - t0_abs_s
    vel_d = df["velD"].to_numpy(dtype=float)
    performance_window_start_s = _first_upward_threshold_crossing_time(
        time_s=relative_time_s,
        values=vel_d,
        threshold=10.0,
        minimum_time_s=0.0,
    )
    performance_window_start_utc = None
    if performance_window_start_s is not None:
        pw_start_idx = int(np.searchsorted(relative_time_s, performance_window_start_s, side="left"))
        pw_start_idx = min(pw_start_idx, len(df) - 1)
        performance_window_start_utc = df["time"].iloc[pw_start_idx].isoformat()

    performance_window_end_s = None
    validation_window_start_s = None
    validation_window_end_s = None
    window_quality = None
    rule_best = None
    performance_window_complete = False
    validation_missing_accuracy = False
    validation_failed = False

    if performance_window_start_s is not None:
        post_time = post["t_rel_s"].to_numpy(dtype=float)
        post_altitude = post["hAGL_m"].to_numpy(dtype=float)
        start_alt_agl = float(np.interp(performance_window_start_s, post_time, post_altitude))
        if start_alt_agl > resolved_breakoff_altitude_agl_m:
            end_alt_agl = max(
                start_alt_agl - PERFORMANCE_WINDOW_VERTICAL_DROP_M,
                resolved_breakoff_altitude_agl_m,
            )
            crossing_time = _first_altitude_crossing_time(
                time_s=post_time,
                altitude_m=post_altitude,
                start_s=performance_window_start_s,
                target_altitude_m=end_alt_agl,
            )
            if crossing_time is not None:
                performance_window_end_s = float(crossing_time)
                performance_window_complete = True
            else:
                performance_window_end_s = float(post_time[-1])

            validation_window_end_s = performance_window_end_s
            performance_end_altitude = float(np.interp(performance_window_end_s, post_time, post_altitude))
            validation_start_altitude = performance_end_altitude + VALIDATION_WINDOW_VERTICAL_DROP_M
            validation_window_start_s = _first_altitude_crossing_time(
                time_s=post_time,
                altitude_m=post_altitude,
                start_s=performance_window_start_s,
                target_altitude_m=validation_start_altitude,
            )
            if validation_window_start_s is None:
                validation_window_start_s = performance_window_start_s

            validation_window = post[
                (post["t_rel_s"] >= validation_window_start_s)
                & (post["t_rel_s"] <= validation_window_end_s)
            ]
            if not validation_window.empty:
                sacc = pd.to_numeric(validation_window["sAcc"], errors="coerce")
                finite_sacc = sacc[np.isfinite(sacc.to_numpy(dtype=float))]
                validation_missing_accuracy = len(finite_sacc) != len(validation_window)
                if len(finite_sacc):
                    valid_count = int((finite_sacc < MAX_SACC_MPS).sum())
                    window_quality = round(float(valid_count / len(validation_window)), 3)
                    validation_failed = bool((finite_sacc >= MAX_SACC_MPS).any())

            rule_best = _best_3s_window(
                post,
                start_limit=performance_window_start_s,
                end_limit=performance_window_end_s,
            )

    curve_window = detect_curve_window(post, sample_rate_hz=sample_rate_hz)
    fs2_track_summary = _build_fs2_track_summary(
        post=post,
        curve_window=curve_window,
        device_type=device_type,
    )
    effective_curve_end_s = float(curve_window["curve_window_end_s"])
    if performance_window_end_s is not None and performance_window_end_s >= 8.0:
        effective_curve_end_s = min(effective_curve_end_s, float(performance_window_end_s))
    analysis_post = _slice_analysis_window(
        post,
        start_s=float(curve_window["curve_window_start_s"]),
        end_s=effective_curve_end_s,
    )
    gap_eval_end_s = min(25.0, effective_curve_end_s)
    gap_eval_post = _slice_analysis_window(
        post,
        start_s=float(curve_window["curve_window_start_s"]),
        end_s=gap_eval_end_s,
    )
    if "SPEED_SPIKE" in quality_flags and not _has_speed_spike_in_window(gap_eval_post):
        quality_flags = [flag for flag in quality_flags if flag != "SPEED_SPIKE"]
        quality_score = min(100.0, float(quality_score) + 10.0)
    rule_gap_post = (
        _slice_analysis_window(
            post,
            start_s=float(performance_window_start_s),
            end_s=float(performance_window_end_s),
        )
        if performance_window_start_s is not None and performance_window_end_s is not None
        else gap_eval_post
    )
    if _has_time_gaps_in_window(rule_gap_post, time_col="t_rel_s"):
        quality_flags.append("TIME_GAPS")
        quality_score = max(0.0, float(quality_score) - 12.0)
    early_end_reason = _detect_early_end_issue(post=post, curve_window=curve_window)
    analysis_blocked = early_end_reason is not None
    if analysis_blocked:
        quality_flags.append("EARLY_JUMP_END")
        quality_score = max(0.0, float(quality_score) - 35.0)

    quality_flags = sorted(set(quality_flags))
    rule_score_status, rule_score_reasons = _rule_score_assessment(
        rule_best=rule_best,
        ground_estimated=ground_estimated,
        performance_window_complete=performance_window_complete,
        validation_quality=window_quality,
        validation_missing_accuracy=validation_missing_accuracy,
        validation_failed=validation_failed,
        quality_flags=quality_flags,
    )

    fixpoints = _fixpoints(post)

    phase_end_limit = float(analysis_post["t_rel_s"].max())
    phases = [
        _phase_stats(analysis_post, phase_spec, end_limit_s=phase_end_limit)
        for phase_spec in TECHNICAL_PHASE_SPECS
        if float(phase_spec["start_s"]) <= phase_end_limit
    ]

    primary_window = rule_best or best_training
    hot_start, hot_end, hot_label, hot_reason = _detect_hot_zone(analysis_post, primary_window)
    neg_score, neg_details = _negative_risk(analysis_post)

    exit_profile = _compute_exit_profile(post=post, sample_rate_hz=sample_rate_hz, fixpoints=fixpoints)
    scorecard = _build_scorecard(
        fixpoints,
        hot_label,
        neg_details["label"],
        primary_window,
        exit_profile,
        phases=phases,
    )
    tips = _generate_tips(fixpoints, hot_label, neg_details, scorecard)

    post_rows = post.reset_index(drop=True)
    row_count = len(post_rows)
    t_rel = post_rows["t_rel_s"].to_numpy(dtype=float)
    per_sample_flags: list[str] = []
    median_dt = float(np.median(np.diff(t_rel))) if len(t_rel) > 1 else 0.2
    gps_fix_values = post_rows["gpsFix"].to_numpy()
    sacc_values = post_rows["sAcc"].to_numpy()
    numsv_values = post_rows["numSV"].to_numpy()
    for i in range(row_count):
        row_flags: list[str] = []
        if gps_fix_values[i] != 3:
            row_flags.append("LOW_GPS_FIX")
        if sacc_values[i] >= MAX_SACC_MPS:
            row_flags.append("HIGH_SPEED_ACCURACY_ERROR")
        if numsv_values[i] < MIN_NUM_SV:
            row_flags.append("LOW_NUM_SV")
        if i > 0 and (t_rel[i] - t_rel[i - 1]) > max(0.6, median_dt * 2.5):
            row_flags.append("TIME_GAPS")
        per_sample_flags.append(",".join(row_flags))

    jump_id = str(uuid.uuid4())

    sample_records: list[dict[str, Any]] = []
    time_values = post_rows["time"].tolist()
    lat_values = post_rows["lat"].to_numpy()
    lon_values = post_rows["lon"].to_numpy()
    hmsl_values = post_rows["hMSL"].to_numpy()
    hagl_values = post_rows["hAGL_m"].to_numpy()
    veln_values = post_rows["velN"].to_numpy()
    vele_values = post_rows["velE"].to_numpy()
    veld_values = post_rows["velD"].to_numpy()
    vvert_values = post_rows["vVert_kmh"].to_numpy()
    vhor_values = post_rows["vHor_kmh"].to_numpy()
    vtotal_values = post_rows["vTotal_kmh"].to_numpy()
    angle_values = post_rows["angle_deg"].to_numpy()
    acc_values = post_rows["accVert_mps2"].to_numpy()
    hacc_values = post_rows["hAcc"].to_numpy()
    vacc_values = post_rows["vAcc"].to_numpy()
    for i in range(row_count):
        sample_records.append(
            {
                "jump_id": jump_id,
                "time_utc": time_values[i].isoformat(),
                "t_rel_s": float(t_rel[i]),
                "lat": None if pd.isna(lat_values[i]) else float(lat_values[i]),
                "lon": None if pd.isna(lon_values[i]) else float(lon_values[i]),
                "hMSL_m": float(hmsl_values[i]),
                "hAGL_m": None if pd.isna(hagl_values[i]) else float(hagl_values[i]),
                "velN_mps": float(veln_values[i]),
                "velE_mps": float(vele_values[i]),
                "velD_mps": float(veld_values[i]),
                "vVert_kmh": float(vvert_values[i]),
                "vHor_kmh": float(vhor_values[i]),
                "vTotal_kmh": float(vtotal_values[i]),
                "angle_deg": float(angle_values[i]),
                "accVert_mps2": float(acc_values[i]),
                "hAcc": None if pd.isna(hacc_values[i]) else float(hacc_values[i]),
                "vAcc": None if pd.isna(vacc_values[i]) else float(vacc_values[i]),
                "sAcc": None if pd.isna(sacc_values[i]) else float(sacc_values[i]),
                "gpsFix": None if pd.isna(gps_fix_values[i]) else int(gps_fix_values[i]),
                "numSV": None if pd.isna(numsv_values[i]) else int(numsv_values[i]),
                "quality_flags": per_sample_flags[i],
            }
        )

    analysis_signature = _analysis_signature(
        ground_elevation_m=float(ground_elevation_m),
        ground_elevation_source=ground_elevation_source,
        breakoff_altitude_agl_m=resolved_breakoff_altitude_agl_m,
        manual_t0_utc=manual_t0_utc,
    )
    jump_record = {
        "jump_id": jump_id,
        "jumper_name": jumper_name.strip(),
        "file_name": file_name,
        "device_type": device_type,
        "raw_start_time_utc": raw_start_time_utc,
        "t0_utc": t0_utc,
        "exit_altitude_msl_m": round(exit_altitude_msl, 2),
        "exit_altitude_agl_m": None if exit_altitude_agl is None else round(exit_altitude_agl, 2),
        "ground_elevation_m": None if ground_elevation_m is None else round(float(ground_elevation_m), 2),
        "ground_elevation_source": ground_elevation_source,
        "breakoff_altitude_agl_m": round(resolved_breakoff_altitude_agl_m, 2),
        "analysis_version": ANALYSIS_VERSION,
        "analysis_signature": analysis_signature,
        "is_valid_altitude": 1 if is_valid_altitude else 0,
        "sample_rate_hz": round(sample_rate_hz, 3),
        "quality_score": round(quality_score, 2),
        "quality_flags": json.dumps(sorted(set(quality_flags))),
    }

    notes = {
        "t0_confidence": round(t0_confidence, 3),
        "t0_uncertainty_s": round(t0_uncertainty_s, 3),
        "t0_review_required": bool((not manual_t0_applied) and (t0_confidence < 0.55 or t0_uncertainty_s > 1.2)),
        "t0_reason": t0_reason,
        "t0_manual_override": manual_t0_applied,
        "auto_t0_utc": auto_t0_utc,
        "auto_t0_confidence": round(auto_t0_confidence, 3),
        "auto_t0_uncertainty_s": round(auto_t0_uncertainty_s, 3),
        "auto_t0_reason": auto_t0_reason,
        "analysis_blocked": analysis_blocked,
        "analysis_block_reason": early_end_reason,
        "exit_profile": exit_profile,
        "pw_start_utc": performance_window_start_utc,
        "pw_start_s_from_t0": None if performance_window_start_s is None else round(performance_window_start_s, 3),
        "hot_zone_reason": hot_reason,
        "negative_details": neg_details["details"],
        "ground_level_estimated": ground_estimated,
        "ground_elevation_source": ground_elevation_source,
        "breakoff_altitude_agl_m": round(resolved_breakoff_altitude_agl_m, 2),
        "analysis_version": ANALYSIS_VERSION,
        "rule_score_status": rule_score_status,
        "rule_score_reasons": rule_score_reasons,
        "validation_window_start_s": (
            None if validation_window_start_s is None else round(validation_window_start_s, 3)
        ),
        "validation_window_end_s": (
            None if validation_window_end_s is None else round(validation_window_end_s, 3)
        ),
        "agl_note": "AGL approximiert" if ground_estimated else "AGL aus Ground Elevation berechnet",
        "curve_window_start_s": curve_window["curve_window_start_s"],
        "curve_window_end_s": curve_window["curve_window_end_s"],
        "decel_start_s": curve_window["decel_start_s"],
        "canopy_open_s": curve_window["canopy_open_s"],
        "peak_s": curve_window["peak_s"],
        "curve_window_reason": curve_window["curve_window_reason"],
        "unit_normalization": unit_normalization,
        "leading_gap_repair": leading_gap_repair,
    }
    if fs2_track_summary is not None:
        notes["fs2_track_summary"] = fs2_track_summary

    metrics_record = {
        "jump_id": jump_id,
        "best_3s_start_s": round(best_training.start_s, 2),
        "best_3s_end_s": round(best_training.end_s, 2),
        "best_3s_vVert_mps": round(best_training.avg_vvert_mps, 3),
        "best_3s_vVert_kmh": round(best_training.avg_vvert_kmh, 2),
        "best_3s_vHor_kmh": round(best_training.avg_vhor_kmh, 2),
        "best_3s_angle_deg": round(best_training.avg_angle_deg, 2),
        "training_3s_max_from_t0": round(best_training.avg_vvert_kmh, 2),
        "rule_based_3s_score": None if rule_best is None else round(rule_best.avg_vvert_kmh, 2),
        "rule_based_3s_score_mps": None if rule_best is None else round(rule_best.avg_vvert_mps, 3),
        "performance_window_start_s": None if performance_window_start_s is None else round(performance_window_start_s, 2),
        "performance_window_end_s": None if performance_window_end_s is None else round(performance_window_end_s, 2),
        "validation_window_quality": window_quality,
        "validation_window_start_s": (
            None if validation_window_start_s is None else round(validation_window_start_s, 2)
        ),
        "validation_window_end_s": (
            None if validation_window_end_s is None else round(validation_window_end_s, 2)
        ),
        "rule_score_status": rule_score_status,
        "rule_score_reasons": json.dumps(rule_score_reasons),
        "analysis_version": ANALYSIS_VERSION,
        "hot_zone_start_s": hot_start,
        "hot_zone_end_s": hot_end,
        "negative_risk_score": round(neg_score, 2),
        "notes": json.dumps(notes),
        "fixpoints_json": json.dumps(fixpoints),
        "phases_json": json.dumps(phases),
        "scorecard_json": json.dumps(scorecard),
        "tips_json": json.dumps(tips),
        "quality_flags": json.dumps(sorted(set(quality_flags))),
    }

    chart_data = {
        "time_s": [round(float(x), 3) for x in post["t_rel_s"].tolist()],
        "vVert_kmh": [round(float(x), 3) for x in post["vVert_kmh"].tolist()],
        "vHor_kmh": [round(float(x), 3) for x in post["vHor_kmh"].tolist()],
        "angle_deg": [round(float(x), 3) for x in post["angle_deg"].tolist()],
        "hAGL_m": [None if pd.isna(x) else round(float(x), 3) for x in post["hAGL_m"].tolist()],
        "accVert_mps2": [round(float(x), 3) for x in post["accVert_mps2"].tolist()],
        "velN_mps": [round(float(x), 4) for x in post["velN"].tolist()],
        "velE_mps": [round(float(x), 4) for x in post["velE"].tolist()],
    }

    report = {
        "jump": jump_record,
        "metrics": metrics_record,
        "fixpoints": fixpoints,
        "phases": phases,
        "scorecard": scorecard,
        "tips": tips,
        "notes": notes,
        "chart_data": chart_data,
    }
    return {
        "jump_record": jump_record,
        "metrics_record": metrics_record,
        "sample_records": sample_records,
        "report": report,
    }
    SCORING_GRID_STEP_S,
    SCORING_WINDOW_DURATION_S,
