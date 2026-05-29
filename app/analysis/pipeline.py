from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.curve_window import detect_curve_window
from app.config import (
    DEFAULT_BREAKOFF_ALTITUDE_AGL_M,
    FIXPOINT_SECONDS,
    MAX_SACC_MPS,
    MAX_VALID_EXIT_ALTITUDE_AGL_M,
    MIN_NUM_SV,
    MIN_SAMPLE_RATE_HZ,
    PERFORMANCE_WINDOW_VERTICAL_DROP_M,
    REQUIRED_COLUMNS,
    TARGET_ANGLE_BANDS,
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


def _read_csv(content: bytes) -> pd.DataFrame:
    raw = content.decode("utf-8", errors="replace")
    try:
        df = pd.read_csv(StringIO(raw))
    except Exception as exc:  # pragma: no cover - pandas errors are noisy
        raise AnalysisError(f"CSV konnte nicht gelesen werden: {exc}") from exc

    if df.empty:
        raise AnalysisError("CSV enthaelt keine Datenzeilen.")

    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise AnalysisError(f"Pflichtspalten fehlen: {', '.join(missing)}")

    if "lat" not in df.columns:
        df["lat"] = np.nan
    if "lon" not in df.columns:
        df["lon"] = np.nan

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="ISO8601")
    if df["time"].isna().all():
        raise AnalysisError("Keine gueltigen Zeitstempel in Spalte 'time' gefunden.")

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
        raise AnalysisError("Zu wenige gueltige Samples fuer eine robuste Analyse.")

    return df


def _detect_device_type(df: pd.DataFrame) -> str:
    imu_markers = {"ax", "ay", "az", "gx", "gy", "gz", "accX", "accY", "accZ"}
    columns = {str(c).strip() for c in df.columns}
    if columns.intersection(imu_markers):
        return "FlySight 2"
    return "FlySight 1"


def _safe_interp(x: np.ndarray, y: np.ndarray, x_target: float) -> float | None:
    if len(x) < 2 or x_target < x[0] or x_target > x[-1]:
        return None
    return float(np.interp(x_target, x, y))


def _compute_quality_flags(df: pd.DataFrame, sample_rate_hz: float, dt: np.ndarray) -> tuple[list[str], float]:
    flags: list[str] = []
    if sample_rate_hz < MIN_SAMPLE_RATE_HZ:
        flags.append("LOW_SAMPLE_RATE")

    if (df["gpsFix"] != 3).mean() > 0.05:
        flags.append("LOW_GPS_FIX")

    if (df["sAcc"] >= MAX_SACC_MPS).mean() > 0.05:
        flags.append("HIGH_SPEED_ACCURACY_ERROR")

    median_dt = float(np.nanmedian(dt)) if len(dt) else 0.2
    if np.any(dt > max(median_dt * 2.5, 0.6)):
        flags.append("TIME_GAPS")

    altitude_step = df["hMSL"].diff().abs().to_numpy()
    alt_threshold = max(15.0, float(np.nanmedian(altitude_step)) * 8)
    if np.any(altitude_step > alt_threshold):
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


def _first_sustained_index(
    mask: np.ndarray,
    *,
    start_idx: int,
    min_run: int,
) -> int | None:
    run_start: int | None = None
    run_len = 0
    for i in range(start_idx, len(mask)):
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

    candidate_mask = np.zeros(len(df), dtype=bool)
    candidate_details: dict[int, tuple[float, float, float, float, float, float]] = {}
    max_i = len(df) - future_window - 1
    for i in range(pre_window, max_i):
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

        acc_now = float(smooth.at[i, "accVert"])

        cond_speed = fut_vel >= max(10.0, pre_vel + 8.0) and vel_gain >= 8.0
        cond_acc = acc_now >= 2.2
        cond_drop = fut_drop >= max(4.0, pre_drop + 2.0) and drop_gain >= 2.0
        cond_vhor = vhor_drop >= max(4.0, pre_hor * 0.08)

        if cond_speed and cond_acc and cond_drop and cond_vhor:
            candidate_mask[i] = True
            candidate_details[i] = (fut_vel, vel_gain, acc_now, fut_drop, vhor_drop, pre_hor)

    first_idx = _first_sustained_index(
        candidate_mask,
        start_idx=pre_window,
        min_run=min_run,
    )
    if first_idx is not None:
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
            f"exit by sustained transition: velD={fut_vel:.1f}m/s (gain {vel_gain:.1f}), "
            f"accVert={acc_now:.2f}m/s2, hDropRate={fut_drop:.1f}m/s, vHorDrop={vhor_drop:.1f}m/s"
        )
        return first_idx, min(confidence, 1.0), uncertainty_s, reason

    above_10 = np.where(df["velD"].to_numpy() >= 10.0)[0]
    if len(above_10) > 0:
        idx = int(above_10[0])
        return idx, 0.45, 1.5, "Fallback: erster velD>=10m/s als t_exit."

    return 0, 0.25, 2.0, "Fallback: kein klarer Exit, erster Sample verwendet."


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


def _window_mean(
    t_rel: np.ndarray,
    values: np.ndarray,
    start_s: float,
    end_s: float,
    step_s: float = 0.1,
) -> float:
    if end_s <= start_s or len(t_rel) < 2:
        return float("nan")
    grid = np.arange(start_s, end_s + step_s, step_s)
    if len(grid) < 2:
        return float("nan")
    series = np.interp(grid, t_rel, values)
    return float(series.mean())


def _best_3s_window(df: pd.DataFrame, start_limit: float, end_limit: float | None = None) -> WindowResult | None:
    post = df[df["t_rel_s"] >= start_limit]
    if end_limit is not None:
        post = post[post["t_rel_s"] <= end_limit]
    if post.empty:
        return None

    t_rel = post["t_rel_s"].to_numpy()
    if t_rel[-1] - t_rel[0] < 3.0:
        return None

    grid_step = 0.1
    grid = np.arange(t_rel[0], t_rel[-1] + grid_step, grid_step)
    vvert = np.interp(grid, t_rel, post["vVert_mps"].to_numpy())
    vhor = np.interp(grid, t_rel, post["vHor_kmh"].to_numpy())
    angle = np.interp(grid, t_rel, post["angle_deg"].to_numpy())

    w = int(round(3.0 / grid_step))
    if len(grid) <= w:
        return None

    kernel = np.ones(w) / w
    means = np.convolve(vvert, kernel, mode="valid")
    idx = int(np.argmax(means))
    start = float(grid[idx])
    end = float(start + 3.0)

    return WindowResult(
        start_s=start,
        end_s=end,
        avg_vvert_mps=float(means[idx]),
        avg_vvert_kmh=float(means[idx] * 3.6),
        avg_vhor_kmh=_window_mean(grid, vhor, start, end),
        avg_angle_deg=_window_mean(grid, angle, start, end),
    )


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


def _phase_stats(df: pd.DataFrame, name: str, start_s: float, end_s: float) -> dict[str, Any]:
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
            "max_vVert_kmh": None,
            "comment": "Nicht genug Daten in dieser Phase.",
        }
    avg_vvert = float(seg["vVert_kmh"].mean())
    avg_vhor = float(seg["vHor_kmh"].mean())
    avg_angle = float(seg["angle_deg"].mean())
    max_vvert = float(seg["vVert_kmh"].max())
    return {
        "name": name,
        "start_s": round(start_s, 2),
        "end_s": round(end_s, 2),
        "duration_s": round(max(0.0, end_s - start_s), 2),
        "avg_vVert_kmh": round(avg_vvert, 2),
        "avg_vHor_kmh": round(avg_vhor, 2),
        "avg_angle_deg": round(avg_angle, 2),
        "max_vVert_kmh": round(max_vvert, 2),
        "comment": _phase_comment(name, avg_vvert, avg_vhor, avg_angle),
    }


def _phase_comment(name: str, avg_vvert: float, avg_vhor: float, avg_angle: float) -> str:
    if name == "Startphase":
        if avg_vvert < 190:
            return "Aufbau in den ersten Sekunden eher langsam."
        return "Sauberer frueher Speedaufbau."
    if name == "Beschleunigungsphase":
        if avg_angle < 78:
            return "Winkel in der Aufbauphase etwas zu flach."
        if avg_angle > 87 and avg_vhor < 35:
            return "Sehr steil bei geringer horizontaler Stabilisierung."
        return "Beschleunigung kontrolliert."
    if name == "Phase maximale Geschwindigkeit":
        if avg_vhor < 25:
            return "Peak mit starker horizontaler Komponenteinbusse."
        return "Peak-Phase stabil nutzbar."
    if avg_vvert > 380:
        return "Speed am Ende noch hoch - fruehes Management fuer Breakoff planen."
    return "Endphase kontrolliert."


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
            f"vHor-Drop {drop_pct*100:.0f}% bis {float(vhor.min()):.1f} km/h, Winkel bis {max_angle:.1f}°.",
        )
    if drop_pct < 0.2 and max_angle <= 86:
        return (
            round(start, 2),
            round(end, 2),
            "sehr gut",
            f"stabile horizontale Komponente, Winkel max {max_angle:.1f}°.",
        )
    return (
        round(start, 2),
        round(end, 2),
        "stabil",
        f"moderater vHor-Drop {drop_pct*100:.0f}% bei Winkel bis {max_angle:.1f}°.",
    )


def _negative_risk(df: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    post = df[df["t_rel_s"] >= 0]
    if len(post) < 10:
        return 0.0, {"label": "niedrig", "details": "Zu wenige Daten fuer Heuristik."}

    vhor = post["vHor_kmh"].to_numpy()
    angle = post["angle_deg"].to_numpy()
    vvert = post["vVert_kmh"].to_numpy()

    rolling_max = pd.Series(vhor).rolling(12, min_periods=1).max().to_numpy()
    dip_ratio = np.where(rolling_max > 1e-6, (rolling_max - vhor) / rolling_max, 0.0)
    dip_idx = int(np.argmax(dip_ratio))
    dip_value = float(dip_ratio[dip_idx])

    rebound = 0.0
    if dip_idx < len(vhor) - 5:
        future_max = float(np.max(vhor[dip_idx + 1 : dip_idx + 12]))
        if vhor[dip_idx] > 1e-6:
            rebound = (future_max - float(vhor[dip_idx])) / float(vhor[dip_idx])

    angle_near_vertical = float(np.max(angle[max(0, dip_idx - 4) : min(len(angle), dip_idx + 8)]))
    local_vvert = vvert[max(0, dip_idx - 4) : min(len(vvert), dip_idx + 8)]
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
    score = min(score, 100.0)

    label = "niedrig"
    if score >= 65:
        label = "hoch"
    elif score >= 35:
        label = "mittel"

    details = (
        f"vHor-Dip {dip_value*100:.0f}%, vHor-Min {float(vhor[dip_idx]):.1f} km/h, "
        f"Rebound {rebound*100:.0f}%, Winkel-Max {angle_near_vertical:.1f}°."
    )
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


def _build_scorecard(
    fixpoints: list[dict[str, Any]],
    hot_zone_label: str,
    negative_label: str,
    quality_score: float,
    best_window: WindowResult,
) -> dict[str, Any]:
    fp10 = next((p for p in fixpoints if p["t_rel_s"] == 10.0), None)
    fp20 = next((p for p in fixpoints if p["t_rel_s"] == 20.0), None)

    exit_state = "sauber"
    if not fp10 or fp10["vVert_kmh"] is None:
        exit_state = "verzoegert"
    elif fp10["vVert_kmh"] < 230:
        exit_state = "zu langsam"
    elif fp10["vVert_kmh"] > 300:
        exit_state = "aggressiv"

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

    quality_state = "gut"
    if quality_score < 55:
        quality_state = "kritisch"
    elif quality_score < 75:
        quality_state = "eingeschraenkt"

    return {
        "exit": exit_state,
        "aufbau_0_10": build_state,
        "phase_10_20": angle_state,
        "hot_zone": hot_zone_label,
        "three_second_speed_kmh": round(best_window.avg_vvert_kmh, 2),
        "winkel": angle_state,
        "kipp_risiko": negative_label,
        "datenqualitaet": quality_state,
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
        tips.append("Bis +10s frueher in stabile Speed-Position gehen, Aufbau beginnt zu spaet.")
    if fp20 and fp20["angle_deg"] is not None and fp20["angle_deg"] < 80:
        tips.append("Zwischen +10s und +20s Winkel konsequenter in den Bereich 80-85° bringen.")
    if hot_zone_label == "kritisch":
        tips.append("In der Hot-Zone Druck gleichmaessiger halten, um vHor-Dips zu vermeiden.")
    if neg_details["label"] in {"mittel", "hoch"}:
        tips.append("Auf Rebound-Muster nach vHor-Dip achten: Koerperspannung in Schulter/Huefte stabilisieren.")
    if fp24 and fp24["vHor_kmh"] is not None and fp24["vHor_kmh"] < 25:
        tips.append("Ab +22s leichte Winkelkorrektur testen, damit horizontale Stabilisierung nicht kollabiert.")
    if scorecard["datenqualitaet"] != "gut":
        tips.append("GPS-Qualitaet verbessern (saubere Antennenlage, freie Sicht, stabiler Fix vor dem Exit).")

    if not tips:
        tips = [
            "Aktuelles Profil ist stabil. Fokus auf reproduzierbaren Exit und Peak-Haltephase.",
            "Den Aufbau zwischen +10s und +20s konstant halten und Hot-Zone weiterhin ruhig fliegen.",
        ]

    return tips[:5]


def analyze_flysight_csv(
    *,
    content: bytes,
    file_name: str,
    jumper_name: str,
    ground_elevation_m: float | None,
    breakoff_altitude_agl_m: float | None,
) -> dict[str, Any]:
    df = _read_csv(content)
    device_type = _detect_device_type(df)
    raw_start_time_utc = df["time"].iloc[0].isoformat()
    t_abs_s = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy()
    dt = np.diff(t_abs_s)
    sample_rate_hz = float(1.0 / np.median(dt)) if len(dt) else 0.0
    quality_flags, quality_score = _compute_quality_flags(df, sample_rate_hz, dt)

    t0_idx, t0_confidence, t0_uncertainty_s, t0_reason = _detect_t0(df, t_abs_s, sample_rate_hz)
    t0_utc = df["time"].iloc[t0_idx].isoformat()
    t0_abs_s = float(t_abs_s[t0_idx])
    if t0_confidence < 0.55 or t0_uncertainty_s > 1.2:
        quality_flags.append("NO_CLEAR_EXIT")

    ground_estimated = False
    if ground_elevation_m is None:
        ground_elevation_m = float(df["hMSL"].quantile(0.02))
        ground_estimated = True
        quality_flags.append("NO_GROUND_LEVEL")

    data = _calc_derived(df, t_abs_s, t0_abs_s, ground_elevation_m)
    post = data[data["t_rel_s"] >= 0].copy()
    if post.empty:
        raise AnalysisError("Nach t0 sind keine Samples vorhanden.")

    exit_altitude_msl = float(data["hMSL"].iloc[t0_idx])
    exit_altitude_agl = float(data["hAGL_m"].iloc[t0_idx]) if not np.isnan(data["hAGL_m"].iloc[t0_idx]) else None
    is_valid_altitude = True if exit_altitude_agl is None else exit_altitude_agl <= MAX_VALID_EXIT_ALTITUDE_AGL_M

    if exit_altitude_agl is not None and not is_valid_altitude:
        quality_flags.append("INVALID_EXIT_ALTITUDE")

    best_training = _best_3s_window(post, start_limit=0.0)
    if best_training is None:
        raise AnalysisError("3-Sekunden-Fenster konnte nicht bestimmt werden.")

    vel_d = df["velD"].to_numpy()
    pw_candidates = np.where(vel_d >= 10.0)[0]
    pw_candidates = pw_candidates[pw_candidates >= t0_idx]
    pw_start_idx = int(pw_candidates[0]) if len(pw_candidates) else None
    performance_window_start_s = None
    performance_window_start_utc = None
    if pw_start_idx is not None:
        performance_window_start_s = float(t_abs_s[pw_start_idx] - t0_abs_s)
        performance_window_start_utc = df["time"].iloc[pw_start_idx].isoformat()

    performance_window_end_s = None
    window_quality = None
    rule_best = None

    if performance_window_start_s is not None:
        breakoff = breakoff_altitude_agl_m or DEFAULT_BREAKOFF_ALTITUDE_AGL_M
        start_alt_agl = float(np.interp(performance_window_start_s, post["t_rel_s"], post["hAGL_m"]))
        end_alt_agl = max(start_alt_agl - PERFORMANCE_WINDOW_VERTICAL_DROP_M, breakoff)

        below = post[(post["t_rel_s"] >= performance_window_start_s) & (post["hAGL_m"] <= end_alt_agl)]
        if not below.empty:
            performance_window_end_s = float(below["t_rel_s"].iloc[0])
        else:
            performance_window_end_s = float(post["t_rel_s"].max())

        in_window = post[
            (post["t_rel_s"] >= performance_window_start_s) & (post["t_rel_s"] <= performance_window_end_s)
        ]
        if not in_window.empty:
            valid_ratio = (
                ((in_window["gpsFix"] == 3) & (in_window["sAcc"] < MAX_SACC_MPS)).sum() / len(in_window)
            )
            window_quality = round(float(valid_ratio), 3)

        rule_best = _best_3s_window(
            post,
            start_limit=performance_window_start_s,
            end_limit=performance_window_end_s,
        )

    fixpoints = _fixpoints(post)

    start_end = float(min(5.0, max(2.0, best_training.start_s * 0.25)))
    accel_end = float(max(start_end + 1.5, best_training.start_s - 1.5))
    max_end = float(best_training.end_s)
    tail_end = float(post["t_rel_s"].max())

    phases = [
        _phase_stats(post, "Startphase", 0.0, start_end),
        _phase_stats(post, "Beschleunigungsphase", start_end, accel_end),
        _phase_stats(post, "Phase maximale Geschwindigkeit", best_training.start_s, max_end),
        _phase_stats(post, "Endphase", max_end, tail_end),
    ]

    hot_start, hot_end, hot_label, hot_reason = _detect_hot_zone(post, best_training)
    neg_score, neg_details = _negative_risk(post)
    curve_window = detect_curve_window(post, sample_rate_hz=sample_rate_hz)

    scorecard = _build_scorecard(fixpoints, hot_label, neg_details["label"], quality_score, best_training)
    tips = _generate_tips(fixpoints, hot_label, neg_details, scorecard)

    t_rel = post["t_rel_s"].to_numpy()
    per_sample_flags: list[str] = []
    median_dt = float(np.median(np.diff(t_rel))) if len(t_rel) > 1 else 0.2
    for i, row in post.reset_index(drop=True).iterrows():
        row_flags: list[str] = []
        if row.get("gpsFix", 3) != 3:
            row_flags.append("LOW_GPS_FIX")
        if row.get("sAcc", 0.0) >= MAX_SACC_MPS:
            row_flags.append("HIGH_SPEED_ACCURACY_ERROR")
        if row.get("numSV", MIN_NUM_SV) < MIN_NUM_SV:
            row_flags.append("LOW_NUM_SV")
        if i > 0 and (t_rel[i] - t_rel[i - 1]) > max(0.6, median_dt * 2.5):
            row_flags.append("TIME_GAPS")
        per_sample_flags.append(",".join(row_flags))

    jump_id = str(uuid.uuid4())

    sample_records: list[dict[str, Any]] = []
    post_rows = post.reset_index(drop=True)
    for i, row in post_rows.iterrows():
        sample_records.append(
            {
                "jump_id": jump_id,
                "time_utc": row["time"].isoformat(),
                "t_rel_s": float(row["t_rel_s"]),
                "lat": None if pd.isna(row["lat"]) else float(row["lat"]),
                "lon": None if pd.isna(row["lon"]) else float(row["lon"]),
                "hMSL_m": float(row["hMSL"]),
                "hAGL_m": None if pd.isna(row["hAGL_m"]) else float(row["hAGL_m"]),
                "velN_mps": float(row["velN"]),
                "velE_mps": float(row["velE"]),
                "velD_mps": float(row["velD"]),
                "vVert_kmh": float(row["vVert_kmh"]),
                "vHor_kmh": float(row["vHor_kmh"]),
                "vTotal_kmh": float(row["vTotal_kmh"]),
                "angle_deg": float(row["angle_deg"]),
                "accVert_mps2": float(row["accVert_mps2"]),
                "hAcc": None if pd.isna(row["hAcc"]) else float(row["hAcc"]),
                "vAcc": None if pd.isna(row["vAcc"]) else float(row["vAcc"]),
                "sAcc": None if pd.isna(row["sAcc"]) else float(row["sAcc"]),
                "gpsFix": None if pd.isna(row["gpsFix"]) else int(row["gpsFix"]),
                "numSV": None if pd.isna(row["numSV"]) else int(row["numSV"]),
                "quality_flags": per_sample_flags[i],
            }
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
        "is_valid_altitude": 1 if is_valid_altitude else 0,
        "sample_rate_hz": round(sample_rate_hz, 3),
        "quality_score": round(quality_score, 2),
        "quality_flags": json.dumps(sorted(set(quality_flags))),
    }

    notes = {
        "t0_confidence": round(t0_confidence, 3),
        "t0_uncertainty_s": round(t0_uncertainty_s, 3),
        "t0_reason": t0_reason,
        "pw_start_utc": performance_window_start_utc,
        "pw_start_s_from_t0": None if performance_window_start_s is None else round(performance_window_start_s, 3),
        "hot_zone_reason": hot_reason,
        "negative_details": neg_details["details"],
        "ground_level_estimated": ground_estimated,
        "agl_note": "AGL approximiert" if ground_estimated else "AGL aus Ground Elevation berechnet",
        "curve_window_start_s": curve_window["curve_window_start_s"],
        "curve_window_end_s": curve_window["curve_window_end_s"],
        "decel_start_s": curve_window["decel_start_s"],
        "canopy_open_s": curve_window["canopy_open_s"],
        "peak_s": curve_window["peak_s"],
        "curve_window_reason": curve_window["curve_window_reason"],
    }

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
