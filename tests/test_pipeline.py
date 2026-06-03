from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.analysis.pipeline import AnalysisError, analyze_flysight_csv


def _build_synthetic_csv() -> bytes:
    dt = 0.2
    n = 320
    t = np.arange(n) * dt
    base_time = datetime(2026, 5, 29, 10, 0, 0, tzinfo=timezone.utc)
    times = [base_time + timedelta(seconds=float(x)) for x in t]

    vel_d = np.piecewise(
        t,
        [t < 5, (t >= 5) & (t < 22), t >= 22],
        [
            lambda x: np.clip((x - 3) * 4, 0, None),
            lambda x: 8 + (x - 5) * 7.2,
            lambda x: 130 - (x - 22) * 0.9,
        ],
    )
    vel_d = np.clip(vel_d, 0, 132)

    vel_n = np.piecewise(
        t,
        [t < 5, (t >= 5) & (t < 25), t >= 25],
        [
            lambda x: 65 - x * 1.5,
            lambda x: 55 - (x - 5) * 1.9,
            lambda x: 18 + (x - 25) * 0.5,
        ],
    )
    vel_n = np.clip(vel_n, 10, None)
    vel_e = np.full_like(t, 3.0)

    h = [4400.0]
    for i in range(1, len(t)):
        h.append(h[-1] - float(vel_d[i]) * dt)
    h = np.array(h)

    df = pd.DataFrame(
        {
            "time": [ts.isoformat() for ts in times],
            "lat": 50.0 + np.sin(t / 1000) * 0.001,
            "lon": 8.0 + np.cos(t / 1000) * 0.001,
            "hMSL": h,
            "velN": vel_n,
            "velE": vel_e,
            "velD": vel_d,
            "hAcc": 1.5,
            "vAcc": 1.7,
            "sAcc": 0.8,
            "gpsFix": 3,
            "numSV": 13,
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def _build_exit_test_csv(*, unsteady: bool) -> bytes:
    dt = 0.2
    n = 260
    t = np.arange(n) * dt
    base_time = datetime(2026, 5, 29, 12, 0, 0, tzinfo=timezone.utc)
    times = [base_time + timedelta(seconds=float(x)) for x in t]

    if unsteady:
        # Intentionally oscillatory early profile to trigger "aggressiv" (hard/unsteady exit).
        vel_d = np.where(
            t < 6.0,
            18 + t * 20 + 7 * np.sin(t * 9.0),
            np.where(t < 16.0, 95 + (t - 6.0) * 2.4, 119 - (t - 16.0) * 1.1),
        )
    else:
        # Dynamic but smooth early ramp.
        vel_d = np.where(
            t < 6.0,
            18 + t * 14.5,
            np.where(t < 16.0, 105 + (t - 6.0) * 1.8, 123 - (t - 16.0) * 1.0),
        )
    vel_d = np.clip(vel_d, 0, 128)

    vel_n = np.where(t < 16.0, 44 - t * 1.1, 26 - (t - 16.0) * 0.2)
    vel_n = np.clip(vel_n, 12, None)
    vel_e = np.full_like(t, 2.0)

    h = [4450.0]
    for i in range(1, len(t)):
        h.append(h[-1] - float(vel_d[i]) * dt)
    h = np.array(h)

    df = pd.DataFrame(
        {
            "time": [ts.isoformat() for ts in times],
            "lat": 50.0,
            "lon": 8.0,
            "hMSL": h,
            "velN": vel_n,
            "velE": vel_e,
            "velD": vel_d,
            "hAcc": 1.2,
            "vAcc": 1.5,
            "sAcc": 0.8,
            "gpsFix": 3,
            "numSV": 12,
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def _build_leading_gap_csv() -> bytes:
    dt = 0.2
    n = 320
    t = np.arange(n) * dt
    base_time = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)

    # Introduce a leading data gap: 0.0s, then next point at +2.4s, then regular 0.2s.
    t_with_gap = np.array([0.0] + [2.4 + (i - 1) * dt for i in range(1, n)], dtype=float)
    times = [base_time + timedelta(seconds=float(x)) for x in t_with_gap]

    vel_d = np.piecewise(
        t,
        [t < 5, (t >= 5) & (t < 22), t >= 22],
        [
            lambda x: np.clip((x - 3) * 4, 0, None),
            lambda x: 8 + (x - 5) * 7.2,
            lambda x: 130 - (x - 22) * 0.9,
        ],
    )
    vel_d = np.clip(vel_d, 0, 132)

    vel_n = np.piecewise(
        t,
        [t < 5, (t >= 5) & (t < 25), t >= 25],
        [
            lambda x: 65 - x * 1.5,
            lambda x: 55 - (x - 5) * 1.9,
            lambda x: 18 + (x - 25) * 0.5,
        ],
    )
    vel_n = np.clip(vel_n, 10, None)
    vel_e = np.full_like(t, 3.0)

    h = [4400.0]
    for i in range(1, len(t)):
        h.append(h[-1] - float(vel_d[i]) * dt)
    h = np.array(h)

    df = pd.DataFrame(
        {
            "time": [ts.isoformat() for ts in times],
            "lat": 50.0,
            "lon": 8.0,
            "hMSL": h,
            "velN": vel_n,
            "velE": vel_e,
            "velD": vel_d,
            "hAcc": 1.5,
            "vAcc": 1.7,
            "sAcc": 0.8,
            "gpsFix": 3,
            "numSV": 13,
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def _build_kmh_feet_unit_csv() -> bytes:
    dt = 0.2
    n = 320
    t = np.arange(n) * dt
    base_time = datetime(2026, 5, 29, 16, 0, 0, tzinfo=timezone.utc)
    times = [base_time + timedelta(seconds=float(x)) for x in t]

    # Generate plausible profile in m/s + meters first.
    vel_d_mps = np.piecewise(
        t,
        [t < 6, (t >= 6) & (t < 22), t >= 22],
        [
            lambda x: np.clip((x - 3) * 3.2, 0, None),
            lambda x: 7 + (x - 6) * 6.0,
            lambda x: 124 - (x - 22) * 0.8,
        ],
    )
    vel_d_mps = np.clip(vel_d_mps, 0, 128)
    vel_n_mps = np.piecewise(
        t,
        [t < 6, (t >= 6) & (t < 25), t >= 25],
        [
            lambda x: 58 - x * 1.2,
            lambda x: 50 - (x - 6) * 1.7,
            lambda x: 18 + (x - 25) * 0.4,
        ],
    )
    vel_n_mps = np.clip(vel_n_mps, 8, None)
    vel_e_mps = np.full_like(t, 2.5)

    h_m = [4300.0]
    for i in range(1, len(t)):
        h_m.append(h_m[-1] - float(vel_d_mps[i]) * dt)
    h_m = np.array(h_m)

    # Export intentionally in wrong units: km/h and feet.
    vel_d_kmh = vel_d_mps * 3.6
    vel_n_kmh = vel_n_mps * 3.6
    vel_e_kmh = vel_e_mps * 3.6
    h_ft = h_m * 3.28084

    df = pd.DataFrame(
        {
            "time": [ts.isoformat() for ts in times],
            "lat": 50.0,
            "lon": 8.0,
            "hMSL": h_ft,
            "velN": vel_n_kmh,
            "velE": vel_e_kmh,
            "velD": vel_d_kmh,
            "hAcc": 2.0,
            "vAcc": 2.1,
            "sAcc": 0.9,
            "gpsFix": 3,
            "numSV": 13,
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def _build_flysight2_track_csv() -> bytes:
    dt = 0.2
    n = 320
    t = np.arange(n) * dt
    base_time = datetime(2026, 5, 30, 16, 0, 0, tzinfo=timezone.utc)
    times = [base_time + timedelta(seconds=float(x)) for x in t]

    vel_d = np.piecewise(
        t,
        [t < 5, (t >= 5) & (t < 22), t >= 22],
        [
            lambda x: np.clip((x - 3) * 4, 0, None),
            lambda x: 8 + (x - 5) * 7.2,
            lambda x: 130 - (x - 22) * 0.9,
        ],
    )
    vel_d = np.clip(vel_d, 0, 132)
    vel_n = np.piecewise(
        t,
        [t < 5, (t >= 5) & (t < 25), t >= 25],
        [
            lambda x: 65 - x * 1.5,
            lambda x: 55 - (x - 5) * 1.9,
            lambda x: 18 + (x - 25) * 0.5,
        ],
    )
    vel_n = np.clip(vel_n, 10, None)
    vel_e = np.full_like(t, 3.0)

    h = [4400.0]
    for i in range(1, len(t)):
        h.append(h[-1] - float(vel_d[i]) * dt)
    h = np.array(h)

    lines = [
        "$FLYS,1",
        "$VAR,FIRMWARE_VER,v2023.09.22",
        "$COL,GNSS,time,lat,lon,hMSL,velN,velE,velD,hAcc,vAcc,sAcc,numSV",
        "$UNIT,GNSS,,deg,deg,m,m/s,m/s,m/s,m,m,m/s,",
        "$DATA",
    ]
    for i in range(n):
        ts = times[i].isoformat().replace("+00:00", "Z")
        lat = 50.0 + np.sin(t[i] / 1000.0) * 0.001
        lon = 8.0 + np.cos(t[i] / 1000.0) * 0.001
        lines.append(
            "$GNSS,"
            f"{ts},{lat:.7f},{lon:.7f},{h[i]:.3f},{vel_n[i]:.3f},{vel_e[i]:.3f},{vel_d[i]:.3f},"
            "1.50,1.70,0.80,13"
        )
    return "\n".join(lines).encode("utf-8")


def test_pipeline_outputs_core_metrics():
    content = _build_synthetic_csv()
    result = analyze_flysight_csv(
        content=content,
        file_name="synthetic.csv",
        jumper_name="Marlene",
        ground_elevation_m=200.0,
        breakoff_altitude_agl_m=1700.0,
    )

    jump = result["jump_record"]
    metrics = result["metrics_record"]
    fixpoints = result["report"]["fixpoints"]
    phases = result["report"]["phases"]
    notes = result["report"]["notes"]

    assert jump["sample_rate_hz"] >= 4.9
    assert metrics["best_3s_vVert_kmh"] > 430
    assert len(fixpoints) == 5
    assert all(point["vVert_kmh"] is not None for point in fixpoints[:3])
    assert len(phases) == 4
    assert metrics["hot_zone_start_s"] is not None
    assert metrics["negative_risk_score"] >= 0
    assert notes["curve_window_start_s"] == 0.0
    assert notes["curve_window_end_s"] > 10.0
    assert notes["curve_window_end_s"] <= result["report"]["chart_data"]["time_s"][-1]
    assert "t0_confidence" in notes
    assert "t0_uncertainty_s" in notes
    assert "pw_start_s_from_t0" in notes


def test_pipeline_parses_flysight2_track_csv_format():
    result = analyze_flysight_csv(
        content=_build_flysight2_track_csv(),
        file_name="TRACK.CSV",
        jumper_name="Marc",
        ground_elevation_m=200.0,
        breakoff_altitude_agl_m=1700.0,
    )

    jump = result["jump_record"]
    metrics = result["metrics_record"]
    notes = result["report"]["notes"]

    assert jump["device_type"] == "FlySight 2"
    assert jump["sample_rate_hz"] >= 4.8
    assert metrics["best_3s_vVert_kmh"] > 430.0
    assert notes["curve_window_end_s"] > 10.0
    fs2 = notes.get("fs2_track_summary") or {}
    assert fs2.get("available") is True
    assert fs2.get("quality_label") in {"stabil", "grenzwertig", "kritisch"}


def test_pipeline_repairs_leading_orphan_gap():
    result = analyze_flysight_csv(
        content=_build_leading_gap_csv(),
        file_name="leading_gap.csv",
        jumper_name="Test",
        ground_elevation_m=200.0,
        breakoff_altitude_agl_m=1700.0,
    )

    notes = result["report"]["notes"]
    quality_flags = set(json.loads(result["jump_record"]["quality_flags"]))
    chart_time = result["report"]["chart_data"]["time_s"]

    repair = notes.get("leading_gap_repair") or {}
    assert repair.get("applied") is True
    assert repair.get("trimmed_samples", 0) >= 1
    assert repair.get("gap_removed_s", 0.0) >= 2.0
    assert "TIME_GAPS" not in quality_flags
    assert chart_time and chart_time[0] == 0.0


def test_pipeline_normalizes_kmh_and_feet_exports():
    result = analyze_flysight_csv(
        content=_build_kmh_feet_unit_csv(),
        file_name="units_kmh_ft.csv",
        jumper_name="Test",
        ground_elevation_m=None,
        breakoff_altitude_agl_m=1700.0,
    )

    notes = result["report"]["notes"]
    quality_flags = set(json.loads(result["jump_record"]["quality_flags"]))
    unit_note = notes.get("unit_normalization") or {}
    vel_note = unit_note.get("velocity_conversion") or {}
    alt_note = unit_note.get("altitude_conversion") or {}

    assert vel_note.get("applied") is True
    assert vel_note.get("from") == "kmh"
    assert alt_note.get("applied") is True
    assert alt_note.get("from") == "ft"
    assert "SPEED_SPIKE" not in quality_flags
    assert result["metrics_record"]["best_3s_vVert_kmh"] < 520.0


def test_exit_dynamic_stable_is_not_marked_aggressive():
    result = analyze_flysight_csv(
        content=_build_exit_test_csv(unsteady=False),
        file_name="exit_stable.csv",
        jumper_name="Test",
        ground_elevation_m=200.0,
        breakoff_altitude_agl_m=1700.0,
    )
    scorecard = result["report"]["scorecard"]
    fp10 = next(item for item in result["report"]["fixpoints"] if item["t_rel_s"] == 10.0)

    assert fp10["vVert_kmh"] > 300
    assert scorecard["exit"] == "sauber"
    assert scorecard["exit_dynamik"] == "dynamisch_stabil"


def test_exit_dynamic_unsteady_is_marked_aggressive():
    result = analyze_flysight_csv(
        content=_build_exit_test_csv(unsteady=True),
        file_name="exit_unsteady.csv",
        jumper_name="Test",
        ground_elevation_m=200.0,
        breakoff_altitude_agl_m=1700.0,
    )
    scorecard = result["report"]["scorecard"]
    fp10 = next(item for item in result["report"]["fixpoints"] if item["t_rel_s"] == 10.0)

    assert fp10["vVert_kmh"] > 300
    assert scorecard["exit"] == "aggressiv"
    assert scorecard["exit_dynamik"] in {"dynamisch_unruhig", "unruhig"}


def test_t0_anchors_to_main_peak_not_late_secondary_dive():
    dt = 0.2
    n = 1400
    t = np.arange(n) * dt
    rng = np.random.default_rng(42)
    base_time = datetime(2026, 5, 25, 10, 0, 0, tzinfo=timezone.utc)
    times = [base_time + timedelta(seconds=float(x)) for x in t]

    # Main speed event around 120s, secondary weaker event around 170s.
    vel_d = np.zeros_like(t)
    for i, x in enumerate(t):
        if 110 <= x <= 125:
            vel_d[i] = min(125, (x - 110) * 8.5)
        elif 125 < x <= 132:
            vel_d[i] = max(0, 125 - (x - 125) * 15)
        elif 167 <= x <= 172:
            vel_d[i] = min(26, (x - 167) * 5.2)
        elif 172 < x <= 176:
            vel_d[i] = max(0, 26 - (x - 172) * 6.5)
        else:
            vel_d[i] = max(0, rng.normal(0.4, 0.2))

    vel_n = np.where(
        (t >= 110) & (t <= 132),
        np.maximum(3.0, 50 - (t - 110) * 2.1),
        np.where((t >= 167) & (t <= 176), np.maximum(4.0, 34 - (t - 167) * 1.6), 42.0),
    )
    vel_e = np.full_like(t, 2.0)

    h = [4500.0]
    for i in range(1, len(t)):
        h.append(h[-1] - float(vel_d[i]) * dt)
    h = np.array(h)

    df = pd.DataFrame(
        {
            "time": [ts.isoformat() for ts in times],
            "lat": 50.0,
            "lon": 8.0,
            "hMSL": h,
            "velN": vel_n,
            "velE": vel_e,
            "velD": vel_d,
            "hAcc": 1.2,
            "vAcc": 1.4,
            "sAcc": 0.7,
            "gpsFix": 3,
            "numSV": 14,
        }
    )

    result = analyze_flysight_csv(
        content=df.to_csv(index=False).encode("utf-8"),
        file_name="two_events.csv",
        jumper_name="Test",
        ground_elevation_m=200.0,
        breakoff_altitude_agl_m=1700.0,
    )

    t0 = datetime.fromisoformat(result["jump_record"]["t0_utc"].replace("Z", "+00:00"))
    rel_t0 = (t0 - base_time).total_seconds()
    assert 109.0 <= rel_t0 <= 116.0


def test_t0_soft_plausibility_fallback_handles_low_vhor_drop_profile():
    dt = 0.2
    n = 420
    t = np.arange(n) * dt
    base_time = datetime(2026, 5, 29, 10, 0, 0, tzinfo=timezone.utc)
    times = [base_time + timedelta(seconds=float(x)) for x in t]

    vel_d = np.piecewise(
        t,
        [t < 12, (t >= 12) & (t < 30), t >= 30],
        [
            lambda x: np.clip((x - 10) * 3.5, 0, None),
            lambda x: 7 + (x - 12) * 6.6,
            lambda x: np.maximum(0, 125 - (x - 30) * 1.3),
        ],
    )
    vel_d = np.clip(vel_d, 0, 128)

    # Deliberately keep horizontal component nearly flat so strict vHor drop test is hard to satisfy.
    vel_n = np.where(t < 40, 39.0 - (t * 0.04), 37.4)
    vel_e = np.full_like(t, 2.4)

    h = [4450.0]
    for i in range(1, len(t)):
        h.append(h[-1] - float(vel_d[i]) * dt)
    h = np.array(h)

    df = pd.DataFrame(
        {
            "time": [ts.isoformat() for ts in times],
            "lat": 50.0,
            "lon": 8.0,
            "hMSL": h,
            "velN": vel_n,
            "velE": vel_e,
            "velD": vel_d,
            "hAcc": 1.2,
            "vAcc": 1.4,
            "sAcc": 0.7,
            "gpsFix": 3,
            "numSV": 14,
        }
    )

    result = analyze_flysight_csv(
        content=df.to_csv(index=False).encode("utf-8"),
        file_name="soft_fallback.csv",
        jumper_name="Test",
        ground_elevation_m=200.0,
        breakoff_altitude_agl_m=1700.0,
    )

    quality_flags = json.loads(result["jump_record"]["quality_flags"])
    notes = result["report"]["notes"]

    assert "NO_CLEAR_EXIT" not in quality_flags
    assert notes["t0_confidence"] >= 0.55
    assert notes["t0_review_required"] is False


def test_t0_is_shifted_to_ramp_start_when_detection_lands_too_late():
    dt = 0.2
    n = 1200
    t = np.arange(n) * dt
    base_time = datetime(2026, 5, 30, 10, 0, 0, tzinfo=timezone.utc)
    times = [base_time + timedelta(seconds=float(x)) for x in t]

    # Long pre-track, then jump ramp:
    # 0-120s: near static
    # 120-130s: soft ramp (crosses 10 m/s early)
    # 130-152s: strong ramp to peak
    # >152s: decel
    vel_d = np.zeros_like(t)
    for i, x in enumerate(t):
        if x < 120.0:
            vel_d[i] = 0.5
        elif x < 130.0:
            vel_d[i] = 4.0 + (x - 120.0) * 2.0
        elif x < 152.0:
            vel_d[i] = 24.0 + (x - 130.0) * 4.2
        elif x < 160.0:
            vel_d[i] = 116.4 - (x - 152.0) * 8.5
        else:
            vel_d[i] = max(0.0, 48.4 - (x - 160.0) * 0.8)
    vel_d = np.clip(vel_d, 0, 124)

    vel_n = np.where(
        t < 120.0,
        0.5,
        np.where(t < 130.0, 42.0 - (t - 120.0) * 0.3, np.maximum(7.0, 39.0 - (t - 130.0) * 1.3)),
    )
    vel_e = np.full_like(t, 1.5)

    h = [4500.0]
    for i in range(1, len(t)):
        h.append(h[-1] - float(vel_d[i]) * dt)
    h = np.array(h)

    df = pd.DataFrame(
        {
            "time": [ts.isoformat() for ts in times],
            "lat": 50.0,
            "lon": 8.0,
            "hMSL": h,
            "velN": vel_n,
            "velE": vel_e,
            "velD": vel_d,
            "hAcc": 1.3,
            "vAcc": 1.4,
            "sAcc": 0.8,
            "gpsFix": 3,
            "numSV": 14,
        }
    )

    result = analyze_flysight_csv(
        content=df.to_csv(index=False).encode("utf-8"),
        file_name="late_detect_shift.csv",
        jumper_name="Test",
        ground_elevation_m=200.0,
        breakoff_altitude_agl_m=1700.0,
    )

    notes = result["report"]["notes"]
    first_point = result["report"]["chart_data"]["vVert_kmh"][0]
    t0 = datetime.fromisoformat(result["jump_record"]["t0_utc"].replace("Z", "+00:00"))
    rel_t0 = (t0 - base_time).total_seconds()

    # Ramp starts around velD=10m/s at t~123s; t0 should be near that region and not late in high-speed phase.
    assert 122.0 <= rel_t0 <= 125.5
    assert first_point < 70.0
    assert notes.get("t0_reason")


def test_rejects_non_jump_recording_without_freefall_event():
    dt = 0.2
    n = 300
    t = np.arange(n) * dt
    base_time = datetime(2026, 5, 29, 10, 0, 0, tzinfo=timezone.utc)
    times = [base_time + timedelta(seconds=float(x)) for x in t]

    vel_d = 0.2 + 0.3 * np.sin(t / 6.0)
    vel_n = 0.3 + 0.2 * np.cos(t / 4.0)
    vel_e = 0.1 + 0.1 * np.sin(t / 5.0)

    h = [350.0]
    for i in range(1, len(t)):
        h.append(h[-1] - float(vel_d[i]) * dt)
    h = np.array(h)

    df = pd.DataFrame(
        {
            "time": [ts.isoformat() for ts in times],
            "lat": 50.0,
            "lon": 8.0,
            "hMSL": h,
            "velN": vel_n,
            "velE": vel_e,
            "velD": vel_d,
            "hAcc": 4.0,
            "vAcc": 4.0,
            "sAcc": 1.5,
            "gpsFix": 3,
            "numSV": 8,
        }
    )

    try:
        analyze_flysight_csv(
            content=df.to_csv(index=False).encode("utf-8"),
            file_name="non_jump.csv",
            jumper_name="Test",
            ground_elevation_m=200.0,
            breakoff_altitude_agl_m=1700.0,
        )
        assert False, "AnalysisError expected for non-jump profile"
    except AnalysisError as exc:
        assert "Kein plausibler Sprung erkannt" in str(exc)
