from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.analysis.pipeline import analyze_flysight_csv


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
