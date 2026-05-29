from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.analysis.curve_window import detect_curve_window
from app.database import get_connection


def save_analysis_result(result: dict[str, Any]) -> str:
    jump = result["jump_record"]
    metrics = result["metrics_record"]
    samples = result["sample_records"]

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO jumps (
                jump_id, jumper_name, file_name, device_type, raw_start_time_utc, t0_utc,
                exit_altitude_msl_m, exit_altitude_agl_m, ground_elevation_m, is_valid_altitude,
                sample_rate_hz, quality_score, quality_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                jump["jump_id"],
                jump["jumper_name"],
                jump["file_name"],
                jump["device_type"],
                jump["raw_start_time_utc"],
                jump["t0_utc"],
                jump["exit_altitude_msl_m"],
                jump["exit_altitude_agl_m"],
                jump["ground_elevation_m"],
                jump["is_valid_altitude"],
                jump["sample_rate_hz"],
                jump["quality_score"],
                jump["quality_flags"],
            ),
        )

        conn.executemany(
            """
            INSERT INTO samples (
                jump_id, time_utc, t_rel_s, lat, lon, hMSL_m, hAGL_m, velN_mps, velE_mps, velD_mps,
                vVert_kmh, vHor_kmh, vTotal_kmh, angle_deg, accVert_mps2, hAcc, vAcc, sAcc,
                gpsFix, numSV, quality_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s["jump_id"],
                    s["time_utc"],
                    s["t_rel_s"],
                    s["lat"],
                    s["lon"],
                    s["hMSL_m"],
                    s["hAGL_m"],
                    s["velN_mps"],
                    s["velE_mps"],
                    s["velD_mps"],
                    s["vVert_kmh"],
                    s["vHor_kmh"],
                    s["vTotal_kmh"],
                    s["angle_deg"],
                    s["accVert_mps2"],
                    s["hAcc"],
                    s["vAcc"],
                    s["sAcc"],
                    s["gpsFix"],
                    s["numSV"],
                    s["quality_flags"],
                )
                for s in samples
            ],
        )

        conn.execute(
            """
            INSERT INTO metrics (
                jump_id, best_3s_start_s, best_3s_end_s, best_3s_vVert_mps, best_3s_vVert_kmh,
                best_3s_vHor_kmh, best_3s_angle_deg, training_3s_max_from_t0, rule_based_3s_score,
                rule_based_3s_score_mps, performance_window_start_s, performance_window_end_s,
                validation_window_quality, hot_zone_start_s, hot_zone_end_s, negative_risk_score,
                notes, fixpoints_json, phases_json, scorecard_json, tips_json, quality_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metrics["jump_id"],
                metrics["best_3s_start_s"],
                metrics["best_3s_end_s"],
                metrics["best_3s_vVert_mps"],
                metrics["best_3s_vVert_kmh"],
                metrics["best_3s_vHor_kmh"],
                metrics["best_3s_angle_deg"],
                metrics["training_3s_max_from_t0"],
                metrics["rule_based_3s_score"],
                metrics["rule_based_3s_score_mps"],
                metrics["performance_window_start_s"],
                metrics["performance_window_end_s"],
                metrics["validation_window_quality"],
                metrics["hot_zone_start_s"],
                metrics["hot_zone_end_s"],
                metrics["negative_risk_score"],
                metrics["notes"],
                metrics["fixpoints_json"],
                metrics["phases_json"],
                metrics["scorecard_json"],
                metrics["tips_json"],
                metrics["quality_flags"],
            ),
        )
        conn.commit()
    return jump["jump_id"]


def list_recent_jumps(limit: int = 30) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                j.jump_id,
                j.jumper_name,
                j.file_name,
                j.t0_utc,
                j.sample_rate_hz,
                j.quality_score,
                j.is_valid_altitude,
                m.best_3s_vVert_kmh,
                m.rule_based_3s_score
            FROM jumps j
            JOIN metrics m ON m.jump_id = j.jump_id
            ORDER BY j.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_jumpers() -> list[str]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT jumper_name FROM jumps ORDER BY jumper_name COLLATE NOCASE ASC"
        ).fetchall()
    return [str(row["jumper_name"]) for row in rows]


def list_jumps_for_jumper(jumper_name: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                j.jump_id,
                j.file_name,
                j.t0_utc,
                j.quality_score,
                j.sample_rate_hz,
                j.is_valid_altitude,
                m.best_3s_vVert_kmh,
                m.rule_based_3s_score
            FROM jumps j
            JOIN metrics m ON m.jump_id = j.jump_id
            WHERE j.jumper_name = ?
            ORDER BY j.created_at DESC
            """,
            (jumper_name,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_jump_report(jump_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        jump = conn.execute("SELECT * FROM jumps WHERE jump_id = ?", (jump_id,)).fetchone()
        metrics = conn.execute("SELECT * FROM metrics WHERE jump_id = ?", (jump_id,)).fetchone()
        samples = conn.execute(
            """
            SELECT
                t_rel_s, vVert_kmh, vHor_kmh, angle_deg, hAGL_m, accVert_mps2
            FROM samples
            WHERE jump_id = ?
            ORDER BY t_rel_s ASC
            """,
            (jump_id,),
        ).fetchall()

    if jump is None or metrics is None:
        return None

    jump_dict = dict(jump)
    metrics_dict = dict(metrics)
    notes = json.loads(metrics_dict["notes"]) if metrics_dict.get("notes") else {}
    fixpoints = json.loads(metrics_dict["fixpoints_json"])
    phases = json.loads(metrics_dict["phases_json"])
    scorecard = json.loads(metrics_dict["scorecard_json"])
    tips = json.loads(metrics_dict["tips_json"])
    quality_flags = json.loads(jump_dict["quality_flags"])

    chart_data = {
        "time_s": [float(row["t_rel_s"]) for row in samples],
        "vVert_kmh": [float(row["vVert_kmh"]) for row in samples],
        "vHor_kmh": [float(row["vHor_kmh"]) for row in samples],
        "angle_deg": [float(row["angle_deg"]) for row in samples],
        "hAGL_m": [None if row["hAGL_m"] is None else float(row["hAGL_m"]) for row in samples],
        "accVert_mps2": [float(row["accVert_mps2"]) for row in samples],
    }

    # Backward-compatible fallback for older records without curve window metadata.
    if "curve_window_start_s" not in notes or "curve_window_end_s" not in notes:
        sample_df = pd.DataFrame(
            {
                "t_rel_s": chart_data["time_s"],
                "vVert_kmh": chart_data["vVert_kmh"],
            }
        )
        window = detect_curve_window(sample_df, sample_rate_hz=jump_dict.get("sample_rate_hz"))
        notes.update(window)

    notes.setdefault("pw_start_utc", None)
    notes.setdefault("pw_start_s_from_t0", metrics_dict.get("performance_window_start_s"))
    notes.setdefault("t0_uncertainty_s", None)

    return {
        "jump": jump_dict,
        "metrics": metrics_dict,
        "notes": notes,
        "fixpoints": fixpoints,
        "phases": phases,
        "scorecard": scorecard,
        "tips": tips,
        "quality_flags": quality_flags,
        "chart_data": chart_data,
    }
