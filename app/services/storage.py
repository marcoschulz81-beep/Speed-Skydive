from __future__ import annotations

import json
import math
from statistics import median
from typing import Any

import pandas as pd

from app.analysis.curve_window import detect_curve_window
from app.database import get_connection

VALID_JUMP_CONTEXTS = {"unknown", "training", "competition"}
MAX_JUMP_FEEDBACK_CHARS = 2000


def normalize_jump_context(raw: Any, *, default: str = "unknown") -> str:
    value = str(raw or "").strip().lower()
    if value in VALID_JUMP_CONTEXTS:
        return value
    return default


def normalize_jump_feedback_text(raw: Any) -> str:
    raw_text = str(raw or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in raw_text.split("\n")]
    text = "\n".join(line for line in lines).strip()
    if len(text) > MAX_JUMP_FEEDBACK_CHARS:
        return text[:MAX_JUMP_FEEDBACK_CHARS].rstrip()
    return text


def save_analysis_result(
    result: dict[str, Any],
    *,
    jump_context: str = "unknown",
    is_reference_only: bool = False,
    source_file_sha256: str | None = None,
    source_file_path: str | None = None,
) -> tuple[str, bool]:
    """
    Returns (jump_id, is_duplicate).
    When duplicate, no new rows are inserted and existing jump_id is returned.
    """
    jump = result["jump_record"]

    with get_connection() as conn:
        duplicate_id = _find_duplicate_jump_id(
            conn=conn,
            jumper_name=jump["jumper_name"],
            file_name=jump["file_name"],
            raw_start_time_utc=jump["raw_start_time_utc"],
            source_file_sha256=source_file_sha256,
            analysis_signature=str(jump.get("analysis_signature") or "legacy"),
        )
        if duplicate_id is not None:
            if source_file_sha256:
                conn.execute(
                    """
                    UPDATE jumps
                    SET source_file_sha256 = COALESCE(source_file_sha256, ?)
                    WHERE jump_id = ?
                    """,
                    (source_file_sha256, duplicate_id),
                )
                conn.commit()
            return duplicate_id, True

        _insert_analysis_result(
            conn=conn,
            result=result,
            jump_context=normalize_jump_context(jump_context),
            is_reference_only=is_reference_only,
            source_file_sha256=source_file_sha256,
            source_file_path=source_file_path,
        )
        conn.commit()
    return jump["jump_id"], False


def find_duplicate_jump_by_source_hash(
    *, jumper_name: str, source_file_sha256: str, analysis_signature: str = "legacy"
) -> str | None:
    if not source_file_sha256:
        return None
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT jump_id FROM jumps
            WHERE jumper_name = ? AND source_file_sha256 = ? AND analysis_signature = ?
            LIMIT 1
            """,
            (jumper_name, source_file_sha256, analysis_signature),
        ).fetchone()
    return None if row is None else str(row["jump_id"])


def replace_analysis_result(
    *,
    jump_id: str,
    result: dict[str, Any],
    jump_context: str | None = None,
    is_reference_only: bool = False,
    source_file_sha256: str | None = None,
    source_file_path: str | None = None,
) -> str:
    """
    Replace an existing jump analysis in-place (same jump_id).
    """
    _override_result_jump_id(result=result, jump_id=jump_id)
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT jump_context FROM jumps WHERE jump_id = ? LIMIT 1",
            (jump_id,),
        ).fetchone()
        feedback_text = _fetch_jump_feedback_text(conn=conn, jump_id=jump_id)
        resolved_context = normalize_jump_context(
            jump_context if jump_context is not None else (None if existing is None else existing["jump_context"])
        )
        conn.execute("DELETE FROM jumps WHERE jump_id = ?", (jump_id,))
        _insert_analysis_result(
            conn=conn,
            result=result,
            jump_context=resolved_context,
            is_reference_only=is_reference_only,
            source_file_sha256=source_file_sha256,
            source_file_path=source_file_path,
        )
        if feedback_text:
            _upsert_jump_feedback(conn=conn, jump_id=jump_id, feedback_text=feedback_text)
        conn.commit()
    return jump_id


def update_jump_context(jump_id: str, jump_context: str) -> bool:
    resolved_context = normalize_jump_context(jump_context, default="")
    if resolved_context not in VALID_JUMP_CONTEXTS:
        return False

    with get_connection() as conn:
        row = conn.execute(
            "SELECT jump_id FROM jumps WHERE jump_id = ? LIMIT 1",
            (jump_id,),
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE jumps SET jump_context = ? WHERE jump_id = ?",
            (resolved_context, jump_id),
        )
        conn.commit()
    return True


def get_jump_feedback(jump_id: str) -> dict[str, Any]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT jump_id, feedback_text, created_at, updated_at
            FROM jump_feedback
            WHERE jump_id = ?
            LIMIT 1
            """,
            (jump_id,),
        ).fetchone()
    if row is None:
        return {"available": False, "text": "", "created_at": None, "updated_at": None}
    return {
        "available": True,
        "text": str(row["feedback_text"] or ""),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def upsert_jump_feedback(jump_id: str, feedback_text: Any) -> bool:
    text = normalize_jump_feedback_text(feedback_text)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT jump_id FROM jumps WHERE jump_id = ? LIMIT 1",
            (jump_id,),
        ).fetchone()
        if row is None:
            return False
        if not text:
            conn.execute("DELETE FROM jump_feedback WHERE jump_id = ?", (jump_id,))
        else:
            _upsert_jump_feedback(conn=conn, jump_id=jump_id, feedback_text=text)
        conn.commit()
    return True


def get_coaching_snapshot(jump_id: str) -> dict[str, Any]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT snapshot_json, created_at, updated_at
            FROM coaching_snapshots
            WHERE jump_id = ?
            LIMIT 1
            """,
            (jump_id,),
        ).fetchone()
    if row is None:
        return {"available": False}
    try:
        payload = json.loads(row["snapshot_json"])
    except (TypeError, json.JSONDecodeError):
        return {"available": False, "reason": "INVALID_SNAPSHOT_JSON"}
    if not isinstance(payload, dict):
        return {"available": False, "reason": "INVALID_SNAPSHOT_PAYLOAD"}
    payload["available"] = True
    payload["created_at"] = row["created_at"]
    payload["updated_at"] = row["updated_at"]
    return payload


def upsert_coaching_snapshot(jump_id: str, snapshot: dict[str, Any]) -> bool:
    if not isinstance(snapshot, dict) or not snapshot:
        return False
    with get_connection() as conn:
        row = conn.execute(
            "SELECT jump_id FROM jumps WHERE jump_id = ? LIMIT 1",
            (jump_id,),
        ).fetchone()
        if row is None:
            return False
        payload = dict(snapshot)
        payload.pop("created_at", None)
        payload.pop("updated_at", None)
        payload["available"] = True
        snapshot_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        conn.execute(
            """
            INSERT INTO coaching_snapshots (jump_id, snapshot_json, created_at, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(jump_id) DO UPDATE SET
                snapshot_json = excluded.snapshot_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (jump_id, snapshot_json),
        )
        conn.commit()
    return True


def delete_jump(jump_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT jump_id FROM jumps WHERE jump_id = ? LIMIT 1",
            (jump_id,),
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM jumps WHERE jump_id = ?", (jump_id,))
        conn.commit()
    return True


def _fetch_jump_feedback_text(*, conn, jump_id: str) -> str | None:
    row = conn.execute(
        "SELECT feedback_text FROM jump_feedback WHERE jump_id = ? LIMIT 1",
        (jump_id,),
    ).fetchone()
    if row is None:
        return None
    text = normalize_jump_feedback_text(row["feedback_text"])
    return text or None


def _upsert_jump_feedback(*, conn, jump_id: str, feedback_text: str) -> None:
    text = normalize_jump_feedback_text(feedback_text)
    if not text:
        conn.execute("DELETE FROM jump_feedback WHERE jump_id = ?", (jump_id,))
        return
    conn.execute(
        """
        INSERT INTO jump_feedback (jump_id, feedback_text, created_at, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(jump_id) DO UPDATE SET
            feedback_text = excluded.feedback_text,
            updated_at = CURRENT_TIMESTAMP
        """,
        (jump_id, text),
    )


def _find_duplicate_jump_id(
    *,
    conn,
    jumper_name: str,
    file_name: str,
    raw_start_time_utc: str,
    source_file_sha256: str | None,
    analysis_signature: str,
) -> str | None:
    if source_file_sha256:
        row = conn.execute(
            """
            SELECT jump_id FROM jumps
            WHERE jumper_name = ? AND source_file_sha256 = ? AND analysis_signature = ?
            LIMIT 1
            """,
            (jumper_name, source_file_sha256, analysis_signature),
        ).fetchone()
        if row is not None:
            return str(row["jump_id"])

    row = conn.execute(
        """
        SELECT jump_id FROM jumps
        WHERE jumper_name = ? AND file_name = ? AND raw_start_time_utc = ? AND analysis_signature = ?
        LIMIT 1
        """,
        (jumper_name, file_name, raw_start_time_utc, analysis_signature),
    ).fetchone()
    return None if row is None else str(row["jump_id"])


def _insert_analysis_result(
    *,
    conn,
    result: dict[str, Any],
    jump_context: str,
    is_reference_only: bool,
    source_file_sha256: str | None,
    source_file_path: str | None,
) -> None:
    jump = result["jump_record"]
    metrics = result["metrics_record"]
    samples = result["sample_records"]

    conn.execute(
        """
        INSERT INTO jumps (
            jump_id, jumper_name, file_name, device_type, is_reference_only, jump_context, source_file_sha256, source_file_path, raw_start_time_utc, t0_utc,
            exit_altitude_msl_m, exit_altitude_agl_m, ground_elevation_m, ground_elevation_source,
            breakoff_altitude_agl_m, analysis_version, analysis_signature, is_valid_altitude,
            sample_rate_hz, quality_score, quality_flags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            jump["jump_id"],
            jump["jumper_name"],
            jump["file_name"],
            jump["device_type"],
            1 if is_reference_only else 0,
            normalize_jump_context(jump_context),
            source_file_sha256,
            source_file_path,
            jump["raw_start_time_utc"],
            jump["t0_utc"],
            jump["exit_altitude_msl_m"],
            jump["exit_altitude_agl_m"],
            jump["ground_elevation_m"],
            jump.get("ground_elevation_source", "estimated"),
            jump.get("breakoff_altitude_agl_m", 1707.0),
            jump.get("analysis_version", "legacy"),
            jump.get("analysis_signature", "legacy"),
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
        (
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
        ),
    )

    conn.execute(
        """
        INSERT INTO metrics (
            jump_id, best_3s_start_s, best_3s_end_s, best_3s_vVert_mps, best_3s_vVert_kmh,
            best_3s_vHor_kmh, best_3s_angle_deg, training_3s_max_from_t0, rule_based_3s_score,
            rule_based_3s_score_mps, performance_window_start_s, performance_window_end_s,
            validation_window_quality, validation_window_start_s, validation_window_end_s,
            rule_score_status, rule_score_reasons, analysis_version,
            hot_zone_start_s, hot_zone_end_s, negative_risk_score,
            notes, fixpoints_json, phases_json, scorecard_json, tips_json, quality_flags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            metrics.get("validation_window_start_s"),
            metrics.get("validation_window_end_s"),
            metrics.get("rule_score_status", "estimated"),
            metrics.get("rule_score_reasons", "[]"),
            metrics.get("analysis_version", "legacy"),
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


def _override_result_jump_id(*, result: dict[str, Any], jump_id: str) -> None:
    result["jump_record"]["jump_id"] = jump_id
    result["metrics_record"]["jump_id"] = jump_id
    for sample in result["sample_records"]:
        sample["jump_id"] = jump_id


def list_recent_jumps(limit: int = 30) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                j.jump_id,
                j.jumper_name,
                j.file_name,
                j.jump_context,
                j.t0_utc,
                j.sample_rate_hz,
                j.quality_score,
                j.quality_flags,
                j.is_valid_altitude,
                m.best_3s_vVert_kmh,
                m.rule_based_3s_score,
                m.rule_score_status,
                m.analysis_version
            FROM jumps j
            JOIN metrics m ON m.jump_id = j.jump_id
            WHERE j.is_reference_only = 0
            ORDER BY j.t0_utc DESC, j.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_jumpers() -> list[str]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT jumper_name
            FROM jumps
            WHERE is_reference_only = 0
            ORDER BY jumper_name COLLATE NOCASE ASC
            """
        ).fetchall()
    return [str(row["jumper_name"]) for row in rows]


def list_jumps_for_jumper(jumper_name: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                j.jump_id,
                j.file_name,
                j.jump_context,
                j.t0_utc,
                j.quality_score,
                j.quality_flags,
                j.sample_rate_hz,
                j.is_valid_altitude,
                m.best_3s_vVert_kmh,
                m.rule_based_3s_score,
                m.rule_score_status,
                m.analysis_version
            FROM jumps j
            JOIN metrics m ON m.jump_id = j.jump_id
            WHERE j.jumper_name = ?
              AND j.is_reference_only = 0
            ORDER BY j.t0_utc DESC, j.created_at DESC
            """,
            (jumper_name,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_jump_summary(jump_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                j.jump_id,
                j.jumper_name,
                j.file_name,
                j.jump_context,
                j.t0_utc,
                j.quality_score,
                j.sample_rate_hz,
                j.is_valid_altitude,
                m.best_3s_vVert_kmh,
                m.rule_based_3s_score,
                m.rule_score_status,
                m.negative_risk_score
            FROM jumps j
            JOIN metrics m ON m.jump_id = j.jump_id
            WHERE j.jump_id = ?
            LIMIT 1
            """,
            (jump_id,),
        ).fetchone()
    return None if row is None else dict(row)


def get_jump_source_metadata(jump_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                jump_id,
                jumper_name,
                file_name,
                source_file_sha256,
                source_file_path,
                jump_context,
                is_reference_only,
                t0_utc,
                ground_elevation_m,
                ground_elevation_source,
                breakoff_altitude_agl_m,
                analysis_version,
                analysis_signature
            FROM jumps
            WHERE jump_id = ?
            LIMIT 1
            """,
            (jump_id,),
        ).fetchone()
    return None if row is None else dict(row)


def get_best_jump_for_jumper(
    jumper_name: str,
    *,
    exclude_jump_id: str | None = None,
) -> dict[str, Any] | None:
    with get_connection() as conn:
        if exclude_jump_id:
            row = conn.execute(
                """
                SELECT
                    j.jump_id,
                    j.jumper_name,
                    j.file_name,
                    j.jump_context,
                    j.t0_utc,
                    m.best_3s_vVert_kmh,
                    m.rule_based_3s_score,
                    m.rule_score_status
                FROM jumps j
                JOIN metrics m ON m.jump_id = j.jump_id
                WHERE j.jumper_name = ?
                  AND j.jump_id != ?
                  AND j.is_reference_only = 0
                  AND m.rule_score_status = 'valid'
                  AND m.rule_based_3s_score IS NOT NULL
                  AND j.quality_flags NOT LIKE '%EARLY_JUMP_END%'
                  AND j.quality_flags NOT LIKE '%SPEED_SPIKE%'
                  AND j.quality_flags NOT LIKE '%TIME_GAPS%'
                  AND j.quality_flags NOT LIKE '%NO_CLEAR_EXIT%'
                  AND j.quality_flags NOT LIKE '%INVALID_EXIT_ALTITUDE%'
                  AND j.quality_flags NOT LIKE '%HIGH_SPEED_ACCURACY_ERROR%'
                  AND j.quality_flags NOT LIKE '%LOW_GPS_FIX%'
                ORDER BY m.rule_based_3s_score DESC, j.t0_utc DESC, j.created_at DESC
                LIMIT 1
                """,
                (jumper_name, exclude_jump_id),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT
                    j.jump_id,
                    j.jumper_name,
                    j.file_name,
                    j.jump_context,
                    j.t0_utc,
                    m.best_3s_vVert_kmh,
                    m.rule_based_3s_score,
                    m.rule_score_status
                FROM jumps j
                JOIN metrics m ON m.jump_id = j.jump_id
                WHERE j.jumper_name = ?
                  AND j.is_reference_only = 0
                  AND m.rule_score_status = 'valid'
                  AND m.rule_based_3s_score IS NOT NULL
                  AND j.quality_flags NOT LIKE '%EARLY_JUMP_END%'
                  AND j.quality_flags NOT LIKE '%SPEED_SPIKE%'
                  AND j.quality_flags NOT LIKE '%TIME_GAPS%'
                  AND j.quality_flags NOT LIKE '%NO_CLEAR_EXIT%'
                  AND j.quality_flags NOT LIKE '%INVALID_EXIT_ALTITUDE%'
                  AND j.quality_flags NOT LIKE '%HIGH_SPEED_ACCURACY_ERROR%'
                  AND j.quality_flags NOT LIKE '%LOW_GPS_FIX%'
                ORDER BY m.rule_based_3s_score DESC, j.t0_utc DESC, j.created_at DESC
                LIMIT 1
                """,
                (jumper_name,),
            ).fetchone()
    return None if row is None else dict(row)


def get_best_external_reference(
    *,
    current_jump_id: str,
    current_jumper_name: str,
) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                j.jump_id,
                j.jumper_name,
                j.file_name,
                j.jump_context,
                j.t0_utc,
                m.best_3s_vVert_kmh,
                m.rule_based_3s_score,
                m.rule_score_status
            FROM jumps j
            JOIN metrics m ON m.jump_id = j.jump_id
            WHERE j.jump_id != ?
              AND j.jumper_name != ?
              AND m.rule_score_status = 'valid'
              AND m.rule_based_3s_score IS NOT NULL
              AND j.quality_flags NOT LIKE '%EARLY_JUMP_END%'
              AND j.quality_flags NOT LIKE '%SPEED_SPIKE%'
              AND j.quality_flags NOT LIKE '%TIME_GAPS%'
              AND j.quality_flags NOT LIKE '%NO_CLEAR_EXIT%'
              AND j.quality_flags NOT LIKE '%INVALID_EXIT_ALTITUDE%'
              AND j.quality_flags NOT LIKE '%HIGH_SPEED_ACCURACY_ERROR%'
              AND j.quality_flags NOT LIKE '%LOW_GPS_FIX%'
            ORDER BY m.rule_based_3s_score DESC, j.t0_utc DESC, j.created_at DESC
            LIMIT 1
            """,
            (current_jump_id, current_jumper_name),
        ).fetchone()
    return None if row is None else dict(row)


def list_top_global_references(
    *,
    limit: int = 5,
    exclude_jump_id: str | None = None,
) -> list[dict[str, Any]]:
    hard_flags = [
        "EARLY_JUMP_END",
        "SPEED_SPIKE",
        "TIME_GAPS",
        "NO_CLEAR_EXIT",
        "INVALID_EXIT_ALTITUDE",
        "HIGH_SPEED_ACCURACY_ERROR",
        "LOW_GPS_FIX",
    ]

    where_lines = [
        "j.quality_flags NOT LIKE ?",
        "j.quality_flags NOT LIKE ?",
        "j.quality_flags NOT LIKE ?",
        "j.quality_flags NOT LIKE ?",
        "j.quality_flags NOT LIKE ?",
        "j.quality_flags NOT LIKE ?",
        "j.quality_flags NOT LIKE ?",
    ]
    params: list[Any] = [f"%{flag}%" for flag in hard_flags]

    if exclude_jump_id:
        where_lines.append("j.jump_id != ?")
        params.append(exclude_jump_id)

    params.append(max(1, int(limit)))

    query = f"""
        SELECT
            j.jump_id,
            j.jumper_name,
            j.file_name,
            j.jump_context,
            j.t0_utc,
            j.quality_score,
            j.quality_flags,
            j.is_reference_only,
            m.best_3s_vVert_kmh,
            m.rule_based_3s_score,
            m.rule_score_status
        FROM jumps j
        JOIN metrics m ON m.jump_id = j.jump_id
        WHERE {' AND '.join(where_lines)}
          AND m.rule_score_status = 'valid'
          AND m.rule_based_3s_score IS NOT NULL
        ORDER BY m.rule_based_3s_score DESC, j.t0_utc DESC, j.created_at DESC
        LIMIT ?
    """

    with get_connection() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def list_compare_candidates(current_jump_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                j.jump_id,
                j.jumper_name,
                j.file_name,
                j.jump_context,
                j.t0_utc,
                j.quality_score,
                m.best_3s_vVert_kmh,
                m.rule_based_3s_score,
                m.rule_score_status
            FROM jumps j
            JOIN metrics m ON m.jump_id = j.jump_id
            WHERE j.jump_id != ?
              AND j.is_reference_only = 0
            ORDER BY j.t0_utc DESC, j.created_at DESC, j.jumper_name COLLATE NOCASE ASC
            """,
            (current_jump_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_jump_report(jump_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        jump = conn.execute("SELECT * FROM jumps WHERE jump_id = ?", (jump_id,)).fetchone()
        metrics = conn.execute("SELECT * FROM metrics WHERE jump_id = ?", (jump_id,)).fetchone()
        feedback = conn.execute(
            """
            SELECT feedback_text, created_at, updated_at
            FROM jump_feedback
            WHERE jump_id = ?
            LIMIT 1
            """,
            (jump_id,),
        ).fetchone()
        coaching_snapshot = conn.execute(
            """
            SELECT snapshot_json, created_at, updated_at
            FROM coaching_snapshots
            WHERE jump_id = ?
            LIMIT 1
            """,
            (jump_id,),
        ).fetchone()
        samples = conn.execute(
            """
            SELECT
                t_rel_s, vVert_kmh, vHor_kmh, angle_deg, hAGL_m, accVert_mps2, velN_mps, velE_mps
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
    raw_rule_reasons = metrics_dict.get("rule_score_reasons")
    if isinstance(raw_rule_reasons, str):
        try:
            parsed_rule_reasons = json.loads(raw_rule_reasons)
        except json.JSONDecodeError:
            parsed_rule_reasons = []
        metrics_dict["rule_score_reasons"] = (
            parsed_rule_reasons if isinstance(parsed_rule_reasons, list) else []
        )
    metrics_dict.setdefault("rule_score_status", "estimated")
    metrics_dict.setdefault("analysis_version", jump_dict.get("analysis_version", "legacy"))
    notes = json.loads(metrics_dict["notes"]) if metrics_dict.get("notes") else {}
    fixpoints = json.loads(metrics_dict["fixpoints_json"])
    phases = json.loads(metrics_dict["phases_json"])
    scorecard = json.loads(metrics_dict["scorecard_json"])
    tips = json.loads(metrics_dict["tips_json"])
    quality_flags = json.loads(jump_dict["quality_flags"])
    coaching_snapshot_payload: dict[str, Any] = {"available": False}
    if coaching_snapshot is not None:
        try:
            raw_snapshot = json.loads(coaching_snapshot["snapshot_json"])
        except (TypeError, json.JSONDecodeError):
            raw_snapshot = {}
        if isinstance(raw_snapshot, dict):
            coaching_snapshot_payload = dict(raw_snapshot)
            coaching_snapshot_payload["available"] = True
            coaching_snapshot_payload["created_at"] = coaching_snapshot["created_at"]
            coaching_snapshot_payload["updated_at"] = coaching_snapshot["updated_at"]

    chart_data = {
        "time_s": [float(row["t_rel_s"]) for row in samples],
        "vVert_kmh": [float(row["vVert_kmh"]) for row in samples],
        "vHor_kmh": [float(row["vHor_kmh"]) for row in samples],
        "angle_deg": [float(row["angle_deg"]) for row in samples],
        "hAGL_m": [None if row["hAGL_m"] is None else float(row["hAGL_m"]) for row in samples],
        "accVert_mps2": [float(row["accVert_mps2"]) for row in samples],
        "velN_mps": [float(row["velN_mps"]) for row in samples],
        "velE_mps": [float(row["velE_mps"]) for row in samples],
    }
    forward_track = _build_forward_track_from_samples(samples)
    chart_data["forward_m"] = forward_track["forward_m"]
    chart_data["backtrack_m"] = forward_track["backtrack_m"]
    notes["forward_track"] = forward_track["summary"]

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
    notes.setdefault(
        "t0_review_required",
        bool(
            notes.get("t0_confidence") is not None
            and notes.get("t0_uncertainty_s") is not None
            and (float(notes["t0_confidence"]) < 0.55 or float(notes["t0_uncertainty_s"]) > 1.2)
        ),
    )

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
        "feedback": (
            {"available": False, "text": "", "created_at": None, "updated_at": None}
            if feedback is None
            else {
                "available": True,
                "text": str(feedback["feedback_text"] or ""),
                "created_at": feedback["created_at"],
                "updated_at": feedback["updated_at"],
            }
        ),
        "coaching_snapshot": coaching_snapshot_payload,
    }


def _build_forward_track_from_samples(samples: list[Any]) -> dict[str, Any]:
    if not samples:
        return {
            "forward_m": [],
            "backtrack_m": [],
            "summary": {"available": False},
        }

    t: list[float] = []
    vn: list[float] = []
    ve: list[float] = []
    for row in samples:
        t.append(float(row["t_rel_s"]))
        vn.append(float(row["velN_mps"]))
        ve.append(float(row["velE_mps"]))

    if len(t) < 2:
        return {
            "forward_m": [0.0 for _ in t],
            "backtrack_m": [0.0 for _ in t],
            "summary": {"available": False},
        }

    ref_indices = [i for i, ts in enumerate(t) if 1.0 <= ts <= 8.0]
    if len(ref_indices) < 3:
        ref_indices = [i for i, ts in enumerate(t) if 0.0 <= ts <= 12.0]
    if len(ref_indices) < 3:
        ref_indices = list(range(min(20, len(t))))

    ref_vn = [vn[i] for i in ref_indices]
    ref_ve = [ve[i] for i in ref_indices]
    axis_n = float(median(ref_vn)) if ref_vn else 0.0
    axis_e = float(median(ref_ve)) if ref_ve else 0.0
    axis_norm = math.hypot(axis_n, axis_e)
    if axis_norm < 1e-6:
        speeds = [math.hypot(vn[i], ve[i]) for i in range(len(t))]
        best_idx = int(max(range(len(speeds)), key=lambda i: speeds[i]))
        axis_n = float(vn[best_idx])
        axis_e = float(ve[best_idx])
        axis_norm = math.hypot(axis_n, axis_e)
    if axis_norm < 1e-6:
        return {
            "forward_m": [0.0 for _ in t],
            "backtrack_m": [0.0 for _ in t],
            "summary": {"available": False},
        }

    unit_n = axis_n / axis_norm
    unit_e = axis_e / axis_norm
    forward_v = [vn[i] * unit_n + ve[i] * unit_e for i in range(len(t))]

    forward_m: list[float] = [0.0]
    for i in range(1, len(t)):
        dt = float(max(0.0, t[i] - t[i - 1]))
        seg = 0.5 * (forward_v[i - 1] + forward_v[i]) * dt
        forward_m.append(float(forward_m[-1] + seg))

    running_max: list[float] = []
    backtrack_m: list[float] = []
    cur_max = float("-inf")
    for value in forward_m:
        cur_max = max(cur_max, float(value))
        running_max.append(cur_max)
        backtrack_m.append(float(cur_max - float(value)))

    max_forward = float(max(forward_m)) if forward_m else 0.0
    max_backtrack = float(max(backtrack_m)) if backtrack_m else 0.0
    backtrack_ratio_pct = float((max_backtrack / max(max_forward, 1e-6)) * 100.0) if max_forward > 0 else 0.0
    threshold = max(8.0, 0.08 * max_forward)
    drift_start_s = None
    for i in range(len(t)):
        if t[i] < 8.0:
            continue
        if backtrack_m[i] >= threshold:
            drift_start_s = float(t[i])
            break

    label = "stabil"
    if max_backtrack >= 28.0 and backtrack_ratio_pct >= 12.0:
        label = "negativ"
    elif max_backtrack >= 14.0 and backtrack_ratio_pct >= 7.0:
        label = "leicht_negativ"

    summary = {
        "available": True,
        "max_forward_m": round(max_forward, 2),
        "max_backtrack_m": round(max_backtrack, 2),
        "backtrack_ratio_pct": round(backtrack_ratio_pct, 2),
        "drift_start_s": None if drift_start_s is None else round(drift_start_s, 2),
        "label": label,
    }
    return {
        "forward_m": [round(float(x), 3) for x in forward_m],
        "backtrack_m": [round(float(x), 3) for x in backtrack_m],
        "summary": summary,
    }
