from __future__ import annotations

import json
from collections import Counter
from typing import Any

from app.analysis.evaluation import REFERENCE_HARD_FLAGS, RULE_SCORE_INVALID, RULE_SCORE_VALID
from app.config import ANALYSIS_VERSION, SCORING_WINDOW_DURATION_S
from app.database import get_connection, init_db


def _json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    try:
        value = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def audit_database() -> dict[str, Any]:
    init_db()
    issues: list[str] = []
    with get_connection() as conn:
        foreign_keys = [tuple(row) for row in conn.execute("PRAGMA foreign_key_check").fetchall()]
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT j.jump_id, j.jumper_name, j.file_name, j.analysis_version AS jump_version,
                       j.quality_flags AS jump_flags, j.ground_elevation_source,
                       j.breakoff_altitude_agl_m, m.analysis_version AS metric_version,
                       m.best_3s_start_s, m.best_3s_end_s, m.rule_based_3s_score,
                       m.rule_score_status, m.rule_score_reasons,
                       m.performance_window_start_s, m.performance_window_end_s,
                       m.validation_window_start_s, m.validation_window_end_s
                FROM jumps j JOIN metrics m ON m.jump_id = j.jump_id
                ORDER BY j.t0_utc ASC
                """
            ).fetchall()
        ]
        counts = {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("jumps", "metrics", "samples", "jump_feedback", "coaching_snapshots")
        }

    status_counts: Counter[str] = Counter()
    for row in rows:
        jump_id = str(row["jump_id"])
        status = str(row.get("rule_score_status") or "")
        status_counts[status] += 1
        reasons = [str(item) for item in _json_list(row.get("rule_score_reasons"))]
        flags = {str(item) for item in _json_list(row.get("jump_flags"))}
        duration = float(row["best_3s_end_s"]) - float(row["best_3s_start_s"])
        if abs(duration - SCORING_WINDOW_DURATION_S) > 0.011:
            issues.append(f"{jump_id}: Training-Fensterdauer {duration:.3f}s")
        if row.get("jump_version") != ANALYSIS_VERSION or row.get("metric_version") != ANALYSIS_VERSION:
            issues.append(f"{jump_id}: Versionsabweichung")
        if status == RULE_SCORE_VALID:
            if row.get("rule_based_3s_score") is None or reasons:
                issues.append(f"{jump_id}: gültiger Regel-Score ohne Wert oder mit Gründen")
            if flags & REFERENCE_HARD_FLAGS:
                issues.append(f"{jump_id}: gültiger Regel-Score trotz hartem Qualitätsflag")
        if status == RULE_SCORE_INVALID and not reasons:
            issues.append(f"{jump_id}: ungültiger Regel-Score ohne Grund")
        pw_start = row.get("performance_window_start_s")
        pw_end = row.get("performance_window_end_s")
        val_start = row.get("validation_window_start_s")
        val_end = row.get("validation_window_end_s")
        if pw_start is not None and pw_end is not None and float(pw_start) >= float(pw_end):
            issues.append(f"{jump_id}: Performance-Fenster nicht aufsteigend")
        if val_start is not None and val_end is not None and float(val_start) > float(val_end):
            issues.append(f"{jump_id}: Validierungsfenster nicht aufsteigend")
        if pw_start is not None and val_start is not None and float(val_start) < float(pw_start) - 0.02:
            issues.append(f"{jump_id}: Validierung beginnt vor Performance-Fenster")
        if pw_end is not None and val_end is not None and float(val_end) > float(pw_end) + 0.02:
            issues.append(f"{jump_id}: Validierung endet nach Performance-Fenster")
        if row.get("ground_elevation_source") not in {"manual", "estimated"}:
            issues.append(f"{jump_id}: unbekannte Bodenhöhenquelle")
        if row.get("breakoff_altitude_agl_m") is None or float(row["breakoff_altitude_agl_m"]) <= 0:
            issues.append(f"{jump_id}: ungültige Breakoff-Höhe")

    if foreign_keys:
        issues.append(f"Foreign-Key-Verletzungen: {len(foreign_keys)}")
    if counts["jumps"] != counts["metrics"] or counts["jumps"] != len(rows):
        issues.append("Sprung-/Metrikanzahl stimmt nicht überein")

    return {
        "analysis_version": ANALYSIS_VERSION,
        "counts": counts,
        "rule_score_status": dict(sorted(status_counts.items())),
        "foreign_key_violations": len(foreign_keys),
        "issue_count": len(issues),
        "issues": issues,
    }


def main() -> int:
    result = audit_database()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
