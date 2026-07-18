from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from app.config import ANALYSIS_VERSION, RAW_UPLOAD_DIR
from app.database import get_connection, init_db
from app.services.dropzone_matching import analyze_flysight_with_dropzone, list_dropzone_match_contexts
from app.services.storage import (
    get_jump_report,
    get_jump_source_metadata,
    replace_analysis_result,
    save_dropzone_match_attempt,
)


def _source_path(metadata: dict[str, Any]) -> Path | None:
    stored = str(metadata.get("source_file_path") or "").strip()
    if stored:
        path = Path(stored)
        if path.is_file():
            return path
    source_hash = str(metadata.get("source_file_sha256") or "").strip()
    if source_hash:
        for suffix in (".csv", ".CSV"):
            candidate = RAW_UPLOAD_DIR / f"{source_hash}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def _repair_orphans() -> dict[str, int]:
    tables = (
        "samples",
        "metrics",
        "jump_feedback",
        "coaching_snapshots",
        "dropzone_match_attempts",
    )
    removed: dict[str, int] = {}
    with get_connection() as conn:
        for table in tables:
            count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE jump_id NOT IN (SELECT jump_id FROM jumps)"
                ).fetchone()[0]
            )
            removed[table] = count
            if count:
                conn.execute(f"DELETE FROM {table} WHERE jump_id NOT IN (SELECT jump_id FROM jumps)")
        conn.commit()
    return removed


def reprocess_all(
    *,
    dry_run: bool = False,
    limit: int | None = None,
    dropzone_rollout: bool = False,
    json_output: bool = False,
) -> int:
    init_db()
    removed = {} if dry_run else _repair_orphans()
    if removed and not json_output:
        print("Orphan-Reparatur: " + ", ".join(f"{key}={value}" for key, value in removed.items()))

    with get_connection() as conn:
        rows = [
            dict(row)
            for row in conn.execute("SELECT jump_id FROM jumps ORDER BY t0_utc ASC, created_at ASC").fetchall()
        ]
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    failures: list[tuple[str, str]] = []
    match_status_counts: Counter[str] = Counter()
    rollout_skip_counts: Counter[str] = Counter()
    preservation_counts: Counter[str] = Counter()
    rule_status_changes: Counter[str] = Counter()
    ground_deltas: list[float] = []
    score_deltas: list[float] = []
    analyzed = 0
    eligible = 0
    applied = 0
    attempts_recorded = 0
    zones = list_dropzone_match_contexts()
    for index, row in enumerate(rows, start=1):
        jump_id = str(row["jump_id"])
        report = get_jump_report(jump_id)
        metadata = get_jump_source_metadata(jump_id)
        if report is None or metadata is None:
            failures.append((jump_id, "REPORT_OR_METADATA_MISSING"))
            continue
        source_path = _source_path(metadata)
        if source_path is None:
            failures.append((jump_id, "SOURCE_FILE_MISSING"))
            continue

        content = source_path.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        jump = report.get("jump", {})
        metrics = report.get("metrics", {})
        notes = report.get("notes", {})
        manual_t0_utc = str(jump.get("t0_utc") or "") if notes.get("t0_manual_override") else None
        ground_source = str(jump.get("ground_elevation_source") or "estimated")
        ground = jump.get("ground_elevation_m") if ground_source == "manual" else None
        manual_dropzone_id = (
            str(jump.get("dropzone_id"))
            if jump.get("dropzone_id") and jump.get("dropzone_assignment_source") == "catalog_manual"
            else None
        )
        try:
            result = analyze_flysight_with_dropzone(
                content=content,
                file_name=str(jump.get("file_name") or source_path.name),
                jumper_name=str(jump.get("jumper_name") or ""),
                ground_elevation_m=ground,
                breakoff_altitude_agl_m=jump.get("breakoff_altitude_agl_m"),
                manual_t0_utc=manual_t0_utc,
                manual_dropzone_id=manual_dropzone_id,
                zones=zones,
            )
            analyzed += 1
            match = result.get("dropzone_match") or {}
            match_status = str(match.get("status") or "unavailable")
            match_status_counts[match_status] += 1
            rollout_eligible = match_status == "accepted"
            if rollout_eligible:
                eligible += 1
                if ground_source == "manual":
                    preservation_counts["manual_ground_preserved"] += 1
            else:
                rollout_skip_counts[f"match_{match_status}"] += 1

            old_ground = jump.get("ground_elevation_m")
            new_ground = result["jump_record"].get("ground_elevation_m")
            if old_ground is not None and new_ground is not None:
                ground_deltas.append(float(new_ground) - float(old_ground))
            old_score = metrics.get("rule_based_3s_score")
            new_score = result["metrics_record"].get("rule_based_3s_score")
            if old_score is not None and new_score is not None:
                score_deltas.append(float(new_score) - float(old_score))
            old_status = str(metrics.get("rule_score_status") or "unknown")
            new_status = str(result["metrics_record"].get("rule_score_status") or "unknown")
            rule_status_changes[f"{old_status}->{new_status}"] += 1

            should_apply = not dry_run and (not dropzone_rollout or rollout_eligible)
            if should_apply:
                replace_analysis_result(
                    jump_id=jump_id,
                    result=result,
                    jump_context=str(jump.get("jump_context") or "unknown"),
                    is_reference_only=bool(jump.get("is_reference_only")),
                    source_file_sha256=source_hash,
                    source_file_path=str(source_path.resolve()),
                )
                applied += 1
                attempts_recorded += 1
            elif not dry_run and dropzone_rollout:
                save_dropzone_match_attempt(jump_id, match)
                attempts_recorded += 1
        except Exception as exc:
            failures.append((jump_id, str(exc)))
        if not json_output and (index % 25 == 0 or index == len(rows)):
            print(f"Fortschritt: {index}/{len(rows)}, analysiert={analyzed}, Fehler={len(failures)}")

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "total": len(rows),
        "analyzed": analyzed,
        "eligible": eligible,
        "applied": applied,
        "attempts_recorded": attempts_recorded,
        "failures": len(failures),
        "dry_run": dry_run,
        "dropzone_rollout": dropzone_rollout,
        "match_status": dict(sorted(match_status_counts.items())),
        "rollout_skips": dict(sorted(rollout_skip_counts.items())),
        "preserved": dict(sorted(preservation_counts.items())),
        "rule_status_changes": dict(sorted(rule_status_changes.items())),
        "ground_delta_m": _delta_summary(ground_deltas),
        "rule_score_delta_kmh": _delta_summary(score_deltas),
        "failure_details": [{"jump_id": jump_id, "reason": reason} for jump_id, reason in failures],
    }
    if json_output:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"Reanalyse {ANALYSIS_VERSION}: gesamt={len(rows)}, analysiert={analyzed}, "
            f"freigegeben={eligible}, geschrieben={applied}, Fehler={len(failures)}, dry_run={dry_run}"
        )
        print("Match-Status: " + ", ".join(f"{key}={value}" for key, value in sorted(match_status_counts.items())))
        for jump_id, reason in failures:
            print(f"FEHLER {jump_id}: {reason}", file=sys.stderr)
    return 1 if failures else 0


def _delta_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "mean_absolute": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "mean": round(mean(values), 3),
        "mean_absolute": round(mean(abs(value) for value in values), 3),
        "minimum": round(min(values), 3),
        "maximum": round(max(values), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Alle gespeicherten Sprünge mit der aktuellen Analyseversion neu berechnen."
    )
    parser.add_argument("--dry-run", action="store_true", help="Analysieren, aber Datenbank nicht ersetzen.")
    parser.add_argument("--limit", type=int, default=None, help="Optional nur die ersten N Sprünge bearbeiten.")
    parser.add_argument(
        "--dropzone-rollout",
        action="store_true",
        help="Nur sichere automatische Dropzone-Treffer ersetzen; manuelle Bodenhöhen bleiben erhalten.",
    )
    parser.add_argument("--json", action="store_true", help="Zusammenfassung als JSON ausgeben.")
    args = parser.parse_args()
    return reprocess_all(
        dry_run=bool(args.dry_run),
        limit=args.limit,
        dropzone_rollout=bool(args.dropzone_rollout),
        json_output=bool(args.json),
    )


if __name__ == "__main__":
    raise SystemExit(main())
