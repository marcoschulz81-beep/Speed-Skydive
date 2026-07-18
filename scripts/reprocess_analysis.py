from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

from app.analysis.pipeline import analyze_flysight_csv
from app.config import ANALYSIS_VERSION, RAW_UPLOAD_DIR
from app.database import get_connection, init_db
from app.services.storage import get_jump_report, get_jump_source_metadata, replace_analysis_result


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
    tables = ("samples", "metrics", "jump_feedback", "coaching_snapshots")
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
                conn.execute(
                    f"DELETE FROM {table} WHERE jump_id NOT IN (SELECT jump_id FROM jumps)"
                )
        conn.commit()
    return removed


def reprocess_all(*, dry_run: bool = False, limit: int | None = None) -> int:
    init_db()
    removed = {} if dry_run else _repair_orphans()
    if removed:
        print("Orphan-Reparatur: " + ", ".join(f"{key}={value}" for key, value in removed.items()))

    with get_connection() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT jump_id FROM jumps ORDER BY t0_utc ASC, created_at ASC"
            ).fetchall()
        ]
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    failures: list[tuple[str, str]] = []
    processed = 0
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
        notes = report.get("notes", {})
        manual_t0_utc = str(jump.get("t0_utc") or "") if notes.get("t0_manual_override") else None
        ground = (
            None
            if str(jump.get("ground_elevation_source") or "estimated") == "estimated"
            else jump.get("ground_elevation_m")
        )
        try:
            result = analyze_flysight_csv(
                content=content,
                file_name=str(jump.get("file_name") or source_path.name),
                jumper_name=str(jump.get("jumper_name") or ""),
                ground_elevation_m=ground,
                breakoff_altitude_agl_m=jump.get("breakoff_altitude_agl_m"),
                manual_t0_utc=manual_t0_utc,
            )
            if not dry_run:
                replace_analysis_result(
                    jump_id=jump_id,
                    result=result,
                    jump_context=str(jump.get("jump_context") or "unknown"),
                    is_reference_only=bool(jump.get("is_reference_only")),
                    source_file_sha256=source_hash,
                    source_file_path=str(source_path.resolve()),
                )
            processed += 1
        except Exception as exc:
            failures.append((jump_id, str(exc)))
        if index % 25 == 0 or index == len(rows):
            print(f"Fortschritt: {index}/{len(rows)}, erfolgreich={processed}, Fehler={len(failures)}")

    print(
        f"Reanalyse {ANALYSIS_VERSION}: gesamt={len(rows)}, erfolgreich={processed}, "
        f"Fehler={len(failures)}, dry_run={dry_run}"
    )
    for jump_id, reason in failures:
        print(f"FEHLER {jump_id}: {reason}", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Alle gespeicherten Sprünge mit der aktuellen Analyseversion neu berechnen.")
    parser.add_argument("--dry-run", action="store_true", help="Analysieren, aber Datenbank nicht ersetzen.")
    parser.add_argument("--limit", type=int, default=None, help="Optional nur die ersten N Sprünge bearbeiten.")
    args = parser.parse_args()
    return reprocess_all(dry_run=bool(args.dry_run), limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
