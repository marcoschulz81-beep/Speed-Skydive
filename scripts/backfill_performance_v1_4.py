from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import DATABASE_PATH
from app.database import get_connection, init_db
from app.main import (
    _build_cached_jumper_summary,
    _get_marco_top15_profile,
    _persist_analysis_feature,
    _reference_signature_from_profile,
)
from app.services.storage import (
    audit_jump_series,
    backfill_jump_series,
    get_jump_report,
    list_analysis_features,
    list_jumpers,
    list_jumps_for_jumper,
)


def _create_database_backup(backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"speed_skydive-pre-v1.4.0-backfill-{stamp}.db"
    source = sqlite3.connect(DATABASE_PATH)
    destination = sqlite3.connect(target)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return target


def _all_jump_rows() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT jump_id, jumper_name FROM jumps ORDER BY created_at, jump_id"
        ).fetchall()
    return [dict(row) for row in rows]


def _backfill_features(*, force: bool = False) -> dict[str, int]:
    if force:
        with get_connection() as conn:
            conn.execute("DELETE FROM jumper_profile_snapshots")
            conn.execute("DELETE FROM jump_analysis_features")
            conn.commit()
    rows = _all_jump_rows()
    jump_ids = [str(row["jump_id"]) for row in rows]
    existing = list_analysis_features(jump_ids)
    source_created = 0
    for row in rows:
        jump_id = str(row["jump_id"])
        if jump_id in existing:
            continue
        report = get_jump_report(jump_id, prefer_compact_series=False)
        if report is None:
            continue
        _persist_analysis_feature(report, marco_profile=None, include_record=False)
        source_created += 1

    marco_profile = _get_marco_top15_profile(limit=15)
    reference_signature = _reference_signature_from_profile(marco_profile)
    existing = list_analysis_features(jump_ids, reference_signature=reference_signature)
    records_created = 0
    for row in rows:
        jump_id = str(row["jump_id"])
        feature = existing.get(jump_id)
        if isinstance(feature, dict) and isinstance(feature.get("record"), dict):
            continue
        report = get_jump_report(jump_id, prefer_compact_series=False)
        if report is None:
            continue
        _persist_analysis_feature(report, marco_profile=marco_profile, include_record=True)
        records_created += 1

    profiles_created = 0
    for jumper_name in list_jumpers():
        jump_rows = list_jumps_for_jumper(jumper_name)
        if not jump_rows:
            continue
        _build_cached_jumper_summary(jumper_name=jumper_name, jumps=jump_rows)
        profiles_created += 1
    return {
        "sources_created": source_created,
        "records_created": records_created,
        "profiles_created": profiles_created,
    }


def _database_audit() -> dict[str, Any]:
    with get_connection() as conn:
        quick_check = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        foreign_key_violations = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        migrations = [
            dict(row)
            for row in conn.execute(
                "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]
    return {
        "quick_check": quick_check,
        "foreign_key_violations": foreign_key_violations,
        "migrations": migrations,
        "series": audit_jump_series(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Additiver, wiederholbarer Backfill fuer die Performance-Speicher von App v1.4.0."
    )
    parser.add_argument("--apply", action="store_true", help="Backup erstellen und Backfill ausfuehren.")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Abgeleitete Features/Profile nach dem Backup vollstaendig neu aufbauen.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=DATABASE_PATH.parents[2] / "Speed-Skydive-backups",
        help="Zielordner fuer das automatische SQLite-Backup.",
    )
    args = parser.parse_args()

    if not args.apply:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "database": str(DATABASE_PATH),
                    "exists": DATABASE_PATH.exists(),
                    "next_step": "Mit --apply wird zuerst ein konsistentes DB-Backup erstellt.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not DATABASE_PATH.exists():
        raise SystemExit(f"Datenbank nicht gefunden: {DATABASE_PATH}")
    backup_path = _create_database_backup(args.backup_dir.resolve())
    init_db()
    series_result = backfill_jump_series()
    feature_result = _backfill_features(force=bool(args.rebuild_features))
    audit = _database_audit()
    result = {
        "mode": "applied",
        "backup": str(backup_path),
        "series_backfill": series_result,
        "feature_backfill": feature_result,
        "audit": audit,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if audit["quick_check"] == "ok" and audit["foreign_key_violations"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
