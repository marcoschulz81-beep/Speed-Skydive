from __future__ import annotations

from pathlib import Path

import pytest

from app import database
from app.database import init_db
from app.services.storage import (
    JUMP_SERIES_ENCODING,
    _decode_jump_series,
    _encode_jump_series,
    get_ai_coaching_result,
    upsert_ai_coaching_result,
)


def test_v1_4_migration_is_idempotent_and_enables_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "migration.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)

    init_db()
    init_db()

    with database.get_connection() as conn:
        migration = conn.execute(
            "SELECT name FROM schema_migrations WHERE version = 1"
        ).fetchone()
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    assert migration is not None
    assert migration["name"] == "performance_storage_v1"
    assert {
        "jump_series",
        "jump_analysis_features",
        "jumper_profile_snapshots",
        "ai_coaching_results",
    }.issubset(tables)
    assert journal_mode == "wal"


def test_compact_series_round_trip_preserves_numeric_values() -> None:
    samples = [
        {
            "t_rel_s": float(index) * 0.2,
            "vVert_kmh": 250.0 + index,
            "vHor_kmh": 100.0 - index,
            "angle_deg": 70.0 + index * 0.1,
            "hAGL_m": None if index == 1 else 3000.0 - index * 10.0,
            "accVert_mps2": 2.5 + index * 0.01,
            "velN_mps": 20.0 - index * 0.2,
            "velE_mps": 1.0 + index * 0.05,
        }
        for index in range(8)
    ]

    payload, point_count, start_s, end_s = _encode_jump_series(samples)
    decoded = _decode_jump_series(
        {
            "encoding": JUMP_SERIES_ENCODING,
            "point_count": point_count,
            "payload": payload,
        }
    )

    assert decoded is not None
    assert start_s == 0.0
    assert end_s == pytest.approx(1.4)
    assert decoded["time_s"] == pytest.approx([row["t_rel_s"] for row in samples])
    assert decoded["vVert_kmh"] == pytest.approx([row["vVert_kmh"] for row in samples])
    assert decoded["hAGL_m"][1] is None


def test_ai_coaching_cache_survives_process_local_cache_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "ai-cache.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    init_db()

    upsert_ai_coaching_result(
        "cache-key",
        jump_id=None,
        analysis_signature="signature",
        model="test-model",
        prompt_version="10",
        view_mode="expert",
        status="ready",
        payload={"available": True, "summary": "Gespeichert"},
    )
    cached = get_ai_coaching_result("cache-key")

    assert cached is not None
    assert cached["_cache_status"] == "ready"
    assert cached["summary"] == "Gespeichert"
