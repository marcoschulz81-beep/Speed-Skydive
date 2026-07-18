from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

import pytest

from app import database
from app.database import get_connection, init_db
from app.services.dropzone_catalog import (
    CatalogError,
    audit_catalog,
    import_historical_observations,
    sync_catalog,
    validate_catalog,
)


def _catalog() -> dict:
    return {
        "catalog_version": "test-1",
        "dropzones": [
            {
                "dropzone_id": "dz-test",
                "name": "Testplatz",
                "country_code": "DE",
                "latitude": 52.0,
                "longitude": 13.0,
                "match_radius_m": 1500.0,
                "ground_elevation_m": 42.0,
                "published_elevation_m": 41.0,
                "ground_elevation_uncertainty_m": 5.0,
                "ground_elevation_source": "test",
                "status": "trusted",
                "catalog_revision": 1,
            }
        ],
        "zones": [
            {
                "zone_id": "zone-test",
                "dropzone_id": "dz-test",
                "name": "Test-Landezone",
                "zone_kind": "landing",
                "latitude": 52.0,
                "longitude": 13.0,
                "match_radius_m": 1500.0,
                "ground_elevation_m": 42.0,
                "status": "active",
            }
        ],
        "operators": [
            {
                "operator_id": "operator-test",
                "name": "Testbetreiber",
                "country_code": "DE",
                "website_url": "https://example.test/",
                "status": "listed",
            }
        ],
        "operator_assignments": [
            {
                "dropzone_id": "dz-test",
                "operator_id": "operator-test",
                "is_primary": True,
                "status": "active",
            }
        ],
        "sources": [
            {
                "source_id": "source-test",
                "dropzone_id": "dz-test",
                "source_kind": "manual",
                "source_name": "Testquelle",
                "trust_level": "primary",
                "supports_fields": ["ground_elevation_m"],
                "details": {},
                "retrieved_at": "2026-07-18",
            }
        ],
    }


@pytest.fixture
def catalog_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "dropzones.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    init_db()
    return db_path


def test_catalog_sync_is_valid_idempotent_and_auditable(catalog_db: Path) -> None:
    first = sync_catalog(_catalog())
    second = sync_catalog(_catalog())

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["dropzones"] == 1
    assert second["zones"] == 1
    assert second["operators"] == 1
    assert second["operator_assignments"] == 1
    assert second["sources"] == 1
    assert second["issues"] == []

    with sqlite3.connect(catalog_db) as conn:
        metadata = conn.execute(
            "SELECT catalog_version, dropzone_count, operator_count FROM dropzone_catalog_metadata"
        ).fetchone()
    assert metadata == ("test-1", 1, 1)


def test_catalog_sync_preserves_verified_ground_value(catalog_db: Path) -> None:
    sync_catalog(_catalog())
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE dropzones
            SET status = 'verified', ground_elevation_m = 47.5,
                ground_elevation_source = 'manual', ground_elevation_uncertainty_m = 1.0
            WHERE dropzone_id = 'dz-test'
            """
        )
        conn.commit()

    changed = copy.deepcopy(_catalog())
    changed["dropzones"][0]["ground_elevation_m"] = 10.0
    sync_catalog(changed)

    with sqlite3.connect(catalog_db) as conn:
        row = conn.execute(
            """
            SELECT status, ground_elevation_m, ground_elevation_source,
                   ground_elevation_uncertainty_m
            FROM dropzones WHERE dropzone_id = 'dz-test'
            """
        ).fetchone()
    assert row == ("verified", 47.5, "manual", 1.0)


def test_catalog_validation_rejects_broken_relationship() -> None:
    broken = _catalog()
    broken["zones"][0]["dropzone_id"] = "missing"

    with pytest.raises(CatalogError, match="unbekannte Dropzone"):
        validate_catalog(broken)


def test_historical_observation_is_stored_without_assigning_jump(catalog_db: Path) -> None:
    sync_catalog(_catalog())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO jumps (
                jump_id, jumper_name, file_name, device_type, raw_start_time_utc, t0_utc,
                exit_altitude_msl_m, is_valid_altitude, sample_rate_hz, quality_score, quality_flags
            ) VALUES (
                'jump-test', 'Test', 'test.csv', 'FlySight',
                '2026-07-18T10:00:00Z', '2026-07-18T10:01:00Z',
                4000.0, 1, 10.0, 1.0, '[]'
            )
            """
        )
        rows = []
        for index in range(61):
            t_rel_s = 100.0 + index / 10.0
            rows.append(
                (
                    "jump-test",
                    f"2026-07-18T10:02:{index / 10:04.1f}Z",
                    t_rel_s,
                    52.0 + (index % 3 - 1) * 0.000001,
                    13.0 + (index % 3 - 1) * 0.000001,
                    42.0 + (index % 3 - 1) * 0.1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    1.0,
                    1.0,
                    3,
                    12,
                    "[]",
                )
            )
        conn.executemany(
            """
            INSERT INTO samples (
                jump_id, time_utc, t_rel_s, lat, lon, hMSL_m,
                velN_mps, velE_mps, velD_mps, vVert_kmh, vHor_kmh,
                vTotal_kmh, angle_deg, vAcc, sAcc, gpsFix, numSV, quality_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    result = import_historical_observations()

    assert result["accepted"] == 1
    assert result["outside_catalog"] == 0
    audit = audit_catalog()
    assert audit["observations"] == 1
    with sqlite3.connect(catalog_db) as conn:
        observation = conn.execute(
            """
            SELECT dropzone_id, jump_id, quality_status, details_json
            FROM dropzone_observations
            """
        ).fetchone()
        jump_assignment = conn.execute(
            "SELECT dropzone_id, dropzone_zone_id FROM jumps WHERE jump_id = 'jump-test'"
        ).fetchone()
        metadata_audit = json.loads(conn.execute("SELECT audit_json FROM dropzone_catalog_metadata").fetchone()[0])
    assert observation[:3] == ("dz-test", "jump-test", "accepted")
    assert json.loads(observation[3])["algorithm"] == "stable-ground-v1"
    assert jump_assignment == (None, None)
    assert metadata_audit["observations"] == 1
