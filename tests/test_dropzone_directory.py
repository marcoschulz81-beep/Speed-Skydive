from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import database
from app.database import get_connection, init_db
from app.main import app
from app.services.dropzone_catalog import sync_catalog
from app.services.dropzone_directory import get_dropzone_detail, list_dropzones


def _catalog() -> dict:
    return {
        "catalog_version": "directory-test-1",
        "dropzones": [
            {
                "dropzone_id": "dz-test",
                "name": "Testplatz Nord",
                "country_code": "DE",
                "region": "DE-BB",
                "locality": "Teststadt",
                "icao_code": "EDZZ",
                "airport_ident": "EDZZ",
                "latitude": 52.0,
                "longitude": 13.0,
                "match_radius_m": 1500.0,
                "ground_elevation_m": 42.0,
                "published_elevation_m": 41.0,
                "ground_elevation_uncertainty_m": 5.0,
                "ground_elevation_source": "test",
                "status": "trusted",
                "catalog_revision": 2,
            },
            {
                "dropzone_id": "dz-candidate",
                "name": "Prüfplatz Süd",
                "country_code": "AT",
                "latitude": 48.0,
                "longitude": 14.0,
                "match_radius_m": 1200.0,
                "ground_elevation_m": None,
                "published_elevation_m": None,
                "ground_elevation_uncertainty_m": None,
                "ground_elevation_source": None,
                "status": "candidate",
                "catalog_revision": 1,
            },
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
            },
            {
                "zone_id": "zone-candidate",
                "dropzone_id": "dz-candidate",
                "name": "Prüf-Landezone",
                "zone_kind": "landing",
                "latitude": 48.0,
                "longitude": 14.0,
                "match_radius_m": 1200.0,
                "ground_elevation_m": None,
                "status": "candidate",
            },
        ],
        "operators": [
            {
                "operator_id": "operator-test",
                "name": "Testbetreiber",
                "country_code": "DE",
                "website_url": "javascript:alert(1)",
                "status": "confirmed",
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
                "source_kind": "official",
                "source_name": "Offizielle Testquelle",
                "source_url": "https://example.test/dropzone",
                "source_ref": "TEST-REF",
                "trust_level": "primary",
                "supports_fields": ["name", "ground_elevation_m"],
                "details": {"note": "test"},
                "retrieved_at": "2026-07-18",
            },
            {
                "source_id": "source-zone",
                "zone_id": "zone-test",
                "source_kind": "terrain",
                "source_name": "Zonenhöhe",
                "source_url": "javascript:alert(2)",
                "trust_level": "supporting",
                "supports_fields": ["ground_elevation_m"],
                "details": {},
                "retrieved_at": "2026-07-18",
            },
            {
                "source_id": "source-candidate",
                "dropzone_id": "dz-candidate",
                "source_kind": "directory",
                "source_name": "Kandidatenquelle",
                "trust_level": "discovery",
                "supports_fields": ["name"],
                "details": {},
                "retrieved_at": "2026-07-18",
            },
        ],
    }


@pytest.fixture
def directory_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "directory.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    init_db()
    sync_catalog(_catalog())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO jumps (
                jump_id, jumper_name, file_name, device_type, raw_start_time_utc, t0_utc,
                exit_altitude_msl_m, is_valid_altitude, sample_rate_hz, quality_score, quality_flags,
                dropzone_id, dropzone_zone_id, dropzone_assignment_source,
                dropzone_assignment_confidence, dropzone_distance_m
            ) VALUES (
                'jump-private', 'Nicht anzeigen', 'private.csv', 'FlySight',
                '2026-07-18T10:00:00Z', '2026-07-18T10:01:00Z',
                4000.0, 1, 10.0, 1.0, '[]',
                'dz-test', 'zone-test', 'catalog_auto', 0.91, 84.0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO dropzone_match_attempts (
                attempt_id, jump_id, algorithm_version, catalog_version, match_status,
                assignment_source, nearest_dropzone_id, nearest_zone_id,
                nearest_distance_m, confidence, details_json
            ) VALUES (
                'attempt-test', 'jump-private', 'test-1', 'directory-test-1', 'accepted',
                'catalog_auto', 'dz-test', 'zone-test', 84.0, 0.91, '{}'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO dropzone_observations (
                observation_id, dropzone_id, jump_id, observed_at, latitude, longitude,
                ground_elevation_m, altitude_mad_m, sample_count, duration_s,
                source_kind, quality_status, details_json
            ) VALUES (
                'observation-test', 'dz-test', 'jump-private', '2026-07-18T10:02:00Z',
                52.0, 13.0, 42.5, 0.8, 60, 6.0, 'historical_gps', 'accepted', '{}'
            )
            """
        )
        conn.commit()
    return db_path


def test_directory_filters_by_name_location_code_operator_country_and_status(directory_db: Path) -> None:
    assert [row["dropzone_id"] for row in list_dropzones(query="Teststadt")["items"]] == ["dz-test"]
    assert [row["dropzone_id"] for row in list_dropzones(query="EDZZ")["items"]] == ["dz-test"]
    assert [row["dropzone_id"] for row in list_dropzones(query="Testbetreiber")["items"]] == ["dz-test"]

    result = list_dropzones(country_code="de", status="trusted")
    assert result["total_count"] == 2
    assert result["filtered_count"] == 1
    assert result["items"][0]["source_count"] == 2
    assert result["items"][0]["assigned_jump_count"] == 1
    assert result["filters"] == {"query": "", "country_code": "DE", "status": "trusted"}


def test_detail_contains_evidence_and_only_aggregate_jump_data(directory_db: Path) -> None:
    detail = get_dropzone_detail("dz-test")

    assert detail is not None
    assert detail["dropzone"]["name"] == "Testplatz Nord"
    assert detail["metadata"]["catalog_version"] == "directory-test-1"
    assert [row["name"] for row in detail["zones"]] == ["Test-Landezone"]
    assert detail["operators"][0]["website_url"] is None
    assert len(detail["sources"]) == 2
    assert detail["sources"][0]["supports_fields"] == ["name", "ground_elevation_m"]
    assert detail["sources"][1]["source_url"] is None
    assert detail["assignment_summary"]["total"] == 1
    assert detail["assignment_summary"]["automatic"] == 1
    assert detail["assignment_summary"]["average_confidence"] == pytest.approx(0.91)
    assert detail["match_statuses"][0]["match_status"] == "accepted"
    assert detail["observation_groups"][0]["observation_count"] == 1
    assert "jumper_name" not in detail["assignment_summary"]


def test_dropzone_pages_render_filters_details_and_404(directory_db: Path) -> None:
    with TestClient(app) as client:
        overview = client.get("/dropzones?q=Testbetreiber&country=DE&status=trusted&view=simple")
        detail = client.get("/dropzones/dz-test?view=expert")
        missing = client.get("/dropzones/not-there")

    assert overview.status_code == 200
    assert "Testplatz Nord" in overview.text
    assert "Prüfplatz Süd" not in overview.text
    assert 'href="/dropzones/dz-test?view=simple"' in overview.text
    assert detail.status_code == 200
    assert "Quellen und Nachweise" in detail.text
    assert "Offizielle Testquelle" in detail.text
    assert "Nicht anzeigen" not in detail.text
    assert "javascript:" not in detail.text
    assert missing.status_code == 404


def test_report_template_links_assigned_dropzone_to_catalog() -> None:
    template = (Path(__file__).parents[1] / "app" / "templates" / "jump_detail.html").read_text(encoding="utf-8")
    assert "/dropzones/{{ report.dropzone.dropzone_id | urlencode }}?view={{ vm }}" in template
