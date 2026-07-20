from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import database
from app.database import init_db
from app.services import dropzone_matching
from app.services.dropzone_catalog import sync_catalog
from app.services.dropzone_matching import (
    analyze_flysight_with_dropzone,
    extract_stable_ground_observation,
    list_dropzone_match_contexts,
    match_ground_observation,
)
from app.services.storage import get_jump_report, save_analysis_result


def _zone(
    *,
    dropzone_id: str = "dz-test",
    longitude: float = 13.0,
    status: str = "trusted",
) -> dict:
    return {
        "zone_id": f"zone-{dropzone_id}",
        "dropzone_id": dropzone_id,
        "zone_name": "Test-Landezone",
        "zone_kind": "landing",
        "latitude": 52.0,
        "longitude": longitude,
        "match_radius_m": 1500.0,
        "zone_status": "active",
        "dropzone_name": f"Testplatz {dropzone_id}",
        "country_code": "DE",
        "ground_elevation_m": 42.0,
        "ground_elevation_uncertainty_m": 5.0,
        "ground_elevation_source": "test",
        "dropzone_status": status,
        "catalog_revision": 1,
        "catalog_version": "test-1",
    }


def _observation() -> dict:
    return {
        "observed_at": "2026-07-18T10:00:00Z",
        "latitude": 52.0,
        "longitude": 13.0,
        "ground_elevation_m": 43.0,
        "altitude_mad_m": 0.2,
        "coordinate_p95_radius_m": 1.0,
        "sample_count": 61,
        "duration_s": 6.0,
    }


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
                "published_elevation_m": 42.0,
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
        "operators": [],
        "operator_assignments": [],
        "sources": [
            {
                "source_id": "source-test",
                "dropzone_id": "dz-test",
                "source_kind": "manual",
                "source_name": "Testquelle",
                "trust_level": "primary",
                "retrieved_at": "2026-07-18",
            }
        ],
    }


def _jump_with_ground_sequence() -> bytes:
    dt = 0.2
    t = np.arange(0.0, 100.2, dt)
    base_time = datetime(2026, 7, 18, 10, 0, 0, tzinfo=timezone.utc)
    vel_d = np.piecewise(
        t,
        [
            t < 5.0,
            (t >= 5.0) & (t < 10.0),
            (t >= 10.0) & (t < 25.0),
            (t >= 25.0) & (t < 35.0),
            (t >= 35.0) & (t < 75.0),
            t >= 75.0,
        ],
        [
            0.0,
            lambda x: (x - 5.0) * 12.0,
            lambda x: 60.0 + (x - 10.0) * 3.2,
            lambda x: 108.0 - (x - 25.0) * 10.0,
            8.0,
            0.0,
        ],
    )
    vel_d = np.clip(vel_d, 0.0, None)
    vel_n = np.where(t < 5.0, 45.0, np.where(t < 35.0, np.maximum(12.0, 42.0 - t), np.where(t < 75.0, 8.0, 0.0)))
    vel_e = np.where(t < 75.0, 1.0, 0.0)
    altitude = [3600.0]
    for index in range(1, len(t)):
        altitude.append(altitude[-1] - float(vel_d[index]) * dt)
    altitude_array = np.asarray(altitude)
    ground = float(altitude_array[t >= 75.0][0])
    altitude_array[t >= 75.0] = ground
    latitude = np.where(t < 75.0, 51.99, 52.0)
    longitude = np.where(t < 75.0, 12.99, 13.0)
    frame = pd.DataFrame(
        {
            "time": [(base_time + timedelta(seconds=float(value))).isoformat() for value in t],
            "lat": latitude,
            "lon": longitude,
            "hMSL": altitude_array,
            "velN": vel_n,
            "velE": vel_e,
            "velD": vel_d,
            "hAcc": 1.0,
            "vAcc": 1.0,
            "sAcc": 0.5,
            "gpsFix": 3,
            "numSV": 12,
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def test_match_accepts_only_trusted_unambiguous_zone() -> None:
    result = match_ground_observation(_observation(), [_zone()])

    assert result["status"] == "accepted"
    assert result["dropzone_id"] == "dz-test"
    assert result["assignment_source"] == "catalog_auto"
    assert result["confidence"] >= 0.75


def test_candidate_and_ambiguous_matches_are_not_accepted() -> None:
    candidate = match_ground_observation(_observation(), [_zone(status="candidate")])
    ambiguous = match_ground_observation(
        _observation(),
        [_zone(dropzone_id="dz-first"), _zone(dropzone_id="dz-second", longitude=13.002)],
    )

    assert candidate["status"] == "candidate"
    assert candidate["assignment_source"] is None
    assert ambiguous["status"] == "ambiguous"
    assert ambiguous["assignment_source"] is None


def test_manual_selection_can_confirm_candidate_without_ground_sequence() -> None:
    result = match_ground_observation(
        None,
        [_zone(status="candidate")],
        manual_dropzone_id="dz-test",
    )

    assert result["status"] == "manual"
    assert result["assignment_source"] == "catalog_manual"
    assert result["confidence"] == 1.0


def test_stable_ground_extraction_rejects_short_sequence() -> None:
    samples = []
    for index in range(20):
        samples.append(
            {
                "time_utc": f"2026-07-18T10:00:{index / 10:04.1f}Z",
                "t_rel_s": 50.0 + index / 10.0,
                "lat": 52.0,
                "lon": 13.0,
                "hMSL_m": 42.0,
                "velD_mps": 0.0,
                "vTotal_kmh": 0.0,
                "vAcc": 1.0,
                "sAcc": 0.5,
                "gpsFix": 3,
                "numSV": 12,
            }
        )

    assert extract_stable_ground_observation(samples) is None


def test_full_analysis_persists_assignment_attempt_and_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "matching.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    init_db()
    catalog = _catalog()
    expected_ground = pd.read_csv(pd.io.common.BytesIO(_jump_with_ground_sequence()))["hMSL"].iloc[-1]
    catalog["dropzones"][0]["ground_elevation_m"] = float(expected_ground)
    catalog["zones"][0]["ground_elevation_m"] = float(expected_ground)
    sync_catalog(catalog)

    result = analyze_flysight_with_dropzone(
        content=_jump_with_ground_sequence(),
        file_name="dropzone.csv",
        jumper_name="Test",
        ground_elevation_m=None,
        breakoff_altitude_agl_m=1707.0,
        zones=list_dropzone_match_contexts(),
    )
    jump_id, duplicate = save_analysis_result(result)
    report = get_jump_report(jump_id)
    full_sample_report = get_jump_report(jump_id, prefer_compact_series=False)

    assert duplicate is False
    assert report is not None
    assert full_sample_report is not None
    assert report["jump"]["ground_elevation_source"] == "dropzone_catalog"
    assert report["jump"]["dropzone_id"] == "dz-test"
    assert report["dropzone"]["name"] == "Testplatz"
    assert report["dropzone_match"]["match_status"] == "accepted"
    assert report["dropzone_match"]["confidence"] >= 0.75
    with database.get_connection() as conn:
        sample_count = int(conn.execute("SELECT COUNT(*) FROM samples WHERE jump_id = ?", (jump_id,)).fetchone()[0])
        series_count = int(
            conn.execute("SELECT point_count FROM jump_series WHERE jump_id = ?", (jump_id,)).fetchone()[0]
        )
    assert 0 < series_count < sample_count
    assert report["chart_data"]["time_s"][-1] >= float(report["notes"]["curve_window_end_s"])
    assert report["metrics"] == full_sample_report["metrics"]
    assert report["fixpoints"] == full_sample_report["fixpoints"]
    assert report["phases"] == full_sample_report["phases"]
    for key in ("time_s", "vVert_kmh", "vHor_kmh", "angle_deg", "hAGL_m", "velN_mps", "velE_mps"):
        assert report["chart_data"][key] == full_sample_report["chart_data"][key][:series_count]


def test_manual_ground_keeps_priority_over_automatic_dropzone_height() -> None:
    result = analyze_flysight_with_dropzone(
        content=_jump_with_ground_sequence(),
        file_name="manual-ground.csv",
        jumper_name="Test",
        ground_elevation_m=123.0,
        breakoff_altitude_agl_m=1707.0,
        zones=[_zone()],
    )

    assert result["dropzone_match"]["status"] == "accepted"
    assert result["jump_record"]["dropzone_id"] == "dz-test"
    assert result["jump_record"]["ground_elevation_m"] == 123.0
    assert result["jump_record"]["ground_elevation_source"] == "manual"
    assert json.loads(result["metrics_record"]["notes"])["dropzone_match_status"] == "accepted"


def test_dropzone_upload_parses_and_analyzes_track_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    parse_calls = 0
    analysis_calls = 0
    original_prepare = dropzone_matching.prepare_flysight_csv
    original_analyze = dropzone_matching.analyze_flysight_csv

    def counted_prepare(content: bytes):
        nonlocal parse_calls
        parse_calls += 1
        return original_prepare(content)

    def counted_analyze(**kwargs):
        nonlocal analysis_calls
        analysis_calls += 1
        assert kwargs["prepared_track"] is not None
        return original_analyze(**kwargs)

    monkeypatch.setattr(dropzone_matching, "prepare_flysight_csv", counted_prepare)
    monkeypatch.setattr(dropzone_matching, "analyze_flysight_csv", counted_analyze)

    result = analyze_flysight_with_dropzone(
        content=_jump_with_ground_sequence(),
        file_name="single-pass.csv",
        jumper_name="Test",
        ground_elevation_m=None,
        breakoff_altitude_agl_m=1707.0,
        zones=[_zone()],
    )

    assert result["dropzone_match"]["status"] == "accepted"
    assert parse_calls == 1
    assert analysis_calls == 1
