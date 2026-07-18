from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from statistics import median
from typing import Any

from app.config import BASE_DIR
from app.database import get_connection

DEFAULT_CATALOG_PATH = BASE_DIR / "data" / "dropzones_seed.json"
_DROPZONE_STATUSES = {"candidate", "trusted", "verified", "inactive"}
_OPERATOR_STATUSES = {"listed", "confirmed", "inactive"}
_ZONE_KINDS = {"primary", "landing", "alternate", "historical"}
_ZONE_STATUSES = {"active", "candidate", "inactive"}
_ASSIGNMENT_STATUSES = {"active", "candidate", "inactive"}
_TRUST_LEVELS = {"discovery", "supporting", "primary"}
_SOURCE_KINDS = {
    "directory",
    "official",
    "airport_registry",
    "terrain",
    "historical_gps",
    "manual",
}


class CatalogError(ValueError):
    pass


def load_catalog(path: Path | str = DEFAULT_CATALOG_PATH) -> dict[str, Any]:
    catalog_path = Path(path)
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogError(f"Dropzone-Katalog fehlt: {catalog_path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Dropzone-Katalog ist kein gültiges JSON: {exc}") from exc
    validate_catalog(payload)
    return payload


def validate_catalog(catalog: dict[str, Any]) -> None:
    if not isinstance(catalog, dict):
        raise CatalogError("Der Dropzone-Katalog muss ein JSON-Objekt sein.")
    if not str(catalog.get("catalog_version") or "").strip():
        raise CatalogError("catalog_version fehlt.")

    dropzones = _require_list(catalog, "dropzones")
    operators = _require_list(catalog, "operators")
    zones = _require_list(catalog, "zones")
    assignments = _require_list(catalog, "operator_assignments")
    sources = _require_list(catalog, "sources")

    dropzone_ids = _unique_ids(dropzones, "dropzone_id")
    operator_ids = _unique_ids(operators, "operator_id")
    zone_ids = _unique_ids(zones, "zone_id")
    _unique_ids(sources, "source_id")

    for row in dropzones:
        _require_text(row, "name")
        _require_text(row, "country_code")
        _validate_coordinate(row)
        if row.get("status") not in _DROPZONE_STATUSES:
            raise CatalogError(f"Ungültiger Dropzone-Status: {row.get('status')}")
        if float(row.get("match_radius_m") or 0.0) <= 0.0:
            raise CatalogError(f"Ungültiger Erkennungsradius für {row['dropzone_id']}")

    for row in operators:
        _require_text(row, "name")
        _require_text(row, "country_code")
        if row.get("status") not in _OPERATOR_STATUSES:
            raise CatalogError(f"Ungültiger Betreiberstatus: {row.get('status')}")

    for row in zones:
        if row.get("dropzone_id") not in dropzone_ids:
            raise CatalogError(f"Zone verweist auf unbekannte Dropzone: {row.get('dropzone_id')}")
        _require_text(row, "name")
        _validate_coordinate(row)
        if row.get("zone_kind", "landing") not in _ZONE_KINDS:
            raise CatalogError(f"Ungültige Zonenart: {row.get('zone_kind')}")
        if row.get("status", "active") not in _ZONE_STATUSES:
            raise CatalogError(f"Ungültiger Zonenstatus: {row.get('status')}")
        if float(row.get("match_radius_m") or 0.0) <= 0.0:
            raise CatalogError(f"Ungültiger Erkennungsradius für {row['zone_id']}")

    assignment_keys: set[tuple[str, str]] = set()
    for row in assignments:
        if row.get("dropzone_id") not in dropzone_ids:
            raise CatalogError(f"Betreiberzuordnung verweist auf unbekannte Dropzone: {row}")
        if row.get("operator_id") not in operator_ids:
            raise CatalogError(f"Betreiberzuordnung verweist auf unbekannten Betreiber: {row}")
        key = (str(row["dropzone_id"]), str(row["operator_id"]))
        if key in assignment_keys:
            raise CatalogError(f"Doppelte Betreiberzuordnung: {key}")
        assignment_keys.add(key)
        if row.get("status", "active") not in _ASSIGNMENT_STATUSES:
            raise CatalogError(f"Ungültiger Zuordnungsstatus: {row.get('status')}")

    for row in sources:
        if row.get("source_kind") not in _SOURCE_KINDS:
            raise CatalogError(f"Ungültige Quellenart: {row.get('source_kind')}")
        if row.get("dropzone_id") and row["dropzone_id"] not in dropzone_ids:
            raise CatalogError(f"Quelle verweist auf unbekannte Dropzone: {row}")
        if row.get("operator_id") and row["operator_id"] not in operator_ids:
            raise CatalogError(f"Quelle verweist auf unbekannten Betreiber: {row}")
        if row.get("zone_id") and row["zone_id"] not in zone_ids:
            raise CatalogError(f"Quelle verweist auf unbekannte Zone: {row}")
        if not any(row.get(key) for key in ("dropzone_id", "operator_id", "zone_id")):
            raise CatalogError(f"Quelle hat kein Bezugsobjekt: {row}")
        if row.get("trust_level", "supporting") not in _TRUST_LEVELS:
            raise CatalogError(f"Ungültige Quellenvertrauensstufe: {row.get('trust_level')}")
        _require_text(row, "source_name")
        _require_text(row, "retrieved_at")

    zone_dropzone_ids = {str(row["dropzone_id"]) for row in zones}
    source_dropzone_ids = {str(row["dropzone_id"]) for row in sources if row.get("dropzone_id")}
    assigned_operator_ids = {str(row["operator_id"]) for row in assignments}
    missing_zones = dropzone_ids - zone_dropzone_ids
    missing_sources = dropzone_ids - source_dropzone_ids
    missing_assignments = operator_ids - assigned_operator_ids
    if missing_zones:
        raise CatalogError(f"Dropzones ohne Zone: {sorted(missing_zones)}")
    if missing_sources:
        raise CatalogError(f"Dropzones ohne Quelle: {sorted(missing_sources)}")
    if missing_assignments:
        raise CatalogError(f"Betreiber ohne Dropzone: {sorted(missing_assignments)}")
    for row in dropzones:
        if row["status"] in {"trusted", "verified"} and row.get("ground_elevation_m") is None:
            raise CatalogError(f"Vertrauenswürdige Dropzone ohne Bodenhöhe: {row['dropzone_id']}")


def sync_catalog(
    catalog: dict[str, Any],
    *,
    conn: sqlite3.Connection | None = None,
    source_bytes: bytes | None = None,
) -> dict[str, Any]:
    validate_catalog(catalog)
    if conn is None:
        with get_connection() as owned_conn:
            result = _sync_catalog(owned_conn, catalog, source_bytes=source_bytes)
            owned_conn.commit()
            return result
    result = _sync_catalog(conn, catalog, source_bytes=source_bytes)
    conn.commit()
    return result


def sync_catalog_file(path: Path | str = DEFAULT_CATALOG_PATH) -> dict[str, Any]:
    catalog_path = Path(path)
    try:
        source_bytes = catalog_path.read_bytes()
        catalog = json.loads(source_bytes.decode("utf-8"))
    except FileNotFoundError as exc:
        raise CatalogError(f"Dropzone-Katalog fehlt: {catalog_path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Dropzone-Katalog ist kein gültiges UTF-8-JSON: {exc}") from exc
    return sync_catalog(catalog, source_bytes=source_bytes)


def audit_catalog(*, conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    if conn is None:
        with get_connection() as owned_conn:
            return _audit_catalog(owned_conn)
    return _audit_catalog(conn)


def import_historical_observations(
    *,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Extract reproducible landing observations without assigning jumps.

    The importer uses only stable, low-speed GNSS sequences after the freefall.
    It records evidence for known catalog zones, but deliberately does not
    mutate ``jumps.dropzone_id`` or a catalog ground elevation.
    """
    if conn is None:
        with get_connection() as owned_conn:
            result = _import_historical_observations(owned_conn)
            owned_conn.commit()
            return result
    result = _import_historical_observations(conn)
    conn.commit()
    return result


def _import_historical_observations(conn: sqlite3.Connection) -> dict[str, Any]:
    zones = [
        dict(row)
        for row in conn.execute(
            """
            SELECT z.zone_id, z.dropzone_id, z.latitude, z.longitude, z.match_radius_m
            FROM dropzone_zones z
            JOIN dropzones d ON d.dropzone_id = z.dropzone_id
            WHERE z.status IN ('active', 'candidate') AND d.status <> 'inactive'
            """
        )
    ]
    if not zones:
        raise CatalogError("Historienimport nicht möglich: Der Dropzone-Katalog ist leer.")

    jump_ids = [
        str(row[0])
        for row in conn.execute(
            """
            SELECT DISTINCT jump_id
            FROM samples
            WHERE t_rel_s >= 45.0
              AND lat IS NOT NULL AND lon IS NOT NULL
              AND gpsFix >= 3 AND numSV >= 6
              AND vTotal_kmh <= 10.0 AND ABS(velD_mps) <= 1.0
              AND (sAcc IS NULL OR sAcc <= 3.0)
              AND (vAcc IS NULL OR vAcc <= 10.0)
            ORDER BY jump_id
            """
        )
    ]

    conn.execute(
        "DELETE FROM dropzone_observations WHERE source_kind = 'historical_gps' AND observation_id LIKE 'hist-jump-%'"
    )
    result: dict[str, Any] = {
        "candidate_jumps": len(jump_ids),
        "accepted": 0,
        "no_stable_sequence": 0,
        "outside_catalog": 0,
        "by_dropzone": {},
    }
    for jump_id in jump_ids:
        rows = list(
            conn.execute(
                """
                SELECT time_utc, t_rel_s, lat, lon, hMSL_m
                FROM samples
                WHERE jump_id = ? AND t_rel_s >= 45.0
                  AND lat IS NOT NULL AND lon IS NOT NULL
                  AND gpsFix >= 3 AND numSV >= 6
                  AND vTotal_kmh <= 10.0 AND ABS(velD_mps) <= 1.0
                  AND (sAcc IS NULL OR sAcc <= 3.0)
                  AND (vAcc IS NULL OR vAcc <= 10.0)
                ORDER BY t_rel_s
                """,
                (jump_id,),
            )
        )
        observation = _first_stable_ground_sequence(rows)
        if observation is None:
            result["no_stable_sequence"] += 1
            continue

        nearest = min(
            zones,
            key=lambda zone: _haversine_m(
                observation["latitude"],
                observation["longitude"],
                float(zone["latitude"]),
                float(zone["longitude"]),
            ),
        )
        distance_m = _haversine_m(
            observation["latitude"],
            observation["longitude"],
            float(nearest["latitude"]),
            float(nearest["longitude"]),
        )
        if distance_m > float(nearest["match_radius_m"]):
            result["outside_catalog"] += 1
            continue

        details = {
            "algorithm": "stable-ground-v1",
            "distance_to_zone_m": round(distance_m, 1),
            "coordinate_p95_radius_m": observation["coordinate_p95_radius_m"],
            "thresholds": {
                "minimum_duration_s": 5.0,
                "maximum_gap_s": 0.35,
                "maximum_altitude_mad_m": 5.0,
                "maximum_coordinate_p95_radius_m": 100.0,
                "maximum_total_speed_kmh": 10.0,
            },
        }
        conn.execute(
            """
            INSERT INTO dropzone_observations (
                observation_id, dropzone_id, jump_id, observed_at, latitude, longitude,
                ground_elevation_m, altitude_mad_m, sample_count, duration_s,
                source_kind, quality_status, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'historical_gps', 'accepted', ?)
            """,
            (
                f"hist-jump-{jump_id}",
                nearest["dropzone_id"],
                jump_id,
                observation["observed_at"],
                observation["latitude"],
                observation["longitude"],
                observation["ground_elevation_m"],
                observation["altitude_mad_m"],
                observation["sample_count"],
                observation["duration_s"],
                json.dumps(details, ensure_ascii=False, sort_keys=True),
            ),
        )
        result["accepted"] += 1
        counts = result["by_dropzone"]
        counts[nearest["dropzone_id"]] = int(counts.get(nearest["dropzone_id"], 0)) + 1

    result["by_dropzone"] = dict(sorted(result["by_dropzone"].items()))
    current_audit = _audit_catalog(conn)
    conn.execute(
        "UPDATE dropzone_catalog_metadata SET audit_json = ? WHERE catalog_key = 'default'",
        (json.dumps(current_audit, ensure_ascii=False, sort_keys=True),),
    )
    return result


def _first_stable_ground_sequence(rows: list[sqlite3.Row]) -> dict[str, Any] | None:
    segment: list[sqlite3.Row] = []
    for row in rows:
        if segment and float(row["t_rel_s"]) - float(segment[-1]["t_rel_s"]) > 0.35:
            candidate = _summarize_ground_segment(segment)
            if candidate is not None:
                return candidate
            segment = []
        segment.append(row)
    return _summarize_ground_segment(segment)


def _summarize_ground_segment(segment: list[sqlite3.Row]) -> dict[str, Any] | None:
    if len(segment) < 10:
        return None
    duration_s = float(segment[-1]["t_rel_s"]) - float(segment[0]["t_rel_s"])
    if duration_s < 5.0:
        return None

    latitudes = [float(row["lat"]) for row in segment]
    longitudes = [float(row["lon"]) for row in segment]
    altitudes = [float(row["hMSL_m"]) for row in segment]
    latitude = float(median(latitudes))
    longitude = float(median(longitudes))
    ground_elevation_m = float(median(altitudes))
    altitude_mad_m = float(median(abs(value - ground_elevation_m) for value in altitudes))
    radii = sorted(_haversine_m(latitude, longitude, lat, lon) for lat, lon in zip(latitudes, longitudes, strict=True))
    p95_index = min(len(radii) - 1, max(0, math.ceil(len(radii) * 0.95) - 1))
    coordinate_p95_radius_m = float(radii[p95_index])
    if altitude_mad_m > 5.0 or coordinate_p95_radius_m > 100.0:
        return None
    return {
        "observed_at": str(segment[0]["time_utc"]),
        "latitude": round(latitude, 7),
        "longitude": round(longitude, 7),
        "ground_elevation_m": round(ground_elevation_m, 2),
        "altitude_mad_m": round(altitude_mad_m, 3),
        "coordinate_p95_radius_m": round(coordinate_p95_radius_m, 2),
        "sample_count": len(segment),
        "duration_s": round(duration_s, 2),
    }


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = math.radians(lon2 - lon1)
    value = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    return 2.0 * radius_m * math.asin(math.sqrt(value))


def _sync_catalog(
    conn: sqlite3.Connection,
    catalog: dict[str, Any],
    *,
    source_bytes: bytes | None,
) -> dict[str, Any]:
    for row in catalog["dropzones"]:
        conn.execute(
            """
            INSERT INTO dropzones (
                dropzone_id, name, country_code, region, locality, icao_code, airport_ident,
                latitude, longitude, match_radius_m, ground_elevation_m, published_elevation_m,
                ground_elevation_uncertainty_m, ground_elevation_source, status, catalog_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dropzone_id) DO UPDATE SET
                name = excluded.name,
                country_code = excluded.country_code,
                region = excluded.region,
                locality = excluded.locality,
                icao_code = excluded.icao_code,
                airport_ident = excluded.airport_ident,
                latitude = excluded.latitude,
                longitude = excluded.longitude,
                match_radius_m = excluded.match_radius_m,
                ground_elevation_m = CASE
                    WHEN dropzones.status = 'verified' THEN dropzones.ground_elevation_m
                    ELSE excluded.ground_elevation_m
                END,
                published_elevation_m = excluded.published_elevation_m,
                ground_elevation_uncertainty_m = CASE
                    WHEN dropzones.status = 'verified' THEN dropzones.ground_elevation_uncertainty_m
                    ELSE excluded.ground_elevation_uncertainty_m
                END,
                ground_elevation_source = CASE
                    WHEN dropzones.status = 'verified' THEN dropzones.ground_elevation_source
                    ELSE excluded.ground_elevation_source
                END,
                status = CASE
                    WHEN dropzones.status = 'verified' THEN 'verified'
                    ELSE excluded.status
                END,
                catalog_revision = MAX(dropzones.catalog_revision, excluded.catalog_revision),
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                row["dropzone_id"],
                row["name"],
                row["country_code"],
                row.get("region"),
                row.get("locality"),
                row.get("icao_code"),
                row.get("airport_ident"),
                row["latitude"],
                row["longitude"],
                row["match_radius_m"],
                row.get("ground_elevation_m"),
                row.get("published_elevation_m"),
                row.get("ground_elevation_uncertainty_m"),
                row.get("ground_elevation_source"),
                row["status"],
                row.get("catalog_revision", 1),
            ),
        )

    for row in catalog["zones"]:
        conn.execute(
            """
            INSERT INTO dropzone_zones (
                zone_id, dropzone_id, name, zone_kind, latitude, longitude, match_radius_m,
                ground_elevation_m, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(zone_id) DO UPDATE SET
                dropzone_id = excluded.dropzone_id,
                name = excluded.name,
                zone_kind = excluded.zone_kind,
                latitude = excluded.latitude,
                longitude = excluded.longitude,
                match_radius_m = excluded.match_radius_m,
                ground_elevation_m = excluded.ground_elevation_m,
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                row["zone_id"],
                row["dropzone_id"],
                row["name"],
                row.get("zone_kind", "landing"),
                row["latitude"],
                row["longitude"],
                row.get("match_radius_m", 1500.0),
                row.get("ground_elevation_m"),
                row.get("status", "active"),
            ),
        )

    for row in catalog["operators"]:
        conn.execute(
            """
            INSERT INTO dropzone_operators (
                operator_id, name, country_code, website_url, status
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(operator_id) DO UPDATE SET
                name = excluded.name,
                country_code = excluded.country_code,
                website_url = excluded.website_url,
                status = CASE
                    WHEN dropzone_operators.status IN ('confirmed', 'inactive')
                        THEN dropzone_operators.status
                    ELSE excluded.status
                END,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                row["operator_id"],
                row["name"],
                row["country_code"],
                row.get("website_url"),
                row.get("status", "listed"),
            ),
        )

    for row in catalog["operator_assignments"]:
        conn.execute(
            """
            INSERT INTO dropzone_operator_assignments (
                dropzone_id, operator_id, is_primary, status
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(dropzone_id, operator_id) DO UPDATE SET
                is_primary = excluded.is_primary,
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                row["dropzone_id"],
                row["operator_id"],
                1 if row.get("is_primary") else 0,
                row.get("status", "active"),
            ),
        )

    for row in catalog["sources"]:
        conn.execute(
            """
            INSERT INTO dropzone_sources (
                source_id, dropzone_id, zone_id, operator_id, source_kind, source_name,
                source_url, source_ref, trust_level, supports_fields_json, details_json,
                retrieved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                dropzone_id = excluded.dropzone_id,
                zone_id = excluded.zone_id,
                operator_id = excluded.operator_id,
                source_kind = excluded.source_kind,
                source_name = excluded.source_name,
                source_url = excluded.source_url,
                source_ref = excluded.source_ref,
                trust_level = excluded.trust_level,
                supports_fields_json = excluded.supports_fields_json,
                details_json = excluded.details_json,
                retrieved_at = excluded.retrieved_at
            """,
            (
                row["source_id"],
                row.get("dropzone_id"),
                row.get("zone_id"),
                row.get("operator_id"),
                row["source_kind"],
                row["source_name"],
                row.get("source_url"),
                row.get("source_ref"),
                row.get("trust_level", "supporting"),
                json.dumps(row.get("supports_fields", []), ensure_ascii=False),
                json.dumps(row.get("details", {}), ensure_ascii=False, sort_keys=True),
                row["retrieved_at"],
            ),
        )

    audit = _audit_catalog(conn)
    digest_payload = source_bytes or json.dumps(catalog, ensure_ascii=False, sort_keys=True).encode("utf-8")
    conn.execute(
        """
        INSERT INTO dropzone_catalog_metadata (
            catalog_key, catalog_version, source_sha256, dropzone_count, operator_count, audit_json
        ) VALUES ('default', ?, ?, ?, ?, ?)
        ON CONFLICT(catalog_key) DO UPDATE SET
            catalog_version = excluded.catalog_version,
            source_sha256 = excluded.source_sha256,
            dropzone_count = excluded.dropzone_count,
            operator_count = excluded.operator_count,
            imported_at = CURRENT_TIMESTAMP,
            audit_json = excluded.audit_json
        """,
        (
            catalog["catalog_version"],
            hashlib.sha256(digest_payload).hexdigest(),
            len(catalog["dropzones"]),
            len(catalog["operators"]),
            json.dumps(audit, ensure_ascii=False, sort_keys=True),
        ),
    )
    return audit


def _audit_catalog(conn: sqlite3.Connection) -> dict[str, Any]:
    def count(query: str, params: tuple[Any, ...] = ()) -> int:
        return int(conn.execute(query, params).fetchone()[0])

    status_counts = {
        str(row[0]): int(row[1])
        for row in conn.execute("SELECT status, COUNT(*) FROM dropzones GROUP BY status ORDER BY status")
    }
    country_counts = {
        str(row[0]): int(row[1])
        for row in conn.execute(
            "SELECT country_code, COUNT(*) FROM dropzones GROUP BY country_code ORDER BY country_code"
        )
    }
    source_counts = {
        str(row[0]): int(row[1])
        for row in conn.execute(
            "SELECT source_kind, COUNT(*) FROM dropzone_sources GROUP BY source_kind ORDER BY source_kind"
        )
    }
    match_status_counts = {
        str(row[0]): int(row[1])
        for row in conn.execute(
            "SELECT match_status, COUNT(*) FROM dropzone_match_attempts GROUP BY match_status ORDER BY match_status"
        )
    }
    issues: list[str] = []
    checks = [
        (
            "dropzones_without_zone",
            """SELECT COUNT(*) FROM dropzones d
               WHERE NOT EXISTS (SELECT 1 FROM dropzone_zones z WHERE z.dropzone_id = d.dropzone_id)""",
        ),
        (
            "trusted_without_ground_elevation",
            "SELECT COUNT(*) FROM dropzones WHERE status IN ('trusted', 'verified') AND ground_elevation_m IS NULL",
        ),
        (
            "dropzones_without_source",
            """SELECT COUNT(*) FROM dropzones d
               WHERE NOT EXISTS (SELECT 1 FROM dropzone_sources s WHERE s.dropzone_id = d.dropzone_id)""",
        ),
        (
            "operators_without_dropzone",
            """SELECT COUNT(*) FROM dropzone_operators o
               WHERE NOT EXISTS (
                   SELECT 1 FROM dropzone_operator_assignments a WHERE a.operator_id = o.operator_id
               )""",
        ),
    ]
    check_counts: dict[str, int] = {}
    for key, query in checks:
        value = count(query)
        check_counts[key] = value
        if value:
            issues.append(f"{key}: {value}")

    return {
        "dropzones": count("SELECT COUNT(*) FROM dropzones"),
        "zones": count("SELECT COUNT(*) FROM dropzone_zones"),
        "operators": count("SELECT COUNT(*) FROM dropzone_operators"),
        "operator_assignments": count("SELECT COUNT(*) FROM dropzone_operator_assignments"),
        "sources": count("SELECT COUNT(*) FROM dropzone_sources"),
        "observations": count("SELECT COUNT(*) FROM dropzone_observations"),
        "match_attempts": count("SELECT COUNT(*) FROM dropzone_match_attempts"),
        "match_status_counts": match_status_counts,
        "status_counts": status_counts,
        "country_counts": country_counts,
        "source_counts": source_counts,
        "checks": check_counts,
        "issues": issues,
        "ok": not issues,
    }


def _require_list(catalog: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = catalog.get(key)
    if not isinstance(value, list):
        raise CatalogError(f"{key} muss eine Liste sein.")
    if any(not isinstance(row, dict) for row in value):
        raise CatalogError(f"{key} darf nur JSON-Objekte enthalten.")
    return value


def _unique_ids(rows: list[dict[str, Any]], key: str) -> set[str]:
    result: set[str] = set()
    for row in rows:
        value = _require_text(row, key)
        if value in result:
            raise CatalogError(f"Doppelte ID in {key}: {value}")
        result.add(value)
    return result


def _require_text(row: dict[str, Any], key: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise CatalogError(f"Pflichtfeld fehlt: {key}")
    return value


def _validate_coordinate(row: dict[str, Any]) -> None:
    try:
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CatalogError(f"Ungültige Koordinate: {row}") from exc
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        raise CatalogError(f"Koordinate außerhalb des Wertebereichs: {row}")
