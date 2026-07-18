from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from app.database import get_connection

DROPZONE_STATUSES = {"candidate", "trusted", "verified", "inactive"}


def list_dropzones(
    *,
    query: str | None = None,
    country_code: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Return the read-only catalog overview with aggregate, non-personal statistics."""
    normalized_query = " ".join(str(query or "").split())[:120]
    normalized_country = str(country_code or "").strip().upper()[:2]
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in DROPZONE_STATUSES:
        normalized_status = ""

    conditions: list[str] = []
    parameters: list[Any] = []
    if normalized_query:
        pattern = f"%{normalized_query.lower()}%"
        conditions.append(
            """
            (
                LOWER(d.name) LIKE ? OR LOWER(COALESCE(d.locality, '')) LIKE ?
                OR LOWER(COALESCE(d.region, '')) LIKE ? OR LOWER(COALESCE(d.icao_code, '')) LIKE ?
                OR LOWER(COALESCE(d.airport_ident, '')) LIKE ?
                OR EXISTS (
                    SELECT 1
                    FROM dropzone_operator_assignments search_assignment
                    JOIN dropzone_operators search_operator
                      ON search_operator.operator_id = search_assignment.operator_id
                    WHERE search_assignment.dropzone_id = d.dropzone_id
                      AND LOWER(search_operator.name) LIKE ?
                )
            )
            """
        )
        parameters.extend([pattern] * 6)
    if normalized_country:
        conditions.append("d.country_code = ?")
        parameters.append(normalized_country)
    if normalized_status:
        conditions.append("d.status = ?")
        parameters.append(normalized_status)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT d.*,
                   (SELECT COUNT(*) FROM dropzone_zones z WHERE z.dropzone_id = d.dropzone_id) AS zone_count,
                   (SELECT COUNT(*) FROM dropzone_sources s WHERE s.dropzone_id = d.dropzone_id) AS direct_source_count,
                   (
                       SELECT COUNT(DISTINCT s.source_id)
                       FROM dropzone_sources s
                       WHERE s.dropzone_id = d.dropzone_id
                          OR EXISTS (
                              SELECT 1 FROM dropzone_zones source_zone
                              WHERE source_zone.dropzone_id = d.dropzone_id
                                AND source_zone.zone_id = s.zone_id
                          )
                          OR EXISTS (
                              SELECT 1 FROM dropzone_operator_assignments source_assignment
                              WHERE source_assignment.dropzone_id = d.dropzone_id
                                AND source_assignment.operator_id = s.operator_id
                          )
                   ) AS source_count,
                   (SELECT COUNT(*) FROM jumps j WHERE j.dropzone_id = d.dropzone_id) AS assigned_jump_count,
                   (
                       SELECT GROUP_CONCAT(o.name, ' | ')
                       FROM dropzone_operator_assignments a
                       JOIN dropzone_operators o ON o.operator_id = a.operator_id
                       WHERE a.dropzone_id = d.dropzone_id AND a.status = 'active'
                   ) AS operator_names
            FROM dropzones d
            {where_clause}
            ORDER BY d.country_code, d.name COLLATE NOCASE
            """,
            parameters,
        ).fetchall()
        countries = [
            dict(row)
            for row in conn.execute(
                """
                SELECT country_code, COUNT(*) AS dropzone_count
                FROM dropzones
                GROUP BY country_code
                ORDER BY country_code
                """
            ).fetchall()
        ]
        status_counts = {
            str(row["status"]): int(row["dropzone_count"])
            for row in conn.execute(
                """
                SELECT status, COUNT(*) AS dropzone_count
                FROM dropzones
                GROUP BY status
                ORDER BY status
                """
            ).fetchall()
        }
        metadata_row = conn.execute(
            """
            SELECT catalog_version, dropzone_count, operator_count, imported_at
            FROM dropzone_catalog_metadata
            WHERE catalog_key = 'default'
            LIMIT 1
            """
        ).fetchone()

    items = [dict(row) for row in rows]
    return {
        "items": items,
        "filters": {
            "query": normalized_query,
            "country_code": normalized_country,
            "status": normalized_status,
        },
        "countries": countries,
        "status_counts": status_counts,
        "total_count": sum(status_counts.values()),
        "filtered_count": len(items),
        "metadata": None if metadata_row is None else dict(metadata_row),
    }


def get_dropzone_detail(dropzone_id: str) -> dict[str, Any] | None:
    """Return complete catalog evidence and aggregate usage for a single dropzone."""
    normalized_id = str(dropzone_id or "").strip()
    if not normalized_id:
        return None

    with get_connection() as conn:
        dropzone_row = conn.execute(
            "SELECT * FROM dropzones WHERE dropzone_id = ? LIMIT 1",
            (normalized_id,),
        ).fetchone()
        if dropzone_row is None:
            return None

        zones = [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM dropzone_zones
                WHERE dropzone_id = ?
                ORDER BY
                    CASE zone_kind
                        WHEN 'primary' THEN 0
                        WHEN 'landing' THEN 1
                        WHEN 'alternate' THEN 2
                        ELSE 3
                    END,
                    name COLLATE NOCASE
                """,
                (normalized_id,),
            ).fetchall()
        ]
        operators = [
            dict(row)
            for row in conn.execute(
                """
                SELECT o.*, a.is_primary, a.status AS assignment_status
                FROM dropzone_operator_assignments a
                JOIN dropzone_operators o ON o.operator_id = a.operator_id
                WHERE a.dropzone_id = ?
                ORDER BY a.is_primary DESC, o.name COLLATE NOCASE
                """,
                (normalized_id,),
            ).fetchall()
        ]
        source_rows = conn.execute(
            """
            SELECT DISTINCT s.*
            FROM dropzone_sources s
            WHERE s.dropzone_id = ?
               OR EXISTS (
                   SELECT 1 FROM dropzone_zones z
                   WHERE z.dropzone_id = ? AND z.zone_id = s.zone_id
               )
               OR EXISTS (
                   SELECT 1 FROM dropzone_operator_assignments a
                   WHERE a.dropzone_id = ? AND a.operator_id = s.operator_id
               )
            ORDER BY
                CASE s.trust_level
                    WHEN 'primary' THEN 0
                    WHEN 'supporting' THEN 1
                    ELSE 2
                END,
                s.source_kind,
                s.source_name COLLATE NOCASE
            """,
            (normalized_id, normalized_id, normalized_id),
        ).fetchall()
        assignment_summary_row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN dropzone_assignment_source = 'catalog_auto' THEN 1 ELSE 0 END) AS automatic,
                   SUM(CASE WHEN dropzone_assignment_source = 'catalog_manual' THEN 1 ELSE 0 END) AS manual,
                   AVG(dropzone_assignment_confidence) AS average_confidence,
                   MIN(dropzone_assignment_confidence) AS minimum_confidence,
                   MAX(dropzone_assignment_confidence) AS maximum_confidence,
                   AVG(dropzone_distance_m) AS average_distance_m,
                   MAX(t0_utc) AS latest_jump_at
            FROM jumps
            WHERE dropzone_id = ?
            """,
            (normalized_id,),
        ).fetchone()
        assignment_sources = [
            dict(row)
            for row in conn.execute(
                """
                SELECT COALESCE(dropzone_assignment_source, 'unknown') AS assignment_source,
                       COUNT(*) AS assignment_count
                FROM jumps
                WHERE dropzone_id = ?
                GROUP BY COALESCE(dropzone_assignment_source, 'unknown')
                ORDER BY assignment_count DESC, assignment_source
                """,
                (normalized_id,),
            ).fetchall()
        ]
        match_statuses = [
            dict(row)
            for row in conn.execute(
                """
                SELECT match_status, COUNT(*) AS attempt_count,
                       AVG(confidence) AS average_confidence,
                       AVG(nearest_distance_m) AS average_distance_m
                FROM dropzone_match_attempts
                WHERE nearest_dropzone_id = ?
                GROUP BY match_status
                ORDER BY attempt_count DESC, match_status
                """,
                (normalized_id,),
            ).fetchall()
        ]
        observation_groups = [
            dict(row)
            for row in conn.execute(
                """
                SELECT source_kind, quality_status, COUNT(*) AS observation_count,
                       AVG(ground_elevation_m) AS average_ground_elevation_m,
                       AVG(altitude_mad_m) AS average_altitude_mad_m,
                       SUM(COALESCE(sample_count, 0)) AS sample_count
                FROM dropzone_observations
                WHERE dropzone_id = ?
                GROUP BY source_kind, quality_status
                ORDER BY source_kind, quality_status
                """,
                (normalized_id,),
            ).fetchall()
        ]
        metadata_row = conn.execute(
            """
            SELECT catalog_version, imported_at
            FROM dropzone_catalog_metadata
            WHERE catalog_key = 'default'
            LIMIT 1
            """
        ).fetchone()

    sources = [_source_payload(row) for row in source_rows]
    operator_payloads = []
    for operator in operators:
        operator["website_url"] = _safe_external_url(operator.get("website_url"))
        operator_payloads.append(operator)

    assignment_summary = dict(assignment_summary_row) if assignment_summary_row is not None else {}
    for key in ("total", "automatic", "manual"):
        assignment_summary[key] = int(assignment_summary.get(key) or 0)

    return {
        "dropzone": dict(dropzone_row),
        "zones": zones,
        "operators": operator_payloads,
        "sources": sources,
        "assignment_summary": assignment_summary,
        "assignment_sources": assignment_sources,
        "match_statuses": match_statuses,
        "observation_groups": observation_groups,
        "metadata": None if metadata_row is None else dict(metadata_row),
    }


def _source_payload(row: Any) -> dict[str, Any]:
    payload = dict(row)
    payload["source_url"] = _safe_external_url(payload.get("source_url"))
    payload["supports_fields"] = _json_list(payload.pop("supports_fields_json", "[]"))
    payload["details"] = _json_dict(payload.pop("details_json", "{}"))
    return payload


def _safe_external_url(raw: Any) -> str | None:
    value = str(raw or "").strip()
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return value


def _json_list(raw: Any) -> list[Any]:
    try:
        parsed = json.loads(str(raw or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _json_dict(raw: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(raw or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
