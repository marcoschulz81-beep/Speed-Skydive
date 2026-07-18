from __future__ import annotations

import hashlib
import json
import math
from statistics import median
from typing import Any

from app.analysis.pipeline import AnalysisError, analyze_flysight_csv
from app.database import get_connection

MATCH_ALGORITHM_VERSION = "stable-ground-match-v1"
MIN_GROUND_SEQUENCE_START_S = 45.0
MIN_GROUND_SEQUENCE_DURATION_S = 5.0
MAX_GROUND_SAMPLE_GAP_S = 0.35
MAX_TOTAL_SPEED_KMH = 10.0
MAX_VERTICAL_SPEED_MPS = 1.0
MAX_ALTITUDE_MAD_M = 5.0
MAX_COORDINATE_P95_RADIUS_M = 100.0
MAX_AUTO_MATCH_DISTANCE_M = 750.0
MIN_AUTO_CONFIDENCE = 0.75
AMBIGUITY_DISTANCE_MARGIN_M = 500.0
AMBIGUITY_DISTANCE_RATIO = 2.0


def list_dropzone_match_contexts() -> list[dict[str, Any]]:
    with get_connection() as conn:
        metadata = conn.execute(
            "SELECT catalog_version FROM dropzone_catalog_metadata WHERE catalog_key = 'default'"
        ).fetchone()
        catalog_version = None if metadata is None else str(metadata["catalog_version"])
        rows = conn.execute(
            """
            SELECT
                z.zone_id, z.dropzone_id, z.name AS zone_name, z.zone_kind,
                z.latitude, z.longitude, z.match_radius_m, z.status AS zone_status,
                d.name AS dropzone_name, d.country_code, d.ground_elevation_m,
                d.ground_elevation_uncertainty_m, d.ground_elevation_source,
                d.status AS dropzone_status, d.catalog_revision
            FROM dropzone_zones z
            JOIN dropzones d ON d.dropzone_id = z.dropzone_id
            WHERE d.status <> 'inactive' AND z.status <> 'inactive'
            ORDER BY d.name COLLATE NOCASE, z.zone_kind, z.name COLLATE NOCASE
            """
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["catalog_version"] = catalog_version
        result.append(item)
    return result


def list_dropzone_choices() -> list[dict[str, Any]]:
    contexts = list_dropzone_match_contexts()
    choices: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in contexts:
        dropzone_id = str(item["dropzone_id"])
        if dropzone_id in seen:
            continue
        seen.add(dropzone_id)
        choices.append(
            {
                "dropzone_id": dropzone_id,
                "name": str(item["dropzone_name"]),
                "country_code": str(item["country_code"]),
                "status": str(item["dropzone_status"]),
                "ground_elevation_m": item.get("ground_elevation_m"),
            }
        )
    return choices


def extract_stable_ground_observation(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for sample in samples:
        if not _is_eligible_ground_sample(sample):
            continue
        eligible.append(sample)

    segment: list[dict[str, Any]] = []
    for sample in eligible:
        if segment and float(sample["t_rel_s"]) - float(segment[-1]["t_rel_s"]) > MAX_GROUND_SAMPLE_GAP_S:
            observation = _summarize_ground_segment(segment)
            if observation is not None:
                return observation
            segment = []
        segment.append(sample)
    return _summarize_ground_segment(segment)


def match_ground_observation(
    observation: dict[str, Any] | None,
    zones: list[dict[str, Any]],
    *,
    manual_dropzone_id: str | None = None,
) -> dict[str, Any]:
    catalog_version = next((item.get("catalog_version") for item in zones if item.get("catalog_version")), None)
    base: dict[str, Any] = {
        "algorithm_version": MATCH_ALGORITHM_VERSION,
        "catalog_version": catalog_version,
        "status": "insufficient" if observation is None else "unmatched",
        "assignment_source": None,
        "dropzone_id": None,
        "zone_id": None,
        "dropzone_name": None,
        "zone_name": None,
        "dropzone_status": None,
        "catalog_revision": None,
        "ground_elevation_m": None,
        "ground_elevation_source": None,
        "distance_m": None,
        "second_distance_m": None,
        "confidence": None,
        "observation": observation,
        "details": {},
    }
    if not zones:
        base["status"] = "unavailable"
        base["details"] = {"reason": "DROPZONE_CATALOG_EMPTY"}
        return base

    if manual_dropzone_id:
        selected = next((item for item in zones if item["dropzone_id"] == manual_dropzone_id), None)
        if selected is None:
            raise AnalysisError("Die manuell gewählte Dropzone ist nicht mehr im aktiven Katalog vorhanden.")
        distance_m = None
        if observation is not None:
            distance_m = _haversine_m(
                float(observation["latitude"]),
                float(observation["longitude"]),
                float(selected["latitude"]),
                float(selected["longitude"]),
            )
        return _matched_payload(
            base,
            selected,
            status="manual",
            assignment_source="catalog_manual",
            confidence=1.0,
            distance_m=distance_m,
            second_distance_m=None,
            details={"reason": "MANUAL_DROPZONE_SELECTION"},
        )

    if observation is None:
        base["details"] = {"reason": "NO_STABLE_GROUND_SEQUENCE"}
        return base

    ranked = []
    for zone in zones:
        distance_m = _haversine_m(
            float(observation["latitude"]),
            float(observation["longitude"]),
            float(zone["latitude"]),
            float(zone["longitude"]),
        )
        ranked.append((distance_m, zone))
    ranked.sort(key=lambda item: item[0])
    nearest_distance_m, nearest = ranked[0]
    base["distance_m"] = round(nearest_distance_m, 1)
    base["dropzone_id"] = nearest["dropzone_id"]
    base["zone_id"] = nearest["zone_id"]
    base["dropzone_name"] = nearest["dropzone_name"]
    base["zone_name"] = nearest["zone_name"]
    base["dropzone_status"] = nearest["dropzone_status"]
    base["catalog_revision"] = nearest["catalog_revision"]

    effective_radius_m = min(float(nearest["match_radius_m"]), MAX_AUTO_MATCH_DISTANCE_M)
    if nearest_distance_m > effective_radius_m:
        base["details"] = {
            "reason": "OUTSIDE_MATCH_RADIUS",
            "effective_radius_m": effective_radius_m,
        }
        return base

    grouped = _nearest_by_dropzone(ranked)
    second_distance_m = grouped[1][0] if len(grouped) > 1 else None
    base["second_distance_m"] = None if second_distance_m is None else round(second_distance_m, 1)
    if second_distance_m is not None:
        difference_m = second_distance_m - nearest_distance_m
        ratio = math.inf if nearest_distance_m <= 0.0 else second_distance_m / nearest_distance_m
        if difference_m < AMBIGUITY_DISTANCE_MARGIN_M or ratio < AMBIGUITY_DISTANCE_RATIO:
            base["status"] = "ambiguous"
            base["details"] = {
                "reason": "MULTIPLE_NEARBY_DROPZONES",
                "distance_margin_m": round(difference_m, 1),
                "distance_ratio": None if not math.isfinite(ratio) else round(ratio, 3),
            }
            return base

    confidence = _match_confidence(
        observation=observation,
        distance_m=nearest_distance_m,
        effective_radius_m=effective_radius_m,
    )
    base["confidence"] = confidence
    if str(nearest["dropzone_status"]) not in {"trusted", "verified"}:
        return _matched_payload(
            base,
            nearest,
            status="candidate",
            assignment_source=None,
            confidence=confidence,
            distance_m=nearest_distance_m,
            second_distance_m=second_distance_m,
            details={"reason": "DROPZONE_NOT_TRUSTED"},
        )
    if nearest.get("ground_elevation_m") is None:
        base["status"] = "candidate"
        base["details"] = {"reason": "DROPZONE_GROUND_ELEVATION_MISSING"}
        return base
    if confidence < MIN_AUTO_CONFIDENCE:
        return _matched_payload(
            base,
            nearest,
            status="low_confidence",
            assignment_source=None,
            confidence=confidence,
            distance_m=nearest_distance_m,
            second_distance_m=second_distance_m,
            details={"reason": "CONFIDENCE_BELOW_THRESHOLD", "minimum_confidence": MIN_AUTO_CONFIDENCE},
        )
    return _matched_payload(
        base,
        nearest,
        status="accepted",
        assignment_source="catalog_auto",
        confidence=confidence,
        distance_m=nearest_distance_m,
        second_distance_m=second_distance_m,
        details={"reason": "TRUSTED_UNAMBIGUOUS_MATCH", "effective_radius_m": effective_radius_m},
    )


def analyze_flysight_with_dropzone(
    *,
    content: bytes,
    file_name: str,
    jumper_name: str,
    ground_elevation_m: float | None,
    breakoff_altitude_agl_m: float | None,
    manual_t0_utc: str | None = None,
    manual_dropzone_id: str | None = None,
    zones: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    match_contexts = list_dropzone_match_contexts() if zones is None else zones
    initial = analyze_flysight_csv(
        content=content,
        file_name=file_name,
        jumper_name=jumper_name,
        ground_elevation_m=ground_elevation_m,
        breakoff_altitude_agl_m=breakoff_altitude_agl_m,
        manual_t0_utc=manual_t0_utc,
    )
    observation = extract_stable_ground_observation(initial["sample_records"])
    match = match_ground_observation(
        observation,
        match_contexts,
        manual_dropzone_id=manual_dropzone_id,
    )

    result = initial
    matched_ground = match.get("ground_elevation_m")
    if ground_elevation_m is None and match["status"] in {"accepted", "manual"} and matched_ground is not None:
        source = "dropzone_manual" if match["status"] == "manual" else "dropzone_catalog"
        result = analyze_flysight_csv(
            content=content,
            file_name=file_name,
            jumper_name=jumper_name,
            ground_elevation_m=float(matched_ground),
            breakoff_altitude_agl_m=breakoff_altitude_agl_m,
            manual_t0_utc=manual_t0_utc,
            ground_elevation_source_override=source,
        )
        result_observation = extract_stable_ground_observation(result["sample_records"])
        if result_observation is not None:
            match["observation"] = result_observation

    _attach_dropzone_match(result, match)
    return result


def audit_dropzone_matches() -> dict[str, Any]:
    with get_connection() as conn:
        status_counts = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                "SELECT match_status, COUNT(*) FROM dropzone_match_attempts GROUP BY match_status ORDER BY match_status"
            )
        }
        pending = [
            dict(row)
            for row in conn.execute(
                """
                SELECT attempt_id, jump_id, match_status, observed_at, latitude, longitude,
                       ground_elevation_m, nearest_dropzone_id, nearest_distance_m,
                       second_distance_m, confidence, catalog_version, created_at
                FROM dropzone_match_attempts
                WHERE match_status IN ('candidate', 'ambiguous', 'unmatched', 'low_confidence')
                ORDER BY created_at DESC, attempt_id
                """
            ).fetchall()
        ]
    return {
        "attempts": sum(status_counts.values()),
        "status_counts": status_counts,
        "pending_count": len(pending),
        "pending": pending,
    }


def _attach_dropzone_match(result: dict[str, Any], match: dict[str, Any]) -> None:
    jump = result["jump_record"]
    assigned = match["status"] in {"accepted", "manual"}
    jump["dropzone_id"] = match.get("dropzone_id") if assigned else None
    jump["dropzone_zone_id"] = match.get("zone_id") if assigned else None
    jump["dropzone_revision"] = match.get("catalog_revision") if assigned else None
    jump["dropzone_assignment_source"] = match.get("assignment_source") if assigned else None
    jump["dropzone_assignment_confidence"] = match.get("confidence") if assigned else None
    jump["dropzone_distance_m"] = match.get("distance_m") if assigned else None
    signature_payload = {
        "base": jump["analysis_signature"],
        "algorithm": match.get("algorithm_version"),
        "catalog_version": match.get("catalog_version"),
        "status": match.get("status"),
        "dropzone_id": jump["dropzone_id"],
        "zone_id": jump["dropzone_zone_id"],
        "revision": jump["dropzone_revision"],
        "assignment_source": jump["dropzone_assignment_source"],
    }
    jump["analysis_signature"] = hashlib.sha256(
        json.dumps(signature_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result["dropzone_match"] = match
    result["report"]["dropzone_match"] = match
    result["report"]["notes"]["dropzone_match_status"] = match["status"]
    result["report"]["notes"]["dropzone_match_algorithm"] = match["algorithm_version"]
    stored_notes = json.loads(str(result["metrics_record"].get("notes") or "{}"))
    if isinstance(stored_notes, dict):
        stored_notes["dropzone_match_status"] = match["status"]
        stored_notes["dropzone_match_algorithm"] = match["algorithm_version"]
        result["metrics_record"]["notes"] = json.dumps(stored_notes)


def _is_eligible_ground_sample(sample: dict[str, Any]) -> bool:
    try:
        t_rel_s = float(sample["t_rel_s"])
        latitude = float(sample["lat"])
        longitude = float(sample["lon"])
        altitude = float(sample["hMSL_m"])
        total_speed = float(sample["vTotal_kmh"])
        vertical_speed = float(sample["velD_mps"])
        gps_fix = float(sample["gpsFix"])
        satellites = float(sample["numSV"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (t_rel_s, latitude, longitude, altitude, total_speed, vertical_speed)):
        return False
    if t_rel_s < MIN_GROUND_SEQUENCE_START_S:
        return False
    if gps_fix < 3 or satellites < 6:
        return False
    if total_speed > MAX_TOTAL_SPEED_KMH or abs(vertical_speed) > MAX_VERTICAL_SPEED_MPS:
        return False
    for key, limit in (("sAcc", 3.0), ("vAcc", 10.0)):
        value = sample.get(key)
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return False
            if math.isfinite(numeric) and numeric > limit:
                return False
    return True


def _summarize_ground_segment(segment: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(segment) < 10:
        return None
    duration_s = float(segment[-1]["t_rel_s"]) - float(segment[0]["t_rel_s"])
    if duration_s < MIN_GROUND_SEQUENCE_DURATION_S:
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
    if altitude_mad_m > MAX_ALTITUDE_MAD_M or coordinate_p95_radius_m > MAX_COORDINATE_P95_RADIUS_M:
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


def _matched_payload(
    base: dict[str, Any],
    zone: dict[str, Any],
    *,
    status: str,
    assignment_source: str | None,
    confidence: float,
    distance_m: float | None,
    second_distance_m: float | None,
    details: dict[str, Any],
) -> dict[str, Any]:
    base.update(
        {
            "status": status,
            "assignment_source": assignment_source,
            "dropzone_id": zone["dropzone_id"],
            "zone_id": zone["zone_id"],
            "dropzone_name": zone["dropzone_name"],
            "zone_name": zone["zone_name"],
            "dropzone_status": zone["dropzone_status"],
            "catalog_revision": zone["catalog_revision"],
            "ground_elevation_m": zone.get("ground_elevation_m"),
            "ground_elevation_source": zone.get("ground_elevation_source"),
            "distance_m": None if distance_m is None else round(distance_m, 1),
            "second_distance_m": None if second_distance_m is None else round(second_distance_m, 1),
            "confidence": round(float(confidence), 3),
            "details": details,
        }
    )
    return base


def _nearest_by_dropzone(ranked: list[tuple[float, dict[str, Any]]]) -> list[tuple[float, dict[str, Any]]]:
    result: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for distance_m, zone in ranked:
        dropzone_id = str(zone["dropzone_id"])
        if dropzone_id in seen:
            continue
        seen.add(dropzone_id)
        result.append((distance_m, zone))
    return result


def _match_confidence(*, observation: dict[str, Any], distance_m: float, effective_radius_m: float) -> float:
    distance_score = max(0.0, 1.0 - distance_m / max(effective_radius_m, 1.0))
    coordinate_score = max(
        0.0,
        1.0 - float(observation["coordinate_p95_radius_m"]) / MAX_COORDINATE_P95_RADIUS_M,
    )
    altitude_score = max(0.0, 1.0 - float(observation["altitude_mad_m"]) / MAX_ALTITUDE_MAD_M)
    duration_score = min(1.0, float(observation["duration_s"]) / 10.0)
    value = 0.35 + 0.35 * distance_score + 0.15 * coordinate_score + 0.10 * altitude_score + 0.05 * duration_score
    return round(min(1.0, max(0.0, value)), 3)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = math.radians(lon2 - lon1)
    value = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    return 2.0 * radius_m * math.asin(math.sqrt(value))
