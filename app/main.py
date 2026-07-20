from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from html import escape as html_escape
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import quote_plus

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.analysis.comparison import build_jump_comparison
from app.analysis.evaluation import effective_eval_window_end_s, is_reference_eligible
from app.analysis.feedback import build_feedback_coaching_context, build_feedback_training_profile
from app.analysis.lateral import analyze_lateral_dynamics
from app.analysis.pipeline import AnalysisError
from app.analysis.potential import build_speed_potential_preview
from app.analysis.review import build_jump_review
from app.config import (
    AI_COACHING_ENABLED,
    AI_COACHING_INCLUDE_IDENTIFIERS,
    AI_COACHING_MAX_REQUESTS_PER_DAY,
    AI_COACHING_MODEL,
    AI_COACHING_TIMEOUT_S,
    ANALYSIS_VERSION,
    APP_VERSION,
    BASE_DIR,
    COACH_VIEW_ENABLED,
    MAX_UPLOAD_BYTES,
    RAW_UPLOAD_DIR,
    TECHNICAL_PHASE_SPECS,
)
from app.database import init_db
from app.services.ai_coach import AI_COACHING_SCHEMA_VERSION, generate_ai_coaching_texts
from app.services.dropzone_directory import get_dropzone_detail, list_dropzones
from app.services.dropzone_matching import analyze_flysight_with_dropzone, list_dropzone_choices
from app.services.storage import (
    VALID_JUMP_CONTEXTS,
    delete_jump,
    get_best_jump_for_jumper,
    get_jump_report,
    get_jump_source_metadata,
    get_jump_summary,
    list_compare_candidates,
    list_jumpers,
    list_jumps_for_jumper,
    list_recent_jumps,
    list_top_global_references,
    normalize_jump_context,
    replace_analysis_result,
    save_analysis_result,
    update_jump_context,
    upsert_coaching_snapshot,
    upsert_jump_feedback,
)
from app.text_utils import normalize_german_text


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    init_db()
    RAW_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Speed-Skydive Analyzer", version=APP_VERSION, lifespan=_lifespan)
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
templates.env.globals["coach_view_enabled"] = COACH_VIEW_ENABLED
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")

_MARCO_PROFILE_CACHE: dict[int, tuple[str, dict[str, Any] | None]] = {}
_JUMPER_SUMMARY_CACHE: dict[str, tuple[str, dict[str, Any]]] = {}
_PLOTLY_JS_CACHE: str | None = None
_SEMANTIC_DEDUPE_STOPWORDS: set[str] = {
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "einen",
    "einem",
    "und",
    "oder",
    "mit",
    "im",
    "in",
    "auf",
    "bei",
    "bis",
    "von",
    "zu",
    "für",
    "nach",
    "vor",
    "ist",
    "war",
    "wird",
    "du",
    "dein",
    "deine",
    "deinen",
    "deinem",
    "deiner",
    "deines",
    "dass",
    "damit",
    "hier",
    "dort",
    "diesem",
    "diesen",
    "dieser",
}
_SEMANTIC_DEDUPE_STOPWORDS = {
    normalize_german_text(item) for item in _SEMANTIC_DEDUPE_STOPWORDS
}

_VIEW_MODE_SIMPLE = "simple"
_VIEW_MODE_EXPERT = "expert"
_INDEX_RECENT_JUMPS_LIMIT = 10
_TRUTHY_QUERY_VALUES = {"1", "true", "yes", "open"}
_JUMP_CONTEXT_DEFAULT = "training"
_TIMING_REFERENCE_MIN_USABLE_JUMPS = 5
_TIMING_REFERENCE_MIN_BEST_KMH = 430.0
_TIMING_REFERENCE_MIN_TOP3_AVG_KMH = 415.0
_TIMING_REFERENCE_MIN_BASIS_JUMPS = 3
_TIMING_REFERENCE_MAX_ANCHOR_SPREAD_S = 2.0
_TIMING_REFERENCE_MATURITY_LABELS = {
    "off": "nicht bewertet",
    "soft": "noch nicht belastbar",
    "active": "belastbar",
}
_JUMP_CONTEXT_LABELS = {
    "unknown": "Unbekannt",
    "training": "Training",
    "competition": "Wettkampf",
}
_JUMP_CONTEXT_OPTIONS = [
    {"value": "training", "label": "Training"},
    {"value": "competition", "label": "Wettkampf"},
    {"value": "unknown", "label": "Unbekannt"},
]
_PERFORMANCE_BAND_LABELS = {
    "basis": "Basis",
    "aufbau": "Aufbau",
    "schnell": "Schnell",
    "elite": "Elite",
}
_PROFILE_CONFIDENCE_LABELS = {
    "low": "niedrig",
    "medium": "mittel",
    "good": "gut",
    "stable": "stabil",
}
_DROPZONE_STATUS_LABELS = {
    "candidate": "Kandidat",
    "trusted": "Vertrauenswürdig",
    "verified": "Verifiziert",
    "inactive": "Inaktiv",
}
_ZONE_KIND_LABELS = {
    "primary": "Primärzone",
    "landing": "Landezone",
    "alternate": "Ausweichzone",
    "historical": "Historische Zone",
}
_ZONE_STATUS_LABELS = {"active": "Aktiv", "candidate": "Kandidat", "inactive": "Inaktiv"}
_OPERATOR_STATUS_LABELS = {"listed": "Gelistet", "confirmed": "Bestätigt", "inactive": "Inaktiv"}
_SOURCE_KIND_LABELS = {
    "directory": "Verzeichnis",
    "official": "Offizielle Quelle",
    "airport_registry": "Flugplatzregister",
    "terrain": "Geländemodell",
    "historical_gps": "Historische GNSS-Daten",
    "manual": "Manuelle Prüfung",
}
_TRUST_LEVEL_LABELS = {
    "discovery": "Fundstelle",
    "supporting": "Stützend",
    "primary": "Primärbeleg",
}
_ASSIGNMENT_SOURCE_LABELS = {
    "catalog_auto": "Automatisch",
    "catalog_manual": "Manuell bestätigt",
    "unknown": "Unbekannt",
}
_MATCH_STATUS_LABELS = {
    "accepted": "Akzeptiert",
    "manual": "Manuell",
    "candidate": "Kandidat",
    "ambiguous": "Mehrdeutig",
    "unmatched": "Ohne Treffer",
    "low_confidence": "Niedrige Konfidenz",
    "insufficient": "Daten unzureichend",
    "unavailable": "Nicht verfügbar",
}
_OBSERVATION_QUALITY_LABELS = {
    "candidate": "Kandidat",
    "accepted": "Akzeptiert",
    "rejected": "Verworfen",
}

_SIMPLE_GLOSSARY_ITEMS: list[tuple[str, str]] = [
    ("Druck halten", "Körper ruhig und fest im Luftstrom lassen, ohne hektische Bewegungen."),
    ("Druck", "Wie stark du den Körper in den Luftstrom stellst, um Geschwindigkeit aufzubauen."),
    ("kleine Korrekturen", "Kurze, minimale Bewegungen statt großer später Gegenbewegungen."),
    ("Korrekturen", "Bewusste kleine Bewegungen, um Richtung und Lage nachzujustieren."),
    ("Linie", "Deine Flugrichtung und Körperlage, die möglichst ruhig bleiben soll."),
    ("Aufbauphase", "Abschnitt nach dem Start, in dem du kontrolliert Geschwindigkeit aufbaust."),
    ("schnelle Phase", "Der schnellste Teil des Sprungs kurz vor dem Abbremsen."),
    ("Beschleunigung", "Wie schnell deine Geschwindigkeit zunimmt."),
    ("flacher gehen", "Winkel etwas weniger steil machen, um den Flug zu beruhigen."),
    ("Richtungswechsel", "Wenn die Flugrichtung oft hin und her springt."),
    ("waagerechte Geschwindigkeit", "Der horizontale Anteil deiner GPS-Geschwindigkeit ueber Grund; nicht die Koerperausrichtung."),
    ("horizontale Reserve", "Wie viel waagerechte Geschwindigkeit noch bleibt, wenn der Sprung sehr steil und schnell wird."),
    ("Vorwärtsrichtung", "Der horizontale Anteil deiner GPS-Bewegung ueber Grund; nicht die Koerperausrichtung."),
]
_SIMPLE_GLOSSARY_LOOKUP: dict[str, str] = {
    normalize_german_text(key): tip for key, tip in _SIMPLE_GLOSSARY_ITEMS
}
_SIMPLE_GLOSSARY_PATTERN: re.Pattern[str] | None = (
    re.compile(
        "|".join(re.escape(item[0]) for item in sorted(_SIMPLE_GLOSSARY_ITEMS, key=lambda row: len(row[0]), reverse=True)),
        flags=re.IGNORECASE,
    )
    if _SIMPLE_GLOSSARY_ITEMS
    else None
)


def _normalize_view_mode(raw: str | None) -> str:
    mode = str(raw or "").strip().lower()
    if mode == _VIEW_MODE_SIMPLE:
        return _VIEW_MODE_SIMPLE
    return _VIEW_MODE_EXPERT


def _normalize_jump_context(raw: str | None, *, default: str = "unknown") -> str:
    return normalize_jump_context(raw, default=default)


def _jump_context_label(raw: str | None) -> str:
    return _JUMP_CONTEXT_LABELS.get(_normalize_jump_context(raw), _JUMP_CONTEXT_LABELS["unknown"])


templates.env.globals["jump_context_label"] = _jump_context_label
templates.env.globals["jump_context_options"] = _JUMP_CONTEXT_OPTIONS


def _is_enabled_query(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in _TRUTHY_QUERY_VALUES


def _clear_derived_caches() -> None:
    _JUMPER_SUMMARY_CACHE.clear()
    _MARCO_PROFILE_CACHE.clear()


def _report_for_client(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "chart_data"}


def _safe_json_dumps(value: Any) -> str:
    """Serialize JSON for an inline script without allowing an HTML end-tag escape."""
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _chart_data_for_client(report: dict[str, Any], *, max_points: int = 3200) -> dict[str, Any]:
    chart = report.get("chart_data", {}) if isinstance(report.get("chart_data"), dict) else {}
    time_s = chart.get("time_s", []) if isinstance(chart.get("time_s"), list) else []
    if not time_s:
        return chart

    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    curve_start = _to_float(notes.get("curve_window_start_s"))
    curve_end = _to_float(notes.get("curve_window_end_s"))
    max_time = _max_time_s(time_s)
    start_s = min(0.0, float(curve_start if curve_start is not None else 0.0))
    end_s = float(curve_end if curve_end is not None else (max_time if max_time is not None else time_s[-1]))

    indices: list[int] = []
    for i, raw_t in enumerate(time_s):
        t = _to_float(raw_t)
        if t is None:
            continue
        if start_s <= float(t) <= end_s:
            indices.append(i)

    filtered = _chart_data_at_indices(chart, indices) if indices else chart
    return _downsample_chart_data(filtered, max_points=max_points)


def _build_t0_diagnostics(report: dict[str, Any]) -> dict[str, Any]:
    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    jump = report.get("jump", {}) if isinstance(report.get("jump"), dict) else {}
    chart = report.get("chart_data", {}) if isinstance(report.get("chart_data"), dict) else {}
    flags = {str(flag) for flag in (report.get("quality_flags") or [])}

    first_vvert = _first_chart_number(chart, "vVert_kmh")
    first_vhor = _first_chart_number(chart, "vHor_kmh")
    first_angle = _first_chart_number(chart, "angle_deg")
    pw_start = _to_float(metrics.get("performance_window_start_s"))
    pw_end = _to_float(metrics.get("performance_window_end_s"))
    validation_quality = _to_float(metrics.get("validation_window_quality"))

    warnings: list[str] = []
    if bool(notes.get("t0_review_required")):
        warnings.append("Automatische t0-Erkennung ist unsicher; Kurvenvergleich und Phasenzeiten pruefen.")
    if first_vvert is not None and first_vvert >= 90.0:
        warnings.append("Der erste Punkt nach t0 ist bereits sehr schnell; t0 koennte zu spaet liegen.")
    elif first_vvert is not None and first_vvert >= 65.0:
        warnings.append("Der erste Punkt nach t0 ist schon dynamisch; bei auffaelligem Vergleich t0 pruefen.")
    if first_angle is not None and first_angle >= 65.0:
        warnings.append("Der erste Winkel nach t0 ist bereits steil; Exit-/Startanker visuell gegenpruefen.")
    if "NO_GROUND_LEVEL" in flags:
        warnings.append("Bodenhoehe wurde geschaetzt; AGL- und Performance-Window-Grenzen sind weniger belastbar.")

    return {
        "file_name": str(jump.get("file_name") or ""),
        "t0_utc": str(jump.get("t0_utc") or ""),
        "t0_confidence": _to_float(notes.get("t0_confidence")),
        "t0_uncertainty_s": _to_float(notes.get("t0_uncertainty_s")),
        "t0_review_required": bool(notes.get("t0_review_required")),
        "t0_manual_override": bool(notes.get("t0_manual_override")),
        "t0_reason": str(notes.get("t0_reason") or ""),
        "auto_t0_utc": str(notes.get("auto_t0_utc") or ""),
        "auto_t0_confidence": _to_float(notes.get("auto_t0_confidence")),
        "first_vvert_kmh": first_vvert,
        "first_vhor_kmh": first_vhor,
        "first_angle_deg": first_angle,
        "performance_window_start_s": pw_start,
        "performance_window_end_s": pw_end,
        "validation_window_quality": validation_quality,
        "ground_level_estimated": bool(notes.get("ground_level_estimated") or ("NO_GROUND_LEVEL" in flags)),
        "agl_note": str(notes.get("agl_note") or ""),
        "vhor_basis": "GPS-Geschwindigkeit ueber Grund; aktuell nicht windkorrigiert.",
        "warnings": _unique_texts(warnings),
    }


def _first_chart_number(chart: dict[str, Any], key: str) -> float | None:
    values = chart.get(key, []) if isinstance(chart.get(key), list) else []
    for raw in values:
        value = _to_float(raw)
        if value is not None:
            return float(value)
    return None


def _build_compare_alignment(
    *,
    reference_report: dict[str, Any],
    comparison_report: dict[str, Any],
) -> dict[str, Any]:
    reference = _build_t0_diagnostics(reference_report)
    comparison = _build_t0_diagnostics(comparison_report)
    warnings: list[str] = []
    for label, diag in [("Referenz", reference), ("Vergleich", comparison)]:
        for warning in diag.get("warnings", []):
            warnings.append(f"{label}: {warning}")
    return {
        "reference": reference,
        "comparison": comparison,
        "warnings": _unique_texts(warnings),
    }


def _chart_data_at_indices(chart: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    time_s = chart.get("time_s", []) if isinstance(chart.get("time_s"), list) else []
    source_len = len(time_s)
    out: dict[str, Any] = {}
    for key, value in chart.items():
        if isinstance(value, list) and len(value) == source_len:
            out[key] = [value[i] for i in indices]
        else:
            out[key] = value
    return out


def _downsample_chart_data(chart: dict[str, Any], *, max_points: int) -> dict[str, Any]:
    time_s = chart.get("time_s", []) if isinstance(chart.get("time_s"), list) else []
    n = len(time_s)
    if n <= max(2, int(max_points)):
        return chart

    value_keys = [
        key
        for key in [
            "vVert_kmh",
            "vHor_kmh",
            "angle_deg",
            "hAGL_m",
            "accVert_mps2",
            "forward_m",
            "backtrack_m",
        ]
        if isinstance(chart.get(key), list) and len(chart.get(key)) == n
    ]
    max_points = max(2, int(max_points))
    max_indices_per_bucket = max(2, 2 + (2 * len(value_keys)))
    bucket_count = max(1, max_points // max_indices_per_bucket)
    bucket_size = n / float(bucket_count)
    selected: set[int] = {0, n - 1}

    for bucket in range(bucket_count):
        start = int(bucket * bucket_size)
        end = int((bucket + 1) * bucket_size)
        end = min(n, max(start + 1, end))
        selected.add(start)
        selected.add(end - 1)
        for key in value_keys:
            values = chart.get(key, [])
            numeric_items: list[tuple[float, int]] = []
            for idx in range(start, end):
                value = _to_float(values[idx])
                if value is not None:
                    numeric_items.append((float(value), idx))
            if not numeric_items:
                continue
            selected.add(min(numeric_items, key=lambda item: item[0])[1])
            selected.add(max(numeric_items, key=lambda item: item[0])[1])

    selected_indices = sorted(selected)
    if len(selected_indices) > max_points:
        step = (len(selected_indices) - 1) / float(max_points - 1)
        selected_indices = sorted({selected_indices[int(round(i * step))] for i in range(max_points)})
        selected_indices[0] = 0
        selected_indices[-1] = n - 1

    out = _chart_data_at_indices(chart, selected_indices)
    out["display_downsampled"] = True
    out["source_points"] = n
    out["display_points"] = len(selected_indices)
    return out


def _jumper_overview_simple_status(*, stable_count: int, unstable_count: int) -> str:
    total = max(0, int(stable_count)) + max(0, int(unstable_count))
    if total < 2:
        return "Noch offen"
    stable_ratio = float(max(0, int(stable_count))) / float(total)
    if stable_ratio >= 0.7:
        return "Stabil"
    if stable_ratio >= 0.45:
        return "Wechselhaft"
    return "Unruhig"


@app.get("/plotly.min.js", include_in_schema=False)
def plotly_bundle() -> Response:
    global _PLOTLY_JS_CACHE
    if _PLOTLY_JS_CACHE is None:
        from plotly.offline import get_plotlyjs

        _PLOTLY_JS_CACHE = get_plotlyjs()

    return Response(
        content=_PLOTLY_JS_CACHE,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/")
def index(
    request: Request,
    message: str | None = None,
    error: str | None = None,
    view: str | None = None,
    show_jumpers: str | None = None,
):
    jumpers = list_jumpers()
    view_mode = _normalize_view_mode(view)
    show_jumpers_panel = _is_enabled_query(show_jumpers)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "message": message,
            "error": error,
            "view_mode": view_mode,
            "recent_jumps": list_recent_jumps(limit=_INDEX_RECENT_JUMPS_LIMIT),
            "jumpers": jumpers,
            "show_jumpers_panel": show_jumpers_panel,
            "jumper_overview": _build_jumpers_overview(jumpers) if show_jumpers_panel else [],
            "needs_plotly": False,
        },
    )


@app.get("/dropzones")
def dropzone_directory(
    request: Request,
    q: str | None = None,
    country: str | None = None,
    status: str | None = None,
    view: str | None = None,
):
    view_mode = _normalize_view_mode(view)
    directory = list_dropzones(query=q, country_code=country, status=status)
    return templates.TemplateResponse(
        request,
        "dropzones.html",
        {
            "directory": directory,
            "view_mode": view_mode,
            "dropzone_status_labels": _DROPZONE_STATUS_LABELS,
            "needs_plotly": False,
        },
    )


@app.get("/dropzones/{dropzone_id}")
def dropzone_detail(request: Request, dropzone_id: str, view: str | None = None):
    view_mode = _normalize_view_mode(view)
    detail = get_dropzone_detail(dropzone_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Dropzone nicht gefunden.")
    return templates.TemplateResponse(
        request,
        "dropzone_detail.html",
        {
            "detail": detail,
            "view_mode": view_mode,
            "dropzone_status_labels": _DROPZONE_STATUS_LABELS,
            "zone_kind_labels": _ZONE_KIND_LABELS,
            "zone_status_labels": _ZONE_STATUS_LABELS,
            "operator_status_labels": _OPERATOR_STATUS_LABELS,
            "source_kind_labels": _SOURCE_KIND_LABELS,
            "trust_level_labels": _TRUST_LEVEL_LABELS,
            "assignment_source_labels": _ASSIGNMENT_SOURCE_LABELS,
            "match_status_labels": _MATCH_STATUS_LABELS,
            "observation_quality_labels": _OBSERVATION_QUALITY_LABELS,
            "needs_plotly": False,
        },
    )


@app.post("/analyze")
async def analyze_upload(
    request: Request,
    jumper_name: str = Form(...),
    csv_file: UploadFile = File(...),
    jump_context: str = Form(_JUMP_CONTEXT_DEFAULT),
    ground_elevation_m: str = Form(""),
    breakoff_altitude_agl_m: str = Form(""),
    jump_feedback: str = Form(""),
    view_mode: str = Form(_VIEW_MODE_EXPERT),
):
    resolved_view_mode = _normalize_view_mode(view_mode)
    jumper = jumper_name.strip()
    if not jumper:
        return _render_index_with_error(request, "Springername ist erforderlich.", view_mode=resolved_view_mode)

    uploaded_file_name = str(csv_file.filename or "").strip()
    if not uploaded_file_name.lower().endswith(".csv"):
        return _render_index_with_error(request, "Bitte eine CSV-Datei hochladen.", view_mode=resolved_view_mode)

    resolved_jump_context = _normalize_jump_context(jump_context, default="")
    if resolved_jump_context not in VALID_JUMP_CONTEXTS:
        return _render_index_with_error(request, "Ungültiger Sprung-Kontext.", view_mode=resolved_view_mode)

    content = await csv_file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        return _render_index_with_error(request, "Die hochgeladene Datei ist leer.", view_mode=resolved_view_mode)
    if len(content) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        return _render_index_with_error(
            request,
            f"Die CSV-Datei ist größer als das erlaubte Limit von {max_mb:.0f} MB.",
            view_mode=resolved_view_mode,
        )

    try:
        ground = _parse_optional_float(ground_elevation_m)
        breakoff = _parse_optional_float(breakoff_altitude_agl_m)
    except ValueError as exc:
        return _render_index_with_error(request, str(exc), view_mode=resolved_view_mode)

    source_hash = hashlib.sha256(content).hexdigest()
    try:
        result = analyze_flysight_with_dropzone(
            content=content,
            file_name=uploaded_file_name,
            jumper_name=jumper,
            ground_elevation_m=ground,
            breakoff_altitude_agl_m=breakoff,
        )
    except AnalysisError as exc:
        return _render_index_with_error(request, str(exc), view_mode=resolved_view_mode)
    except Exception as exc:  # pragma: no cover
        return _render_index_with_error(request, f"Unerwarteter Analysefehler: {exc}", view_mode=resolved_view_mode)

    source_path = _cache_uploaded_file(source_hash=source_hash, original_name=uploaded_file_name, content=content)
    jump_id, is_duplicate = save_analysis_result(
        result,
        jump_context=resolved_jump_context,
        source_file_sha256=source_hash,
        source_file_path=str(source_path),
    )
    if str(jump_feedback or "").strip():
        upsert_jump_feedback(jump_id, jump_feedback)
    if not is_duplicate:
        _clear_derived_caches()
    jump_url = f"/jumps/{jump_id}?view={resolved_view_mode}"
    if is_duplicate:
        return RedirectResponse(url=jump_url, status_code=303)
    return RedirectResponse(url=jump_url, status_code=303)


@app.get("/jumps/{jump_id}")
def jump_detail(
    request: Request,
    jump_id: str,
    message: str | None = None,
    error: str | None = None,
    view: str | None = None,
):
    report = get_jump_report(jump_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Sprung nicht gefunden.")
    view_mode = _normalize_view_mode(view)

    best_compare: dict[str, Any] | None = None
    best_reference: dict[str, Any] | None = None
    best_history_reference: dict[str, Any] | None = None
    best_history_report: dict[str, Any] | None = None
    top_reference_jumps = list_top_global_references(limit=5, exclude_jump_id=jump_id)
    top_reference_compares: list[dict[str, Any]] = []

    best_candidate = get_best_jump_for_jumper(report["jump"]["jumper_name"], exclude_jump_id=jump_id)
    if best_candidate is not None:
        best_report = get_jump_report(best_candidate["jump_id"])
        if best_report is not None:
            best_reference = best_candidate
            best_compare = build_jump_comparison(left_report=report, right_report=best_report)
    best_history_reference, best_history_report = _pick_best_history_reference(report["jump"]["jumper_name"])

    for item in top_reference_jumps:
        ref_report = get_jump_report(item["jump_id"])
        if ref_report is None:
            continue
        top_reference_compares.append(
            build_jump_comparison(left_report=report, right_report=ref_report)
        )

    jumper_history_rows = list_jumps_for_jumper(report["jump"]["jumper_name"])
    previous_jump_row = _find_previous_jump_row(
        jumps=jumper_history_rows,
        current_jump_id=str(report["jump"].get("jump_id") or jump_id),
    )
    previous_jump_report = (
        get_jump_report(str(previous_jump_row["jump_id"]))
        if previous_jump_row is not None and previous_jump_row.get("jump_id")
        else None
    )
    jumper_history_reports: list[dict[str, Any]] = []
    for item in jumper_history_rows:
        item_report = get_jump_report(item["jump_id"])
        if item_report is not None:
            jumper_history_reports.append(item_report)
    jumper_summary = _build_jumper_summary(
        jumper_name=report["jump"]["jumper_name"],
        jumps=jumper_history_rows,
    )
    jumper_stability_reference = (
        jumper_summary.get("stability_reference")
        if isinstance(jumper_summary, dict)
        else None
    )
    tip_effect_profile = (
        jumper_summary.get("tip_effect_profile")
        if isinstance(jumper_summary, dict)
        else None
    )

    speed_potential = build_speed_potential_preview(
        current_report=report,
        historical_reports=jumper_history_reports,
    )
    marco_profile = _get_marco_top15_profile(limit=15)
    previous_review = (
        build_jump_review(
            previous_jump_report,
            jumper_stability_reference=jumper_stability_reference,
            tip_effect_profile=tip_effect_profile,
        )
        if previous_jump_report is not None
        else None
    )
    tip_follow_up = _build_tip_follow_up(
        current_report=report,
        previous_report=previous_jump_report,
        marco_profile=marco_profile,
        previous_coaching_goals=(
            previous_review.get("coaching_goals", [])
            if isinstance(previous_review, dict)
            else []
        ),
    )
    review = build_jump_review(
        report,
        best_compare=best_compare,
        top_reference_compares=top_reference_compares,
        jumper_stability_reference=jumper_stability_reference,
        tip_effect_profile=tip_effect_profile,
    )
    feedback_context = build_feedback_coaching_context(report, review=review)
    scorecard_rows = _build_scorecard_rows(report, marco_profile=marco_profile)
    scorecard_rows = _annotate_scorecard_with_reference(
        rows=scorecard_rows,
        report=report,
        reference_report=best_history_report,
    )
    phase_rows = _build_phase_rows_with_reference(
        report=report,
        reference_report=best_history_report,
    )
    jump_brief = _build_jump_brief_summary(
        report=report,
        review=review,
        scorecard_rows=scorecard_rows,
        best_reference=best_reference,
        top_reference_jumps=top_reference_jumps,
        feedback_context=feedback_context,
    )
    jump_brief_simple = _build_jump_brief_simple(
        report=report,
        jump_brief=jump_brief,
        scorecard_rows=scorecard_rows,
        feedback_context=feedback_context,
    )
    quality_issue_lines = _build_quality_issue_lines(report.get("quality_flags", []))
    quality_issue_lines.extend(_build_rule_score_issue_lines(report.get("metrics", {})))
    quality_issue_lines.extend(_build_fs2_quality_issue_lines(report.get("notes", {})))
    quality_issue_lines = _unique_texts(quality_issue_lines)
    t0_diagnostics = _build_t0_diagnostics(report)
    ai_coaching = _build_ai_coaching(
        report=report,
        review=review,
        jump_brief=jump_brief,
        jump_brief_simple=jump_brief_simple,
        scorecard_rows=scorecard_rows,
        tip_follow_up=tip_follow_up,
        jumper_summary=jumper_summary,
        feedback_context=feedback_context,
        quality_issue_lines=quality_issue_lines,
        view_mode=view_mode,
    )
    if view_mode == _VIEW_MODE_EXPERT:
        coaching_snapshot = _build_coaching_snapshot(
            report=report,
            review=review,
            jump_brief=jump_brief,
            jump_brief_simple=jump_brief_simple,
            ai_coaching=ai_coaching,
            feedback_context=feedback_context,
            view_mode=view_mode,
        )
        if coaching_snapshot.get("available"):
            upsert_coaching_snapshot(jump_id, coaching_snapshot)
            report["coaching_snapshot"] = coaching_snapshot

    return templates.TemplateResponse(
        request,
        "jump_detail.html",
        {
            "jump_id": jump_id,
            "report": report,
            "review": review,
            "scorecard_rows": scorecard_rows,
            "jump_brief": jump_brief,
            "jump_brief_simple": jump_brief_simple,
            "speed_potential": speed_potential,
            "jumper_summary": jumper_summary,
            "best_reference": best_reference,
            "best_history_reference": best_history_reference,
            "top_reference_jumps": top_reference_jumps,
            "phase_rows": phase_rows,
            "tip_follow_up": tip_follow_up,
            "ai_coaching": ai_coaching,
            "feedback_context": feedback_context,
            "chart_data_json": _safe_json_dumps(_chart_data_for_client(report)),
            "report_meta_json": _safe_json_dumps(_report_for_client(report)),
            "quality_flags_json": _safe_json_dumps(report["quality_flags"]),
            "quality_issue_lines": quality_issue_lines,
            "t0_diagnostics": t0_diagnostics,
            "dropzone_options": list_dropzone_choices() if view_mode == _VIEW_MODE_EXPERT else [],
            "message": message,
            "error": error,
            "view_mode": view_mode,
            "needs_plotly": True,
        },
    )


@app.post("/jumps/{jump_id}/reprocess-t0")
def reprocess_t0(jump_id: str, view: str | None = None):
    view_mode = _normalize_view_mode(view)
    report = get_jump_report(jump_id)
    source_meta = get_jump_source_metadata(jump_id)
    if report is None or source_meta is None:
        raise HTTPException(status_code=404, detail="Sprung nicht gefunden.")

    source_path = _resolve_source_file_for_jump(source_meta)
    if source_path is None or not source_path.exists():
        msg = "Original-CSV nicht gefunden. Bitte Datei erneut hochladen."
        return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(msg)}", status_code=303)

    try:
        content = source_path.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        cached_path = _cache_uploaded_file(
            source_hash=source_hash,
            original_name=report["jump"]["file_name"],
            content=content,
        )
        ground_elevation_m = (
            report["jump"].get("ground_elevation_m")
            if report["jump"].get("ground_elevation_source") == "manual"
            else None
        )
        manual_dropzone_id = (
            report["jump"].get("dropzone_id")
            if report["jump"].get("dropzone_assignment_source") == "catalog_manual"
            else None
        )
        new_result = analyze_flysight_with_dropzone(
            content=content,
            file_name=report["jump"]["file_name"],
            jumper_name=report["jump"]["jumper_name"],
            ground_elevation_m=ground_elevation_m,
            breakoff_altitude_agl_m=report["jump"].get("breakoff_altitude_agl_m"),
            manual_dropzone_id=manual_dropzone_id,
        )
        replace_analysis_result(
            jump_id=jump_id,
            result=new_result,
            is_reference_only=bool(report["jump"].get("is_reference_only")),
            source_file_sha256=source_hash,
            source_file_path=str(cached_path),
        )
        _clear_derived_caches()
    except AnalysisError as exc:
        return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(str(exc))}", status_code=303)
    except Exception as exc:  # pragma: no cover
        msg = f"Unerwarteter Reanalysefehler: {exc}"
        return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(msg)}", status_code=303)

    ok_msg = "Absprung wurde neu erkannt und der Datensatz aktualisiert."
    return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&message={quote_plus(ok_msg)}", status_code=303)


@app.post("/jumps/{jump_id}/manual-t0")
def manual_t0(
    jump_id: str,
    new_t0_rel_s: str = Form(...),
    view: str | None = None,
):
    view_mode = _normalize_view_mode(view)
    report = get_jump_report(jump_id)
    source_meta = get_jump_source_metadata(jump_id)
    if report is None or source_meta is None:
        raise HTTPException(status_code=404, detail="Sprung nicht gefunden.")

    try:
        rel_s = float(str(new_t0_rel_s).replace(",", ".").strip())
    except ValueError:
        msg = "Bitte eine gueltige Sekundenangabe fuer den neuen Absprung eingeben."
        return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(msg)}", status_code=303)
    if not math.isfinite(rel_s) or abs(rel_s) > 120.0:
        msg = "Manuelle t0-Korrektur muss zwischen -120s und +120s relativ zum aktuellen t0 liegen."
        return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(msg)}", status_code=303)
    if abs(rel_s) < 0.05:
        msg = "Bitte eine Kurvenzeit ungleich 0.0s eingeben, wenn t0 manuell geaendert werden soll."
        return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(msg)}", status_code=303)

    source_path = _resolve_source_file_for_jump(source_meta)
    if source_path is None or not source_path.exists():
        msg = "Original-CSV nicht gefunden. Bitte Datei erneut hochladen."
        return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(msg)}", status_code=303)

    try:
        manual_t0_utc = _manual_t0_utc_from_current(report["jump"]["t0_utc"], rel_s)
        content = source_path.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        cached_path = _cache_uploaded_file(
            source_hash=source_hash,
            original_name=report["jump"]["file_name"],
            content=content,
        )
        ground_elevation_m = (
            report["jump"].get("ground_elevation_m")
            if report["jump"].get("ground_elevation_source") == "manual"
            else None
        )
        manual_dropzone_id = (
            report["jump"].get("dropzone_id")
            if report["jump"].get("dropzone_assignment_source") == "catalog_manual"
            else None
        )
        new_result = analyze_flysight_with_dropzone(
            content=content,
            file_name=report["jump"]["file_name"],
            jumper_name=report["jump"]["jumper_name"],
            ground_elevation_m=ground_elevation_m,
            breakoff_altitude_agl_m=report["jump"].get("breakoff_altitude_agl_m"),
            manual_dropzone_id=manual_dropzone_id,
            manual_t0_utc=manual_t0_utc,
        )
        replace_analysis_result(
            jump_id=jump_id,
            result=new_result,
            is_reference_only=bool(report["jump"].get("is_reference_only")),
            source_file_sha256=source_hash,
            source_file_path=str(cached_path),
        )
        _clear_derived_caches()
    except AnalysisError as exc:
        return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(str(exc))}", status_code=303)
    except Exception as exc:  # pragma: no cover
        msg = f"Unerwarteter Reanalysefehler: {exc}"
        return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(msg)}", status_code=303)

    ok_msg = f"Absprung wurde manuell auf aktuelle Kurvenzeit {rel_s:+.1f}s gesetzt und neu ausgewertet."
    return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&message={quote_plus(ok_msg)}", status_code=303)


@app.post("/jumps/{jump_id}/dropzone")
def update_jump_dropzone_route(
    jump_id: str,
    dropzone_id: str = Form(...),
    view: str | None = None,
):
    view_mode = _normalize_view_mode(view)
    report = get_jump_report(jump_id)
    source_meta = get_jump_source_metadata(jump_id)
    if report is None or source_meta is None:
        raise HTTPException(status_code=404, detail="Sprung nicht gefunden.")

    selected = str(dropzone_id or "").strip()
    manual_dropzone_id = None if selected in {"", "auto"} else selected
    if manual_dropzone_id is not None:
        valid_ids = {str(item["dropzone_id"]) for item in list_dropzone_choices()}
        if manual_dropzone_id not in valid_ids:
            msg = "Die gewählte Dropzone ist nicht im aktiven Katalog vorhanden."
            return RedirectResponse(
                url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(msg)}",
                status_code=303,
            )

    source_path = _resolve_source_file_for_jump(source_meta)
    if source_path is None or not source_path.exists():
        msg = "Original-CSV nicht gefunden. Bitte Datei erneut hochladen."
        return RedirectResponse(
            url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(msg)}",
            status_code=303,
        )

    try:
        content = source_path.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        cached_path = _cache_uploaded_file(
            source_hash=source_hash,
            original_name=report["jump"]["file_name"],
            content=content,
        )
        notes = report.get("notes") or {}
        manual_t0_utc = report["jump"].get("t0_utc") if notes.get("t0_manual_override") else None
        manual_ground = (
            report["jump"].get("ground_elevation_m")
            if report["jump"].get("ground_elevation_source") == "manual"
            else None
        )
        new_result = analyze_flysight_with_dropzone(
            content=content,
            file_name=report["jump"]["file_name"],
            jumper_name=report["jump"]["jumper_name"],
            ground_elevation_m=manual_ground,
            breakoff_altitude_agl_m=report["jump"].get("breakoff_altitude_agl_m"),
            manual_t0_utc=manual_t0_utc,
            manual_dropzone_id=manual_dropzone_id,
        )
        replace_analysis_result(
            jump_id=jump_id,
            result=new_result,
            jump_context=str(report["jump"].get("jump_context") or "unknown"),
            is_reference_only=bool(report["jump"].get("is_reference_only")),
            source_file_sha256=source_hash,
            source_file_path=str(cached_path),
        )
        _clear_derived_caches()
    except AnalysisError as exc:
        return RedirectResponse(
            url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(str(exc))}",
            status_code=303,
        )
    except Exception as exc:  # pragma: no cover
        msg = f"Unerwarteter Dropzone-Reanalysefehler: {exc}"
        return RedirectResponse(
            url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(msg)}",
            status_code=303,
        )

    match = new_result.get("dropzone_match") or {}
    if match.get("status") in {"accepted", "manual"}:
        ok_msg = f"Dropzone {match.get('dropzone_name')} wurde gespeichert und der Sprung neu ausgewertet."
    else:
        ok_msg = "Keine sichere Dropzone erkannt; die bisherige automatische Bodenhöhenschätzung wird verwendet."
    return RedirectResponse(
        url=f"/jumps/{jump_id}?view={view_mode}&message={quote_plus(ok_msg)}",
        status_code=303,
    )


@app.post("/jumps/{jump_id}/context")
def update_jump_context_route(
    jump_id: str,
    jump_context: str = Form(...),
    view: str | None = None,
):
    view_mode = _normalize_view_mode(view)
    resolved_context = _normalize_jump_context(jump_context, default="")
    if resolved_context not in VALID_JUMP_CONTEXTS:
        msg = "Ungültiger Sprung-Kontext."
        return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&error={quote_plus(msg)}", status_code=303)

    changed = update_jump_context(jump_id, resolved_context)
    if not changed:
        raise HTTPException(status_code=404, detail="Sprung nicht gefunden.")

    _clear_derived_caches()
    ok_msg = f"Sprung-Kontext aktualisiert: {_jump_context_label(resolved_context)}."
    return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&message={quote_plus(ok_msg)}", status_code=303)


@app.post("/jumps/{jump_id}/feedback")
def update_jump_feedback_route(
    jump_id: str,
    jump_feedback: str = Form(""),
    view: str | None = None,
):
    view_mode = _normalize_view_mode(view)
    changed = upsert_jump_feedback(jump_id, jump_feedback)
    if not changed:
        raise HTTPException(status_code=404, detail="Sprung nicht gefunden.")

    _clear_derived_caches()
    ok_msg = "Sprungfeedback aktualisiert." if str(jump_feedback or "").strip() else "Sprungfeedback entfernt."
    return RedirectResponse(url=f"/jumps/{jump_id}?view={view_mode}&message={quote_plus(ok_msg)}", status_code=303)


@app.post("/jumps/{jump_id}/delete")
def delete_jump_dataset(jump_id: str, view: str | None = None):
    view_mode = _normalize_view_mode(view)
    summary = get_jump_summary(jump_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Sprung nicht gefunden.")

    file_name = str(summary.get("file_name") or jump_id)
    deleted = delete_jump(jump_id)
    if not deleted:
        msg = "Datensatz konnte nicht gelöscht werden."
        return RedirectResponse(url=f"/?view={view_mode}&error={quote_plus(msg)}", status_code=303)
    _clear_derived_caches()

    ok_msg = f"Datensatz gelöscht: {file_name}"
    return RedirectResponse(url=f"/?view={view_mode}&message={quote_plus(ok_msg)}", status_code=303)


@app.get("/jumps/{jump_id}/compare")
def jump_compare(
    request: Request,
    jump_id: str,
    compare_jump_id: str | None = None,
    preset: str | None = None,
    view: str | None = None,
):
    base_report = get_jump_report(jump_id)
    if base_report is None:
        raise HTTPException(status_code=404, detail="Basis-Sprung nicht gefunden.")

    base_jump = base_report["jump"]
    base_summary = get_jump_summary(jump_id)
    if base_summary is None:
        raise HTTPException(status_code=404, detail="Basis-Sprung nicht gefunden.")

    same_jumper_jumps = list_jumps_for_jumper(base_jump["jumper_name"])
    for jump in same_jumper_jumps:
        jump["is_current"] = jump["jump_id"] == jump_id

    compare_candidates = list_compare_candidates(jump_id)
    same_candidates = [item for item in compare_candidates if item["jumper_name"] == base_jump["jumper_name"]]
    other_candidates = [item for item in compare_candidates if item["jumper_name"] != base_jump["jumper_name"]]

    compare_error: str | None = None
    compare_result: dict[str, Any] | None = None
    compare_chart_json: str | None = None
    compare_alignment: dict[str, Any] | None = None
    view_mode = _normalize_view_mode(view)

    if preset == "best":
        best = get_best_jump_for_jumper(base_jump["jumper_name"], exclude_jump_id=jump_id)
        if best is None:
            compare_error = "Kein weiterer Sprung dieses Springers vorhanden, um mit dem besten Sprung zu vergleichen."
        else:
            compare_jump_id = best["jump_id"]

    if compare_jump_id:
        compare_summary = get_jump_summary(compare_jump_id)
        if compare_summary is None:
            compare_error = "Vergleichssprung nicht gefunden."
        elif compare_jump_id == jump_id:
            compare_error = "Bitte einen anderen Sprung als Vergleich auswählen."
        else:
            compare_report = get_jump_report(compare_jump_id)
            if compare_report is None:
                compare_error = "Vergleichssprung nicht gefunden."
            else:
                compare_result = build_jump_comparison(
                    left_report=base_report,
                    right_report=compare_report,
                )
                reports_by_id = {
                    str(base_report["jump"]["jump_id"]): base_report,
                    str(compare_report["jump"]["jump_id"]): compare_report,
                }
                compare_alignment = _build_compare_alignment(
                    reference_report=reports_by_id[str(compare_result["reference"]["jump_id"])],
                    comparison_report=reports_by_id[str(compare_result["comparison"]["jump_id"])],
                )
                compare_chart_json = _safe_json_dumps(compare_result["charts"])

    return templates.TemplateResponse(
        request,
        "jump_compare.html",
        {
            "jump_id": jump_id,
            "base_report": base_report,
            "base_summary": base_summary,
            "same_jumper_jumps": same_jumper_jumps,
            "same_candidates": same_candidates,
            "other_candidates": other_candidates,
            "compare_error": compare_error,
            "compare_result": compare_result,
            "compare_alignment": compare_alignment,
            "compare_chart_json": compare_chart_json,
            "selected_compare_jump_id": compare_jump_id,
            "view_mode": view_mode,
            "needs_plotly": True,
        },
    )


@app.get("/jumpers/{jumper_name}")
def jumper_view(request: Request, jumper_name: str, view: str | None = None):
    jumps = list_jumps_for_jumper(jumper_name)
    if not jumps:
        raise HTTPException(status_code=404, detail="Springer nicht gefunden.")

    jumps = _annotate_best_jump(jumps)
    jumper_summary = _build_jumper_summary(jumper_name=jumper_name, jumps=jumps)
    view_mode = _normalize_view_mode(view)
    return templates.TemplateResponse(
        request,
        "jumper_detail.html",
        {
            "jumper_name": jumper_name,
            "view_mode": view_mode,
            "jumps": jumps,
            "jumper_summary": jumper_summary,
            "compare_result": None,
            "compare_alignment": None,
            "compare_chart_json": None,
            "compare_error": None,
            "left_jump_id": None,
            "right_jump_id": None,
            "needs_plotly": True,
        },
    )


@app.get("/jumpers/{jumper_name}/compare")
def jumper_compare(
    request: Request,
    jumper_name: str,
    left_jump_id: str,
    right_jump_id: str,
    view: str | None = None,
):
    jumps = list_jumps_for_jumper(jumper_name)
    if not jumps:
        raise HTTPException(status_code=404, detail="Springer nicht gefunden.")

    jumps = _annotate_best_jump(jumps)
    jumper_summary = _build_jumper_summary(jumper_name=jumper_name, jumps=jumps)
    view_mode = _normalize_view_mode(view)
    jump_ids = {jump["jump_id"] for jump in jumps}
    compare_error: str | None = None
    compare_result: dict[str, Any] | None = None
    compare_chart_json: str | None = None

    if left_jump_id == right_jump_id:
        compare_error = "Bitte zwei unterschiedliche Sprünge auswählen."
    elif left_jump_id not in jump_ids or right_jump_id not in jump_ids:
        compare_error = "Vergleich ungültig: Sprünge gehoeren nicht zu diesem Springer."
    else:
        left_report = get_jump_report(left_jump_id)
        right_report = get_jump_report(right_jump_id)
        if left_report is None or right_report is None:
            compare_error = "Vergleich ungültig: Mindestens ein Sprung wurde nicht gefunden."
        else:
            compare_result = build_jump_comparison(left_report=left_report, right_report=right_report)
            reports_by_id = {
                str(left_report["jump"]["jump_id"]): left_report,
                str(right_report["jump"]["jump_id"]): right_report,
            }
            compare_alignment = _build_compare_alignment(
                reference_report=reports_by_id[str(compare_result["reference"]["jump_id"])],
                comparison_report=reports_by_id[str(compare_result["comparison"]["jump_id"])],
            )
            compare_chart_json = _safe_json_dumps(compare_result["charts"])

    return templates.TemplateResponse(
        request,
        "jumper_detail.html",
        {
            "jumper_name": jumper_name,
            "view_mode": view_mode,
            "jumps": jumps,
            "jumper_summary": jumper_summary,
            "compare_result": compare_result,
            "compare_alignment": compare_alignment,
            "compare_chart_json": compare_chart_json,
            "compare_error": compare_error,
            "left_jump_id": left_jump_id,
            "right_jump_id": right_jump_id,
            "needs_plotly": True,
        },
    )


@app.get("/coach")
def coach_view(request: Request, jumper_name: str | None = None, view: str | None = None):
    if not COACH_VIEW_ENABLED:
        raise HTTPException(status_code=404, detail="Traineransicht ist deaktiviert.")

    jumpers = list_jumpers()
    selected_jumper = jumper_name.strip() if jumper_name else None
    view_mode = _normalize_view_mode(view)
    if selected_jumper and selected_jumper not in jumpers:
        raise HTTPException(status_code=404, detail="Springer nicht gefunden.")

    top_reference_jumps = list_top_global_references(limit=5)
    top_reference_reports: dict[str, dict[str, Any]] = {}
    for item in top_reference_jumps:
        rep = get_jump_report(item["jump_id"])
        if rep is not None:
            top_reference_reports[item["jump_id"]] = rep

    target_jumpers = [selected_jumper] if selected_jumper else jumpers
    groups: list[dict[str, Any]] = []

    for jumper in target_jumpers:
        jump_rows = _sort_by_t0_desc(list_jumps_for_jumper(jumper))
        if not jump_rows:
            continue
        jump_rows = _annotate_best_jump(jump_rows)
        jumper_summary = _build_jumper_summary(jumper_name=jumper, jumps=jump_rows)
        best = next((item for item in jump_rows if item.get("is_best")), None)
        best_report = get_jump_report(best["jump_id"]) if best is not None else None

        jumps_for_coach: list[dict[str, Any]] = []
        for item in jump_rows:
            report = get_jump_report(item["jump_id"])
            if report is None:
                continue

            best_compare: dict[str, Any] | None = None
            delta_to_best: float | None = None
            if (
                best is not None
                and best_report is not None
                and item["jump_id"] != best["jump_id"]
                and item.get("rule_score_status") == "valid"
            ):
                best_compare = build_jump_comparison(left_report=report, right_report=best_report)
                by_label = {row["label"]: row for row in best_compare.get("summary", [])}
                row = by_label.get("Rule Score")
                if row and row.get("delta") is not None:
                    delta_to_best = float(row["delta"])

            top_reference_compares: list[dict[str, Any]] = []
            for top_item in top_reference_jumps:
                ref_id = str(top_item.get("jump_id"))
                if ref_id == str(item["jump_id"]):
                    continue
                ref_report = top_reference_reports.get(ref_id)
                if ref_report is None:
                    continue
                top_reference_compares.append(
                    build_jump_comparison(left_report=report, right_report=ref_report)
                )

            review = build_jump_review(
                report,
                best_compare=best_compare,
                top_reference_compares=top_reference_compares,
                jumper_stability_reference=(
                    jumper_summary.get("stability_reference")
                    if isinstance(jumper_summary, dict)
                    else None
                ),
                tip_effect_profile=(
                    jumper_summary.get("tip_effect_profile")
                    if isinstance(jumper_summary, dict)
                    else None
                ),
            )
            status = _coach_status(report)
            jumps_for_coach.append(
                {
                    "jump_id": item["jump_id"],
                    "file_name": item["file_name"],
                    "jump_context": item.get("jump_context", "unknown"),
                    "t0_utc": item["t0_utc"],
                    "best_3s_vVert_kmh": item["best_3s_vVert_kmh"],
                    "rule_based_3s_score": item.get("rule_based_3s_score"),
                    "rule_score_status": item.get("rule_score_status"),
                    "delta_to_best": delta_to_best,
                    "is_best": bool(item.get("is_best")),
                    "status": status,
                    "good_points": review["good"][:2],
                    "weak_points": review["not_good"][:2],
                    "actions": review["improve"][:3],
                }
            )

        jumps_for_coach = _sort_by_t0_desc(jumps_for_coach)
        groups.append(
            {
                "jumper_name": jumper,
                "best_jump_id": None if best is None else best["jump_id"],
                "best_file_name": None if best is None else best["file_name"],
                "best_score": None if best is None else best.get("rule_based_3s_score"),
                "latest_t0_utc": jumps_for_coach[0]["t0_utc"] if jumps_for_coach else None,
                "stability_reference": jumper_summary.get("stability_reference")
                if isinstance(jumper_summary, dict)
                else None,
                "jumps": jumps_for_coach,
            }
        )

    groups = sorted(groups, key=lambda item: _t0_sort_key(item.get("latest_t0_utc")), reverse=True)

    return templates.TemplateResponse(
        request,
        "coach.html",
        {
            "jumpers": jumpers,
            "selected_jumper": selected_jumper,
            "groups": groups,
            "view_mode": view_mode,
        },
    )


def _parse_optional_float(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Ungültiger Zahlenwert: {raw}") from exc


def _build_legacy_scorecard_rows(
    report: dict[str, Any],
    *,
    marco_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    metrics = report.get("metrics", {})
    notes = report.get("notes", {})
    scorecard = report.get("scorecard", {})
    chart = report.get("chart_data", {})
    snapshot = _scorecard_metric_snapshot(report)

    v10 = _to_float(snapshot.get("v10"))
    a20 = _to_float(snapshot.get("angle_20"))
    gain_10_20 = _to_float(snapshot.get("gain_10_20"))
    build_coverage = _to_float(snapshot.get("build_coverage")) or 0.0
    hot_coverage = _to_float(snapshot.get("hot_coverage")) or 0.0
    build_start_s = _to_float(snapshot.get("build_start_s"))
    build_end_s = _to_float(snapshot.get("build_end_s"))
    build_window_label = (
        "-"
        if build_start_s is None or build_end_s is None
        else f"+{build_start_s:.1f}s bis +{build_end_s:.1f}s"
    )

    exit_profile = notes.get("exit_profile", {}) if isinstance(notes.get("exit_profile"), dict) else {}
    carry_ratio = _to_float(exit_profile.get("carry_ratio"))
    exit_unsteady = bool(exit_profile.get("unsteady"))
    if carry_ratio is None:
        carry_ratio = _estimate_carry_ratio(chart)

    # 1) Exit
    exit_score = 50
    if v10 is not None:
        if 240.0 <= v10 <= 320.0:
            exit_score += 22
        elif 220.0 <= v10 < 240.0:
            exit_score += 10
        elif v10 < 220.0:
            exit_score -= 10
        else:
            exit_score += 12
    else:
        exit_score -= 12
    if carry_ratio is not None:
        if carry_ratio >= 0.8:
            exit_score += 18
        elif carry_ratio >= 0.7:
            exit_score += 10
        elif carry_ratio >= 0.6:
            exit_score += 3
        else:
            exit_score -= 10
    if exit_unsteady:
        exit_score -= 10
    exit_score = _clamp_score(exit_score)
    exit_reason_lines = _build_exit_reason_lines(
        v10=v10,
        carry_ratio=carry_ratio,
        exit_unsteady=exit_unsteady,
    )
    exit_reason = _join_reason_lines(exit_reason_lines)

    # 2) Aufbau 10-20s
    build_score = 50
    if gain_10_20 is not None:
        if 95.0 <= gain_10_20 <= 170.0:
            build_score += 24
        elif 80.0 <= gain_10_20 < 95.0:
            build_score += 12
        elif gain_10_20 < 80.0:
            build_score -= 12
        else:
            build_score += 8
    else:
        build_score -= 15
    if a20 is not None:
        if 82.0 <= a20 <= 85.5:
            build_score += 18
        elif 80.0 <= a20 <= 87.0:
            build_score += 10
        else:
            build_score -= 12
    else:
        build_score -= 8
    build_score = _clamp_score(build_score)
    if build_coverage < 0.75:
        build_score = max(build_score, 60)
    build_reason_lines = _build_build_reason_lines(
        gain_10_20=gain_10_20,
        angle_20=a20,
        phase_window_label=build_window_label,
    )
    if build_coverage < 0.75:
        build_reason_lines.insert(0, "Die Aufbauphase ist im individuellen Messfenster nicht vollständig abgedeckt.")
    build_reason = _join_reason_lines(build_reason_lines)

    # 3) Hot-Zone
    hot_zone = _build_hot_zone_assessment(
        notes=notes,
        metrics=metrics,
        scorecard=scorecard,
        chart=chart,
    )
    hot_score = int(hot_zone["score"])
    hot_reason = str(hot_zone["reason"])
    hot_reason_lines = list(hot_zone.get("reason_lines") or [hot_reason])
    if hot_coverage < 0.75:
        hot_reason_lines.insert(0, "Die Hot-Zone ist im individuellen Messfenster nur eingeschränkt abgedeckt.")

    # 4) Stabilität / Kipp-Risiko
    risk_score_raw = _to_float(metrics.get("negative_risk_score"))
    risk_label = str(scorecard.get("kipp_risiko") or "")
    if risk_score_raw is not None:
        stability_score = _clamp_score(100 - risk_score_raw)
    else:
        stability_score = 70
    if risk_label == "hoch":
        stability_score = min(stability_score, 45)
    elif risk_label == "mittel":
        stability_score = min(stability_score, 65)
    raw_negative_details = str(notes.get("negative_details") or "")
    stability_reason_lines = _describe_stability_reason_lines(
        raw_negative_details=raw_negative_details,
        risk_label=risk_label,
    )
    forward_eval_end_s = _forward_track_eval_end_s(notes=notes)
    stability_reason_lines.extend(
        _build_forward_track_reason_lines(
            chart=chart,
            end_s=forward_eval_end_s,
        )
    )
    stability_reason = _join_reason_lines(stability_reason_lines)

    if marco_profile is not None:
        marco_scores = _compute_marco_percent_scores(
            snapshot=snapshot,
            marco_profile=marco_profile,
            exit_unsteady=exit_unsteady,
            risk_score_raw=risk_score_raw,
        )
        exit_score = marco_scores["exit"]
        build_score = marco_scores["build"]
        hot_score = marco_scores["hot"]
        stability_score = marco_scores["stability"]
        exit_reason_lines = _add_reference_score_context(
            phase_name="Exit",
            score=exit_score,
            reason_lines=exit_reason_lines,
            snapshot=snapshot,
            marco_profile=marco_profile,
        )
        build_reason_lines = _add_reference_score_context(
            phase_name="Aufbau 10-20s",
            score=build_score,
            reason_lines=build_reason_lines,
            snapshot=snapshot,
            marco_profile=marco_profile,
        )
        hot_reason_lines = _add_reference_score_context(
            phase_name="Hot-Zone",
            score=hot_score,
            reason_lines=hot_reason_lines,
            snapshot=snapshot,
            marco_profile=marco_profile,
        )
        stability_reason_lines = _add_reference_score_context(
            phase_name="Stabilität / Kipp-Risiko",
            score=stability_score,
            reason_lines=stability_reason_lines,
            snapshot=snapshot,
            marco_profile=marco_profile,
        )
        exit_reason = _join_reason_lines(exit_reason_lines)
        build_reason = _join_reason_lines(build_reason_lines)
        hot_reason = _join_reason_lines(hot_reason_lines)
        stability_reason = _join_reason_lines(stability_reason_lines)

    rows = [
        {
            "name": "Exit",
            "status": _score_status_label(exit_score),
            "score": exit_score,
            "reason": exit_reason,
            "reason_lines": exit_reason_lines,
        },
        {
            "name": "Aufbau 10-20s",
            "status": _score_status_label(build_score),
            "score": build_score,
            "reason": build_reason,
            "reason_lines": build_reason_lines,
        },
        {
            "name": "Hot-Zone",
            "status": _score_status_label(hot_score),
            "score": hot_score,
            "reason": hot_reason,
            "reason_lines": hot_reason_lines,
        },
        {
            "name": "Stabilität / Kipp-Risiko",
            "status": _score_status_label(stability_score),
            "score": stability_score,
            "reason": stability_reason,
            "reason_lines": stability_reason_lines,
        },
    ]
    return rows


def _build_scorecard_rows(
    report: dict[str, Any],
    *,
    marco_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    legacy_rows = _build_legacy_scorecard_rows(report, marco_profile=marco_profile)
    phase_rows = _phase_rows_for_report(report)
    technical_rows = _build_technical_phase_scorecard_rows(
        report=report,
        phase_rows=phase_rows,
        legacy_rows=legacy_rows,
    )
    return technical_rows if technical_rows else legacy_rows


def _build_technical_phase_scorecard_rows(
    *,
    report: dict[str, Any],
    phase_rows: list[dict[str, Any]],
    legacy_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_name = {str(row.get("name") or ""): row for row in phase_rows}
    legacy_scores = {
        "exit": _legacy_score_for_phase(legacy_rows, "exit"),
        "build": _legacy_score_for_phase(legacy_rows, "aufbau"),
        "hot": _legacy_score_for_phase(legacy_rows, "hot"),
        "stability": _legacy_score_for_phase(legacy_rows, "stabil"),
    }
    legacy_reasons = {
        "exit": _legacy_reason_lines_for_phase(legacy_rows, "exit"),
        "build": _legacy_reason_lines_for_phase(legacy_rows, "aufbau"),
        "hot": _legacy_reason_lines_for_phase(legacy_rows, "hot"),
        "stability": _legacy_reason_lines_for_phase(legacy_rows, "stabil"),
    }

    chart = report.get("chart_data", {}) if isinstance(report.get("chart_data"), dict) else {}
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    scorecard = report.get("scorecard", {}) if isinstance(report.get("scorecard"), dict) else {}
    reserve_context = _early_horizontal_reserve_context(chart=chart, metrics=metrics)
    out: list[dict[str, Any]] = []

    for spec in TECHNICAL_PHASE_SPECS:
        name = str(spec["name"])
        row = by_name.get(name)
        if row is None:
            continue

        start_s = _to_float(row.get("start_s"))
        end_s = _to_float(row.get("end_s"))
        target_status = str(row.get("target_status") or "")
        angle_score = _technical_phase_status_score(target_status)
        gain = None
        vhor_min = None
        turns = None
        dur_400 = None
        if start_s is not None and end_s is not None:
            gain = _window_delta(
                time_s=chart.get("time_s", []),
                values=chart.get("vVert_kmh", []),
                start_s=float(start_s),
                end_s=float(end_s),
            )
            vhor_min = _window_min(
                time_s=chart.get("time_s", []),
                values=chart.get("vHor_kmh", []),
                start_s=float(start_s),
                end_s=float(end_s),
            )
            turns = _window_turn_count(
                time_s=chart.get("time_s", []),
                values=chart.get("angle_deg", []),
                start_s=float(start_s),
                end_s=float(end_s),
                eps=0.35,
            )
            dur_400 = _duration_above_threshold(
                time_s=chart.get("time_s", []),
                values=chart.get("vVert_kmh", []),
                threshold=400.0,
                start_s=float(start_s),
                end_s=float(end_s),
            )

        score = _technical_phase_score(
            phase_name=name,
            angle_score=angle_score,
            legacy_scores=legacy_scores,
            gain_kmh=gain,
            vhor_min_kmh=vhor_min,
            turns=turns,
            dur_400_s=dur_400,
            negative_risk_score=_to_float(metrics.get("negative_risk_score")),
            risk_label=str(scorecard.get("kipp_risiko") or ""),
            early_reserve_collapse=bool(reserve_context.get("collapse")),
        )
        reason_lines = _technical_phase_score_reason_lines(
            row=row,
            phase_name=name,
            gain_kmh=gain,
            vhor_min_kmh=vhor_min,
            turns=turns,
            dur_400_s=dur_400,
            reserve_context=reserve_context,
            legacy_reason_lines=_phase_relevant_legacy_reason_lines(
                phase_name=name,
                legacy_reasons=legacy_reasons,
            ),
        )
        out.append(
            {
                "name": name,
                "status": _score_status_label(score),
                "score": score,
                "reason": _join_reason_lines(reason_lines),
                "reason_lines": reason_lines,
                "target_status": target_status,
                "target_angle_label": row.get("target_angle_label"),
                "start_s": row.get("start_s"),
                "end_s": row.get("end_s"),
                "avg_angle_deg": row.get("avg_angle_deg"),
                "angle_start_deg": row.get("angle_start_deg"),
                "angle_end_deg": row.get("angle_end_deg"),
                "angle_delta_deg": row.get("angle_delta_deg"),
                "avg_vVert_kmh": row.get("avg_vVert_kmh"),
                "avg_vHor_kmh": row.get("avg_vHor_kmh"),
                "min_angle_deg": row.get("min_angle_deg"),
                "max_angle_deg": row.get("max_angle_deg"),
            }
        )

    return out


def _legacy_score_for_phase(rows: list[dict[str, Any]], token: str) -> int | None:
    token_norm = normalize_german_text(token)
    for row in rows:
        name = normalize_german_text(str(row.get("name") or ""))
        if token_norm in name:
            try:
                return int(row.get("score", 0))
            except (TypeError, ValueError):
                return None
    return None


def _legacy_reason_lines_for_phase(rows: list[dict[str, Any]], token: str) -> list[str]:
    token_norm = normalize_german_text(token)
    for row in rows:
        name = normalize_german_text(str(row.get("name") or ""))
        if token_norm not in name:
            continue
        raw_lines = row.get("reason_lines")
        if isinstance(raw_lines, list):
            return [str(item).strip() for item in raw_lines if str(item).strip()]
        reason = str(row.get("reason") or "").strip()
        return [reason] if reason else []
    return []


def _technical_phase_status_score(status: str) -> int | None:
    status_norm = normalize_german_text(status)
    if not status_norm or "unbekannt" in status_norm:
        return None
    if "nicht belastbar" in status_norm:
        return 58
    if "im zielbereich" in status_norm:
        return 88
    if "eher flach" in status_norm or "kurz zu steil" in status_norm:
        return 70
    if "zu flach" in status_norm or "zu steil" in status_norm:
        return 50
    return 68


def _technical_phase_score(
    *,
    phase_name: str,
    angle_score: int | None,
    legacy_scores: dict[str, int | None],
    gain_kmh: float | None,
    vhor_min_kmh: float | None,
    turns: float | None,
    dur_400_s: float | None,
    negative_risk_score: float | None,
    risk_label: str,
    early_reserve_collapse: bool = False,
) -> int:
    parts: list[tuple[int | None, float]] = []
    if phase_name == "Exit / Stabilisierung":
        parts = [(angle_score, 0.35), (legacy_scores.get("exit"), 0.65)]
    elif phase_name == "Dive-Aufbau":
        parts = [
            (angle_score, 0.45),
            (legacy_scores.get("build"), 0.25),
            (_phase_gain_score(gain_kmh, phase_name=phase_name), 0.30),
        ]
    elif phase_name == "Hauptbeschleunigung":
        parts = [
            (angle_score, 0.45),
            (legacy_scores.get("build"), 0.35),
            (_phase_gain_score(gain_kmh, phase_name=phase_name), 0.20),
        ]
    elif phase_name == "Hot-Zone Aufbau":
        parts = [
            (angle_score, 0.35),
            (legacy_scores.get("hot"), 0.30),
            (_phase_vhor_score(vhor_min_kmh), 0.20),
            (_phase_gain_score(gain_kmh, phase_name=phase_name), 0.15),
        ]
    elif phase_name == "Max-Speed Fenster":
        risk_score = None if negative_risk_score is None else _clamp_score(100.0 - float(negative_risk_score))
        if risk_label == "hoch":
            risk_score = min(45, risk_score if risk_score is not None else 45)
        elif risk_label == "mittel":
            risk_score = min(65, risk_score if risk_score is not None else 65)
        parts = [
            (angle_score, 0.25),
            (legacy_scores.get("hot"), 0.20),
            (legacy_scores.get("stability"), 0.20),
            (_phase_vhor_score(vhor_min_kmh), 0.15),
            (_phase_turn_score(turns), 0.10),
            (_phase_hold_score(dur_400_s), 0.10),
            (risk_score, 0.10),
        ]

    values = [(float(score), float(weight)) for score, weight in parts if score is not None and weight > 0]
    if not values:
        score = 60
    else:
        total_weight = sum(weight for _, weight in values)
        if total_weight <= 0:
            score = 60
        else:
            score = _clamp_score(sum(score * weight for score, weight in values) / total_weight)

    if early_reserve_collapse:
        if phase_name == "Hot-Zone Aufbau":
            score = min(score, 62)
        elif phase_name == "Max-Speed Fenster":
            score = min(score, 64)
    return score


def _phase_gain_score(gain_kmh: float | None, *, phase_name: str) -> int | None:
    if gain_kmh is None:
        return None
    gain = float(gain_kmh)
    min_expected = {
        "Dive-Aufbau": 45.0,
        "Hauptbeschleunigung": 55.0,
        "Hot-Zone Aufbau": 20.0,
        "Max-Speed Fenster": 0.0,
    }.get(phase_name, 20.0)
    strong_expected = min_expected * 1.6
    if gain >= strong_expected:
        return 90
    if gain >= min_expected:
        return 76
    if gain >= 0.0:
        return 60
    return 42


def _phase_vhor_score(vhor_min_kmh: float | None) -> int | None:
    if vhor_min_kmh is None:
        return None
    value = float(vhor_min_kmh)
    if value >= 40.0:
        return 92
    if value >= 30.0:
        return 80
    if value >= 25.0:
        return 66
    return 44


def _phase_turn_score(turns: float | None) -> int | None:
    if turns is None:
        return None
    value = float(turns)
    if value <= 1.0:
        return 90
    if value <= 3.0:
        return 76
    if value <= 5.0:
        return 62
    return 44


def _phase_hold_score(dur_400_s: float | None) -> int | None:
    if dur_400_s is None:
        return None
    value = float(dur_400_s)
    if value >= 3.0:
        return 90
    if value >= 1.5:
        return 76
    if value > 0.0:
        return 62
    return 48


def _technical_phase_score_reason_lines(
    *,
    row: dict[str, Any],
    phase_name: str,
    gain_kmh: float | None,
    vhor_min_kmh: float | None,
    turns: float | None,
    dur_400_s: float | None,
    reserve_context: dict[str, Any] | None,
    legacy_reason_lines: list[str],
) -> list[str]:
    start_s = _to_float(row.get("start_s"))
    end_s = _to_float(row.get("end_s"))
    window_label = "-" if start_s is None or end_s is None else f"+{start_s:.1f}s bis +{end_s:.1f}s"
    status = str(row.get("target_status") or "unbekannt")
    target = str(row.get("target_angle_label") or "-")
    lines = [f"{window_label}, Zielwinkel {target}."]
    avg_angle = _to_float(row.get("avg_angle_deg"))
    angle_start = _to_float(row.get("angle_start_deg"))
    angle_end = _to_float(row.get("angle_end_deg"))
    angle_delta = _to_float(row.get("angle_delta_deg"))
    min_angle = _to_float(row.get("min_angle_deg"))
    max_angle = _to_float(row.get("max_angle_deg"))
    avg_vvert = _to_float(row.get("avg_vVert_kmh"))
    avg_vhor = _to_float(row.get("avg_vHor_kmh"))
    if avg_angle is not None or avg_vvert is not None or avg_vhor is not None:
        lines.append(
            f"Istwerte: Avg Winkel {_fmt1(avg_angle)} Grad, Avg vVert {_fmt1(avg_vvert)} km/h, "
            f"Avg vHor {_fmt1(avg_vhor)} km/h."
        )
    lines.append(
        _technical_phase_status_reason(
            status=status,
            avg_angle=avg_angle,
            min_angle=min_angle,
            max_angle=max_angle,
            target=target,
        )
    )
    angle_progression_line = _scorecard_angle_progression_line(
        status=status,
        start_angle=angle_start,
        end_angle=angle_end,
        delta_angle=angle_delta,
    )
    if angle_progression_line:
        lines.append(angle_progression_line)
    reserve_line = ""
    if phase_name in {"Hot-Zone Aufbau", "Max-Speed Fenster"}:
        reserve_line = _early_horizontal_reserve_reason_line(reserve_context)
        if reserve_line:
            lines.append(reserve_line)
    comment = str(row.get("comment") or "").strip()
    if reserve_line and phase_name == "Max-Speed Fenster" and "technisch nutzbar" in normalize_german_text(comment):
        comment = (
            "Das spaetere 3s-Fenster ist nutzbar, aber erst nach vorherigem Verlust der horizontalen Reserve."
        )
    elif reserve_line and comment == "Phase liegt im technischen Zielbereich.":
        comment = "Der Winkel-Durchschnitt passt, aber die horizontale Reserve bricht vorher zu frueh weg."
    if (
        status == "kurz zu steil"
        and "durchschnitt liegt im zielbereich" in normalize_german_text(comment)
        and "max winkel" in normalize_german_text(comment)
    ):
        comment = ""
    if comment:
        lines.append(comment)
    if phase_name in {"Dive-Aufbau", "Hauptbeschleunigung", "Hot-Zone Aufbau"}:
        lines.append(f"vVert-Zuwachs im Fenster {_fmt1(gain_kmh)} km/h.")
    if phase_name in {"Hot-Zone Aufbau", "Max-Speed Fenster"}:
        lines.append(f"vHor-Min im Fenster {_fmt1(vhor_min_kmh)} km/h.")
    if phase_name == "Max-Speed Fenster":
        lines.append(f">400-km/h-Dauer im Fenster {_fmt1(dur_400_s)}s, Winkel-Korrekturen {_fmt1(turns)}.")

    return lines


def _scorecard_angle_progression_line(
    *,
    status: str,
    start_angle: float | None,
    end_angle: float | None,
    delta_angle: float | None,
) -> str:
    status_norm = normalize_german_text(str(status or ""))
    if not any(marker in status_norm for marker in ["zu steil", "zu flach", "eher flach", "kurz zu steil"]):
        return ""
    text = _format_angle_progression(
        start_angle=start_angle,
        end_angle=end_angle,
        delta_angle=delta_angle,
    )
    if text == "-":
        return ""
    return f"Winkelverlauf: {text}."


def _technical_phase_status_reason(
    *,
    status: str,
    avg_angle: float | None,
    min_angle: float | None,
    max_angle: float | None,
    target: str,
) -> str:
    status_text = str(status or "unbekannt")
    target_text = str(target or "-")
    if status_text == "zu flach" and avg_angle is not None:
        return f"Bewertung: zu flach, weil der Avg Winkel {_fmt1(avg_angle)} Grad unter dem Ziel {target_text} liegt."
    if status_text == "zu steil" and avg_angle is not None:
        return f"Bewertung: zu steil, weil der Avg Winkel {_fmt1(avg_angle)} Grad ueber dem Ziel {target_text} liegt."
    if status_text == "eher flach":
        if avg_angle is not None and min_angle is not None:
            return (
                f"Bewertung: eher flach, weil der Avg Winkel {_fmt1(avg_angle)} Grad zum Zielband "
                f"{target_text} passt, der Min Winkel {_fmt1(min_angle)} Grad aber darunter liegt."
            )
        if min_angle is not None:
            return (
                f"Bewertung: eher flach, weil der Durchschnitt zwar passt, "
                f"aber der Min Winkel {_fmt1(min_angle)} Grad unter dem Ziel {target_text} liegt."
            )
        return "Bewertung: eher flach, weil einzelne Abschnitte unter dem Zielwinkel liegen."
    if status_text == "kurz zu steil":
        if avg_angle is not None and max_angle is not None:
            return (
                f"Bewertung: kurz zu steil, weil der Avg Winkel {_fmt1(avg_angle)} Grad zum Zielband "
                f"{target_text} passt, der Max Winkel {_fmt1(max_angle)} Grad aber kurz darueber liegt."
            )
        if max_angle is not None:
            return (
                f"Bewertung: kurz zu steil, weil der Max Winkel {_fmt1(max_angle)} Grad "
                f"ueber dem Ziel {target_text} liegt."
            )
        return "Bewertung: kurz zu steil, weil einzelne Abschnitte ueber dem Zielwinkel liegen."
    if status_text == "im Zielbereich":
        return f"Bewertung: im Zielbereich, weil der Avg Winkel {_fmt1(avg_angle)} Grad zum Ziel {target_text} passt."
    if status_text == "nicht belastbar":
        return "Bewertung: nicht belastbar, weil in dieser Phase nicht genug verwertbare Daten vorliegen."
    return f"Bewertung: {status_text}."


def _scorecard_row_has_angle_issue(row: dict[str, Any]) -> bool:
    status = normalize_german_text(str(row.get("target_status") or ""))
    return any(
        marker in status
        for marker in ["zu steil", "zu flach", "eher flach", "kurz zu steil"]
    )


def _phase_relevant_legacy_reason_lines(
    *,
    phase_name: str,
    legacy_reasons: dict[str, list[str]],
) -> list[str]:
    if phase_name == "Exit / Stabilisierung":
        return legacy_reasons.get("exit", [])
    if phase_name in {"Dive-Aufbau", "Hauptbeschleunigung"}:
        return legacy_reasons.get("build", [])
    if phase_name == "Hot-Zone Aufbau":
        return legacy_reasons.get("hot", [])
    if phase_name == "Max-Speed Fenster":
        return (legacy_reasons.get("hot", [])[:1] + legacy_reasons.get("stability", [])[:2])[:3]
    return []


def _build_phase_rows_with_reference(
    *,
    report: dict[str, Any],
    reference_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    phases = _phase_rows_for_report(report)
    phase_has_gap = _phase_gap_map(report)
    ref_by_name: dict[str, dict[str, Any]] = {}
    if reference_report is not None:
        for row in _phase_rows_for_report(reference_report):
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            ref_by_name[name] = row

    out: list[dict[str, Any]] = []
    for row in phases:
        name = str(row.get("name") or "").strip()
        ref_row = ref_by_name.get(name, {})
        item = dict(row)
        gap_in_phase = bool(phase_has_gap.get(name))
        if gap_in_phase:
            item["avg_vVert_display"] = "nicht belastbar (Datenlücke)"
            item["avg_vHor_display"] = "nicht belastbar (Datenlücke)"
            item["avg_angle_display"] = "nicht belastbar (Datenlücke)"
            item["angle_progression_display"] = "nicht belastbar (Datenluecke)"
            out.append(item)
            continue
        item["avg_vVert_display"] = _format_value_with_reference(
            value=row.get("avg_vVert_kmh"),
            reference=ref_row.get("avg_vVert_kmh"),
            higher_is_better=True,
            tolerance=0.5,
            decimals=2,
        )
        item["avg_vHor_display"] = _format_value_with_reference(
            value=row.get("avg_vHor_kmh"),
            reference=ref_row.get("avg_vHor_kmh"),
            higher_is_better=True,
            tolerance=0.5,
            decimals=2,
        )
        item["avg_angle_display"] = _format_value_with_reference(
            value=row.get("avg_angle_deg"),
            reference=ref_row.get("avg_angle_deg"),
            higher_is_better=None,
            tolerance=0.3,
            decimals=2,
        )
        item["angle_progression_display"] = _format_angle_progression(
            start_angle=row.get("angle_start_deg"),
            end_angle=row.get("angle_end_deg"),
            delta_angle=row.get("angle_delta_deg"),
        )
        out.append(item)
    return out


def _phase_rows_for_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    dynamic_rows = _build_normalized_phase_rows(report)
    if dynamic_rows:
        return dynamic_rows
    return list(report.get("phases", []) or [])


def _build_normalized_phase_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    chart = report.get("chart_data", {}) if isinstance(report.get("chart_data"), dict) else {}
    eval_end = effective_eval_window_end_s(
        metrics=report.get("metrics", {}), notes=notes, chart_data=chart
    )
    if eval_end is None or eval_end < 8.0:
        return []

    technical_rows: list[dict[str, Any]] = []
    for spec in TECHNICAL_PHASE_SPECS:
        name = str(spec["name"])
        start_s = float(spec["start_s"])
        end_s = min(float(spec["end_s"]), float(eval_end))
        target_low = float(spec["angle_min"])
        target_high = float(spec["angle_max"])
        target_label = f"{target_low:.0f}-{target_high:.0f} Grad"
        if end_s <= start_s + 0.75:
            display_end_s = max(start_s, end_s)
            technical_rows.append(
                {
                    "name": name,
                    "start_s": round(start_s, 2),
                    "end_s": round(display_end_s, 2),
                    "duration_s": round(max(0.0, display_end_s - start_s), 2),
                    "avg_vVert_kmh": None,
                    "avg_vHor_kmh": None,
                    "avg_angle_deg": None,
                    "angle_start_deg": None,
                    "angle_end_deg": None,
                    "angle_delta_deg": None,
                    "min_angle_deg": None,
                    "max_angle_deg": None,
                    "max_vVert_kmh": None,
                    "vVert_gain_kmh": None,
                    "coverage_ratio": 0.0,
                    "target_angle_label": target_label,
                    "target_status": "nicht belastbar",
                    "comment": "Bewertungsfenster endet vor oder direkt zu Beginn dieser Phase.",
                }
            )
            continue

        coverage = _window_coverage_ratio(
            time_s=chart.get("time_s", []),
            start_s=start_s,
            end_s=end_s,
        )
        avg_vvert = _window_mean(
            time_s=chart.get("time_s", []),
            values=chart.get("vVert_kmh", []),
            start_s=start_s,
            end_s=end_s,
        )
        avg_vhor = _window_mean(
            time_s=chart.get("time_s", []),
            values=chart.get("vHor_kmh", []),
            start_s=start_s,
            end_s=end_s,
        )
        avg_angle = _window_mean(
            time_s=chart.get("time_s", []),
            values=chart.get("angle_deg", []),
            start_s=start_s,
            end_s=end_s,
        )
        min_angle = _window_min(
            time_s=chart.get("time_s", []),
            values=chart.get("angle_deg", []),
            start_s=start_s,
            end_s=end_s,
        )
        max_angle = _window_max(
            time_s=chart.get("time_s", []),
            values=chart.get("angle_deg", []),
            start_s=start_s,
            end_s=end_s,
        )
        angle_start = _series_value_at(
            time_s=chart.get("time_s", []),
            values=chart.get("angle_deg", []),
            target_s=start_s,
        )
        angle_end = _series_value_at(
            time_s=chart.get("time_s", []),
            values=chart.get("angle_deg", []),
            target_s=end_s,
        )
        angle_delta = None if angle_start is None or angle_end is None else float(angle_end - angle_start)
        max_vvert = _window_max(
            time_s=chart.get("time_s", []),
            values=chart.get("vVert_kmh", []),
            start_s=start_s,
            end_s=end_s,
        )
        vvert_gain = _window_delta(
            time_s=chart.get("time_s", []),
            values=chart.get("vVert_kmh", []),
            start_s=start_s,
            end_s=end_s,
        )

        if coverage < 0.75:
            technical_rows.append(
                {
                    "name": name,
                    "start_s": round(start_s, 2),
                    "end_s": round(end_s, 2),
                    "duration_s": round(max(0.0, end_s - start_s), 2),
                    "avg_vVert_kmh": None,
                    "avg_vHor_kmh": None,
                    "avg_angle_deg": None,
                    "angle_start_deg": None,
                    "angle_end_deg": None,
                    "angle_delta_deg": None,
                    "min_angle_deg": None,
                    "max_angle_deg": None,
                    "max_vVert_kmh": None,
                    "vVert_gain_kmh": None,
                    "coverage_ratio": round(coverage, 2),
                    "target_angle_label": target_label,
                    "target_status": "nicht belastbar",
                    "comment": "Nicht genug Daten in dieser Phase.",
                }
            )
            continue

        technical_rows.append(
            {
                "name": name,
                "start_s": round(start_s, 2),
                "end_s": round(end_s, 2),
                "duration_s": round(max(0.0, end_s - start_s), 2),
                "avg_vVert_kmh": None if avg_vvert is None else round(avg_vvert, 2),
                "avg_vHor_kmh": None if avg_vhor is None else round(avg_vhor, 2),
                "avg_angle_deg": None if avg_angle is None else round(avg_angle, 2),
                "angle_start_deg": None if angle_start is None else round(angle_start, 2),
                "angle_end_deg": None if angle_end is None else round(angle_end, 2),
                "angle_delta_deg": None if angle_delta is None else round(angle_delta, 2),
                "min_angle_deg": None if min_angle is None else round(min_angle, 2),
                "max_angle_deg": None if max_angle is None else round(max_angle, 2),
                "max_vVert_kmh": None if max_vvert is None else round(max_vvert, 2),
                "vVert_gain_kmh": None if vvert_gain is None else round(vvert_gain, 2),
                "coverage_ratio": round(coverage, 2),
                "target_angle_label": target_label,
                "target_status": _technical_phase_status(
                    avg_angle=avg_angle,
                    min_angle=min_angle,
                    max_angle=max_angle,
                    target_low=target_low,
                    target_high=target_high,
                ),
                "comment": _technical_phase_comment(
                    name=name,
                    avg_vvert=avg_vvert,
                    avg_vhor=avg_vhor,
                    avg_angle=avg_angle,
                    min_angle=min_angle,
                    max_angle=max_angle,
                    target_low=target_low,
                    target_high=target_high,
                ),
            }
        )
    if technical_rows:
        return technical_rows

    phase_specs = [
        ("Exitphase", 0.00, 0.15),
        ("Aufbauphase", 0.15, 0.45),
        ("Hauptbeschleunigung", 0.45, 0.70),
        ("Hot-Zone", 0.70, 0.90),
        ("Spätphase", 0.90, 1.00),
    ]
    rows: list[dict[str, Any]] = []
    for name, tau0, tau1 in phase_specs:
        start_s = _tau_to_time(tau0=tau0, tau1=tau1, eval_end_s=eval_end)[0]
        end_s = _tau_to_time(tau0=tau0, tau1=tau1, eval_end_s=eval_end)[1]
        if end_s <= start_s + 0.15:
            continue

        coverage = _window_coverage_ratio(
            time_s=chart.get("time_s", []),
            start_s=start_s,
            end_s=end_s,
        )
        avg_vvert = _window_mean(
            time_s=chart.get("time_s", []),
            values=chart.get("vVert_kmh", []),
            start_s=start_s,
            end_s=end_s,
        )
        avg_vhor = _window_mean(
            time_s=chart.get("time_s", []),
            values=chart.get("vHor_kmh", []),
            start_s=start_s,
            end_s=end_s,
        )
        avg_angle = _window_mean(
            time_s=chart.get("time_s", []),
            values=chart.get("angle_deg", []),
            start_s=start_s,
            end_s=end_s,
        )
        max_vvert = _window_max(
            time_s=chart.get("time_s", []),
            values=chart.get("vVert_kmh", []),
            start_s=start_s,
            end_s=end_s,
        )

        if coverage < 0.75:
            rows.append(
                {
                    "name": name,
                    "start_s": round(start_s, 2),
                    "end_s": round(end_s, 2),
                    "duration_s": round(max(0.0, end_s - start_s), 2),
                    "avg_vVert_kmh": None,
                    "avg_vHor_kmh": None,
                    "avg_angle_deg": None,
                    "max_vVert_kmh": None,
                    "coverage_ratio": round(coverage, 2),
                    "comment": "Nicht genug Daten in dieser Phase.",
                }
            )
            continue

        rows.append(
            {
                "name": name,
                "start_s": round(start_s, 2),
                "end_s": round(end_s, 2),
                "duration_s": round(max(0.0, end_s - start_s), 2),
                "avg_vVert_kmh": None if avg_vvert is None else round(avg_vvert, 2),
                "avg_vHor_kmh": None if avg_vhor is None else round(avg_vhor, 2),
                "avg_angle_deg": None if avg_angle is None else round(avg_angle, 2),
                "max_vVert_kmh": None if max_vvert is None else round(max_vvert, 2),
                "coverage_ratio": round(coverage, 2),
                "comment": _phase_comment_light(
                    name,
                    0.0 if avg_vvert is None else float(avg_vvert),
                    0.0 if avg_vhor is None else float(avg_vhor),
                    0.0 if avg_angle is None else float(avg_angle),
                ),
            }
        )
    return rows


def _tau_to_time(*, tau0: float, tau1: float, eval_end_s: float) -> tuple[float, float]:
    end_s = max(0.0, float(eval_end_s))
    start = max(0.0, min(end_s, float(tau0) * end_s))
    end = max(start, min(end_s, float(tau1) * end_s))
    return start, end


def _max_time_s(time_s: list[Any]) -> float | None:
    values = [float(v) for raw in time_s for v in [_to_float(raw)] if v is not None]
    return None if not values else float(max(values))


def _window_coverage_ratio(*, time_s: list[Any], start_s: float, end_s: float) -> float:
    if not time_s or end_s <= start_s:
        return 0.0
    t = [float(v) for raw in time_s for v in [_to_float(raw)] if v is not None]
    if len(t) < 3:
        return 0.0
    t.sort()
    dts = [t[i] - t[i - 1] for i in range(1, len(t)) if (t[i] - t[i - 1]) > 0]
    if not dts:
        return 0.0
    med_dt = float(_percentile(sorted(dts), 0.5))
    if med_dt <= 0:
        return 0.0
    count = sum(1 for value in t if start_s <= value <= end_s)
    expected = ((end_s - start_s) / med_dt) + 1.0
    if expected <= 0:
        return 0.0
    return float(max(0.0, min(1.0, count / expected)))


def _technical_phase_status(
    *,
    avg_angle: float | None,
    min_angle: float | None,
    max_angle: float | None,
    target_low: float,
    target_high: float,
) -> str:
    if avg_angle is None:
        return "unbekannt"
    avg = float(avg_angle)
    low = float(target_low)
    high = float(target_high)
    max_val = None if max_angle is None else float(max_angle)
    min_val = None if min_angle is None else float(min_angle)
    if avg < low - 1.5:
        return "zu flach"
    if avg > high + 1.0:
        return "zu steil"
    if max_val is not None and max_val > high + 2.0:
        return "kurz zu steil"
    if min_val is not None and min_val < low - 3.0 and avg < low + 0.8:
        return "eher flach"
    return "im Zielbereich"


def _technical_phase_comment(
    *,
    name: str,
    avg_vvert: float | None,
    avg_vhor: float | None,
    avg_angle: float | None,
    min_angle: float | None,
    max_angle: float | None,
    target_low: float,
    target_high: float,
) -> str:
    if avg_angle is None:
        return "Phase nicht belastbar."
    status = _technical_phase_status(
        avg_angle=avg_angle,
        min_angle=min_angle,
        max_angle=max_angle,
        target_low=target_low,
        target_high=target_high,
    )
    vhor = None if avg_vhor is None else float(avg_vhor)
    vvert = None if avg_vvert is None else float(avg_vvert)

    if status == "kurz zu steil":
        max_val = None if max_angle is None else float(max_angle)
        target_text = f"{float(target_low):.0f}-{float(target_high):.0f} Grad"
        if max_val is not None and vhor is not None and vhor < 28.0:
            return (
                f"Durchschnitt liegt im Zielbereich; Max Winkel {max_val:.1f} Grad liegt bei knapper "
                f"horizontaler Reserve kurz ueber dem Ziel {target_text}."
            )
        if max_val is not None:
            return (
                f"Durchschnitt liegt im Zielbereich; Max Winkel {max_val:.1f} Grad liegt kurz "
                f"ueber dem Ziel {target_text}."
            )
        return "Durchschnitt liegt im Zielbereich; einzelne Messpunkte liegen kurz ueber dem Zielwinkel."
    if status == "zu steil":
        if vhor is not None and vhor < 28.0:
            return "Zu steil bei knapper horizontaler Reserve; Risiko fuer Nachkorrekturen."
        return "Winkel liegt ueber dem Technikmodell; nur sinnvoll, wenn die Linie ruhig bleibt."
    if status in {"zu flach", "eher flach"}:
        if name == "Max-Speed Fenster" and status == "eher flach":
            return "Durchschnitt liegt im Zielwinkel, aber einzelne Abschnitte fallen darunter; Stabilitaet des 3s-Fensters separat pruefen."
        if name == "Dive-Aufbau":
            return "Aufbau bleibt flach; Druck kommt wahrscheinlich spaeter."
        return "Winkel liegt unter dem Technikmodell; Speed-Aufbau kann spaeter fehlen."
    if name == "Max-Speed Fenster":
        if vvert is not None and vhor is not None and vvert >= 390.0 and vhor >= 25.0:
            return "Max-Speed-Fenster ist technisch nutzbar: hoch und noch mit horizontaler Reserve."
        return "Max-Speed-Fenster liegt im Zielwinkel; Stabilitaet des 3s-Fensters separat pruefen."
    return "Phase liegt im technischen Zielbereich."


def _phase_comment_light(name: str, avg_vvert: float, avg_vhor: float, avg_angle: float) -> str:
    if name == "Exitphase":
        if avg_vvert < 190.0:
            return "Der Start baut noch wenig vertikalen Speed auf."
        return "Der Start ist sauber und kontrolliert."
    if name == "Aufbauphase":
        if avg_angle < 70.0:
            return "Im Aufbau bleibt der Winkel eher flach."
        return "Der Aufbau geht kontrolliert in den Dive."
    if name == "Hauptbeschleunigung":
        if avg_angle > 87.0 and avg_vhor < 28.0:
            return "Sehr steil bei knapper horizontaler Reserve."
        return "Hauptaufbau stabil und nutzbar."
    if name == "Hot-Zone":
        if avg_vhor < 24.0:
            return "In der Hot-Zone ist die horizontale Reserve knapp."
        return "Hot-Zone ist stabil nutzbar."
    if avg_vvert > 380.0:
        return "Speed bleibt hoch, Ausstieg sauber timen."
    return "Spätphase kontrolliert."


def _phase_gap_map(report: dict[str, Any]) -> dict[str, bool]:
    phases = _phase_rows_for_report(report)
    chart = report.get("chart_data", {}) if isinstance(report.get("chart_data"), dict) else {}
    time_s = chart.get("time_s", []) or []
    if len(time_s) < 3:
        return {str(row.get("name") or ""): False for row in phases}

    t: list[float] = []
    for raw in time_s:
        value = _to_float(raw)
        if value is None:
            continue
        t.append(float(value))
    if len(t) < 3:
        return {str(row.get("name") or ""): False for row in phases}

    dts: list[float] = []
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt > 0:
            dts.append(dt)
    if not dts:
        return {str(row.get("name") or ""): False for row in phases}

    med_dt = float(_percentile(sorted(dts), 0.5))
    gap_threshold = max(0.6, med_dt * 2.5)

    gap_segments: list[tuple[float, float]] = []
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt > gap_threshold:
            gap_segments.append((t[i - 1], t[i]))

    out: dict[str, bool] = {}
    for row in phases:
        name = str(row.get("name") or "")
        start_s = _to_float(row.get("start_s"))
        end_s = _to_float(row.get("end_s"))
        if start_s is None or end_s is None or not gap_segments:
            out[name] = False
            continue
        has_gap = False
        for g0, g1 in gap_segments:
            mid = (g0 + g1) * 0.5
            if float(start_s) <= mid <= float(end_s):
                has_gap = True
                break
        out[name] = has_gap
    return out


def _annotate_scorecard_with_reference(
    *,
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    reference_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if reference_report is None:
        return [dict(row, benchmark_lines=[]) for row in rows]

    current = _scorecard_metric_snapshot(report)
    reference = _scorecard_metric_snapshot(reference_report)
    current_phases = {
        str(row.get("name") or "").strip(): row
        for row in _phase_rows_for_report(report)
        if str(row.get("name") or "").strip()
    }
    reference_phases = {
        str(row.get("name") or "").strip(): row
        for row in _phase_rows_for_report(reference_report)
        if str(row.get("name") or "").strip()
    }
    out: list[dict[str, Any]] = []

    for row in rows:
        name = str(row.get("name") or "").strip()
        benchmark_lines: list[str] = []
        phase_row = current_phases.get(name)
        ref_phase_row = reference_phases.get(name)
        if phase_row is not None and ref_phase_row is not None:
            line = _metric_compare_line(
                label="Avg vVert",
                current=phase_row.get("avg_vVert_kmh"),
                reference=ref_phase_row.get("avg_vVert_kmh"),
                unit="km/h",
                decimals=1,
                higher_is_better=None,
                tolerance=0.5,
            )
            if line:
                benchmark_lines.append(line)
            line = _metric_compare_line(
                label="Avg vHor",
                current=phase_row.get("avg_vHor_kmh"),
                reference=ref_phase_row.get("avg_vHor_kmh"),
                unit="km/h",
                decimals=1,
                higher_is_better=None,
                tolerance=0.5,
            )
            if line:
                benchmark_lines.append(line)
            line = _metric_compare_line(
                label="Avg Winkel",
                current=phase_row.get("avg_angle_deg"),
                reference=ref_phase_row.get("avg_angle_deg"),
                unit="Grad",
                decimals=1,
                higher_is_better=None,
                tolerance=0.3,
            )
            if line:
                benchmark_lines.append(line)
        elif name == "Exit":
            line = _metric_compare_line(
                label="vVert +10s",
                current=current.get("v10"),
                reference=reference.get("v10"),
                unit="km/h",
                decimals=1,
                higher_is_better=True,
                tolerance=0.5,
            )
            if line:
                benchmark_lines.append(line)
            line = _metric_compare_line(
                label="Druck-Mitnahme",
                current=current.get("carry_ratio"),
                reference=reference.get("carry_ratio"),
                unit="",
                decimals=2,
                higher_is_better=True,
                tolerance=0.03,
            )
            if line:
                benchmark_lines.append(line)
        elif name == "Aufbau 10-20s":
            line = _metric_compare_line(
                label="Zuwachs +10 bis +20s",
                current=current.get("gain_10_20"),
                reference=reference.get("gain_10_20"),
                unit="km/h",
                decimals=1,
                higher_is_better=True,
                tolerance=0.7,
            )
            if line:
                benchmark_lines.append(line)
            line = _metric_compare_line(
                label="Winkel bei +20s",
                current=current.get("angle_20"),
                reference=reference.get("angle_20"),
                unit="Grad",
                decimals=1,
                higher_is_better=None,
                tolerance=0.3,
            )
            if line:
                benchmark_lines.append(line)
        elif name == "Hot-Zone":
            line = _metric_compare_line(
                label="Dauer >400 km/h",
                current=current.get("dur_400"),
                reference=reference.get("dur_400"),
                unit="s",
                decimals=1,
                higher_is_better=True,
                tolerance=0.2,
            )
            if line:
                benchmark_lines.append(line)
            line = _metric_compare_line(
                label="vHor-Min Hot-Zone",
                current=current.get("vhor_min_20_25"),
                reference=reference.get("vhor_min_20_25"),
                unit="km/h",
                decimals=1,
                higher_is_better=True,
                tolerance=0.5,
            )
            if line:
                benchmark_lines.append(line)
            line = _metric_compare_line(
                label="vVert-Zuwachs Hot-Zone",
                current=current.get("vvert_gain_20_25"),
                reference=reference.get("vvert_gain_20_25"),
                unit="km/h",
                decimals=1,
                higher_is_better=True,
                tolerance=0.7,
            )
            if line:
                benchmark_lines.append(line)
            line = _metric_compare_line(
                label="Korrekturen Hot-Zone",
                current=current.get("angle_turns_20_25"),
                reference=reference.get("angle_turns_20_25"),
                unit="",
                decimals=0,
                higher_is_better=False,
                tolerance=0.5,
            )
            if line:
                benchmark_lines.append(line)
        elif name == "Stabilität / Kipp-Risiko":
            line = _metric_compare_line(
                label="vHor-Min Hot-Zone",
                current=current.get("vhor_min_20_25"),
                reference=reference.get("vhor_min_20_25"),
                unit="km/h",
                decimals=1,
                higher_is_better=True,
                tolerance=0.5,
            )
            if line:
                benchmark_lines.append(line)
            line = _metric_compare_line(
                label="Korrekturen Hot-Zone",
                current=current.get("angle_turns_20_25"),
                reference=reference.get("angle_turns_20_25"),
                unit="",
                decimals=0,
                higher_is_better=False,
                tolerance=0.5,
            )
            if line:
                benchmark_lines.append(line)

        out.append(dict(row, benchmark_lines=benchmark_lines))

    return out


def _scorecard_metric_snapshot(report: dict[str, Any]) -> dict[str, float | None]:
    fixpoints = report.get("fixpoints", []) or []
    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    chart = report.get("chart_data", {}) if isinstance(report.get("chart_data"), dict) else {}

    fp10 = _fixpoint_at(fixpoints, 10.0)
    fp15 = _fixpoint_at(fixpoints, 15.0)
    fp20 = _fixpoint_at(fixpoints, 20.0)
    v10 = _to_float(fp10.get("vVert_kmh")) if fp10 else None
    angle_10 = _to_float(fp10.get("angle_deg")) if fp10 else None
    angle_15 = _to_float(fp15.get("angle_deg")) if fp15 else None
    v20_fixed = _to_float(fp20.get("vVert_kmh")) if fp20 else None
    angle_20_fixed = _to_float(fp20.get("angle_deg")) if fp20 else None
    gain_10_20_fixed = None if v10 is None or v20_fixed is None else float(v20_fixed - v10)

    exit_profile = notes.get("exit_profile", {}) if isinstance(notes.get("exit_profile"), dict) else {}
    carry_ratio = _to_float(exit_profile.get("carry_ratio"))
    if carry_ratio is None:
        carry_ratio = _estimate_carry_ratio(chart)

    eval_end = effective_eval_window_end_s(
        metrics=report.get("metrics", {}), notes=notes, chart_data=chart
    )
    if eval_end is None:
        curve_end = _to_float(notes.get("curve_window_end_s"))
        eval_end = 25.0 if curve_end is None else float(curve_end)

    build_start_s = 3.0
    build_end_s = min(15.0, float(eval_end))
    if build_end_s <= build_start_s + 0.75:
        build_start_s, build_end_s = _tau_to_time(tau0=0.15, tau1=0.70, eval_end_s=eval_end)

    build_angle_start_s = 8.0
    build_angle_end_s = min(22.0, float(eval_end))
    if build_angle_end_s <= build_angle_start_s + 0.75:
        build_angle_start_s, build_angle_end_s = _tau_to_time(tau0=0.45, tau1=0.70, eval_end_s=eval_end)

    hot_start_s = 20.0
    hot_end_s = min(28.0, float(eval_end))
    if hot_end_s <= hot_start_s + 0.75:
        hot_start_s, hot_end_s = _tau_to_time(tau0=0.70, tau1=0.90, eval_end_s=eval_end)

    build_coverage = _window_coverage_ratio(
        time_s=chart.get("time_s", []),
        start_s=build_start_s,
        end_s=build_end_s,
    )
    hot_coverage = _window_coverage_ratio(
        time_s=chart.get("time_s", []),
        start_s=hot_start_s,
        end_s=hot_end_s,
    )

    gain_build_norm = _window_delta(
        time_s=chart.get("time_s", []),
        values=chart.get("vVert_kmh", []),
        start_s=build_start_s,
        end_s=build_end_s,
    )
    angle_build_norm = _window_mean(
        time_s=chart.get("time_s", []),
        values=chart.get("angle_deg", []),
        start_s=build_angle_start_s,
        end_s=build_angle_end_s,
    )

    dur_400_hot_norm = _duration_above_threshold(
        time_s=chart.get("time_s", []),
        values=chart.get("vVert_kmh", []),
        threshold=400.0,
        start_s=hot_start_s,
        end_s=hot_end_s,
    )
    vhor_min_hot_norm = _window_min(
        time_s=chart.get("time_s", []),
        values=chart.get("vHor_kmh", []),
        start_s=hot_start_s,
        end_s=hot_end_s,
    )
    vvert_gain_hot_norm = _window_delta(
        time_s=chart.get("time_s", []),
        values=chart.get("vVert_kmh", []),
        start_s=hot_start_s,
        end_s=hot_end_s,
    )
    angle_turns_hot_norm = _window_turn_count(
        time_s=chart.get("time_s", []),
        values=chart.get("angle_deg", []),
        start_s=hot_start_s,
        end_s=hot_end_s,
        eps=0.35,
    )

    gain_10_20 = gain_10_20_fixed
    angle_20 = angle_20_fixed
    if build_coverage >= 0.75:
        gain_10_20 = gain_build_norm if gain_build_norm is not None else gain_10_20
        angle_20 = angle_build_norm if angle_build_norm is not None else angle_20

    dur_400 = dur_400_hot_norm if hot_coverage >= 0.75 else None
    vhor_min_20_25 = vhor_min_hot_norm if hot_coverage >= 0.75 else None
    vvert_gain_20_25 = vvert_gain_hot_norm if hot_coverage >= 0.75 else None
    angle_turns_20_25 = angle_turns_hot_norm if hot_coverage >= 0.75 else None

    return {
        "v10": v10,
        "angle_10": angle_10,
        "angle_15": angle_15,
        "carry_ratio": carry_ratio,
        "gain_10_20": gain_10_20,
        "angle_20": angle_20,
        "dur_400": dur_400,
        "vhor_min_20_25": vhor_min_20_25,
        "vvert_gain_20_25": vvert_gain_20_25,
        "angle_turns_20_25": None if angle_turns_20_25 is None else float(angle_turns_20_25),
        "eval_end_s": float(eval_end),
        "build_coverage": float(build_coverage),
        "hot_coverage": float(hot_coverage),
        "build_start_s": float(build_start_s),
        "build_end_s": float(build_end_s),
        "hot_start_s": float(hot_start_s),
        "hot_end_s": float(hot_end_s),
    }


def _get_marco_top15_profile(*, limit: int = 15) -> dict[str, Any] | None:
    rows = list_jumps_for_jumper("Marco Hepp")
    if not rows:
        return None
    signature = _marco_profile_signature(rows)
    cached = _MARCO_PROFILE_CACHE.get(int(limit))
    if cached is not None and cached[0] == signature:
        return cached[1]

    profile = _build_marco_top15_profile(limit=limit, rows=rows)
    _MARCO_PROFILE_CACHE[int(limit)] = (signature, profile)
    return profile


def _marco_profile_signature(rows: list[dict[str, Any]]) -> str:
    # DB-side change detection: if any relevant row changes, signature changes and cache refreshes.
    parts: list[str] = []
    for row in rows:
        parts.append(
            "|".join(
                [
                    str(row.get("jump_id") or ""),
                    str(row.get("t0_utc") or ""),
                    str(row.get("best_3s_vVert_kmh") or ""),
                    str(row.get("quality_flags") or ""),
                ]
            )
        )
    parts.sort()
    payload = "||".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _jumper_summary_signature(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        parts.append(
            "|".join(
                [
                    str(row.get("jump_id") or ""),
                    str(row.get("file_name") or ""),
                    str(row.get("jump_context") or ""),
                    str(row.get("t0_utc") or ""),
                    str(row.get("quality_score") or ""),
                    str(row.get("quality_flags") or ""),
                    str(row.get("sample_rate_hz") or ""),
                    str(row.get("is_valid_altitude") or ""),
                    str(row.get("best_3s_vVert_kmh") or ""),
                    str(row.get("rule_based_3s_score") or ""),
                    str(row.get("rule_score_status") or ""),
                    str(row.get("analysis_version") or ""),
                ]
            )
        )
    parts.sort()
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()


def _build_marco_top15_profile(
    *,
    limit: int = 15,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if rows is None:
        rows = list_jumps_for_jumper("Marco Hepp")
    if not rows:
        return None

    ordered_rows = sorted(
        rows,
        key=lambda item: float(_to_float(item.get("rule_based_3s_score")) or float("-inf")),
        reverse=True,
    )

    candidates: list[dict[str, Any]] = []
    target = max(1, int(limit))
    for row in ordered_rows:
        jump_id = str(row.get("jump_id") or "")
        if not jump_id:
            continue
        report = get_jump_report(jump_id)
        if report is None:
            continue
        if not _is_clean_reference_jump(row, report):
            continue
        snap = _scorecard_metric_snapshot(report)
        best_3s = _to_float(report.get("metrics", {}).get("rule_based_3s_score"))
        if best_3s is None:
            continue
        snap["best_3s"] = best_3s
        candidates.append(snap)
        if len(candidates) >= target:
            break

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda item: float(item.get("best_3s") or 0.0), reverse=True)
    top = candidates[:target]
    if not top:
        return None

    profile = {
        "count": len(top),
        "v10_ref": _max_metric(top, "v10"),
        "carry_ref": _max_metric(top, "carry_ratio"),
        "gain_10_20_ref": _max_metric(top, "gain_10_20"),
        "angle_20_low": _quantile_metric(top, "angle_20", 0.25),
        "angle_20_high": _quantile_metric(top, "angle_20", 0.75),
        "dur_400_ref": _max_metric(top, "dur_400"),
        "vhor_min_ref": _max_metric(top, "vhor_min_20_25"),
        "vvert_gain_20_25_ref": _max_metric(top, "vvert_gain_20_25"),
        "turns_ref": _min_metric(top, "angle_turns_20_25"),
    }

    return profile


def _compute_marco_percent_scores(
    *,
    snapshot: dict[str, float | None],
    marco_profile: dict[str, Any],
    exit_unsteady: bool,
    risk_score_raw: float | None,
) -> dict[str, int]:
    # Exit
    exit_parts: list[float] = []
    exit_parts.append(_ratio_higher(snapshot.get("v10"), marco_profile.get("v10_ref")))
    exit_parts.append(_ratio_higher(snapshot.get("carry_ratio"), marco_profile.get("carry_ref")))
    exit_score = _combine_scores(exit_parts)
    if exit_unsteady:
        exit_score = max(0, exit_score - 8)

    # Build
    build_parts: list[float] = []
    build_parts.append(_ratio_higher(snapshot.get("gain_10_20"), marco_profile.get("gain_10_20_ref")))
    build_parts.append(
        _score_angle_band(
            value=snapshot.get("angle_20"),
            band_low=marco_profile.get("angle_20_low"),
            band_high=marco_profile.get("angle_20_high"),
        )
    )
    build_score = _combine_scores(build_parts)

    # Hot-Zone
    hot_parts: list[float] = []
    hot_parts.append(_ratio_higher(snapshot.get("dur_400"), marco_profile.get("dur_400_ref")))
    hot_parts.append(_ratio_higher(snapshot.get("vhor_min_20_25"), marco_profile.get("vhor_min_ref")))
    hot_parts.append(_ratio_higher(snapshot.get("vvert_gain_20_25"), marco_profile.get("vvert_gain_20_25_ref")))
    hot_parts.append(_score_lower_is_better(snapshot.get("angle_turns_20_25"), marco_profile.get("turns_ref"), slack=4.0))
    hot_score = _combine_scores(hot_parts)

    # Stability
    stability_parts: list[float] = []
    stability_parts.append(_ratio_higher(snapshot.get("vhor_min_20_25"), marco_profile.get("vhor_min_ref")))
    stability_parts.append(_score_lower_is_better(snapshot.get("angle_turns_20_25"), marco_profile.get("turns_ref"), slack=5.0))
    stability_score = _combine_scores(stability_parts)
    if risk_score_raw is not None:
        risk_factor = _clamp_score(100 - float(risk_score_raw))
        stability_score = _clamp_score(stability_score * 0.6 + risk_factor * 0.4)

    return {
        "exit": _clamp_score(exit_score),
        "build": _clamp_score(build_score),
        "hot": _clamp_score(hot_score),
        "stability": _clamp_score(stability_score),
    }


def _add_reference_score_context(
    *,
    phase_name: str,
    score: int,
    reason_lines: list[str],
    snapshot: dict[str, float | None],
    marco_profile: dict[str, Any],
) -> list[str]:
    if int(score) >= 70:
        return reason_lines

    lines: list[str] = []
    if phase_name == "Exit":
        v10 = _to_float(snapshot.get("v10"))
        v10_ref = _to_float(marco_profile.get("v10_ref"))
        carry = _to_float(snapshot.get("carry_ratio"))
        carry_ref = _to_float(marco_profile.get("carry_ref"))
        if v10 is not None and v10_ref is not None:
            lines.append(f"Referenzscore: vVert +10s {_fmt1(v10)} km/h statt Referenz {_fmt1(v10_ref)} km/h.")
        if carry is not None and carry_ref is not None:
            lines.append(f"Referenzscore: Druck-Mitnahme {_fmt2(carry)} statt Referenz {_fmt2(carry_ref)}.")
    elif phase_name in {"Aufbau 10-20s", "Dive-Aufbau", "Hauptbeschleunigung"}:
        gain = _to_float(snapshot.get("gain_10_20"))
        gain_ref = _to_float(marco_profile.get("gain_10_20_ref"))
        angle = _to_float(snapshot.get("angle_20"))
        angle_low = _to_float(marco_profile.get("angle_20_low"))
        angle_high = _to_float(marco_profile.get("angle_20_high"))
        if gain is not None and gain_ref is not None:
            lines.append(
                f"Referenzscore: Zuwachs im Aufbau {_fmt1(gain)} km/h statt Referenz {_fmt1(gain_ref)} km/h."
            )
        if angle is not None and angle_low is not None and angle_high is not None:
            if angle < angle_low:
                lines.append(
                    "Referenzscore: Aufbauwinkel "
                    f"{_fmt1(angle)} Grad liegt unter dem Referenzband {_fmt1(angle_low)} bis {_fmt1(angle_high)} Grad."
                )
            elif angle > angle_high:
                lines.append(
                    "Referenzscore: Aufbauwinkel "
                    f"{_fmt1(angle)} Grad liegt über dem Referenzband {_fmt1(angle_low)} bis {_fmt1(angle_high)} Grad."
                )
    elif phase_name == "Hot-Zone":
        dur_400 = _to_float(snapshot.get("dur_400"))
        dur_ref = _to_float(marco_profile.get("dur_400_ref"))
        vhor_min = _to_float(snapshot.get("vhor_min_20_25"))
        vhor_ref = _to_float(marco_profile.get("vhor_min_ref"))
        gain_hot = _to_float(snapshot.get("vvert_gain_20_25"))
        gain_hot_ref = _to_float(marco_profile.get("vvert_gain_20_25_ref"))
        if dur_400 is not None and dur_ref is not None:
            lines.append(f"Referenzscore: Dauer über 400 km/h {_fmt1(dur_400)}s statt Referenz {_fmt1(dur_ref)}s.")
        if vhor_min is not None and vhor_ref is not None:
            lines.append(f"Referenzscore: vHor-Min Hot-Zone {_fmt1(vhor_min)} km/h statt Referenz {_fmt1(vhor_ref)} km/h.")
        if gain_hot is not None and gain_hot_ref is not None:
            lines.append(
                f"Referenzscore: zusätzlicher Speed Hot-Zone {_fmt1(gain_hot)} km/h statt Referenz {_fmt1(gain_hot_ref)} km/h."
            )
    elif phase_name == "Stabilität / Kipp-Risiko":
        vhor_min = _to_float(snapshot.get("vhor_min_20_25"))
        vhor_ref = _to_float(marco_profile.get("vhor_min_ref"))
        turns = _to_float(snapshot.get("angle_turns_20_25"))
        turns_ref = _to_float(marco_profile.get("turns_ref"))
        if vhor_min is not None and vhor_ref is not None:
            lines.append(f"Referenzscore: vHor-Min Hot-Zone {_fmt1(vhor_min)} km/h statt Referenz {_fmt1(vhor_ref)} km/h.")
        if turns is not None and turns_ref is not None:
            lines.append(f"Referenzscore: Korrekturen Hot-Zone {_fmt1(turns)} statt Referenz {_fmt1(turns_ref)}.")

    if not lines:
        return reason_lines
    return lines + reason_lines


def _combine_scores(values: list[float | None]) -> int:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return 0
    return _clamp_score(sum(valid) / len(valid))


def _ratio_higher(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference <= 0:
        return None
    pct = (float(value) / float(reference)) * 100.0
    return float(max(0.0, min(110.0, pct)))


def _score_lower_is_better(value: float | None, reference: float | None, *, slack: float) -> float | None:
    if value is None:
        return None
    ref = 0.0 if reference is None else float(reference)
    val = float(value)
    if val <= ref:
        return 100.0
    if slack <= 0:
        return 0.0
    penalty = ((val - ref) / float(slack)) * 100.0
    return float(max(0.0, 100.0 - penalty))


def _score_angle_band(value: float | None, *, band_low: float | None, band_high: float | None) -> float | None:
    if value is None:
        return None
    val = float(value)
    if band_low is None or band_high is None:
        return 100.0
    low = float(min(band_low, band_high))
    high = float(max(band_low, band_high))
    if low <= val <= high:
        return 100.0
    if val < low:
        diff = low - val
    else:
        diff = val - high
    # around 3 deg outside band -> major penalty
    penalty = min(100.0, (diff / 3.0) * 100.0)
    return float(max(0.0, 100.0 - penalty))


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = _to_float(row.get(key))
        if value is None:
            continue
        out.append(float(value))
    return out


def _max_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = _metric_values(rows, key)
    return None if not vals else float(max(vals))


def _min_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = _metric_values(rows, key)
    return None if not vals else float(min(vals))


def _quantile_metric(rows: list[dict[str, Any]], key: str, q: float) -> float | None:
    vals = sorted(_metric_values(rows, key))
    if not vals:
        return None
    return float(_percentile(vals, q))


def _format_value_with_reference(
    *,
    value: Any,
    reference: Any,
    higher_is_better: bool | None,
    tolerance: float,
    decimals: int,
) -> str:
    current = _to_float(value)
    if current is None:
        return "-"
    current_text = f"{current:.{decimals}f}"
    ref = _to_float(reference)
    if ref is None:
        return current_text
    ref_text = f"{ref:.{decimals}f}"
    delta = current - ref
    if higher_is_better is None:
        return f"{current_text} (Ref {ref_text} | Delta {delta:+.{decimals}f})"
    if _is_near_best(
        current=current,
        reference=ref,
        higher_is_better=higher_is_better,
        tolerance=tolerance,
    ):
        return f"{current_text} (Ref {ref_text} | bestes Ergebnis bisher)"
    return f"{current_text} (Ref {ref_text} | {delta:+.{decimals}f})"


def _format_angle_progression(*, start_angle: Any, end_angle: Any, delta_angle: Any) -> str:
    start = _to_float(start_angle)
    end = _to_float(end_angle)
    delta = _to_float(delta_angle)
    if start is None or end is None:
        return "-"
    if delta is None:
        delta = end - start
    return f"{start:.1f} -> {end:.1f} Grad ({delta:+.1f})"


def _metric_compare_line(
    *,
    label: str,
    current: Any,
    reference: Any,
    unit: str,
    decimals: int,
    higher_is_better: bool | None,
    tolerance: float,
) -> str | None:
    cur = _to_float(current)
    ref = _to_float(reference)
    if cur is None or ref is None:
        return None

    unit_text = f" {unit}" if unit else ""
    cur_text = f"{cur:.{decimals}f}"
    ref_text = f"{ref:.{decimals}f}"
    delta = cur - ref
    if higher_is_better is None:
        return (
            f"{label}: {cur_text}{unit_text} "
            f"(Ref {ref_text}{unit_text} | Delta {delta:+.{decimals}f}{unit_text})"
        )
    if _is_near_best(
        current=cur,
        reference=ref,
        higher_is_better=higher_is_better,
        tolerance=tolerance,
    ):
        return f"{label}: {cur_text}{unit_text} (Ref {ref_text}{unit_text} | bestes Ergebnis bisher)"
    return f"{label}: {cur_text}{unit_text} (Ref {ref_text}{unit_text} | {delta:+.{decimals}f}{unit_text})"


def _is_near_best(
    *,
    current: float,
    reference: float,
    higher_is_better: bool | None,
    tolerance: float,
) -> bool:
    if higher_is_better is True:
        return current >= (reference - tolerance)
    if higher_is_better is False:
        return current <= (reference + tolerance)
    return abs(current - reference) <= tolerance


def _build_jump_brief_summary(
    *,
    report: dict[str, Any],
    review: dict[str, list[str]],
    scorecard_rows: list[dict[str, Any]],
    best_reference: dict[str, Any] | None,
    top_reference_jumps: list[dict[str, Any]],
    feedback_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    notes = report.get("notes", {})
    best_3s_kmh = _to_float(metrics.get("best_3s_vVert_kmh"))

    basis_lines: list[str] = []
    if best_reference:
        basis_lines.append(
            "Vergleichsbasis: bester gespeicherter Sprung dieses Springers "
            f"({best_reference.get('file_name')}, {best_reference.get('t0_utc')})."
        )
    if top_reference_jumps and best_3s_kmh is not None and best_3s_kmh >= 500.0:
        basis_lines.append("Zusatz-Benchmark: Top-5 schnellste plausible Sprünge aller Springer.")

    best_3s_start = _to_float(metrics.get("best_3s_start_s"))
    best_3s_end = _to_float(metrics.get("best_3s_end_s"))
    curve_start = _to_float(notes.get("curve_window_start_s"))
    curve_end = _to_float(notes.get("curve_window_end_s"))
    blocked = bool(notes.get("analysis_blocked"))
    blocked_reason = str(notes.get("analysis_block_reason") or "").strip()

    key_facts: list[str] = []
    if best_3s_kmh is not None and best_3s_start is not None and best_3s_end is not None:
        key_facts.append(
            f"Bestes 3s-Fenster: +{best_3s_start:.1f}s bis +{best_3s_end:.1f}s mit {best_3s_kmh:.1f} km/h."
        )
    if curve_start is not None and curve_end is not None:
        key_facts.append(f"Ausgewerteter Bereich: +{curve_start:.1f}s bis +{curve_end:.1f}s.")

    if blocked:
        summary = blocked_reason or "Sprungdaten nicht korrekt. Der Sprung endet zu früh."
        return {
            "summary": summary,
            "basis_lines": basis_lines,
            "key_facts": key_facts,
            "main_issues": [summary],
            "strengths": ["Keine belastbare Technikbewertung möglich."],
            "actions": [
                "Sprung mit korrekter Absprungerkennung neu einlesen.",
                "Wenn der Track wirklich so kurz ist, nicht für Technikvergleich nutzen.",
            ],
        }

    weak_rows_all = [row for row in scorecard_rows if int(row.get("score", 0)) < 70]
    primary_diagnosis = (
        review.get("primary_diagnosis", {})
        if isinstance(review.get("primary_diagnosis"), dict)
        else {}
    )
    has_primary_diagnosis = bool(primary_diagnosis.get("available"))
    weak_rows = sorted(
        weak_rows_all,
        key=lambda row: int(row.get("score", 0)),
    )
    weak_rows_timeline = sorted(
        weak_rows_all,
        key=lambda row: (
            _scorecard_phase_rank(str(row.get("name", ""))),
            int(row.get("score", 0)),
        ),
    )
    strong_rows = sorted(
        [
            row
            for row in scorecard_rows
            if int(row.get("score", 0)) >= 70 and not _scorecard_row_has_angle_issue(row)
        ],
        key=lambda row: int(row.get("score", 0)),
        reverse=True,
    )

    if has_primary_diagnosis:
        title = str(primary_diagnosis.get("title") or "Hauptdiagnose").strip()
        speed_text = f" Top-Speed: {best_3s_kmh:.1f} km/h." if best_3s_kmh is not None else ""
        summary = f"Hauptdiagnose: {title}.{speed_text}".strip()
    elif not weak_rows:
        summary = (
            f"Sehr stabiler Sprung mit wenigen Schwaechen. "
            f"Top-Speed: {best_3s_kmh:.1f} km/h."
            if best_3s_kmh is not None
            else "Sehr stabiler Sprung mit wenigen Schwaechen."
        )
    else:
        focus_names = ", ".join(str(row.get("name", "")).strip() for row in weak_rows[:2] if row.get("name"))
        speed_text = f" Top-Speed: {best_3s_kmh:.1f} km/h." if best_3s_kmh is not None else ""
        summary = f"Die größten Baustellen liegen bei {focus_names}.{speed_text}".strip()

    review_not_good = _sort_texts_by_timeline(review.get("not_good", [])[:4])
    main_issue_candidates = review_not_good + [
        f"{row.get('name')}: {_primary_issue_line_from_row(row)}"
        for row in weak_rows_timeline[:3]
        if _primary_issue_line_from_row(row)
    ]
    main_issues = _compact_main_issues(main_issue_candidates, max_items=4)
    if has_primary_diagnosis:
        primary_issue = str(
            primary_diagnosis.get("main_issue")
            or primary_diagnosis.get("summary")
            or primary_diagnosis.get("title")
            or ""
        ).strip()
        if primary_issue:
            main_issues = _merge_priority_texts(primary_issue, main_issues, max_items=4)
    if not main_issues:
        main_issues = ["Keine klaren Hauptprobleme in den Hauptdaten gefunden."]
    feedback_hint = _feedback_context_line(feedback_context, key="coaching_hint")
    if feedback_hint:
        main_issues = _unique_texts_semantic(main_issues + [f"Feedback-Abgleich: {feedback_hint}"])[:4]
    if not has_primary_diagnosis and weak_rows:
        issue_focus_names = _focus_names_from_issue_texts(main_issues)
        if issue_focus_names:
            focus_names = ", ".join(issue_focus_names[:2])
            speed_text = f" Top-Speed: {best_3s_kmh:.1f} km/h." if best_3s_kmh is not None else ""
            summary = f"Die größten Baustellen liegen bei {focus_names}.{speed_text}".strip()

    strengths = _unique_texts_semantic(
        [
            f"{row.get('name')}: {row.get('reason')}"
            for row in strong_rows[:2]
            if row.get("reason")
        ]
        + review.get("good", [])[:3]
    )[:4]
    if not strengths:
        strengths = ["Der Sprung ist insgesamt verwertbar und stabil genug für Coaching."]

    cleaned_actions = [_strip_priority_prefix(item) for item in review.get("improve", []) if item]
    cleaned_actions = _sort_texts_by_timeline(_unique_texts(cleaned_actions))
    actions = _build_issue_aligned_actions(
        main_issues=main_issues,
        actions=cleaned_actions,
        max_items=5,
    )
    actions = _filter_conflicting_build_pressure_actions(
        main_issues=main_issues,
        actions=actions,
    )
    if has_primary_diagnosis:
        primary_action = str(
            primary_diagnosis.get("next_focus")
            or primary_diagnosis.get("action")
            or ""
        ).strip()
        if primary_action:
            actions = _merge_priority_texts(primary_action, actions, max_items=4)
        actions = _filter_actions_for_primary_diagnosis(primary_diagnosis, actions, max_items=3)
    feedback_focus = _feedback_context_line(feedback_context, key="next_focus_hint")
    if feedback_focus:
        if actions:
            actions = _unique_texts([actions[0], feedback_focus] + actions[1:])[:4]
        else:
            actions = [feedback_focus]
    if not actions:
        actions = ["Ablauf stabil wiederholen und nur kleine Korrekturen setzen."]
    actions = _harmonize_action_texts(
        actions,
        max_items=5,
        preserve_first=has_primary_diagnosis,
    )
    return {
        "summary": summary,
        "basis_lines": basis_lines,
        "key_facts": key_facts,
        "primary_diagnosis": primary_diagnosis if has_primary_diagnosis else {"available": False},
        "main_issues": main_issues,
        "strengths": strengths,
        "actions": actions,
    }


def _build_jump_brief_simple(
    *,
    report: dict[str, Any],
    jump_brief: dict[str, Any],
    scorecard_rows: list[dict[str, Any]],
    feedback_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    blocked = bool(notes.get("analysis_blocked"))
    blocked_reason = str(notes.get("analysis_block_reason") or "").strip()

    if blocked:
        summary = blocked_reason or "Sprungdaten nicht korrekt. Der Sprung endet zu früh."
        summary_ui = _render_simple_glossary(summary)
        return {
            "summary": summary,
            "summary_ui": summary_ui,
            "basis_line": "",
            "basis_line_ui": "",
            "main_issues": [summary],
            "main_issues_ui": [summary_ui],
            "strengths": ["Keine belastbare Bewertung möglich."],
            "strengths_ui": ["Keine belastbare Bewertung möglich."],
            "actions": [
                "Sprungdatei neu einlesen und Absprungzeitpunkt pruefen.",
                "Diesen Sprung nicht für den Technikvergleich nutzen.",
            ],
            "actions_ui": [
                _render_simple_glossary("Sprungdatei neu einlesen und Absprungzeitpunkt pruefen."),
                _render_simple_glossary("Diesen Sprung nicht für den Technikvergleich nutzen."),
            ],
        }

    primary_diagnosis = (
        jump_brief.get("primary_diagnosis", {})
        if isinstance(jump_brief.get("primary_diagnosis"), dict)
        else {}
    )
    if primary_diagnosis.get("available"):
        summary = str(primary_diagnosis.get("title") or "Ein Hauptproblem ist klar sichtbar.").strip()
        main_issue = str(primary_diagnosis.get("summary") or primary_diagnosis.get("main_issue") or "").strip()
        next_focus = str(primary_diagnosis.get("next_focus") or primary_diagnosis.get("action") or "").strip()
        feedback_hint = _feedback_context_line(feedback_context, key="coaching_hint")
        feedback_focus = _feedback_context_line(feedback_context, key="next_focus_hint")
        if not main_issue:
            main_issue = "Der Sprung verliert in der schnellen Phase Stabilitaet."
        if feedback_hint:
            main_issue = f"{main_issue} {feedback_hint}"
        if not next_focus:
            next_focus = "Im naechsten Sprung nur einen Fokus setzen: ruhiger Uebergang in die schnelle Phase."
        elif feedback_focus:
            next_focus = feedback_focus
        strengths = _unique_texts(
            [
                _simplify_coaching_line(item)
                for item in (jump_brief.get("strengths") or [])
                if item
            ]
        )[:2]
        if not strengths:
            strengths = ["Der Sprung war insgesamt nutzbar."]
        return {
            "summary": summary,
            "summary_ui": _render_simple_glossary(summary),
            "basis_line": "",
            "basis_line_ui": "",
            "primary_diagnosis": primary_diagnosis,
            "main_issues": [main_issue],
            "main_issues_ui": [_render_simple_glossary(main_issue)],
            "strengths": strengths,
            "strengths_ui": [_render_simple_glossary(item) for item in strengths],
            "actions": [next_focus],
            "actions_ui": [_render_simple_glossary(next_focus)],
        }

    weak_rows = sorted(
        [row for row in scorecard_rows if int(row.get("score", 0)) < 70],
        key=lambda row: (
            _scorecard_phase_rank(str(row.get("name", ""))),
            int(row.get("score", 0)),
        ),
    )
    strong_rows = sorted(
        [
            row
            for row in scorecard_rows
            if int(row.get("score", 0)) >= 70 and not _scorecard_row_has_angle_issue(row)
        ],
        key=lambda row: int(row.get("score", 0)),
        reverse=True,
    )

    issue_focus_names = _focus_names_from_issue_texts(
        [str(item) for item in (jump_brief.get("main_issues") or []) if item]
    )
    weak_labels = _unique_texts(
        [_simple_phase_label(name) for name in issue_focus_names[:2]]
        if issue_focus_names
        else [_simple_phase_label(str(row.get("name", ""))) for row in weak_rows[:2]]
    )
    if weak_labels:
        if len(weak_labels) == 1:
            summary = f"Dein größter Hebel liegt aktuell bei: {weak_labels[0]}."
        else:
            summary = f"Deine größten Hebel liegen aktuell bei: {weak_labels[0]} und {weak_labels[1]}."
    else:
        summary = "Stabiler Sprung mit guter Linie. Jetzt vor allem ruhig wiederholen."

    fallback = []
    for item in (jump_brief.get("main_issues") or []):
        if not item or _hide_in_simple_view(item):
            continue
        simplified = _simple_issue_from_brief_text(item) or _simplify_coaching_line(item)
        if _is_malformed_simple_line(simplified):
            continue
        fallback.append(simplified)
    row_issues = [_simple_issue_for_scorecard_row(row) for row in weak_rows[:2]]
    main_issues = _unique_texts([item for item in fallback + row_issues if item])[:2]
    if not main_issues:
        main_issues = _unique_texts(
            [
                _simple_issue_for_phase(
                    phase_name=str(row.get("name", "")),
                    score=int(row.get("score", 0)),
                )
                for row in weak_rows[:2]
            ]
        )
    if not main_issues:
        main_issues = ["Keine klaren Hauptprobleme sichtbar."]
    feedback_hint = _feedback_context_line(feedback_context, key="coaching_hint")
    if feedback_hint:
        main_issues = _unique_texts([f"{main_issues[0]} {feedback_hint}"] + main_issues[1:])[:2]

    strengths = _unique_texts(
        [
            _simple_strength_for_phase(
                phase_name=str(row.get("name", "")),
                score=int(row.get("score", 0)),
            )
            for row in strong_rows[:2]
        ]
    )
    if not strengths:
        fallback = [
            _simplify_coaching_line(item)
            for item in (jump_brief.get("strengths") or [])
            if item
        ]
        strengths = _unique_texts([item for item in fallback if item])[:2]
    if not strengths:
        strengths = ["Der Sprung war insgesamt nutzbar."]

    actions = _unique_texts(
        [
            _simple_action_for_phase(
                phase_name=str(row.get("name", "")),
                score=int(row.get("score", 0)),
            )
            for row in weak_rows
        ]
    )
    if issue_focus_names:
        issue_actions = [
            _simple_action_for_phase(phase_name=str(name), score=50)
            for name in issue_focus_names[:2]
        ]
        actions = _unique_texts(issue_actions + actions)
    if not actions:
        actions = ["Diesen Ablauf im nächsten Sprung ruhig und sauber wiederholen."]
    feedback_focus = _feedback_context_line(feedback_context, key="next_focus_hint")
    if feedback_focus:
        actions = _unique_texts([feedback_focus] + actions)
    actions = _harmonize_action_texts(actions, max_items=3)

    basis_line = ""
    for item in jump_brief.get("basis_lines") or []:
        text = str(item).strip().lower()
        if "bester gespeicherter sprung dieses springers" in text:
            basis_line = "Vergleich: dein bester gespeicherter Sprung."
            break

    return {
        "summary": summary,
        "summary_ui": _render_simple_glossary(summary),
        "basis_line": basis_line,
        "basis_line_ui": _render_simple_glossary(basis_line),
        "main_issues": main_issues,
        "main_issues_ui": [_render_simple_glossary(item) for item in main_issues],
        "strengths": strengths,
        "strengths_ui": [_render_simple_glossary(item) for item in strengths],
        "actions": actions,
        "actions_ui": [_render_simple_glossary(item) for item in actions],
    }


def _feedback_context_line(feedback_context: dict[str, Any] | None, *, key: str) -> str:
    if not isinstance(feedback_context, dict) or not feedback_context.get("available"):
        return ""
    text = str(feedback_context.get(key) or "").strip()
    if not text:
        return ""
    return _truncate_text(text, 360)


def _build_ai_coaching(
    *,
    report: dict[str, Any],
    review: dict[str, Any],
    jump_brief: dict[str, Any],
    jump_brief_simple: dict[str, Any],
    scorecard_rows: list[dict[str, Any]],
    tip_follow_up: dict[str, Any],
    jumper_summary: dict[str, Any],
    feedback_context: dict[str, Any],
    quality_issue_lines: list[str],
    view_mode: str,
) -> dict[str, Any]:
    payload = _build_ai_coaching_payload(
        report=report,
        review=review,
        jump_brief=jump_brief,
        jump_brief_simple=jump_brief_simple,
        scorecard_rows=scorecard_rows,
        tip_follow_up=tip_follow_up,
        jumper_summary=jumper_summary,
        feedback_context=feedback_context,
        quality_issue_lines=quality_issue_lines,
        view_mode=view_mode,
    )
    result = generate_ai_coaching_texts(
        payload,
        view_mode=view_mode,
        enabled=AI_COACHING_ENABLED,
        model=AI_COACHING_MODEL,
        timeout_s=AI_COACHING_TIMEOUT_S,
        max_requests_per_day=AI_COACHING_MAX_REQUESTS_PER_DAY,
    )
    result["payload_schema_version"] = AI_COACHING_SCHEMA_VERSION
    return result


def _build_coaching_snapshot(
    *,
    report: dict[str, Any],
    review: dict[str, Any],
    jump_brief: dict[str, Any],
    jump_brief_simple: dict[str, Any],
    ai_coaching: dict[str, Any],
    feedback_context: dict[str, Any],
    view_mode: str,
) -> dict[str, Any]:
    jump = report.get("jump", {}) if isinstance(report.get("jump"), dict) else {}
    focus_text = ""
    source = "fallback"
    if isinstance(ai_coaching, dict) and ai_coaching.get("available"):
        focus_text = str(ai_coaching.get("next_jump_focus") or "").strip()
        source = "ai"
    if not focus_text:
        brief = jump_brief_simple if view_mode == "simple" else jump_brief
        actions = brief.get("actions") if isinstance(brief, dict) else []
        focus_text = next((str(item).strip() for item in actions or [] if str(item or "").strip()), "")
        source = "rule"
    if not focus_text:
        return {"available": False, "reason": "NO_FOCUS_TEXT"}
    source_kind = source

    feedback_goal = _feedback_snapshot_goal(feedback_context)
    review_goals = [
        _snapshot_goal_from_review_goal(goal)
        for goal in (review.get("coaching_goals") if isinstance(review, dict) else []) or []
        if isinstance(goal, dict)
    ]
    review_goals = [goal for goal in review_goals if goal]

    focus_uses_feedback = _snapshot_focus_uses_feedback(
        focus_text=focus_text,
        feedback_focus=str((feedback_context or {}).get("next_focus_hint") or ""),
    )
    display_targets: list[dict[str, Any]] = []
    display_phase = "Coaching-Fokus"
    matched_review_goal = _snapshot_matching_review_goal(focus_text=focus_text, review_goals=review_goals)
    if focus_uses_feedback and feedback_goal:
        display_targets = list(feedback_goal.get("target_metrics") or [])
        display_phase = str(feedback_goal.get("phase") or display_phase)
        source = f"{source}+feedback"
    elif matched_review_goal:
        display_targets = list(matched_review_goal.get("target_metrics") or [])
        display_phase = str(matched_review_goal.get("phase") or display_phase)
    elif review_goals and source_kind == "rule":
        display_targets = list(review_goals[0].get("target_metrics") or [])
        display_phase = str(review_goals[0].get("phase") or display_phase)
    elif feedback_goal:
        display_targets = list(feedback_goal.get("target_metrics") or [])
        display_phase = str(feedback_goal.get("phase") or display_phase)
        source = f"{source}+feedback"

    goals: list[dict[str, Any]] = []
    if display_targets:
        goals.append(
            {
                "id": "display_focus",
                "phase": display_phase,
                "text": focus_text,
                "target_metrics": _dedupe_target_metrics(display_targets),
            }
        )
    if feedback_goal and not _snapshot_goal_is_duplicate(feedback_goal, goals):
        goals.append(feedback_goal)
    for goal in review_goals:
        if len(goals) >= 4:
            break
        if _snapshot_goal_is_duplicate(goal, goals):
            continue
        goals.append(goal)

    if not goals:
        return {"available": False, "reason": "NO_TARGET_METRICS", "focus_text": focus_text}

    return {
        "available": True,
        "schema_version": 1,
        "analysis_version": str(
            (report.get("metrics") or {}).get("analysis_version")
            or jump.get("analysis_version")
            or ANALYSIS_VERSION
        ),
        "jump_id": str(jump.get("jump_id") or ""),
        "view_mode": view_mode,
        "source": source,
        "focus_text": focus_text,
        "goals": goals[:4],
        "feedback_used": bool(feedback_goal),
        "ai_used": bool(isinstance(ai_coaching, dict) and ai_coaching.get("available")),
    }


def _snapshot_goal_from_review_goal(goal: dict[str, Any]) -> dict[str, Any] | None:
    text = str(goal.get("text") or goal.get("display_text") or "").strip()
    target_metrics = goal.get("target_metrics") if isinstance(goal.get("target_metrics"), list) else []
    if not text or not target_metrics:
        return None
    return {
        "id": str(goal.get("id") or "review_goal"),
        "phase": str(goal.get("phase") or "Coaching-Ziel"),
        "text": _strip_priority_prefix(text),
        "target_metrics": _dedupe_target_metrics(target_metrics),
    }


def _feedback_snapshot_goal(feedback_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(feedback_context, dict) or not feedback_context.get("available"):
        return None
    text = str(feedback_context.get("next_focus_hint") or "").strip()
    if not text:
        return None
    evidence = feedback_context.get("evidence") if isinstance(feedback_context.get("evidence"), dict) else {}
    metrics: list[dict[str, Any]] = []

    def add(metric: str, label: str, direction: str, min_delta: float, unit: str, decimals: int = 1) -> None:
        metrics.append(
            {
                "metric": metric,
                "label": label,
                "direction": direction,
                "min_delta": float(min_delta),
                "unit": unit,
                "decimals": int(decimals),
            }
        )

    phase = "Max-Speed Fenster"
    if evidence.get("end_unstable") or evidence.get("late_hard_transition"):
        phase = "Hot-Zone Aufbau"
        add("vhor_min_20_25", "vHor-Min 20-25s", "increase", 2.0, " km/h")
        add("angle_turns_20_25", "Korrekturen 20-25s", "decrease", 1.0, "", 0)
        add("negative_risk_score", "Risiko-Score", "decrease", 5.0, "")
    if evidence.get("build_too_flat"):
        phase = "Hauptbeschleunigung"
        add("gain_10_20", "Zuwachs +10 bis +20s", "increase", 8.0, " km/h")
    if not metrics:
        add("angle_turns_20_25", "Korrekturen 20-25s", "decrease", 1.0, "", 0)

    return {
        "id": "feedback_focus",
        "phase": phase,
        "text": text,
        "target_metrics": _dedupe_target_metrics(metrics),
    }


def _snapshot_focus_uses_feedback(*, focus_text: str, feedback_focus: str) -> bool:
    focus_norm = normalize_german_text(focus_text)
    feedback_norm = normalize_german_text(feedback_focus)
    if not focus_norm or not feedback_norm:
        return False
    if _is_semantic_near_duplicate_text(focus_text, feedback_focus):
        return True
    feedback_markers = [
        "kompakt",
        "arm",
        "schulter",
        "bein",
        "schrittweise",
        "kleiner testen",
        "gefuehl",
        "feedback",
    ]
    return any(marker in focus_norm for marker in feedback_markers)


def _snapshot_matching_review_goal(*, focus_text: str, review_goals: list[dict[str, Any]]) -> dict[str, Any] | None:
    focus = str(focus_text or "").strip()
    if not focus or not review_goals:
        return None
    for goal in review_goals:
        goal_text = str(goal.get("text") or goal.get("display_text") or "").strip()
        if goal_text and _is_semantic_near_duplicate_text(focus, goal_text):
            return goal

    focus_topic = _topic_from_text(focus)
    focus_tokens = _semantic_tokens(focus)
    best_goal: dict[str, Any] | None = None
    best_score = 0.0
    for goal in review_goals:
        goal_text = str(goal.get("text") or goal.get("display_text") or "").strip()
        if not goal_text:
            continue
        goal_topic = _topic_from_text(goal_text)
        phase_topic = _topic_from_text(str(goal.get("phase") or ""))
        goal_tokens = _semantic_tokens(goal_text)
        overlap = len(focus_tokens & goal_tokens)
        score = 0.0
        if focus_topic != "other" and focus_topic == goal_topic:
            score += 4.0
        if focus_topic != "other" and focus_topic == phase_topic:
            score += 2.0
        if overlap:
            score += min(3.0, float(overlap))
            score += min(2.0, float(overlap) / max(1.0, float(len(focus_tokens))) * 4.0)
        if score > best_score:
            best_goal = goal
            best_score = score
    return best_goal if best_score >= 4.0 else None


def _snapshot_goal_is_duplicate(candidate: dict[str, Any], existing: list[dict[str, Any]]) -> bool:
    text = str(candidate.get("text") or "")
    metrics = {
        str(item.get("metric") or "")
        for item in candidate.get("target_metrics") or []
        if isinstance(item, dict)
    }
    for item in existing:
        other_text = str(item.get("text") or "")
        other_metrics = {
            str(metric.get("metric") or "")
            for metric in item.get("target_metrics") or []
            if isinstance(metric, dict)
        }
        if text and other_text and _is_semantic_near_duplicate_text(text, other_text):
            return True
        if metrics and metrics == other_metrics:
            return True
    return False


def _dedupe_target_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in metrics:
        if not isinstance(raw, dict):
            continue
        metric = str(raw.get("metric") or "").strip()
        direction = str(raw.get("direction") or "").strip()
        if not metric or not direction:
            continue
        key = (metric, direction)
        if key in seen:
            continue
        seen.add(key)
        raw_decimals = raw.get("decimals")
        decimals = int(raw_decimals) if raw_decimals is not None else 1
        out.append(
            {
                "metric": metric,
                "label": str(raw.get("label") or metric),
                "direction": direction,
                "min_delta": float(_to_float(raw.get("min_delta")) or 0.0),
                "unit": str(raw.get("unit") or ""),
                "decimals": decimals,
            }
        )
    return out


def _build_ai_coaching_payload(
    *,
    report: dict[str, Any],
    review: dict[str, Any],
    jump_brief: dict[str, Any],
    jump_brief_simple: dict[str, Any],
    scorecard_rows: list[dict[str, Any]],
    tip_follow_up: dict[str, Any],
    jumper_summary: dict[str, Any],
    quality_issue_lines: list[str],
    view_mode: str,
    feedback_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    jump = report.get("jump", {}) if isinstance(report.get("jump"), dict) else {}
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    performance_profile = (
        jumper_summary.get("performance_profile", {})
        if isinstance(jumper_summary.get("performance_profile"), dict)
        else {}
    )
    selected_brief = jump_brief

    return {
        "schema_version": AI_COACHING_SCHEMA_VERSION,
        "view_mode": view_mode,
        "jump": {
            "jumper_name": str(jump.get("jumper_name") or "") if AI_COACHING_INCLUDE_IDENTIFIERS else "",
            "file_name": str(jump.get("file_name") or "") if AI_COACHING_INCLUDE_IDENTIFIERS else "",
            "jump_context": _normalize_jump_context(str(jump.get("jump_context") or "unknown")),
            "jump_context_label": _jump_context_label(str(jump.get("jump_context") or "unknown")),
            "t0_utc": str(jump.get("t0_utc") or "") if AI_COACHING_INCLUDE_IDENTIFIERS else "",
        },
        "performance_profile": _compact_performance_profile_for_ai(performance_profile),
        "personal_timing": _compact_timing_reference_for_ai(
            jumper_summary.get("timing_reference")
            if isinstance(jumper_summary, dict)
            else None
        ),
        "feedback_training_profile": _compact_feedback_training_profile_for_ai(
            jumper_summary.get("feedback_training_profile")
            if isinstance(jumper_summary, dict)
            else None
        ),
        "metrics": {
            "best_3s_vVert_kmh": _round_float(metrics.get("best_3s_vVert_kmh"), 1),
            "best_3s_start_s": _round_float(metrics.get("best_3s_start_s"), 1),
            "best_3s_end_s": _round_float(metrics.get("best_3s_end_s"), 1),
            "best_3s_vHor_kmh": _round_float(metrics.get("best_3s_vHor_kmh"), 1),
            "rule_based_3s_score": _round_float(metrics.get("rule_based_3s_score"), 1),
            "negative_risk_score": _round_float(metrics.get("negative_risk_score"), 1),
            "curve_window_start_s": _round_float(notes.get("curve_window_start_s"), 1),
            "curve_window_end_s": _round_float(notes.get("curve_window_end_s"), 1),
        },
        "scorecard": [
            {
                "phase": str(row.get("name") or ""),
                "score": int(row.get("score", 0) or 0),
                "status": str(row.get("status") or ""),
                "reason": _truncate_text(str(row.get("reason") or ""), 260),
            }
            for row in scorecard_rows[:5]
        ],
        "technical_phases": _compact_phase_rows_for_ai(_phase_rows_for_report(report)),
        "review": {
            "happened": _limit_texts(review.get("happened"), max_items=6, max_len=260),
            "good": _limit_texts(review.get("good"), max_items=4, max_len=220),
            "not_good": _limit_texts(review.get("not_good"), max_items=5, max_len=260),
            "coaching_goals": _compact_coaching_goals_for_ai(review.get("coaching_goals")),
        },
        "primary_diagnosis": _compact_primary_diagnosis_for_ai(review.get("primary_diagnosis")),
        "technical_assessment": _compact_technical_assessment_for_ai(review.get("technical_assessment")),
        "jump_brief": {
            "summary": _truncate_text(str(selected_brief.get("summary") or ""), 280),
            "main_issues": _limit_texts(selected_brief.get("main_issues"), max_items=4, max_len=260),
            "strengths": _limit_texts(selected_brief.get("strengths"), max_items=3, max_len=220),
            "actions": _limit_texts(selected_brief.get("actions"), max_items=4, max_len=420),
        },
        "tip_follow_up": _compact_tip_follow_up_for_ai(tip_follow_up),
        "jump_feedback": _compact_feedback_context_for_ai(feedback_context),
        "quality": {
            "analysis_blocked": bool(notes.get("analysis_blocked")),
            "quality_flags": [str(item) for item in (report.get("quality_flags") or [])[:8]],
            "quality_issue_lines": _limit_texts(quality_issue_lines, max_items=4, max_len=240),
        },
    }


def _compact_primary_diagnosis_for_ai(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value.get("available"):
        return {"available": False}
    evidence = value.get("evidence") if isinstance(value.get("evidence"), dict) else {}
    return {
        "available": True,
        "severity": str(value.get("severity") or ""),
        "phase": str(value.get("phase") or ""),
        "title": _truncate_text(str(value.get("title") or ""), 120),
        "summary": _truncate_text(str(value.get("summary") or ""), 260),
        "main_issue": _truncate_text(str(value.get("main_issue") or ""), 260),
        "next_focus": _truncate_text(str(value.get("next_focus") or ""), 260),
        "evidence": {
            "angle_20": _round_float(evidence.get("angle_20"), 1),
            "angle_peak": _round_float(evidence.get("angle_peak"), 1),
            "angle_peak_s": _round_float(evidence.get("angle_peak_s"), 1),
            "angle_gain_20_peak": _round_float(evidence.get("angle_gain_20_peak"), 1),
            "vhor_20": _round_float(evidence.get("vhor_20"), 1),
            "vhor_min_after_20": _round_float(evidence.get("vhor_min_after_20"), 1),
            "vhor_drop_after_20_pct": _round_float(evidence.get("vhor_drop_after_20_pct"), 0),
            "vhor_min_time_s": _round_float(evidence.get("vhor_min_time_s"), 1),
            "best_3s_start_s": _round_float(evidence.get("best_3s_start_s"), 1),
            "best_3s_end_s": _round_float(evidence.get("best_3s_end_s"), 1),
            "vhor_rebound_to_best_end_pct": _round_float(evidence.get("vhor_rebound_to_best_end_pct"), 0),
            "angle_at_vhor_min": _round_float(evidence.get("angle_at_vhor_min"), 1),
            "forward_loss_to_best_end_m": _round_float(evidence.get("forward_loss_to_best_end_m"), 1),
        },
    }


def _compact_feedback_context_for_ai(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value.get("available"):
        return {"available": False}
    evidence = value.get("evidence") if isinstance(value.get("evidence"), dict) else {}
    lateral = evidence.get("lateral") if isinstance(evidence.get("lateral"), dict) else {}
    return {
        "available": True,
        "subjective_text_excerpt": _truncate_text(str(value.get("text_excerpt") or ""), 500),
        "intents": [
            {"key": str(item.get("key") or ""), "label": str(item.get("label") or "")}
            for item in (value.get("intents") or [])[:6]
            if isinstance(item, dict)
        ],
        "felt_issues": [
            {"key": str(item.get("key") or ""), "label": str(item.get("label") or "")}
            for item in (value.get("felt_issues") or [])[:5]
            if isinstance(item, dict)
        ],
        "question_present": bool(value.get("question_present")),
        "match_lines": _limit_texts(value.get("match_lines"), max_items=4, max_len=260),
        "coaching_hint": _truncate_text(str(value.get("coaching_hint") or ""), 360),
        "next_focus_hint": _truncate_text(str(value.get("next_focus_hint") or ""), 280),
        "confidence": str(value.get("confidence") or ""),
        "caution": str(value.get("caution") or ""),
        "evidence": {
            "objective_signals": _limit_texts(evidence.get("objective_signals"), max_items=6, max_len=120),
            "end_unstable": bool(evidence.get("end_unstable")),
            "late_hard_transition": bool(evidence.get("late_hard_transition")),
            "too_fast_steep": bool(evidence.get("too_fast_steep")),
            "build_too_flat": bool(evidence.get("build_too_flat")),
            "hot_zone_label": str(evidence.get("hot_zone_label") or ""),
            "risk_label": str(evidence.get("risk_label") or ""),
            "lateral_hot_pattern": str(lateral.get("hot_pattern") or ""),
            "lateral_hot_vlat_abs_mean_kmh": _round_float(lateral.get("hot_vlat_abs_mean_kmh"), 1),
            "lateral_hot_heading_rate_rms_dps": _round_float(lateral.get("hot_heading_rate_rms_dps"), 1),
        },
    }


def _compact_feedback_training_profile_for_ai(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value.get("available"):
        return {"available": False, "reason": str((value or {}).get("reason") or "") if isinstance(value, dict) else ""}
    return {
        "available": True,
        "feedback_count": int(value.get("feedback_count") or 0),
        "recent_feedback_count": int(value.get("recent_feedback_count") or 0),
        "summary": _truncate_text(str(value.get("summary") or ""), 260),
        "lines": _limit_texts(value.get("lines"), max_items=4, max_len=260),
        "top_focus_keys": [str(item) for item in (value.get("top_focus_keys") or [])[:4]],
    }


def _compact_phase_rows_for_ai(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in rows[:5]:
        if not isinstance(raw, dict):
            continue
        out.append(_compact_technical_phase_for_ai(raw))
    return out


def _compact_technical_phase_for_ai(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(raw.get("name") or ""),
        "start_s": _round_float(raw.get("start_s"), 1),
        "end_s": _round_float(raw.get("end_s"), 1),
        "target_angle": str(raw.get("target_angle_label") or ""),
        "angle_status": str(raw.get("target_status") or raw.get("angle_status") or ""),
        "avg_angle_deg": _round_float(raw.get("avg_angle_deg"), 1),
        "angle_start_deg": _round_float(raw.get("angle_start_deg"), 1),
        "angle_end_deg": _round_float(raw.get("angle_end_deg"), 1),
        "angle_delta_deg": _round_float(raw.get("angle_delta_deg"), 1),
        "avg_vVert_kmh": _round_float(raw.get("avg_vVert_kmh"), 1),
        "avg_vHor_kmh": _round_float(raw.get("avg_vHor_kmh"), 1),
        "max_angle_deg": _round_float(raw.get("max_angle_deg"), 1),
        "min_vhor_kmh": _round_float(raw.get("min_vhor_kmh"), 1),
        "vvert_gain_kmh": _round_float(raw.get("vvert_gain_kmh", raw.get("vVert_gain_kmh")), 1),
        "comment": _truncate_text(str(raw.get("comment") or ""), 180),
    }


def _compact_technical_assessment_for_ai(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value.get("available"):
        return {"available": False}

    best_window = value.get("best_window_quality") if isinstance(value.get("best_window_quality"), dict) else {}
    jerk = value.get("jerk_quality") if isinstance(value.get("jerk_quality"), dict) else {}
    phase_rows: list[dict[str, Any]] = []
    all_phase_rows: list[dict[str, Any]] = []
    for raw in value.get("phases") or []:
        if not isinstance(raw, dict):
            continue
        if len(all_phase_rows) < 5:
            all_phase_rows.append(_compact_technical_phase_for_ai(raw))
        if not (raw.get("steep_without_gain") or raw.get("oversteep_vhor_cost")):
            continue
        phase_rows.append(
            {
                "name": str(raw.get("name") or ""),
                "start_s": _round_float(raw.get("start_s"), 1),
                "end_s": _round_float(raw.get("end_s"), 1),
                "angle_status": str(raw.get("angle_status") or ""),
                "avg_angle_deg": _round_float(raw.get("avg_angle_deg"), 1),
                "max_angle_deg": _round_float(raw.get("max_angle_deg"), 1),
                "vvert_gain_kmh": _round_float(raw.get("vvert_gain_kmh"), 1),
                "min_vhor_kmh": _round_float(raw.get("min_vhor_kmh"), 1),
                "acc_mean_mps2": _round_float(raw.get("acc_mean_mps2"), 2),
                "steep_without_gain": bool(raw.get("steep_without_gain")),
                "oversteep_vhor_cost": bool(raw.get("oversteep_vhor_cost")),
            }
        )
        if len(phase_rows) >= 3:
            break

    return {
        "available": True,
        "summary_line": _truncate_text(str(value.get("summary_line") or ""), 220),
        "best_window_quality": {
            "available": bool(best_window.get("available")),
            "label": str(best_window.get("label") or ""),
            "source": str(best_window.get("source") or ""),
            "vvert_std_kmh": _round_float(best_window.get("vvert_std_kmh"), 1),
            "angle_std_deg": _round_float(best_window.get("angle_std_deg"), 1),
            "vhor_min_kmh": _round_float(best_window.get("vhor_min_kmh"), 1),
            "vvert_drop_after_kmh": _round_float(best_window.get("vvert_drop_after_kmh"), 1),
            "drop_after_evaluable": bool(best_window.get("drop_after_evaluable")),
            "drop_after_reason": str(best_window.get("drop_after_reason") or ""),
            "drop_after_window_s": _round_float(best_window.get("drop_after_window_s"), 1),
            "raw_best_start_s": _round_float(best_window.get("raw_best_start_s"), 1),
            "raw_best_end_s": _round_float(best_window.get("raw_best_end_s"), 1),
            "rule_based_3s_score": _round_float(best_window.get("rule_based_3s_score"), 1),
        },
        "jerk_quality": {
            "available": bool(jerk.get("available")),
            "label": str(jerk.get("label") or ""),
            "jerk_rms_mps3": _round_float(jerk.get("jerk_rms_mps3"), 1),
        },
        "phases": all_phase_rows,
        "phase_issues": phase_rows,
        "issues": _limit_texts(value.get("issues"), max_items=3, max_len=240),
        "actions": [
            _truncate_text(str(item.get("text") or ""), 260)
            for item in (value.get("actions") or [])
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ][:3],
    }


def _compact_performance_profile_for_ai(profile: dict[str, Any]) -> dict[str, Any]:
    if not profile or not profile.get("available"):
        return {"available": False, "summary": str(profile.get("reason") or "Noch kein Leistungsprofil.")}
    return {
        "available": True,
        "summary": str(profile.get("summary") or ""),
        "performance_band": str(profile.get("performance_band") or ""),
        "performance_band_label": str(profile.get("performance_band_label") or ""),
        "confidence": str(profile.get("confidence") or ""),
        "confidence_label": str(profile.get("confidence_label") or ""),
        "valid_jump_count": int(profile.get("valid_jump_count") or 0),
        "top_available_avg_kmh": _round_float(profile.get("top_available_avg_kmh"), 1),
    }


def _compact_timing_reference_for_ai(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value.get("available") or str(value.get("maturity") or "active") != "active":
        return {
            "available": False,
            "reason": str((value or {}).get("reason") or "") if isinstance(value, dict) else "",
            "maturity": str((value or {}).get("maturity") or "off") if isinstance(value, dict) else "off",
        }
    anchors = value.get("anchors") if isinstance(value.get("anchors"), dict) else {}
    compact_anchors: dict[str, dict[str, Any]] = {}
    for key in [
        "angle_75_time_s",
        "angle_82_time_s",
        "angle_85_time_s",
        "best_3s_start_s",
        "best_3s_end_s",
        "vvert_peak_time_s",
        "decel_start_s",
    ]:
        raw = anchors.get(key) if isinstance(anchors, dict) else None
        if not isinstance(raw, dict):
            continue
        compact_anchors[key] = {
            "label": str(raw.get("label") or key),
            "low_s": _round_float(raw.get("low"), 1),
            "high_s": _round_float(raw.get("high"), 1),
            "median_s": _round_float(raw.get("median"), 1),
        }
    return {
        "available": True,
        "maturity": "active",
        "confidence": str(value.get("confidence") or ""),
        "basis_count": int(value.get("basis_count") or 0),
        "usable_count": int(value.get("usable_count") or 0),
        "source": str(value.get("source") or ""),
        "best_reference_kmh": _round_float(value.get("best_reference_kmh"), 1),
        "top3_avg_kmh": _round_float(value.get("top3_avg_kmh"), 1),
        "max_timing_spread_s": _round_float(value.get("max_timing_spread_s"), 1),
        "summary_lines": _limit_texts(value.get("summary_lines"), max_items=3, max_len=220),
        "anchors": compact_anchors,
    }


def _compact_coaching_goals_for_ai(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in value[:5]:
        if not isinstance(raw, dict):
            continue
        target_metrics: list[dict[str, Any]] = []
        for target in raw.get("target_metrics") or []:
            if not isinstance(target, dict):
                continue
            target_metrics.append(
                {
                    "metric": str(target.get("metric") or ""),
                    "label": str(target.get("label") or ""),
                    "direction": str(target.get("direction") or ""),
                    "min_delta": _round_float(target.get("min_delta"), 2),
                    "unit": str(target.get("unit") or ""),
                }
            )
        out.append(
            {
                "id": str(raw.get("id") or ""),
                "priority": int(raw.get("priority") or 0),
                "phase": str(raw.get("phase") or ""),
                "text": _truncate_text(str(raw.get("text") or ""), 260),
                "target_metrics": target_metrics[:4],
            }
        )
    return out


def _compact_tip_follow_up_for_ai(tip_follow_up: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tip_follow_up, dict) or not tip_follow_up.get("available"):
        return {
            "available": False,
            "reason": str((tip_follow_up or {}).get("reason") or ""),
        }
    entries: list[dict[str, Any]] = []
    for raw in tip_follow_up.get("entries") or []:
        if not isinstance(raw, dict):
            continue
        entries.append(
            {
                "phase": str(raw.get("phase") or ""),
                "status": str(raw.get("status") or ""),
                "status_key": str(raw.get("status_key") or ""),
                "goal_text": _truncate_text(str(raw.get("goal_text") or ""), 260),
                "detail": _truncate_text(str(raw.get("detail") or ""), 420),
            }
        )
    return {
        "available": True,
        "summary": str(tip_follow_up.get("summary") or ""),
        "quality_note": str(tip_follow_up.get("quality_note") or ""),
        "compact_note": str(tip_follow_up.get("compact_note") or ""),
        "entries": entries[:4],
    }


def _round_float(value: Any, decimals: int = 1) -> float | None:
    number = _to_float(value)
    if number is None:
        return None
    return round(float(number), int(decimals))


def _limit_texts(value: Any, *, max_items: int, max_len: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in value:
        text = _truncate_text(str(raw or ""), max_len)
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _truncate_text(value: str, max_len: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)].rstrip() + "..."


def _render_simple_glossary(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    pattern = _SIMPLE_GLOSSARY_PATTERN
    if pattern is None:
        return html_escape(raw)

    out: list[str] = []
    last = 0
    for match in pattern.finditer(raw):
        start, end = match.span()
        if start > last:
            out.append(html_escape(raw[last:start]))
        token = match.group(0)
        tip = _SIMPLE_GLOSSARY_LOOKUP.get(normalize_german_text(token))
        if not tip:
            out.append(html_escape(token))
        else:
            out.append(
                f'<span class="glossary-term" tabindex="0" title="{html_escape(tip, quote=True)}">'
                f"{html_escape(token)}"
                "</span>"
            )
        last = end
    if last < len(raw):
        out.append(html_escape(raw[last:]))
    return "".join(out)


def _simple_phase_label(phase_name: str) -> str:
    lowered = normalize_german_text(phase_name.strip())
    if "exit" in lowered:
        return "Startphase"
    if "dive" in lowered:
        return "Dive-Aufbau"
    if "hauptbeschleunigung" in lowered:
        return "Hauptbeschleunigung"
    if "max-speed" in lowered or "max speed" in lowered:
        return "Max-Speed-Fenster"
    if "hot" in lowered and "aufbau" in lowered:
        return "Hot-Zone-Aufbau"
    if "aufbau" in lowered and "hot" not in lowered:
        return "Aufbauphase"
    if "hot" in lowered:
        return "schnelle Phase"
    if "stabilität" in lowered or "kipp" in lowered:
        return "Flugruhe"
    return "Flugverlauf"


def _simple_issue_for_phase(*, phase_name: str, score: int) -> str:
    lowered = normalize_german_text(phase_name.strip())
    if "exit" in lowered:
        if score < 55:
            return "Der Start war zu unruhig. Du verlierst direkt nach dem Absprung zu viel Druck."
        return "Der Start war okay, aber der Druck wurde nicht stabil genug mitgenommen."
    if "dive" in lowered:
        if score < 55:
            return "Der Dive-Aufbau passt noch nicht zum Zielwinkel. Der Druck kommt dadurch zu unruhig."
        return "Der Dive-Aufbau braucht noch einen ruhigeren Zielwinkel."
    if "hauptbeschleunigung" in lowered:
        if score < 55:
            return "In der Hauptbeschleunigung fehlt ein sauberer Speed-Aufbau im Zielwinkel."
        return "Die Hauptbeschleunigung ist da, aber noch nicht gleichmaessig genug."
    if "hot" in lowered and "aufbau" in lowered:
        if score < 55:
            return "Im Hot-Zone-Aufbau fehlt der ruhige Uebergang in den Zielwinkel."
        return "Der Hot-Zone-Aufbau ist noch nicht ruhig genug."
    if "max-speed" in lowered or "max speed" in lowered:
        if score < 55:
            return "Im Max-Speed-Fenster geht zu viel Stabilitaet oder vHor verloren."
        return "Das Max-Speed-Fenster ist noch nicht ruhig genug gehalten."
    if "aufbau" in lowered:
        if score < 55:
            return "Im Mittelteil bricht der Aufbau zu früh ab. Die Beschleunigung bleibt nicht konstant."
        return "Im Mittelteil fehlt noch ein ruhiger, gleichmäßiger Aufbau."
    if "hot" in lowered:
        if score < 55:
            return "In der schnellen Phase geht zu viel Stabilität verloren. Dadurch fällt Speed weg."
        return "In der schnellen Phase gab es zu viele Korrekturen. Das kostet Tempo."
    if "stabilität" in lowered or "kipp" in lowered:
        if score < 55:
            return "Der Flug war im schnellen Teil deutlich unruhig."
        return "Der Flug war stellenweise unruhig, vor allem im späten schnellen Abschnitt."
    return "Der Flug war nicht durchgehend ruhig und stabil."


def _simple_issue_for_scorecard_row(row: dict[str, Any]) -> str:
    phase_name = str(row.get("name") or "")
    status = normalize_german_text(str(row.get("target_status") or ""))
    avg_angle = _to_float(row.get("avg_angle_deg"))
    target = str(row.get("target_angle_label") or "").strip()
    label = _simple_phase_label(phase_name)
    if "zu steil" in status and avg_angle is not None and target:
        return f"{label}: Der mittlere Winkel liegt mit {_fmt1(avg_angle)} Grad ueber dem Ziel {target}."
    if "zu flach" in status and avg_angle is not None and target:
        return f"{label}: Der mittlere Winkel liegt mit {_fmt1(avg_angle)} Grad unter dem Ziel {target}."
    if "eher flach" in status and target:
        return f"{label}: Der Durchschnitt passt knapp, aber einzelne Abschnitte fallen unter das Ziel {target}."
    if "kurz zu steil" in status and target:
        return f"{label}: Der Durchschnitt passt, aber einzelne Abschnitte werden zu steil fuer das Ziel {target}."
    return _simple_issue_for_phase(phase_name=phase_name, score=int(row.get("score", 0) or 0))


def _simple_strength_for_phase(*, phase_name: str, score: int) -> str:
    lowered = normalize_german_text(phase_name.strip())
    if "exit" in lowered:
        return "Der Start war kontrolliert."
    if "dive" in lowered:
        return "Der Dive-Aufbau war kontrolliert."
    if "hauptbeschleunigung" in lowered:
        return "Die Hauptbeschleunigung lag gut im Zielbereich."
    if "max-speed" in lowered or "max speed" in lowered:
        return "Das Max-Speed-Fenster war gut nutzbar."
    if "hot" in lowered and "aufbau" in lowered:
        return "Der Hot-Zone-Aufbau war kontrolliert."
    if "aufbau" in lowered:
        return "Der Aufbau in die Beschleunigung war sauber."
    if "hot" in lowered:
        return "Die schnelle Phase war über weite Strecken stabil."
    if "stabilität" in lowered or "kipp" in lowered:
        return "Die Fluglinie blieb über weite Strecken ruhig."
    return "Der Sprung war in diesem Bereich stabil."


def _simple_action_for_phase(*, phase_name: str, score: int) -> str:
    lowered = normalize_german_text(phase_name.strip())
    if "exit" in lowered:
        return (
            "Startphase: Nach dem Absprung 2-3 Sekunden ruhig Druck halten "
            "und nur kleine Korrekturen machen."
        )
    if "dive" in lowered or "hauptbeschleunigung" in lowered:
        return (
            "Aufbauphase: Zwischen +10s und +20s gleichmäßig weiter beschleunigen, "
            "ohne hektische Richtungswechsel."
        )
    if "aufbau" in lowered and "hot" not in lowered:
        return (
            "Aufbauphase: Zwischen +10s und +20s gleichmäßig weiter beschleunigen, "
            "ohne hektische Richtungswechsel."
        )
    if "hot" in lowered:
        return (
            "Schnelle Phase: Kleine, frühe Korrekturen setzen. "
            "Späte große Korrekturen vermeiden."
        )
    if "stabilität" in lowered or "kipp" in lowered:
        if score < 55:
            return (
                "Wenn der Flug unruhig wird: kurz etwas flacher gehen, "
                "Linie beruhigen, dann wieder sauber aufbauen."
            )
        return "Linie ruhig halten und jede Korrektur so klein wie möglich setzen."
    return "Ruhiger fliegen: kleine Korrekturen, klare Linie, keine hektischen Nachbewegungen."


def _simplify_coaching_line(text: str) -> str:
    line = str(text or "").strip()
    if not line:
        return ""
    line = re.sub(r"^([^:]{2,40}):\s*", "", line)
    line = re.sub(r"\([^)]*\)", "", line)
    line = re.sub(r"\bMesswerte\b.*", "", line, flags=re.IGNORECASE)
    line = re.sub(
        r"[+-]?\d+(?:[.,]\d+)?\s*(?:km/h|m/s2|m/s|grad|deg|s|%)",
        "",
        line,
        flags=re.IGNORECASE,
    )
    replacements = [
        (r"\bhot-zone\b", "schnelle Phase"),
        (r"\bhot phase\b", "schnelle Phase"),
        (r"im relevanten bereich vorwärtsgerichtet", "waagerecht stabil"),
        (r"vorwärtsbewegung", "waagerechte Geschwindigkeit"),
        (r"vorwaertsbewegung", "waagerechte Geschwindigkeit"),
        (r"\bvhor\b", "waagerechte Geschwindigkeit"),
        (r"\bvvert\b", "Geschwindigkeit nach unten"),
    ]
    for pattern, repl in replacements:
        line = re.sub(pattern, repl, line, flags=re.IGNORECASE)
    line = re.sub(r"\bin der schnelle Phase\b", "in der schnellen Phase", line, flags=re.IGNORECASE)
    line = re.sub(r"\s+", " ", line).strip(" .;,-")
    return line


def _simple_issue_from_brief_text(text: str) -> str:
    raw = str(text or "")
    lowered = normalize_german_text(str(text or ""))
    if not lowered:
        return ""
    label = _simple_phase_label(raw.split(":", 1)[0]) if ":" in raw else "Diese Phase"
    if "bewertung:" in lowered and "avg winkel" in lowered and "unter dem ziel" in lowered:
        return f"{label}: Der Winkel liegt im Durchschnitt unter dem Zielbereich."
    if "bewertung:" in lowered and "avg winkel" in lowered and "ueber dem ziel" in lowered:
        return f"{label}: Der Winkel liegt im Durchschnitt ueber dem Zielbereich."
    if ("+10 bis +15" in lowered or "10 bis 15" in lowered) and "zuwachs" in lowered:
        return "Zwischen +10s und +15s kommt noch zu wenig zusaetzlicher Speed."
    if "hot-zone" in lowered and (
        "ineffizient" in lowered or "hoehenverlust" in lowered or "hohenverlust" in lowered
    ):
        return "In der Hot-Zone geht viel Hoehe verloren, aber es kommt nur wenig zusaetzlicher Speed dazu."
    if "hot-zone" in lowered and "vhor" in lowered and "verloren" in lowered:
        return "In der Hot-Zone faellt die waagerechte Geschwindigkeit zu stark ab."
    return ""


def _is_malformed_simple_line(text: str) -> bool:
    lowered = normalize_german_text(str(text or "").strip())
    if not lowered:
        return True
    malformed_markers = [
        "bis ,",
        "zielwinkel 60",
        "geschwindigkeit nach unten-zuwachs",
        "istwerte avg",
        "avg waagerechte geschwindigkeit",
        "avg vorwaertsrichtung",
    ]
    return any(marker in lowered for marker in malformed_markers)


def _hide_in_simple_view(text: str) -> bool:
    lowered = normalize_german_text(str(text or ""))
    hidden_markers = ["top-5", "top5", "benchmark", "referenzscore"]
    return any(normalize_german_text(marker) in lowered for marker in hidden_markers)


def _build_performance_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    context_counts = {"training": 0, "competition": 0, "unknown": 0}
    for row in records:
        context = _normalize_jump_context(str(row.get("jump_context") or "unknown"))
        context_counts[context] = context_counts.get(context, 0) + 1

    hard_flags = {
        "EARLY_JUMP_END",
        "SPEED_SPIKE",
        "TIME_GAPS",
        "NO_CLEAR_EXIT",
        "INVALID_EXIT_ALTITUDE",
        "LOW_GPS_FIX",
        "HIGH_SPEED_ACCURACY_ERROR",
    }
    usable: list[dict[str, Any]] = []
    excluded = 0
    ground_estimated_count = 0
    for row in records:
        flags = _parse_quality_flags(row.get("quality_flags"))
        if "NO_GROUND_LEVEL" in flags:
            ground_estimated_count += 1
        if row.get("learning_eligible") is False:
            excluded += 1
            continue
        if bool(row.get("analysis_blocked")) or bool(row.get("t0_review_required")) or (flags & hard_flags):
            excluded += 1
            continue
        if _to_float(row.get("rule_score_kmh")) is None:
            excluded += 1
            continue
        usable.append(row)

    rule_values = [
        float(value)
        for row in usable
        for value in [_to_float(row.get("rule_score_kmh"))]
        if value is not None
    ]
    training_values = [
        float(value)
        for row in usable
        for value in [_to_float(row.get("best_3s_kmh"))]
        if value is not None
    ]
    basis_values = rule_values
    if not basis_values:
        return {
            "available": False,
            "reason": "Noch keine gueltigen Spruenge fuer ein Leistungsprofil.",
            "valid_jump_count": 0,
            "excluded_count": excluded,
            "training_count": context_counts.get("training", 0),
            "competition_count": context_counts.get("competition", 0),
            "unknown_count": context_counts.get("unknown", 0),
        }

    basis_count = len(basis_values)
    top_label_count = min(10, basis_count)
    top_available = _top_avg(basis_values, top_label_count)
    confidence = _performance_profile_confidence(basis_count)
    if (
        ground_estimated_count
        and confidence == "stable"
        and ground_estimated_count >= max(2, basis_count // 2)
    ):
        confidence = "good"

    source = "rule"
    source_label = "Regel-Score"
    band = _performance_band_for_speed(top_available)
    summary = (
        f"{_PERFORMANCE_BAND_LABELS.get(band, band)}: Top-{top_label_count} {source_label} "
        f"{float(top_available):.1f} km/h, Profil {_PROFILE_CONFIDENCE_LABELS.get(confidence, confidence)}."
    )

    return {
        "available": True,
        "score_source": source,
        "score_source_label": source_label,
        "valid_jump_count": basis_count,
        "usable_record_count": len(usable),
        "excluded_count": excluded,
        "ground_estimated_count": ground_estimated_count,
        "top1_rule_kmh": _top_avg(rule_values, 1),
        "top3_rule_avg_kmh": _top_avg(rule_values, 3),
        "top5_rule_avg_kmh": _top_avg(rule_values, 5),
        "top10_rule_avg_kmh": _top_avg(rule_values, 10),
        "top_available_rule_avg_kmh": _top_avg(rule_values, min(10, len(rule_values))),
        "top1_training_kmh": _top_avg(training_values, 1),
        "top3_training_avg_kmh": _top_avg(training_values, 3),
        "top5_training_avg_kmh": _top_avg(training_values, 5),
        "top10_training_avg_kmh": _top_avg(training_values, 10),
        "top_available_training_avg_kmh": _top_avg(training_values, min(10, len(training_values))),
        "top_available_avg_kmh": round(float(top_available), 2),
        "top_available_count": top_label_count,
        "confidence": confidence,
        "confidence_label": _PROFILE_CONFIDENCE_LABELS.get(confidence, confidence),
        "performance_band": band,
        "performance_band_label": _PERFORMANCE_BAND_LABELS.get(band, band),
        "training_count": context_counts.get("training", 0),
        "competition_count": context_counts.get("competition", 0),
        "unknown_count": context_counts.get("unknown", 0),
        "summary": summary,
    }


def _top_avg(values: list[float], limit: int) -> float | None:
    if not values or limit <= 0:
        return None
    selected = sorted([float(value) for value in values], reverse=True)[: int(limit)]
    if not selected:
        return None
    return round(float(sum(selected) / len(selected)), 2)


def _performance_profile_confidence(valid_count: int) -> str:
    count = max(0, int(valid_count))
    if count >= 10:
        return "stable"
    if count >= 6:
        return "good"
    if count >= 3:
        return "medium"
    return "low"


def _performance_band_for_speed(speed_kmh: float | None) -> str:
    if speed_kmh is None:
        return "basis"
    speed = float(speed_kmh)
    if speed >= 500.0:
        return "elite"
    if speed >= 430.0:
        return "schnell"
    if speed >= 350.0:
        return "aufbau"
    return "basis"


def _build_jumper_summary(*, jumper_name: str, jumps: list[dict[str, Any]]) -> dict[str, Any]:
    if not jumps:
        return {"available": False, "reason": "Keine Sprünge vorhanden."}

    rows = _sort_by_t0_desc(jumps)
    signature = _jumper_summary_signature(rows)
    cache_key = jumper_name.casefold()
    cached = _JUMPER_SUMMARY_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]

    reports: list[dict[str, Any]] = []
    for row in rows:
        report = get_jump_report(str(row.get("jump_id")))
        if report is not None:
            reports.append(report)

    if not reports:
        result = {"available": False, "reason": "Keine auswertbaren Sprünge vorhanden."}
        _JUMPER_SUMMARY_CACHE[cache_key] = (signature, result)
        return result

    reports = sorted(reports, key=lambda item: _t0_sort_key(item.get("jump", {}).get("t0_utc")))
    marco_profile = _get_marco_top15_profile(limit=15)
    records = [_build_jumper_record(item, marco_profile=marco_profile) for item in reports]
    records = [item for item in records if item]
    if not records:
        result = {"available": False, "reason": "Keine auswertbaren Sprünge vorhanden."}
        _JUMPER_SUMMARY_CACHE[cache_key] = (signature, result)
        return result

    learning_records = [item for item in records if bool(item.get("learning_eligible"))]
    best_record = (
        max(
            learning_records,
            key=lambda item: _to_float(item.get("rule_score_kmh")) or float("-inf"),
        )
        if learning_records
        else None
    )
    trend_rows = _build_jumper_trend_rows(learning_records)
    better_rows = [row for row in trend_rows if row.get("status") == "besser"]
    worse_rows = [row for row in trend_rows if row.get("status") == "schlechter"]

    if not trend_rows:
        trend_summary = "Zu wenig Sprünge für einen belastbaren Verlauf (mindestens 4 nötig)."
    elif worse_rows and better_rows:
        trend_summary = "Gemischter Verlauf: einige Punkte wurden besser, andere zuletzt schlechter."
    elif worse_rows:
        trend_summary = "Zuletzt eher rückläufig: zentrale Punkte sind aktuell schlechter als früher."
    elif better_rows:
        trend_summary = "Positiver Verlauf: zentrale Punkte sind zuletzt besser geworden."
    else:
        trend_summary = "In Summe stabiler Verlauf ohne klare Trendverschiebung."

    improved_points = [row["trend_text"] for row in better_rows[:4]]
    worse_points = [row["trend_text"] for row in worse_rows[:4]]
    earlier_better_points = [row["earlier_better_text"] for row in worse_rows[:4]]
    focus_actions = _build_jumper_focus_actions(worse_rows=worse_rows)
    performance_profile = _build_performance_profile(records)
    stability_reference = _build_jumper_stability_reference(learning_records)
    stability_reference["performance_profile"] = performance_profile
    timing_reference = _build_jumper_timing_reference(learning_records)
    stability_reference["timing_reference"] = timing_reference
    tip_effect_profile = _build_tip_effect_profile(learning_records, performance_profile=performance_profile)
    feedback_training_profile = build_feedback_training_profile(learning_records)

    result = {
        "available": True,
        "jumper_name": jumper_name,
        "jump_count": len(records),
        "learning_jump_count": len(learning_records),
        "excluded_learning_jump_count": len(records) - len(learning_records),
        "best_speed_kmh": None if best_record is None else best_record.get("rule_score_kmh"),
        "best_file_name": None if best_record is None else best_record.get("file_name"),
        "best_t0_utc": None if best_record is None else best_record.get("t0_utc"),
        "best_score_source": "rule_based_3s_score",
        "trend_summary": trend_summary,
        "trend_rows": trend_rows,
        "improved_points": improved_points,
        "worse_points": worse_points,
        "earlier_better_points": earlier_better_points,
        "focus_actions": focus_actions,
        "performance_profile": performance_profile,
        "timing_reference": timing_reference,
        "stability_reference": stability_reference,
        "tip_effect_profile": tip_effect_profile,
        "feedback_training_profile": feedback_training_profile,
    }
    _JUMPER_SUMMARY_CACHE[cache_key] = (signature, result)
    return result


def _empty_jumper_timing_reference(
    reason: str,
    *,
    maturity: str = "off",
    usable_count: int = 0,
    basis_count: int = 0,
    source: str = "",
    best_reference_kmh: float | None = None,
    top3_avg_kmh: float | None = None,
    max_timing_spread_s: float | None = None,
    maturity_reasons: list[str] | None = None,
) -> dict[str, Any]:
    maturity_key = str(maturity or "off").strip().lower()
    if maturity_key not in _TIMING_REFERENCE_MATURITY_LABELS:
        maturity_key = "off"
    return {
        "available": False,
        "reason": reason,
        "confidence": "unavailable",
        "basis_count": int(basis_count),
        "usable_count": int(usable_count),
        "source": source,
        "maturity": maturity_key,
        "maturity_label": _TIMING_REFERENCE_MATURITY_LABELS[maturity_key],
        "maturity_reasons": list(maturity_reasons or []),
        "best_reference_kmh": _round_float(best_reference_kmh, 1),
        "top3_avg_kmh": _round_float(top3_avg_kmh, 1),
        "max_timing_spread_s": _round_float(max_timing_spread_s, 1),
        "anchors": {},
        "summary_lines": [],
    }


def _timing_anchor_spread_s(anchors: dict[str, dict[str, Any]], key: str) -> float | None:
    raw = anchors.get(key)
    if not isinstance(raw, dict):
        return None
    low = _to_float(raw.get("low"))
    high = _to_float(raw.get("high"))
    if low is None or high is None:
        return None
    return max(0.0, float(high) - float(low))


def _timing_reference_maturity(
    *,
    usable_rows: list[dict[str, Any]],
    basis_rows: list[dict[str, Any]],
    anchors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    speed_values = [
        float(value)
        for row in usable_rows
        for value in [_to_float(row.get("rule_score_kmh"))]
        if value is not None
    ]
    best_reference = max(speed_values) if speed_values else None
    top3_avg = _top_avg(speed_values, 3) if speed_values else None
    angle_82_spread = _timing_anchor_spread_s(anchors, "angle_82_time_s")
    best_start_spread = _timing_anchor_spread_s(anchors, "best_3s_start_s")
    spread_values = [value for value in [angle_82_spread, best_start_spread] if value is not None]
    max_spread = max(spread_values) if spread_values else None

    if len(usable_rows) < _TIMING_REFERENCE_MIN_USABLE_JUMPS:
        return {
            "state": "off",
            "reason": (
                "Mindestens "
                f"{_TIMING_REFERENCE_MIN_USABLE_JUMPS} verwertbare Spruenge noetig; "
                f"aktuell {len(usable_rows)}."
            ),
            "reasons": ["Zu wenige verwertbare Spruenge."],
            "best_reference_kmh": best_reference,
            "top3_avg_kmh": top3_avg,
            "max_timing_spread_s": max_spread,
        }

    reasons: list[str] = []
    if len(basis_rows) < _TIMING_REFERENCE_MIN_BASIS_JUMPS:
        reasons.append(
            f"Nur {len(basis_rows)} kontrollierte Basis-Spruenge; "
            f"mindestens {_TIMING_REFERENCE_MIN_BASIS_JUMPS} noetig."
        )
    if best_reference is None or best_reference < _TIMING_REFERENCE_MIN_BEST_KMH:
        reasons.append(
            f"Beste Referenz unter {_TIMING_REFERENCE_MIN_BEST_KMH:.0f} km/h."
        )
    if top3_avg is None or top3_avg < _TIMING_REFERENCE_MIN_TOP3_AVG_KMH:
        reasons.append(
            f"Top-3-Schnitt unter {_TIMING_REFERENCE_MIN_TOP3_AVG_KMH:.0f} km/h."
        )
    if angle_82_spread is None:
        reasons.append("82-Grad-Timing noch nicht stabil genug erfasst.")
    elif angle_82_spread > _TIMING_REFERENCE_MAX_ANCHOR_SPREAD_S:
        reasons.append(
            "82-Grad-Timing streut zu stark "
            f"({angle_82_spread:.1f}s > {_TIMING_REFERENCE_MAX_ANCHOR_SPREAD_S:.1f}s)."
        )
    if best_start_spread is None:
        reasons.append("Start des besten 3s-Fensters noch nicht stabil genug erfasst.")
    elif best_start_spread > _TIMING_REFERENCE_MAX_ANCHOR_SPREAD_S:
        reasons.append(
            "Start des besten 3s-Fensters streut zu stark "
            f"({best_start_spread:.1f}s > {_TIMING_REFERENCE_MAX_ANCHOR_SPREAD_S:.1f}s)."
        )

    if reasons:
        return {
            "state": "soft",
            "reason": "Persoenliches Timing noch nicht belastbar: " + " ".join(reasons),
            "reasons": reasons,
            "best_reference_kmh": best_reference,
            "top3_avg_kmh": top3_avg,
            "max_timing_spread_s": max_spread,
        }
    return {
        "state": "active",
        "reason": None,
        "reasons": [],
        "best_reference_kmh": best_reference,
        "top3_avg_kmh": top3_avg,
        "max_timing_spread_s": max_spread,
    }


def _build_jumper_timing_reference(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return _empty_jumper_timing_reference("Keine Spruenge vorhanden.")

    usable = [
        row
        for row in records
        if bool(row.get("learning_eligible", True))
        and not bool(row.get("analysis_blocked"))
        and _to_float(row.get("best_3s_start_s")) is not None
        and _to_float(row.get("best_3s_end_s")) is not None
        and (
            _to_float(row.get("build_coverage")) is None
            or float(_to_float(row.get("build_coverage")) or 0.0) >= 0.70
        )
        and (
            _to_float(row.get("hot_coverage")) is None
            or float(_to_float(row.get("hot_coverage")) or 0.0) >= 0.70
        )
    ]
    if len(usable) < _TIMING_REFERENCE_MIN_USABLE_JUMPS:
        return _empty_jumper_timing_reference(
            (
                "Zu wenige verwertbare Spruenge fuer eine persoenliche Timing-Referenz "
                f"(mindestens {_TIMING_REFERENCE_MIN_USABLE_JUMPS} noetig)."
            ),
            maturity="off",
            usable_count=len(usable),
        )

    stable_rows = [
        row
        for row in usable
        if _is_stable_geometry(row)
        and int(row.get("stability_score") or 0) >= 62
        and int(row.get("hot_score") or 0) >= 58
        and int(row.get("build_score") or 0) >= 58
    ]
    if len(stable_rows) >= 2:
        basis_rows = stable_rows
        source = "stable_control"
    else:
        ranked = sorted(
            usable,
            key=lambda row: (
                _control_reference_score(row),
                float(_to_float(row.get("rule_score_kmh")) or 0.0),
            ),
            reverse=True,
        )
        take_count = min(len(ranked), max(2, min(8, (len(ranked) + 1) // 2)))
        basis_rows = ranked[:take_count]
        source = "best_available"

    anchor_specs = [
        ("angle_60_time_s", "60 Grad"),
        ("angle_75_time_s", "75 Grad"),
        ("angle_82_time_s", "82 Grad"),
        ("angle_85_time_s", "85 Grad"),
        ("best_3s_start_s", "Start bestes 3s-Fenster"),
        ("best_3s_end_s", "Ende bestes 3s-Fenster"),
        ("vvert_peak_time_s", "vVert-Peak"),
        ("decel_start_s", "Decel/Recovery"),
    ]
    anchors: dict[str, dict[str, Any]] = {}
    for key, label in anchor_specs:
        stats = _metric_stats(basis_rows, key)
        if stats is None:
            continue
        anchors[key] = {
            "label": label,
            "low": stats["low"],
            "high": stats["high"],
            "median": stats["median"],
        }

    if not anchors.get("best_3s_start_s"):
        return _empty_jumper_timing_reference(
            "Timing vorhanden, aber kein belastbarer Startbereich fuer das 3s-Fenster.",
            maturity="soft",
            usable_count=len(usable),
            basis_count=len(basis_rows),
            source=source,
        )

    maturity = _timing_reference_maturity(
        usable_rows=usable,
        basis_rows=basis_rows,
        anchors=anchors,
    )
    if maturity.get("state") != "active":
        return _empty_jumper_timing_reference(
            str(maturity.get("reason") or "Persoenliches Timing noch nicht belastbar."),
            maturity=str(maturity.get("state") or "soft"),
            usable_count=len(usable),
            basis_count=len(basis_rows),
            source=source,
            best_reference_kmh=_to_float(maturity.get("best_reference_kmh")),
            top3_avg_kmh=_to_float(maturity.get("top3_avg_kmh")),
            max_timing_spread_s=_to_float(maturity.get("max_timing_spread_s")),
            maturity_reasons=[
                str(item)
                for item in (maturity.get("reasons") or [])
                if str(item or "").strip()
            ],
        )

    basis_count = len(basis_rows)
    if basis_count >= 6 and source == "stable_control":
        confidence = "stable"
    elif basis_count >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    summary_lines = _build_jumper_timing_summary_lines(anchors)
    if not summary_lines:
        summary_lines = [
            "Persoenliches Timing vorhanden, aber noch ohne engen gemeinsamen Korridor."
        ]

    return {
        "available": True,
        "reason": None,
        "confidence": confidence,
        "basis_count": basis_count,
        "usable_count": len(usable),
        "source": source,
        "maturity": "active",
        "maturity_label": _TIMING_REFERENCE_MATURITY_LABELS["active"],
        "maturity_reasons": [],
        "best_reference_kmh": _round_float(maturity.get("best_reference_kmh"), 1),
        "top3_avg_kmh": _round_float(maturity.get("top3_avg_kmh"), 1),
        "max_timing_spread_s": _round_float(maturity.get("max_timing_spread_s"), 1),
        "anchors": anchors,
        "summary_lines": summary_lines,
    }


def _fmt_timing_band(stats: dict[str, Any]) -> str:
    low = _to_float(stats.get("low"))
    high = _to_float(stats.get("high"))
    if low is None or high is None:
        return "-"
    return f"+{_fmt_band(float(low), float(high), 1)}s"


def _build_jumper_timing_summary_lines(anchors: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    best_start = anchors.get("best_3s_start_s")
    best_end = anchors.get("best_3s_end_s")
    angle_82 = anchors.get("angle_82_time_s")
    angle_85 = anchors.get("angle_85_time_s")
    vpeak = anchors.get("vvert_peak_time_s")
    decel = anchors.get("decel_start_s")

    if angle_82 is not None and best_start is not None:
        lines.append(
            f"Persoenliches Timing: 82 Grad meist bei {_fmt_timing_band(angle_82)}; "
            f"bestes 3s-Fenster startet meist {_fmt_timing_band(best_start)}."
        )
    elif best_start is not None:
        lines.append(
            f"Bestes 3s-Fenster startet in stabileren Spruengen meist {_fmt_timing_band(best_start)}."
        )

    if angle_85 is not None:
        lines.append(
            f"Oberer Winkelbereich: 85 Grad wird meist um {_fmt_timing_band(angle_85)} erreicht."
        )

    if decel is not None:
        lines.append(
            f"Reserve: Decel/Recovery beginnt meist {_fmt_timing_band(decel)}; "
            "faellt vHor vorher ab, ist das ein frueher Stabilitaetshinweis."
        )
    elif vpeak is not None and best_end is not None:
        lines.append(
            f"Peak-Timing: vVert-Peak meist {_fmt_timing_band(vpeak)}, "
            f"3s-Fenster endet meist {_fmt_timing_band(best_end)}."
        )

    return lines[:3]


def _build_jumpers_overview(jumpers: list[str]) -> list[dict[str, Any]]:
    overview: list[dict[str, Any]] = []
    for jumper_name in jumpers:
        jump_rows = _sort_by_t0_desc(list_jumps_for_jumper(jumper_name))
        if not jump_rows:
            continue
        summary = _build_jumper_summary(jumper_name=jumper_name, jumps=jump_rows)
        stability_ref = summary.get("stability_reference", {}) if summary else {}
        performance_profile = summary.get("performance_profile", {}) if isinstance(summary, dict) else {}
        latest_t0 = jump_rows[0].get("t0_utc")
        stable_count = int(stability_ref.get("stable_count") or 0)
        unstable_count = int(stability_ref.get("unstable_count") or 0)
        focus_actions = summary.get("focus_actions") if isinstance(summary, dict) else []
        focus_short = "-"
        if isinstance(focus_actions, list):
            first_focus = next((item for item in focus_actions if isinstance(item, str) and item.strip()), None)
            if first_focus:
                focus_short = first_focus
        if focus_short == "-":
            focus_short = str(stability_ref.get("unstable_short") or "-")

        overview.append(
            {
                "jumper_name": jumper_name,
                "jump_count": int(summary.get("jump_count") or len(jump_rows)),
                "latest_t0_utc": latest_t0,
                "best_speed_kmh": summary.get("best_speed_kmh"),
                "stable_short": str(stability_ref.get("stable_short") or "-"),
                "unstable_short": str(stability_ref.get("unstable_short") or "-"),
                "stable_count": stable_count,
                "unstable_count": unstable_count,
                "trend_summary": str(summary.get("trend_summary") or "-"),
                "focus_short": focus_short,
                "performance_profile": performance_profile,
                "performance_label": (
                    str(performance_profile.get("performance_band_label") or "-")
                    if isinstance(performance_profile, dict) and performance_profile.get("available")
                    else "-"
                ),
                "performance_summary": (
                    str(performance_profile.get("summary") or "-")
                    if isinstance(performance_profile, dict) and performance_profile.get("available")
                    else "-"
                ),
                "simple_status": _jumper_overview_simple_status(
                    stable_count=stable_count,
                    unstable_count=unstable_count,
                ),
            }
        )
    return sorted(overview, key=lambda row: _t0_sort_key(row.get("latest_t0_utc")), reverse=True)


def _build_jumper_stability_reference(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "available": False,
            "reason": "Keine Sprünge vorhanden.",
            "stable_lines": [],
            "unstable_lines": [],
            "stable_short": "-",
            "unstable_short": "-",
            "stable_count": 0,
            "unstable_count": 0,
            "bands": {"stable": {}, "unstable": {}},
            "thresholds": {},
            "capability_profile": {"mode": "basis", "label": "Basis", "text": "Zu wenige Daten für eine persönliche Einstufung."},
            "asymmetry_profile": {"available": False, "reason": "Zu wenige Sprünge für eine Seitlinien-Analyse."},
        }

    raw_usable = [
        row
        for row in records
        if bool(row.get("learning_eligible", True))
        and not bool(row.get("analysis_blocked"))
        and _to_float(row.get("vvert_10s")) is not None
        and _to_float(row.get("vvert_20s")) is not None
        and _to_float(row.get("angle_20s")) is not None
        and (
            _to_float(row.get("build_coverage")) is None
            or _to_float(row.get("build_coverage")) >= 0.75
        )
        and (
            _to_float(row.get("hot_coverage")) is None
            or _to_float(row.get("hot_coverage")) >= 0.75
        )
    ]
    usable = [
        row
        for row in raw_usable
        if (
            (_to_float(row.get("gain_10_20")) is not None)
            and 40.0 <= float(_to_float(row.get("gain_10_20"))) <= 220.0
            and 170.0 <= float(_to_float(row.get("vvert_10s"))) <= 380.0
            and 250.0 <= float(_to_float(row.get("vvert_20s"))) <= 470.0
            and 60.0 <= float(_to_float(row.get("angle_20s"))) <= 88.8
        )
    ]
    if len(usable) < 5:
        usable = raw_usable
    if len(usable) < 5:
        return {
            "available": False,
            "reason": "Zu wenige verwertbare Sprünge für eine persönliche Stabilitäts-Referenz.",
            "stable_lines": [],
            "unstable_lines": [],
            "stable_short": "-",
            "unstable_short": "-",
            "stable_count": 0,
            "unstable_count": 0,
            "bands": {"stable": {}, "unstable": {}},
            "thresholds": {},
            "capability_profile": {"mode": "basis", "label": "Basis", "text": "Zu wenige Daten für eine persönliche Einstufung."},
            "asymmetry_profile": {"available": False, "reason": "Zu wenige Sprünge für eine Seitlinien-Analyse."},
        }

    stable_rows = [
        row
        for row in usable
        if (int(row.get("stability_score") or 0) >= 70)
        and (int(row.get("hot_score") or 0) >= 65)
        and (int(row.get("build_score") or 0) >= 65)
        and _is_stable_geometry(row)
    ]
    if len(stable_rows) < 2:
        stable_rows = [
            row
            for row in usable
            if (int(row.get("stability_score") or 0) >= 62) and (int(row.get("hot_score") or 0) >= 58)
            and _is_stable_geometry(row)
        ]
    if len(stable_rows) < 2:
        stable_rows = []

    unstable_rows = [
        row
        for row in usable
        if (int(row.get("stability_score") or 100) < 55)
        or (int(row.get("hot_score") or 100) < 55)
        or ((_to_float(row.get("vhor_min_20_25")) is not None) and (_to_float(row.get("vhor_min_20_25")) < 25.0))
        or ((_to_float(row.get("angle_turns_20_25")) is not None) and (_to_float(row.get("angle_turns_20_25")) >= 5.0))
    ]
    if len(unstable_rows) < 2:
        unstable_rows = [
            row
            for row in usable
            if (int(row.get("stability_score") or 100) < 62)
            or (int(row.get("hot_score") or 100) < 62)
        ]
    stable_row_ids = {id(row) for row in stable_rows}
    unstable_rows = [row for row in unstable_rows if id(row) not in stable_row_ids]
    if len(unstable_rows) < 2:
        unstable_rows = []

    stable_v10_stats = _metric_stats(stable_rows, "vvert_10s")
    stable_v15_stats = _metric_stats(stable_rows, "vvert_15s")
    stable_v20_stats = _metric_stats(stable_rows, "vvert_20s")
    stable_a10_stats = _metric_stats(stable_rows, "angle_10s")
    stable_a15_stats = _metric_stats(stable_rows, "angle_15s")
    stable_a20_stats = _metric_stats(stable_rows, "angle_20s")
    stable_gain_stats = _metric_stats(stable_rows, "gain_10_20")
    stable_gain_10_15_stats = _metric_stats(stable_rows, "gain_10_15")
    stable_gain_15_20_stats = _metric_stats(stable_rows, "gain_15_20")
    stable_vhor_stats = _metric_stats(stable_rows, "vhor_min_20_25")
    stable_turns_stats = _metric_stats(stable_rows, "angle_turns_20_25")
    stable_lat_abs_stats = _metric_stats(stable_rows, "lateral_hot_abs_mean_kmh")
    stable_lat_heading_stats = _metric_stats(stable_rows, "lateral_hot_heading_rate_rms_dps")
    stable_lat_alat_stats = _metric_stats(stable_rows, "lateral_hot_alat_peak_mps2")
    stable_lat_turn_stats = _metric_stats(stable_rows, "lateral_hot_sign_changes")

    unstable_a20_stats = _metric_stats(unstable_rows, "angle_20s")
    unstable_vhor_stats = _metric_stats(unstable_rows, "vhor_min_20_25")
    unstable_turns_stats = _metric_stats(unstable_rows, "angle_turns_20_25")
    unstable_gain_stats = _metric_stats(unstable_rows, "gain_10_20")
    unstable_gain_10_15_stats = _metric_stats(unstable_rows, "gain_10_15")
    unstable_gain_15_20_stats = _metric_stats(unstable_rows, "gain_15_20")
    unstable_lat_abs_stats = _metric_stats(unstable_rows, "lateral_hot_abs_mean_kmh")
    unstable_lat_heading_stats = _metric_stats(unstable_rows, "lateral_hot_heading_rate_rms_dps")
    unstable_lat_alat_stats = _metric_stats(unstable_rows, "lateral_hot_alat_peak_mps2")
    unstable_lat_turn_stats = _metric_stats(unstable_rows, "lateral_hot_sign_changes")

    stable_lines: list[str] = []
    for label, speed_key, angle_key in [
        ("+10s", "vvert_10s", "angle_10s"),
        ("+15s", "vvert_15s", "angle_15s"),
        ("+20s", "vvert_20s", "angle_20s"),
    ]:
        speed_stats = _metric_stats(stable_rows, speed_key)
        angle_stats = _metric_stats(stable_rows, angle_key)
        if speed_stats is None or angle_stats is None:
            continue
        stable_lines.append(
            f"{label}: vVert meist {_fmt_band(speed_stats['low'], speed_stats['high'], 0)} km/h, "
            f"Winkel {_fmt_band(angle_stats['low'], angle_stats['high'], 1)} Grad."
        )

    stable_vhor = stable_vhor_stats
    stable_turns = stable_turns_stats
    if stable_vhor is not None and stable_turns is not None:
        stable_lines.append(
            "Hot-Zone stabil bei "
            f"vHor-Min 20-25s meist >= {stable_vhor['low']:.1f} km/h "
            f"und Korrekturen meist <= {int(round(stable_turns['high']))}."
        )
    if stable_lat_abs_stats is not None and stable_lat_heading_stats is not None:
        stable_lines.append(
            "Seitliche Linie in stabilen Sprüngen: "
            f"|Seitbewegung| in der Hot-Zone meist {_fmt_band(stable_lat_abs_stats['low'], stable_lat_abs_stats['high'], 1)} km/h, "
            f"Richtungswechsel meist <= {int(round((stable_lat_turn_stats or {'high': 2.0})['high']))}."
        )

    unstable_lines: list[str] = []
    stable_angle20 = stable_a20_stats
    unstable_angle20 = unstable_a20_stats
    angle_limit: float | None = None
    if stable_angle20 is not None and unstable_angle20 is not None:
        angle_limit = max(stable_angle20["high"], stable_angle20["median"] + 0.6)
        if unstable_angle20["median"] >= angle_limit - 0.4:
            unstable_lines.append(
                f"Ab +20s wird es oft unruhig, wenn der Winkel über etwa {angle_limit:.1f} Grad geht."
            )

    unstable_vhor = unstable_vhor_stats
    if stable_vhor is not None and unstable_vhor is not None:
        floor = stable_vhor["low"]
        if unstable_vhor["median"] <= floor + 1.0:
            unstable_lines.append(
                f"Wird oft instabil, wenn vHor-Min im Segment 20-25s unter ca. {floor:.1f} km/h fällt."
            )
    if unstable_vhor is not None and unstable_vhor["median"] < 25.0:
        unstable_lines.append(
            f"In unruhigen Sprüngen liegt vHor-Min 20-25s oft nur bei {_fmt_band(unstable_vhor['low'], unstable_vhor['high'], 1)} km/h."
        )

    unstable_turns = unstable_turns_stats
    turns_limit: int | None = None
    if stable_turns is not None and unstable_turns is not None:
        turns_limit = max(3, int(round(stable_turns["high"] + 1.0)))
        if unstable_turns["median"] >= turns_limit - 0.5:
            unstable_lines.append(
                f"Unruhe steigt deutlich, wenn im Segment 20-25s etwa {turns_limit}+ Richtungswechsel auftreten."
            )
    if unstable_turns is not None and unstable_turns["median"] >= 3.0:
        unstable_lines.append(
            f"In unruhigen Sprüngen gibt es im Segment 20-25s meist {_fmt_band(unstable_turns['low'], unstable_turns['high'], 0)} Richtungswechsel."
        )
    if (
        stable_lat_abs_stats is not None
        and unstable_lat_abs_stats is not None
        and unstable_lat_abs_stats["median"] > (stable_lat_abs_stats["high"] + 1.2)
    ):
        unstable_lines.append(
            "In unruhigen Sprüngen ist die Seitbewegung in der Hot-Zone klar höher als in deinen stabilen Sprüngen."
        )
    if (
        stable_lat_heading_stats is not None
        and unstable_lat_heading_stats is not None
        and unstable_lat_heading_stats["median"] > (stable_lat_heading_stats["high"] + 1.2)
    ):
        unstable_lines.append(
            "In unruhigen Sprüngen drehst du in der Hot-Zone häufiger die Richtung. Das deutet auf späte Nachkorrekturen."
        )

    stable_gain = stable_gain_stats
    unstable_gain = unstable_gain_stats
    if stable_gain is not None and unstable_gain is not None and unstable_gain["median"] + 8.0 < stable_gain["median"]:
        unstable_lines.append(
            f"Wenn der Aufbau +10 bis +20s unter etwa {stable_gain['low']:.1f} km/h bleibt, wird die Linie häufig später unruhig."
        )
    if unstable_angle20 is not None and unstable_angle20["median"] >= 85.0:
        unstable_lines.append(
            f"Unruhige Sprünge laufen häufig mit +20s-Winkeln um {_fmt_band(unstable_angle20['low'], unstable_angle20['high'], 1)} Grad."
        )

    asymmetry_profile = _build_lateral_asymmetry_profile(usable)
    if asymmetry_profile.get("available"):
        asym_text = str(asymmetry_profile.get("text") or "").strip()
        if asym_text:
            unstable_lines.append(asym_text)

    if not stable_lines and stable_rows:
        stable_lines.append("Stabile Sprünge vorhanden, aber noch zu wenig gemeinsame Fixpunkte für einen engen Korridor.")
    if not unstable_lines and unstable_rows:
        unstable_lines.append("Noch kein eindeutiger Instabilitäts-Trigger; weiter sammeln für schärfere Grenzen.")

    stable_short = stable_lines[0] if stable_lines else "Noch keine stabile Referenz."
    unstable_short = unstable_lines[0] if unstable_lines else "Noch kein klarer Instabilitäts-Trigger."
    if len(stable_lines) > 1:
        stable_short = f"{stable_short} | {stable_lines[1]}"

    recent_window = min(8, len(usable))
    recent_rows = usable[-recent_window:] if recent_window > 0 else []
    stable_recent_hits = 0
    for row in recent_rows:
        if (
            _is_stable_geometry(row)
            and int(row.get("stability_score") or 0) >= 62
            and int(row.get("hot_score") or 0) >= 58
        ):
            stable_recent_hits += 1
    stable_ratio = (stable_recent_hits / recent_window) if recent_window > 0 else 0.0
    stable_ratio_pct = stable_ratio * 100.0
    capability_mode = "build"
    if recent_window < 4:
        capability_mode = "basis"
    elif stable_ratio < 0.45:
        capability_mode = "safe"
    elif stable_ratio >= 0.72:
        capability_mode = "push"
    capability_label_map = {
        "basis": "Basis",
        "safe": "Sicher",
        "build": "Aufbau",
        "push": "Push",
    }
    if capability_mode == "push":
        capability_text = (
            f"Letzte {recent_window} Sprünge: {stable_recent_hits}/{recent_window} stabil "
            f"({stable_ratio_pct:.0f}%). Du kannst vorsichtig in den oberen Zielbereich gehen."
        )
    elif capability_mode == "safe":
        capability_text = (
            f"Letzte {recent_window} Sprünge: {stable_recent_hits}/{recent_window} stabil "
            f"({stable_ratio_pct:.0f}%). Erst Stabilität sichern, dann Tempo pushen."
        )
    elif capability_mode == "basis":
        capability_text = "Zu wenige aktuelle Sprünge für eine sichere Einstufung."
    else:
        capability_text = (
            f"Letzte {recent_window} Sprünge: {stable_recent_hits}/{recent_window} stabil "
            f"({stable_ratio_pct:.0f}%). Stabilität weiter aufbauen, dann schrittweise pushen."
        )
    capability_profile = {
        "mode": capability_mode,
        "label": capability_label_map.get(capability_mode, "Aufbau"),
        "recent_window": recent_window,
        "stable_hits": stable_recent_hits,
        "stable_ratio_pct": round(stable_ratio_pct, 1),
        "text": capability_text,
    }

    bands = {
        "stable": {
            "vvert_10s": stable_v10_stats,
            "vvert_15s": stable_v15_stats,
            "vvert_20s": stable_v20_stats,
            "angle_10s": stable_a10_stats,
            "angle_15s": stable_a15_stats,
            "angle_20s": stable_a20_stats,
            "gain_10_20": stable_gain_stats,
            "gain_10_15": stable_gain_10_15_stats,
            "gain_15_20": stable_gain_15_20_stats,
            "vhor_min_20_25": stable_vhor_stats,
            "angle_turns_20_25": stable_turns_stats,
            "lateral_hot_abs_mean_kmh": stable_lat_abs_stats,
            "lateral_hot_heading_rate_rms_dps": stable_lat_heading_stats,
            "lateral_hot_alat_peak_mps2": stable_lat_alat_stats,
            "lateral_hot_sign_changes": stable_lat_turn_stats,
        },
        "unstable": {
            "angle_20s": unstable_a20_stats,
            "gain_10_20": unstable_gain_stats,
            "gain_10_15": unstable_gain_10_15_stats,
            "gain_15_20": unstable_gain_15_20_stats,
            "vhor_min_20_25": unstable_vhor_stats,
            "angle_turns_20_25": unstable_turns_stats,
            "lateral_hot_abs_mean_kmh": unstable_lat_abs_stats,
            "lateral_hot_heading_rate_rms_dps": unstable_lat_heading_stats,
            "lateral_hot_alat_peak_mps2": unstable_lat_alat_stats,
            "lateral_hot_sign_changes": unstable_lat_turn_stats,
        },
    }
    thresholds = {
        "angle_20_target_low": None if stable_a20_stats is None else stable_a20_stats["low"],
        "angle_20_target_high": None if stable_a20_stats is None else stable_a20_stats["high"],
        "angle_20_risk_above": angle_limit,
        "phase_0_10_angle_low": None if stable_a10_stats is None else stable_a10_stats["low"],
        "phase_0_10_angle_high": None if stable_a10_stats is None else stable_a10_stats["high"],
        "vhor_min_20_25_floor": None if stable_vhor_stats is None else stable_vhor_stats["low"],
        "angle_turns_20_25_max": None if stable_turns_stats is None else int(round(stable_turns_stats["high"])),
        "gain_10_20_target_low": None if stable_gain_stats is None else stable_gain_stats["low"],
        "gain_10_20_target_high": None if stable_gain_stats is None else stable_gain_stats["high"],
        "vvert_10_target_low": None if stable_v10_stats is None else stable_v10_stats["low"],
        "vvert_10_target_high": None if stable_v10_stats is None else stable_v10_stats["high"],
        "phase_0_10_vvert_low": None if stable_v10_stats is None else stable_v10_stats["low"],
        "phase_0_10_vvert_high": None if stable_v10_stats is None else stable_v10_stats["high"],
        "phase_10_15_gain_low": None if stable_gain_10_15_stats is None else stable_gain_10_15_stats["low"],
        "phase_10_15_gain_high": None if stable_gain_10_15_stats is None else stable_gain_10_15_stats["high"],
        "phase_15_20_gain_low": None if stable_gain_15_20_stats is None else stable_gain_15_20_stats["low"],
        "phase_15_20_gain_high": None if stable_gain_15_20_stats is None else stable_gain_15_20_stats["high"],
        "phase_10_15_angle_low": None if stable_a15_stats is None else stable_a15_stats["low"],
        "phase_10_15_angle_high": None if stable_a15_stats is None else stable_a15_stats["high"],
        "phase_15_20_angle_low": None if stable_a20_stats is None else stable_a20_stats["low"],
        "phase_15_20_angle_high": None if stable_a20_stats is None else stable_a20_stats["high"],
        "lateral_hot_abs_target_high": None if stable_lat_abs_stats is None else stable_lat_abs_stats["high"],
        "lateral_hot_heading_rate_target_high": None if stable_lat_heading_stats is None else stable_lat_heading_stats["high"],
        "lateral_hot_alat_peak_target_high": None if stable_lat_alat_stats is None else stable_lat_alat_stats["high"],
        "lateral_hot_sign_changes_max": None if stable_lat_turn_stats is None else int(round(stable_lat_turn_stats["high"])),
    }

    return {
        "available": bool(stable_lines or unstable_lines),
        "reason": None,
        "stable_lines": stable_lines,
        "unstable_lines": unstable_lines,
        "stable_short": stable_short,
        "unstable_short": unstable_short,
        "stable_count": len(stable_rows),
        "unstable_count": len(unstable_rows),
        "bands": bands,
        "thresholds": thresholds,
        "capability_profile": capability_profile,
        "asymmetry_profile": asymmetry_profile,
    }


def _build_lateral_asymmetry_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    votes: list[str] = []
    cost_votes: list[str] = []
    for row in rows:
        abs_mean = _to_float(row.get("lateral_hot_abs_mean_kmh"))
        signed_mean = _to_float(row.get("lateral_hot_signed_mean_kmh"))
        if abs_mean is None or signed_mean is None:
            continue
        if abs_mean < 5.0 or abs(signed_mean) < 1.8:
            continue
        direction = "rechts" if signed_mean > 0 else "links"
        votes.append(direction)
        if bool(row.get("lateral_cost_event")):
            cost_votes.append(direction)

    if len(votes) < 4:
        return {
            "available": False,
            "reason": "Zu wenige Sprünge für eine belastbare Seitlinien-Asymmetrie.",
        }

    right_hits = sum(1 for item in votes if item == "rechts")
    left_hits = sum(1 for item in votes if item == "links")
    if right_hits >= left_hits:
        direction = "rechts"
        hits = right_hits
    else:
        direction = "links"
        hits = left_hits

    hit_ratio = hits / len(votes)
    if hit_ratio < 0.65:
        return {
            "available": False,
            "reason": "Keine klare Seitlinien-Richtung über mehrere Sprünge.",
            "sample_count": len(votes),
        }

    cost_hits = sum(1 for item in cost_votes if item == direction)
    cost_ratio = (cost_hits / len(cost_votes)) if cost_votes else 0.0
    direction_label = "Rechtsdrift" if direction == "rechts" else "Linksdrift"
    text = (
        f"Wiederkehrendes Muster: {direction_label} in der Hot-Zone "
        f"({hits}/{len(votes)} Sprünge, {hit_ratio * 100.0:.0f}%)."
    )
    return {
        "available": True,
        "direction": direction,
        "sample_count": len(votes),
        "hit_count": hits,
        "hit_ratio_pct": round(hit_ratio * 100.0, 1),
        "cost_event_count": len(cost_votes),
        "cost_event_direction_count": cost_hits,
        "cost_event_direction_ratio_pct": round(cost_ratio * 100.0, 1),
        "text": text,
        "note": "Ohne Windkorrektur als Musterhinweis werten, nicht als sichere Körperdiagnose.",
    }


def _control_reference_score(row: dict[str, Any]) -> float:
    parts = [
        _to_float(row.get("stability_score")),
        _to_float(row.get("hot_score")),
        _to_float(row.get("build_score")),
        _to_float(row.get("exit_score")),
    ]
    values = [float(item) for item in parts if item is not None]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _score_value_from_map(score_map: dict[str, int], *names: str) -> int | None:
    for name in names:
        if name in score_map:
            return score_map[name]
    normalized = {normalize_german_text(key): value for key, value in score_map.items()}
    for name in names:
        value = normalized.get(normalize_german_text(name))
        if value is not None:
            return value
    return None


def _average_optional_scores(values: list[int | None]) -> int | None:
    numeric = [int(value) for value in values if value is not None]
    if not numeric:
        return None
    return int(round(sum(numeric) / len(numeric)))


def _is_stable_geometry(row: dict[str, Any]) -> bool:
    vhor_min = _to_float(row.get("vhor_min_20_25"))
    turns = _to_float(row.get("angle_turns_20_25"))
    if vhor_min is not None and vhor_min < 25.0:
        return False
    if turns is not None and turns > 4.0:
        return False
    return True


def _metric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values: list[float] = []
    for row in rows:
        value = _to_float(row.get(key))
        if value is None:
            continue
        values.append(float(value))
    if len(values) < 2:
        return None
    values.sort()
    low = _percentile(values, 0.25) if len(values) >= 4 else values[0]
    high = _percentile(values, 0.75) if len(values) >= 4 else values[-1]
    return {
        "low": float(low),
        "high": float(high),
        "median": float(_percentile(values, 0.5)),
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    qq = max(0.0, min(1.0, float(q)))
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * qq
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    weight = pos - lo
    return float(values[lo] * (1.0 - weight) + values[hi] * weight)


def _fmt_band(low: float, high: float, decimals: int) -> str:
    if decimals <= 0:
        if round(low) == round(high):
            return f"{round(low):.0f}"
        return f"{round(low):.0f} bis {round(high):.0f}"
    if round(low, decimals) == round(high, decimals):
        return f"{low:.{decimals}f}"
    return f"{low:.{decimals}f} bis {high:.{decimals}f}"


def _chart_time_value_pairs(
    chart: dict[str, Any],
    value_key: str,
    *,
    end_s: float | None = None,
) -> list[tuple[float, float]]:
    time_values = chart.get("time_s", []) if isinstance(chart, dict) else []
    metric_values = chart.get(value_key, []) if isinstance(chart, dict) else []
    pairs: list[tuple[float, float]] = []
    for raw_t, raw_value in zip(time_values or [], metric_values or []):
        t = _to_float(raw_t)
        value = _to_float(raw_value)
        if t is None or value is None:
            continue
        if t < 0.0:
            continue
        if end_s is not None and t > end_s:
            continue
        pairs.append((float(t), float(value)))
    pairs.sort(key=lambda item: item[0])
    return pairs


def _first_angle_crossing_time(
    chart: dict[str, Any],
    threshold_deg: float,
    *,
    end_s: float | None = None,
) -> float | None:
    pairs = _chart_time_value_pairs(chart, "angle_deg", end_s=end_s)
    if not pairs:
        return None
    threshold = float(threshold_deg)
    first_t, first_angle = pairs[0]
    if first_angle >= threshold:
        return first_t

    prev_t, prev_angle = pairs[0]
    for t, angle in pairs[1:]:
        if prev_angle < threshold <= angle:
            delta = angle - prev_angle
            if abs(delta) <= 1e-9:
                return t
            ratio = (threshold - prev_angle) / delta
            ratio = max(0.0, min(1.0, ratio))
            return float(prev_t + ((t - prev_t) * ratio))
        prev_t, prev_angle = t, angle
    return None


def _vvert_peak_time(chart: dict[str, Any], *, end_s: float | None = None) -> float | None:
    pairs = _chart_time_value_pairs(chart, "vVert_kmh", end_s=end_s)
    if not pairs:
        return None
    return max(pairs, key=lambda item: item[1])[0]


def _build_jumper_record(
    report: dict[str, Any],
    *,
    marco_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    jump = report.get("jump", {})
    metrics = report.get("metrics", {})
    notes = report.get("notes", {})
    quality_flags = report.get("quality_flags", [])
    fixpoints = report.get("fixpoints", [])
    score_rows = _build_scorecard_rows(report, marco_profile=marco_profile)
    score_map = {str(row.get("name")): int(row.get("score", 0)) for row in score_rows}

    fp10 = _fixpoint_at(fixpoints, 10.0)
    fp15 = _fixpoint_at(fixpoints, 15.0)
    fp20 = _fixpoint_at(fixpoints, 20.0)
    snapshot = _scorecard_metric_snapshot(report)
    v10 = _to_float(snapshot.get("v10"))
    v15 = _to_float(fp15.get("vVert_kmh")) if fp15 else None
    v20 = _to_float(fp20.get("vVert_kmh")) if fp20 else None
    a10 = _to_float(fp10.get("angle_deg")) if fp10 else None
    a15 = _to_float(fp15.get("angle_deg")) if fp15 else None
    a20 = _to_float(snapshot.get("angle_20"))
    gain_10_20 = _to_float(snapshot.get("gain_10_20"))
    gain_10_15 = None if v10 is None or v15 is None else float(v15 - v10)
    gain_15_20 = None if v15 is None or v20 is None else float(v20 - v15)

    hold_400 = _to_float(snapshot.get("dur_400"))
    vhor_min_20_25 = _to_float(snapshot.get("vhor_min_20_25"))
    angle_turns_20_25 = _to_float(snapshot.get("angle_turns_20_25"))
    build_coverage = _to_float(snapshot.get("build_coverage"))
    hot_coverage = _to_float(snapshot.get("hot_coverage"))
    chart = report.get("chart_data", {})
    eval_end_s = effective_eval_window_end_s(metrics=metrics, notes=notes, chart_data=chart)
    lateral = analyze_lateral_dynamics(
        chart,
        eval_end_s=eval_end_s,
    )
    lateral_event = lateral.get("speed_cost_event", {}) if isinstance(lateral, dict) else {}
    lateral_high_speed = lateral.get("high_speed", {}) if isinstance(lateral, dict) else {}

    exit_score = _score_value_from_map(score_map, "Exit / Stabilisierung", "Exit")
    dive_score = _score_value_from_map(score_map, "Dive-Aufbau")
    main_accel_score = _score_value_from_map(score_map, "Hauptbeschleunigung", "Aufbau 10-20s")
    hot_build_score = _score_value_from_map(score_map, "Hot-Zone Aufbau", "Hot-Zone")
    max_speed_score = _score_value_from_map(score_map, "Max-Speed Fenster", "Stabilität / Kipp-Risiko")
    build_score = _average_optional_scores([dive_score, main_accel_score])
    hot_score = _average_optional_scores([hot_build_score, max_speed_score])
    stability_score = max_speed_score

    core_scores = [
        exit_score,
        dive_score,
        main_accel_score,
        hot_build_score,
        max_speed_score,
    ]
    core_score_values = [float(s) for s in core_scores if s is not None]
    overall_score = int(round(mean(core_score_values))) if core_score_values else None

    return {
        "jump_id": jump.get("jump_id"),
        "file_name": jump.get("file_name"),
        "jump_context": _normalize_jump_context(str(jump.get("jump_context") or "unknown")),
        "feedback_context": build_feedback_coaching_context(report),
        "t0_utc": jump.get("t0_utc"),
        "best_3s_kmh": _to_float(metrics.get("best_3s_vVert_kmh")),
        "best_3s_start_s": _to_float(metrics.get("best_3s_start_s")),
        "best_3s_end_s": _to_float(metrics.get("best_3s_end_s")),
        "rule_score_kmh": _to_float(metrics.get("rule_based_3s_score")),
        "rule_score_status": metrics.get("rule_score_status"),
        "learning_eligible": is_reference_eligible(
            metrics=metrics,
            notes=notes if isinstance(notes, dict) else {},
            quality_flags=quality_flags,
        ),
        "overall_score": overall_score,
        "exit_score": exit_score,
        "dive_score": dive_score,
        "main_accel_score": main_accel_score,
        "build_score": build_score,
        "hot_build_score": hot_build_score,
        "max_speed_score": max_speed_score,
        "hot_score": hot_score,
        "stability_score": stability_score,
        "vvert_10s": v10,
        "vvert_15s": v15,
        "vvert_20s": v20,
        "angle_10s": a10,
        "angle_15s": a15,
        "angle_20s": a20,
        "gain_10_20": gain_10_20,
        "gain_10_15": gain_10_15,
        "gain_15_20": gain_15_20,
        "angle_60_time_s": _first_angle_crossing_time(chart, 60.0, end_s=eval_end_s),
        "angle_75_time_s": _first_angle_crossing_time(chart, 75.0, end_s=eval_end_s),
        "angle_82_time_s": _first_angle_crossing_time(chart, 82.0, end_s=eval_end_s),
        "angle_85_time_s": _first_angle_crossing_time(chart, 85.0, end_s=eval_end_s),
        "vvert_peak_time_s": _vvert_peak_time(chart, end_s=eval_end_s),
        "decel_start_s": _to_float(notes.get("decel_start_s")),
        "curve_window_end_s": _to_float(notes.get("curve_window_end_s")),
        "eval_end_s": eval_end_s,
        "hold_400_s": hold_400,
        "vhor_min_20_25": vhor_min_20_25,
        "angle_turns_20_25": angle_turns_20_25,
        "lateral_hot_abs_mean_kmh": _to_float(lateral.get("hot_vlat_abs_mean_kmh")) if isinstance(lateral, dict) else None,
        "lateral_hot_signed_mean_kmh": _to_float(lateral.get("hot_vlat_signed_mean_kmh")) if isinstance(lateral, dict) else None,
        "lateral_hot_heading_rate_rms_dps": _to_float(lateral.get("hot_heading_rate_rms_dps")) if isinstance(lateral, dict) else None,
        "lateral_hot_alat_peak_mps2": _to_float(lateral.get("hot_alat_peak_mps2")) if isinstance(lateral, dict) else None,
        "lateral_hot_sign_changes": _to_float(lateral.get("hot_sign_changes")) if isinstance(lateral, dict) else None,
        "lateral_hot_direction": str(lateral.get("hot_direction") or "") if isinstance(lateral, dict) else "",
        "lateral_hot_pattern": str(lateral.get("hot_pattern") or "") if isinstance(lateral, dict) else "",
        "lateral_hot_cross_track_span_m": _to_float(lateral.get("hot_cross_track_span_m")) if isinstance(lateral, dict) else None,
        "lateral_cost_event": bool(lateral_event.get("available") and lateral_event.get("likely_speed_cost")),
        "lateral_cost_event_time_s": _to_float(lateral_event.get("t_peak_s")) if isinstance(lateral_event, dict) else None,
        "lateral_cost_event_direction": str(lateral_event.get("direction") or "") if isinstance(lateral_event, dict) else "",
        "high_speed_theta_std_deg": _to_float(lateral_high_speed.get("theta_std_deg")) if isinstance(lateral_high_speed, dict) else None,
        "high_speed_vlat_abs_mean_kmh": _to_float(lateral_high_speed.get("vlat_abs_mean_kmh")) if isinstance(lateral_high_speed, dict) else None,
        "high_speed_heading_rate_rms_dps": _to_float(lateral_high_speed.get("heading_rate_rms_dps")) if isinstance(lateral_high_speed, dict) else None,
        "build_coverage": build_coverage,
        "hot_coverage": hot_coverage,
        "quality_flags": quality_flags,
        "analysis_blocked": bool(notes.get("analysis_blocked")),
        "t0_review_required": bool(notes.get("t0_review_required")),
    }


def _build_jumper_trend_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) < 4:
        return []

    split = max(2, len(records) // 2)
    early = records[:split]
    recent = records[split:]
    if len(recent) < 2:
        return []

    specs = [
        {"key": "rule_score_kmh", "name": "Regel-Score", "unit": "km/h", "higher_is_better": True, "threshold": 2.5},
        {"key": "exit_score", "name": "Exit / Stabilisierung", "unit": "Score", "higher_is_better": True, "threshold": 5.0},
        {"key": "dive_score", "name": "Dive-Aufbau", "unit": "Score", "higher_is_better": True, "threshold": 5.0},
        {"key": "main_accel_score", "name": "Hauptbeschleunigung", "unit": "Score", "higher_is_better": True, "threshold": 5.0},
        {"key": "hot_build_score", "name": "Hot-Zone Aufbau", "unit": "Score", "higher_is_better": True, "threshold": 5.0},
        {"key": "max_speed_score", "name": "Max-Speed Fenster", "unit": "Score", "higher_is_better": True, "threshold": 5.0},
        {
            "key": "angle_turns_20_25",
            "name": "Korrekturen 20-25s",
            "unit": "Anzahl",
            "higher_is_better": False,
            "threshold": 0.8,
        },
        {
            "key": "lateral_hot_abs_mean_kmh",
            "name": "Seitbewegung Hot-Zone",
            "unit": "km/h",
            "higher_is_better": False,
            "threshold": 1.2,
        },
        {
            "key": "high_speed_theta_std_deg",
            "name": "Winkelruhe bei hohem Speed",
            "unit": "Grad",
            "higher_is_better": False,
            "threshold": 0.4,
        },
    ]

    out: list[dict[str, Any]] = []
    for spec in specs:
        early_avg = _mean_metric(early, spec["key"])
        recent_avg = _mean_metric(recent, spec["key"])
        if early_avg is None or recent_avg is None:
            continue
        delta = recent_avg - early_avg
        threshold = float(spec["threshold"])
        if abs(delta) < threshold:
            status = "stabil"
        else:
            is_better = delta > 0 if bool(spec["higher_is_better"]) else delta < 0
            status = "besser" if is_better else "schlechter"

        sign = "+" if delta > 0 else ""
        trend_text = (
            f"{spec['name']}: {status} ({early_avg:.1f} -> {recent_avg:.1f} {spec['unit']}, "
            f"Delta {sign}{delta:.1f})."
        )
        if status == "schlechter":
            earlier_better_text = (
                f"{spec['name']} war früher besser ({early_avg:.1f}) als zuletzt ({recent_avg:.1f})."
            )
        else:
            earlier_better_text = ""

        out.append(
            {
                "name": spec["name"],
                "status": status,
                "early_value": round(early_avg, 2),
                "recent_value": round(recent_avg, 2),
                "delta": round(delta, 2),
                "unit": spec["unit"],
                "trend_text": trend_text,
                "earlier_better_text": earlier_better_text,
            }
        )

    order = {"schlechter": 0, "besser": 1, "stabil": 2}
    out.sort(key=lambda row: order.get(str(row.get("status")), 3))
    return out


def _build_jumper_focus_actions(*, worse_rows: list[dict[str, Any]]) -> list[str]:
    if not worse_rows:
        return ["Kein klarer Rückschritt sichtbar. Fokus auf stabile Wiederholung der zuletzt guten Linie."]

    action_map = {
        "Top-Speed": "In der späten schnellen Phase länger stabil bleiben und den Ausstieg später setzen.",
        "Exit": "Die ersten Sekunden nach dem Exit ruhiger und konstanter aufbauen, ohne harte Gegenkorrektur.",
        "Exit / Stabilisierung": "Die ersten Sekunden nach dem Exit ruhiger und konstanter aufbauen, ohne harte Gegenkorrektur.",
        "Dive-Aufbau": "Von +3s bis +8s kontrolliert in den Zielwinkel gehen, ohne den Druck zu spät zu setzen.",
        "Hauptbeschleunigung": "Von +8s bis +15s den Speed mit stabilem Winkel aufbauen, statt nur steiler zu werden.",
        "Hot-Zone Aufbau": "Von +15s bis +22s früher in den Zielbereich kommen und die horizontale Reserve schützen.",
        "Max-Speed Fenster": "Im Max-Speed-Fenster den Winkel halten, vHor nicht abreißen lassen und Korrekturen klein halten.",
        "Aufbau 10-20s": "Zwischen +10s und +20s den Druck gleichmäßiger steigern, damit der Speed sauberer zunimmt.",
        "Hot-Zone": "Ab +20s kleinere, frühe Korrekturen setzen, damit die schnelle Zone stabil gehalten wird.",
        "Stabilität / Kipp-Risiko": "Körperspannung in Schulter, Rumpf und Hüfte früher stabilisieren.",
        "Korrekturen 20-25s": "Im Segment 20-25s weniger große Nachkorrekturen, stattdessen frühe Mini-Korrekturen.",
        "Seitbewegung Hot-Zone": "In der Hot-Zone seitliche Lenkimpulse reduzieren und den Druck gleichmäßiger nach unten halten.",
        "Winkelruhe bei hohem Speed": "Sobald der Speed hoch ist, nur noch kleine frühe Korrekturen fliegen, damit der Winkel ruhig bleibt.",
    }

    actions: list[str] = []
    for row in worse_rows:
        name = str(row.get("name"))
        text = action_map.get(name)
        if not text or text in actions:
            continue
        actions.append(text)
        if len(actions) >= 4:
            break
    return actions


def _build_tip_effect_profile(
    records: list[dict[str, Any]],
    *,
    performance_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usable = [
        row
        for row in records
        if bool(row.get("learning_eligible", True)) and not bool(row.get("analysis_blocked"))
    ]
    if len(usable) < 3:
        return {"available": False}

    recent_window = min(8, len(usable))
    recent = usable[-recent_window:]
    split = max(2, recent_window // 2)
    early = recent[:split]
    late = recent[split:]
    if len(late) < 2:
        late = recent[-2:]
        early = recent[:-2]
    if len(early) < 2:
        return {"available": False}

    phases = [
        {"key": "exit_score", "rank": 0, "label": "Exit / Stabilisierung"},
        {"key": "dive_score", "rank": 1, "label": "Dive-Aufbau"},
        {"key": "main_accel_score", "rank": 2, "label": "Hauptbeschleunigung"},
        {"key": "hot_build_score", "rank": 3, "label": "Hot-Zone Aufbau"},
        {"key": "max_speed_score", "rank": 4, "label": "Max-Speed Fenster"},
    ]
    low_thresholds = {
        "exit_score": 68.0,
        "dive_score": 70.0,
        "main_accel_score": 70.0,
        "hot_build_score": 70.0,
        "max_speed_score": 68.0,
    }
    low_hit_min = max(2, int(round(recent_window * 0.35)))

    metrics: dict[str, dict[str, float | int | None]] = {}
    phase_boosts: dict[int, int] = {}
    for phase in phases:
        key = str(phase["key"])
        recent_avg = _mean_metric(recent, key)
        if recent_avg is None:
            continue
        early_avg = _mean_metric(early, key)
        late_avg = _mean_metric(late, key)
        delta = None
        if early_avg is not None and late_avg is not None:
            delta = float(late_avg - early_avg)

        low_threshold = float(low_thresholds.get(key, 70.0))
        low_hits = 0
        for row in recent:
            value = _to_float(row.get(key))
            if value is not None and value < low_threshold:
                low_hits += 1

        boost = 0
        if recent_avg < low_threshold:
            boost += 2
        if low_hits >= low_hit_min:
            boost += 1
        if delta is not None and delta <= -4.0:
            boost += 2
        elif delta is not None and delta >= 4.0 and recent_avg >= low_threshold + 3.0:
            boost -= 1
        boost = int(max(0, min(6, boost)))
        if boost > 0:
            phase_boosts[int(phase["rank"])] = boost

        metrics[key] = {
            "recent_avg": round(float(recent_avg), 1),
            "early_avg": None if early_avg is None else round(float(early_avg), 1),
            "late_avg": None if late_avg is None else round(float(late_avg), 1),
            "delta": None if delta is None else round(float(delta), 1),
            "low_hits": int(low_hits),
            "low_threshold": round(low_threshold, 1),
            "boost": boost,
        }

    if not metrics:
        return {"available": False}

    if performance_profile and performance_profile.get("available"):
        band = str(performance_profile.get("performance_band") or "")
        confidence = str(performance_profile.get("confidence") or "")
        if confidence in {"medium", "good", "stable"}:
            if band in {"schnell", "elite"}:
                hot_metric = metrics.get("hot_build_score", {})
                stability_metric = metrics.get("max_speed_score", {})
                hot_avg = _to_float(hot_metric.get("recent_avg"))
                stability_avg = _to_float(stability_metric.get("recent_avg"))
                if hot_avg is not None and hot_avg < 78.0:
                    phase_boosts[3] = min(6, int(phase_boosts.get(3, 0)) + 1)
                if stability_avg is not None and stability_avg < 76.0:
                    phase_boosts[4] = min(6, int(phase_boosts.get(4, 0)) + 1)
            elif band in {"basis", "aufbau"}:
                exit_metric = metrics.get("exit_score", {})
                build_metric = metrics.get("dive_score", {})
                exit_avg = _to_float(exit_metric.get("recent_avg"))
                build_avg = _to_float(build_metric.get("recent_avg"))
                if exit_avg is not None and exit_avg < 76.0:
                    phase_boosts[0] = min(6, int(phase_boosts.get(0, 0)) + 1)
                if build_avg is not None and build_avg < 78.0:
                    phase_boosts[1] = min(6, int(phase_boosts.get(1, 0)) + 1)

    focus = max(
        phases,
        key=lambda phase: (
            int(phase_boosts.get(int(phase["rank"]), 0)),
            -float(metrics.get(str(phase["key"]), {}).get("recent_avg") or 0.0),
        ),
    )
    focus_key = str(focus["key"])
    focus_rank = int(focus["rank"])
    focus_label = str(focus["label"])
    focus_metric = metrics.get(focus_key, {})
    delta = _to_float(focus_metric.get("delta"))
    if delta is not None and delta <= -4.0:
        trend_hint = f"Trend zuletzt rückläufig ({delta:.1f} Punkte)"
    elif delta is not None and delta >= 4.0:
        trend_hint = f"Trend zuletzt besser (+{delta:.1f} Punkte)"
    else:
        trend_hint = "Trend zuletzt stabil"

    summary_line = f"Verlauf letzter {recent_window} Sprünge: Fokus aktuell {focus_label} ({trend_hint})."
    if performance_profile and performance_profile.get("available"):
        summary_line = f"{summary_line} Leistungsprofil: {performance_profile.get('summary')}"

    return {
        "available": True,
        "recent_window": recent_window,
        "phase_boosts": phase_boosts,
        "focus_phase": focus_rank,
        "focus_key": focus_key,
        "focus_label": focus_label,
        "focus_boost": int(phase_boosts.get(focus_rank, 0)),
        "trend_hint": trend_hint,
        "summary_line": summary_line,
        "metrics": metrics,
        "performance_profile": performance_profile or {"available": False},
    }


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    fallback_keys = {
        "dive_score": ["build_score"],
        "main_accel_score": ["build_score"],
        "hot_build_score": ["hot_score"],
        "max_speed_score": ["stability_score", "hot_score"],
    }
    values: list[float] = []
    for row in rows:
        value = _to_float(row.get(key))
        if value is None:
            for fallback_key in fallback_keys.get(key, []):
                value = _to_float(row.get(fallback_key))
                if value is not None:
                    break
        if value is None:
            continue
        values.append(value)
    if not values:
        return None
    return float(mean(values))


def _strip_priority_prefix(text: str) -> str:
    return re.sub(r"^\s*Priorit(?:aet|ät)\s+\d+\s*:\s*", "", text.strip(), flags=re.IGNORECASE)


def _find_previous_jump_row(*, jumps: list[dict[str, Any]], current_jump_id: str) -> dict[str, Any] | None:
    for idx, row in enumerate(jumps):
        if str(row.get("jump_id") or "") != current_jump_id:
            continue
        if idx + 1 < len(jumps):
            return jumps[idx + 1]
        return None
    return None


def _tip_focus_order() -> list[str]:
    return [
        "Exit / Stabilisierung",
        "Dive-Aufbau",
        "Hauptbeschleunigung",
        "Hot-Zone Aufbau",
        "Max-Speed Fenster",
    ]


def _tip_focus_from_previous(
    *,
    previous_score_rows: list[dict[str, Any]],
    previous_tips: list[str],
) -> list[str]:
    names = {str(row.get("name") or ""): int(row.get("score", 0)) for row in previous_score_rows}
    selected: set[str] = {name for name, score in names.items() if score < 70}

    def mark(phase_name: str, token_match: bool) -> None:
        if token_match:
            selected.add(phase_name)

    for raw_tip in previous_tips:
        text = normalize_german_text(str(raw_tip or "").strip())
        if not text:
            continue
        mark(
            "Exit / Stabilisierung",
            any(token in text for token in ["exit", "start", "absprung", "druck-mitnahme", "druck mitnahme"]),
        )
        mark("Dive-Aufbau", any(token in text for token in ["dive", "+3s", "+8s", "frueher aufbau", "frueher druck"]))
        mark(
            "Hauptbeschleunigung",
            any(token in text for token in ["aufbau", "+10s", "+15s", "tauchwinkel", "winkel"]),
        )
        mark(
            "Hot-Zone Aufbau",
            any(token in text for token in ["hot-zone aufbau", "+20s", "hot-zone", "hot zone", "vhor"]),
        )
        mark(
            "Max-Speed Fenster",
            any(token in text for token in ["max-speed", "3s", "peak", "400", "stabil", "kipp", "koerperspannung", "korrektur"]),
        )
        mark("Exit", any(token in text for token in ["exit", "start", "absprung", "druck-mitnahme", "druck mitnahme"]))
        mark("Aufbau 10-20s", any(token in text for token in ["aufbau", "+10s", "+15s", "+20s", "tauchwinkel", "winkel"]))
        mark("Hot-Zone", any(token in text for token in ["hot-zone", "hot zone", "hot-phase", "hot phase", "400", "vhor"]))
        mark(
            "Stabilität / Kipp-Risiko",
            any(token in text for token in ["stabil", "kipp", "körperspannung", "korrektur"]),
        )

    if not selected and names:
        ranked = sorted(names.items(), key=lambda item: int(item[1]))
        for phase_name, _ in ranked[:2]:
            selected.add(phase_name)

    canonical_selected: set[str] = set()
    for phase_name in selected:
        canonical_selected.update(_canonical_focus_phase_names(phase_name))
    selected = canonical_selected

    ordered = [name for name in _tip_focus_order() if name in selected]
    return ordered


def _canonical_focus_phase_names(phase_name: str) -> list[str]:
    normalized = normalize_german_text(str(phase_name or ""))
    if "exit" in normalized or "start" in normalized:
        return ["Exit / Stabilisierung"]
    if "dive" in normalized:
        return ["Dive-Aufbau"]
    if "hauptbeschleunigung" in normalized:
        return ["Hauptbeschleunigung"]
    if "aufbau 10-20" in normalized:
        return ["Dive-Aufbau", "Hauptbeschleunigung"]
    if "hot-zone aufbau" in normalized:
        return ["Hot-Zone Aufbau"]
    if normalized == "hot-zone" or normalized == "hot zone" or "hot-phase" in normalized:
        return ["Hot-Zone Aufbau", "Max-Speed Fenster"]
    if "max-speed" in normalized or "max speed" in normalized:
        return ["Max-Speed Fenster"]
    if "stabil" in normalized or "kipp" in normalized:
        return ["Max-Speed Fenster"]
    return [phase_name] if phase_name in _tip_focus_order() else []


def _tip_follow_status(*, score_delta: int, positive_hits: int, negative_hits: int) -> tuple[str, str]:
    if positive_hits > 0 and negative_hits > 0:
        return "uneindeutig", "Uneindeutig"
    if positive_hits > 0 or score_delta >= 4:
        return "verbessert", "Verbessert"
    if negative_hits > 0 or score_delta <= -4:
        return "verschlechtert", "Verschlechtert"
    return "unveraendert", "Unverändert"


def _fmt_delta(value: float | None, *, unit: str, decimals: int = 1) -> str:
    if value is None:
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}{unit}"


def _goal_metric_value(
    metric: str,
    *,
    snapshot: dict[str, float | None],
    report: dict[str, Any],
) -> float | None:
    value = _to_float(snapshot.get(metric))
    if value is not None:
        return value
    if metric == "negative_risk_score":
        return _to_float((report.get("metrics") or {}).get("negative_risk_score"))
    return None


def _evaluate_goal_metric(
    target: dict[str, Any],
    *,
    prev_snapshot: dict[str, float | None],
    curr_snapshot: dict[str, float | None],
    prev_report: dict[str, Any],
    curr_report: dict[str, Any],
) -> dict[str, Any] | None:
    metric = str(target.get("metric") or "").strip()
    if not metric:
        return None
    prev_value = _goal_metric_value(metric, snapshot=prev_snapshot, report=prev_report)
    curr_value = _goal_metric_value(metric, snapshot=curr_snapshot, report=curr_report)
    if prev_value is None or curr_value is None:
        return None

    direction = str(target.get("direction") or "").strip().lower()
    min_delta = abs(float(_to_float(target.get("min_delta")) or 0.0))
    unit = str(target.get("unit") or "")
    raw_decimals = target.get("decimals")
    decimals = int(raw_decimals) if raw_decimals is not None else 1
    delta = float(curr_value - prev_value)
    label = str(target.get("label") or metric)

    improved = False
    worsened = False
    if direction == "increase":
        improved = delta >= min_delta
        worsened = delta <= -min_delta
    elif direction == "decrease":
        improved = delta <= -min_delta
        worsened = delta >= min_delta
    else:
        improved = abs(delta) >= min_delta

    status_key = "met" if improved else "missed" if worsened else "partial"
    detail = (
        f"{label}: {prev_value:.{decimals}f} -> {curr_value:.{decimals}f}{unit} "
        f"({_fmt_delta(delta, unit=unit, decimals=decimals)})"
    )
    return {
        "metric": metric,
        "label": label,
        "status_key": status_key,
        "detail": detail,
        "delta": round(delta, decimals),
    }


def _goal_follow_status(*, positive_hits: int, negative_hits: int, total_hits: int) -> tuple[str, str]:
    if total_hits <= 0:
        return "nicht_messbar", "Nicht messbar"
    if positive_hits >= 1 and negative_hits >= 1:
        return "uneindeutig", "Uneindeutig"
    if positive_hits >= 1:
        return "verbessert", "Verbessert"
    if negative_hits >= 1:
        return "verschlechtert", "Verschlechtert"
    return "unveraendert", "Unverändert"


def _build_goal_follow_item(
    *,
    goal: dict[str, Any],
    prev_snapshot: dict[str, float | None],
    curr_snapshot: dict[str, float | None],
    prev_report: dict[str, Any],
    curr_report: dict[str, Any],
) -> dict[str, Any] | None:
    target_metrics = goal.get("target_metrics") if isinstance(goal.get("target_metrics"), list) else []
    evaluated = [
        item
        for target in target_metrics
        for item in [
            _evaluate_goal_metric(
                target,
                prev_snapshot=prev_snapshot,
                curr_snapshot=curr_snapshot,
                prev_report=prev_report,
                curr_report=curr_report,
            )
        ]
        if item is not None
    ]
    if not evaluated:
        return None

    positive_hits = sum(1 for item in evaluated if item.get("status_key") == "met")
    negative_hits = sum(1 for item in evaluated if item.get("status_key") == "missed")
    status_key, status_label = _goal_follow_status(
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        total_hits=len(evaluated),
    )
    message_prefix = {
        "verbessert": "Das konkrete Ziel hat sich messbar verbessert.",
        "verschlechtert": "Das konkrete Ziel hat sich messbar verschlechtert.",
        "unveraendert": "Das konkrete Ziel ist innerhalb der definierten Schwelle unverändert.",
        "uneindeutig": "Die Messwerte des konkreten Ziels entwickeln sich uneindeutig.",
        "nicht_messbar": "Das konkrete Ziel ist mit diesen Daten nicht messbar.",
    }[status_key]
    goal_text = _strip_priority_prefix(str(goal.get("text") or goal.get("display_text") or "")).strip()
    details = "; ".join(str(item.get("detail") or "") for item in evaluated if item.get("detail"))
    detail_text = message_prefix
    if details:
        detail_text = f"{detail_text} Messwerte: {details}."

    return {
        "phase": str(goal.get("phase") or "Coaching-Ziel"),
        "status_key": status_key,
        "status": status_label,
        "goal_text": goal_text,
        "detail": detail_text,
        "target_results": evaluated,
    }


def _goal_follow_group(item: dict[str, Any]) -> str:
    metrics = {
        str(result.get("metric") or "")
        for result in item.get("target_results", [])
        if isinstance(result, dict)
    }
    phase = normalize_german_text(str(item.get("phase") or ""))
    text = normalize_german_text(str(item.get("goal_text") or ""))
    if "v10" in metrics or "carry_ratio" in metrics:
        return "exit"
    if metrics.intersection({"angle_10", "angle_15", "angle_20"}):
        return "build_angle"
    if "gain_10_20" in metrics and not metrics.intersection({"vhor_min_20_25", "dur_400"}):
        return "build_speed"
    if metrics.intersection({"vhor_min_20_25", "dur_400"}):
        return "hot_zone_control"
    if metrics.intersection({"negative_risk_score", "angle_turns_20_25"}):
        return "stability"
    if "hot" in phase or "hot" in text or "20-25" in text or "peak" in text:
        return "hot_zone_control"
    return phase or text[:40] or "general"


def _phase_for_follow_group(group: str, fallback: str) -> str:
    if group == "exit":
        return "Exit / Stabilisierung"
    if group in {"build_angle", "build_speed"}:
        return "Hauptbeschleunigung"
    if group == "hot_zone_control":
        return "Hot-Zone Aufbau"
    if group == "stability":
        return "Max-Speed Fenster"
    return fallback


def _compact_goal_follow_items(items: list[dict[str, Any]], *, max_items: int = 8) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in items:
        group = _goal_follow_group(item)
        if group in grouped:
            current = grouped[group]
            current["related_goal_count"] = int(current.get("related_goal_count") or 1) + 1
            results = current.setdefault("target_results", [])
            known = {
                (str(result.get("metric") or ""), str(result.get("detail") or ""))
                for result in results
                if isinstance(result, dict)
            }
            for result in item.get("target_results", []):
                if not isinstance(result, dict):
                    continue
                identity = (str(result.get("metric") or ""), str(result.get("detail") or ""))
                if identity not in known:
                    results.append(result)
                    known.add(identity)
            continue
        copied = dict(item)
        copied["phase"] = _phase_for_follow_group(group, str(item.get("phase") or "Coaching-Ziel"))
        copied["related_goal_count"] = 1
        grouped[group] = copied
        order.append(group)

    compacted = [grouped[group] for group in order[:max_items]]
    hidden_count = max(0, len(order) - len(compacted))
    for item in compacted:
        results = [result for result in item.get("target_results", []) if isinstance(result, dict)]
        positive_hits = sum(1 for result in results if result.get("status_key") == "met")
        negative_hits = sum(1 for result in results if result.get("status_key") == "missed")
        status_key, status_label = _goal_follow_status(
            positive_hits=positive_hits,
            negative_hits=negative_hits,
            total_hits=len(results),
        )
        item["status_key"] = status_key
        item["status"] = status_label
        prefix = {
            "verbessert": "Die zusammengehörigen Zielwerte haben sich messbar verbessert.",
            "verschlechtert": "Die zusammengehörigen Zielwerte haben sich messbar verschlechtert.",
            "unveraendert": "Die zusammengehörigen Zielwerte sind innerhalb der Schwellen unverändert.",
            "uneindeutig": "Die zusammengehörigen Zielwerte entwickeln sich uneindeutig.",
            "nicht_messbar": "Die zusammengehörigen Zielwerte sind nicht messbar.",
        }[status_key]
        details = "; ".join(str(result.get("detail") or "") for result in results if result.get("detail"))
        item["detail"] = f"{prefix} Messwerte: {details}." if details else prefix
        related = int(item.get("related_goal_count") or 1)
        if related > 1:
            detail = str(item.get("detail") or "").strip()
            suffix = f" {related} ähnliche frühere Ziele wurden hier zusammengefasst."
            item["detail"] = f"{detail}{suffix}" if detail else suffix.strip()
    return compacted, hidden_count


def _coaching_snapshot_goals_for_follow_up(previous_report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshot = previous_report.get("coaching_snapshot")
    if not isinstance(snapshot, dict) or not snapshot.get("available"):
        return [], {}
    goals = snapshot.get("goals")
    if not isinstance(goals, list):
        return [], snapshot
    usable_goals: list[dict[str, Any]] = []
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        target_metrics = goal.get("target_metrics")
        if not isinstance(target_metrics, list) or not target_metrics:
            continue
        text = str(goal.get("text") or goal.get("display_text") or "").strip()
        if not text:
            continue
        usable_goals.append(goal)
    return usable_goals, snapshot


def _tip_follow_quality_note(
    *,
    current_report: dict[str, Any],
    previous_report: dict[str, Any],
) -> str:
    current_flags = {str(flag) for flag in (current_report.get("quality_flags") or [])}
    previous_flags = {str(flag) for flag in (previous_report.get("quality_flags") or [])}
    if "TIME_GAPS" in current_flags or "TIME_GAPS" in previous_flags:
        return "Rückblick eingeschränkt belastbar, weil aktuelle oder vorherige Daten Zeitlücken enthalten."
    return ""


def _build_tip_follow_item(
    *,
    phase_name: str,
    prev_score: int | None,
    curr_score: int | None,
    prev_snapshot: dict[str, float | None],
    curr_snapshot: dict[str, float | None],
    prev_report: dict[str, Any],
    curr_report: dict[str, Any],
) -> dict[str, Any]:
    score_before = int(prev_score or 0)
    score_now = int(curr_score or 0)
    score_delta = int(score_now - score_before)
    positive_hits = 0
    negative_hits = 0
    metric_parts: list[str] = []
    canonical_phase_names = _canonical_focus_phase_names(phase_name)
    canonical_phase = canonical_phase_names[0] if canonical_phase_names else phase_name

    if canonical_phase == "Exit / Stabilisierung":
        prev_v10 = _to_float(prev_snapshot.get("v10"))
        curr_v10 = _to_float(curr_snapshot.get("v10"))
        if prev_v10 is not None and curr_v10 is not None:
            dv = curr_v10 - prev_v10
            metric_parts.append(f"vVert +10s {prev_v10:.1f} -> {curr_v10:.1f} km/h ({_fmt_delta(dv, unit=' km/h')})")
            if dv >= 8.0:
                positive_hits += 1
            elif dv <= -8.0:
                negative_hits += 1
        prev_carry = _to_float(prev_snapshot.get("carry_ratio"))
        curr_carry = _to_float(curr_snapshot.get("carry_ratio"))
        if prev_carry is not None and curr_carry is not None:
            dc = curr_carry - prev_carry
            metric_parts.append(f"Druck-Mitnahme {prev_carry:.2f} -> {curr_carry:.2f} ({_fmt_delta(dc, unit='')})")
            if dc >= 0.04:
                positive_hits += 1
            elif dc <= -0.04:
                negative_hits += 1
    elif canonical_phase in {"Dive-Aufbau", "Hauptbeschleunigung"}:
        prev_gain = _to_float(prev_snapshot.get("gain_10_20"))
        curr_gain = _to_float(curr_snapshot.get("gain_10_20"))
        if prev_gain is not None and curr_gain is not None:
            dg = curr_gain - prev_gain
            metric_parts.append(f"Aufbau-Zuwachs {prev_gain:.1f} -> {curr_gain:.1f} km/h ({_fmt_delta(dg, unit=' km/h')})")
            if dg >= 12.0:
                positive_hits += 1
            elif dg <= -12.0:
                negative_hits += 1
        prev_angle = _to_float(prev_snapshot.get("angle_20"))
        curr_angle = _to_float(curr_snapshot.get("angle_20"))
        if prev_angle is not None and curr_angle is not None:
            metric_parts.append(f"Winkel Richtung Hot-Zone {prev_angle:.1f} -> {curr_angle:.1f} Grad")
    elif canonical_phase == "Hot-Zone Aufbau":
        prev_dur = _to_float(prev_snapshot.get("dur_400"))
        curr_dur = _to_float(curr_snapshot.get("dur_400"))
        if prev_dur is not None and curr_dur is not None:
            dd = curr_dur - prev_dur
            metric_parts.append(f">400 km/h {prev_dur:.1f}s -> {curr_dur:.1f}s ({_fmt_delta(dd, unit='s')})")
            if dd >= 0.3:
                positive_hits += 1
            elif dd <= -0.3:
                negative_hits += 1
        prev_vhor = _to_float(prev_snapshot.get("vhor_min_20_25"))
        curr_vhor = _to_float(curr_snapshot.get("vhor_min_20_25"))
        if prev_vhor is not None and curr_vhor is not None:
            dv = curr_vhor - prev_vhor
            metric_parts.append(f"vHor-Min 20-25s {prev_vhor:.1f} -> {curr_vhor:.1f} km/h ({_fmt_delta(dv, unit=' km/h')})")
            if dv >= 2.0:
                positive_hits += 1
            elif dv <= -2.0:
                negative_hits += 1
        prev_turns = _to_float(prev_snapshot.get("angle_turns_20_25"))
        curr_turns = _to_float(curr_snapshot.get("angle_turns_20_25"))
        if prev_turns is not None and curr_turns is not None:
            dt = curr_turns - prev_turns
            metric_parts.append(f"Korrekturen 20-25s {prev_turns:.0f} -> {curr_turns:.0f} ({_fmt_delta(dt, unit='')})")
            if dt <= -1.0:
                positive_hits += 1
            elif dt >= 1.0:
                negative_hits += 1
    else:
        prev_turns = _to_float(prev_snapshot.get("angle_turns_20_25"))
        curr_turns = _to_float(curr_snapshot.get("angle_turns_20_25"))
        if prev_turns is not None and curr_turns is not None:
            dt = curr_turns - prev_turns
            metric_parts.append(f"Korrekturen 20-25s {prev_turns:.0f} -> {curr_turns:.0f} ({_fmt_delta(dt, unit='')})")
            if dt <= -1.0:
                positive_hits += 1
            elif dt >= 1.0:
                negative_hits += 1
        prev_risk = _to_float((prev_report.get("metrics") or {}).get("negative_risk_score"))
        curr_risk = _to_float((curr_report.get("metrics") or {}).get("negative_risk_score"))
        if prev_risk is not None and curr_risk is not None:
            dr = curr_risk - prev_risk
            metric_parts.append(
                f"Risiko-Score {prev_risk:.1f} -> {curr_risk:.1f} ({_fmt_delta(dr, unit='')}, niedriger ist besser)"
            )
            if dr <= -5.0:
                positive_hits += 1
            elif dr >= 5.0:
                negative_hits += 1

    status_key, status_label = _tip_follow_status(
        score_delta=score_delta,
        positive_hits=positive_hits,
        negative_hits=negative_hits,
    )
    message_prefix = {
        "verbessert": "Der Schwerpunkt wurde messbar verbessert.",
        "verschlechtert": "Der Schwerpunkt hat sich messbar verschlechtert.",
        "unveraendert": "Der Schwerpunkt ist innerhalb der definierten Schwellen unverändert.",
        "uneindeutig": "Die Messwerte des Schwerpunkts entwickeln sich uneindeutig.",
    }[status_key]
    detail_text = f"{message_prefix} Score {score_before} -> {score_now} ({_fmt_delta(float(score_delta), unit='', decimals=0)})."
    if metric_parts:
        detail_text = f"{detail_text} Messwerte: " + "; ".join(metric_parts) + "."
    return {
        "phase": phase_name,
        "status_key": status_key,
        "status": status_label,
        "detail": detail_text,
    }


def _build_tip_follow_up(
    *,
    current_report: dict[str, Any],
    previous_report: dict[str, Any] | None,
    marco_profile: dict[str, Any] | None,
    previous_coaching_goals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if previous_report is None:
        return {
            "available": False,
            "reason": "Kein vorheriger Sprung vorhanden. Tipp-Umsetzung wird ab dem nächsten Sprung sichtbar.",
        }

    current_rows = _build_scorecard_rows(current_report, marco_profile=marco_profile)
    previous_rows = _build_scorecard_rows(previous_report, marco_profile=marco_profile)
    if not current_rows or not previous_rows:
        return {
            "available": False,
            "reason": "Zu wenige Daten für Tipp-Umsetzung.",
        }

    current_score_map = {str(row.get("name") or ""): int(row.get("score", 0)) for row in current_rows}
    previous_score_map = {str(row.get("name") or ""): int(row.get("score", 0)) for row in previous_rows}
    previous_tips = previous_report.get("tips") if isinstance(previous_report.get("tips"), list) else []
    focus_phases = _tip_focus_from_previous(
        previous_score_rows=previous_rows,
        previous_tips=[str(item) for item in previous_tips],
    )

    current_snapshot = _scorecard_metric_snapshot(current_report)
    previous_snapshot = _scorecard_metric_snapshot(previous_report)
    structured_items: list[dict[str, Any]] = []
    snapshot_goals, coaching_snapshot = _coaching_snapshot_goals_for_follow_up(previous_report)
    snapshot_used = False

    def build_structured_items(goals: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        built_items: list[dict[str, Any]] = []
        for goal in (goals or [])[:5]:
            if not isinstance(goal, dict):
                continue
            item = _build_goal_follow_item(
                goal=goal,
                prev_snapshot=previous_snapshot,
                curr_snapshot=current_snapshot,
                prev_report=previous_report,
                curr_report=current_report,
            )
            if item is not None:
                built_items.append(item)
        return built_items

    if snapshot_goals:
        structured_items = build_structured_items(snapshot_goals)
        snapshot_used = bool(structured_items)
    if not structured_items:
        structured_items = build_structured_items(previous_coaching_goals)
        snapshot_used = False

    if structured_items:
        items, hidden_count = _compact_goal_follow_items(structured_items, max_items=8)
    else:
        if not focus_phases:
            return {
                "available": False,
                "reason": "Keine klaren Schwerpunkte aus dem vorherigen Sprung gefunden.",
            }
        items = [
            _build_tip_follow_item(
                phase_name=phase_name,
                prev_score=previous_score_map.get(phase_name),
                curr_score=current_score_map.get(phase_name),
                prev_snapshot=previous_snapshot,
                curr_snapshot=current_snapshot,
                prev_report=previous_report,
                curr_report=current_report,
            )
            for phase_name in focus_phases[:2]
        ]
        hidden_count = max(0, len(focus_phases) - len(items))
    done_count = sum(1 for item in items if item.get("status_key") == "umgesetzt")
    open_count = sum(1 for item in items if item.get("status_key") == "offen")
    partial_count = sum(1 for item in items if item.get("status_key") == "teilweise")
    mixed_count = sum(1 for item in items if item.get("status_key") == "gemischt")
    count_parts = []
    if done_count:
        count_parts.append(f"{done_count} umgesetzt")
    if mixed_count:
        count_parts.append(f"{mixed_count} gemischt")
    if partial_count:
        count_parts.append(f"{partial_count} teilweise")
    if open_count:
        count_parts.append(f"{open_count} offen")
    summary = "Umsetzung letzter Fokus: " + (", ".join(count_parts) if count_parts else "keine klare Veränderung") + "."
    quality_note = _tip_follow_quality_note(current_report=current_report, previous_report=previous_report)
    return {
        "available": True,
        "summary": summary,
        "quality_note": quality_note,
        "compact_note": (
            f"Weitere frühere Schwerpunkte ausgeblendet: {hidden_count}."
            if hidden_count > 0
            else ""
        ),
        "previous_jump_id": previous_report.get("jump", {}).get("jump_id"),
        "previous_file_name": previous_report.get("jump", {}).get("file_name"),
        "previous_t0_utc": previous_report.get("jump", {}).get("t0_utc"),
        "entries": items,
        "snapshot_used": snapshot_used,
        "snapshot_source": str(coaching_snapshot.get("source") or "") if coaching_snapshot else "",
        "snapshot_focus_text": str(coaching_snapshot.get("focus_text") or "") if coaching_snapshot else "",
    }


def _unique_texts(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _merge_priority_texts(primary: str, items: list[str], *, max_items: int) -> list[str]:
    primary_text = str(primary or "").strip()
    if not primary_text:
        return _unique_texts_semantic(items)[:max_items]
    out = [primary_text]
    for item in _unique_texts_semantic(items):
        if len(out) >= max_items:
            break
        if _is_semantic_near_duplicate_text(primary_text, item):
            continue
        out.append(item)
    return out


def _filter_actions_for_primary_diagnosis(
    primary_diagnosis: dict[str, Any],
    actions: list[str],
    *,
    max_items: int,
) -> list[str]:
    pattern = str(primary_diagnosis.get("pattern") or "")
    if pattern not in {"late_hard_steepening_vhor_collapse", "early_horizontal_reserve_collapse"}:
        return actions[:max_items]
    if not actions:
        return []

    out = [actions[0]]
    skip_markers = [
        "mehr druck aufbauen",
        "mindestens +180",
        "fehlen gegen die referenz",
        "bis +15s fehlen",
        "zuwachs",
        "steiler werden",
    ]
    keep_markers = [
        "uebergang",
        "übergang",
        "vhor",
        "horizontale reserve",
        "waagerechte geschwindigkeit",
        "hot-zone",
        "hotzone",
        "3s-fenster",
        "korrektur",
        "druck laenger tragen",
        "druck länger tragen",
    ]
    for action in actions[1:]:
        if len(out) >= max_items:
            break
        lowered = normalize_german_text(action)
        if "steilflug" in lowered and any(marker in lowered for marker in ["uebergang", "übergang"]):
            continue
        should_skip = any(normalize_german_text(marker) in lowered for marker in skip_markers)
        should_keep = any(normalize_german_text(marker) in lowered for marker in keep_markers)
        if should_skip and not should_keep:
            continue
        out.append(action)
    return out


def _normalize_text_for_semantic_dedupe(text: str) -> str:
    lowered = normalize_german_text(text.strip())
    lowered = lowered.replace("-", " ")
    lowered = lowered.replace(":", " ")
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _semantic_tokens(text: str) -> set[str]:
    normalized = _normalize_text_for_semantic_dedupe(text)
    if not normalized:
        return set()
    tokens: set[str] = set()
    for token in normalized.split():
        if len(token) <= 2:
            continue
        if token in _SEMANTIC_DEDUPE_STOPWORDS:
            continue
        if any(char.isdigit() for char in token):
            continue
        tokens.add(token)
    return tokens


def _strength_semantic_bucket(text: str) -> str | None:
    tokens = _semantic_tokens(text)
    if not tokens:
        return None
    has_start = bool(tokens & {"start", "absprung", "sprungstart", "exit"})
    has_control = any(token.startswith("kontroll") for token in tokens) or bool(
        tokens & {"sauber", "ruhig", "stabil"}
    )
    if has_start and has_control:
        return "exit_start_control"
    has_carry = "druck" in tokens and bool(tokens & {"mitnahme", "mitgenommen", "übergang", "dynamik"})
    if has_carry:
        return "exit_pressure_carry"
    return None


def _is_semantic_near_duplicate_text(left: str, right: str) -> bool:
    left_text = str(left).strip()
    right_text = str(right).strip()
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True

    left_bucket = _strength_semantic_bucket(left_text)
    right_bucket = _strength_semantic_bucket(right_text)
    if left_bucket and left_bucket == right_bucket:
        return True

    if _topic_from_text(left_text) != _topic_from_text(right_text):
        return False

    left_tokens = _semantic_tokens(left_text)
    right_tokens = _semantic_tokens(right_text)
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens & right_tokens
    if len(overlap) < 3:
        return False
    coverage = len(overlap) / float(min(len(left_tokens), len(right_tokens)))
    return coverage >= 0.60


def _unique_texts_semantic(items: list[str]) -> list[str]:
    unique_items = _unique_texts(items)
    out: list[str] = []
    for item in unique_items:
        if any(_is_semantic_near_duplicate_text(item, existing) for existing in out):
            continue
        out.append(item)
    return out


def _is_scorecard_context_only_line(text: str) -> bool:
    lowered = normalize_german_text(str(text or "").strip())
    if not lowered:
        return False
    if lowered.startswith("+") and "zielwinkel" in lowered:
        return True
    if lowered.startswith("istwerte") or lowered.startswith("messwerte"):
        return True
    if "avg winkel" in lowered and ("avg vvert" in lowered or "avg geschwindigkeit" in lowered):
        return True
    context_markers = [
        "der tauchwinkel ist dabei eher zu flach",
        "der tauchwinkel ist dabei zu steil",
        "der tauchwinkel liegt in einem guten bereich",
        "du kannst die sehr schnelle phase spuerbar halten",
        "phase liegt im technischen zielbereich",
        "max-speed-fenster ist technisch nutzbar",
        "du kannst die sehr schnelle phase spürbar halten",
    ]
    return any(normalize_german_text(marker) in lowered for marker in context_markers)


def _primary_issue_line_from_row(row: dict[str, Any]) -> str:
    score = int(row.get("score", 100) or 100)
    candidates: list[str] = []
    reason_lines = row.get("reason_lines")
    if isinstance(reason_lines, list):
        for item in reason_lines:
            text = str(item).strip()
            if not text:
                continue
            if text.lower().startswith("messwerte"):
                continue
            if normalize_german_text(text).startswith("referenzscore"):
                continue
            if _is_scorecard_context_only_line(text):
                continue
            candidates.append(text)
    reason = str(row.get("reason") or "").strip()
    if reason:
        parts = re.split(r"(?<=[.!?])\s+", reason)
        for part in parts:
            text = str(part).strip()
            if not text:
                continue
            if text.lower().startswith("messwerte"):
                continue
            if normalize_german_text(text).startswith("referenzscore"):
                continue
            if _is_scorecard_context_only_line(text):
                continue
            candidates.append(text)

    if not candidates:
        return ""

    for text in candidates:
        if _is_issue_like_text(text):
            return text

    # If the row is weak but no actual issue sentence exists, avoid showing a positive line as "problem".
    if score < 70:
        return ""
    return candidates[0]


def _is_issue_like_text(text: str) -> bool:
    lowered = normalize_german_text(text.strip())
    if not lowered:
        return False
    if lowered.startswith("messwerte"):
        return False
    if lowered.startswith("referenzscore"):
        return False
    if _is_scorecard_context_only_line(text):
        return False

    negative_markers = [
        "zu flach",
        "zu steil",
        "zu schwach",
        "zu kurz",
        "nur kurz",
        "zu wenig",
        "fällt",
        "fällt",
        "einbruch",
        "bricht",
        "instabil",
        "kipp",
        "risiko",
        "unruhig",
        "rückläufig",
        "nachkorrektur",
        "fehlt",
        "unter ",
        "nicht",
        "problem",
    ]
    if any(normalize_german_text(marker) in lowered for marker in negative_markers):
        return True

    positive_markers = [
        "sauber",
        "gut",
        "optimal",
        "kontrolliert",
        "stabil",
        "sehr stark",
        "passt",
        "ruhig",
        "genug",
        "dynamisch",
        "spürbar halten",
        "spuerbar halten",
    ]
    if any(normalize_german_text(marker) in lowered for marker in positive_markers):
        return False

    # Unknown wording: keep it as issue to avoid hiding potential signals.
    return True


def _compact_main_issues(items: list[str], *, max_items: int) -> list[str]:
    if max_items <= 0:
        return []
    unique_items = _unique_texts_semantic(items)
    ordered = sorted(
        unique_items,
        key=lambda text: (
            _timeline_rank_from_text(text),
            -_tip_specificity_score(text),
            len(text),
            text.lower(),
        ),
    )
    out: list[str] = []
    seen_topics: set[str] = set()
    for item in ordered:
        topic = _topic_from_text(item)
        if topic != "other" and topic in seen_topics:
            continue
        out.append(item)
        if topic != "other":
            seen_topics.add(topic)
        if len(out) >= max_items:
            break
    return out


def _scorecard_phase_rank(name: str) -> int:
    lowered = normalize_german_text(name.strip())
    if "exit" in lowered:
        return 0
    if "dive" in lowered:
        return 1
    if "hauptbeschleunigung" in lowered:
        return 2
    if "hot-zone aufbau" in lowered or ("hot" in lowered and "aufbau" in lowered):
        return 3
    if "max-speed" in lowered or "max speed" in lowered:
        return 4
    if "aufbau" in lowered:
        return 2
    if "hot-zone" in lowered or "hot" in lowered:
        return 3
    if "stabilität" in lowered or "kipp" in lowered:
        return 4
    return 5


def _topic_from_text(text: str) -> str:
    lowered = normalize_german_text(text.strip())
    fast_unrest = any(
        normalize_german_text(token) in lowered
        for token in ["schnell", "schnelle phase", "peak", "schlussteil", "+20s", "+20 bis +25"]
    ) and any(
        normalize_german_text(token) in lowered
        for token in [
            "unruhig",
            "seitbewegung",
            "richtungsdrehen",
            "winkelschwankung",
            "korrektur",
            "gegenkorrektur",
            "lenkimpuls",
        ]
    )
    if fast_unrest:
        return "stability"
    if any(normalize_german_text(token) in lowered for token in ["hot-zone", "hot-phase", "+20 bis +25", "peak-phase", "schlussteil"]):
        return "hot"
    if any(
        normalize_german_text(token) in lowered
        for token in [
            "+10s",
            "+15s",
            "+20s",
            "aufbau",
            "aufbauphase",
            "winkel",
            "stabilitaetsbereich",
            "stabilitätsbereich",
        ]
    ):
        return "build"
    if any(
        normalize_german_text(token) in lowered
        for token in [
            "stabilität",
            "kipp",
            "körperspannung",
            "kurvenverlauf",
            "unruhig",
            "seitbewegung",
            "richtungsdrehen",
            "winkelschwankung",
            "korrektur",
            "gegenkorrektur",
            "lenkimpuls",
        ]
    ):
        return "stability"
    if any(normalize_german_text(token) in lowered for token in ["exit", "absprung", "startphase", "ersten 2 sekunden"]):
        return "exit"
    if any(normalize_german_text(token) in lowered for token in ["zuwachs", "druckaufbau", "druck aufbauen"]):
        return "build"
    return "other"


def _timeline_rank_from_text(text: str) -> int:
    lowered = normalize_german_text(text.strip())
    marks = [float(m.group(1)) for m in re.finditer(r"\+([0-9]+(?:\.[0-9]+)?)s", lowered)]
    if marks:
        first_t = min(marks)
        if first_t <= 3.0:
            return 0
        if first_t <= 10.0:
            return 1
        if first_t <= 15.0:
            return 2
        if first_t <= 20.0:
            return 3
        if first_t <= 25.0:
            return 4
        return 5

    topic = _topic_from_text(text)
    if topic == "exit":
        return 0
    if topic == "build":
        return 2
    if topic == "hot":
        return 4
    if topic == "stability":
        return 5
    return 6


def _tip_specificity_score(text: str) -> int:
    lowered = normalize_german_text(text.strip())
    score = 0
    if re.search(r"\+([0-9]+(?:\.[0-9]+)?)s", lowered):
        score += 2
    if re.search(r"\d", lowered):
        score += 1
    if any(normalize_german_text(token) in lowered for token in ["km/h", "m/s", "deg", "grad", "korridor", "ziel"]):
        score += 1
    if any(normalize_german_text(token) in lowered for token in ["vhor", "vvert", "winkel", "zuwachs"]):
        score += 1

    generic_markers = [
        "linie ruhiger halten",
        "kleine, frühe korrekturen",
        "druck konstanter halten",
        "ablauf stabil wiederholen",
        "körperspannung früher stabilisieren",
        "ab +20s sauberer und ruhiger arbeiten",
        "seitlinie in der hot-zone beruhigen",
        "in der peak-phase druck ruhiger halten",
    ]
    if any(normalize_german_text(token) in lowered for token in generic_markers):
        score -= 1
    return score


def _generic_tip_penalty(text: str) -> int:
    lowered = normalize_german_text(text.strip())
    strongly_generic_markers = [
        "ab +20s sauberer und ruhiger arbeiten",
        "seitlinie in der hot-zone beruhigen",
        "in der peak-phase druck ruhiger halten",
    ]
    if any(normalize_german_text(token) in lowered for token in strongly_generic_markers):
        return 2

    has_numeric_context = bool(re.search(r"\d", lowered)) or "+10s" in lowered or "+15s" in lowered or "+20s" in lowered
    has_metric_or_phase_context = any(
        normalize_german_text(token) in lowered
        for token in [
            "vhor",
            "vvert",
            "winkel",
            "hot-phase",
            "hot-zone",
            "aufbau",
            "exit",
            "segment",
            "korridor",
        ]
    )
    generic_markers = [
        "linie ruhiger halten",
        "kleine, frühe korrekturen",
        "druck konstanter halten",
        "ablauf stabil wiederholen",
    ]
    if any(normalize_german_text(token) in lowered for token in generic_markers) and not (has_numeric_context or has_metric_or_phase_context):
        return 1
    return 0


def _sort_texts_by_timeline(items: list[str]) -> list[str]:
    unique_items = _unique_texts(items)
    return sorted(
        unique_items,
        key=lambda text: (
            _timeline_rank_from_text(text),
            -_tip_specificity_score(text),
            text.lower(),
        ),
    )


def _topic_matches_issue(*, issue_topic: str, action_topic: str) -> bool:
    if issue_topic == action_topic:
        return True
    if issue_topic == "hot" and action_topic in {"hot", "stability"}:
        return True
    if issue_topic == "stability" and action_topic in {"stability", "hot"}:
        return True
    if issue_topic == "build" and action_topic in {"build", "exit"}:
        return True
    return False


def _focus_names_from_issue_texts(items: list[str]) -> list[str]:
    names_by_topic = {
        "exit": "Exit / Stabilisierung",
        "build": "Hauptbeschleunigung",
        "hot": "Hot-Zone Aufbau",
        "stability": "Max-Speed Fenster",
    }
    out: list[str] = []
    for item in items:
        topic = _topic_from_text(item)
        name = names_by_topic.get(topic)
        if not name or name in out:
            continue
        out.append(name)
        if len(out) >= 2:
            break
    return out


def _has_steep_build_issue(items: list[str]) -> bool:
    lowered = normalize_german_text(" ".join(str(item or "") for item in items))
    markers = [
        "zu steil",
        "zu schnell steil",
        "maximal steil",
        "tauchwinkel",
        "nicht sofort steiler",
    ]
    return any(normalize_german_text(marker) in lowered for marker in markers)


def _is_build_angle_control_action(text: str) -> bool:
    lowered = normalize_german_text(text)
    markers = [
        "flacher",
        "zuerst stabil",
        "zielkorridor",
        "zielbereich",
        "nicht sofort steiler",
        "stabil im bereich",
    ]
    return any(normalize_german_text(marker) in lowered for marker in markers)


def _is_generic_build_pressure_action(text: str) -> bool:
    lowered = normalize_german_text(text)
    generic_markers = ["mehr druck aufbauen", "mindestens +180", "zuwachs"]
    safe_markers = ["erst stabilität sichern", "erst stabilitaet sichern", "nicht sofort steiler", "flacher"]
    return any(normalize_german_text(marker) in lowered for marker in generic_markers) and not any(
        normalize_german_text(marker) in lowered for marker in safe_markers
    )


def _filter_conflicting_build_pressure_actions(*, main_issues: list[str], actions: list[str]) -> list[str]:
    if not actions or not _has_steep_build_issue(main_issues):
        return actions
    if not any(_is_build_angle_control_action(action) for action in actions):
        return actions
    return [action for action in actions if not _is_generic_build_pressure_action(action)]


def _harmonize_action_texts(
    actions: list[str],
    *,
    max_items: int,
    preserve_first: bool = False,
) -> list[str]:
    if max_items <= 0:
        return []
    unique_actions = _unique_texts([str(item).strip() for item in actions if str(item or "").strip()])
    if len(unique_actions) <= 1:
        return unique_actions[:max_items]

    grouped_entries: list[dict[str, Any]] = []
    family_entries: dict[str, dict[str, Any]] = {}
    for text in unique_actions:
        family = _action_harmonization_family(text)
        if not family:
            grouped_entries.append({"family": "", "texts": [text]})
            continue
        entry = family_entries.get(family)
        if entry is None:
            entry = {"family": family, "texts": []}
            family_entries[family] = entry
            grouped_entries.append(entry)
        entry["texts"].append(text)

    merged: list[str] = []
    for entry in grouped_entries:
        texts = [str(item).strip() for item in entry.get("texts", []) if str(item).strip()]
        if not texts:
            continue
        family = str(entry.get("family") or "")
        if family and len(texts) >= 2:
            merged.append(_merge_action_family_text(family=family, texts=texts))
        else:
            merged.append(texts[0])
    merged = _unique_texts(merged)
    if preserve_first and unique_actions:
        first = unique_actions[0]
        if first in merged:
            rest = [item for item in merged if item != first]
            return [first] + _sort_texts_by_timeline(rest)[: max(0, max_items - 1)]
    return _sort_texts_by_timeline(merged)[:max_items]


def _action_harmonization_family(text: str) -> str:
    lowered = normalize_german_text(text)
    if not lowered:
        return ""
    if _is_build_specific_action_text(lowered):
        return ""
    correction_markers = [
        "korrektur",
        "gegenkorrektur",
        "lenkimpuls",
        "lenkimpulse",
        "beschleunigungskurve",
        "kurvenverlauf",
        "richtung ruhiger",
        "linie ruhiger",
    ]
    if not any(normalize_german_text(marker) in lowered for marker in correction_markers):
        return ""
    fast_markers = [
        "letzte schnelle phase",
        "schnelle phase",
        "hot-zone",
        "hot zone",
        "hot-phase",
        "peak",
        "schlussteil",
        "max-speed",
        "+20",
        "vhor",
        "horizontale reserve",
        "waagerechte geschwindigkeit",
        "beschleunigungskurve",
    ]
    if any(normalize_german_text(marker) in lowered for marker in fast_markers):
        return "fast_correction_control"
    return ""


def _is_build_specific_action_text(lowered: str) -> bool:
    build_markers = [
        "+10 bis +15",
        "+15 bis +20",
        "+10s",
        "+15s",
        "aufbau",
        "zuwachs",
        "stabilitaetsbereich",
        "stabilitÃ¤tsbereich",
        "zielbereich",
        "zielkorridor",
    ]
    fast_markers = [
        "letzte schnelle phase",
        "schnelle phase",
        "hot-zone",
        "hot phase",
        "hot-phase",
        "peak",
        "schlussteil",
    ]
    return any(normalize_german_text(marker) in lowered for marker in build_markers) and not any(
        normalize_german_text(marker) in lowered for marker in fast_markers
    )


def _merge_action_family_text(*, family: str, texts: list[str]) -> str:
    if family != "fast_correction_control":
        return sorted(
            texts,
            key=lambda text: (
                -_tip_specificity_score(text),
                _generic_tip_penalty(text),
                _timeline_rank_from_text(text),
                len(text),
            ),
        )[0]

    combined = normalize_german_text(" ".join(texts))
    has_reserve = any(
        normalize_german_text(marker) in combined
        for marker in ["vhor", "horizontale reserve", "waagerechte geschwindigkeit"]
    )
    has_curve = "beschleunigungskurve" in combined or "kurvenverlauf" in combined
    if has_reserve:
        if has_curve:
            return (
                "In der letzten schnellen Phase die horizontale Reserve schuetzen: "
                "Korrekturen frueher und kleiner setzen, damit Lenkimpulse und Beschleunigungskurve ruhiger werden."
            )
        return (
            "In der letzten schnellen Phase die horizontale Reserve schuetzen: "
            "Korrekturen frueher und kleiner setzen statt spaeter grosser Gegenkorrektur."
        )
    return (
        "In der letzten schnellen Phase Korrekturen frueher und kleiner setzen, "
        "damit Lenkimpulse und Beschleunigungskurve ruhiger werden."
    )


def _build_issue_aligned_actions(*, main_issues: list[str], actions: list[str], max_items: int) -> list[str]:
    if max_items <= 0:
        return []
    unique_actions = _unique_texts(actions)
    if not unique_actions:
        return []

    issue_topics = _unique_texts([_topic_from_text(item) for item in main_issues if _topic_from_text(item) != "other"])
    structured = [
        {
            "text": text,
            "topic": _topic_from_text(text),
            "timeline": _timeline_rank_from_text(text),
            "specificity": _tip_specificity_score(text),
            "generic_penalty": _generic_tip_penalty(text),
        }
        for text in unique_actions
    ]

    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            int(item["generic_penalty"]),
            -int(item["specificity"]),
            int(item["timeline"]),
            str(item["text"]).lower(),
        )

    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    for issue_topic in issue_topics:
        candidates = [
            item
            for item in structured
            if item["text"] not in used and _topic_matches_issue(issue_topic=issue_topic, action_topic=str(item["topic"]))
        ]
        if not candidates:
            continue
        chosen = sorted(candidates, key=sort_key)[0]
        selected.append(chosen)
        used.add(str(chosen["text"]))
        if len(selected) >= max_items:
            break

    for item in sorted((entry for entry in structured if entry["text"] not in used), key=sort_key):
        if len(selected) >= max_items:
            break
        if item["generic_penalty"] > 0:
            specific_count = sum(1 for entry in selected if int(entry.get("generic_penalty", 0)) == 0)
            if specific_count >= 2:
                continue
        selected.append(item)
        used.add(str(item["text"]))

    if not selected:
        selected = sorted(structured, key=sort_key)[:max_items]

    ordered = sorted(selected, key=sort_key)
    return [str(item["text"]) for item in ordered[:max_items]]


def _fixpoint_at(fixpoints: list[dict[str, Any]], t_rel_s: float) -> dict[str, Any] | None:
    for item in fixpoints:
        try:
            if abs(float(item.get("t_rel_s", -1.0)) - t_rel_s) < 1e-6:
                return item
        except Exception:
            continue
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_score(value: float) -> int:
    return int(max(0.0, min(100.0, round(float(value)))))


def _status_from_score(score: int) -> str:
    if score >= 85:
        return "Sehr gut"
    if score >= 70:
        return "Gut"
    if score >= 55:
        return "Mittel"
    return "Kritisch"


def _score_status_label(score: int) -> str:
    s = int(max(0, min(100, score)))
    if s >= 95:
        return f"{s}% von 100% (Sehr gut - Marco Hepp Niveau)"
    return f"{s}% von 100%"


def _fmt1(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _fmt2(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _join_reason_lines(lines: list[str]) -> str:
    return " ".join(str(line).strip() for line in lines if str(line).strip())


def _build_exit_reason_lines(*, v10: float | None, carry_ratio: float | None, exit_unsteady: bool) -> list[str]:
    lines: list[str] = []
    if v10 is None:
        lines.append("Der Start konnte nicht sauber bewertet werden.")
    elif v10 < 220.0:
        lines.append("Der Start bringt noch zu wenig Tempo in den Sprung.")
    elif v10 <= 320.0:
        if exit_unsteady:
            lines.append("Der Start ist kraftvoll, aber noch etwas unruhig.")
        else:
            lines.append("Der Start ist sauber und kontrolliert.")
    else:
        lines.append("Der Start ist sehr dynamisch.")

    if carry_ratio is not None:
        if carry_ratio < 0.65:
            lines.append("Nach dem Exit fällt der Druck früh ab.")
        elif carry_ratio < 0.78:
            lines.append("Der Druck wird teilweise mitgenommen, aber nicht durchgehend.")
        else:
            lines.append("Du nimmst den Druck nach dem Exit gut mit.")

    if v10 is not None or carry_ratio is not None:
        lines.append(f"Messwerte: vVert +10s {_fmt1(v10)} km/h, Druck-Mitnahme {_fmt2(carry_ratio)}.")
    return lines


def _build_build_reason_lines(
    *,
    gain_10_20: float | None,
    angle_20: float | None,
    phase_window_label: str | None = None,
) -> list[str]:
    lines: list[str] = []
    phase_text = "in der Aufbauphase"
    if phase_window_label:
        phase_text = f"im Aufbaufenster ({phase_window_label})"
    if gain_10_20 is None:
        lines.append(f"Der Speed-Aufbau {phase_text} konnte nicht sicher bewertet werden.")
    elif gain_10_20 < 80.0:
        lines.append(f"Du baust {phase_text} zu wenig zusätzlichen Speed auf.")
    elif gain_10_20 < 95.0:
        lines.append(f"Der Aufbau ist {phase_text} da, aber noch etwas zu schwach.")
    elif gain_10_20 <= 170.0:
        lines.append(f"Der Speed-Aufbau ist {phase_text} gut.")
    else:
        if angle_20 is not None and angle_20 > 85.5:
            lines.append(
                f"Der Aufbau ist {phase_text} sehr stark. "
                "Wichtig: nicht zusätzlich steiler drücken, sondern die Linie ruhig halten, "
                "damit der hohe Druck stabil bleibt."
            )
        else:
            lines.append(f"Der Aufbau ist {phase_text} sehr stark und dabei kontrolliert.")

    if angle_20 is not None:
        if angle_20 < 80.0:
            lines.append("Der Tauchwinkel ist dabei eher zu flach.")
        elif angle_20 > 87.0:
            lines.append("Der Tauchwinkel ist dabei zu steil.")
        else:
            lines.append("Der Tauchwinkel liegt in einem guten Bereich.")

    lines.append(
        f"Messwerte ({phase_text}): Zuwachs {_fmt1(gain_10_20)} km/h, mittlerer Zielwinkel {_fmt1(angle_20)} Grad."
    )
    return lines


def _build_forward_track_reason_lines(*, chart: dict[str, Any], end_s: float | None = None) -> list[str]:
    time_s = chart.get("time_s", []) or []
    forward_m = chart.get("forward_m", []) or []
    if not time_s or not forward_m or len(time_s) != len(forward_m):
        return []

    pairs: list[tuple[float, float]] = []
    for i, raw_t in enumerate(time_s):
        t_val = _to_float(raw_t)
        f_val = _to_float(forward_m[i])
        if t_val is None or f_val is None:
            continue
        if t_val < 0.0:
            continue
        if end_s is not None and t_val > float(end_s):
            continue
        pairs.append((float(t_val), float(f_val)))
    if len(pairs) < 5:
        return []

    t = [item[0] for item in pairs]
    fwd = [item[1] for item in pairs]
    running_max: list[float] = []
    backtrack: list[float] = []
    cur_max = float("-inf")
    for value in fwd:
        cur_max = max(cur_max, float(value))
        running_max.append(cur_max)
        backtrack.append(float(cur_max - float(value)))

    max_forward = float(max(fwd)) if fwd else 0.0
    max_backtrack = float(max(backtrack)) if backtrack else 0.0
    if max_forward < 10.0:
        return []
    backtrack_ratio_pct = float((max_backtrack / max(max_forward, 1e-6)) * 100.0)

    threshold = max(8.0, 0.08 * max_forward)
    drift_start_s = None
    for i in range(len(t)):
        if t[i] < 8.0:
            continue
        if backtrack[i] >= threshold:
            drift_start_s = float(t[i])
            break

    lines: list[str] = []
    if max_backtrack >= 28.0 and backtrack_ratio_pct >= 12.0:
        start_text = "" if drift_start_s is None else f" ab +{drift_start_s:.1f}s"
        lines.append(
            "Die Fluglinie driftet vor Abbremsbeginn klar nach hinten"
            f"{start_text}. Das passt zu instabilen Korrekturen/Kippmomenten."
        )
    elif max_backtrack >= 14.0 and backtrack_ratio_pct >= 7.0:
        lines.append("Vor Abbremsbeginn geht die Fluglinie teilweise wieder zurück.")
    else:
        lines.append("Bis zum Abbremsbeginn bleibt die Fluglinie vorwärtsgerichtet.")

    lines.append(
        "Fluglinie-Messwerte bis Abbremsbeginn: Vorwärts-Max "
        f"{_fmt1(max_forward)} m, Rückdrift {_fmt1(max_backtrack)} m ({_fmt1(backtrack_ratio_pct)}%)."
    )
    return lines


def _forward_track_eval_end_s(*, notes: dict[str, Any]) -> float | None:
    decel_start = _to_float(notes.get("decel_start_s"))
    curve_end = _to_float(notes.get("curve_window_end_s"))
    coaching_cap = 25.0

    if decel_start is not None and decel_start >= 8.0:
        end_s = decel_start
        if curve_end is not None:
            end_s = min(end_s, curve_end)
        return max(8.0, float(end_s))

    if curve_end is not None:
        return max(8.0, float(min(curve_end, coaching_cap)))

    return coaching_cap


def _build_quality_issue_lines(raw_flags: Any) -> list[str]:
    if raw_flags is None:
        return []
    try:
        if isinstance(raw_flags, str):
            flags = set(json.loads(raw_flags))
        else:
            flags = set(raw_flags)
    except Exception:
        return []

    priority = [
        "EARLY_JUMP_END",
        "NO_CLEAR_EXIT",
        "INVALID_EXIT_ALTITUDE",
        "TIME_GAPS",
        "LOW_GPS_FIX",
        "HIGH_SPEED_ACCURACY_ERROR",
        "SPEED_SPIKE",
    ]
    mapping = {
        "EARLY_JUMP_END": "Sprung endet für die Auswertung zu früh.",
        "NO_CLEAR_EXIT": "Absprungzeit war nicht eindeutig; Detailwerte koennen verschoben sein.",
        "INVALID_EXIT_ALTITUDE": "Exit-Höhe wirkt unplausibel.",
        "TIME_GAPS": "Im relevanten Bereich gibt es Datenlücken.",
        "LOW_GPS_FIX": "GPS-Fix war zeitweise schwach.",
        "HIGH_SPEED_ACCURACY_ERROR": "Geschwindigkeitsgenauigkeit war zeitweise eingeschränkt.",
        "SPEED_SPIKE": "Unplausible Speed-Spitze erkannt.",
    }
    lines: list[str] = []
    for flag in priority:
        if flag in flags:
            lines.append(mapping[flag])
    return lines


def _build_rule_score_issue_lines(metrics: Any) -> list[str]:
    if not isinstance(metrics, dict):
        return []
    status = str(metrics.get("rule_score_status") or "estimated").strip().lower()
    if status == "valid":
        return []
    lines = [
        (
            "Der Regel-Score ist nur ein Schätzwert und wird nicht für Bestwerte, Referenzen oder Lernen genutzt."
            if status == "estimated"
            else "Für diesen Sprung gibt es keinen gültigen Regel-Score; er wird nicht für Bestwerte, Referenzen oder Lernen genutzt."
        )
    ]
    raw_reasons = metrics.get("rule_score_reasons")
    if isinstance(raw_reasons, str):
        try:
            raw_reasons = json.loads(raw_reasons)
        except json.JSONDecodeError:
            raw_reasons = []
    reasons = {str(item) for item in (raw_reasons or [])}
    mapping = {
        "NO_RULE_WINDOW": "Kein vollständiges regelkonformes 3s-Fenster vorhanden.",
        "PERFORMANCE_WINDOW_INCOMPLETE": "Das Performance-Fenster ist unvollständig oder beginnt bereits unterhalb der Breakoff-Höhe.",
        "VALIDATION_ACCURACY_FAILED": "Die GPS-Geschwindigkeitsgenauigkeit überschreitet im Validierungsfenster den Grenzwert.",
        "VALIDATION_ACCURACY_UNKNOWN": "Die GPS-Geschwindigkeitsgenauigkeit ist im Validierungsfenster nicht vollständig bekannt.",
        "GROUND_LEVEL_ESTIMATED": "Die Bodenhöhe wurde geschätzt; AGL- und Fenstergrenzen sind nicht offiziell belastbar.",
        "LOW_SAMPLE_RATE": "Die Samplingrate ist für einen belastbaren Regel-Score zu niedrig.",
    }
    for reason in sorted(reasons):
        text = mapping.get(reason)
        if text:
            lines.append(text)
    return lines


def _build_fs2_quality_issue_lines(notes: dict[str, Any]) -> list[str]:
    if not isinstance(notes, dict):
        return []
    fs2 = notes.get("fs2_track_summary", {})
    if not isinstance(fs2, dict) or not bool(fs2.get("available")):
        return []

    label = str(fs2.get("quality_label") or "").strip().lower()
    if label not in {"kritisch", "grenzwertig"}:
        # stabile FS2-Werte nicht extra anzeigen
        return []

    start = _to_float(fs2.get("window_start_s"))
    end = _to_float(fs2.get("window_end_s"))
    sacc = _to_float(fs2.get("sAcc_p95"))
    hacc = _to_float(fs2.get("hAcc_p95"))
    sv = _to_float(fs2.get("numSV_p10"))

    if start is not None and end is not None:
        window_txt = f"+{start:.1f}s bis +{end:.1f}s"
    else:
        window_txt = "letzte schnelle Phase"

    level = "kritisch" if label == "kritisch" else "grenzwertig"
    if None not in {sacc, hacc, sv}:
        return [
            "Messqualität (FS2): "
            f"{level} in der schnellen Phase ({window_txt}) "
            f"(seitliche Schwankung {_fmt2(sacc)} m/s, Höhen-Unschärfe {_fmt1(hacc)} m, Satellitenreserve {_fmt1(sv)})."
        ]
    return [f"Messqualität (FS2): {level} in der schnellen Phase ({window_txt})."]


def _build_hot_zone_assessment(
    *,
    notes: dict[str, Any],
    metrics: dict[str, Any],
    scorecard: dict[str, Any],
    chart: dict[str, Any],
) -> dict[str, Any]:
    eval_end = effective_eval_window_end_s(metrics=metrics, notes=notes, chart_data=chart)
    if eval_end is None:
        curve_end = _to_float(notes.get("curve_window_end_s"))
        eval_end = 25.0 if curve_end is None else float(curve_end)
    hot_start_s, hot_end_s = _tau_to_time(tau0=0.70, tau1=0.90, eval_end_s=eval_end)
    hot_coverage = _window_coverage_ratio(
        time_s=chart.get("time_s", []),
        start_s=hot_start_s,
        end_s=hot_end_s,
    )
    window_end = hot_end_s

    dur_400 = _duration_above_threshold(
        time_s=chart.get("time_s", []),
        values=chart.get("vVert_kmh", []),
        threshold=400.0,
        start_s=hot_start_s,
        end_s=window_end,
    )
    dur_390 = _duration_above_threshold(
        time_s=chart.get("time_s", []),
        values=chart.get("vVert_kmh", []),
        threshold=390.0,
        start_s=hot_start_s,
        end_s=window_end,
    )
    vhor_min_20_25 = _window_min(
        time_s=chart.get("time_s", []),
        values=chart.get("vHor_kmh", []),
        start_s=hot_start_s,
        end_s=window_end,
    )
    angle_turns_20_25 = _window_turn_count(
        time_s=chart.get("time_s", []),
        values=chart.get("angle_deg", []),
        start_s=hot_start_s,
        end_s=window_end,
        eps=0.35,
    )
    vvert_gain_20_25 = _window_delta(
        time_s=chart.get("time_s", []),
        values=chart.get("vVert_kmh", []),
        start_s=hot_start_s,
        end_s=window_end,
    )
    hot_label = str(scorecard.get("hot_zone") or "")

    # Speed result in hot zone (0..100)
    speed_score = 0
    if dur_400 is None:
        speed_score = 45
    elif dur_400 >= 1.5:
        speed_score = 92
    elif dur_400 >= 0.8:
        speed_score = 78
    elif dur_400 >= 0.3:
        speed_score = 63
    else:
        speed_score = 46
    if dur_390 is not None:
        if dur_390 >= 4.0:
            speed_score += 8
        elif dur_390 >= 2.0:
            speed_score += 4
        else:
            speed_score -= 4
    speed_score = _clamp_score(speed_score)

    # Stability in hot zone (0..100)
    stability_score = 50
    if vhor_min_20_25 is not None:
        if vhor_min_20_25 >= 32.0:
            stability_score += 26
        elif vhor_min_20_25 >= 28.0:
            stability_score += 18
        elif vhor_min_20_25 >= 25.0:
            stability_score += 10
        else:
            stability_score -= 12
    if angle_turns_20_25 is not None:
        if angle_turns_20_25 <= 2:
            stability_score += 10
        elif angle_turns_20_25 <= 4:
            stability_score += 3
        else:
            stability_score -= 8
    if hot_label == "kritisch":
        stability_score -= 8
    elif hot_label in {"stabil", "sehr gut"}:
        stability_score += 3
    stability_score = _clamp_score(stability_score)

    # Combined hot-zone score: speed result + stability
    combined = _clamp_score((speed_score * 0.6) + (stability_score * 0.4))
    if hot_coverage < 0.75:
        combined = max(combined, 60)

    # Build simple-language explanation.
    reason_lines: list[str] = []
    if hot_coverage < 0.75:
        reason_lines.append("Die Hot-Zone ist in diesem Sprung nicht vollständig abgedeckt.")

    if dur_400 is not None:
        if dur_400 < 0.3:
            reason_lines.append("Du erreichst die sehr schnelle Phase, kannst sie aber kaum halten.")
        elif dur_400 < 1.0:
            reason_lines.append("Du kommst in die sehr schnelle Phase, hältst sie aber nur kurz.")
        else:
            reason_lines.append("Du kannst die sehr schnelle Phase spuerbar halten.")

    if vhor_min_20_25 is not None:
        if vhor_min_20_25 < 20.0:
            reason_lines.append("In der Endphase faellt die waagerechte Geschwindigkeit fast komplett weg.")
        elif vhor_min_20_25 < 25.0:
            reason_lines.append("In der Endphase faellt die waagerechte Geschwindigkeit deutlich ab.")
        elif vhor_min_20_25 < 30.0:
            reason_lines.append("In der Endphase wird die horizontale Reserve knapp.")
        else:
            reason_lines.append("Die horizontale Reserve bleibt in der Endphase stabil.")

    if angle_turns_20_25 is not None:
        if angle_turns_20_25 >= 5:
            reason_lines.append("In der heißen Zone musst du mehrfach nachkorrigieren.")
        elif angle_turns_20_25 >= 3:
            reason_lines.append("In der heißen Zone gibt es einige Nachkorrekturen.")

    if vvert_gain_20_25 is not None and vvert_gain_20_25 < 8.0:
        reason_lines.append("In der Hot-Zone kommt kaum noch zusätzlicher Speed dazu.")

    hot_label = f"Hot-Zone +{hot_start_s:.1f}s bis +{hot_end_s:.1f}s"
    reason_lines.append(
        f"In deiner {hot_label} konntest du >400 km/h für {_fmt1(dur_400)}s halten. "
        f"Die kleinste horizontale Reserve lag bei {_fmt1(vhor_min_20_25)} km/h, "
        f"der zusätzliche Speed in dieser Phase bei {_fmt1(vvert_gain_20_25)} km/h."
    )
    impact_parts: list[str] = []
    if dur_400 is not None:
        if dur_400 < 0.3:
            impact_parts.append("die sehr schnelle Zone bricht sofort wieder weg")
        elif dur_400 < 1.0:
            impact_parts.append("die sehr schnelle Zone ist noch zu kurz")
    if vhor_min_20_25 is not None:
        if vhor_min_20_25 < 20.0:
            impact_parts.append("die horizontale Reserve faellt fast komplett weg und grosse Gegenkorrekturen werden wahrscheinlicher")
        elif vhor_min_20_25 < 25.0:
            impact_parts.append("die horizontale Reserve ist zu knapp und die Linie wird leichter unruhig")
    if vvert_gain_20_25 is not None and vvert_gain_20_25 < 8.0:
        impact_parts.append("im letzten schnellen Abschnitt entsteht kaum noch zusätzlicher Speed")
    if impact_parts:
        reason_lines.append(f"Das bedeutet: {'; '.join(impact_parts)}.")
    reason = _join_reason_lines(reason_lines)

    return {
        "score": combined,
        "reason": reason,
        "reason_lines": reason_lines,
        "speed_score": speed_score,
        "stability_score": stability_score,
    }


def _early_horizontal_reserve_context(*, chart: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    best_start = _to_float(metrics.get("best_3s_start_s"))
    best_end = _to_float(metrics.get("best_3s_end_s"))
    time_s = chart.get("time_s", []) or []
    vhor_values = chart.get("vHor_kmh", []) or []
    angle_values = chart.get("angle_deg", []) or []
    if (
        best_start is None
        or best_end is None
        or best_end <= best_start
        or not time_s
        or not vhor_values
        or len(time_s) != len(vhor_values)
    ):
        return {"available": False, "collapse": False}

    search_start = max(18.0, float(best_start) - 7.0)
    search_end = float(best_end)
    samples: list[tuple[float, float]] = []
    for i, raw_t in enumerate(time_s):
        t = _to_float(raw_t)
        v = _to_float(vhor_values[i])
        if t is None or v is None:
            continue
        if search_start <= t <= search_end:
            samples.append((float(t), float(v)))
    if len(samples) < 4:
        return {"available": False, "collapse": False}

    min_t, min_vhor = min(samples, key=lambda item: item[1])
    vhor_at_best_start = _series_value_at(time_s=time_s, values=vhor_values, target_s=float(best_start))
    vhor_at_best_end = _series_value_at(time_s=time_s, values=vhor_values, target_s=float(best_end))
    rebound_to_best_end_pct = None
    if vhor_at_best_end is not None and min_vhor > 1e-6:
        rebound_to_best_end_pct = max(0.0, (float(vhor_at_best_end) - min_vhor) / min_vhor * 100.0)

    angle_at_min = None
    angle_peak_near_min = None
    if angle_values and len(angle_values) == len(time_s):
        angle_at_min = _series_value_at(time_s=time_s, values=angle_values, target_s=min_t)
        near_angles: list[float] = []
        for i, raw_t in enumerate(time_s):
            t = _to_float(raw_t)
            angle = _to_float(angle_values[i])
            if t is None or angle is None:
                continue
            if abs(float(t) - min_t) <= 1.2:
                near_angles.append(float(angle))
        if near_angles:
            angle_peak_near_min = max(near_angles)

    forward_loss_to_best_end = None
    forward_peak_s = None
    forward_values = chart.get("forward_m", []) or []
    if forward_values and len(forward_values) == len(time_s):
        forward_samples: list[tuple[float, float]] = []
        for i, raw_t in enumerate(time_s):
            t = _to_float(raw_t)
            v = _to_float(forward_values[i])
            if t is None or v is None:
                continue
            if max(0.0, float(best_start) - 8.0) <= t <= float(best_end):
                forward_samples.append((float(t), float(v)))
        if len(forward_samples) >= 4:
            forward_peak_s, forward_peak = max(forward_samples, key=lambda item: item[1])
            forward_at_best_end = _series_value_at(time_s=time_s, values=forward_values, target_s=float(best_end))
            if forward_at_best_end is not None:
                forward_loss_to_best_end = max(0.0, float(forward_peak) - float(forward_at_best_end))

    forward_reversal = bool(
        forward_loss_to_best_end is not None
        and forward_loss_to_best_end >= 8.0
        and forward_peak_s is not None
        and forward_peak_s <= float(best_start) + 0.5
        and abs(forward_peak_s - min_t) <= 2.5
    )
    steep_at_min = bool(
        (angle_at_min is not None and angle_at_min >= 85.5)
        or (angle_peak_near_min is not None and angle_peak_near_min >= 86.0)
    )
    collapse = bool(
        min_t <= float(best_start) - 0.7
        and min_vhor <= 30.0
        and rebound_to_best_end_pct is not None
        and rebound_to_best_end_pct >= 25.0
        and (steep_at_min or forward_reversal)
    )
    return {
        "available": True,
        "collapse": collapse,
        "vhor_min_time_s": min_t,
        "vhor_min_kmh": min_vhor,
        "best_3s_start_s": float(best_start),
        "best_3s_end_s": float(best_end),
        "vhor_at_best_start_kmh": vhor_at_best_start,
        "vhor_at_best_end_kmh": vhor_at_best_end,
        "vhor_rebound_to_best_end_pct": rebound_to_best_end_pct,
        "angle_at_vhor_min_deg": angle_at_min,
        "angle_peak_near_vhor_min_deg": angle_peak_near_min,
        "forward_reversal_near_vhor_min": forward_reversal,
        "forward_peak_s": forward_peak_s,
        "forward_loss_to_best_end_m": forward_loss_to_best_end,
    }


def _early_horizontal_reserve_reason_line(reserve_context: dict[str, Any] | None) -> str:
    if not isinstance(reserve_context, dict) or not reserve_context.get("collapse"):
        return ""
    min_t = _to_float(reserve_context.get("vhor_min_time_s"))
    min_vhor = _to_float(reserve_context.get("vhor_min_kmh"))
    best_start = _to_float(reserve_context.get("best_3s_start_s"))
    best_end = _to_float(reserve_context.get("best_3s_end_s"))
    rebound = _to_float(reserve_context.get("vhor_rebound_to_best_end_pct"))
    parts: list[str] = []
    if min_t is not None and min_vhor is not None:
        parts.append(f"vHor-Min +{min_t:.1f}s bei {_fmt1(min_vhor)} km/h")
    if best_start is not None and best_end is not None:
        parts.append(f"vor dem 3s-Fenster +{best_start:.1f}s bis +{best_end:.1f}s")
    if rebound is not None:
        parts.append(f"danach vHor-Rebound {_fmt1(rebound)}%")
    if not parts:
        return "Horizontale Reserve wird vor dem 3s-Fenster zu frueh verbraucht."
    return "Horizontale Reserve zu frueh verbraucht: " + ", ".join(parts) + "."


def _series_value_at(*, time_s: list[Any], values: list[Any], target_s: float) -> float | None:
    if not time_s or not values or len(time_s) != len(values):
        return None
    pairs: list[tuple[float, float]] = []
    for i, raw_t in enumerate(time_s):
        t = _to_float(raw_t)
        v = _to_float(values[i])
        if t is None or v is None:
            continue
        pairs.append((float(t), float(v)))
    if not pairs:
        return None
    pairs.sort(key=lambda item: item[0])
    if target_s < pairs[0][0] or target_s > pairs[-1][0]:
        return None
    for idx, (t, v) in enumerate(pairs):
        if abs(t - target_s) < 1e-9:
            return v
        if t > target_s:
            prev_t, prev_v = pairs[idx - 1]
            if abs(t - prev_t) < 1e-9:
                return v
            ratio = (target_s - prev_t) / (t - prev_t)
            return float(prev_v + (v - prev_v) * ratio)
    return pairs[-1][1]


def _window_min(
    *,
    time_s: list[Any],
    values: list[Any],
    start_s: float,
    end_s: float,
) -> float | None:
    if not time_s or not values or len(time_s) != len(values) or end_s <= start_s:
        return None
    mins: list[float] = []
    for i, raw_t in enumerate(time_s):
        t = _to_float(raw_t)
        v = _to_float(values[i])
        if t is None or v is None:
            continue
        if start_s <= t <= end_s:
            mins.append(v)
    if not mins:
        return None
    return float(min(mins))


def _window_max(
    *,
    time_s: list[Any],
    values: list[Any],
    start_s: float,
    end_s: float,
) -> float | None:
    if not time_s or not values or len(time_s) != len(values) or end_s <= start_s:
        return None
    vals: list[float] = []
    for i, raw_t in enumerate(time_s):
        t = _to_float(raw_t)
        v = _to_float(values[i])
        if t is None or v is None:
            continue
        if start_s <= t <= end_s:
            vals.append(v)
    if not vals:
        return None
    return float(max(vals))


def _window_mean(
    *,
    time_s: list[Any],
    values: list[Any],
    start_s: float,
    end_s: float,
) -> float | None:
    if not time_s or not values or len(time_s) != len(values) or end_s <= start_s:
        return None
    vals: list[float] = []
    for i, raw_t in enumerate(time_s):
        t = _to_float(raw_t)
        v = _to_float(values[i])
        if t is None or v is None:
            continue
        if start_s <= t <= end_s:
            vals.append(v)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _window_turn_count(
    *,
    time_s: list[Any],
    values: list[Any],
    start_s: float,
    end_s: float,
    eps: float,
) -> int | None:
    if not time_s or not values or len(time_s) != len(values) or end_s <= start_s:
        return None
    seq: list[float] = []
    for i, raw_t in enumerate(time_s):
        t = _to_float(raw_t)
        v = _to_float(values[i])
        if t is None or v is None:
            continue
        if start_s <= t <= end_s:
            seq.append(v)
    if len(seq) < 4:
        return None
    turns = 0
    prev_sign = 0
    for i in range(1, len(seq)):
        d = seq[i] - seq[i - 1]
        sign = 1 if d > eps else -1 if d < -eps else 0
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            turns += 1
        prev_sign = sign
    return turns


def _window_delta(
    *,
    time_s: list[Any],
    values: list[Any],
    start_s: float,
    end_s: float,
) -> float | None:
    if not time_s or not values or len(time_s) != len(values) or end_s <= start_s:
        return None
    samples: list[tuple[float, float]] = []
    for i, raw_t in enumerate(time_s):
        t = _to_float(raw_t)
        v = _to_float(values[i])
        if t is None or v is None:
            continue
        if start_s <= t <= end_s:
            samples.append((t, v))
    if len(samples) < 2:
        return None
    return float(samples[-1][1] - samples[0][1])


def _duration_above_threshold(
    *,
    time_s: list[Any],
    values: list[Any],
    threshold: float,
    start_s: float,
    end_s: float,
) -> float | None:
    if not time_s or not values or len(time_s) != len(values) or end_s <= start_s:
        return None

    samples: list[tuple[float, float]] = []
    for i, raw_t in enumerate(time_s):
        t = _to_float(raw_t)
        v = _to_float(values[i])
        if t is None or v is None:
            continue
        if start_s <= t <= end_s:
            samples.append((t, v))
    if len(samples) < 2:
        return None

    dur = 0.0
    for i in range(1, len(samples)):
        t0, v0 = samples[i - 1]
        t1, v1 = samples[i]
        if v0 >= threshold and v1 >= threshold:
            dur += (t1 - t0)
    return float(dur)


def _estimate_carry_ratio(chart_data: dict[str, Any]) -> float | None:
    time_s = chart_data.get("time_s", []) or []
    acc = chart_data.get("accVert_mps2", []) or []
    if not time_s or not acc or len(time_s) != len(acc):
        return None

    early: list[float] = []
    mid: list[float] = []
    for i, raw_t in enumerate(time_s):
        t = _to_float(raw_t)
        a = _to_float(acc[i])
        if t is None or a is None:
            continue
        if 0.0 <= t <= 2.0:
            early.append(a)
        elif 2.0 < t <= 6.0:
            mid.append(a)

    if len(early) < 2 or len(mid) < 2:
        return None
    early_mean = sum(early) / len(early)
    mid_mean = sum(mid) / len(mid)
    if abs(early_mean) < 1e-6:
        return None
    return float(mid_mean / early_mean)


def _parse_negative_details(text: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "vhor_dip_pct": None,
        "vhor_min_kmh": None,
        "rebound_pct": None,
        "angle_max_deg": None,
    }
    if not text:
        return out

    patterns = {
        "vhor_dip_pct": r"vHor-Dip\s*([0-9]+(?:\.[0-9]+)?)%",
        "vhor_min_kmh": r"vHor-Min\s*([0-9]+(?:\.[0-9]+)?)\s*km/h",
        "rebound_pct": r"Rebound\s*([0-9]+(?:\.[0-9]+)?)%",
        "angle_max_deg": r"Winkel-Max\s*([0-9]+(?:\.[0-9]+)?)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            out[key] = float(m.group(1))
        except Exception:
            out[key] = None
    return out


def _describe_stability_reason_lines(*, raw_negative_details: str, risk_label: str) -> list[str]:
    parsed = _parse_negative_details(raw_negative_details)
    dip = parsed.get("vhor_dip_pct")
    vmin = parsed.get("vhor_min_kmh")
    rebound = parsed.get("rebound_pct")
    angle = parsed.get("angle_max_deg")

    if dip is None and vmin is None and rebound is None and angle is None:
        if risk_label == "hoch":
            return ["In der schnellen Phase ist die Linie instabil und es gibt klare Nachkorrekturen."]
        if risk_label == "mittel":
            return ["In der schnellen Phase gibt es erkennbare Nachkorrekturen."]
        return ["Der Verlauf ist insgesamt stabil."]

    lines: list[str] = []
    if (dip is not None and dip >= 30.0) or (vmin is not None and vmin < 24.0):
        lines.append("In der schnellen Phase faellt die waagerechte Geschwindigkeit zu stark ab.")
    elif (dip is not None and dip >= 20.0) or (vmin is not None and vmin < 28.0):
        lines.append("In der schnellen Phase faellt die waagerechte Geschwindigkeit merklich ab.")
    else:
        lines.append("Die waagerechte Geschwindigkeit bleibt ueber weite Strecken stabil.")

    if rebound is not None:
        if rebound >= 120.0:
            lines.append("Danach musst du stark gegensteuern, dadurch wird die Linie unruhig.")
        elif rebound >= 70.0:
            lines.append("Danach folgt eine sichtbare Gegenkorrektur.")

    if angle is not None and angle >= 86.5:
        lines.append("Der Tauchwinkel wird zeitweise zu steil, dadurch steigt das Kipp-Risiko.")
    elif angle is not None:
        lines.append("Der Winkel ist nicht das Hauptproblem, entscheidend ist die Stabilität der Linie.")

    lines.append(
        "Messwerte: vHor-Dip "
        f"{_fmt1(dip)}%, vHor-Min {_fmt1(vmin)} km/h, Rebound {_fmt1(rebound)}%, Winkel-Max {_fmt1(angle)} Grad."
    )
    return lines


def _describe_stability_reason(*, raw_negative_details: str, risk_label: str) -> str:
    return _join_reason_lines(
        _describe_stability_reason_lines(
            raw_negative_details=raw_negative_details,
            risk_label=risk_label,
        )
    )


def _render_index_with_error(request: Request, error: str, view_mode: str | None = None):
    jumpers = list_jumpers()
    resolved_view_mode = _normalize_view_mode(view_mode or request.query_params.get("view"))
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "message": None,
            "error": error,
            "view_mode": resolved_view_mode,
            "recent_jumps": list_recent_jumps(limit=_INDEX_RECENT_JUMPS_LIMIT),
            "jumpers": jumpers,
            "show_jumpers_panel": False,
            "jumper_overview": [],
            "needs_plotly": False,
        },
        status_code=400,
    )


def _cache_uploaded_file(*, source_hash: str, original_name: str, content: bytes) -> Path:
    ext = Path(original_name).suffix or ".csv"
    cache_path = RAW_UPLOAD_DIR / f"{source_hash}{ext.lower()}"
    if not cache_path.exists():
        cache_path.write_bytes(content)
    return cache_path


def _resolve_source_file_for_jump(source_meta: dict[str, Any]) -> Path | None:
    source_path = source_meta.get("source_file_path")
    if source_path:
        path = Path(source_path)
        if path.exists():
            return path

    source_hash = source_meta.get("source_file_sha256")
    if source_hash:
        for ext in [".csv", ".CSV"]:
            candidate = RAW_UPLOAD_DIR / f"{source_hash}{ext}"
            if candidate.exists():
                return candidate

    file_name = source_meta.get("file_name")
    if file_name:
        candidate = Path.home() / "Downloads" / str(file_name)
        if candidate.exists():
            return candidate

    return None


def _manual_t0_utc_from_current(current_t0_utc: Any, rel_s: float) -> str:
    text = str(current_t0_utc or "").strip()
    if not text:
        raise AnalysisError("Aktueller t0-Zeitpunkt fehlt.")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        current = datetime.fromisoformat(text)
    except ValueError as exc:
        raise AnalysisError("Aktueller t0-Zeitpunkt ist ungueltig.") from exc
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return (current + timedelta(seconds=float(rel_s))).isoformat()


def _annotate_best_jump(jumps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not jumps:
        return jumps
    eligible = [item for item in jumps if _is_jump_eligible_for_best(item)]
    for jump in jumps:
        jump["is_best"] = False
    if not eligible:
        return jumps
    best_jump = max(
        eligible,
        key=lambda item: (
            float(item["rule_based_3s_score"])
            if item.get("rule_based_3s_score") is not None
            else float("-inf")
        ),
    )
    best_id = best_jump["jump_id"]
    for jump in jumps:
        jump["is_best"] = jump["jump_id"] == best_id
    return jumps


def _is_jump_eligible_for_best(jump: dict[str, Any]) -> bool:
    flags = _parse_quality_flags(jump.get("quality_flags"))
    return bool(
        jump.get("rule_score_status") in {None, "valid"}
        and _to_float(jump.get("rule_based_3s_score")) is not None
        and not flags
        & {
            "EARLY_JUMP_END",
            "SPEED_SPIKE",
            "TIME_GAPS",
            "NO_CLEAR_EXIT",
            "INVALID_EXIT_ALTITUDE",
            "LOW_GPS_FIX",
            "HIGH_SPEED_ACCURACY_ERROR",
        }
    )


def _pick_best_history_reference(jumper_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    rows = list_jumps_for_jumper(jumper_name)
    if not rows:
        return None, None

    clean_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    fallback_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        jump_id = str(row.get("jump_id") or "")
        if not jump_id:
            continue
        report = get_jump_report(jump_id)
        if report is None:
            continue
        pair = (row, report)
        fallback_candidates.append(pair)
        if _is_clean_reference_jump(row, report):
            clean_candidates.append(pair)

    pool = clean_candidates
    if not pool:
        return None, None

    def _sort_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, float]:
        row, report = item
        speed = _to_float(report.get("metrics", {}).get("rule_based_3s_score"))
        return (
            float("-inf") if speed is None else float(speed),
            _t0_sort_key(row.get("t0_utc")),
        )

    best_row, best_report = max(pool, key=_sort_key)
    return best_row, best_report


def _is_clean_reference_jump(jump_row: dict[str, Any], report: dict[str, Any]) -> bool:
    flags = _parse_quality_flags(jump_row.get("quality_flags"))
    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    if not is_reference_eligible(metrics=metrics, notes=notes, quality_flags=flags):
        return False
    if not _has_complete_reference_window(report, start_s=0.0, end_s=25.0):
        return False
    return True


def _has_complete_reference_window(report: dict[str, Any], *, start_s: float, end_s: float) -> bool:
    chart = report.get("chart_data", {}) if isinstance(report.get("chart_data"), dict) else {}
    raw_time = chart.get("time_s", []) or []

    times: list[float] = []
    for raw in raw_time:
        value = _to_float(raw)
        if value is None:
            continue
        times.append(float(value))
    if len(times) < 3:
        return False

    times = sorted(times)
    dts: list[float] = []
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt > 0:
            dts.append(dt)
    if not dts:
        return False

    med_dt = float(_percentile(sorted(dts), 0.5))
    gap_threshold = max(0.6, med_dt * 2.5)
    start_tol = max(0.5, med_dt * 2.0)
    end_tol = max(0.5, med_dt * 2.0)

    window_samples = [t for t in times if float(start_s) <= t <= float(end_s)]
    if not window_samples:
        return False

    first = min(window_samples)
    last = max(window_samples)
    if first > float(start_s) + start_tol:
        return False
    if last < float(end_s) - end_tol:
        return False

    expected_count = ((float(end_s) - float(start_s)) / med_dt) + 1.0 if med_dt > 0 else float(len(window_samples))
    if expected_count > 0 and (len(window_samples) / expected_count) < 0.85:
        return False

    for i in range(1, len(times)):
        left = times[i - 1]
        right = times[i]
        dt = right - left
        if dt <= gap_threshold:
            continue
        if left < float(end_s) and right > float(start_s):
            return False

    return True


def _parse_quality_flags(raw_flags: Any) -> set[str]:
    if raw_flags is None:
        return set()
    try:
        return set(json.loads(raw_flags)) if isinstance(raw_flags, str) else set(raw_flags)
    except Exception:
        return set()


def _coach_status(report: dict[str, Any]) -> dict[str, str]:
    scorecard = report.get("scorecard", {})
    notes = report.get("notes", {})
    metrics = report.get("metrics", {})
    quality_issues = _build_quality_issue_lines(report.get("quality_flags", []))
    quality_issues.extend(_build_rule_score_issue_lines(metrics))
    phase_statuses = {
        str(row.get("name") or ""): str(row.get("target_status") or "")
        for row in _phase_rows_for_report(report)
        if isinstance(row, dict)
    }

    if bool(notes.get("analysis_blocked")):
        return {"level": "red", "label": "Rot", "reason": "Sprung endet zu früh / Daten unplausibel"}
    if bool(notes.get("t0_review_required")):
        return {"level": "red", "label": "Rot", "reason": "Absprungzeit unsicher"}
    if metrics.get("rule_score_status") == "invalid":
        return {"level": "red", "label": "Rot", "reason": "Kein gültiger Regel-Score"}
    if scorecard.get("kipp_risiko") == "hoch" or scorecard.get("hot_zone") == "kritisch":
        return {"level": "red", "label": "Rot", "reason": "Instabile schnelle Phase"}

    if scorecard.get("kipp_risiko") == "mittel":
        return {"level": "yellow", "label": "Gelb", "reason": "Mittleres Stabilitätsrisiko"}
    if scorecard.get("phase_10_20") in {"zu flach", "zu steil"} or any(
        status in {"zu flach", "zu steil", "kurz zu steil"}
        for status in phase_statuses.values()
    ):
        return {"level": "yellow", "label": "Gelb", "reason": "Winkel im Aufbau noch unruhig"}
    if quality_issues:
        return {"level": "yellow", "label": "Gelb", "reason": "Datenhinweise vorhanden"}

    return {"level": "green", "label": "Gruen", "reason": "Stabiler Sprungverlauf"}


def _sort_by_t0_desc(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: _t0_sort_key(item.get("t0_utc")), reverse=True)


def _t0_sort_key(raw_t0: Any) -> float:
    if raw_t0 is None:
        return float("-inf")
    text = str(raw_t0).strip()
    if not text:
        return float("-inf")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("-inf")
