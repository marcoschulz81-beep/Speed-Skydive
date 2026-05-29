from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.analysis.pipeline import AnalysisError, analyze_flysight_csv
from app.analysis.comparison import build_jump_comparison
from app.config import BASE_DIR, RAW_UPLOAD_DIR
from app.database import init_db
from app.report_pdf import build_pdf
from app.services.storage import (
    get_best_jump_for_jumper,
    get_jump_report,
    get_jump_source_metadata,
    get_jump_summary,
    list_compare_candidates,
    list_jumps_for_jumper,
    list_jumpers,
    list_recent_jumps,
    replace_analysis_result,
    save_analysis_result,
)

app = FastAPI(title="Speed-Skydive Analyzer", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()
    RAW_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def index(request: Request, message: str | None = None, error: str | None = None):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "message": message,
            "error": error,
            "recent_jumps": list_recent_jumps(),
            "jumpers": list_jumpers(),
        },
    )


@app.post("/analyze")
async def analyze_upload(
    request: Request,
    jumper_name: str = Form(...),
    csv_file: UploadFile = File(...),
    ground_elevation_m: str = Form(""),
    breakoff_altitude_agl_m: str = Form(""),
):
    jumper = jumper_name.strip()
    if not jumper:
        return _render_index_with_error(request, "Springername ist erforderlich.")

    if not csv_file.filename.lower().endswith(".csv"):
        return _render_index_with_error(request, "Bitte eine CSV-Datei hochladen.")

    content = await csv_file.read()
    if not content:
        return _render_index_with_error(request, "Die hochgeladene Datei ist leer.")

    try:
        ground = _parse_optional_float(ground_elevation_m)
        breakoff = _parse_optional_float(breakoff_altitude_agl_m)
    except ValueError as exc:
        return _render_index_with_error(request, str(exc))

    try:
        result = analyze_flysight_csv(
            content=content,
            file_name=csv_file.filename,
            jumper_name=jumper,
            ground_elevation_m=ground,
            breakoff_altitude_agl_m=breakoff,
        )
    except AnalysisError as exc:
        return _render_index_with_error(request, str(exc))
    except Exception as exc:  # pragma: no cover
        return _render_index_with_error(request, f"Unerwarteter Analysefehler: {exc}")

    source_hash = hashlib.sha256(content).hexdigest()
    source_path = _cache_uploaded_file(source_hash=source_hash, original_name=csv_file.filename, content=content)
    jump_id, is_duplicate = save_analysis_result(
        result,
        source_file_sha256=source_hash,
        source_file_path=str(source_path),
    )
    if is_duplicate:
        return RedirectResponse(url=f"/jumps/{jump_id}", status_code=303)
    return RedirectResponse(url=f"/jumps/{jump_id}", status_code=303)


@app.get("/jumps/{jump_id}")
def jump_detail(request: Request, jump_id: str, message: str | None = None, error: str | None = None):
    report = get_jump_report(jump_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Sprung nicht gefunden.")

    return templates.TemplateResponse(
        request,
        "jump_detail.html",
        {
            "jump_id": jump_id,
            "report": report,
            "chart_data_json": json.dumps(report["chart_data"]),
            "quality_flags_json": json.dumps(report["quality_flags"]),
            "message": message,
            "error": error,
        },
    )


@app.post("/jumps/{jump_id}/reprocess-t0")
def reprocess_t0(jump_id: str):
    report = get_jump_report(jump_id)
    source_meta = get_jump_source_metadata(jump_id)
    if report is None or source_meta is None:
        raise HTTPException(status_code=404, detail="Sprung nicht gefunden.")

    source_path = _resolve_source_file_for_jump(source_meta)
    if source_path is None or not source_path.exists():
        msg = "Original-CSV nicht gefunden. Bitte Datei erneut hochladen."
        return RedirectResponse(url=f"/jumps/{jump_id}?error={quote_plus(msg)}", status_code=303)

    try:
        content = source_path.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        cached_path = _cache_uploaded_file(
            source_hash=source_hash,
            original_name=report["jump"]["file_name"],
            content=content,
        )
        new_result = analyze_flysight_csv(
            content=content,
            file_name=report["jump"]["file_name"],
            jumper_name=report["jump"]["jumper_name"],
            ground_elevation_m=report["jump"].get("ground_elevation_m"),
            breakoff_altitude_agl_m=None,
        )
        replace_analysis_result(
            jump_id=jump_id,
            result=new_result,
            source_file_sha256=source_hash,
            source_file_path=str(cached_path),
        )
    except AnalysisError as exc:
        return RedirectResponse(url=f"/jumps/{jump_id}?error={quote_plus(str(exc))}", status_code=303)
    except Exception as exc:  # pragma: no cover
        msg = f"Unerwarteter Reanalysefehler: {exc}"
        return RedirectResponse(url=f"/jumps/{jump_id}?error={quote_plus(msg)}", status_code=303)

    ok_msg = "Absprung wurde neu erkannt und der Datensatz aktualisiert."
    return RedirectResponse(url=f"/jumps/{jump_id}?message={quote_plus(ok_msg)}", status_code=303)


@app.get("/jumps/{jump_id}/compare")
def jump_compare(
    request: Request,
    jump_id: str,
    compare_jump_id: str | None = None,
    preset: str | None = None,
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
            compare_error = "Bitte einen anderen Sprung als Vergleich auswaehlen."
        else:
            compare_report = get_jump_report(compare_jump_id)
            if compare_report is None:
                compare_error = "Vergleichssprung nicht gefunden."
            else:
                compare_result = build_jump_comparison(
                    left_report=base_report,
                    right_report=compare_report,
                )
                compare_chart_json = json.dumps(compare_result["charts"])

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
            "compare_chart_json": compare_chart_json,
            "selected_compare_jump_id": compare_jump_id,
        },
    )


@app.get("/jumps/{jump_id}/report.pdf")
def jump_pdf(jump_id: str):
    report = get_jump_report(jump_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Sprung nicht gefunden.")
    pdf_bytes = build_pdf(report)
    file_name = f"speed_report_{jump_id}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{file_name}"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@app.get("/jumpers/{jumper_name}")
def jumper_view(request: Request, jumper_name: str):
    jumps = list_jumps_for_jumper(jumper_name)
    if not jumps:
        raise HTTPException(status_code=404, detail="Springer nicht gefunden.")

    jumps = _annotate_best_jump(jumps)
    return templates.TemplateResponse(
        request,
        "jumper_detail.html",
        {
            "jumper_name": jumper_name,
            "jumps": jumps,
            "compare_result": None,
            "compare_chart_json": None,
            "compare_error": None,
            "left_jump_id": None,
            "right_jump_id": None,
        },
    )


@app.get("/jumpers/{jumper_name}/compare")
def jumper_compare(request: Request, jumper_name: str, left_jump_id: str, right_jump_id: str):
    jumps = list_jumps_for_jumper(jumper_name)
    if not jumps:
        raise HTTPException(status_code=404, detail="Springer nicht gefunden.")

    jumps = _annotate_best_jump(jumps)
    jump_ids = {jump["jump_id"] for jump in jumps}
    compare_error: str | None = None
    compare_result: dict[str, Any] | None = None
    compare_chart_json: str | None = None

    if left_jump_id == right_jump_id:
        compare_error = "Bitte zwei unterschiedliche Spruenge auswaehlen."
    elif left_jump_id not in jump_ids or right_jump_id not in jump_ids:
        compare_error = "Vergleich ungueltig: Spruenge gehoeren nicht zu diesem Springer."
    else:
        left_report = get_jump_report(left_jump_id)
        right_report = get_jump_report(right_jump_id)
        if left_report is None or right_report is None:
            compare_error = "Vergleich ungueltig: Mindestens ein Sprung wurde nicht gefunden."
        else:
            compare_result = build_jump_comparison(left_report=left_report, right_report=right_report)
            compare_chart_json = json.dumps(compare_result["charts"])

    return templates.TemplateResponse(
        request,
        "jumper_detail.html",
        {
            "jumper_name": jumper_name,
            "jumps": jumps,
            "compare_result": compare_result,
            "compare_chart_json": compare_chart_json,
            "compare_error": compare_error,
            "left_jump_id": left_jump_id,
            "right_jump_id": right_jump_id,
        },
    )


def _parse_optional_float(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Ungueltiger Zahlenwert: {raw}") from exc


def _render_index_with_error(request: Request, error: str):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "message": None,
            "error": error,
            "recent_jumps": list_recent_jumps(),
            "jumpers": list_jumpers(),
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


def _annotate_best_jump(jumps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not jumps:
        return jumps
    best_jump = max(
        jumps,
        key=lambda item: float(item["best_3s_vVert_kmh"]) if item["best_3s_vVert_kmh"] is not None else float("-inf"),
    )
    best_id = best_jump["jump_id"]
    for jump in jumps:
        jump["is_best"] = jump["jump_id"] == best_id
    return jumps
