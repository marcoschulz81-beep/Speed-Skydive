from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.analysis.comparison import build_jump_comparison
from app.analysis.pipeline import AnalysisError, analyze_flysight_csv
from app.config import BASE_DIR
from app.database import init_db
from app.report_pdf import build_pdf
from app.services.storage import (
    get_jump_report,
    list_jumps_for_jumper,
    list_jumpers,
    list_recent_jumps,
    save_analysis_result,
)

app = FastAPI(title="Speed-Skydive Analyzer", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()


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
    jump_id, is_duplicate = save_analysis_result(result, source_file_sha256=source_hash)
    if is_duplicate:
        return RedirectResponse(url=f"/jumps/{jump_id}", status_code=303)
    return RedirectResponse(url=f"/jumps/{jump_id}", status_code=303)


@app.get("/jumps/{jump_id}")
def jump_detail(request: Request, jump_id: str):
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
