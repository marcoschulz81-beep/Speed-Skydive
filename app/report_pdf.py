from __future__ import annotations

import io
from typing import Any

from fpdf import FPDF


def _line(pdf: FPDF, text: str, h: int = 6) -> None:
    pdf.multi_cell(0, h, text)


def build_pdf(report: dict[str, Any]) -> bytes:
    jump = report["jump"]
    metrics = report["metrics"]
    fixpoints = report["fixpoints"]
    phases = report["phases"]
    tips = report["tips"]
    scorecard = report["scorecard"]
    notes = report["notes"]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    _line(pdf, "FlySight Speed-Skydiving-Auswertung")
    pdf.set_font("Helvetica", size=10)
    _line(pdf, f"Springer: {jump['jumper_name']}")
    _line(pdf, f"Datei: {jump['file_name']}")
    _line(pdf, f"t0 UTC: {jump['t0_utc']}")
    _line(pdf, f"Qualitaetsscore: {jump['quality_score']}")
    _line(pdf, f"Exit AGL: {jump['exit_altitude_agl_m']} m")
    _line(pdf, f"Exit gueltig <=4267.2m: {'Ja' if jump['is_valid_altitude'] else 'Nein'}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    _line(pdf, "Speed Score")
    pdf.set_font("Helvetica", size=10)
    _line(pdf, f"Bestes 3s-Fenster: +{metrics['best_3s_start_s']}s bis +{metrics['best_3s_end_s']}s")
    _line(pdf, f"Training 3s Max: {metrics['best_3s_vVert_kmh']} km/h ({metrics['best_3s_vVert_mps']} m/s)")
    _line(pdf, f"Regelnaher 3s-Score: {metrics['rule_based_3s_score']} km/h")
    _line(
        pdf,
        f"Performance Window: {metrics['performance_window_start_s']}s bis {metrics['performance_window_end_s']}s",
    )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    _line(pdf, "Fixpunkte")
    pdf.set_font("Helvetica", size=10)
    for point in fixpoints:
        _line(
            pdf,
            f"+{point['t_rel_s']}s: vVert={_fmt(point['vVert_kmh'])} km/h, "
            f"vHor={_fmt(point['vHor_kmh'])} km/h, Winkel={_fmt(point['angle_deg'])}°, "
            f"hAGL={_fmt(point['hAGL_m'])} m",
        )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    _line(pdf, "Phasen")
    pdf.set_font("Helvetica", size=10)
    for phase in phases:
        _line(
            pdf,
            f"{phase['name']}: +{phase['start_s']}s..+{phase['end_s']}s, "
            f"ØvVert={_fmt(phase['avg_vVert_kmh'])} km/h, ØWinkel={_fmt(phase['avg_angle_deg'])}°, "
            f"Kommentar: {phase['comment']}",
        )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    _line(pdf, "Scorecard")
    pdf.set_font("Helvetica", size=10)
    for key, value in scorecard.items():
        _line(pdf, f"{key}: {value}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    _line(pdf, "Notizen")
    pdf.set_font("Helvetica", size=10)
    _line(pdf, f"t0-Begruendung: {notes.get('t0_reason', '-')}")
    _line(pdf, f"Hot-Zone: {notes.get('hot_zone_reason', '-')}")
    _line(pdf, f"Negativ/Kippen: {notes.get('negative_details', '-')}")
    _line(pdf, f"AGL-Hinweis: {notes.get('agl_note', '-')}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    _line(pdf, "Konkrete Tipps")
    pdf.set_font("Helvetica", size=10)
    for idx, tip in enumerate(tips, start=1):
        _line(pdf, f"{idx}. {tip}")

    raw = pdf.output(dest="S")
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    elif isinstance(raw, str):
        raw = raw.encode("latin-1", errors="replace")
    return raw


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)

