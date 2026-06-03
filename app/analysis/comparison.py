from __future__ import annotations

from typing import Any


def build_jump_comparison(
    *,
    left_report: dict[str, Any],
    right_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare two jump reports.
    The faster jump (best_3s_vVert_kmh) is always used as reference.
    Delta is always: comparison - reference.
    """
    reference_report, comparison_report = _order_reports_by_speed(left_report=left_report, right_report=right_report)
    reference_jump = reference_report["jump"]
    comparison_jump = comparison_report["jump"]
    reference_metrics = reference_report["metrics"]
    comparison_metrics = comparison_report["metrics"]

    summary = [
        _summary_row(
            label="3s Max (Training)",
            key="best_3s_vVert_kmh",
            unit="km/h",
            better_when="higher",
            reference=reference_metrics,
            comparison=comparison_metrics,
        ),
        _summary_row(
            label="Rule Score",
            key="rule_based_3s_score",
            unit="km/h",
            better_when="higher",
            reference=reference_metrics,
            comparison=comparison_metrics,
        ),
        _summary_row(
            label="Negativ-Risiko",
            key="negative_risk_score",
            unit="score",
            better_when="lower",
            reference=reference_metrics,
            comparison=comparison_metrics,
        ),
    ]

    reference_fix = {float(item["t_rel_s"]): item for item in reference_report["fixpoints"]}
    comparison_fix = {float(item["t_rel_s"]): item for item in comparison_report["fixpoints"]}
    all_fix_times = sorted(set(reference_fix.keys()) | set(comparison_fix.keys()))

    fixpoint_rows: list[dict[str, Any]] = []
    for t_sec in all_fix_times:
        ref = reference_fix.get(t_sec, {})
        cmp = comparison_fix.get(t_sec, {})
        fixpoint_rows.append(
            {
                "t_rel_s": t_sec,
                "left_vVert_kmh": _num(ref.get("vVert_kmh")),
                "right_vVert_kmh": _num(cmp.get("vVert_kmh")),
                "delta_vVert_kmh": _delta(ref.get("vVert_kmh"), cmp.get("vVert_kmh")),
                "left_vHor_kmh": _num(ref.get("vHor_kmh")),
                "right_vHor_kmh": _num(cmp.get("vHor_kmh")),
                "delta_vHor_kmh": _delta(ref.get("vHor_kmh"), cmp.get("vHor_kmh")),
                "left_angle_deg": _num(ref.get("angle_deg")),
                "right_angle_deg": _num(cmp.get("angle_deg")),
                "delta_angle_deg": _delta(ref.get("angle_deg"), cmp.get("angle_deg")),
            }
        )

    insights = _build_insights(
        summary=summary,
        fixpoint_rows=fixpoint_rows,
        reference_name=reference_jump["file_name"],
        comparison_name=comparison_jump["file_name"],
    )
    reference_strengths, comparison_strengths = _build_both_sides_strengths(
        fixpoint_rows=fixpoint_rows,
        reference_name=reference_jump["file_name"],
        comparison_name=comparison_jump["file_name"],
    )
    brief = _build_compare_brief(
        reference_report=reference_report,
        comparison_report=comparison_report,
        summary=summary,
        fixpoint_rows=fixpoint_rows,
        reference_strengths=reference_strengths,
        comparison_strengths=comparison_strengths,
    )

    left_chart = _windowed_chart_series(reference_report)
    right_chart = _windowed_chart_series(comparison_report)
    x_end_candidates = [left_chart["window_end_s"], right_chart["window_end_s"]]
    x_end = max(x_end_candidates) if x_end_candidates else 0.0

    comparison_chart = {
        "left": {
            "label": _short_label(reference_jump["file_name"], reference_jump["t0_utc"]),
            "file_name": reference_jump["file_name"],
            "time_s": left_chart["time_s"],
            "vVert_kmh": left_chart["vVert_kmh"],
            "vHor_kmh": left_chart["vHor_kmh"],
            "angle_deg": left_chart["angle_deg"],
            "window_start_s": left_chart["window_start_s"],
            "window_end_s": left_chart["window_end_s"],
        },
        "right": {
            "label": _short_label(comparison_jump["file_name"], comparison_jump["t0_utc"]),
            "file_name": comparison_jump["file_name"],
            "time_s": right_chart["time_s"],
            "vVert_kmh": right_chart["vVert_kmh"],
            "vHor_kmh": right_chart["vHor_kmh"],
            "angle_deg": right_chart["angle_deg"],
            "window_start_s": right_chart["window_start_s"],
            "window_end_s": right_chart["window_end_s"],
        },
        "x_axis_start_s": 0.0,
        "x_axis_end_s": round(float(max(8.0, x_end)), 2),
    }

    return {
        "left": {
            "jump_id": reference_jump["jump_id"],
            "file_name": reference_jump["file_name"],
            "t0_utc": reference_jump["t0_utc"],
        },
        "right": {
            "jump_id": comparison_jump["jump_id"],
            "file_name": comparison_jump["file_name"],
            "t0_utc": comparison_jump["t0_utc"],
        },
        "reference": {
            "jump_id": reference_jump["jump_id"],
            "file_name": reference_jump["file_name"],
            "t0_utc": reference_jump["t0_utc"],
            "best_3s_vVert_kmh": _num(reference_metrics.get("best_3s_vVert_kmh")),
        },
        "comparison": {
            "jump_id": comparison_jump["jump_id"],
            "file_name": comparison_jump["file_name"],
            "t0_utc": comparison_jump["t0_utc"],
            "best_3s_vVert_kmh": _num(comparison_metrics.get("best_3s_vVert_kmh")),
        },
        "summary": summary,
        "fixpoint_rows": fixpoint_rows,
        "insights": insights,
        "reference_strengths": reference_strengths,
        "comparison_strengths": comparison_strengths,
        "brief": brief,
        "charts": comparison_chart,
    }


def _summary_row(
    *,
    label: str,
    key: str,
    unit: str,
    better_when: str,
    reference: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    left_v = _num(reference.get(key))
    right_v = _num(comparison.get(key))
    delta = _delta(left_v, right_v)
    trend = "gleich"
    if delta is not None:
        if abs(delta) < 1e-6:
            trend = "gleich"
        elif better_when == "higher":
            trend = "vergleich besser" if delta > 0 else "referenz besser"
        else:
            trend = "vergleich besser" if delta < 0 else "referenz besser"
    return {
        "label": label,
        "left": left_v,
        "right": right_v,
        "delta": delta,
        "unit": unit,
        "trend": trend,
    }


def _build_insights(
    *,
    summary: list[dict[str, Any]],
    fixpoint_rows: list[dict[str, Any]],
    reference_name: str,
    comparison_name: str,
) -> list[str]:
    insights: list[str] = []

    score_row = next((row for row in summary if row["label"] == "3s Max (Training)"), None)
    if score_row and score_row["delta"] is not None:
        if score_row["delta"] > 0:
            insights.append(
                f"{comparison_name} ist beim 3s-Max um {score_row['delta']:.2f} km/h schneller als die Referenz."
            )
        elif score_row["delta"] < 0:
            insights.append(
                f"{comparison_name} ist beim 3s-Max um {abs(score_row['delta']):.2f} km/h langsamer als die Referenz."
            )
        else:
            insights.append("3s-Max ist in beiden Spruengen gleich.")

    vvert_rows = [row for row in fixpoint_rows if row.get("delta_vVert_kmh") is not None]
    if vvert_rows:
        biggest_loss = min(vvert_rows, key=lambda row: float(row["delta_vVert_kmh"]))
        if float(biggest_loss["delta_vVert_kmh"]) < -5.0:
            insights.append(
                f"Groesster Unterschied bei vVert: um +{biggest_loss['t_rel_s']}s liegt {comparison_name} "
                f"{abs(float(biggest_loss['delta_vVert_kmh'])):.1f} km/h unter der Referenz."
            )
        biggest_gain = max(vvert_rows, key=lambda row: float(row["delta_vVert_kmh"]))
        if float(biggest_gain["delta_vVert_kmh"]) > 5.0:
            insights.append(
                f"Positiver Punkt im langsameren Sprung: um +{biggest_gain['t_rel_s']}s liegt {comparison_name} "
                f"{float(biggest_gain['delta_vVert_kmh']):.1f} km/h ueber der Referenz."
            )

    vhor_rows = [row for row in fixpoint_rows if row.get("delta_vHor_kmh") is not None]
    if vhor_rows:
        biggest_vhor_gain = max(vhor_rows, key=lambda row: float(row["delta_vHor_kmh"]))
        if float(biggest_vhor_gain["delta_vHor_kmh"]) > 4.0:
            insights.append(
                f"{comparison_name} ist bei +{biggest_vhor_gain['t_rel_s']}s in vHor um "
                f"{float(biggest_vhor_gain['delta_vHor_kmh']):.1f} km/h besser."
            )
        biggest_vhor_loss = min(vhor_rows, key=lambda row: float(row["delta_vHor_kmh"]))
        if float(biggest_vhor_loss["delta_vHor_kmh"]) < -6.0:
            insights.append(
                f"Schwaechster Punkt in vHor: bei +{biggest_vhor_loss['t_rel_s']}s liegt {comparison_name} um "
                f"{abs(float(biggest_vhor_loss['delta_vHor_kmh'])):.1f} km/h unter der Referenz."
            )

    risk_row = next((row for row in summary if row["label"] == "Negativ-Risiko"), None)
    if risk_row and risk_row["delta"] is not None:
        if risk_row["delta"] < -2.0:
            insights.append(f"{comparison_name} hat hier ein niedrigeres Stabilitaetsrisiko als die Referenz.")
        elif risk_row["delta"] > 2.0:
            insights.append(f"{comparison_name} zeigt ein hoeheres Stabilitaetsrisiko als die Referenz.")

    if not insights:
        insights.append("Keine belastbaren Unterschiede gefunden (fehlende oder identische Werte).")
    return insights


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: Any, right: Any) -> float | None:
    l = _num(left)
    r = _num(right)
    if l is None or r is None:
        return None
    return round(r - l, 3)


def _order_reports_by_speed(
    *,
    left_report: dict[str, Any],
    right_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    left_speed = _num(left_report.get("metrics", {}).get("best_3s_vVert_kmh"))
    right_speed = _num(right_report.get("metrics", {}).get("best_3s_vVert_kmh"))

    if left_speed is None and right_speed is None:
        return left_report, right_report
    if left_speed is None:
        return right_report, left_report
    if right_speed is None:
        return left_report, right_report
    if right_speed > left_speed:
        return right_report, left_report
    return left_report, right_report


def _build_both_sides_strengths(
    *,
    fixpoint_rows: list[dict[str, Any]],
    reference_name: str,
    comparison_name: str,
) -> tuple[list[str], list[str]]:
    reference_strengths: list[str] = []
    comparison_strengths: list[str] = []

    vhor_rows = [row for row in fixpoint_rows if row.get("delta_vHor_kmh") is not None]
    if vhor_rows:
        best_for_reference = min(vhor_rows, key=lambda row: float(row["delta_vHor_kmh"]))
        if float(best_for_reference["delta_vHor_kmh"]) < -3.0:
            reference_strengths.append(
                f"{reference_name}: bei +{best_for_reference['t_rel_s']}s deutlich mehr vHor "
                f"({abs(float(best_for_reference['delta_vHor_kmh'])):.1f} km/h)."
            )
        best_for_comparison = max(vhor_rows, key=lambda row: float(row["delta_vHor_kmh"]))
        if float(best_for_comparison["delta_vHor_kmh"]) > 3.0:
            comparison_strengths.append(
                f"{comparison_name}: bei +{best_for_comparison['t_rel_s']}s mehr vHor "
                f"({float(best_for_comparison['delta_vHor_kmh']):.1f} km/h)."
            )

    angle_rows = [row for row in fixpoint_rows if row.get("left_angle_deg") is not None and row.get("right_angle_deg") is not None]
    for row in angle_rows:
        ref_dist = abs(float(row["left_angle_deg"]) - 84.0)
        cmp_dist = abs(float(row["right_angle_deg"]) - 84.0)
        if ref_dist + 0.8 < cmp_dist:
            reference_strengths.append(
                f"{reference_name}: bei +{row['t_rel_s']}s naeher am Zielwinkel."
            )
            break
    for row in angle_rows:
        ref_dist = abs(float(row["left_angle_deg"]) - 84.0)
        cmp_dist = abs(float(row["right_angle_deg"]) - 84.0)
        if cmp_dist + 0.8 < ref_dist:
            comparison_strengths.append(
                f"{comparison_name}: bei +{row['t_rel_s']}s naeher am Zielwinkel."
            )
            break

    if not reference_strengths:
        reference_strengths.append(f"{reference_name}: insgesamt stabilerer Gesamtverlauf.")
    if not comparison_strengths:
        comparison_strengths.append(
            f"{comparison_name}: es gibt Teilbereiche, die besser sind und als Lernpunkt genutzt werden koennen."
        )
    return reference_strengths[:2], comparison_strengths[:2]


def _short_label(file_name: str, t0_utc: str) -> str:
    ts = t0_utc.replace("T", " ").replace("+00:00", "Z")
    if "." in ts:
        ts = ts.split(".", 1)[0] + "Z"
    return f"{file_name} ({ts})"


def _windowed_chart_series(report: dict[str, Any]) -> dict[str, Any]:
    notes = report.get("notes", {}) or {}
    chart = report.get("chart_data", {}) or {}

    time_s = chart.get("time_s", []) or []
    vvert = chart.get("vVert_kmh", []) or []
    vhor = chart.get("vHor_kmh", []) or []
    angle = chart.get("angle_deg", []) or []

    if not time_s:
        return {
            "time_s": [],
            "vVert_kmh": [],
            "vHor_kmh": [],
            "angle_deg": [],
            "window_start_s": 0.0,
            "window_end_s": 0.0,
        }

    raw_start = notes.get("curve_window_start_s", 0.0)
    raw_end = notes.get("curve_window_end_s", max(time_s))

    try:
        start_s = float(raw_start)
    except (TypeError, ValueError):
        start_s = 0.0
    try:
        end_s = float(raw_end)
    except (TypeError, ValueError):
        end_s = float(max(time_s))

    if end_s < start_s:
        start_s, end_s = 0.0, float(max(time_s))

    filtered_time: list[float] = []
    filtered_vvert: list[float] = []
    filtered_vhor: list[float] = []
    filtered_angle: list[float] = []

    for i, t in enumerate(time_s):
        try:
            t_val = float(t)
        except (TypeError, ValueError):
            continue
        if t_val < start_s or t_val > end_s:
            continue
        filtered_time.append(t_val)
        filtered_vvert.append(float(vvert[i]))
        filtered_vhor.append(float(vhor[i]))
        filtered_angle.append(float(angle[i]))

    if not filtered_time:
        filtered_time = [float(x) for x in time_s]
        filtered_vvert = [float(x) for x in vvert]
        filtered_vhor = [float(x) for x in vhor]
        filtered_angle = [float(x) for x in angle]
        start_s = float(min(filtered_time)) if filtered_time else 0.0
        end_s = float(max(filtered_time)) if filtered_time else 0.0

    return {
        "time_s": filtered_time,
        "vVert_kmh": filtered_vvert,
        "vHor_kmh": filtered_vhor,
        "angle_deg": filtered_angle,
        "window_start_s": round(start_s, 2),
        "window_end_s": round(end_s, 2),
    }


def _build_compare_brief(
    *,
    reference_report: dict[str, Any],
    comparison_report: dict[str, Any],
    summary: list[dict[str, Any]],
    fixpoint_rows: list[dict[str, Any]],
    reference_strengths: list[str],
    comparison_strengths: list[str],
) -> dict[str, Any]:
    ref_jump = reference_report["jump"]
    cmp_jump = comparison_report["jump"]
    ref_metrics = reference_report["metrics"]
    cmp_metrics = comparison_report["metrics"]

    ref_3s = _num(ref_metrics.get("best_3s_vVert_kmh"))
    cmp_3s = _num(cmp_metrics.get("best_3s_vVert_kmh"))
    delta_3s = None if ref_3s is None or cmp_3s is None else float(cmp_3s - ref_3s)

    basis_lines = [
        f"Referenz (schneller): {ref_jump['file_name']} ({ref_jump['t0_utc']}).",
        f"Vergleich: {cmp_jump['file_name']} ({cmp_jump['t0_utc']}).",
    ]

    key_facts: list[str] = []
    if delta_3s is not None:
        key_facts.append(
            f"3s-Max: Referenz {ref_3s:.1f} km/h, Vergleich {cmp_3s:.1f} km/h (Delta {delta_3s:+.1f})."
        )

    worst_vvert = _pick_fixpoint_row(fixpoint_rows, key="delta_vVert_kmh", direction="min")
    if worst_vvert and _num(worst_vvert.get("delta_vVert_kmh")) is not None and float(worst_vvert["delta_vVert_kmh"]) < -5.0:
        key_facts.append(
            f"Groesster Rueckstand vVert bei +{float(worst_vvert['t_rel_s']):.0f}s: "
            f"{abs(float(worst_vvert['delta_vVert_kmh'])):.1f} km/h."
        )

    worst_vhor = _pick_fixpoint_row(fixpoint_rows, key="delta_vHor_kmh", direction="min")
    if worst_vhor and _num(worst_vhor.get("delta_vHor_kmh")) is not None and float(worst_vhor["delta_vHor_kmh"]) < -4.0:
        key_facts.append(
            f"Groesster Rueckstand vHor bei +{float(worst_vhor['t_rel_s']):.0f}s: "
            f"{abs(float(worst_vhor['delta_vHor_kmh'])):.1f} km/h."
        )

    segment_pairs = _build_segment_pairs(
        ref_chart=reference_report.get("chart_data", {}),
        cmp_chart=comparison_report.get("chart_data", {}),
        ref_window_end=_window_end_for_compare(reference_report),
        cmp_window_end=_window_end_for_compare(comparison_report),
    )

    main_issues: list[str] = []
    actions: list[str] = []
    strengths: list[str] = []
    strengths.extend([f"Referenz: {item}" for item in reference_strengths[:2]])
    strengths.extend([f"Vergleich: {item}" for item in comparison_strengths[:2]])

    for item in segment_pairs:
        label = item["label"]
        ref_seg = item["reference"]
        cmp_seg = item["comparison"]
        if not ref_seg or not cmp_seg:
            continue

        gain_delta = _safe_delta(cmp_seg.get("vvert_gain"), ref_seg.get("vvert_gain"))
        angle_delta = _safe_delta(cmp_seg.get("angle_end"), ref_seg.get("angle_end"))
        turns_delta = _safe_delta(cmp_seg.get("turns"), ref_seg.get("turns"))
        min_vhor_delta = _safe_delta(cmp_seg.get("vhor_min"), ref_seg.get("vhor_min"))

        if gain_delta is not None and gain_delta < -10.0:
            main_issues.append(
                f"Im Segment {label} baut der Vergleich klar weniger vertikalen Speed auf ({gain_delta:.1f} km/h Delta)."
            )
            actions.append(
                f"Im Segment {label} frueher konstant Druck aufbauen, damit der vVert-Aufbau nicht abreisst."
            )

        if label in {"10-15s", "15-20s"} and angle_delta is not None and angle_delta < -2.0:
            main_issues.append(
                f"Im Segment {label} ist der Vergleich flacher als die Referenz ({angle_delta:.1f} Grad Delta)."
            )
            actions.append(
                f"Im Segment {label} den Tauchwinkel frueher stabil in den Zielbereich bringen (kleine, fruehe Korrekturen)."
            )

        if label in {"20-25s"} and angle_delta is not None and angle_delta > 2.0:
            main_issues.append(
                f"Im Segment {label} wird der Vergleich deutlich steiler als die Referenz ({angle_delta:+.1f} Grad)."
            )
            actions.append(
                "In der heißen Zone nicht zu steil werden; Linie ruhiger bei hohem Speed halten."
            )

        if label in {"20-25s"} and min_vhor_delta is not None and min_vhor_delta < -5.0:
            main_issues.append(
                f"Im Segment {label} faellt vHor im Vergleich deutlich tiefer ({min_vhor_delta:.1f} km/h Delta)."
            )
            actions.append(
                "Ab +20s Koerperspannung frueher stabilisieren, damit vHor nicht zu stark einbricht."
            )

        if label in {"20-25s"} and turns_delta is not None and turns_delta >= 2.0:
            main_issues.append(
                f"Im Segment {label} zeigt der Vergleich mehr Nachkorrekturen als die Referenz (+{turns_delta:.0f})."
            )
            actions.append(
                "In der Endphase kleinere Korrekturen frueher setzen statt spaete grobe Gegenbewegungen."
            )

        if gain_delta is not None and gain_delta > 8.0:
            strengths.append(
                f"Vergleich: Im Segment {label} ist der vertikale Aufbau besser als in der Referenz ({gain_delta:+.1f} km/h)."
            )
        if min_vhor_delta is not None and min_vhor_delta > 5.0:
            strengths.append(
                f"Vergleich: Im Segment {label} bleibt mehr horizontale Reserve erhalten ({min_vhor_delta:+.1f} km/h)."
            )

    main_issues = _unique_keep_order(main_issues)[:5]
    strengths = _unique_keep_order(strengths)[:5]
    actions = _unique_keep_order(actions)[:5]

    if not strengths:
        strengths = ["Beide Spruenge sind verwertbar; die Unterschiede liegen vor allem in der spaeten Stabilitaet."]
    if not main_issues:
        main_issues = ["Keine grosse technische Abweichung zwischen den beiden Spruengen erkennbar."]
    if not actions:
        actions = ["Ablauf des schnelleren Sprungs moeglichst exakt reproduzieren und nur kleine Korrekturen setzen."]

    if delta_3s is None:
        summary_text = "Vergleich liegt vor, aber 3s-Max konnte nicht sicher gegenuebergestellt werden."
    elif delta_3s <= -8.0:
        summary_text = (
            "Der Vergleichssprung ist klar langsamer. Hauptunterschied: "
            "in den spaeten Phasen geht Stabilitaet verloren und der Aufbau bricht frueher ab."
        )
    elif delta_3s < 0.0:
        summary_text = (
            "Der Vergleichssprung ist etwas langsamer. Mit stabilerer Linie in der Hot-Zone ist der Rueckstand schliessbar."
        )
    else:
        summary_text = (
            "Der Vergleichssprung ist gleich schnell oder schneller. Die Starken des Vergleichssprungs koennen als neue Referenz dienen."
        )

    return {
        "summary": summary_text,
        "basis_lines": basis_lines,
        "key_facts": key_facts[:4],
        "main_issues": main_issues,
        "strengths": strengths,
        "actions": actions,
    }


def _window_end_for_compare(report: dict[str, Any]) -> float:
    notes = report.get("notes", {}) or {}
    raw_end = _num(notes.get("curve_window_end_s"))
    if raw_end is None:
        chart = report.get("chart_data", {}) or {}
        time_s = chart.get("time_s", []) or []
        if not time_s:
            return 25.0
        try:
            raw_end = float(max(time_s))
        except Exception:
            raw_end = 25.0
    return min(25.0, max(10.0, float(raw_end)))


def _build_segment_pairs(
    *,
    ref_chart: dict[str, Any],
    cmp_chart: dict[str, Any],
    ref_window_end: float,
    cmp_window_end: float,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for start_s, end_s, label in [
        (0.0, 10.0, "0-10s"),
        (10.0, 15.0, "10-15s"),
        (15.0, 20.0, "15-20s"),
        (20.0, 25.0, "20-25s"),
    ]:
        ref_seg = _segment_stats(ref_chart, start_s, min(end_s, ref_window_end))
        cmp_seg = _segment_stats(cmp_chart, start_s, min(end_s, cmp_window_end))
        pairs.append({"label": label, "reference": ref_seg, "comparison": cmp_seg})
    return pairs


def _segment_stats(chart: dict[str, Any], start_s: float, end_s: float) -> dict[str, float] | None:
    if end_s <= start_s + 0.4:
        return None
    time_s = chart.get("time_s", []) or []
    vvert = chart.get("vVert_kmh", []) or []
    vhor = chart.get("vHor_kmh", []) or []
    angle = chart.get("angle_deg", []) or []
    if not time_s or len(time_s) != len(vvert) or len(time_s) != len(vhor) or len(time_s) != len(angle):
        return None

    t: list[float] = []
    vv: list[float] = []
    vh: list[float] = []
    ang: list[float] = []
    for i, raw_t in enumerate(time_s):
        t_val = _num(raw_t)
        vv_val = _num(vvert[i])
        vh_val = _num(vhor[i])
        a_val = _num(angle[i])
        if None in {t_val, vv_val, vh_val, a_val}:
            continue
        t.append(float(t_val))
        vv.append(float(vv_val))
        vh.append(float(vh_val))
        ang.append(float(a_val))
    if len(t) < 4:
        return None
    if t[0] > end_s or t[-1] < start_s:
        return None

    vv_start = _interp_linear(t, vv, start_s)
    vv_end = _interp_linear(t, vv, end_s)
    vh_start = _interp_linear(t, vh, start_s)
    vh_end = _interp_linear(t, vh, end_s)
    angle_end = _interp_linear(t, ang, end_s)
    if None in {vv_start, vv_end, vh_start, vh_end, angle_end}:
        return None

    in_idx = [i for i, ts in enumerate(t) if start_s <= ts <= end_s]
    if len(in_idx) < 2:
        return None
    min_vhor = min(vh[i] for i in in_idx)
    turns = _turn_count([ang[i] for i in in_idx], eps=0.35)

    return {
        "vvert_gain": float(vv_end - vv_start),
        "vhor_drop": float(vh_start - vh_end),
        "vhor_min": float(min_vhor),
        "angle_end": float(angle_end),
        "turns": float(turns),
    }


def _interp_linear(x: list[float], y: list[float], target: float) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    if target < x[0] or target > x[-1]:
        return None
    for i in range(1, len(x)):
        if x[i] < target:
            continue
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[i - 1], y[i]
        if x1 == x0:
            return float(y1)
        factor = (target - x0) / (x1 - x0)
        return float(y0 + factor * (y1 - y0))
    return float(y[-1])


def _turn_count(values: list[float], *, eps: float) -> int:
    if len(values) < 3:
        return 0
    turns = 0
    prev = 0
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        sign = 1 if d > eps else -1 if d < -eps else 0
        if sign == 0:
            continue
        if prev != 0 and sign != prev:
            turns += 1
        prev = sign
    return turns


def _pick_fixpoint_row(
    rows: list[dict[str, Any]],
    *,
    key: str,
    direction: str,
) -> dict[str, Any] | None:
    usable = [row for row in rows if _num(row.get(key)) is not None]
    if not usable:
        return None
    if direction == "max":
        return max(usable, key=lambda row: float(_num(row.get(key)) or 0.0))
    return min(usable, key=lambda row: float(_num(row.get(key)) or 0.0))


def _safe_delta(a: Any, b: Any) -> float | None:
    av = _num(a)
    bv = _num(b)
    if av is None or bv is None:
        return None
    return float(av - bv)


def _unique_keep_order(items: list[str]) -> list[str]:
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
