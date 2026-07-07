from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.analysis.lateral import analyze_lateral_dynamics
from app.text_utils import normalize_german_text


_BODY_COMPACT_KEYS = {"arms_closer", "shoulders_compact", "legs_narrow", "head_quiet"}


def build_feedback_coaching_context(
    report: dict[str, Any],
    *,
    review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feedback = report.get("feedback") if isinstance(report.get("feedback"), dict) else {}
    text = _clean_feedback_text(feedback.get("text") if isinstance(feedback, dict) else "")
    if not text:
        return {"available": False, "text": "", "intents": [], "felt_issues": []}

    intents = _extract_intents(text)
    felt_issues = _extract_felt_issues(text)
    evidence = _feedback_evidence(report=report, review=review)
    match_lines = _build_match_lines(
        intents=intents,
        felt_issues=felt_issues,
        evidence=evidence,
    )
    coaching_hint = _build_coaching_hint(match_lines=match_lines, evidence=evidence)
    next_focus_hint = _build_next_focus_hint(intents=intents, felt_issues=felt_issues, evidence=evidence)
    profile_focus_keys = _profile_focus_keys(intents=intents, felt_issues=felt_issues)

    confidence = "low"
    if match_lines and evidence.get("objective_signals"):
        confidence = "medium"
    if len(match_lines) >= 2 and evidence.get("strong_signal_count", 0) >= 2:
        confidence = "high"

    return {
        "available": True,
        "text": text,
        "text_excerpt": _truncate(text, 500),
        "intents": intents,
        "felt_issues": felt_issues,
        "question_present": _question_present(text),
        "match_lines": match_lines[:4],
        "coaching_hint": coaching_hint,
        "next_focus_hint": next_focus_hint,
        "profile_focus_keys": profile_focus_keys,
        "confidence": confidence,
        "caution": "Feedback ist subjektiver Kontext; Messdaten bleiben die Bewertungsgrundlage.",
        "evidence": evidence,
    }


def build_feedback_training_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    feedback_records = [
        row
        for row in records
        if isinstance(row.get("feedback_context"), dict)
        and row["feedback_context"].get("available")
    ]
    if not feedback_records:
        return {"available": False, "reason": "Noch kein Sprungfeedback vorhanden."}

    recent = feedback_records[-min(8, len(feedback_records)) :]
    key_counter: Counter[str] = Counter()
    compact_with_unrest = 0
    compact_count = 0
    later_count = 0
    later_late_hard = 0
    end_unrest_count = 0
    for row in recent:
        context = row.get("feedback_context") if isinstance(row.get("feedback_context"), dict) else {}
        keys = [str(item) for item in (context.get("profile_focus_keys") or [])]
        key_counter.update(keys)
        if _BODY_COMPACT_KEYS.intersection(keys):
            compact_count += 1
            if _feedback_has_unrest_signal(context):
                compact_with_unrest += 1
        if "later_steepening" in keys or "smoother_curve" in keys:
            later_count += 1
            evidence = context.get("evidence") if isinstance(context.get("evidence"), dict) else {}
            if evidence.get("late_hard_transition"):
                later_late_hard += 1
        if "end_instability" in keys and _feedback_has_unrest_signal(context):
            end_unrest_count += 1

    lines: list[str] = []
    if key_counter:
        labels = [_focus_label(key) for key, _count in key_counter.most_common(2)]
        lines.append("Aktueller Trainingskontext aus Feedback: " + ", ".join(labels) + ".")
    if compact_count >= 2 and compact_with_unrest >= max(1, compact_count // 2):
        lines.append(
            "Kompaktere Koerperlinie taucht mehrfach zusammen mit Unruhe auf; "
            "diese Aenderung eher kleiner und spaeter testen."
        )
    if later_count >= 2 and later_late_hard >= 1:
        lines.append(
            "Der Fokus auf spaeteren/saubereren Aufbau ist sinnvoll, aber der Uebergang in die schnelle Phase "
            "muss gleichmaessiger werden."
        )
    if end_unrest_count >= 2:
        lines.append("Das Gefuehl von spaeter Unruhe wird in mehreren Spruengen durch Messdaten gestuetzt.")

    if not lines:
        lines.append("Feedback vorhanden; noch zu wenig Wiederholung fuer einen stabilen Trainingskontext.")

    return {
        "available": True,
        "feedback_count": len(feedback_records),
        "recent_feedback_count": len(recent),
        "summary": lines[0],
        "lines": lines[:4],
        "top_focus_keys": [key for key, _count in key_counter.most_common(4)],
    }


def _clean_feedback_text(value: Any) -> str:
    text = str(value or "").replace("\r", "\n").strip()
    text = re.sub(r"\s+", " ", text)
    return _truncate(text, 2000)


def _extract_intents(text: str) -> list[dict[str, str]]:
    norm = normalize_german_text(text)
    out: list[dict[str, str]] = []

    def add(key: str, label: str) -> None:
        if not any(item["key"] == key for item in out):
            out.append({"key": key, "label": label})

    if _has_any(norm, ["arm", "arme", "hand", "haende", "ellbogen"]) and _has_any(
        norm, ["enger", "naeher", "naher", "anlegen", "koerper", "beine", "kompakt"]
    ):
        add("arms_closer", "Arme/Haende enger")
    if "schulter" in norm and _has_any(norm, ["kompakt", "hoch", "enger", "schmal", "ruhig"]):
        add("shoulders_compact", "Schultern kompakter")
    if _has_any(norm, ["fuss", "fuesse", "beine", "knie", "zehen"]) and _has_any(
        norm, ["enger", "zusammen", "schmal", "lang", "gestreckt"]
    ):
        add("legs_narrow", "Beine/Fuesse schmaler")
    if _has_any(norm, ["kopf", "helm", "nacken"]) and _has_any(norm, ["ruhig", "linie", "stabil", "gerade"]):
        add("head_quiet", "Kopf/Helm ruhiger")
    if _has_any(
        norm,
        [
            "nicht sofort steil",
            "spaeter steil",
            "spaeter in den steil",
            "progressiv",
            "sauberere kurve",
            "saubere kurve",
            "ruhigere kurve",
            "nicht direkt steil",
        ],
    ):
        add("later_steepening", "spaeter/progressiver steil werden")
    if _has_any(norm, ["frueher druck", "frueher steil", "schneller steil", "mehr druck"]):
        add("earlier_pressure", "frueher Druck aufbauen")
    if _has_any(
        norm,
        ["laenger ziehen", "laenger zu ziehen", "laenger halten", "laenger durchziehen", "laenger in der linie"],
    ):
        add("longer_hold", "laenger halten")
    if _has_any(norm, ["stabilitaet", "stabiler", "ruhiger", "ruhig fliegen"]):
        add("stability_focus", "Stabilitaet")
    return out


def _extract_felt_issues(text: str) -> list[dict[str, str]]:
    norm = normalize_german_text(text)
    out: list[dict[str, str]] = []

    def add(key: str, label: str) -> None:
        if not any(item["key"] == key for item in out):
            out.append({"key": key, "label": label})

    if _has_any(norm, ["gewackelt", "zerrissen", "zerissen", "unruhig", "instabil", "weggekippt", "kippt"]):
        if _has_any(norm, ["ende", "schluss", "spaet", "spaeter", "hot", "peak"]) or not out:
            add("end_instability", "spaete Unruhe/Instabilitaet gefuehlt")
    if _has_any(norm, ["zu flach", "nicht steil genug", "flach"]):
        add("felt_too_flat", "zu flach gefuehlt")
    if _has_any(norm, ["zu steil", "zu schnell steil", "uebersteuert"]):
        add("felt_too_steep", "zu steil/uebersteuert gefuehlt")
    if _has_any(norm, ["exit", "absprung"]) and _has_any(norm, ["schlecht", "unruhig", "hart", "verpasst"]):
        add("exit_issue", "Exit-Problem gefuehlt")
    return out


def _feedback_evidence(report: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
    scorecard = report.get("scorecard", {}) if isinstance(report.get("scorecard"), dict) else {}
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
    fixpoints = report.get("fixpoints", []) if isinstance(report.get("fixpoints"), list) else []
    primary = review.get("primary_diagnosis", {}) if isinstance(review, dict) and isinstance(review.get("primary_diagnosis"), dict) else {}
    technical = review.get("technical_assessment", {}) if isinstance(review, dict) and isinstance(review.get("technical_assessment"), dict) else {}

    lateral = analyze_lateral_dynamics(
        report.get("chart_data", {}) if isinstance(report.get("chart_data"), dict) else {},
        eval_end_s=_eval_end_s(notes=notes, metrics=metrics),
    )
    lateral_event = lateral.get("speed_cost_event", {}) if isinstance(lateral, dict) else {}
    high_speed = lateral.get("high_speed", {}) if isinstance(lateral, dict) else {}
    best_window = technical.get("best_window_quality", {}) if isinstance(technical.get("best_window_quality"), dict) else {}
    jerk = technical.get("jerk_quality", {}) if isinstance(technical.get("jerk_quality"), dict) else {}

    angle10 = _fixpoint_value(fixpoints, 10.0, "angle_deg")
    angle15 = _fixpoint_value(fixpoints, 15.0, "angle_deg")
    angle20 = _fixpoint_value(fixpoints, 20.0, "angle_deg")
    phase_10_20 = str(scorecard.get("phase_10_20") or "")
    hot_zone = str(scorecard.get("hot_zone") or "")
    risk = str(scorecard.get("kipp_risiko") or "")
    pattern = str(primary.get("pattern") or "")

    strong_count = 0
    objective_signals: list[str] = []
    if hot_zone == "kritisch":
        strong_count += 1
        objective_signals.append("Hot-Zone kritisch")
    if risk in {"mittel", "hoch"}:
        strong_count += 1
        objective_signals.append(f"Kipp-Risiko {risk}")
    if bool(high_speed.get("unstable")):
        strong_count += 1
        objective_signals.append("Linie bei hohem Speed unruhig")
    if bool(lateral_event.get("available") and lateral_event.get("likely_speed_cost")):
        strong_count += 1
        objective_signals.append("seitliche Korrektur kostet Speed")
    if str(jerk.get("label") or "") in {"unruhig", "kritisch"}:
        strong_count += 1
        objective_signals.append(f"Beschleunigungsruhe {jerk.get('label')}")
    if str(best_window.get("label") or "") in {"unruhig", "kritisch"}:
        strong_count += 1
        objective_signals.append(f"3s-Fenster {best_window.get('label')}")

    end_unstable = bool(strong_count >= 1)
    if hot_zone == "kritisch" and risk == "hoch":
        end_unstable = True

    return {
        "objective_signals": objective_signals[:6],
        "strong_signal_count": strong_count,
        "end_unstable": end_unstable,
        "hot_zone_label": hot_zone,
        "risk_label": risk,
        "phase_10_20_label": phase_10_20,
        "primary_pattern": pattern,
        "too_fast_steep": pattern == "too_fast_steep_not_hold",
        "late_hard_transition": pattern == "late_hard_steepening_vhor_collapse",
        "rollback_instability": pattern == "rollback_with_instability",
        "build_too_flat": phase_10_20 == "zu flach",
        "build_too_steep": phase_10_20 == "zu steil",
        "angle10": angle10,
        "angle15": angle15,
        "angle20": angle20,
        "best_3s_vVert_kmh": _num(metrics.get("best_3s_vVert_kmh")),
        "lateral": {
            "available": bool(lateral.get("available")) if isinstance(lateral, dict) else False,
            "hot_pattern": str(lateral.get("hot_pattern") or "") if isinstance(lateral, dict) else "",
            "hot_vlat_abs_mean_kmh": _round(lateral.get("hot_vlat_abs_mean_kmh")) if isinstance(lateral, dict) else None,
            "hot_heading_rate_rms_dps": _round(lateral.get("hot_heading_rate_rms_dps")) if isinstance(lateral, dict) else None,
            "speed_cost_event": bool(lateral_event.get("available") and lateral_event.get("likely_speed_cost")),
            "high_speed_unstable": bool(high_speed.get("unstable")),
        },
        "best_window_quality": {
            "available": bool(best_window.get("available")),
            "label": str(best_window.get("label") or ""),
        },
        "jerk_quality": {
            "available": bool(jerk.get("available")),
            "label": str(jerk.get("label") or ""),
        },
    }


def _build_match_lines(
    *,
    intents: list[dict[str, str]],
    felt_issues: list[dict[str, str]],
    evidence: dict[str, Any],
) -> list[str]:
    intent_keys = {item["key"] for item in intents}
    issue_keys = {item["key"] for item in felt_issues}
    lines: list[str] = []

    if "end_instability" in issue_keys:
        if evidence.get("end_unstable"):
            signals = ", ".join(evidence.get("objective_signals") or [])
            if signals:
                lines.append(f"Dein Gefuehl von spaeter Unruhe passt zu den Messdaten ({signals}).")
            else:
                lines.append("Dein Gefuehl von spaeter Unruhe passt zu den Messdaten.")
        else:
            lines.append("Dein Gefuehl von Unruhe ist in den Messdaten nicht klar als starker Speedverlust sichtbar.")

    if _BODY_COMPACT_KEYS.intersection(intent_keys):
        if evidence.get("end_unstable"):
            lines.append(
                "Die kompaktere Koerperhaltung ist ohne Video nur eine Hypothese; "
                "die Wirkung in den Daten spricht aber fuer zu viel Unruhe nach der Aenderung."
            )
        else:
            lines.append(
                "Der Fokus auf eine kompaktere Linie wird beruecksichtigt; "
                "die Daten zeigen keinen klaren Instabilitaetsbeweis dadurch."
            )

    if "later_steepening" in intent_keys:
        if evidence.get("late_hard_transition"):
            lines.append(
                "Der spaetere Aufbau passt als Idee, aber der Uebergang in den Steilflug kommt noch zu hart."
            )
        elif evidence.get("too_fast_steep"):
            lines.append("Das Ziel, nicht sofort steil zu werden, wird in den Messdaten noch nicht sauber erreicht.")
        elif evidence.get("build_too_flat"):
            lines.append(
                "Der ruhigere Aufbau wirkt plausibel, aber der Speed-Aufbau bleibt dadurch noch zu schwach."
            )
        else:
            lines.append("Der Fokus auf eine sauberere Kurve passt zur aktuellen Coaching-Auswertung.")

    if "longer_hold" in intent_keys:
        if evidence.get("end_unstable"):
            lines.append("Laenger ziehen ist erst sinnvoll, wenn die schnelle Phase ruhiger gehalten wird.")
        else:
            lines.append("Der Fokus auf laengeres Halten kann im naechsten Sprung ueber die Hot-Zone geprueft werden.")

    return _unique(lines)


def _build_coaching_hint(*, match_lines: list[str], evidence: dict[str, Any]) -> str:
    if match_lines:
        return " ".join(match_lines[:2])
    if evidence.get("objective_signals"):
        return "Das Feedback wird als subjektiver Kontext genutzt; die Messdaten bleiben fuehrend."
    return "Feedback vorhanden, aber ohne klare messbare Zusatzspur."


def _build_next_focus_hint(
    *,
    intents: list[dict[str, str]],
    felt_issues: list[dict[str, str]],
    evidence: dict[str, Any],
) -> str:
    intent_keys = {item["key"] for item in intents}
    issue_keys = {item["key"] for item in felt_issues}
    if _BODY_COMPACT_KEYS.intersection(intent_keys) and evidence.get("end_unstable"):
        return (
            "Kompaktere Haltung nur kleiner testen: erst Linie und Richtung ruhig halten, "
            "dann Arme/Schultern/Beine schrittweise enger fuehren."
        )
    if "later_steepening" in intent_keys and evidence.get("late_hard_transition"):
        return (
            "Nicht spaeter hart steil werden: ab +15s gleichmaessiger aufbauen und vHor/Seitlinie ruhig halten."
        )
    if "end_instability" in issue_keys and evidence.get("end_unstable"):
        return "Naechster Fokus: die letzte schnelle Phase mit kleineren fruehen Korrekturen beruhigen."
    if "later_steepening" in intent_keys and evidence.get("build_too_flat"):
        return "Ruhigen Aufbau behalten, aber den Druck frueher gleichmaessig steigern statt nur flach zu bleiben."
    return ""


def _profile_focus_keys(*, intents: list[dict[str, str]], felt_issues: list[dict[str, str]]) -> list[str]:
    keys = [item["key"] for item in intents] + [item["key"] for item in felt_issues]
    return _unique(keys)


def _feedback_has_unrest_signal(context: dict[str, Any]) -> bool:
    evidence = context.get("evidence") if isinstance(context.get("evidence"), dict) else {}
    return bool(evidence.get("end_unstable") or evidence.get("strong_signal_count", 0) >= 1)


def _focus_label(key: str) -> str:
    return {
        "arms_closer": "Arme/Haende enger",
        "shoulders_compact": "Schultern kompakter",
        "legs_narrow": "Beine/Fuesse schmaler",
        "head_quiet": "Kopf ruhiger",
        "later_steepening": "spaeter/progressiver steil",
        "smoother_curve": "saubere Kurve",
        "earlier_pressure": "frueher Druck",
        "longer_hold": "laenger halten",
        "stability_focus": "Stabilitaet",
        "end_instability": "spaete Unruhe",
    }.get(key, key.replace("_", " "))


def _question_present(text: str) -> bool:
    norm = normalize_german_text(text)
    return "?" in text or _has_any(norm, ["was habe ich falsch", "warum", "wieso", "was war falsch"])


def _eval_end_s(*, notes: dict[str, Any], metrics: dict[str, Any]) -> float | None:
    candidates = [
        _num(notes.get("decel_start_s")),
        _num(notes.get("curve_window_end_s")),
        _num(metrics.get("performance_window_end_s")),
        _num(notes.get("canopy_open_s")),
    ]
    valid = [float(item) for item in candidates if item is not None and float(item) >= 8.0]
    return min(valid) if valid else None


def _fixpoint_value(fixpoints: list[Any], target: float, key: str) -> float | None:
    for item in fixpoints:
        if not isinstance(item, dict):
            continue
        t_rel = _num(item.get("t_rel_s"))
        if t_rel is not None and abs(float(t_rel) - float(target)) < 0.05:
            return _num(item.get(key))
    return None


def _has_any(value: str, tokens: list[str]) -> bool:
    return any(token in value for token in tokens)


def _truncate(text: str, max_len: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_len:
        return clean
    return clean[:max_len].rstrip()


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _round(value: Any, decimals: int = 1) -> float | None:
    number = _num(value)
    if number is None:
        return None
    return round(float(number), decimals)


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = normalize_german_text(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out
