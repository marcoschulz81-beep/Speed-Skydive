from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import date
from difflib import SequenceMatcher
from typing import Any, Callable

AI_COACHING_SCHEMA_VERSION = 10
AI_PROFILE_COACHING_PROMPT_VERSION = 2

_REQUIRED_TEXT_FIELDS = [
    "summary",
    "main_issue",
    "coaching_text",
    "next_jump_focus",
    "confidence_note",
]
_MAX_FIELD_LENGTHS = {
    "summary": 280,
    "main_issue": 320,
    "coaching_text": 1800,
    "next_jump_focus": 560,
    "confidence_note": 280,
}
_DANGLING_TEXT_ENDINGS = {
    "aber",
    "am",
    "an",
    "auf",
    "bei",
    "bis",
    "damit",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "fuer",
    "gegen",
    "im",
    "in",
    "mit",
    "nach",
    "oder",
    "ohne",
    "sodass",
    "sowie",
    "ueber",
    "um",
    "und",
    "unter",
    "vom",
    "von",
    "vor",
    "wenn",
    "weil",
    "zu",
    "zum",
    "zur",
}
_AI_COACHING_CACHE: dict[str, dict[str, Any]] = {}
_AI_COACHING_DAILY_USAGE: dict[str, int] = {}
_LOGGER = logging.getLogger(__name__)

ClientFactory = Callable[[str, float], Any]


def generate_ai_coaching_texts(
    payload: dict[str, Any],
    *,
    view_mode: str,
    enabled: bool,
    model: str,
    timeout_s: float,
    max_requests_per_day: int | None = None,
    api_key: str | None = None,
    client_factory: ClientFactory | None = None,
) -> dict[str, Any]:
    if not enabled:
        return _unavailable(enabled=False, reason="KI-Coaching ist deaktiviert.")

    resolved_key = str(api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not resolved_key:
        return _unavailable(
            enabled=True,
            reason="OPENAI_API_KEY ist nicht gesetzt. Die normale Analyse bleibt aktiv.",
        )

    clean_payload = _compact_payload(payload)
    report_kind = str(clean_payload.get("report_kind") or "jump").strip().lower()
    cache_key = _cache_key(payload=clean_payload, view_mode=view_mode, model=model)
    cached = _AI_COACHING_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached, cached=True)

    if _daily_limit_reached(max_requests_per_day):
        return _unavailable(
            enabled=True,
            reason="KI-Coaching Tageslimit ist erreicht. Die normale Analyse bleibt aktiv.",
        )

    try:
        client = (
            client_factory(resolved_key, timeout_s)
            if client_factory is not None
            else _default_client(resolved_key, timeout_s)
        )
        _track_daily_request()
        response = client.responses.create(
            model=model,
            instructions=_instructions_for_view(view_mode, report_kind=report_kind),
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(clean_payload, ensure_ascii=False, sort_keys=True),
                        }
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "speed_skydive_ai_coaching",
                    "strict": True,
                    "schema": _response_schema(report_kind=report_kind),
                }
            },
            reasoning={"effort": "low"},
            max_output_tokens=1800,
        )
    except ImportError:
        return _unavailable(
            enabled=True,
            reason="OpenAI-Python-Paket fehlt. Bitte requirements.txt installieren.",
        )
    except Exception as exc:
        _LOGGER.warning("AI coaching request failed: %s", type(exc).__name__)
        return _unavailable(
            enabled=True,
            reason=f"KI-Coaching konnte nicht erzeugt werden ({_safe_exception_category(exc)}). Die normale Analyse bleibt aktiv.",
            error_type=type(exc).__name__,
        )

    try:
        parsed = json.loads(_response_text(response))
    except Exception:
        return _unavailable(
            enabled=True,
            reason="KI-Antwort war kein gueltiges JSON. Fallback auf normale Analyse.",
        )

    validated = _validate_ai_response(parsed, payload=clean_payload)
    if validated is None:
        return _unavailable(
            enabled=True,
            reason="KI-Antwort entsprach nicht dem erwarteten Schema. Fallback auf normale Analyse.",
        )

    result = {
        "available": True,
        "enabled": True,
        "source": "openai",
        "cached": False,
        "model": model,
        "view_mode": view_mode,
        **validated,
    }
    _AI_COACHING_CACHE[cache_key] = result
    return dict(result)


def clear_ai_coaching_cache() -> None:
    _AI_COACHING_CACHE.clear()
    _AI_COACHING_DAILY_USAGE.clear()


def _daily_limit_reached(max_requests_per_day: int | None) -> bool:
    if max_requests_per_day is None:
        return False
    limit = int(max_requests_per_day)
    if limit <= 0:
        return False
    return _AI_COACHING_DAILY_USAGE.get(_usage_day_key(), 0) >= limit


def _track_daily_request() -> None:
    key = _usage_day_key()
    _AI_COACHING_DAILY_USAGE[key] = int(_AI_COACHING_DAILY_USAGE.get(key, 0)) + 1


def _usage_day_key() -> str:
    return date.today().isoformat()


def _default_client(api_key: str, timeout_s: float) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key, timeout=float(timeout_s))


def _unavailable(*, enabled: bool, reason: str, error_type: str | None = None) -> dict[str, Any]:
    out = {
        "available": False,
        "enabled": bool(enabled),
        "reason": reason,
        "source": "fallback",
    }
    if error_type:
        out["error_type"] = str(error_type)
    return out


def _safe_exception_category(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timeout" in text or "timed out" in text:
        return "Timeout"
    if "rate" in name or "rate limit" in text or "429" in text:
        return "Rate Limit"
    if "auth" in name or "permission" in name or "401" in text or "403" in text:
        return "Auth/API-Key"
    if "connect" in name or "network" in name or "connection" in text or "dns" in text:
        return "Netzwerk"
    return "Clientfehler"


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "schema_version",
        "report_kind",
        "profile_prompt_version",
        "view_mode",
        "jump",
        "profile",
        "performance_profile",
        "feedback_training_profile",
        "metrics",
        "scorecard",
        "review",
        "primary_diagnosis",
        "technical_assessment",
        "jump_brief",
        "tip_follow_up",
        "jump_feedback",
        "quality",
    }
    clean = {key: value for key, value in payload.items() if key in allowed_keys}
    clean["schema_version"] = AI_COACHING_SCHEMA_VERSION
    return clean


def _cache_key(*, payload: dict[str, Any], view_mode: str, model: str) -> str:
    raw = json.dumps(
        {"payload": payload, "view_mode": view_mode, "model": model},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _instructions_for_view(view_mode: str, *, report_kind: str = "jump") -> str:
    is_simple = view_mode == "simple"
    is_profile = report_kind == "jumper_profile"
    style = (
        "Schreibe sehr einfach, direkt und ohne Fachjargon. Maximal drei kurze Saetze pro Feld. "
        "Nutze keine internen Begriffe wie Technikmodell, technische Bewertung, technische Analyse, "
        "vHor, vVert, Jerk, RMS, Speed-Drop oder 3s-Window. Schreibe stattdessen waagerechte Geschwindigkeit, "
        "horizontale Reserve, vertikale Geschwindigkeit, harte Korrekturen, Geschwindigkeitseinbruch und 3s-Fenster. "
        "Vermeide Vorwaertsbewegung oder Vorwaertsrichtung, weil GPS nicht die Koerperausrichtung misst."
        if is_simple
        else "Schreibe technisch praezise, aber knapp. Messwerte duerfen genannt werden, wenn sie im Payload stehen."
    )
    focus_rule = (
        "next_jump_focus ist genau eine konkrete Aufgabe fuer den naechsten Sprung. "
        "next_jump_focus soll kurz, messbar und handlungsorientiert sein. "
        if is_simple
        else (
            "next_jump_focus ist der KI-Coaching-Fokus: Formuliere in ein bis zwei Saetzen, wie die "
            "regelbasierten Tipps aus jump_brief.actions zusammenhaengen und welche Prioritaet daraus folgt. "
            "Du darfst Dopplungen sprachlich glaetten, aber keine Tipps ersetzen, keine Tipps frei weglassen "
            "und keine neuen technischen Zielwerte erfinden. Die pruefbare Tipp-Liste bleibt separat sichtbar. "
        )
    )
    profile_rule = ""
    if is_profile:
        profile_rule = (
            "Dieser Auftrag bewertet kein einzelnes Ereignis, sondern das Springerprofil ueber mehrere Spruenge. "
            "Beziehe summary, main_issue und coaching_text auf wiederkehrende Muster und den Verlauf in profile, review "
            "und jump_brief. Unterscheide klar zwischen stabilen Mustern, Verbesserungen und noch offenen Punkten. "
            "Behaupte keine Ursache allein aus einer Korrelation und schreibe nicht 'bei diesem Sprung'. "
            "summary muss aus hoechstens zwei vollstaendigen, kurzen Saetzen bestehen. "
            "next_jump_focus bleibt genau eine konkrete Aufgabe fuer den naechsten Sprung. "
        )
        if is_simple:
            profile_rule += (
                "Nutze in der einfachen Profilansicht alltaegliche Woerter und ganze, kurze Saetze. "
                "Vermeide Dezimal-Grenzwerte und ungewoehnliche Wortzusammensetzungen; runde notwendige Werte sinnvoll. "
            )
    return (
        "Du bist ein Speed-Skydiving-Coach. Formuliere Coaching-Texte ausschliesslich aus den gelieferten "
        "Analyse-Fakten. Erfinde keine Messwerte, keine Ursachen und keine Sicherheitsdiagnosen. "
        f"{profile_rule}"
        "Die deterministische Analyse ist die Quelle der Wahrheit; wenn Datenqualitaet eingeschraenkt ist, "
        "formuliere vorsichtig. Wenn primary_diagnosis.available=true ist, behandle diese Diagnose als Hauptursache "
        "und formuliere keine widerspruechlichen Ziele wie gleichzeitig steiler und flacher werden. "
        "Wenn technical_assessment vorhanden ist, nutze es als zusaetzliche technische Evidenz fuer Phasenmodell, "
        "3s-Fenster-Qualitaet und Beschleunigungsruhe. Nutze einen Speed-Drop nach dem 3s-Fenster nur dann als "
        "Problem, wenn best_window_quality.drop_after_evaluable=true ist; bei nicht belastbarem Folgefenster nicht "
        "aus einem Ausstiegs-/Decel-Drop auf instabile Technik schliessen. "
        "Wenn jump_feedback.available=true ist, nutze es als subjektiven Kontext des Springers. Behandle Feedback nie "
        "als Messwert und behaupte keine konkrete Koerperhaltung sicher ohne Messbeleg. Formuliere stattdessen, ob "
        "das Gefuehl oder der Versuch durch Messdaten bestaetigt, teilweise bestaetigt oder nicht klar sichtbar ist. "
        "Wenn das Feedback eine Technikidee nennt, pruefe die gelieferten Evidenzen und leite daraus einen kleinen, "
        "sicheren naechsten Schritt ab. Wenn feedback_training_profile vorhanden ist, nutze es nur als wiederkehrenden "
        "Trainingskontext, nicht als harte Bewertung. "
        "Trenne die Ausgabefelder strikt: summary ist nur das Kurzfazit, main_issue ist nur die Diagnose, "
        "coaching_text erklaert warum der Fehler entsteht und was im Sprung passiert. "
        f"{focus_rule}"
        "coaching_text darf die konkrete Fokus-Anweisung nicht "
        "wiederholen und soll nicht mit Imperativen wie 'im naechsten Sprung' beginnen. next_jump_focus soll "
        "messbar und handlungsorientiert sein. Ein reines Ergebnisziel wie 'Zuwachs um 12 km/h erhoehen' reicht "
        "nicht aus; beschreibe immer auch wie der Springer das technisch versuchen soll, z. B. Timing, Druck, "
        "Winkel, Linie, Korrekturen oder Guardrail. Wenn ein persoenlicher Winkel- oder Stabilitaetskorridor genannt wird, "
        "behandle ihn als Stabilitaets-Grenze bzw. Guardrail, nicht als allgemeines Speed-Ideal. Kombiniere nicht "
        "'Winkel halten' und 'Zuwachs erhoehen' zu einem harten Doppelziel; formuliere stattdessen: frueher Druck "
        "aufbauen, aber Winkel/Hot-Zone stabil halten. "
        "Nenne keine internen Payload-Feldnamen wie angle_peak, vhor_min_after_20 oder pattern; nutze klare deutsche "
        "Messwertbezeichnungen. Vergleiche nicht mit globalen Top-5, wenn diese nicht ausdruecklich im Payload stehen. "
        "Antworte nur als JSON gemaess Schema. "
        f"{style}"
    )


def _response_schema(*, report_kind: str = "jump") -> dict[str, Any]:
    is_profile = report_kind == "jumper_profile"
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(_REQUIRED_TEXT_FIELDS),
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "Kurzes Gesamtfazit zum Springerprofil auf Basis aller gelieferten Fakten."
                    if is_profile
                    else "Kurzes Gesamtfazit zum Sprung auf Basis der gelieferten Fakten."
                ),
            },
            "main_issue": {
                "type": "string",
                "description": (
                    "Wichtigstes wiederkehrendes Problem oder groesster Hebel im Springerprofil."
                    if is_profile
                    else "Wichtigstes Problem oder Hebel fuer diesen Sprung."
                ),
            },
            "coaching_text": {
                "type": "string",
                "description": (
                    "Erklaerender Coaching-Text zu Verlauf, Ursache, Wirkung und Flugmechanik. "
                    "Nicht die konkrete Aufgabe aus next_jump_focus wiederholen."
                ),
            },
            "next_jump_focus": {
                "type": "string",
                "description": (
                    "Simple: eine konkrete, messbare Aufgabe fuer den naechsten Sprung. "
                    "Expert: ein kurzer KI-Coaching-Fokus, der die regelbasierten Tipps aus jump_brief.actions "
                    "als Zusammenhang erklaert, ohne neue Ziele oder Messwerte zu erfinden."
                ),
            },
            "confidence_note": {
                "type": "string",
                "description": "Kurzer Hinweis zur Daten- oder Profil-Sicherheit.",
            },
        },
    }


def _response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if direct:
        return str(direct)
    if isinstance(response, dict):
        direct = response.get("output_text")
        if direct:
            return str(direct)
        output = response.get("output")
    else:
        output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                else:
                    text = getattr(part, "text", None)
                if text:
                    return str(text)
    return ""


def _validate_ai_response(value: Any, *, payload: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    out: dict[str, str] = {}
    for field in _REQUIRED_TEXT_FIELDS:
        raw = value.get(field)
        if not isinstance(raw, str):
            return None
        text = " ".join(raw.strip().split())
        if not text:
            return None
        out[field] = _limit_text(text, _MAX_FIELD_LENGTHS[field])
    return _separate_coaching_roles(out, payload=payload)


def _separate_coaching_roles(value: dict[str, str], *, payload: dict[str, Any]) -> dict[str, str]:
    out = dict(value)
    focus_replacement = _build_actionable_next_focus(out, payload=payload)
    if focus_replacement:
        out["next_jump_focus"] = _limit_text(
            focus_replacement,
            _MAX_FIELD_LENGTHS["next_jump_focus"],
        )
    summary_replacement = _build_safe_summary_replacement(out, payload=payload)
    if summary_replacement:
        out["summary"] = _limit_text(summary_replacement, _MAX_FIELD_LENGTHS["summary"])

    coaching_text = out.get("coaching_text", "")
    next_focus = out.get("next_jump_focus", "")
    if not _texts_are_too_similar(coaching_text, next_focus):
        return _sanitize_texts_for_view(out, payload=payload)

    replacement = _build_explanatory_coaching_text(out, payload=payload)
    if replacement:
        out["coaching_text"] = _limit_text(
            replacement,
            _MAX_FIELD_LENGTHS["coaching_text"],
        )
    return _sanitize_texts_for_view(out, payload=payload)


def _sanitize_texts_for_view(value: dict[str, str], *, payload: dict[str, Any]) -> dict[str, str]:
    if str(payload.get("view_mode") or "").strip().lower() != "simple":
        return value
    out = dict(value)
    for field in _REQUIRED_TEXT_FIELDS:
        out[field] = _limit_text(
            _sanitize_simple_ai_text(out.get(field, "")),
            _MAX_FIELD_LENGTHS[field],
        )
    return out


def _limit_text(text: str, max_len: int) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if len(cleaned) <= max_len:
        return cleaned

    boundary = max_len
    sentence_boundary = max(
        cleaned.rfind(".", 0, max_len + 1),
        cleaned.rfind("!", 0, max_len + 1),
        cleaned.rfind("?", 0, max_len + 1),
    )
    if sentence_boundary >= 80:
        boundary = sentence_boundary + 1
    else:
        word_boundary = cleaned.rfind(" ", 0, max_len + 1)
        if word_boundary >= max(40, int(max_len * 0.75)):
            boundary = word_boundary

    limited = _trim_dangling_text_tail(cleaned[:boundary])
    if not limited:
        limited = cleaned[:boundary].rstrip(" ,;:-")
    if not limited.endswith((".", "!", "?")):
        if len(limited) >= max_len:
            limited = _trim_dangling_text_tail(limited[: max_len - 1])
        limited = f"{limited}."
    return limited


def _trim_dangling_text_tail(text: str) -> str:
    cleaned = str(text or "").rstrip(" ,;:-")
    while cleaned:
        match = re.search(r"(.+?)\s+([^\s]+)$", cleaned)
        if not match:
            break
        tail = match.group(2).strip(".,;:!?()[]{}\"'").casefold()
        if tail not in _DANGLING_TEXT_ENDINGS:
            break
        cleaned = match.group(1).rstrip(" ,;:-")
    return cleaned


def _sanitize_simple_ai_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    replacements = [
        (r"\b[Dd]as\s+Technikmodell\s+best(?:ae|ä)tigt\b", "Die Messwerte zeigen"),
        (r"\b[Dd]as\s+Technikmodell\s+zeigt\b", "Die Messwerte zeigen"),
        (r"\b[Dd]as\s+technische\s+Modell\s+best(?:ae|ä)tigt\b", "Die Messwerte zeigen"),
        (r"\b[Dd]as\s+technische\s+Modell\s+zeigt\b", "Die Messwerte zeigen"),
        (r"\b[Dd]ie\s+technische\s+Analyse\s+zeigt\b", "Die Messwerte zeigen"),
        (r"\b[Tt]echnische\s+Analyse\s+zeigt\b", "Die Messwerte zeigen"),
        (r"\b[Tt]echnische\s+Analyse\b", "Messwerte"),
        (r"\b[Tt]echnische\s+Daten\s+zeigen\b", "Die Messwerte zeigen"),
        (r"\b[Tt]echnische\s+Daten\b", "Messwerte"),
        (r"\b[Tt]echnikbewertung\s+zeigt\b", "Die Messwerte zeigen"),
        (r"\b[Tt]echnikbewertung\b", "Messwerte"),
        (r"\b[Tt]echnikdaten\b", "Messwerte"),
        (r"\b[Tt]echnisch\s+bedeutet\s+das\s*:\s*", "Das bedeutet: "),
        (r"\b[Tt]echnisch\s+entsteht\s+das\b", "Das entsteht"),
        (r"\b[Tt]echnisch\s+ergibt\s+das\b", "Das ergibt"),
        (r"\b[Tt]echnisch\s+passt\s+das\s+zu\b", "Das passt zu"),
        (
            r"\b[Tt]echnisch\s+f(?:ue|ü)hren\s+fr(?:ue|ü)he,\s+kleine\s+Korrekturen\s+und\s+eine\s+ruhige\s+Linie\s+zu\b",
            "Fruehe, kleine Korrekturen und eine ruhige Linie fuehren zu",
        ),
        (r"\b[Tt]echnisch\s+zeigt\b", "Die Messwerte zeigen"),
        (r"\b[Tt]echnische\s+Bewertung\s+zeigt\b", "Die Messwerte zeigen"),
        (r"\b[Tt]echnische\s+Hinweise\s+zeigen\b", "Die Messwerte zeigen"),
        (r"\b[Tt]echnisch(?:e|er|es|en)?\b", ""),
        (r"\b[Tt]echnikmodell\b", "Messwerte"),
        (r"\b[Tt]echnisches\s+Modell\b", "Messwerte"),
        (r"\b[Tt]echnische[sn]?\s+Modell\b", "Messwerte"),
        (r"\b3s[\u2010-\u2015-]?Window\b", "3s-Fenster"),
        (
            r"\btrotz\s+ruhig(?:em|en|er)?\s+Jerk[\u2010-\u2015\s-]*(?:Wert|Werte)\b",
            "obwohl die harten Korrekturen gering sind",
        ),
        (
            r"\bruhig(?:em|en|er)?\s+Jerk[\u2010-\u2015\s-]*(?:Wert|Werte)\b",
            "wenig harte Korrekturen",
        ),
        (r"\bniedrige\s+Jerk[\u2010-\u2015\s-]*Werte\b", "wenig harte Korrekturen"),
        (r"\bgeringe\s+Jerk[\u2010-\u2015\s-]*Werte\b", "wenig harte Korrekturen"),
        (r"\bhohe\s+Jerk[\u2010-\u2015\s-]*(?:Werte|RMS)\b", "harte Korrekturen"),
        (r"\bJerk[\u2010-\u2015\s-]*(?:Wert|Werte)\b", "harte Korrekturen"),
        (r"\bRuckwerte\b", "Unruhe"),
        (
            r"\bniedrig(?:e|en|em|er)?\s+vHor[\u2010-\u2015\s-]*(?:Minimum|Minima|Min)\b",
            "niedrige Werte der waagerechten Geschwindigkeit",
        ),
        (
            r"\bMindest[\u2010-\u2015\s-]*vHor\b",
            "Mindestwert der waagerechten Geschwindigkeit",
        ),
        (r"\bmin\.\s*vHor\b", "Minimum der waagerechten Geschwindigkeit"),
        (
            r"\bvHor[\u2010-\u2015\s-]*(?:Minimum|Minima|Min)\b",
            "Minimum der waagerechten Geschwindigkeit",
        ),
        (
            r"\bvHor[\u2010-\u2015\s-]*(?:Einbruch|Drop)\b",
            "Einbruch der waagerechten Geschwindigkeit",
        ),
        (r"\bvHor[\u2010-\u2015\s-]*Abfall\b", "Abfall der waagerechten Geschwindigkeit"),
        (r"\bvHor[\u2010-\u2015\s-]*Reserve\b", "horizontale Reserve"),
        (
            r"\bvVert[\u2010-\u2015\s-]*(?:Einbruch|Drop)\b",
            "Abfall der vertikalen Geschwindigkeit",
        ),
        (r"\bvHor\b", "waagerechte Geschwindigkeit"),
        (r"\bvVert\b", "vertikale Geschwindigkeit"),
        (r"\bwaagereine\s+Geschwindigkeit\b", "waagerechte Geschwindigkeit"),
        (
            r"\bMindest[\u2010-\u2015\s-]*waagerechte\s+Geschwindigkeit\b",
            "Mindestwert der waagerechten Geschwindigkeit",
        ),
        (
            r"\bwaagerechte\s+Geschwindigkeit[\u2010-\u2015\s-]*(?:Einbruch|Drop)\b",
            "Einbruch der waagerechten Geschwindigkeit",
        ),
        (
            r"\bvertikale\s+Geschwindigkeit[\u2010-\u2015\s-]*(?:Einbruch|Drop)\b",
            "Abfall der vertikalen Geschwindigkeit",
        ),
        (r"\b[Vv]orw(?:ae|ä|Ã¤)rtsbewegung\b", "waagerechte Geschwindigkeit"),
        (r"\b[Vv]orw(?:ae|ä|Ã¤)rtsrichtung\b", "waagerechte Richtung"),
        (r"\b[Vv]orw(?:ae|ä|Ã¤)rtsreserve\b", "horizontale Reserve"),
        (r"\bwaagerechte\s+Reserve\b", "horizontale Reserve"),
        (
            r"\b[Ss]obald\s+die\s+waagerechte\s+Geschwindigkeit\s+hoch\s+(?:ist|wird)\b",
            "Sobald der Sprung schnell wird",
        ),
        (r"\bripples\b", "Unruhe"),
        (r"\bSpeed[\u2010-\u2015\s-]*Drop\b", "Geschwindigkeitseinbruch"),
    ]
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = re.sub(
        r"\b[Dd]ie\s+Messwerte\s+zeigen\s+das\s+Best[\u2010-\u2015-]?3s[\u2010-\u2015-]?Fenster\s+geringe\b",
        "Die Messwerte zeigen im besten 3s-Fenster geringe",
        cleaned,
    )
    cleaned = re.sub(r"\b[Dd]ie\s+Messwerte\s+zeigen\s+das\s+Messmodell\b", "Die Messwerte zeigen", cleaned)
    cleaned = re.sub(r"\b[Dd]ie\s+Messwerte\s+zeigen\s+das\s+Modell\s+ein\b", "Die Messwerte zeigen ein", cleaned)
    cleaned = re.sub(r"\b[Dd]ie\s+Messwerte\s+zeigen\s+sich\s+das\s+in\b", "Die Messwerte zeigen das in", cleaned)
    cleaned = re.sub(r"\b[Dd]ie\s+Messwerte\s+best(?:ae|ä|Ã¤)tigt\b", "Die Messwerte zeigen", cleaned)
    cleaned = re.sub(r"\b[Dd]as\s+Messmodell\s+zeigt\b", "Die Messwerte zeigen", cleaned)
    cleaned = re.sub(r"\b[Mm]essmodell\b", "Messwerte", cleaned)
    cleaned = re.sub(r"\bharte\s+Korrekturen\s*\(\s*harte\s+Korrekturen\s*\)", "harte Korrekturen", cleaned)
    cleaned = re.sub(r"\bharte\s+Korrekturen[\u2010-\u2015\s-]*RMS\b", "harte Korrekturen", cleaned)
    cleaned = re.sub(r"\b[Dd]en\s+harte\s+Korrekturen\b", "die harten Korrekturen", cleaned)
    cleaned = re.sub(r"(^|[.!?]\s+)passt\s+das\s+zu\b", lambda match: f"{match.group(1)}Das passt zu", cleaned)
    cleaned = re.sub(r"(^|[.!?]\s+)ergibt\s+das\b", lambda match: f"{match.group(1)}Das ergibt", cleaned)
    cleaned = re.sub(r"(^|[.!?]\s+)f(?:ue|ü)hrt\s+das\s+zu\b", lambda match: f"{match.group(1)}Das fuehrt zu", cleaned)
    cleaned = re.sub(
        r"(^|[.!?]\s+)f(?:ue|ü)hren\s+fr(?:ue|ü)he,\s+kleine\s+Korrekturen\s+und\s+eine\s+ruhige\s+Linie\s+zu\b",
        lambda match: f"{match.group(1)}Fruehe, kleine Korrekturen und eine ruhige Linie fuehren zu",
        cleaned,
    )
    cleaned = re.sub(
        r"\bmit\s+([^.;]{1,120}?)\s+und\s+niedrige\s+Werte\s+der\s+waagerechten\s+Geschwindigkeit\b",
        r"mit \1 und niedriger waagerechter Geschwindigkeit",
        cleaned,
    )
    cleaned = re.sub(
        r"\bmit\s+(sehr\s+)?niedrige\s+Werte\s+der\s+waagerechten\s+Geschwindigkeit\b",
        lambda match: f"mit {match.group(1) or ''}niedrigen Werten der waagerechten Geschwindigkeit",
        cleaned,
    )
    cleaned = re.sub(
        r"\bJerk[\u2010-\u2015\s-]*RMS\s*(?:\([^)]+\)|[0-9]+(?:[,.][0-9]+)?\s*m/s(?:3|³)?)",
        "harte Korrekturen",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bJerk\b", "harte Korrekturen", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("Die Die Messwerte", "Die Messwerte")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _build_safe_summary_replacement(value: dict[str, str], *, payload: dict[str, Any]) -> str:
    jump_brief = payload.get("jump_brief") if isinstance(payload.get("jump_brief"), dict) else {}
    deterministic_summary = _clean_focus_text(str(jump_brief.get("summary") or ""))
    if not deterministic_summary:
        return ""
    if _summary_conflicts_with_focus(value.get("summary", ""), value.get("next_jump_focus", "")):
        return deterministic_summary
    return ""


def _summary_conflicts_with_focus(summary: str, focus: str) -> bool:
    summary_norm = _normalize_for_similarity(summary)
    focus_norm = _normalize_for_similarity(focus)
    if not summary_norm or not focus_norm:
        return False
    focus_wants_earlier_pressure = "frueher" in focus_norm and "druck" in focus_norm
    if focus_wants_earlier_pressure and any(
        marker in summary_norm
        for marker in ["zu frueh", "zu fruehes", "zu fruehes druecken", "zu aggressiv"]
    ):
        return True
    return False


def _build_actionable_next_focus(value: dict[str, str], *, payload: dict[str, Any]) -> str:
    current = str(value.get("next_jump_focus") or "")
    jump_brief = payload.get("jump_brief") if isinstance(payload.get("jump_brief"), dict) else {}
    primary = payload.get("primary_diagnosis") if isinstance(payload.get("primary_diagnosis"), dict) else {}
    actions = [str(item or "") for item in (jump_brief.get("actions") or [])]
    view_mode = str(payload.get("view_mode") or "").strip().lower()

    if view_mode != "simple":
        current_clean = _clean_focus_text(current)
        if _expert_focus_is_usable(current_clean):
            return current_clean
        rule_summary = _build_expert_focus_from_actions(actions=actions, primary=primary)
        if rule_summary:
            return rule_summary

    candidates: list[str] = []
    if primary and primary.get("available"):
        candidates.append(str(primary.get("next_focus") or ""))
    if _focus_mentions_build_gain(current):
        candidates.extend([item for item in actions if _action_matches_build_gain(item)])
    candidates.extend(actions)

    seen: set[str] = set()
    for raw in candidates:
        text = _clean_focus_text(raw)
        if not text:
            continue
        key = _normalize_for_similarity(text)
        if not key or key in seen:
            continue
        seen.add(key)
        if _focus_has_actionable_how(text):
            return _focus_text_for_view(text, payload=payload)
    return ""


def _expert_focus_is_usable(text: str) -> bool:
    if not text:
        return False
    if _focus_has_untrusted_precision(text):
        return False
    if _focus_is_metric_only(text):
        return False
    return _focus_has_actionable_how(text)


def _focus_has_untrusted_precision(text: str) -> bool:
    normalized = str(text or "").replace(",", ".")
    return bool(
        re.search(r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*s\b", normalized, flags=re.IGNORECASE)
    )


def _focus_is_metric_only(text: str) -> bool:
    norm = _normalize_for_similarity(text)
    if not norm:
        return True
    result_markers = ["zuwachs", "km/h", "score", "um >=", "erhoehen", "erhöhen"]
    how_markers = [
        "druck",
        "winkel",
        "linie",
        "korrig",
        "stabil",
        "halten",
        "schieben",
        "aufbauen",
        "tragen",
        "fliegen",
    ]
    return any(marker in norm for marker in result_markers) and not any(marker in norm for marker in how_markers)


def _build_expert_focus_from_actions(*, actions: list[str], primary: dict[str, Any]) -> str:
    clean_actions = [_clean_focus_text(item) for item in actions]
    clean_actions = [item for item in clean_actions if item]
    if primary and primary.get("available"):
        primary_focus = _clean_focus_text(str(primary.get("next_focus") or primary.get("action") or ""))
        if primary_focus and not _focus_has_untrusted_precision(primary_focus):
            return primary_focus

    if len(clean_actions) < 2:
        return clean_actions[0] if clean_actions else ""

    combined = _normalize_for_similarity(" ".join(clean_actions))
    has_build = any(marker in combined for marker in ["+10", "+15", "aufbau", "druck", "zuwachs"])
    has_guardrail = any(marker in combined for marker in ["+20", "stabilitaetsbereich", "stabilitätsbereich", "winkel"])
    has_fast_correction = any(
        marker in combined
        for marker in ["letzte schnelle phase", "lenkimpuls", "beschleunigungskurve", "gegenkorrektur", "korrektur"]
    )
    has_reserve = any(marker in combined for marker in ["vhor", "horizontale reserve", "waagerechte geschwindigkeit"])

    if has_build and has_guardrail and has_fast_correction:
        return (
            "Der Fokus liegt auf dem Aufbau in die schnelle Phase: frueher Druck aufnehmen, aber bei +20s "
            "die persoenliche Winkel-Grenze und die Linie stabil halten. Wenn das ruhiger gelingt, werden "
            "spaete Gegenkorrekturen in der letzten schnellen Phase weniger noetig."
        )
    if has_build and has_guardrail:
        return (
            "Die Aufbau-Tipps gehoeren zusammen: frueher Druck aufnehmen, aber den Winkel bei +20s als "
            "Stabilitaets-Grenze behandeln. Erst wenn Linie und horizontale Reserve stabil bleiben, wieder "
            "steiler pushen."
        )
    if has_reserve and has_fast_correction:
        return (
            "Der Fokus liegt auf der letzten schnellen Phase: horizontale Reserve laenger halten und "
            "Korrekturen frueher kleiner setzen, damit die Linie nicht erst spaet gerettet werden muss."
        )
    if has_fast_correction:
        return (
            "Die Tipps zielen auf eine ruhigere schnelle Phase: weniger spaete Lenkimpulse, fruehere kleine "
            "Korrekturen und eine gleichmaessigere Beschleunigungskurve."
        )
    return " ".join(clean_actions[:2])


def _focus_text_for_view(text: str, *, payload: dict[str, Any]) -> str:
    if str(payload.get("view_mode") or "").strip().lower() != "simple":
        return text
    return _simplify_focus_for_simple(text)


def _simplify_focus_for_simple(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    def round_deg(match: re.Match[str]) -> str:
        low = int(float(match.group(1).replace(",", ".")) + 0.5)
        high = int(float(match.group(2).replace(",", ".")) + 0.5)
        return f"{low} bis {high} Grad"

    cleaned = re.sub(
        r"(\d+(?:[,.]\d+)?)\s+bis\s+(\d+(?:[,.]\d+)?)\s+Grad",
        round_deg,
        cleaned,
    )
    return cleaned


def _focus_mentions_build_gain(text: str) -> bool:
    norm = _normalize_for_similarity(text)
    return any(marker in norm for marker in ["zuwachs", "aufbau", "+10", "+15", "+20", "segment"])


def _action_matches_build_gain(text: str) -> bool:
    norm = _normalize_for_similarity(text)
    if not norm:
        return False
    return any(marker in norm for marker in ["druck", "zuwachs", "aufbau", "segment +10", "+10", "+15"])


def _focus_has_actionable_how(text: str) -> bool:
    norm = _normalize_for_similarity(text)
    if not norm:
        return False
    action_markers = [
        "druck",
        "winkel",
        "linie",
        "ruhig",
        "stabil",
        "stabilisieren",
        "halten",
        "bleiben",
        "korridor",
        "guardrail",
        "korrektur",
        "lenk",
        "vhor",
        "flacher",
        "steiler",
        "gleichmaessig",
        "frueher",
        "spaeter",
        "schrittweise",
        "nicht",
        "kleine",
        "aufbauen",
        "tragen",
        "schieben",
        "fliegen",
    ]
    return any(marker in norm for marker in action_markers)


def _clean_focus_text(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"^Priorit(?:ae|\u00e4)t\s*\d+\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ;,-")


def _build_explanatory_coaching_text(value: dict[str, str], *, payload: dict[str, Any]) -> str:
    primary = payload.get("primary_diagnosis") if isinstance(payload.get("primary_diagnosis"), dict) else {}
    jump_brief = payload.get("jump_brief") if isinstance(payload.get("jump_brief"), dict) else {}

    candidates: list[str] = []
    if primary and primary.get("available"):
        candidates.extend(
            [
                str(primary.get("summary") or ""),
                str(primary.get("main_issue") or ""),
            ]
        )
    candidates.extend(
        [
            str(value.get("main_issue") or ""),
            *[str(item or "") for item in (jump_brief.get("main_issues") or [])[:2]],
        ]
    )

    for raw in candidates:
        text = _clean_explanatory_sentence(raw)
        if text and not _texts_are_too_similar(text, value.get("next_jump_focus", "")):
            return (
                f"{text} Darum sollte der nächste Sprung auf eine einzelne Korrektur reduziert werden."
            )

    return (
        "Der wichtigste Hebel liegt in der Ursache der Hauptdiagnose, nicht in mehreren gleichzeitigen Korrekturen. "
        "Darum sollte der nächste Sprung auf eine einzelne Aufgabe reduziert werden."
    )


def _clean_explanatory_sentence(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"^([^:]{2,40}):\s*", "", cleaned)
    cleaned = cleaned.strip(" .;,-")
    if not cleaned:
        return ""
    return f"{cleaned}."


def _texts_are_too_similar(left: str, right: str) -> bool:
    left_norm = _normalize_for_similarity(left)
    right_norm = _normalize_for_similarity(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if len(left_norm) >= 24 and len(right_norm) >= 24:
        shorter, longer = sorted([left_norm, right_norm], key=len)
        if shorter in longer:
            return True
    if SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.82:
        return True

    left_tokens = _similarity_tokens(left_norm)
    right_tokens = _similarity_tokens(right_norm)
    if len(left_tokens) < 4 or len(right_tokens) < 4:
        return False
    overlap = left_tokens & right_tokens
    coverage = len(overlap) / float(min(len(left_tokens), len(right_tokens)))
    return coverage >= 0.72


def _normalize_for_similarity(text: str) -> str:
    lowered = str(text or "").casefold()
    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
    }
    for src, dst in replacements.items():
        lowered = lowered.replace(src, dst)
    lowered = re.sub(r"[^a-z0-9+\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _similarity_tokens(text: str) -> set[str]:
    stopwords = {
        "der",
        "die",
        "das",
        "den",
        "dem",
        "und",
        "oder",
        "mit",
        "bei",
        "bis",
        "zum",
        "zur",
        "ein",
        "eine",
        "einen",
        "im",
        "in",
        "auf",
        "fuer",
        "fur",
        "naechsten",
        "sprung",
    }
    return {token for token in text.split() if len(token) > 2 and token not in stopwords}
