from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from typing import Any, Callable

AI_COACHING_SCHEMA_VERSION = 1

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
    "coaching_text": 900,
    "next_jump_focus": 420,
    "confidence_note": 280,
}
_AI_COACHING_CACHE: dict[str, dict[str, Any]] = {}
_AI_COACHING_DAILY_USAGE: dict[str, int] = {}

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
            instructions=_instructions_for_view(view_mode),
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
                    "schema": _response_schema(),
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
    except Exception:
        return _unavailable(
            enabled=True,
            reason="KI-Coaching konnte nicht erzeugt werden. Die normale Analyse bleibt aktiv.",
        )

    try:
        parsed = json.loads(_response_text(response))
    except Exception:
        return _unavailable(
            enabled=True,
            reason="KI-Antwort war kein gueltiges JSON. Fallback auf normale Analyse.",
        )

    validated = _validate_ai_response(parsed)
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


def _unavailable(*, enabled: bool, reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "enabled": bool(enabled),
        "reason": reason,
        "source": "fallback",
    }


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "schema_version",
        "view_mode",
        "jump",
        "performance_profile",
        "metrics",
        "scorecard",
        "review",
        "jump_brief",
        "tip_follow_up",
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


def _instructions_for_view(view_mode: str) -> str:
    style = (
        "Schreibe sehr einfach, direkt und ohne Fachjargon. Maximal drei kurze Saetze pro Feld."
        if view_mode == "simple"
        else "Schreibe technisch praezise, aber knapp. Messwerte duerfen genannt werden, wenn sie im Payload stehen."
    )
    return (
        "Du bist ein Speed-Skydiving-Coach. Formuliere Coaching-Texte ausschliesslich aus den gelieferten "
        "Analyse-Fakten. Erfinde keine Messwerte, keine Ursachen und keine Sicherheitsdiagnosen. "
        "Die deterministische Analyse ist die Quelle der Wahrheit; wenn Datenqualitaet eingeschraenkt ist, "
        "formuliere vorsichtig. Antworte nur als JSON gemaess Schema. "
        f"{style}"
    )


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(_REQUIRED_TEXT_FIELDS),
        "properties": {
            "summary": {
                "type": "string",
                "description": "Kurzes Gesamtfazit zum Sprung auf Basis der gelieferten Fakten.",
            },
            "main_issue": {
                "type": "string",
                "description": "Wichtigstes Problem oder Hebel fuer diesen Sprung.",
            },
            "coaching_text": {
                "type": "string",
                "description": "Konkrete Coaching-Erklaerung fuer den Springer.",
            },
            "next_jump_focus": {
                "type": "string",
                "description": "Ein klarer Fokus fuer den naechsten Sprung.",
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


def _validate_ai_response(value: Any) -> dict[str, str] | None:
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
        out[field] = text[: _MAX_FIELD_LENGTHS[field]]
    return out
