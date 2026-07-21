from __future__ import annotations

import json
from types import SimpleNamespace

from app.services.ai_coach import _limit_text, clear_ai_coaching_cache, generate_ai_coaching_texts


class _FakeResponses:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class _FakeClient:
    def __init__(self, output_text: str):
        self.responses = _FakeResponses(output_text)


def test_ai_coaching_returns_disabled_without_api_call():
    result = generate_ai_coaching_texts(
        {"schema_version": 1},
        view_mode="expert",
        enabled=False,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="",
        client_factory=lambda _key, _timeout: _FakeClient("{}"),
    )

    assert result["available"] is False
    assert result["enabled"] is False


def test_ai_coaching_requires_api_key_when_enabled(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = generate_ai_coaching_texts(
        {"schema_version": 1},
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="",
        client_factory=lambda _key, _timeout: _FakeClient("{}"),
    )

    assert result["available"] is False
    assert result["enabled"] is True
    assert "OPENAI_API_KEY" in result["reason"]


def test_ai_coaching_accepts_valid_json_schema_response():
    clear_ai_coaching_cache()
    response_payload = {
        "summary": "Der Sprung ist verwertbar.",
        "main_issue": "Der groesste Hebel liegt im Aufbau.",
        "coaching_text": "Bleib bis +15s ruhiger im Winkel und steigere erst danach.",
        "next_jump_focus": "Ein klarer Fokus: Aufbau ruhiger halten.",
        "confidence_note": "Datenbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    result = generate_ai_coaching_texts(
        {"schema_version": 1, "metrics": {"best_3s_vVert_kmh": 420.0}},
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert result["available"] is True
    assert result["summary"] == response_payload["summary"]
    assert result["source"] == "openai"
    request = fake_client.responses.calls[0]
    assert request["model"] == "gpt-5-mini"
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["reasoning"] == {"effort": "low"}
    assert request["max_output_tokens"] == 1800
    assert "coaching_text erklaert" in request["instructions"]
    assert "next_jump_focus ist der KI-Coaching-Fokus" in request["instructions"]
    assert "regelbasierten Tipps aus jump_brief.actions" in request["instructions"]
    assert "Ein reines Ergebnisziel" in request["instructions"]


def test_ai_coaching_uses_profile_context_for_jumper_profile():
    clear_ai_coaching_cache()
    response_payload = {
        "summary": "Das Profil entwickelt sich stabil nach oben.",
        "main_issue": "Der wichtigste Hebel ist die reproduzierbare schnelle Phase.",
        "coaching_text": "Mehrere Spruenge zeigen Fortschritt, die Wiederholbarkeit schwankt aber noch.",
        "next_jump_focus": "Im naechsten Sprung die zuletzt gute Linie ohne zusaetzliche Korrektur wiederholen.",
        "confidence_note": "Die Profilbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    result = generate_ai_coaching_texts(
        {
            "report_kind": "jumper_profile",
            "profile_prompt_version": 1,
            "profile": {"jump_count": 12, "trend_summary": "Positiver Verlauf."},
            "jump_brief": {"actions": ["Die zuletzt gute Linie stabil wiederholen."]},
        },
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert result["available"] is True
    request = fake_client.responses.calls[0]
    sent_payload = json.loads(request["input"][0]["content"][0]["text"])
    assert sent_payload["report_kind"] == "jumper_profile"
    assert sent_payload["profile"]["jump_count"] == 12
    assert "Springerprofil ueber mehrere Spruenge" in request["instructions"]
    assert "bei diesem Sprung" in request["instructions"]
    schema = request["text"]["format"]["schema"]
    assert "Springerprofil" in schema["properties"]["summary"]["description"]


def test_ai_coaching_preserves_long_expert_coaching_text_past_old_limit():
    clear_ai_coaching_cache()
    long_explanation = " ".join(
        [
            (
                "Die Messdaten zeigen einen spaeten harten Winkelaufbau mit knapper "
                "Vorwaertsreserve und einem unruhigen 3s-Fenster."
            )
            for _ in range(10)
        ]
    )
    assert len(long_explanation) > 900
    response_payload = {
        "summary": "Der Sprung ist verwertbar.",
        "main_issue": "Der groesste Hebel liegt in der spaeten schnellen Phase.",
        "coaching_text": long_explanation,
        "next_jump_focus": "Ab +15s schrittweise aufbauen und die Linie ruhig halten.",
        "confidence_note": "Datenbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    result = generate_ai_coaching_texts(
        {"schema_version": 1},
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert result["available"] is True
    assert result["coaching_text"] == long_explanation


def test_ai_coaching_long_text_limit_ends_on_clean_boundary():
    clear_ai_coaching_cache()
    very_long_explanation = " ".join(
        [
            f"Satz {idx} beschreibt die Ursache und Wirkung im Sprungverlauf."
            for idx in range(80)
        ]
    )
    assert len(very_long_explanation) > 1800
    response_payload = {
        "summary": "Der Sprung ist verwertbar.",
        "main_issue": "Der groesste Hebel liegt in der spaeten schnellen Phase.",
        "coaching_text": very_long_explanation,
        "next_jump_focus": "Ab +15s schrittweise aufbauen und die Linie ruhig halten.",
        "confidence_note": "Datenbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    result = generate_ai_coaching_texts(
        {"schema_version": 1},
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert result["available"] is True
    assert len(result["coaching_text"]) <= 1800
    assert result["coaching_text"].endswith(".")
    assert not result["coaching_text"].endswith(" .")


def test_ai_text_limit_does_not_finish_on_dangling_connector():
    text = (
        "Prioritaet: Aufbau ruhig halten und die Linie vor dem "
        "3s-Fenster stabilisieren."
    )
    limited = _limit_text(text, len("Prioritaet: Aufbau ruhig halten und die Linie vor dem"))

    assert limited == "Prioritaet: Aufbau ruhig halten und die Linie."
    assert "vor." not in limited
    assert "dem." not in limited


def test_ai_text_limit_prefers_an_earlier_complete_sentence():
    first_sentence = (
        "Das Profil zeigt ueber mehrere Spruenge einen stabilen und klar positiven Verlauf."
    )
    second_sentence = (
        "Die spaete Phase bleibt jedoch ein wiederkehrender Schwerpunkt mit mehreren weiteren Einzelheiten."
    )

    limited = _limit_text(f"{first_sentence} {second_sentence}", len(first_sentence) + 35)

    assert limited == first_sentence


def test_ai_coaching_expert_focus_keeps_reported_focus_complete():
    clear_ai_coaching_cache()
    focus = (
        "Prioritaet: Aufbau im Segment +10 bis +15s frueher beginnen, dabei Winkel nicht "
        "erzwingen und horizontale Reserve schuetzen. Konkret: leichter, frueher Druckaufbau "
        "zwischen +10s und +15s mit Fokus auf stabiler vHor (Guardrail: nicht unter deine "
        "bisherigen vHor-Min-Werte gehen) und deutlich kleinere, fruehere Korrekturen vor +20s, "
        "sodass das Zielzuwachsfenster (+10 bis +20s) gesteigert wird und die Linie vor dem "
        "3s-Fenster ruhiger bleibt."
    )
    assert len(focus) > 420
    response_payload = {
        "summary": "Der Sprung ist verwertbar.",
        "main_issue": "Der groesste Hebel liegt im Aufbau.",
        "coaching_text": "Der Aufbau und die horizontale Reserve haengen zusammen.",
        "next_jump_focus": focus,
        "confidence_note": "Datenbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    result = generate_ai_coaching_texts(
        {"schema_version": 1},
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert result["available"] is True
    assert result["next_jump_focus"] == focus
    assert result["next_jump_focus"].endswith("3s-Fenster ruhiger bleibt.")
    assert "vor dem." not in result["next_jump_focus"]


def test_ai_coaching_separates_duplicate_coaching_and_focus_texts():
    clear_ai_coaching_cache()
    duplicate_focus = "Bis +10s zuerst stabil im Bereich 72 bis 76 Grad bleiben."
    response_payload = {
        "summary": "Der Sprung wird zu frueh unruhig.",
        "main_issue": "Der Tauchwinkel wird zu frueh zu steil.",
        "coaching_text": duplicate_focus,
        "next_jump_focus": duplicate_focus,
        "confidence_note": "Datenbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    result = generate_ai_coaching_texts(
        {
            "schema_version": 1,
            "primary_diagnosis": {
                "available": True,
                "summary": "Der Winkel wird zu frueh steil und die Linie wird danach unruhig.",
                "main_issue": "Der fruehe harte Winkelwechsel kostet Stabilitaet.",
            },
            "jump_brief": {
                "main_issues": ["Der Aufbau ist zu hektisch."],
            },
        },
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert result["available"] is True
    assert result["next_jump_focus"] == duplicate_focus
    assert result["coaching_text"] != duplicate_focus
    assert "Winkel wird zu frueh steil" in result["coaching_text"]
    assert "eine einzelne Korrektur" in result["coaching_text"]


def test_ai_coaching_replaces_metric_only_focus_with_actionable_tip():
    clear_ai_coaching_cache()
    metric_only_focus = "Zuwachs im Segment +10-20s um >=12.0 km/h erhoehen."
    actionable_focus = (
        "Im Segment +10 bis +15s frueher Druck aufbauen, aber den Winkel nicht erzwingen "
        "und die Linie ruhiger halten."
    )
    response_payload = {
        "summary": "Der Aufbau liefert zu wenig Zuwachs.",
        "main_issue": "Im Aufbau fehlt frueher Druck.",
        "coaching_text": "Der zu schwache fruehe Aufbau kostet spaeter Stabilitaet.",
        "next_jump_focus": metric_only_focus,
        "confidence_note": "Datenbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    result = generate_ai_coaching_texts(
        {
            "schema_version": 1,
            "jump_brief": {
                "actions": [
                    "Bei +20s nicht ueber den persoenlichen Stabilitaetsbereich schieben.",
                    actionable_focus,
                ],
            },
        },
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert result["available"] is True
    assert "Aufbau-Tipps gehoeren zusammen" in result["next_jump_focus"]
    assert "frueher Druck" in result["next_jump_focus"]
    assert "Stabilitaets-Grenze" in result["next_jump_focus"]
    assert result["next_jump_focus"] != metric_only_focus


def test_ai_coaching_simple_focus_uses_deterministic_action_and_rounds_angle_range():
    clear_ai_coaching_cache()
    generic_focus = "Aufbauphase: Zwischen +10s und +20s gleichmaessig weiter beschleunigen."
    deterministic_focus = (
        "Bis +10s zuerst stabil im Bereich 72.3 bis 76.5 Grad bleiben. "
        "Erst wenn die Linie bis +15s ruhig bleibt, den Winkel in kleinen Schritten weiter aufbauen."
    )
    response_payload = {
        "summary": "Der Aufbau wird zu frueh unruhig.",
        "main_issue": "Der Winkel ist bei +10s zu steil.",
        "coaching_text": "Der zu steile fruehe Winkel fuehrt spaeter zu Nachkorrekturen.",
        "next_jump_focus": generic_focus,
        "confidence_note": "Datenbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    result = generate_ai_coaching_texts(
        {
            "schema_version": 1,
            "view_mode": "simple",
            "jump_brief": {
                "actions": [deterministic_focus],
            },
        },
        view_mode="simple",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert result["available"] is True
    assert "72 bis 77 Grad" in result["next_jump_focus"]
    assert generic_focus != result["next_jump_focus"]


def test_ai_coaching_simple_view_removes_internal_technical_jargon():
    clear_ai_coaching_cache()
    response_payload = {
        "summary": "Das technische Modell zeigt ein ruhiges bestes 3s-Window.",
        "main_issue": "Technische Bewertung zeigt harte Korrekturen und Vorwaertsbewegung geht verloren.",
        "coaching_text": (
            "Die technische Analyse zeigt vHor-Abfall und vVert-Verlust. "
            "Das 3s-Fenster ist unruhig trotz ruhigem Jerk-Wert. "
            "Technikdaten zeigen ein niedriges vHor-Minimum, wenig vHor-Reserve "
            "und waagereine Geschwindigkeit. Technische Daten zeigen Mindest-vHor, "
            "vVert-Drop und ripples in Richtung. Technisch bedeutet das: spaet korrigieren. "
            "Technisch passt das zu harte Korrekturen (harte Korrekturen). "
            "Technisch ergibt das ein kurzes 3s-Peak. "
            "Technisch fuehren fruehe, kleine Korrekturen und eine ruhige Linie zu mehr Speed. "
            "Die Messwerte zeigen das Messmodell ein unruhiges Fenster. "
            "Die Messwerte zeigen das Best-3s-Fenster geringe Winkelschwankung. "
            "Die Messwerte zeigen das Modell ein ruhiges 3s-Fenster. "
            "Die Messwerte bestaetigt eine auffaellige Phase. "
            "Die Messwerte zeigen sich das in steilem Winkel. "
            "Hot-Zone mit Max-Winkel und niedrige Werte der waagerechten Geschwindigkeit. "
            "Danach kommt ein Speed-Drop. Den harte Korrekturen-RMS reduzieren. "
            "Fenster mit sehr niedrige Werte der waagerechten Geschwindigkeit. "
            "Sobald die Vorwaertsbewegung hoch wird, wird die Linie unruhig."
        ),
        "next_jump_focus": "Technische Hinweise zeigen: im Peak kleiner korrigieren.",
        "confidence_note": "Technikmodell ausreichend belastbar.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    result = generate_ai_coaching_texts(
        {
            "schema_version": 1,
            "view_mode": "simple",
            "jump_brief": {
                "actions": ["Im Peak kleine Korrekturen setzen und die Linie ruhig halten."],
            },
        },
        view_mode="simple",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    text_blob = " ".join(
        [
            result["summary"],
            result["main_issue"],
            result["coaching_text"],
            result["next_jump_focus"],
            result["confidence_note"],
        ]
    )
    assert "Technikmodell" not in text_blob
    assert "technische Modell" not in text_blob
    assert "Technische Bewertung" not in text_blob
    assert "Technische Hinweise" not in text_blob
    assert "technische Analyse" not in text_blob
    assert "Jerk-RMS" not in text_blob
    assert "Jerk-Werte" not in text_blob
    assert "Jerk" not in text_blob
    assert "ruhigem harte Korrekturen-Wert" not in text_blob
    assert "Ruckwerte" not in text_blob
    assert "3s-Window" not in text_blob
    assert "vHor" not in text_blob
    assert "vVert" not in text_blob
    assert "Vorwaertsbewegung" not in text_blob
    assert "Vorwärtsbewegung" not in text_blob
    assert "Vorwaertsreserve" not in text_blob
    assert "Vorwärtsreserve" not in text_blob
    assert "Technikdaten" not in text_blob
    assert "waagereine Geschwindigkeit" not in text_blob
    assert "waagerechte Geschwindigkeit-Minimum" not in text_blob
    assert "waagerechte Reserve" not in text_blob
    assert "Technische Daten" not in text_blob
    assert "Mindest-vHor" not in text_blob
    assert "vVert-Drop" not in text_blob
    assert "ripples" not in text_blob
    assert "Technisch bedeutet" not in text_blob
    assert "Technisch passt" not in text_blob
    assert "Technisch ergibt" not in text_blob
    assert "Technisch fuehren" not in text_blob
    assert "harte Korrekturen (harte Korrekturen)" not in text_blob
    assert "Messmodell" not in text_blob
    assert "zeigen das Modell ein" not in text_blob
    assert "Die Messwerte bestaetigt" not in text_blob
    assert "zeigen sich das in" not in text_blob
    assert "zeigen das Best-3s-Fenster geringe" not in text_blob
    assert "und niedrige Werte der waagerechten Geschwindigkeit" not in text_blob
    assert "mit sehr niedrige Werte der waagerechten Geschwindigkeit" not in text_blob
    assert "Speed-Drop" not in text_blob
    assert "harte Korrekturen-RMS" not in text_blob
    assert "Den harte Korrekturen" not in text_blob
    assert "3s-Fenster" in text_blob
    assert "waagerechte Geschwindigkeit" in text_blob
    assert "horizontale Reserve" in text_blob


def test_ai_coaching_simple_prompt_blocks_internal_terms():
    clear_ai_coaching_cache()
    response_payload = {
        "summary": "Der Sprung ist verwertbar.",
        "main_issue": "Der groesste Hebel liegt im Aufbau.",
        "coaching_text": "Die Linie wird spaet unruhig.",
        "next_jump_focus": "Kleine Korrekturen frueher setzen.",
        "confidence_note": "Datenbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    generate_ai_coaching_texts(
        {"schema_version": 1},
        view_mode="simple",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    instructions = fake_client.responses.calls[0]["instructions"]
    assert "Nutze keine internen Begriffe" in instructions
    assert "vHor" in instructions
    assert "Jerk" in instructions
    assert "waagerechte Geschwindigkeit" in instructions
    assert "Vorwaertsbewegung" in instructions


def test_ai_coaching_replaces_suspicious_new_timing_precision_in_focus():
    clear_ai_coaching_cache()
    suspicious_focus = (
        "Ab +10s 0.4-0.8 s frueher Druck aufbauen und den Winkel ruhig halten."
    )
    deterministic_focus = (
        "Im Segment +10 bis +15s frueher Druck aufbauen, aber den Winkel nicht erzwingen "
        "und die Linie ruhiger halten."
    )
    response_payload = {
        "summary": "Der Aufbau liefert zu wenig Zuwachs.",
        "main_issue": "Im Aufbau fehlt frueher Druck.",
        "coaching_text": "Der zu schwache fruehe Aufbau kostet spaeter Stabilitaet.",
        "next_jump_focus": suspicious_focus,
        "confidence_note": "Datenbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    result = generate_ai_coaching_texts(
        {
            "schema_version": 1,
            "jump_brief": {
                "actions": [
                    deterministic_focus,
                    "Die letzte schnelle Phase mit weniger Lenkimpulsen fliegen.",
                ],
            },
        },
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert result["available"] is True
    assert "Der Fokus liegt auf dem Aufbau" in result["next_jump_focus"]
    assert "spaete Gegenkorrekturen" in result["next_jump_focus"]
    assert "0.4-0.8" not in result["next_jump_focus"]


def test_ai_coaching_replaces_summary_when_it_conflicts_with_focus():
    clear_ai_coaching_cache()
    deterministic_summary = "Die groessten Baustellen liegen bei Aufbau 10-20s und Stabilitaet."
    deterministic_focus = "Im Segment +10 bis +15s frueher Druck aufbauen und die Linie ruhiger halten."
    response_payload = {
        "summary": "Der Aufbau ist zu aggressiv: zu fruehes Druecken macht die Linie unruhig.",
        "main_issue": "Im Aufbau fehlt frueher Druck.",
        "coaching_text": "Der zu schwache fruehe Aufbau kostet spaeter Stabilitaet.",
        "next_jump_focus": "Ab +10s frueher Druck aufbauen und die Linie ruhiger halten.",
        "confidence_note": "Datenbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    result = generate_ai_coaching_texts(
        {
            "schema_version": 1,
            "jump_brief": {
                "summary": deterministic_summary,
                "actions": [deterministic_focus],
            },
        },
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert result["available"] is True
    assert result["summary"] == deterministic_summary
    assert result["next_jump_focus"] == "Ab +10s frueher Druck aufbauen und die Linie ruhiger halten."


def test_ai_coaching_daily_limit_blocks_new_uncached_requests():
    clear_ai_coaching_cache()
    response_payload = {
        "summary": "Der Sprung ist verwertbar.",
        "main_issue": "Der groesste Hebel liegt im Aufbau.",
        "coaching_text": "Bleib bis +15s ruhiger im Winkel und steigere erst danach.",
        "next_jump_focus": "Ein klarer Fokus: Aufbau ruhiger halten.",
        "confidence_note": "Datenbasis ist ausreichend.",
    }
    fake_client = _FakeClient(json.dumps(response_payload))

    first = generate_ai_coaching_texts(
        {"schema_version": 1, "metrics": {"best_3s_vVert_kmh": 420.0}},
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        max_requests_per_day=1,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )
    second = generate_ai_coaching_texts(
        {"schema_version": 1, "metrics": {"best_3s_vVert_kmh": 421.0}},
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        max_requests_per_day=1,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert first["available"] is True
    assert second["available"] is False
    assert "Tageslimit" in second["reason"]
    assert len(fake_client.responses.calls) == 1


def test_ai_coaching_does_not_expose_client_exception_text():
    clear_ai_coaching_cache()

    def failing_factory(_key, _timeout):
        raise RuntimeError("secret test token leaked")

    result = generate_ai_coaching_texts(
        {"schema_version": 1},
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=failing_factory,
    )

    assert result["available"] is False
    assert "secret" not in result["reason"]
    assert "test-key" not in result["reason"]
    assert result["error_type"] == "RuntimeError"
    assert "Clientfehler" in result["reason"]


def test_ai_coaching_reports_timeout_category_without_exception_text():
    clear_ai_coaching_cache()

    def failing_factory(_key, _timeout):
        raise TimeoutError("secret timeout detail")

    result = generate_ai_coaching_texts(
        {"schema_version": 1},
        view_mode="expert",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=failing_factory,
    )

    assert result["available"] is False
    assert "Timeout" in result["reason"]
    assert "secret" not in result["reason"]


def test_ai_coaching_rejects_invalid_response_and_falls_back():
    clear_ai_coaching_cache()
    fake_client = _FakeClient(json.dumps({"summary": "zu wenig"}))

    result = generate_ai_coaching_texts(
        {"schema_version": 1},
        view_mode="simple",
        enabled=True,
        model="gpt-5-mini",
        timeout_s=1.0,
        api_key="test-key",
        client_factory=lambda _key, _timeout: fake_client,
    )

    assert result["available"] is False
    assert "Schema" in result["reason"]
