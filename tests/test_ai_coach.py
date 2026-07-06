from __future__ import annotations

import json
from types import SimpleNamespace

from app.services.ai_coach import clear_ai_coaching_cache, generate_ai_coaching_texts


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
    assert "next_jump_focus ist genau eine konkrete Aufgabe" in request["instructions"]
    assert "Ein reines Ergebnisziel" in request["instructions"]


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
    assert result["next_jump_focus"] == actionable_focus
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
        "main_issue": "Technische Bewertung zeigt harte Korrekturen.",
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
            "Fenster mit sehr niedrige Werte der waagerechten Geschwindigkeit."
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
    assert result["next_jump_focus"] == deterministic_focus
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
    assert result["next_jump_focus"] == deterministic_focus


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
