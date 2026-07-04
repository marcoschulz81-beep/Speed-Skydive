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
