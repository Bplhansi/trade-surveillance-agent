"""Tests for the Ollama (free, local) note writer, with a fake Ollama server."""

import json

import httpx
import pytest

from service.llm import OllamaNoteGenerator, generator_from_env
from service.schemas import Case

CASE = Case(alert_id=1, pattern="spoofing", trader_id=42, symbol_id=3, window_seconds=1,
            window_start_ns=0, window_end_ns=1_000_000_000, stats={"large_cancels": 4},
            event_ids=[5, 6])


def fake_ollama(reply: dict, status: int = 200):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "body": json.loads(request.content)})
        return httpx.Response(status, json=reply)

    return httpx.MockTransport(handler), seen


def test_asks_ollama_for_the_note_schema_and_returns_its_text() -> None:
    transport, seen = fake_ollama({"message": {"role": "assistant", "content": '{"x": 1}'},
                                   "done": True})
    gen = OllamaNoteGenerator(model="qwen2.5:3b", base_url="http://ollama:11434",
                              transport=transport)

    assert gen.generate(CASE) == '{"x": 1}'
    body = seen[0]["body"]
    assert seen[0]["path"] == "/api/chat"
    assert body["model"] == "qwen2.5:3b"
    assert body["stream"] is False
    assert set(body["format"]["required"]) == {"summary", "evidence", "next_check"}
    assert body["messages"][0]["role"] == "system"
    assert "Use ONLY the case data" in body["messages"][0]["content"]
    assert '"trader_id":42' in body["messages"][1]["content"]
    assert body["options"]["temperature"] == 0


def test_an_ollama_error_raises_so_the_service_falls_back() -> None:
    transport, _ = fake_ollama({"error": "model not found"}, status=404)
    gen = OllamaNoteGenerator(model="m", base_url="http://ollama:11434", transport=transport)

    with pytest.raises(httpx.HTTPStatusError):
        gen.generate(CASE)


def test_an_unfinished_reply_raises() -> None:
    transport, _ = fake_ollama({"message": {"content": '{"summ'}, "done": False})
    gen = OllamaNoteGenerator(model="m", base_url="http://ollama:11434", transport=transport)

    with pytest.raises(RuntimeError):
        gen.generate(CASE)


def test_provider_is_chosen_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.2:3b")
    monkeypatch.setenv("OLLAMA_URL", "http://host.docker.internal:11434")

    gen = generator_from_env()

    assert isinstance(gen, OllamaNoteGenerator)
    assert gen.model == "llama3.2:3b"
    assert gen.base_url == "http://host.docker.internal:11434"


def test_ollama_has_sensible_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)

    gen = generator_from_env()

    assert gen.model == "qwen2.5:3b"
    assert gen.base_url == "http://127.0.0.1:11434"


def test_no_provider_means_no_llm(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)  # a local .env may set it

    assert generator_from_env() is None


def test_the_schema_sent_is_strict() -> None:
    transport, seen = fake_ollama({"message": {"content": "{}"}, "done": True})
    OllamaNoteGenerator(model="m", base_url="http://o:1", transport=transport).generate(CASE)

    assert seen[0]["body"]["format"]["additionalProperties"] is False


def test_unknown_provider_is_an_error(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "nope")

    with pytest.raises(ValueError):
        generator_from_env()
