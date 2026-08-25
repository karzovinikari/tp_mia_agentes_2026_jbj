"""Tests del KimiProvider — mockean el transporte HTTP, no requieren red.

Verifican la traducción entre el formato interno del framework y el
formato "chat completions" (compatible OpenAI) que expone la API de
Moonshot AI (mensajes, herramientas, tool calls, tokens, raw_response).
"""

from __future__ import annotations

import json

import httpx
import pytest

from student_framework.kimi_provider import KimiProvider


def _chat_completion(
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class _Recorder:
    """Handler de MockTransport que graba el último request y devuelve una respuesta fija."""

    def __init__(self) -> None:
        self.last_request: httpx.Request | None = None
        self.response_json: dict = _chat_completion(content="ok")
        self.status_code: int = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        return httpx.Response(self.status_code, json=self.response_json)

    @property
    def sent_body(self) -> dict:
        assert self.last_request is not None
        return json.loads(self.last_request.content)


@pytest.fixture
def recorder() -> _Recorder:
    return _Recorder()


@pytest.fixture
def provider(recorder: _Recorder) -> KimiProvider:
    p = KimiProvider(api_key="test-key", model="kimi-k2-0711-preview", min_interval_s=0)
    p._client = httpx.Client(
        base_url="https://api.moonshot.ai/v1",
        headers=dict(p._client.headers),
        transport=httpx.MockTransport(recorder),
    )
    return p


# ---------------------------------------------------------------------------
# Construcción
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    # Evita que un .env real del repo (con una key de verdad) contamine este
    # caso: `load_env_files()` solo completa variables ausentes, así que un
    # .env legítimo del desarrollador haría que este test no viera nunca la
    # ausencia de key.
    monkeypatch.setattr(
        "student_framework.kimi_provider.load_env_files", lambda *a, **k: None
    )
    with pytest.raises(RuntimeError, match="KIMI_API_KEY"):
        KimiProvider()


def test_constructor_uses_env_when_args_absent(monkeypatch) -> None:
    monkeypatch.setenv("KIMI_API_KEY", "from-env")
    monkeypatch.setenv("KIMI_MODEL", "kimi-k2-turbo-preview")
    provider = KimiProvider()
    assert provider._api_key == "from-env"
    assert provider._model == "kimi-k2-turbo-preview"


def test_authorization_header_sent(provider: KimiProvider, recorder: _Recorder) -> None:
    provider.chat(messages=[{"role": "user", "content": "hola"}])
    assert recorder.last_request.headers["authorization"] == "Bearer test-key"


# ---------------------------------------------------------------------------
# Respuesta de texto simple
# ---------------------------------------------------------------------------


def test_simple_text_response_parsed(provider: KimiProvider, recorder: _Recorder) -> None:
    recorder.response_json = _chat_completion(
        content="Hola mundo", prompt_tokens=100, completion_tokens=50
    )
    result = provider.chat(messages=[{"role": "user", "content": "hola"}])

    assert result.content == "Hola mundo"
    assert result.tool_calls == []
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.raw_response is not None


def test_empty_content_becomes_none(provider: KimiProvider, recorder: _Recorder) -> None:
    recorder.response_json = _chat_completion(content=None)
    result = provider.chat(messages=[{"role": "user", "content": "x"}])
    assert result.content is None


def test_http_error_raises(
    provider: KimiProvider, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "student_framework.kimi_provider._RATE_LIMIT_WAIT_S", 0.0
    )
    recorder.status_code = 429
    recorder.response_json = {"error": {"message": "rate limited"}}
    with pytest.raises(httpx.HTTPStatusError, match="429"):
        provider.chat(messages=[{"role": "user", "content": "x"}])


def test_non_429_http_error_raises_immediately(
    provider: KimiProvider, recorder: _Recorder
) -> None:
    recorder.status_code = 500
    recorder.response_json = {"error": {"message": "server error"}}
    with pytest.raises(httpx.HTTPStatusError, match="500"):
        provider.chat(messages=[{"role": "user", "content": "x"}])


def test_429_then_success_recovers(
    provider: KimiProvider, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "student_framework.kimi_provider._RATE_LIMIT_WAIT_S", 0.0
    )
    calls = {"n": 0}
    real_call = recorder.__call__

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            recorder.status_code = 429
        else:
            recorder.status_code = 200
        return real_call(request)

    provider._client = httpx.Client(
        base_url="https://api.moonshot.ai/v1",
        headers=dict(provider._client.headers),
        transport=httpx.MockTransport(flaky),
    )
    result = provider.chat(messages=[{"role": "user", "content": "x"}])
    assert result.content == "ok"
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Tool calls — respuesta entrante (de Kimi)
# ---------------------------------------------------------------------------


def test_tool_call_parsed(provider: KimiProvider, recorder: _Recorder) -> None:
    recorder.response_json = _chat_completion(
        content=None,
        tool_calls=[
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "examine",
                    "arguments": json.dumps({"target": "alfombra"}),
                },
            }
        ],
        finish_reason="tool_calls",
    )

    result = provider.chat(messages=[{"role": "user", "content": "explora"}])

    assert result.content is None
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.id == "call_abc123"
    assert tc.name == "examine"
    assert json.loads(tc.arguments) == {"target": "alfombra"}


# ---------------------------------------------------------------------------
# Mensaje saliente: system, usuario, asistente, herramientas
# ---------------------------------------------------------------------------


def test_system_prompt_prepended_to_messages(
    provider: KimiProvider, recorder: _Recorder
) -> None:
    provider.chat(
        messages=[{"role": "user", "content": "hola"}],
        system="Eres un asistente.",
    )
    sent = recorder.sent_body["messages"]
    assert sent[0] == {"role": "system", "content": "Eres un asistente."}
    assert sent[1] == {"role": "user", "content": "hola"}


def test_assistant_with_tool_calls_translated(
    provider: KimiProvider, recorder: _Recorder
) -> None:
    provider.chat(
        messages=[
            {"role": "user", "content": "hola"},
            {
                "role": "assistant",
                "content": "voy a examinar",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "examine",
                            "arguments": json.dumps({"target": "alfombra"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "Tienes una llave."},
        ]
    )

    sent = recorder.sent_body["messages"]
    assert sent[1]["role"] == "assistant"
    assert sent[1]["tool_calls"][0]["id"] == "c1"
    assert sent[1]["tool_calls"][0]["function"]["name"] == "examine"
    assert json.loads(sent[1]["tool_calls"][0]["function"]["arguments"]) == {
        "target": "alfombra"
    }
    assert sent[2] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "Tienes una llave.",
    }


# ---------------------------------------------------------------------------
# Esquemas de herramientas
# ---------------------------------------------------------------------------


def test_tools_wrapped_in_openai_function_format(
    provider: KimiProvider, recorder: _Recorder
) -> None:
    provider.chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[
            {
                "name": "look",
                "description": "Mira la sala.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    tools_sent = recorder.sent_body["tools"]
    assert tools_sent == [
        {
            "type": "function",
            "function": {
                "name": "look",
                "description": "Mira la sala.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_no_tools_means_no_tools_key(
    provider: KimiProvider, recorder: _Recorder
) -> None:
    provider.chat(messages=[{"role": "user", "content": "x"}])
    assert "tools" not in recorder.sent_body


# ---------------------------------------------------------------------------
# temperature / max_tokens / response_format
# ---------------------------------------------------------------------------


def test_temperature_omitted_and_max_tokens_sent(
    provider: KimiProvider, recorder: _Recorder
) -> None:
    """La temperatura NO se envía: toda la línea vigente de Moonshot
    (k2.5/k2.6/k3) responde 400 a cualquier valor distinto de 1
    ("invalid temperature: only 1 is allowed for this model"). El modelo
    original `kimi-k2-0711-preview`, que sí la aceptaba, fue retirado."""
    provider.chat(messages=[{"role": "user", "content": "x"}], temperature=0.7)
    body = recorder.sent_body
    assert "temperature" not in body
    assert body["max_tokens"] == 4096  # default


def test_response_format_activates_json_object_mode(
    provider: KimiProvider, recorder: _Recorder
) -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    provider.chat(
        messages=[{"role": "user", "content": "x"}],
        response_format=schema,
    )
    assert recorder.sent_body["response_format"] == {"type": "json_object"}
