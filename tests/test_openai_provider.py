"""Tests del OpenAIProvider — mockean el transporte HTTP, no requieren red ni API key real.

Mismo patrón que `tests/test_kimi_provider.py` (`httpx.MockTransport`).
Ambos providers comparten `OpenAICompatProvider`, así que estos tests se
concentran en lo que OpenAI hace DISTINTO —sin throttling por defecto,
`response_format` con JSON Schema— más una verificación de que la parte
compartida sigue funcionando desde esta subclase, y de que el throttle es
independiente del de Kimi.
"""

from __future__ import annotations

import json

import httpx
import pytest

from student_framework.kimi_provider import KimiProvider
from student_framework.openai_provider import OpenAIProvider


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
    """Handler de MockTransport: graba el último request y devuelve una respuesta fija."""

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
def provider(recorder: _Recorder) -> OpenAIProvider:
    p = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
    p._client = httpx.Client(
        base_url="https://api.openai.com/v1",
        headers=dict(p._client.headers),
        transport=httpx.MockTransport(recorder),
    )
    return p


# ---------------------------------------------------------------------------
# Construcción y configuración
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Sin esto, un `.env` real del repo repondría la clave y el test mentiría.
    monkeypatch.setattr(
        "student_framework.openai_provider.load_env_files", lambda *a, **k: None
    )
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIProvider()


def test_constructor_uses_env_when_args_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    p = OpenAIProvider()
    assert p._api_key == "env-key"
    assert p._model == "gpt-4o"


def test_default_model_is_gpt_4o_mini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    p = OpenAIProvider(api_key="k")
    assert p._model == "gpt-4o-mini"


def test_no_throttling_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diferencia clave con Kimi: OpenAI no espacia llamadas por defecto."""
    monkeypatch.delenv("OPENAI_MIN_INTERVAL_S", raising=False)
    assert OpenAIProvider(api_key="k")._min_interval_s == 0.0
    # Kimi espacia 21s: la cuenta tiene un tope de 3 RPM.
    assert KimiProvider(api_key="k")._min_interval_s == 21.0


def test_throttle_state_is_independent_from_kimi() -> None:
    """Cada subclase tiene su propio reloj de throttling.

    Si compartieran el estado de la clase base, OpenAI heredaría el
    espaciado de 12s que Kimi necesita y el sweep tardaría horas de más.
    """
    assert OpenAIProvider._throttle_lock is not KimiProvider._throttle_lock

    # Se comparan valores antes/después en vez de asumir estado prístino:
    # otros tests del archivo pueden haber marcado ya el reloj de Kimi.
    kimi_before = KimiProvider._last_request_at
    OpenAIProvider(api_key="k")._throttle()

    assert OpenAIProvider._last_request_at > 0.0
    assert KimiProvider._last_request_at == kimi_before


def test_authorization_header_sent(provider: OpenAIProvider, recorder: _Recorder) -> None:
    provider.chat(messages=[{"role": "user", "content": "hola"}])
    assert recorder.last_request is not None
    assert recorder.last_request.headers["authorization"] == "Bearer test-key"


# ---------------------------------------------------------------------------
# Respuestas
# ---------------------------------------------------------------------------


def test_simple_text_response_parsed(provider: OpenAIProvider, recorder: _Recorder) -> None:
    recorder.response_json = _chat_completion(
        content="la puerta está abierta", prompt_tokens=100, completion_tokens=50
    )
    result = provider.chat(messages=[{"role": "user", "content": "hola"}])

    assert result.content == "la puerta está abierta"
    assert result.tool_calls == []
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.raw_response is not None


def test_empty_content_becomes_none(provider: OpenAIProvider, recorder: _Recorder) -> None:
    recorder.response_json = _chat_completion(content=None)
    assert provider.chat(messages=[{"role": "user", "content": "hola"}]).content is None


def test_tool_call_parsed(provider: OpenAIProvider, recorder: _Recorder) -> None:
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
    result = provider.chat(messages=[{"role": "user", "content": "mirá la alfombra"}])

    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.id == "call_abc123"
    assert tc.name == "examine"
    assert json.loads(tc.arguments) == {"target": "alfombra"}


def test_http_error_raises(
    provider: OpenAIProvider, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OpenAIProvider, "_rate_limit_wait_s", lambda self: 0.0)
    recorder.status_code = 429
    with pytest.raises(httpx.HTTPStatusError, match="429"):
        provider.chat(messages=[{"role": "user", "content": "hola"}])


def test_non_429_http_error_raises_immediately(
    provider: OpenAIProvider, recorder: _Recorder
) -> None:
    """Un 500 se propaga sin dormir: M2 decide si es transitorio."""
    recorder.status_code = 500
    with pytest.raises(httpx.HTTPStatusError):
        provider.chat(messages=[{"role": "user", "content": "hola"}])


# ---------------------------------------------------------------------------
# Mensajes salientes (lógica compartida, verificada desde esta subclase)
# ---------------------------------------------------------------------------


def test_system_prompt_prepended(provider: OpenAIProvider, recorder: _Recorder) -> None:
    provider.chat(
        messages=[{"role": "user", "content": "hola"}], system="Sos un agente."
    )
    sent = recorder.sent_body["messages"]
    assert sent[0] == {"role": "system", "content": "Sos un agente."}
    assert sent[1] == {"role": "user", "content": "hola"}


def test_assistant_with_tool_calls_and_tool_result_translated(
    provider: OpenAIProvider, recorder: _Recorder
) -> None:
    provider.chat(
        messages=[
            {"role": "user", "content": "abrí la puerta"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "take", "arguments": '{"item": "llave"}'},
                    }
                ],
            },
            {"role": "tool", "content": "Tomas llave.", "tool_call_id": "c1"},
        ]
    )
    sent = recorder.sent_body["messages"]
    assert sent[1]["tool_calls"][0]["id"] == "c1"
    assert sent[1]["tool_calls"][0]["function"]["name"] == "take"
    assert json.loads(sent[1]["tool_calls"][0]["function"]["arguments"]) == {"item": "llave"}
    assert sent[2] == {"role": "tool", "tool_call_id": "c1", "content": "Tomas llave."}


def test_tool_schema_wrapped_in_openai_format(
    provider: OpenAIProvider, recorder: _Recorder
) -> None:
    """Se pasa un ToolSchema real (no un dict), ejercitando to_llm_spec()."""
    from mia_world.tools import examine_schema

    provider.chat(messages=[{"role": "user", "content": "hola"}], tools=[examine_schema])
    sent = recorder.sent_body["tools"]
    assert len(sent) == 1
    assert sent[0]["type"] == "function"
    assert sent[0]["function"]["name"] == "examine"
    assert "target" in sent[0]["function"]["parameters"]["properties"]


def test_no_tools_means_no_tools_key(provider: OpenAIProvider, recorder: _Recorder) -> None:
    provider.chat(messages=[{"role": "user", "content": "hola"}])
    assert "tools" not in recorder.sent_body


# ---------------------------------------------------------------------------
# response_format: OpenAI soporta JSON Schema (Kimi no)
# ---------------------------------------------------------------------------


def test_response_format_wraps_bare_json_schema(
    provider: OpenAIProvider, recorder: _Recorder
) -> None:
    schema = {"type": "object", "properties": {"result": {"type": "integer"}}}
    provider.chat(messages=[{"role": "user", "content": "dame un objeto"}], response_format=schema)
    sent = recorder.sent_body["response_format"]
    assert sent["type"] == "json_schema"
    assert sent["json_schema"]["schema"] == schema


def test_response_format_passthrough_when_already_native(
    provider: OpenAIProvider, recorder: _Recorder
) -> None:
    native = {"type": "json_object"}
    provider.chat(messages=[{"role": "user", "content": "hola"}], response_format=native)
    assert recorder.sent_body["response_format"] == native


def test_kimi_still_forces_json_object_mode() -> None:
    """Contraste explícito: Kimi descarta el schema, OpenAI lo reenvía."""
    p = KimiProvider(api_key="k", min_interval_s=0)
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    assert p._build_response_format(schema) == {"type": "json_object"}


# ---------------------------------------------------------------------------
# temperature: los modelos de razonamiento de Moonshot solo aceptan la suya
# ---------------------------------------------------------------------------


def test_openai_sends_temperature(provider: OpenAIProvider, recorder: _Recorder) -> None:
    provider.chat(messages=[{"role": "user", "content": "hola"}], temperature=0.7)
    assert recorder.sent_body["temperature"] == 0.7


def test_kimi_omits_temperature(recorder: _Recorder) -> None:
    """La línea actual de Moonshot (k2.5/k2.6/k3) responde 400 a cualquier
    temperatura distinta de 1, así que el parámetro se omite del body."""
    p = KimiProvider(api_key="test-key", min_interval_s=0)
    p._client = httpx.Client(
        base_url="https://api.moonshot.ai/v1",
        headers=dict(p._client.headers),
        transport=httpx.MockTransport(recorder),
    )
    p.chat(messages=[{"role": "user", "content": "hola"}], temperature=0.2)
    assert "temperature" not in recorder.sent_body
