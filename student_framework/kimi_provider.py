"""Cliente propio para Kimi (Moonshot AI) vía su API HTTP compatible con OpenAI.

`mia_agents/llm_client.py` es fijo y solo trae Bedrock/Ollama. Para un
proveedor distinto, la vía soportada es implementar el protocolo
`mia_agents.protocols.LLMClient` (un único método `chat(...) -> LLMResponse`)
en `student_framework/` y pasarlo por config:

    from student_framework.kimi_provider import KimiProvider
    agent = build_agent({"llm_client": KimiProvider()})

La API de Moonshot usa el formato "chat completions" de OpenAI: mensajes
con roles `system`/`user`/`assistant`/`tool`, tool calls con
`function.arguments` como string JSON, y `usage.prompt_tokens` /
`usage.completion_tokens`. Por eso la traducción de mensajes/tools es casi
idéntica a la de `OllamaProvider`.

Variables de entorno:
  - `KIMI_API_KEY` (o `MOONSHOT_API_KEY`): API key de Moonshot. Obligatoria.
  - `KIMI_MODEL`: modelo (defecto: `kimi-k2-0711-preview`).
  - `KIMI_BASE_URL`: base de la API (defecto: `https://api.moonshot.ai/v1`;
    usar `https://api.moonshot.cn/v1` para la región China).

Nota sobre `response_format`: Moonshot solo soporta modo JSON (`{"type":
"json_object"}`), no JSON Schema estricto. Si se pasa `response_format`,
se activa ese modo, pero la validación del schema sigue siendo
responsabilidad del agente (como ya ocurre con `BedrockProvider`).

Throttling: la cuenta usada durante M3 devolvió 429 (rate limit) incluso
después de los reintentos con backoff del agente (M2) — parece ser una
ventana corta (~30-40s) con cupo bajo, no un simple "N por minuto"
(medido empíricamente: 10s de espaciado falla ~1 de cada 6 llamadas).
Por eso `chat()` combina dos mecanismos:

  1. Espaciado mínimo entre llamadas salientes (`KIMI_MIN_INTERVAL_S`,
     default 12s), compartido entre todas las instancias del proceso —
     un `build_agent` nuevo por trial (como hace `eval/runner.py`) no
     debe resetear el throttling.
  2. Reintento propio ante 429 con espera fija de `_RATE_LIMIT_WAIT_S`
     (40s, bastante más que el backoff corto de M2) antes de propagar el
     error — M2 sigue teniendo su propio retry por encima como red
     adicional para otros errores transitorios.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any

import httpx

from mia_agents._env import load_env_files
from mia_agents.types import LLMResponse, ToolCall, ToolSchema

ToolSpecInput = ToolSchema | dict[str, Any]

_DEFAULT_MODEL = "kimi-k2-0711-preview"
_DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
_RATE_LIMIT_MAX_RETRIES = 2
_RATE_LIMIT_WAIT_S = 40.0


def _tool_specs_as_dicts(tools: list[ToolSpecInput] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    specs: list[dict[str, Any]] = []
    for tool in tools:
        specs.append(tool.to_llm_spec() if isinstance(tool, ToolSchema) else tool)
    return specs


def _arguments_to_dict(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, str):
        if not raw_args:
            return {}
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
    return raw_args or {}


class KimiProvider:
    """Cliente nativo para la API de Kimi (Moonshot AI)."""

    # Compartido entre TODAS las instancias del proceso: un sweep de eval
    # crea un `KimiProvider` nuevo por trial, así que el throttle no puede
    # vivir en `self` o se resetea en cada trial y deja de proteger nada.
    _throttle_lock = threading.Lock()
    _last_request_at: float = 0.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        min_interval_s: float | None = None,
    ) -> None:
        # Levanta configuración de un `.env` (mismo mecanismo que
        # `LLMClient.from_env()` usa para Bedrock/Ollama), sin pisar
        # variables ya presentes en el entorno real.
        load_env_files()
        self._min_interval_s = (
            min_interval_s
            if min_interval_s is not None
            else float(os.environ.get("KIMI_MIN_INTERVAL_S", "12.0"))
        )
        self._api_key = (
            api_key
            or os.environ.get("KIMI_API_KEY")
            or os.environ.get("MOONSHOT_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "Define KIMI_API_KEY (o MOONSHOT_API_KEY) o pasá api_key= "
                "al construir KimiProvider."
            )
        self._model = model or os.environ.get("KIMI_MODEL", _DEFAULT_MODEL)
        base = base_url or os.environ.get("KIMI_BASE_URL", _DEFAULT_BASE_URL)
        self._max_tokens = max_tokens
        self._client = httpx.Client(
            base_url=base,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpecInput] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": self._normalize_messages(messages, system),
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if tools:
            body["tools"] = self._format_tools(tools)
        if response_format is not None:
            # Moonshot no soporta JSON Schema estricto; solo modo JSON.
            body["response_format"] = {"type": "json_object"}

        return self._to_llm_response(self._post_with_rate_limit_retry(body))

    def _post_with_rate_limit_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST con reintento propio ante 429, además del retry de M2.

        El 429 de esta cuenta parece ser una ventana corta (~30-40s) con
        cupo bajo, no un simple "N por minuto": el backoff corto de M2
        (`retry_base_delay · 2^intento`, con default 0.5s) no alcanza a
        esperar que la ventana libere cupo. Acá esperamos `_RATE_LIMIT_WAIT_S`
        fijo — mucho más que el backoff de M2 — antes de reintentar, y solo
        para 429 (cualquier otro error HTTP se propaga tal cual para que M2
        decida si es transitorio).
        """
        last_exc: httpx.HTTPStatusError | None = None
        for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            self._throttle()
            resp = self._client.post("/chat/completions", json=body)
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp.json()
            last_exc = httpx.HTTPStatusError(
                f"Client error '429 Too Many Requests' for url {resp.url!r}",
                request=resp.request,
                response=resp,
            )
            if attempt < _RATE_LIMIT_MAX_RETRIES:
                time.sleep(_RATE_LIMIT_WAIT_S)
        assert last_exc is not None
        raise last_exc

    # -- internos --------------------------------------------------------

    def _throttle(self) -> None:
        """Duerme lo necesario para respetar `min_interval_s` entre llamadas
        salientes. El timestamp vive en la clase (no en `self`) para que
        todas las instancias del proceso compartan un único reloj — un
        sweep de eval crea un `KimiProvider` nuevo por trial."""
        cls = KimiProvider
        with cls._throttle_lock:
            now = time.monotonic()
            wait = cls._last_request_at + self._min_interval_s - now
            if wait > 0:
                time.sleep(wait)
            cls._last_request_at = time.monotonic()

    @staticmethod
    def _normalize_messages(
        messages: list[dict[str, Any]], system: str | None
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            role = m.get("role")
            if role == "system":
                # Ya antepuesto vía el parámetro `system`.
                continue
            if role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.get("tool_call_id", ""),
                        "content": str(m.get("content", "")),
                    }
                )
                continue
            if role == "assistant" and m.get("tool_calls"):
                tcs = [
                    {
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": json.dumps(
                                _arguments_to_dict(
                                    tc.get("function", {}).get("arguments")
                                ),
                                ensure_ascii=False,
                            ),
                        },
                    }
                    for tc in m["tool_calls"]
                ]
                out.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": tcs,
                    }
                )
                continue
            out.append(
                {"role": role or "user", "content": m.get("content", "") or ""}
            )
        return out

    @staticmethod
    def _wrap_tool_spec(spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": spec.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            },
        }

    @classmethod
    def _format_tools(cls, tools: list[ToolSpecInput]) -> list[dict[str, Any]]:
        return [cls._wrap_tool_spec(spec) for spec in _tool_specs_as_dicts(tools)]

    @staticmethod
    def _to_llm_response(data: dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments") or "{}",
                )
            )

        usage = data.get("usage") or {}
        return LLMResponse(
            content=message.get("content") or None,
            tool_calls=tool_calls,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            raw_response=data,
        )
