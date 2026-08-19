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
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx

from mia_agents._env import load_env_files
from mia_agents.types import LLMResponse, ToolCall, ToolSchema

ToolSpecInput = ToolSchema | dict[str, Any]

_DEFAULT_MODEL = "kimi-k2-0711-preview"
_DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"


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

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> None:
        # Levanta configuración de un `.env` (mismo mecanismo que
        # `LLMClient.from_env()` usa para Bedrock/Ollama), sin pisar
        # variables ya presentes en el entorno real.
        load_env_files()
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

        resp = self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        return self._to_llm_response(resp.json())

    # -- internos --------------------------------------------------------

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
