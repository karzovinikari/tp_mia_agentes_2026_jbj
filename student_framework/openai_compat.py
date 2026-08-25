"""Base compartida para proveedores que hablan la API `/chat/completions` de OpenAI.

`mia_agents/llm_client.py` es fijo y solo trae Bedrock/Ollama. Para otros
proveedores, la vía soportada es implementar el protocolo
`mia_agents.protocols.LLMClient` (un único método `chat(...) -> LLMResponse`)
en `student_framework/` y pasarlo por config.

Varios proveedores exponen exactamente el mismo dialecto que OpenAI —
mensajes con roles `system`/`user`/`assistant`/`tool`, tool calls con
`function.arguments` como string JSON, y `usage.prompt_tokens` /
`usage.completion_tokens`. Kimi (Moonshot) es uno; OpenAI es el otro. En
vez de duplicar ~250 líneas de traducción, la parte común vive acá y cada
proveedor concreto se reduce a declarar sus constantes.

Una subclase típica solo define atributos de clase::

    class MiProvider(OpenAICompatProvider):
        _DEFAULT_MODEL = "mi-modelo"
        _DEFAULT_BASE_URL = "https://api.ejemplo.com/v1"
        _API_KEY_ENV_VARS = ("MI_API_KEY",)
        _MODEL_ENV_VAR = "MI_MODEL"
        _BASE_URL_ENV_VAR = "MI_BASE_URL"
        _MIN_INTERVAL_ENV_VAR = "MI_MIN_INTERVAL_S"

Puntos de extensión (métodos, no constantes, para poder ajustarlos en
runtime y en tests): `_load_env`, `_rate_limit_wait_s` y
`_build_response_format`.
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


def tool_specs_as_dicts(tools: list[ToolSpecInput] | None) -> list[dict[str, Any]]:
    """`ToolSchema` -> `to_llm_spec()`; un dict ya normalizado pasa igual."""
    if not tools:
        return []
    return [tool.to_llm_spec() if isinstance(tool, ToolSchema) else tool for tool in tools]


def arguments_to_dict(raw_args: Any) -> dict[str, Any]:
    """`arguments` puede llegar como string JSON o como dict; JSON malformado -> {}."""
    if isinstance(raw_args, str):
        if not raw_args:
            return {}
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
    return raw_args or {}


class OpenAICompatProvider:
    """Cliente HTTP para una API estilo `POST {base_url}/chat/completions`."""

    # --- Configuración que cada subclase declara -------------------------
    _DEFAULT_MODEL: str = ""
    _DEFAULT_BASE_URL: str = ""
    _API_KEY_ENV_VARS: tuple[str, ...] = ()
    _MODEL_ENV_VAR: str = ""
    _BASE_URL_ENV_VAR: str = ""
    _MIN_INTERVAL_ENV_VAR: str = ""
    # Espaciado mínimo entre llamadas salientes. 0 = sin throttling (el
    # default sano para un tier pago); Kimi lo sube porque su cuota es
    # una ventana corta con cupo bajo.
    _DEFAULT_MIN_INTERVAL_S: float = 0.0
    _RATE_LIMIT_MAX_RETRIES: int = 2

    # --- Estado de throttling, POR SUBCLASE ------------------------------
    # Vive en la clase y no en `self` porque un sweep de eval construye un
    # provider nuevo por trial: si viviera en la instancia, el throttle se
    # resetearía en cada trial y no protegería nada.
    _throttle_lock: threading.Lock = threading.Lock()
    _last_request_at: float = 0.0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Cada subclase estrena su propio reloj. Sin esto, todas las
        # subclases compartirían el de la base y un proveedor rápido
        # heredaría el espaciado lento de otro (p. ej. OpenAI esperando los
        # 12s que necesita Kimi).
        cls._throttle_lock = threading.Lock()
        cls._last_request_at = 0.0

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
        self._load_env()

        self._min_interval_s = (
            min_interval_s
            if min_interval_s is not None
            else float(
                os.environ.get(
                    self._MIN_INTERVAL_ENV_VAR, str(self._DEFAULT_MIN_INTERVAL_S)
                )
            )
        )

        self._api_key = api_key or self._api_key_from_env()
        if not self._api_key:
            names = " (o ".join(self._API_KEY_ENV_VARS) + (
                ")" if len(self._API_KEY_ENV_VARS) > 1 else ""
            )
            raise RuntimeError(
                f"Define {names} o pasá api_key= al construir "
                f"{type(self).__name__}."
            )

        self._model = model or os.environ.get(self._MODEL_ENV_VAR, self._DEFAULT_MODEL)
        base = base_url or os.environ.get(self._BASE_URL_ENV_VAR, self._DEFAULT_BASE_URL)
        self._max_tokens = max_tokens
        self._client = httpx.Client(
            base_url=base,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    # -- puntos de extensión ---------------------------------------------

    def _load_env(self) -> None:
        """Carga el `.env`. Es un método (y no una llamada directa) para que
        una subclase pueda enlazarlo al símbolo de su propio módulo, que es
        lo que permite neutralizarlo en tests sin tocar el entorno real."""
        load_env_files()

    def _rate_limit_wait_s(self) -> float:
        """Espera fija tras un 429, antes de reintentar. Método y no
        constante para poder ajustarlo en runtime (los tests lo bajan a 0
        para no dormir de verdad)."""
        return 40.0

    def _adjust_temperature(self, temperature: float) -> float | None:
        """Temperatura efectiva a enviar; `None` la omite del body.

        Algunos modelos (típicamente los de razonamiento) solo aceptan su
        temperatura por defecto y devuelven 400 ante cualquier otro valor.
        Este hook deja que cada proveedor decida sin que el agente tenga
        que saber nada del modelo.
        """
        return temperature

    def _build_response_format(
        self, response_format: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Traduce el `response_format` del framework al del proveedor.

        Por defecto, modo JSON genérico: es el mínimo común denominador.
        Un proveedor que soporte JSON Schema estricto puede devolver el
        schema completo. Devolver `None` lo omite del body.
        """
        return {"type": "json_object"}

    def _api_key_from_env(self) -> str | None:
        for var in self._API_KEY_ENV_VARS:
            value = os.environ.get(var)
            if value:
                return value
        return None

    # -- API pública ------------------------------------------------------

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
            "max_tokens": self._max_tokens,
        }
        effective_temperature = self._adjust_temperature(temperature)
        if effective_temperature is not None:
            body["temperature"] = effective_temperature
        if tools:
            body["tools"] = self._format_tools(tools)
        if response_format is not None:
            translated = self._build_response_format(response_format)
            if translated is not None:
                body["response_format"] = translated

        return self._to_llm_response(self._post_with_rate_limit_retry(body))

    # -- internos ---------------------------------------------------------

    def _post_with_rate_limit_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST con reintento propio ante 429, además del retry de M2.

        El 429 de un tier gratuito suele ser una ventana corta con cupo
        bajo: el backoff de M2 (`retry_base_delay · 2^intento`, default
        0.5s) no alcanza a esperar que la ventana libere. Acá se espera un
        valor fijo, mucho mayor, y SOLO para 429; cualquier otro error HTTP
        se propaga tal cual para que M2 decida si es transitorio.
        """
        last_exc: httpx.HTTPStatusError | None = None
        max_retries = self._RATE_LIMIT_MAX_RETRIES
        for attempt in range(max_retries + 1):
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
            if attempt < max_retries:
                time.sleep(self._rate_limit_wait_s())
        assert last_exc is not None
        raise last_exc

    def _throttle(self) -> None:
        """Duerme lo necesario para respetar `min_interval_s` entre llamadas.

        `type(self)` y no una clase hardcodeada: cada subclase tiene su
        propio reloj (ver `__init_subclass__`).
        """
        cls = type(self)
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
                                arguments_to_dict(
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
            out.append({"role": role or "user", "content": m.get("content", "") or ""})
        return out

    @staticmethod
    def _wrap_tool_spec(spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
            },
        }

    @classmethod
    def _format_tools(cls, tools: list[ToolSpecInput]) -> list[dict[str, Any]]:
        return [cls._wrap_tool_spec(spec) for spec in tool_specs_as_dicts(tools)]

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
                    # Se deja como string JSON: es el contrato de ToolCall.
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
