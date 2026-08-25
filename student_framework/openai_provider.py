"""Cliente propio para la API de OpenAI (`/chat/completions`).

`mia_agents/llm_client.py` es fijo y solo trae Bedrock/Ollama; este
provider sigue la vía soportada — implementar el protocolo
`mia_agents.protocols.LLMClient` en `student_framework/` y pasarlo por
config:

    from student_framework.openai_provider import OpenAIProvider
    agent = build_agent({"llm_client": OpenAIProvider()})

Toda la traducción de mensajes/tools/respuestas vive en
`openai_compat.OpenAICompatProvider` (compartida con `KimiProvider`, que
habla el mismo dialecto). Acá solo quedan las diferencias propias de
OpenAI, que son dos:

  1. **Sin throttling por defecto** (`_DEFAULT_MIN_INTERVAL_S = 0`). Kimi
     necesita 12s de espaciado por su cuota; el tier pago de OpenAI no.
     El reintento ante 429 de la base sigue activo como red por si acaso.
  2. **`response_format` con JSON Schema estricto.** OpenAI soporta
     `{"type": "json_schema", ...}`, no solo el modo JSON genérico, así
     que se reenvía el schema en vez de descartarlo.

Variables de entorno:
  - `OPENAI_API_KEY`: API key. Obligatoria.
  - `OPENAI_MODEL`: modelo (defecto: `gpt-4o-mini`).
  - `OPENAI_BASE_URL`: base de la API (defecto: `https://api.openai.com/v1`).
    Útil para apuntar a un gateway o proxy compatible.
  - `OPENAI_MIN_INTERVAL_S`: espaciado mínimo entre llamadas (defecto: 0).
"""

from __future__ import annotations

from typing import Any

from mia_agents._env import load_env_files

from student_framework.openai_compat import OpenAICompatProvider


# Valores de `type` que OpenAI acepta en `response_format`. Sirven para
# distinguir un `response_format` ya nativo de un JSON Schema pelado: este
# último TAMBIÉN trae `type`, pero con valores de JSON Schema
# ("object", "array", ...), no con estos.
_NATIVE_RESPONSE_FORMAT_TYPES = frozenset({"text", "json_object", "json_schema"})


class OpenAIProvider(OpenAICompatProvider):
    """Cliente nativo para la API de OpenAI."""

    _DEFAULT_MODEL = "gpt-4o-mini"
    _DEFAULT_BASE_URL = "https://api.openai.com/v1"
    _API_KEY_ENV_VARS = ("OPENAI_API_KEY",)
    _MODEL_ENV_VAR = "OPENAI_MODEL"
    _BASE_URL_ENV_VAR = "OPENAI_BASE_URL"
    _MIN_INTERVAL_ENV_VAR = "OPENAI_MIN_INTERVAL_S"
    # Sin espaciado: el tier pago tolera el ritmo de un sweep secuencial.
    _DEFAULT_MIN_INTERVAL_S = 0.0
    _RATE_LIMIT_MAX_RETRIES = 2

    def _load_env(self) -> None:
        load_env_files()

    def _build_response_format(
        self, response_format: dict[str, Any]
    ) -> dict[str, Any]:
        """OpenAI acepta JSON Schema estricto, no solo el modo JSON genérico.

        Si `response_format` ya viene con la forma nativa, se reenvía tal
        cual; si es un JSON Schema pelado, se lo envuelve. No alcanza con
        preguntar si trae `type`: un JSON Schema siempre lo trae (con
        valor "object", "array", ...), así que se compara contra los
        valores nativos de OpenAI. En M2 `structured_call` cierra con la
        tool `final_result`, así que este camino es una mejora oportunista,
        no algo de lo que el agente dependa.
        """
        if response_format.get("type") in _NATIVE_RESPONSE_FORMAT_TYPES:
            return response_format
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": response_format,
                "strict": False,
            },
        }
