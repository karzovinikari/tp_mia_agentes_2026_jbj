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
`usage.completion_tokens`. Toda esa traducción vive en
`openai_compat.OpenAICompatProvider`, compartida con `OpenAIProvider`;
acá solo quedan las diferencias propias de Moonshot.

Variables de entorno:
  - `KIMI_API_KEY` (o `MOONSHOT_API_KEY`): API key de Moonshot. Obligatoria.
  - `KIMI_MODEL`: modelo (defecto: `kimi-k2-0711-preview`).
  - `KIMI_BASE_URL`: base de la API (defecto: `https://api.moonshot.ai/v1`;
    usar `https://api.moonshot.cn/v1` para la región China).
  - `KIMI_MIN_INTERVAL_S`: espaciado mínimo entre llamadas (defecto: 12s).

Nota sobre `response_format`: Moonshot solo soporta modo JSON (`{"type":
"json_object"}`), no JSON Schema estricto. Si se pasa `response_format`,
se activa ese modo, pero la validación del schema sigue siendo
responsabilidad del agente (como ya ocurre con `BedrockProvider`).

Throttling: la cuenta usada durante M3 devolvió 429 (rate limit) incluso
después de los reintentos con backoff del agente (M2) — parece ser una
ventana corta (~30-40s) con cupo bajo, no un simple "N por minuto"
(medido empíricamente: 10s de espaciado falla ~1 de cada 6 llamadas).
Por eso se combinan dos mecanismos, ambos en la clase base:

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

from typing import Any

from mia_agents._env import load_env_files

from student_framework.openai_compat import OpenAICompatProvider

# Global de módulo (no atributo de clase) porque se lee en cada reintento:
# permite ajustar la espera en runtime sin reconstruir el provider.
_RATE_LIMIT_WAIT_S = 40.0


class KimiProvider(OpenAICompatProvider):
    """Cliente nativo para la API de Kimi (Moonshot AI)."""

    # `kimi-k2-0711-preview` (el default original) fue retirado y devuelve
    # 404. La línea vigente es k2.5 / k2.6 / k3.
    _DEFAULT_MODEL = "kimi-k2.6"
    _DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
    _API_KEY_ENV_VARS = ("KIMI_API_KEY", "MOONSHOT_API_KEY")
    _MODEL_ENV_VAR = "KIMI_MODEL"
    _BASE_URL_ENV_VAR = "KIMI_BASE_URL"
    _MIN_INTERVAL_ENV_VAR = "KIMI_MIN_INTERVAL_S"
    # La cuenta usada en M3 tiene un tope de 3 RPM a nivel organización
    # ("request reached organization max RPM: 3"), o sea una llamada cada
    # 20s. Con menos espaciado el sweep se llena de 429 aunque M2 reintente.
    _DEFAULT_MIN_INTERVAL_S = 21.0
    _RATE_LIMIT_MAX_RETRIES = 2

    def _load_env(self) -> None:
        # Se llama al símbolo de ESTE módulo (no al de la base) para que la
        # carga del `.env` sea neutralizable desde acá en los tests.
        load_env_files()

    def _rate_limit_wait_s(self) -> float:
        # Lee el global del módulo en cada llamada, así se puede ajustar en
        # runtime sin reconstruir el provider.
        return _RATE_LIMIT_WAIT_S

    def _adjust_temperature(self, temperature: float) -> float | None:
        # Toda la línea actual de Moonshot (k2.5, k2.6, k3) rechaza con 400
        # cualquier temperatura distinta de 1: "invalid temperature: only 1
        # is allowed for this model". Se omite el parámetro y se usa la
        # default del proveedor. Consecuencia a documentar: los trials de
        # Kimi no corren a la misma temperatura que los de Ollama/OpenAI.
        return None

    def _build_response_format(
        self, response_format: dict[str, Any]
    ) -> dict[str, Any]:
        # Moonshot no soporta JSON Schema estricto; solo modo JSON. El
        # schema se descarta acá y lo valida el agente (M2).
        return {"type": "json_object"}
