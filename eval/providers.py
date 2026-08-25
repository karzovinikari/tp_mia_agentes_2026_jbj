"""Configuración de proveedores LLM para la evaluación.

Por qué existe este módulo: la primera versión del harness no registraba
qué modelo había producido cada trial, y `_config_hash` tampoco lo incluía.
Consecuencia: dos corridas con modelos distintos escribían en el MISMO
path y una pisaba a la otra, dejando 143 trials sin poder atribuir a
ningún modelo. Acá se centraliza (a) el modelo por defecto de cada
proveedor, para que una corrida nunca dependa en silencio de lo que haya
en `.env`, y (b) el slug que separa los resultados por modelo en disco.
"""

from __future__ import annotations

import re

PROVIDERS: tuple[str, ...] = ("auto", "ollama", "bedrock", "kimi", "openai")

# Modelo por defecto de cada proveedor cuando no se pasa `--model`.
# `None` = "lo decide el provider" (Bedrock lee `BEDROCK_MODEL_ID`; Kimi
# tiene su propio default interno). Para Ollama y OpenAI lo fijamos acá
# explícitamente: son los dos casos donde un default implícito ya nos
# arruinó una corrida.
DEFAULT_MODELS: dict[str, str | None] = {
    "auto": None,
    "ollama": "qwen2.5",
    "openai": "gpt-4o-mini",
    "kimi": None,
    "bedrock": None,
}


def resolve_model(provider: str, model: str | None) -> str | None:
    """Modelo pedido explícitamente, o el default del proveedor."""
    if model is not None:
        return model
    return DEFAULT_MODELS.get(provider)


def model_slug(model: str | None) -> str:
    """Slug corto y seguro para nombre de archivo.

    `qwen2.5:latest` -> `qwen2-5-latest`; `gpt-4o-mini` -> `gpt-4o-mini`.
    Permite distinguir de un vistazo, con `ls`, qué modelo produjo cada
    trial.
    """
    if not model:
        return "default"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower()
    return slug[:32] or "default"
