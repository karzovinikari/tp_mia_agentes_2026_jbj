"""Helpers compartidos por `rubric.py` y `errors.py` para leer la traza.

Detalle importante del dominio: las tools de `mia_world` (look/examine/
take/use/go) NUNCA lanzan excepción — devuelven un string `"Error: ..."`
como `tool_output` normal. Eso significa que `AgentStep.error` (poblado
por `MyAgent._execute_tool_call`) solo se activa para fallos del *marco*
(tool desconocida, JSON inválido, excepción del callable) — no para un
`"Error: ..."` de `mia_world`, que viaja en `tool_output`. Cualquier
regla que quiera detectar "este step falló" tiene que mirar ambos campos.
"""

from __future__ import annotations

import json
from typing import Any


def step_failed(step: dict[str, Any]) -> bool:
    """True si el step falló, sea error del framework o de la tool de mia_world."""
    if step.get("error") is not None:
        return True
    output = step.get("tool_output")
    return isinstance(output, str) and output.startswith("Error")


def canonical_call(step: dict[str, Any]) -> tuple[str | None, str]:
    """(tool_name, args canonicalizados) — reordena las claves del JSON de
    `tool_input` para que dos llamadas con los mismos argumentos en
    distinto orden cuenten como la misma acción."""
    tool_name = step.get("tool_name")
    raw = step.get("tool_input") or "{}"
    try:
        parsed = json.loads(raw)
        canonical = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        canonical = raw
    return tool_name, canonical
