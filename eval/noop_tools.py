"""No-ops de las 3 tools de M1, para el Experimento C (stretch).

`build_agent` registra incondicionalmente `calculator`/`file_reader`/
`word_counter` en todo agente — en el dominio de `mia_world` son ruido
puro (el agente nunca las necesita para abrir una puerta). Reemplazarlas
por no-ops no toca `student_framework`: se reusan los `*_schema` reales
(mismo nombre, misma descripción, mismos parámetros — el LLM ve una tool
idéntica) y se re-registran después de `build_agent()`, aprovechando que
`register_tool` sobrescribe por `schema.name`.
"""

from __future__ import annotations

from typing import Callable

from mia_agents.types import ToolSchema
from student_framework.tools.calculator import calculator_schema
from student_framework.tools.file_reader import file_reader_schema
from student_framework.tools.word_counter import word_counter_schema

_NOOP_MESSAGE = "Esta herramienta no está disponible en este escenario."


def make_noop_m1_tools() -> list[tuple[Callable[..., str], ToolSchema]]:
    def noop(**_kwargs: object) -> str:
        return _NOOP_MESSAGE

    return [
        (noop, calculator_schema),
        (noop, file_reader_schema),
        (noop, word_counter_schema),
    ]
