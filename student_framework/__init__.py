"""Paquete propio del grupo.

Implementen el agente en `agent.py` y registren sus herramientas a
continuación, en `build_agent`. Tanto el runner de la CLI como los tests
de conformidad llaman a `build_agent`, por lo que esta es la única puerta
de entrada pública de su entrega.
"""

from __future__ import annotations

from typing import Any

from mia_agents.llm_client import LLMClient
from mia_agents.protocols import Agent

from .agent import MyAgent


def build_agent(config: dict[str, Any] | None = None) -> Agent:
    """Construye y configura su agente.

    `config` es opcional. Si se proporciona `config["llm_client"]`, el
    agente debe usarlo (así es como los tests de conformidad inyectan un
    cliente mock). Si no, se construye a partir del entorno.

    TODO (M1): instancien su agente y llamen a `agent.register_tool(...)`
    por cada una de sus herramientas antes de devolverlo.
    """

    config = config or {} #NO CAMBIAR
    llm = config.get("llm_client") or LLMClient.from_env() #NO CAMBIAR
    kwargs: dict[str, Any] = {"llm_client": llm} #NO CAMBIAR
    
    # Reenvío de los demás parámetros que MyAgent.__init__ ya acepta (M3:
    # la infraestructura de evaluación en eval/ los necesita para variar
    # max_iterations/system_prompt entre experimentos sin tocar este
    # archivo). No cambia el comportamiento por defecto: si config no trae
    # estas claves, kwargs queda igual que antes.
    for key in (
        "max_history_messages",
        "max_iterations",
        "system_prompt",
        "max_retries",
        "retry_base_delay",
    ):
        if key in config:
            kwargs[key] = config[key]

    agent = MyAgent(**kwargs)

    # M3 usa esta bandera para una ablación real: cuando es False, las tools
    # de M1 no se registran y por lo tanto tampoco aparecen en el prompt del
    # modelo. El default sigue siendo True para conservar el contrato de M1.
    register_m1_tools = config.get("register_m1_tools", True)

    # Registro de las tres herramientas obligatorias del M1. Cada par
    # (callable, schema) se importa de su módulo en tools/. El schema lo
    # genera ToolSchema.from_callable dentro de cada módulo.
    from student_framework.tools.calculator import calculator, calculator_schema
    from student_framework.tools.file_reader import file_reader, file_reader_schema
    from student_framework.tools.word_counter import word_counter, word_counter_schema

    if register_m1_tools:
        agent.register_tool(calculator, calculator_schema)
        agent.register_tool(file_reader, file_reader_schema)
        agent.register_tool(word_counter, word_counter_schema)

    return agent
