"""Tests propios del agente (M1).

Complementan `tests/conformance/test_m1.py` cubriendo los casos del
ENUNCIADO_M1.md que la suite de la cátedra no incluye explícitamente:
herramienta desconocida (robustez), terminación por max_iterations, y la
realimentación del resultado de la tool a la segunda llamada del LLM.

Usan el MockLLMClient provisto: deterministas, sin claves de API.
"""

from __future__ import annotations

import json

import pytest

from mia_agents.testing import MockLLMClient, make_recording_tool
from mia_agents.types import AgentResult, LLMResponse, ToolCall, ToolSchema

from student_framework import build_agent
from student_framework.agent import MyAgent


def _agent_with(mock: MockLLMClient) -> MyAgent:
    """Agente construido vía build_agent con el cliente mock inyectado."""
    return build_agent({"llm_client": mock})


# --------------------------------------------------------------------------
# 1. Sin tool: un único turno, sin steps.
# --------------------------------------------------------------------------
def test_sin_tool_un_solo_turno() -> None:
    mock = MockLLMClient([LLMResponse(content="cuatro")])
    agent = _agent_with(mock)

    result = agent.run("¿cuánto es 2+2?")

    assert isinstance(result, AgentResult)
    assert result.answer == "cuatro"
    assert result.steps == []
    assert mock.call_count == 1  # no debe re-llamar al LLM


# --------------------------------------------------------------------------
# 2. Con tool: se ejecuta el callable con los args parseados y se registra
#    exactamente un AgentStep con el output exacto.
# --------------------------------------------------------------------------
def test_con_tool_ejecuta_y_registra_step() -> None:
    tool, schema = make_recording_tool(return_value="42")
    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=schema.name,
                        arguments=json.dumps({"text": "hola"}),
                    )
                ],
            ),
            LLMResponse(content="listo"),
        ]
    )
    agent = _agent_with(mock)
    agent.register_tool(tool, schema)

    result = agent.run("usá la tool")

    # El callable recibió los kwargs parseados desde el JSON string.
    assert tool.calls == [{"text": "hola"}]
    # Dos llamadas al LLM: tool_call + respuesta final.
    assert mock.call_count == 2
    assert result.answer == "listo"
    # Exactamente un step, con nombre y output exactos, sin error.
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == schema.name
    assert result.steps[0].tool_output == "42"
    assert result.steps[0].error is None


# --------------------------------------------------------------------------
# 3. Realimentación: el output de la tool aparece en los `messages` de la
#    SEGUNDA llamada a chat (volcado como mensaje role="tool").
# --------------------------------------------------------------------------
def test_output_de_tool_aparece_en_segunda_llamada() -> None:
    tool, schema = make_recording_tool(return_value="recorded:hola")
    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=schema.name,
                        arguments=json.dumps({"text": "hola"}),
                    )
                ],
            ),
            LLMResponse(content="final"),
        ]
    )
    agent = _agent_with(mock)
    agent.register_tool(tool, schema)

    agent.run("disparar")

    # La segunda llamada (índice 1) debe contener el resultado de la tool.
    second_call_messages = mock.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "recorded:hola"


# --------------------------------------------------------------------------
# 4. Tool desconocida (alucinada): no rompe; el step queda con error no nulo.
# --------------------------------------------------------------------------
def test_tool_desconocida_no_rompe() -> None:
    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="herramienta_que_no_existe",
                        arguments=json.dumps({"x": 1}),
                    )
                ],
            ),
            LLMResponse(content="me recuperé"),
        ]
    )
    agent = _agent_with(mock)

    # No debe lanzar excepción.
    result = agent.run("alucina una tool")

    assert isinstance(result, AgentResult)
    assert result.answer == "me recuperé"
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "herramienta_que_no_existe"
    assert result.steps[0].error is not None  # el fallo quedó registrado
    assert result.steps[0].tool_output is None


# --------------------------------------------------------------------------
# 5. Terminación: si el LLM loopea pidiendo tools sin parar, el agente corta
#    al llegar a max_iterations y devuelve igual un AgentResult válido.
# --------------------------------------------------------------------------
def test_corta_en_max_iterations() -> None:
    tool, schema = make_recording_tool()

    # El mock siempre devuelve un tool_call: nunca llega texto final.
    def tool_call_resp() -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="c", name=schema.name, arguments=json.dumps({"text": "x"}))
            ],
        )

    mock = MockLLMClient([tool_call_resp() for _ in range(10)])
    # max_iterations=3: el agente debe parar tras 3 llamadas al LLM.
    agent = MyAgent(llm_client=mock, max_iterations=3)
    agent.register_tool(tool, schema)

    result = agent.run("loop infinito")

    assert isinstance(result, AgentResult)
    assert mock.call_count == 3  # cortó en el límite, no agotó la cola de 10
    # Se ejecutó la tool en cada iteración: 3 steps.
    assert len(result.steps) == 3


# --------------------------------------------------------------------------
# 6. Entradas básicas: no lanza con mensajes triviales ni cadena vacía.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("msg", ["hola", "¿cuánto es 2+2?", ""])
def test_entradas_basicas_no_lanzan(msg: str) -> None:
    mock = MockLLMClient([LLMResponse(content="ok")])
    agent = _agent_with(mock)

    result = agent.run(msg)

    assert isinstance(result, AgentResult)
    assert result.answer == "ok"


# --------------------------------------------------------------------------
# 7. Integración real con la calculadora (sin mockear la tool): el agente
#    pasa los kwargs y el resultado del cálculo llega al step.
# --------------------------------------------------------------------------
def test_calculadora_real_via_agente() -> None:
    from student_framework.tools.calculator import calculator, calculator_schema

    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=calculator_schema.name,
                        arguments=json.dumps({"a": 17, "b": 23, "operator": "*"}),
                    )
                ],
            ),
            LLMResponse(content="El resultado es 391."),
        ]
    )
    agent = _agent_with(mock)  # ya tiene la calculadora registrada
    # (no re-registramos: build_agent ya la incluye)

    result = agent.run("¿cuánto es 17 * 23?")

    assert result.answer == "El resultado es 391."
    # El step de la calculadora trae el resultado exacto del cómputo.
    calc_steps = [s for s in result.steps if s.tool_name == calculator_schema.name]
    assert len(calc_steps) == 1
    assert calc_steps[0].tool_output == "391"
    assert calc_steps[0].error is None
