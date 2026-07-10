"""Tests propios del Milestone 2.

Complementan `tests/conformance/test_m2.py` cubriendo los criterios de
aprobación del ENUNCIADO_M2 que la suite de la cátedra no ejercita:
invariante de recencia bajo presión de tool_calls, reintentos ante fallos
transitorios (LLM y tools), reparación de structured_call con JSON
malformado, mensajes accionables de las herramientas y un ejemplo
concreto de recuperación end-to-end por cada herramienta.

Deterministas, sin claves de API (MockLLMClient).
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from mia_agents.testing import MockLLMClient, make_recording_tool
from mia_agents.tool_schema import FINAL_RESULT_TOOL_NAME
from mia_agents.types import LLMResponse, ToolCall

from student_framework import build_agent
from student_framework.agent import MyAgent, StructuredOutputError
import student_framework.tools.file_reader as file_reader_module
from student_framework.tools.calculator import calculator
from student_framework.tools.file_reader import file_reader


def _tool_call_resp(name: str, args: dict, call_id: str = "c1") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(args))],
    )


# ---------------------------------------------------------------------------
# Memoria: invariante de recencia y presupuesto
# ---------------------------------------------------------------------------


def test_recencia_bajo_presion_de_tool_calls() -> None:
    """El último mensaje de usuario entra en la ventana aunque un turno con
    muchos tool_calls lo empuje fuera de los últimos N mensajes."""
    tool, schema = make_recording_tool(return_value="ok")
    many_calls = LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id=f"c{i}", name=schema.name, arguments=json.dumps({"text": "x"}))
            for i in range(5)
        ],
    )
    mock = MockLLMClient([many_calls, LLMResponse(content="final")])
    # budget 4: tras el turno con 5 tools, el historial es
    # [user, assistant, tool x5] = 7 mensajes > 4.
    agent = MyAgent(llm_client=mock, max_history_messages=4)
    agent.register_tool(tool, schema)

    result = agent.run("mensaje del usuario que no debe perderse")

    assert result.answer == "final"
    second_messages = mock.calls[1]["messages"]
    # Presupuesto respetado…
    assert len(second_messages) <= 4
    # …y el mensaje de usuario más reciente sigue presente (invariante).
    user_contents = [m["content"] for m in second_messages if m.get("role") == "user"]
    assert "mensaje del usuario que no debe perderse" in user_contents


def test_ventana_nunca_arranca_con_mensaje_tool() -> None:
    """La ventana recortada no puede empezar con role='tool' huérfano
    (los proveedores reales rechazan un toolResult sin su toolUse)."""
    tool, schema = make_recording_tool(return_value="ok")
    responses: list[LLMResponse] = []
    for i in range(4):
        responses.append(_tool_call_resp(schema.name, {"text": f"t{i}"}, f"c{i}"))
    responses.append(LLMResponse(content="fin"))
    mock = MockLLMClient(responses)
    agent = MyAgent(llm_client=mock, max_history_messages=3)
    agent.register_tool(tool, schema)

    agent.run("hola")

    for call in mock.calls:
        msgs = call["messages"]
        assert msgs, "ninguna llamada debería ir con lista vacía"
        assert msgs[0].get("role") != "tool", (
            f"la ventana arrancó con un tool huérfano: {msgs}"
        )


def test_conversacion_larga_sigue_respondiendo() -> None:
    """Decenas de turnos con mensajes grandes: cada run devuelve answer
    no vacío y el presupuesto se respeta siempre (resiliencia del historial)."""
    turns = 30
    budget = 8
    mock = MockLLMClient([LLMResponse(content=f"r{i}") for i in range(turns)])
    agent = build_agent({"llm_client": mock, "max_history_messages": budget})

    for i in range(turns):
        result = agent.run(f"turno {i}: " + "relleno largo " * 200)
        assert result.answer, f"answer vacío en el turno {i}"

    assert max(len(c["messages"]) for c in mock.calls) <= budget


# ---------------------------------------------------------------------------
# Resiliencia: fallos transitorios del LLM y de tools
# ---------------------------------------------------------------------------


def test_timeout_del_llm_se_reintenta_y_termina_ok() -> None:
    mock = MockLLMClient(
        [TimeoutError("simulated timeout"), LLMResponse(content="me recuperé")]
    )
    agent = MyAgent(llm_client=mock, retry_base_delay=0)

    result = agent.run("hola")

    assert result.answer == "me recuperé"
    assert mock.call_count == 2  # 1 fallo + 1 reintento exitoso


def test_error_no_transitorio_aflora_limpio() -> None:
    """Un error no transitorio (p. ej. de programación) NO se reintenta."""
    mock = MockLLMClient([ValueError("bug no transitorio")])
    agent = MyAgent(llm_client=mock, retry_base_delay=0)

    with pytest.raises(ValueError):
        agent.run("hola")
    assert mock.call_count == 1  # sin reintentos


def test_retries_agotados_propagan_el_fallo() -> None:
    mock = MockLLMClient([TimeoutError("t1"), TimeoutError("t2"), TimeoutError("t3")])
    agent = MyAgent(llm_client=mock, max_retries=2, retry_base_delay=0)

    with pytest.raises(TimeoutError):
        agent.run("hola")
    assert mock.call_count == 3  # 1 intento + 2 reintentos


def test_tool_con_fallo_transitorio_se_reintenta() -> None:
    """Una tool que falla una vez con timeout y luego anda: el step sale OK."""
    from mia_agents.types import ToolSchema

    attempts = {"n": 0}

    def flaky(text: str) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("fallo transitorio de la tool")
        return "ok tras reintento"

    schema = ToolSchema.from_callable(flaky, name="flaky", description="tool inestable")
    mock = MockLLMClient(
        [_tool_call_resp("flaky", {"text": "x"}), LLMResponse(content="fin")]
    )
    agent = MyAgent(llm_client=mock, retry_base_delay=0)
    agent.register_tool(flaky, schema)

    result = agent.run("probá la tool")

    assert attempts["n"] == 2
    assert result.steps[0].error is None
    assert result.steps[0].tool_output == "ok tras reintento"


# ---------------------------------------------------------------------------
# Salida estructurada: reparación y fallo limpio
# ---------------------------------------------------------------------------


class _Answer(BaseModel):
    result: int
    comment: str


def _final_result(args: dict | str, call_id: str = "fr") -> LLMResponse:
    arguments = args if isinstance(args, str) else json.dumps(args)
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id=call_id, name=FINAL_RESULT_TOOL_NAME, arguments=arguments)
        ],
    )


def test_structured_repara_texto_libre() -> None:
    mock = MockLLMClient(
        [
            LLMResponse(content="charla libre en vez de la tool"),
            _final_result({"result": 7, "comment": "ok"}),
        ]
    )
    agent = build_agent({"llm_client": mock})

    parsed = agent.structured_call(prompt="dame un objeto", schema=_Answer)

    assert parsed == _Answer(result=7, comment="ok")
    assert mock.call_count == 2
    # El mensaje de reparación menciona la tool y el schema esperado.
    repair_messages = [
        m for m in mock.calls[1]["messages"] if m.get("role") == "user"
    ]
    assert any(FINAL_RESULT_TOOL_NAME in m["content"] for m in repair_messages)


def test_structured_repara_json_malformado() -> None:
    mock = MockLLMClient(
        [
            _final_result("{esto no es json"),
            _final_result({"result": 1, "comment": "bien"}, call_id="fr2"),
        ]
    )
    agent = build_agent({"llm_client": mock})

    parsed = agent.structured_call(prompt="dame un objeto", schema=_Answer)

    assert parsed == _Answer(result=1, comment="bien")
    assert mock.call_count == 2


def test_structured_agotado_lanza_excepcion_propia() -> None:
    mock = MockLLMClient([LLMResponse(content=f"texto {i}") for i in range(3)])
    agent = build_agent({"llm_client": mock})

    with pytest.raises(StructuredOutputError):
        agent.structured_call(
            prompt="dame un objeto", schema=_Answer, max_repair_attempts=2
        )
    assert mock.call_count == 3

    # La primera llamada ofrece final_result en tools (contrato).
    tools = mock.calls[0]["tools"]
    assert any(t.name == FINAL_RESULT_TOOL_NAME for t in tools)


def test_structured_no_toca_el_historial_conversacional() -> None:
    """Decisión de diseño: structured_call es una operación aislada."""
    mock = MockLLMClient(
        [
            _final_result({"result": 1, "comment": "x"}),
            LLMResponse(content="hola"),
        ]
    )
    agent = build_agent({"llm_client": mock})

    agent.structured_call(prompt="extraé algo", schema=_Answer)
    agent.run("charlemos")

    # El run posterior no debe ver el prompt del structured_call.
    run_messages = str(mock.calls[1]["messages"])
    assert "extraé algo" not in run_messages


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_tokens_none_si_nadie_reporta() -> None:
    mock = MockLLMClient([LLMResponse(content="sin tokens")])
    agent = build_agent({"llm_client": mock})

    result = agent.run("hola")

    assert result.input_tokens is None
    assert result.output_tokens is None


# ---------------------------------------------------------------------------
# Herramientas: mensajes accionables y recuperación end-to-end
# ---------------------------------------------------------------------------


def test_calculadora_mensajes_accionables() -> None:
    no_num = calculator(a="tres", b=2, operator="+")
    assert "'a'" in no_num and "'tres'" in no_num  # nombra parámetro y valor

    bad_op = calculator(a=1, b=2, operator="^")
    for op in ("+", "-", "*", "/", "%"):
        assert op in bad_op  # lista los operadores permitidos

    div_zero = calculator(a=1, b=0, operator="/")
    assert "cero" in div_zero and "'b'" in div_zero  # explica la restricción


def test_file_reader_reglas_de_sandbox(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(file_reader_module, "SANDBOX_ROOT", tmp_path)
    (tmp_path / "notas.txt").write_text("contenido", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b", encoding="utf-8")

    assert file_reader(path="notas.txt") == "contenido"
    assert "vacía" in file_reader(path="")
    assert "absoluta" in file_reader(path="/etc/passwd")
    assert "'..'" in file_reader(path="../fuera.txt")

    missing = file_reader(path="no_esta.txt")
    assert "no existe" in missing and "notas.txt" in missing  # lista disponibles

    isdir = file_reader(path="sub")
    assert "directorio" in isdir and "b.txt" in isdir  # lista el contenido


def test_recuperacion_end_to_end_calculadora() -> None:
    """Ejemplo concreto de recuperación: el LLM manda un operador inválido,
    lee el error accionable y reintenta con uno permitido."""
    from student_framework.tools.calculator import calculator_schema

    mock = MockLLMClient(
        [
            _tool_call_resp(
                calculator_schema.name, {"a": 10, "b": 5, "operator": "^"}, "c1"
            ),
            _tool_call_resp(
                calculator_schema.name, {"a": 10, "b": 5, "operator": "*"}, "c2"
            ),
            LLMResponse(content="El resultado es 50."),
        ]
    )
    agent = build_agent({"llm_client": mock})

    result = agent.run("elevá 10 a la 5... digo, multiplicá")

    assert result.answer == "El resultado es 50."
    assert len(result.steps) == 2
    # Primer intento: error recuperable visible para el LLM en los messages.
    second_call = str(mock.calls[1]["messages"])
    assert "Operadores permitidos" in second_call
    # Segundo intento: éxito.
    assert result.steps[1].tool_output == "50"


def test_recuperacion_end_to_end_file_reader(tmp_path, monkeypatch) -> None:
    """Ejemplo concreto de recuperación: ruta inexistente → el error lista
    los archivos disponibles → el LLM reintenta con el nombre correcto."""
    from student_framework.tools.file_reader import file_reader_schema

    monkeypatch.setattr(file_reader_module, "SANDBOX_ROOT", tmp_path)
    (tmp_path / "informe.txt").write_text("dato secreto", encoding="utf-8")

    mock = MockLLMClient(
        [
            _tool_call_resp(file_reader_schema.name, {"path": "reporte.txt"}, "c1"),
            _tool_call_resp(file_reader_schema.name, {"path": "informe.txt"}, "c2"),
            LLMResponse(content="El archivo dice: dato secreto"),
        ]
    )
    agent = build_agent({"llm_client": mock})

    result = agent.run("leé el reporte")

    assert result.answer == "El archivo dice: dato secreto"
    # El primer step falló con mensaje que listaba 'informe.txt'.
    first_error_fed_back = str(mock.calls[1]["messages"])
    assert "informe.txt" in first_error_fed_back
    # El segundo step leyó el archivo correcto.
    assert result.steps[1].tool_output == "dato secreto"
