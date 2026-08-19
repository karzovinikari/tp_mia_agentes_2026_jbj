"""Tests propios de la infraestructura de evaluación (eval/).

Corren `run_trial` con `MockLLMClient` (mismo patrón que
`tests/test_my_agent.py`), nunca contra un LLM real — verifican que
`TrialRecord`, `metrics`, `rubric` y `errors` dan los valores esperados
sobre trazas conocidas de antemano. Sin esto, un bug en `eval/` podría
contaminar silenciosamente el informe "oficial" sin que ningún test lo note.
"""

from __future__ import annotations

import json
from pathlib import Path

from mia_agents.testing import MockLLMClient
from mia_agents.types import LLMResponse, ToolCall

from eval import errors, metrics, rubric, scenario_meta
from eval.runner import TrialRecord, run_trial

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"
EASY_SCENARIO = SCENARIOS_DIR / "01-study-with-key.json"


def _tool_call(name: str, args: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(args))],
    )


def _optimal_study_with_key_responses() -> list[LLMResponse]:
    """La solución óptima de study-with-key (3 tool_calls), una por
    respuesta del LLM, más el texto final — mismo patrón que
    `tests/conformance/test_m3_world.py`."""
    return [
        _tool_call("examine", {"target": "alfombra"}, "c1"),
        _tool_call("take", {"item": "llave_oro"}, "c2"),
        _tool_call("use", {"item": "llave_oro", "target": "puerta_principal"}, "c3"),
        LLMResponse(content="Listo, abrí la puerta con la llave dorada."),
    ]


def _run_easy_trial(**kwargs) -> TrialRecord:
    mock = MockLLMClient(_optimal_study_with_key_responses())
    return run_trial(
        EASY_SCENARIO,
        provider="mock",
        framework_config={},
        trial_index=0,
        llm_client=mock,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# run_trial / TrialRecord
# ---------------------------------------------------------------------------


def test_run_trial_optimal_solution_succeeds() -> None:
    trial = _run_easy_trial()

    assert trial.crashed is False
    assert trial.goal_achieved is True
    assert trial.scenario_id == "study-with-key"
    assert trial.difficulty == "easy"
    assert trial.tool_call_count == 3  # examine, take, use — no cuenta el texto final
    assert trial.optimal_calls == scenario_meta.OPTIMAL_TOOL_CALLS["study-with-key"]
    assert "abierta" in trial.goal_reason


def test_run_trial_fresh_world_per_call() -> None:
    """Dos llamadas a run_trial sobre el mismo scenario_path no deben
    compartir estado: la segunda debe volver a resolver desde cero."""
    trial1 = _run_easy_trial()
    trial2 = _run_easy_trial()

    assert trial1.goal_achieved is True
    assert trial2.goal_achieved is True
    # Si compartieran World, el segundo tomaría 0 pasos porque la puerta
    # ya estaría abierta desde el primer trial.
    assert trial2.tool_call_count == 3


def test_run_trial_crash_is_captured_not_raised() -> None:
    mock = MockLLMClient([RuntimeError("proveedor caído tras agotar reintentos")])
    trial = run_trial(
        EASY_SCENARIO,
        provider="mock",
        framework_config={"max_retries": 0, "retry_base_delay": 0},
        trial_index=0,
        llm_client=mock,
    )

    assert trial.crashed is True
    assert "proveedor caído" in trial.crash_message
    assert trial.goal_achieved is False  # check_goal igual se evalúa sobre el world sin tocar


def test_run_trial_unknown_tool_recorded_as_step_error() -> None:
    mock = MockLLMClient(
        [
            _tool_call("herramienta_inventada", {}, "c1"),
            LLMResponse(content="me rindo"),
        ]
    )
    trial = run_trial(
        EASY_SCENARIO, provider="mock", framework_config={}, trial_index=0, llm_client=mock
    )

    assert trial.crashed is False
    assert trial.goal_achieved is False
    steps = trial.agent_result["steps"]
    assert steps[0]["error"].startswith("Herramienta desconocida:")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_efficiency_ratio_optimal_is_one() -> None:
    trial = _run_easy_trial()
    assert metrics.efficiency_ratio(trial) == 1.0  # 3 óptimo / 3 real


def test_efficiency_ratio_none_when_goal_not_achieved() -> None:
    mock = MockLLMClient([LLMResponse(content="no hice nada")])
    trial = run_trial(
        EASY_SCENARIO, provider="mock", framework_config={}, trial_index=0, llm_client=mock
    )
    assert trial.goal_achieved is False
    assert metrics.efficiency_ratio(trial) is None


def test_success_rate_and_macro() -> None:
    trials = [_run_easy_trial(), _run_easy_trial()]
    assert metrics.success_rate(trials) == 1.0
    assert metrics.success_rate_macro(trials) == 1.0


# ---------------------------------------------------------------------------
# rubric
# ---------------------------------------------------------------------------


def test_rubric_look_first_false_when_first_step_is_not_look() -> None:
    trial = _run_easy_trial()  # empieza con examine, no look
    assert rubric.r1_look_first(trial) == 0.0


def test_rubric_no_hallucination_flags_unknown_tool() -> None:
    mock = MockLLMClient(
        [_tool_call("no_existe", {}, "c1"), LLMResponse(content="fin")]
    )
    trial = run_trial(
        EASY_SCENARIO, provider="mock", framework_config={}, trial_index=0, llm_client=mock
    )
    assert rubric.r3_no_hallucination(trial) == 0.0


def test_rubric_no_loop_detects_repeated_failed_call() -> None:
    """use(llave_oro, puerta_principal) antes de tomarla falla dos veces
    seguidas con el mismo argumento -> R2 debe marcar 0.0."""
    mock = MockLLMClient(
        [
            _tool_call("use", {"item": "llave_oro", "target": "puerta_principal"}, "c1"),
            _tool_call("use", {"item": "llave_oro", "target": "puerta_principal"}, "c2"),
            LLMResponse(content="no pude"),
        ]
    )
    trial = run_trial(
        EASY_SCENARIO, provider="mock", framework_config={}, trial_index=0, llm_client=mock
    )
    assert rubric.r2_no_loop(trial) == 0.0


def test_rubric_sequence_order_not_applicable_for_item_open_goal() -> None:
    trial = _run_easy_trial()  # goal de study-with-key es item_open, no sequence
    assert rubric.r4_sequence_order(trial) is None


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def test_classify_failure_none_when_achieved() -> None:
    trial = _run_easy_trial()
    assert errors.classify_failure(trial) is None


def test_classify_failure_provider_error() -> None:
    mock = MockLLMClient([RuntimeError("boom")])
    trial = run_trial(
        EASY_SCENARIO,
        provider="mock",
        framework_config={"max_retries": 0, "retry_base_delay": 0},
        trial_index=0,
        llm_client=mock,
    )
    assert errors.classify_failure(trial) == "provider_error"


def test_classify_failure_max_iterations_exhausted() -> None:
    """El mock siempre devuelve el mismo tool_call fallido -> nunca converge
    a texto final -> agota max_iterations."""
    loop_call = _tool_call("examine", {"target": "alfombra"}, "c")
    mock = MockLLMClient([loop_call] * 3)
    trial = run_trial(
        EASY_SCENARIO,
        provider="mock",
        framework_config={"max_iterations": 3},
        trial_index=0,
        llm_client=mock,
    )
    assert trial.goal_achieved is False
    assert errors.classify_failure(trial) == "max_iterations_exhausted"


def test_classify_failure_hallucinated_tool() -> None:
    mock = MockLLMClient(
        [_tool_call("teletransportarse", {}, "c1"), LLMResponse(content="no pude")]
    )
    trial = run_trial(
        EASY_SCENARIO, provider="mock", framework_config={}, trial_index=0, llm_client=mock
    )
    assert errors.classify_failure(trial) == "hallucinated_tool_or_args"


def test_error_breakdown_denominators() -> None:
    ok = _run_easy_trial()
    mock = MockLLMClient([LLMResponse(content="no hice nada")])
    fail = run_trial(
        EASY_SCENARIO, provider="mock", framework_config={}, trial_index=0, llm_client=mock
    )
    breakdown = errors.error_breakdown([ok, fail])
    assert breakdown["total_trials"] == 2
    assert breakdown["total_failures"] == 1
    assert sum(breakdown["counts"].values()) == 1


# ---------------------------------------------------------------------------
# scenario_meta — blindaje contra typo de transcripción del enunciado
# ---------------------------------------------------------------------------


def test_optimal_tool_calls_covers_all_scenarios() -> None:
    scenario_files = sorted(SCENARIOS_DIR.glob("*.json"))
    assert len(scenario_files) == 8
    assert set(scenario_meta.OPTIMAL_TOOL_CALLS.keys()) == {
        "study-with-key",
        "color-locks",
        "apartment-keys",
        "library-search",
        "office-sequence",
        "extreme-archive",
        "vault-combination",
        "backtracking-vault",
    }
