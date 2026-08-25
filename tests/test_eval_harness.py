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

import pytest

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


def test_run_trial_invalid_scenario_is_captured_not_raised(tmp_path: Path) -> None:
    missing = tmp_path / "scenario-inexistente.json"

    trial = run_trial(
        missing,
        provider="mock",
        framework_config={},
        trial_index=0,
    )

    assert trial.crashed is True
    assert trial.crash_stage == "scenario_load"
    assert trial.scenario_id == "scenario-inexistente"
    assert trial.goal_achieved is False
    assert errors.classify_failure(trial) == "evaluation_infrastructure_error"


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


def test_rubric_and_errors_flag_json_with_wrong_tool_signature() -> None:
    """JSON válido pero incompleto también es una llamada mal formada."""
    mock = MockLLMClient(
        [
            _tool_call("use", {"item": "llave_oro"}, "c1"),  # falta target
            LLMResponse(content="no pude"),
        ]
    )
    trial = run_trial(
        EASY_SCENARIO, provider="mock", framework_config={}, trial_index=0, llm_client=mock
    )

    assert trial.agent_result["steps"][0]["error"].startswith("Argumentos inválidos:")
    assert rubric.r3_no_hallucination(trial) == 0.0
    assert errors.classify_failure(trial) == "hallucinated_tool_or_args"


def test_rubric_interface_validity_not_applicable_to_crash_without_steps() -> None:
    mock = MockLLMClient([RuntimeError("boom")])
    trial = run_trial(
        EASY_SCENARIO,
        provider="mock",
        framework_config={"max_retries": 0},
        trial_index=0,
        llm_client=mock,
    )
    assert rubric.r3_no_hallucination(trial) is None


def test_unknown_world_object_is_recorded_as_nonexclusive_incident() -> None:
    mock = MockLLMClient(
        [
            _tool_call("examine", {"target": "objeto_inventado"}, "c1"),
            LLMResponse(content="no pude"),
        ]
    )
    trial = run_trial(
        EASY_SCENARIO, provider="mock", framework_config={}, trial_index=0, llm_client=mock
    )

    assert rubric.r3_no_hallucination(trial) == 1.0  # interfaz válida
    assert "unknown_world_object_reference" in errors.incident_categories(trial)


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


def test_named_cohorts_keep_baseline_separate_from_budget_experiment() -> None:
    from eval.cohorts import select_cohort

    baseline = _run_easy_trial()
    baseline.provider = "ollama"
    baseline.framework_config = {"system_prompt": "prompt"}
    budget = _run_easy_trial()
    budget.provider = "ollama"
    budget.framework_config = {"system_prompt": "prompt", "max_iterations": 21}

    assert select_cohort([baseline, budget], "baseline") == [baseline]
    assert select_cohort([baseline, budget], "budget_21") == [budget]
    assert select_cohort([baseline, budget], "analysis") == [baseline, budget]


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


# ---------------------------------------------------------------------------
# Atribución de modelo — la garantía que faltaba y arruinó 143 trials
#
# Contexto: la primera versión del harness no registraba el modelo y el
# `config_hash` tampoco lo incluía, así que dos corridas con modelos
# distintos escribían en el MISMO archivo y una pisaba a la otra. Estos
# tests fijan las dos propiedades que lo impiden.
# ---------------------------------------------------------------------------


def test_trial_record_registra_el_modelo() -> None:
    mock = MockLLMClient(_optimal_study_with_key_responses())
    trial = run_trial(
        EASY_SCENARIO,
        provider="ollama",
        framework_config={},
        trial_index=0,
        model="qwen2.5",
        llm_client=mock,
    )
    assert trial.model == "qwen2.5"


def test_modelo_default_por_proveedor_se_aplica() -> None:
    """Sin `--model`, cada proveedor tiene un default explícito en eval/
    en vez de depender de lo que haya en `.env`."""
    from eval.providers import resolve_model

    assert resolve_model("ollama", None) == "qwen2.5"
    assert resolve_model("openai", None) == "gpt-4o-mini"
    assert resolve_model("ollama", "llama3.2") == "llama3.2"  # explícito gana


def test_modelos_distintos_no_se_pisan_en_disco(tmp_path) -> None:
    """Dos modelos del mismo proveedor y escenario deben producir paths
    distintos. Sin esto, el segundo sweep sobrescribe al primero."""
    from eval.runner import _config_hash, trial_output_path

    cfg, harness = {"max_iterations": 10}, {"noop_m1_tools": False}
    hash_a = _config_hash(cfg, harness, "ollama", "qwen2.5")
    hash_b = _config_hash(cfg, harness, "ollama", "llama3.2")
    assert hash_a != hash_b, "el hash debe depender del modelo"

    path_a = trial_output_path(tmp_path, "study-with-key", "ollama", hash_a, 0, "qwen2.5")
    path_b = trial_output_path(tmp_path, "study-with-key", "ollama", hash_b, 0, "llama3.2")
    assert path_a != path_b
    # El modelo tiene que ser legible en el nombre, no solo estar en el hash.
    assert "qwen2-5" in path_a.name
    assert "llama3-2" in path_b.name


def test_proveedores_distintos_no_se_pisan_en_disco(tmp_path) -> None:
    from eval.runner import _config_hash

    cfg, harness = {}, {"noop_m1_tools": False}
    assert _config_hash(cfg, harness, "openai", "gpt-4o-mini") != _config_hash(
        cfg, harness, "kimi", "gpt-4o-mini"
    )


def test_modulos_distintos_no_comparten_resume() -> None:
    from eval.runner import _config_hash

    cfg, harness = {}, {"noop_m1_tools": False}
    default_hash = _config_hash(
        cfg, harness, "ollama", "qwen2.5", module="student_framework"
    )
    alternate_hash = _config_hash(
        cfg, harness, "ollama", "qwen2.5", module="alternate_agent"
    )
    assert default_hash != alternate_hash


def test_hosts_distintos_no_comparten_resume() -> None:
    from eval.runner import _config_hash

    cfg, harness = {}, {"noop_m1_tools": False}
    hash_a = _config_hash(
        cfg, harness, "ollama", "qwen2.5", host="http://host-a:11434"
    )
    hash_b = _config_hash(
        cfg, harness, "ollama", "qwen2.5", host="http://host-b:11434"
    )
    assert hash_a != hash_b


def test_m1_tools_can_be_absent_from_llm_prompt() -> None:
    mock = MockLLMClient([LLMResponse(content="fin")])
    run_trial(
        EASY_SCENARIO,
        provider="mock",
        framework_config={"register_m1_tools": False},
        trial_index=0,
        llm_client=mock,
    )

    visible_names = {schema.name for schema in mock.calls[0]["tools"]}
    assert {"look", "examine", "take", "use"} <= visible_names
    assert {"calculator", "file_reader", "word_counter"}.isdisjoint(visible_names)


def test_experiment_c_changes_tool_visibility_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eval import experiments

    observed_configs: list[dict[str, object]] = []

    def fake_run_suite(*_args, framework_config, **_kwargs):
        observed_configs.append(framework_config)
        return []

    monkeypatch.setattr(experiments, "run_suite", fake_run_suite)
    experiments.run_experiment_c(
        provider="ollama",
        model="qwen2.5",
        trials=1,
        scenarios_dir=SCENARIOS_DIR,
        out_dir=tmp_path,
    )

    assert observed_configs[0] == {"system_prompt": experiments.ESCAPE_ROOM_SYSTEM_PROMPT}
    assert observed_configs[1] == {
        "system_prompt": experiments.ESCAPE_ROOM_SYSTEM_PROMPT,
        "register_m1_tools": False,
    }
    assert (tmp_path / "experiment_C_ollama_qwen2-5_comparison.md").exists()
    assert (tmp_path / "experiment_C_ollama_qwen2-5_comparison.json").exists()
