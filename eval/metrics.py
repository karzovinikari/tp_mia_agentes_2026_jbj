"""Métricas cuantitativas: success rate y eficiencia.

Ambas son subproductos directos de `TrialRecord` sin instrumentación
extra ni llamadas a LLM — success rate viene de `check_goal` (estado real
del mundo, no del texto del agente); eficiencia compara la cantidad de
`AgentStep` ejecutados contra el óptimo que da `ENUNCIADO_M3.md`.
"""

from __future__ import annotations

from collections import defaultdict

from eval.runner import TrialRecord


def success_rate(trials: list[TrialRecord]) -> float | None:
    """Fracción de trials con goal_achieved=True. None si no hay trials."""
    if not trials:
        return None
    return sum(1 for t in trials if t.goal_achieved) / len(trials)


def success_rate_by_scenario(trials: list[TrialRecord]) -> dict[str, float]:
    groups: dict[str, list[TrialRecord]] = defaultdict(list)
    for t in trials:
        groups[t.scenario_id].append(t)
    return {sid: success_rate(ts) for sid, ts in groups.items()}


def success_rate_macro(trials: list[TrialRecord]) -> float | None:
    """Promedio de success_rate por escenario (no pooled): evita que un
    escenario con muchos trials baratos (study-with-key en Ollama) domine
    sobre uno con pocos trials caros (vault-combination en Bedrock)."""
    per_scenario = success_rate_by_scenario(trials)
    if not per_scenario:
        return None
    return sum(per_scenario.values()) / len(per_scenario)


def efficiency_ratio(trial: TrialRecord) -> float | None:
    """optimal_calls / tool_call_count si goal_achieved, si no None
    (indefinida — se excluye de la agregación, no cuenta como 0).

    Valores en (0, 1]: 1.0 = óptimo, más alto = más eficiente. Se elige
    esta dirección (no `actual/optimal`) para que sea comparable en la
    misma escala "más alto es mejor" que success_rate y rubric_score.
    """
    if not trial.goal_achieved:
        return None
    if trial.optimal_calls is None or not trial.tool_call_count:
        return None
    return trial.optimal_calls / trial.tool_call_count


def mean_efficiency(trials: list[TrialRecord]) -> float | None:
    values = [e for t in trials if (e := efficiency_ratio(t)) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def mean_wall_time(trials: list[TrialRecord]) -> float | None:
    if not trials:
        return None
    return sum(t.wall_time_s for t in trials) / len(trials)


def mean_tokens(trials: list[TrialRecord]) -> tuple[float | None, float | None]:
    """(input_tokens medio, output_tokens medio) sobre trials que reportaron
    tokens. None en cada posición si ningún trial reportó ese campo."""
    inputs = [
        t.agent_result["input_tokens"]
        for t in trials
        if t.agent_result and t.agent_result.get("input_tokens") is not None
    ]
    outputs = [
        t.agent_result["output_tokens"]
        for t in trials
        if t.agent_result and t.agent_result.get("output_tokens") is not None
    ]
    mean_in = sum(inputs) / len(inputs) if inputs else None
    mean_out = sum(outputs) / len(outputs) if outputs else None
    return mean_in, mean_out
