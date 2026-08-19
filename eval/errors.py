"""Taxonomía de modos de fallo, derivada de forma determinista.

Reglas evaluadas en orden de prioridad — la primera que matchea gana.
Las primeras cuatro son *string matching exacto* sobre mensajes que el
propio framework produce (`agent.py`, `mia_world/goals.py`) — no
heurística difusa. Solo `context_window_too_small` es una heurística
declarada como tal (proxy débil, no una prueba).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

from eval.runner import TrialRecord
from eval.scenario_meta import MULTI_ROOM_SCENARIOS
from eval.trace_utils import step_failed

_MAX_ITERATIONS_RE = re.compile(r"^Se alcanzó el límite de \d+ iteraciones sin respuesta final\.")

# Frases que las tools de mia_world devuelven cuando el agente repite una
# acción que ya tuvo éxito antes (proxy de "olvidó que ya lo había hecho").
_ALREADY_DONE_MARKERS = ("ya llevas", "ya está abierta", "ya habías colocado")


def _is_provider_error(trial: TrialRecord) -> bool:
    return trial.crashed


def _is_max_iterations_exhausted(trial: TrialRecord) -> bool:
    err = (trial.agent_result or {}).get("error") if trial.agent_result else None
    return bool(err) and bool(_MAX_ITERATIONS_RE.match(err))


def _is_hallucinated_tool_or_args(trial: TrialRecord) -> bool:
    for step in (trial.agent_result or {}).get("steps", []):
        err = step.get("error")
        if err and (
            err.startswith("Herramienta desconocida:") or err.startswith("Argumentos JSON inválidos:")
        ):
            return True
    return False


def _is_sequence_order_violated(trial: TrialRecord) -> bool:
    if trial.goal.get("type") != "sequence":
        return False
    return "orden" in (trial.goal_reason or "")


def _is_context_window_too_small(trial: TrialRecord) -> bool:
    """Heurística débil: escenario multi-sala + >=2 steps que repiten una
    acción ya completada, separados por >=3 steps de distancia — proxy de
    'perdió de vista un logro anterior', consistente con una ventana de
    historial recortada. No es una prueba, se etiqueta así en el informe."""
    if trial.scenario_id not in MULTI_ROOM_SCENARIOS:
        return False
    steps = (trial.agent_result or {}).get("steps", [])
    hits = [
        i
        for i, s in enumerate(steps)
        if isinstance(s.get("tool_output"), str)
        and any(marker in s["tool_output"].lower() for marker in _ALREADY_DONE_MARKERS)
    ]
    if len(hits) < 2:
        return False
    return (hits[-1] - hits[0]) >= 3


def _is_unrecovered_tool_error(trial: TrialRecord) -> bool:
    steps = (trial.agent_result or {}).get("steps", [])
    if not steps:
        return False
    return step_failed(steps[-1])


# Orden de prioridad: la primera regla que matchea gana.
FAILURE_RULES: list[tuple[str, Callable[[TrialRecord], bool]]] = [
    ("provider_error", _is_provider_error),
    ("max_iterations_exhausted", _is_max_iterations_exhausted),
    ("hallucinated_tool_or_args", _is_hallucinated_tool_or_args),
    ("sequence_order_violated", _is_sequence_order_violated),
    ("context_window_too_small", _is_context_window_too_small),
    ("unrecovered_tool_error", _is_unrecovered_tool_error),
]


def classify_failure(trial: TrialRecord) -> str | None:
    """None si el trial tuvo éxito. Si no, la primera categoría cuya regla
    matchea, o "unclassified" si ninguna aplica."""
    if trial.goal_achieved:
        return None
    for name, rule in FAILURE_RULES:
        if rule(trial):
            return name
    return "unclassified"


def error_breakdown(trials: list[TrialRecord]) -> dict[str, object]:
    """Conteo por categoría + % sobre el total de fallos (denominador
    explícito, no un porcentaje suelto sin contexto)."""
    failures = [t for t in trials if not t.goal_achieved]
    counts: dict[str, int] = defaultdict(int)
    for t in failures:
        counts[classify_failure(t)] += 1
    total_failures = len(failures)
    return {
        "total_trials": len(trials),
        "total_failures": total_failures,
        "counts": dict(counts),
        "pct_of_failures": {
            k: (v / total_failures if total_failures else None) for k, v in counts.items()
        },
    }


def error_breakdown_by_scenario(trials: list[TrialRecord]) -> dict[str, dict[str, object]]:
    groups: dict[str, list[TrialRecord]] = defaultdict(list)
    for t in trials:
        groups[t.scenario_id].append(t)
    return {sid: error_breakdown(ts) for sid, ts in groups.items()}
