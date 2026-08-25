"""Rúbrica cualitativa determinista.

Cinco criterios, cada uno una función pura `TrialRecord -> float | None`
(`None` = no aplicable a este trial, se excluye del promedio de ese trial
en vez de contar como 0 — p. ej. R4 en un escenario sin goal `sequence`).
Deliberadamente NO es LLM-as-judge: cada criterio es string-matching o
aritmética simple sobre mensajes que el propio framework ya produce de
forma determinista (documentados en `CLAUDE.md`), reproducible sin costo
de LLM y defendible oralmente sin apelar a una caja negra.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from eval.metrics import efficiency_ratio
from eval.runner import TrialRecord
from eval.trace_utils import canonical_call, step_failed

CRITERIA = ("R1_look_first", "R2_no_loop", "R3_no_hallucination", "R4_sequence_order", "R5_efficiency")


def _steps(trial: TrialRecord) -> list[dict]:
    if trial.agent_result is None:
        return []
    return trial.agent_result.get("steps", [])


def r1_look_first(trial: TrialRecord) -> float | None:
    """1.0 si la primera acción del agente fue `look` (observar antes de
    actuar a ciegas). None si el agente no ejecutó ninguna tool (respondió
    directo con texto) — no aplica el criterio."""
    steps = _steps(trial)
    if not steps:
        return None
    return 1.0 if steps[0].get("tool_name") == "look" else 0.0


def r2_no_loop(trial: TrialRecord) -> float | None:
    """0.0 si alguna (tool, args canonicalizados) se repite en >=2 steps
    fallidos — el agente no aprendió del error y repitió la misma acción
    inútil. None si no hubo ningún step fallido (no aplica)."""
    steps = _steps(trial)
    failed_calls = [canonical_call(s) for s in steps if step_failed(s)]
    if not failed_calls:
        return None
    counts = Counter(failed_calls)
    return 0.0 if any(c >= 2 for c in counts.values()) else 1.0


def r3_no_hallucination(trial: TrialRecord) -> float | None:
    """0.0 si algún step tiene tool desconocida o argumentos mal formados.

    Cubre JSON inválido y JSON válido que no coincide con la firma de la
    herramienta. No intenta juzgar errores semánticos del mundo (por ejemplo,
    probar una llave válida en la puerta equivocada).
    """
    steps = _steps(trial)
    if trial.crashed and not steps:
        return None
    for step in steps:
        err = step.get("error")
        if err and (
            err.startswith("Herramienta desconocida:")
            or err.startswith("Argumentos JSON inválidos:")
            or err.startswith("Argumentos inválidos:")
        ):
            return 0.0
    return 1.0


def r4_sequence_order(trial: TrialRecord) -> float | None:
    """Solo aplica a goals `sequence`. 0.0 si `goal_reason` señala fallo de
    orden (string exacto que produce `mia_world.goals._check_sequence`)."""
    if trial.goal.get("type") != "sequence":
        return None
    if trial.goal_achieved:
        return 1.0
    reason = trial.goal_reason or ""
    return 0.0 if "orden" in reason else 1.0


def r5_efficiency(trial: TrialRecord) -> float | None:
    """Reusa la métrica cuantitativa de eficiencia — se etiqueta
    explícitamente como derivada, no doble conteo escondido: responde una
    pregunta distinta ('¿fue razonablemente eficiente?')."""
    return efficiency_ratio(trial)


def score_trial(trial: TrialRecord) -> dict[str, float | None]:
    return {
        "R1_look_first": r1_look_first(trial),
        "R2_no_loop": r2_no_loop(trial),
        "R3_no_hallucination": r3_no_hallucination(trial),
        "R4_sequence_order": r4_sequence_order(trial),
        "R5_efficiency": r5_efficiency(trial),
    }


def rubric_score(trial: TrialRecord) -> float | None:
    """Media de los criterios no-None. None si ninguno aplica (trial crasheado
    sin ningún step ejecutado)."""
    values = [v for v in score_trial(trial).values() if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_rubric(trials: list[TrialRecord]) -> dict[str, object]:
    """rubric_score medio + pass-rate de cada criterio por separado (media
    de ese criterio sobre los trials donde es aplicable). El desglose por
    criterio es más defendible oralmente que un solo número compuesto:
    'R2 falla en 35% de los trials extreme' dice más que un promedio."""
    per_criterion: dict[str, list[float]] = defaultdict(list)
    scores: list[float] = []
    for trial in trials:
        s = score_trial(trial)
        for k, v in s.items():
            if v is not None:
                per_criterion[k].append(v)
        rs = rubric_score(trial)
        if rs is not None:
            scores.append(rs)

    return {
        "rubric_score_mean": sum(scores) / len(scores) if scores else None,
        "criteria_pass_rate": {
            k: (sum(vs) / len(vs) if vs else None) for k, vs in per_criterion.items()
        },
        "criteria_applicable_n": {k: len(vs) for k, vs in per_criterion.items()},
        "n_trials": len(trials),
    }
