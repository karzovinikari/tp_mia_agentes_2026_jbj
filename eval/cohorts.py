"""Cohortes reproducibles para separar baseline y experimentos de M3.

Los JSON crudos comparten un mismo árbol porque se fueron generando en varias
etapas. Agregarlos todos juntos mezcla configuraciones distintas. Estas reglas
nombran los cortes que usa el informe y dejan explícito qué entra en cada uno.
"""

from __future__ import annotations

from collections.abc import Callable

from eval.runner import TrialRecord

EVALUATION_PROVIDERS = {"ollama", "bedrock", "openai"}


def _config_is(trial: TrialRecord, expected: dict[str, object]) -> bool:
    config_without_prompt = {
        key: value for key, value in trial.framework_config.items() if key != "system_prompt"
    }
    return config_without_prompt == expected


def _baseline(trial: TrialRecord) -> bool:
    return (
        trial.provider in EVALUATION_PROVIDERS
        and _config_is(trial, {})
        and not trial.harness_options.get("noop_m1_tools", False)
    )


def _budget_21(trial: TrialRecord) -> bool:
    return _config_is(trial, {"max_iterations": 21})


def _experiment_a(trial: TrialRecord) -> bool:
    return _config_is(
        trial,
        {"max_history_messages": trial.framework_config.get("max_history_messages")},
    ) and trial.framework_config.get("max_history_messages") in {8, 16, 50}


def _experiment_b(trial: TrialRecord) -> bool:
    return _config_is(
        trial,
        {"max_iterations": trial.framework_config.get("max_iterations")},
    ) and trial.framework_config.get("max_iterations") in {10, 15, 25}


def _experiment_c_absent(trial: TrialRecord) -> bool:
    return _config_is(trial, {"register_m1_tools": False})


def _experiment_c(trial: TrialRecord) -> bool:
    return _baseline(trial) or _experiment_c_absent(trial)


def _legacy_noop_c(trial: TrialRecord) -> bool:
    return _config_is(trial, {}) and trial.harness_options.get("noop_m1_tools", False)


def _analysis(trial: TrialRecord) -> bool:
    """Todas las corridas usadas como evidencia tras la auditoría.

    Excluye el piloto no-op de C y el único smoke test de Kimi. El baseline
    funciona también como arm visible de C y aparece una sola vez.
    """
    if trial.provider not in EVALUATION_PROVIDERS:
        return False
    return any(
        selector(trial)
        for selector in (_baseline, _budget_21, _experiment_a, _experiment_b, _experiment_c_absent)
    )


COHORTS: dict[str, Callable[[TrialRecord], bool]] = {
    "all": lambda _trial: True,
    "baseline": _baseline,
    "budget_21": _budget_21,
    "experiment_a": _experiment_a,
    "experiment_b": _experiment_b,
    "experiment_c": _experiment_c,
    "experiment_c_absent": _experiment_c_absent,
    "legacy_noop_c": _legacy_noop_c,
    "analysis": _analysis,
}


def select_cohort(trials: list[TrialRecord], name: str) -> list[TrialRecord]:
    try:
        selector = COHORTS[name]
    except KeyError as exc:
        raise ValueError(f"Cohorte desconocida: {name!r}") from exc
    return [trial for trial in trials if selector(trial)]
