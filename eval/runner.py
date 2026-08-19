"""Ejecución de un trial (agente x escenario) y de una corrida completa.

Replica el patrón de integración de `mia_world/cli.py::_cmd_run` (build_agent
→ registrar tools del mundo → run → check_goal), pero con dos invariantes
extra que la CLI fija no necesita (corre un único escenario por proceso) y
que acá son críticas porque un mismo proceso corre decenas de trials:

  1. `World` fresco por trial: `Scenario.initial_world` es mutable: `load_scenario`
     se llama de nuevo en cada trial, nunca se reusa un `Scenario` cacheado.
  2. `MyAgent` fresco por trial: `_history` (M2) persiste entre `run()` de la
     misma instancia — para trials independientes hace falta `build_agent`
     nuevo en cada trial, no una vez afuera del loop.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mia_agents._env import load_env_files
from mia_agents.llm_client import BedrockProvider, LLMClient, OllamaProvider
from mia_world import check_goal, load_scenario, make_world_tools

from eval.scenario_meta import OPTIMAL_TOOL_CALLS


def build_llm_client(
    provider: str,
    *,
    ollama_model: str | None = None,
    ollama_host: str | None = None,
) -> LLMClient:
    """provider en {"auto", "ollama", "bedrock"}.

    "auto" delega en `LLMClient.from_env()` (mismo criterio que usa la
    cátedra: `OLLAMA_HOST` si está, si no `BEDROCK_MODEL_ID`). "ollama" /
    "bedrock" fuerzan el proveedor sin depender de qué haya en `.env` —
    necesario para correr el mismo sweep contra los dos proveedores sin
    reeditar el archivo entre corridas.
    """
    load_env_files()
    if provider == "auto":
        return LLMClient.from_env()
    if provider == "ollama":
        return LLMClient(OllamaProvider(model=ollama_model, host=ollama_host))
    if provider == "bedrock":
        return LLMClient(BedrockProvider())
    raise ValueError(f"provider desconocido: {provider!r} (esperado: auto|ollama|bedrock)")


@dataclass
class TrialRecord:
    trial_id: str
    scenario_id: str
    difficulty: str
    provider: str
    module: str
    framework_config: dict[str, Any]
    harness_options: dict[str, Any]
    trial_index: int
    timestamp: str
    wall_time_s: float
    crashed: bool
    crash_message: str | None
    agent_result: dict[str, Any] | None
    tool_call_count: int | None
    optimal_calls: int | None
    goal: dict[str, Any]
    goal_achieved: bool | None
    goal_reason: str | None
    event_log: list[str] = field(default_factory=list)


def _config_hash(framework_config: dict[str, Any], harness_options: dict[str, Any]) -> str:
    payload = json.dumps(
        {"framework_config": framework_config, "harness_options": harness_options},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def trial_output_path(
    out_dir: Path, scenario_id: str, provider: str, config_hash: str, trial_index: int
) -> Path:
    return out_dir / "raw" / scenario_id / f"{provider}_{config_hash}_t{trial_index}.json"


def run_trial(
    scenario_path: Path,
    *,
    provider: str,
    framework_config: dict[str, Any],
    trial_index: int,
    module: str = "student_framework",
    noop_m1_tools: bool = False,
    ollama_model: str | None = None,
    ollama_host: str | None = None,
    llm_client: LLMClient | None = None,
) -> TrialRecord:
    """Un trial end-to-end. Nunca lanza: cualquier fallo no absorbido por
    el agente (proveedor caído tras agotar los reintentos de M2, bug de
    integración) se captura acá como `crashed=True` para que un sweep
    largo no se caiga entero por un trial.

    `llm_client`, si se pasa, se usa directamente en vez de construirlo vía
    `build_llm_client(provider, ...)` — es lo que permite inyectar un
    `MockLLMClient` en `tests/test_eval_harness.py` sin tocar un LLM real;
    `provider` sigue guardándose en el `TrialRecord` como etiqueta.
    """
    scenario = load_scenario(scenario_path)
    world = scenario.initial_world
    harness_options = {"noop_m1_tools": noop_m1_tools}

    start = time.monotonic()
    crashed = False
    crash_message: str | None = None
    result_dict: dict[str, Any] | None = None
    try:
        llm = llm_client if llm_client is not None else build_llm_client(
            provider, ollama_model=ollama_model, ollama_host=ollama_host
        )
        config: dict[str, Any] = {**framework_config, "llm_client": llm}
        agent_module = importlib.import_module(module)
        agent = agent_module.build_agent(config)
        for fn, schema in make_world_tools(world):
            agent.register_tool(fn, schema)
        if noop_m1_tools:
            from eval.noop_tools import make_noop_m1_tools

            for fn, schema in make_noop_m1_tools():
                agent.register_tool(fn, schema)
        result = agent.run(scenario.user_message)
        result_dict = asdict(result)
    except Exception as exc:  # noqa: BLE001 — un trial nunca debe tumbar el sweep
        crashed = True
        crash_message = f"{type(exc).__name__}: {exc}"
    wall_time_s = time.monotonic() - start

    # check_goal se evalúa siempre, incluso si el agente crasheó: las tools
    # mutan `world` in-place, así que un crash a mitad de camino puede haber
    # dejado el mundo parcialmente resuelto.
    achieved, reason = check_goal(world, scenario.goal)

    config_hash = _config_hash(framework_config, harness_options)
    trial_id = f"{scenario.id}__{provider}__{config_hash}__t{trial_index}"
    tool_call_count = len(result_dict["steps"]) if result_dict is not None else None

    return TrialRecord(
        trial_id=trial_id,
        scenario_id=scenario.id,
        difficulty=scenario.difficulty,
        provider=provider,
        module=module,
        framework_config=framework_config,
        harness_options=harness_options,
        trial_index=trial_index,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_time_s=wall_time_s,
        crashed=crashed,
        crash_message=crash_message,
        agent_result=result_dict,
        tool_call_count=tool_call_count,
        optimal_calls=OPTIMAL_TOOL_CALLS.get(scenario.id),
        goal=scenario.goal,
        goal_achieved=achieved,
        goal_reason=reason,
        event_log=list(world.event_log),
    )


def load_trial_record(path: Path) -> TrialRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrialRecord(**data)


def load_all_trial_records(raw_dir: Path) -> list[TrialRecord]:
    """Relee todos los TrialRecord persistidos bajo `<raw_dir>/*/*.json`,
    sin llamar al LLM — usado por `eval/report.py` para recalcular métricas
    tras un ajuste de umbral sin re-gastar una corrida cara de Bedrock."""
    return [load_trial_record(p) for p in sorted(raw_dir.glob("*/*.json"))]


def run_suite(
    scenario_paths: list[Path],
    *,
    provider: str,
    framework_config: dict[str, Any],
    trials_per_scenario: int,
    out_dir: Path,
    module: str = "student_framework",
    noop_m1_tools: bool = False,
    ollama_model: str | None = None,
    ollama_host: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[TrialRecord]:
    """Corre escenario x trial_index. Resume gratis: si el JSON de un trial
    ya existe en disco, lo relee en vez de volver a llamar al LLM (salvo
    `force=True`). Persiste cada `TrialRecord` a disco INMEDIATAMENTE
    después de calcularlo, no al final, para no perder una corrida cara de
    Bedrock si el proceso muere a mitad de camino."""
    harness_options = {"noop_m1_tools": noop_m1_tools}
    config_hash = _config_hash(framework_config, harness_options)
    records: list[TrialRecord] = []

    for scenario_path in scenario_paths:
        peek = load_scenario(scenario_path)  # solo para id/dificultad (path, dry-run)
        for trial_index in range(trials_per_scenario):
            out_path = trial_output_path(out_dir, peek.id, provider, config_hash, trial_index)
            if dry_run:
                status = "existe" if out_path.exists() else "a correr"
                print(f"[dry-run] {out_path}  ({status})")
                continue
            if out_path.exists() and not force:
                records.append(load_trial_record(out_path))
                continue
            record = run_trial(
                scenario_path,
                provider=provider,
                framework_config=framework_config,
                trial_index=trial_index,
                module=module,
                noop_m1_tools=noop_m1_tools,
                ollama_model=ollama_model,
                ollama_host=ollama_host,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            records.append(record)
            status = "OK" if record.goal_achieved else ("CRASH" if record.crashed else "fail")
            print(
                f"  [{status:>5}] {record.scenario_id} trial {trial_index} "
                f"({record.tool_call_count} calls, {record.wall_time_s:.1f}s)"
            )

    return records
