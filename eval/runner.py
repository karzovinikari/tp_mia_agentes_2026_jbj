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
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mia_agents._env import load_env_files
from mia_agents.llm_client import BedrockProvider, LLMClient, OllamaProvider
from mia_world import check_goal, load_scenario, make_world_tools

from eval.providers import model_slug, resolve_model
from eval.scenario_meta import OPTIMAL_TOOL_CALLS
from student_framework.kimi_provider import KimiProvider
from student_framework.openai_provider import OpenAIProvider


def build_llm_client(
    provider: str,
    *,
    model: str | None = None,
    host: str | None = None,
) -> LLMClient:
    """provider en {"auto", "ollama", "bedrock", "kimi", "openai"}.

    "auto" delega en `LLMClient.from_env()` (mismo criterio que usa la
    cátedra: `OLLAMA_HOST` si está, si no `BEDROCK_MODEL_ID`). Los demás
    fuerzan el proveedor sin depender de qué haya en `.env` — necesario
    para correr el mismo sweep contra varios proveedores sin reeditar el
    archivo entre corridas.

    `model` es genérico a propósito (antes era `ollama_model`, que solo
    servía para un proveedor): así un único `--model` sirve para los
    cuatro. Kimi y OpenAI no pasan por `LLMClient.from_env()` (ese archivo
    es fijo y solo conoce Bedrock/Ollama): viven en `student_framework/` y
    satisfacen el mismo protocolo `LLMClient`.
    """
    load_env_files()
    if provider == "auto":
        return LLMClient.from_env()
    if provider == "ollama":
        return LLMClient(OllamaProvider(model=model, host=host))
    if provider == "bedrock":
        return LLMClient(BedrockProvider(model=model))
    if provider == "kimi":
        return LLMClient(KimiProvider(model=model))
    if provider == "openai":
        return LLMClient(OpenAIProvider(model=model))
    raise ValueError(
        f"provider desconocido: {provider!r} "
        f"(esperado: auto|ollama|bedrock|kimi|openai)"
    )


def describe_model(client: LLMClient, requested: str | None) -> str | None:
    """Modelo efectivo con el que quedó configurado el cliente.

    Todos los providers del proyecto (Ollama, Bedrock, Kimi, OpenAI)
    guardan el modelo resuelto en `_model`. Se lee de ahí para registrar el
    valor real, no el pedido — importa cuando el default lo pone el
    provider (Kimi) o el entorno (Bedrock).
    """
    provider_obj = getattr(client, "_provider", None)
    return getattr(provider_obj, "_model", None) or requested


def resolve_effective_model(
    provider: str, model: str | None, *, host: str | None = None
) -> str | None:
    """Modelo efectivo de un sweep, resuelto UNA vez antes de correr.

    Construye el cliente solo para preguntarle qué modelo quedó (los
    defaults de Kimi/Bedrock viven en el provider, no en `eval/`). Si no se
    puede construir (sin credenciales, provider de test, `--dry-run`), cae
    al valor pedido en vez de romper: resolver el nombre del modelo no debe
    ser condición para listar una matriz de trials.
    """
    requested = resolve_model(provider, model)
    try:
        return describe_model(build_llm_client(provider, model=requested, host=host), requested)
    except Exception:  # noqa: BLE001 — best-effort: es solo para etiquetar
        return requested


def resolve_effective_host(provider: str, host: str | None) -> str | None:
    """Endpoint efectivo cuando la corrida usa Ollama.

    Se registra y entra en el hash porque dos servidores pueden servir pesos
    distintos bajo el mismo nombre de modelo. Para otros proveedores no hay
    override de host en este harness.
    """
    if provider not in {"ollama", "auto"}:
        return None
    load_env_files()
    configured = host or os.environ.get("OLLAMA_HOST")
    if provider == "ollama":
        return configured or "http://localhost:11434"
    return configured  # auto solo elige Ollama si OLLAMA_HOST está definido


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
    # Modelo efectivo del trial. `None` solo en trials viejos (anteriores a
    # que se registrara) o con un cliente inyectado en tests. Default al
    # final para no romper la relectura de records ya persistidos.
    model: str | None = None
    # Endpoint efectivo de Ollama. Los records viejos no lo tenían, por eso
    # queda como campo opcional al final.
    host: str | None = None
    # Permite separar una caída del proveedor de un problema al cargar o
    # evaluar el escenario. En records históricos queda en None.
    crash_stage: str | None = None
    event_log: list[str] = field(default_factory=list)


def _config_hash(
    framework_config: dict[str, Any],
    harness_options: dict[str, Any],
    provider: str = "",
    model: str | None = None,
    *,
    module: str = "student_framework",
    host: str | None = None,
) -> str:
    """Hash que identifica una configuración de corrida.

    Incluye proveedor y modelo, no solo la config del framework: sin eso,
    dos corridas del mismo escenario con modelos distintos colisionan en el
    mismo archivo y la segunda pisa a la primera (fue exactamente lo que
    pasó y dejó 143 trials sin atribuir).
    """
    payload_data: dict[str, Any] = {
        "framework_config": framework_config,
        "harness_options": harness_options,
        "provider": provider,
        "model": model,
    }
    # Compatibilidad: no alteramos el hash histórico de la configuración por
    # defecto, pero cualquier módulo u host explícito distinto sí obtiene su
    # propio namespace de resultados.
    if module != "student_framework":
        payload_data["module"] = module
    if host is not None:
        payload_data["host"] = host
    payload = json.dumps(payload_data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def trial_output_path(
    out_dir: Path,
    scenario_id: str,
    provider: str,
    config_hash: str,
    trial_index: int,
    model: str | None = None,
) -> Path:
    return (
        out_dir
        / "raw"
        / scenario_id
        / f"{provider}_{model_slug(model)}_{config_hash}_t{trial_index}.json"
    )


def run_trial(
    scenario_path: Path,
    *,
    provider: str,
    framework_config: dict[str, Any],
    trial_index: int,
    module: str = "student_framework",
    noop_m1_tools: bool = False,
    model: str | None = None,
    host: str | None = None,
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
    harness_options = {"noop_m1_tools": noop_m1_tools}
    effective_model = resolve_model(provider, model)
    effective_host = resolve_effective_host(provider, host) if llm_client is None else host

    start = time.monotonic()
    scenario = None
    world = None
    crashed = False
    crash_message: str | None = None
    crash_stage: str | None = None
    result_dict: dict[str, Any] | None = None
    try:
        crash_stage = "scenario_load"
        scenario = load_scenario(scenario_path)
        world = scenario.initial_world

        crash_stage = "agent_execution"
        if llm_client is not None:
            llm = llm_client
        else:
            llm = build_llm_client(provider, model=effective_model, host=effective_host)
            # Se registra el modelo REAL del cliente, no el pedido: para
            # Kimi/Bedrock el default lo pone el provider o el entorno.
            effective_model = describe_model(llm, effective_model)
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
    else:
        crash_stage = None
    wall_time_s = time.monotonic() - start

    # check_goal se evalúa siempre, incluso si el agente crasheó: las tools
    # mutan `world` in-place, así que un crash a mitad de camino puede haber
    # dejado el mundo parcialmente resuelto.
    if scenario is not None and world is not None:
        try:
            achieved, reason = check_goal(world, scenario.goal)
        except Exception as exc:  # noqa: BLE001 — mismo contrato: el trial no lanza
            achieved, reason = False, "No se pudo evaluar la meta del escenario."
            crashed = True
            crash_stage = "goal_check"
            detail = f"{type(exc).__name__}: {exc}"
            crash_message = f"{crash_message}; {detail}" if crash_message else detail
    else:
        achieved = False
        reason = "No se pudo cargar el escenario."

    config_hash = _config_hash(
        framework_config,
        harness_options,
        provider,
        effective_model,
        module=module,
        host=effective_host,
    )
    scenario_id = scenario.id if scenario is not None else scenario_path.stem
    trial_id = (
        f"{scenario_id}__{provider}__{model_slug(effective_model)}"
        f"__{config_hash}__t{trial_index}"
    )
    tool_call_count = len(result_dict["steps"]) if result_dict is not None else None

    return TrialRecord(
        trial_id=trial_id,
        scenario_id=scenario_id,
        difficulty=scenario.difficulty if scenario is not None else "unknown",
        provider=provider,
        model=effective_model,
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
        optimal_calls=OPTIMAL_TOOL_CALLS.get(scenario_id),
        goal=scenario.goal if scenario is not None else {},
        goal_achieved=achieved,
        goal_reason=reason,
        host=effective_host,
        crash_stage=crash_stage,
        event_log=list(world.event_log) if world is not None else [],
    )


def load_trial_record(path: Path) -> TrialRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrialRecord(**data)


def load_all_trial_records(raw_dir: Path) -> list[TrialRecord]:
    """Relee todos los TrialRecord persistidos bajo `<raw_dir>/*/*.json`,
    sin llamar al LLM — usado por `eval/report.py` para recalcular métricas
    tras un ajuste de umbral sin re-gastar una corrida cara de Bedrock."""
    return [load_trial_record(p) for p in sorted(raw_dir.glob("*/*.json"))]


def _record_matches_request(
    record: TrialRecord,
    *,
    scenario_id: str,
    provider: str,
    model: str | None,
    module: str,
    host: str | None,
    framework_config: dict[str, Any],
    harness_options: dict[str, Any],
    trial_index: int,
) -> bool:
    """Defensa adicional del resume frente a archivos movidos o hashes viejos."""
    return (
        record.scenario_id == scenario_id
        and record.provider == provider
        and record.model == model
        and record.module == module
        and record.host == host
        and record.framework_config == framework_config
        and record.harness_options == harness_options
        and record.trial_index == trial_index
    )


def run_suite(
    scenario_paths: list[Path],
    *,
    provider: str,
    framework_config: dict[str, Any],
    trials_per_scenario: int,
    out_dir: Path,
    module: str = "student_framework",
    noop_m1_tools: bool = False,
    model: str | None = None,
    host: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[TrialRecord]:
    """Corre escenario x trial_index. Resume gratis: si el JSON de un trial
    ya existe en disco, lo relee en vez de volver a llamar al LLM (salvo
    `force=True`). Persiste cada `TrialRecord` a disco INMEDIATAMENTE
    después de calcularlo, no al final, para no perder una corrida cara de
    Bedrock si el proceso muere a mitad de camino."""
    harness_options = {"noop_m1_tools": noop_m1_tools}
    # El modelo se resuelve UNA vez por sweep y se pasa explícito a cada
    # trial: así todos los trials del sweep usan el mismo modelo, queda
    # registrado, y el path de salida (que depende de él) es estable entre
    # una corrida y su resume.
    effective_model = resolve_effective_model(provider, model, host=host)
    effective_host = resolve_effective_host(provider, host)
    config_hash = _config_hash(
        framework_config,
        harness_options,
        provider,
        effective_model,
        module=module,
        host=effective_host,
    )
    records: list[TrialRecord] = []

    if not dry_run:
        print(f"  modelo: {effective_model or '(default del proveedor)'}")

    for scenario_path in scenario_paths:
        try:
            peek_id = load_scenario(scenario_path).id  # solo para nombrar el path/resume
        except Exception:  # noqa: BLE001 — run_trial persistirá el fallo con detalle
            peek_id = scenario_path.stem
        for trial_index in range(trials_per_scenario):
            out_path = trial_output_path(
                out_dir, peek_id, provider, config_hash, trial_index, effective_model
            )
            if dry_run:
                status = "existe" if out_path.exists() else "a correr"
                print(f"[dry-run] {out_path}  ({status})")
                continue
            if out_path.exists() and not force:
                cached = load_trial_record(out_path)
                if _record_matches_request(
                    cached,
                    scenario_id=peek_id,
                    provider=provider,
                    model=effective_model,
                    module=module,
                    host=effective_host,
                    framework_config=framework_config,
                    harness_options=harness_options,
                    trial_index=trial_index,
                ):
                    records.append(cached)
                    continue
                print(f"  [stale] {out_path} no coincide con la corrida pedida; se recalcula")
            record = run_trial(
                scenario_path,
                provider=provider,
                framework_config=framework_config,
                trial_index=trial_index,
                module=module,
                noop_m1_tools=noop_m1_tools,
                model=effective_model,
                host=effective_host,
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
