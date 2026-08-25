"""Experimentos A/B/C: aíslan qué partes del framework importan para este dominio.

Cada experimento corre varios *arms* (una config distinta) sobre el mismo
subconjunto de escenarios, vía `runner.run_suite` (mismo mecanismo de
resume/persistencia que la corrida baseline), y escribe una tabla
comparativa en `eval/results/experiment_<X>_comparison.md`.

- **A** — sensibilidad a `max_history_messages` en escenarios multi-sala.
  H1: ventana chica degrada éxito/eficiencia (el agente "olvida" el mapa
  o qué ítems ya tiene).
- **B** — sensibilidad a `max_iterations` en escenarios de solución larga.
  H1: el default (10) es insuficiente porque el modelo no batchea varios
  tool_calls por respuesta.
- **C** — tools de M1 visibles vs. ausentes. H1: exponer tools irrelevantes
  en el prompt aumenta elecciones de tool equivocadas o baja el éxito.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mia_world import load_scenario  # noqa: E402

from eval.prompts import ESCAPE_ROOM_SYSTEM_PROMPT  # noqa: E402
from eval.providers import PROVIDERS, model_slug  # noqa: E402
from eval.report import build_summary  # noqa: E402
from eval.runner import TrialRecord, run_suite  # noqa: E402
from eval.scenario_meta import (  # noqa: E402
    LONG_SOLUTION_SCENARIOS,
    MULTI_ROOM_SCENARIOS,
    OPTIMAL_TOOL_CALLS,
)

DEFAULT_SCENARIOS_DIR = _REPO_ROOT / "scenarios"
DEFAULT_OUT_DIR = _REPO_ROOT / "eval" / "results"

EXPERIMENTS: dict[str, dict[str, object]] = {
    "A": {
        "param": "max_history_messages",
        "arms": [8, 16, 50],
        "control": 50,
        "scenario_ids": MULTI_ROOM_SCENARIOS,
        "hypothesis": (
            "Ventana chica degrada éxito/eficiencia en multi-sala porque el "
            "agente olvida el mapa o qué ítems ya tiene."
        ),
    },
    "B": {
        "param": "max_iterations",
        "arms": [10, 15, 25],
        "control": 10,
        "scenario_ids": LONG_SOLUTION_SCENARIOS,
        "hypothesis": (
            "El default de 10 rondas LLM es insuficiente en soluciones largas "
            "porque los modelos observados suelen pedir una sola acción por ronda."
        ),
    },
}


def _scenario_paths_for_ids(scenario_ids: tuple[str, ...], scenarios_dir: Path) -> list[Path]:
    paths: list[Path] = []
    found: set[str] = set()
    for p in sorted(scenarios_dir.glob("*.json")):
        scenario_id = load_scenario(p).id
        if scenario_id in scenario_ids:
            paths.append(p)
            found.add(scenario_id)
    missing = set(scenario_ids) - found
    if missing:
        raise SystemExit(
            f"Faltan escenarios requeridos en {scenarios_dir}: {', '.join(sorted(missing))}."
        )
    return paths


def _fmt_pct(x: float | None) -> str:
    return f"{x * 100:.0f}%" if x is not None else "—"


def _fmt_ratio(x: float | None) -> str:
    return f"{x:.2f}" if x is not None else "—"


def _write_comparison_report(
    name: str,
    param_label: str,
    arm_summaries: dict[str, dict],
    hypothesis: str,
    out_dir: Path,
    *,
    provider: str,
    model: str | None,
) -> Path:
    has_m1_call_metrics = any("m1_tool_calls" in summary for summary in arm_summaries.values())
    table_header = "| arm | n | success (micro) | success (macro) | eficiencia media | top error"
    table_separator = "|---|---:|---:|---:|---:|---"
    if has_m1_call_metrics:
        table_header += " | llamadas M1 | trials con M1"
        table_separator += "|---:|---:"
    table_header += " |"
    table_separator += "|"
    lines = [
        f"# Experimento {name} — comparación de arms ({param_label})",
        "",
        f"Proveedor: **{provider}** · modelo: **{model or '(default del proveedor)'}**",
        "",
        f"**Hipótesis:** {hypothesis}",
        "",
        table_header,
        table_separator,
    ]
    for arm_value, summary in arm_summaries.items():
        counts = summary["error_breakdown_overall"].get("counts", {})
        top_error = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "—"
        row = (
            f"| {arm_value} | {summary['n_trials_total']} | "
            f"{_fmt_pct(summary['success_rate_micro'])} | "
            f"{_fmt_pct(summary['success_rate_macro'])} | "
            f"{_fmt_ratio(summary['mean_efficiency_overall'])} | {top_error}"
        )
        if has_m1_call_metrics:
            row += (
                f" | {summary.get('m1_tool_calls', 0)}"
                f" | {summary.get('trials_with_m1_tool_calls', 0)}"
            )
        lines.append(row + " |")
    lines += ["", "## Desglose de errores por arm", ""]
    for arm_value, summary in arm_summaries.items():
        lines.append(f"### arm = {arm_value}")
        counts = summary["error_breakdown_overall"].get("counts", {})
        if not counts:
            lines.append("(sin fallos)")
        else:
            for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"- {cat}: {n}")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    run_slug = f"{provider}_{model_slug(model)}"
    out_path = out_dir / f"experiment_{name}_{run_slug}_comparison.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "experiment": name,
                "parameter": param_label,
                "provider": provider,
                "model": model,
                "hypothesis": hypothesis,
                "arms": arm_summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


_M1_TOOL_NAMES = {"calculator", "file_reader", "word_counter"}


def _m1_tool_call_stats(records: list[TrialRecord]) -> tuple[int, int]:
    total_calls = 0
    trials_with_calls = 0
    for record in records:
        steps = (record.agent_result or {}).get("steps", [])
        count = sum(1 for step in steps if step.get("tool_name") in _M1_TOOL_NAMES)
        total_calls += count
        trials_with_calls += int(count > 0)
    return total_calls, trials_with_calls


def run_experiment(
    name: str,
    *,
    provider: str,
    trials: int,
    scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    model: str | None = None,
    host: str | None = None,
    force: bool = False,
    extra_framework_config: dict[str, object] | None = None,
    report_suffix: str = "",
) -> dict[str, dict]:
    """Corre todos los arms de un experimento (A o B) y escribe la tabla
    comparativa. Devuelve {arm_value: summary_dict}.

    `extra_framework_config` fija parámetros adicionales iguales en todos
    los arms (p. ej. `{"max_iterations": 25}` para una variante de control
    de A que aísla el efecto de la ventana una vez que se descubrió que el
    default de 10 iteraciones ya domina el resultado — ver `report_suffix`
    para no pisar el archivo de comparación del experimento original)."""
    spec = EXPERIMENTS[name]
    scenario_paths = _scenario_paths_for_ids(spec["scenario_ids"], scenarios_dir)  # type: ignore[arg-type]
    if extra_framework_config and spec["param"] in extra_framework_config:
        raise ValueError(
            f"extra_framework_config no puede sobrescribir el parámetro del arm {spec['param']!r}."
        )

    arm_summaries: dict[str, dict] = {}
    effective_model = model
    for value in spec["arms"]:  # type: ignore[union-attr]
        print(f"--- Experimento {name}{report_suffix}: arm {spec['param']}={value} ---")
        framework_config = {
            "system_prompt": ESCAPE_ROOM_SYSTEM_PROMPT,
            spec["param"]: value,
            **(extra_framework_config or {}),
        }
        records: list[TrialRecord] = run_suite(
            scenario_paths,
            provider=provider,
            framework_config=framework_config,
            trials_per_scenario=trials,
            out_dir=out_dir,
            model=model,
            host=host,
            force=force,
        )
        if records:
            effective_model = records[0].model
        arm_summaries[str(value)] = build_summary(records)

    report_path = _write_comparison_report(
        f"{name}{report_suffix}",
        spec["param"],
        arm_summaries,
        spec["hypothesis"],
        out_dir,
        provider=provider,
        model=effective_model,
    )
    print(f"Comparación escrita en {report_path}")
    return arm_summaries


def run_experiment_c(
    *,
    provider: str,
    trials: int,
    scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    model: str | None = None,
    host: str | None = None,
    force: bool = False,
) -> dict[str, dict]:
    """Experimento C: tools de M1 visibles vs. ausentes, sobre los 8 escenarios.

    El arm ``visible`` reutiliza la configuración baseline. En ``absent``
    `build_agent` no registra calculator/file_reader/word_counter, de modo que
    tampoco aparecen en la lista de tools que recibe el LLM. Esto sí aísla la
    hipótesis de distracción; reemplazar implementaciones por no-ops no lo hacía
    porque el modelo veía exactamente los mismos schemas en ambos arms.
    """
    expected_ids = tuple(OPTIMAL_TOOL_CALLS)
    scenario_paths = _scenario_paths_for_ids(expected_ids, scenarios_dir)
    arm_summaries: dict[str, dict] = {}
    effective_model = model
    for label, register_m1_tools in (("visible", True), ("absent", False)):
        print(f"--- Experimento C: arm m1_tools={label} ---")
        framework_config = {"system_prompt": ESCAPE_ROOM_SYSTEM_PROMPT}
        # Omitir la clave en el arm default conserva el hash histórico y
        # permite reutilizar la corrida baseline ya válida.
        if not register_m1_tools:
            framework_config["register_m1_tools"] = False
        records = run_suite(
            scenario_paths,
            provider=provider,
            framework_config=framework_config,
            trials_per_scenario=trials,
            out_dir=out_dir,
            model=model,
            host=host,
            force=force,
        )
        if records:
            effective_model = records[0].model
        summary = build_summary(records)
        m1_calls, trials_with_m1_calls = _m1_tool_call_stats(records)
        summary["m1_tool_calls"] = m1_calls
        summary["trials_with_m1_tool_calls"] = trials_with_m1_calls
        arm_summaries[label] = summary

    report_path = _write_comparison_report(
        "C",
        "m1_tools (visibles vs. ausentes)",
        arm_summaries,
        "Exponer tools de M1 irrelevantes aumenta elecciones equivocadas o baja el éxito.",
        out_dir,
        provider=provider,
        model=effective_model,
    )
    print(f"Comparación escrita en {report_path}")
    return arm_summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.experiments")
    parser.add_argument("--which", choices=["A", "B", "C"], required=True)
    parser.add_argument("--provider", choices=list(PROVIDERS), default="ollama")
    parser.add_argument("--model", default=None, help="Modelo (default del proveedor si se omite).")
    parser.add_argument("--host", default=None, help="Override del host (solo Ollama).")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--scenarios-dir", default=str(DEFAULT_SCENARIOS_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    kwargs = dict(
        provider=args.provider,
        trials=args.trials,
        scenarios_dir=Path(args.scenarios_dir),
        out_dir=Path(args.out_dir),
        model=args.model,
        host=args.host,
        force=args.force,
    )
    if args.which == "C":
        run_experiment_c(**kwargs)
    else:
        run_experiment(args.which, **kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
