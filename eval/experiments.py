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
- **C** (stretch) — tools de M1 reales vs. no-op. H1: tools irrelevantes
  en el prompt aumentan `hallucinated_tool_or_args` o bajan éxito.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mia_world import load_scenario  # noqa: E402

from eval.prompts import ESCAPE_ROOM_SYSTEM_PROMPT  # noqa: E402
from eval.report import build_summary  # noqa: E402
from eval.runner import TrialRecord, run_suite  # noqa: E402
from eval.scenario_meta import LONG_SOLUTION_SCENARIOS, MULTI_ROOM_SCENARIOS  # noqa: E402

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
            "El default de 10 round-trips es insuficiente en soluciones "
            "largas porque el modelo no batchea varios tool_calls por respuesta."
        ),
    },
}


def _scenario_paths_for_ids(scenario_ids: tuple[str, ...], scenarios_dir: Path) -> list[Path]:
    paths = []
    for p in sorted(scenarios_dir.glob("*.json")):
        if load_scenario(p).id in scenario_ids:
            paths.append(p)
    return paths


def _fmt_pct(x: float | None) -> str:
    return f"{x * 100:.0f}%" if x is not None else "—"


def _fmt_ratio(x: float | None) -> str:
    return f"{x:.2f}" if x is not None else "—"


def _write_comparison_report(
    name: str, param_label: str, arm_summaries: dict[str, dict], hypothesis: str, out_dir: Path
) -> Path:
    lines = [
        f"# Experimento {name} — comparación de arms ({param_label})",
        "",
        f"**Hipótesis:** {hypothesis}",
        "",
        "| arm | n | success (micro) | success (macro) | eficiencia media | top error |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm_value, summary in arm_summaries.items():
        counts = summary["error_breakdown_overall"].get("counts", {})
        top_error = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "—"
        lines.append(
            f"| {arm_value} | {summary['n_trials_total']} | "
            f"{_fmt_pct(summary['success_rate_micro'])} | "
            f"{_fmt_pct(summary['success_rate_macro'])} | "
            f"{_fmt_ratio(summary['mean_efficiency_overall'])} | {top_error} |"
        )
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

    out_path = out_dir / f"experiment_{name}_comparison.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    name: str,
    *,
    provider: str,
    trials: int,
    scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    ollama_model: str | None = None,
    ollama_host: str | None = None,
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
    if not scenario_paths:
        raise SystemExit(f"Experimento {name}: no se encontraron escenarios en {scenarios_dir}.")

    arm_summaries: dict[str, dict] = {}
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
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            force=force,
        )
        arm_summaries[str(value)] = build_summary(records)

    report_path = _write_comparison_report(
        f"{name}{report_suffix}", spec["param"], arm_summaries, spec["hypothesis"], out_dir  # type: ignore[arg-type]
    )
    print(f"Comparación escrita en {report_path}")
    return arm_summaries


def run_experiment_c(
    *,
    provider: str,
    trials: int,
    scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    ollama_model: str | None = None,
    ollama_host: str | None = None,
    force: bool = False,
) -> dict[str, dict]:
    """Experimento C (stretch): tools de M1 reales vs. no-op, sobre los 8
    escenarios. Toggle de `harness_options`, no de `framework_config` —
    manejado aparte de `run_experiment` porque cambia una dimensión distinta."""
    scenario_paths = sorted(scenarios_dir.glob("*.json"))
    arm_summaries: dict[str, dict] = {}
    for label, noop in (("real", False), ("noop", True)):
        print(f"--- Experimento C: arm m1_tools={label} ---")
        framework_config = {"system_prompt": ESCAPE_ROOM_SYSTEM_PROMPT}
        records = run_suite(
            scenario_paths,
            provider=provider,
            framework_config=framework_config,
            trials_per_scenario=trials,
            out_dir=out_dir,
            noop_m1_tools=noop,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            force=force,
        )
        arm_summaries[label] = build_summary(records)

    report_path = _write_comparison_report(
        "C",
        "m1_tools (real vs. noop)",
        arm_summaries,
        "Tools de M1 irrelevantes en el prompt aumentan hallucinated_tool_or_args o bajan éxito.",
        out_dir,
    )
    print(f"Comparación escrita en {report_path}")
    return arm_summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.experiments")
    parser.add_argument("--which", choices=["A", "B", "C"], required=True)
    parser.add_argument("--provider", choices=["auto", "ollama", "bedrock"], default="ollama")
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--ollama-host", default=None)
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
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        force=args.force,
    )
    if args.which == "C":
        run_experiment_c(**kwargs)
    else:
        run_experiment(args.which, **kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
