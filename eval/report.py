"""Agrega TrialRecords en un resumen (JSON + Markdown).

Separado de `runner.py` a propósito: recalcular métricas/rúbrica/errores
no debería requerir volver a llamar al LLM. `python -m eval.report` relee
`eval/results/raw/` y regenera `summary.json`/`summary.md` — útil para
iterar sobre un bug de `metrics.py`/`rubric.py`/`errors.py` sin re-gastar
una corrida cara de Bedrock.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.cohorts import COHORTS, select_cohort  # noqa: E402
from eval.errors import error_breakdown, incident_breakdown  # noqa: E402
from eval.metrics import (  # noqa: E402
    mean_efficiency,
    mean_wall_time,
    success_rate,
    success_rate_macro,
)
from eval.providers import PROVIDERS  # noqa: E402
from eval.rubric import aggregate_rubric  # noqa: E402
from eval.runner import TrialRecord, load_all_trial_records  # noqa: E402


def _scenario_summary(trials: list[TrialRecord]) -> dict[str, object]:
    return {
        "difficulty": trials[0].difficulty if trials else None,
        "n_trials": len(trials),
        "success_rate": success_rate(trials),
        "mean_efficiency": mean_efficiency(trials),
        "mean_wall_time_s": mean_wall_time(trials),
        "rubric": aggregate_rubric(trials),
        "error_breakdown": error_breakdown(trials),
        "incident_breakdown": incident_breakdown(trials),
    }


def build_summary(
    trials: list[TrialRecord], *, selection: dict[str, object] | None = None
) -> dict[str, object]:
    by_scenario: dict[str, list[TrialRecord]] = defaultdict(list)
    for t in trials:
        by_scenario[t.scenario_id].append(t)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_trials_total": len(trials),
        "success_rate_micro": success_rate(trials),
        "success_rate_macro": success_rate_macro(trials),
        "mean_efficiency_overall": mean_efficiency(trials),
        "rubric_overall": aggregate_rubric(trials),
        "error_breakdown_overall": error_breakdown(trials),
        "incident_breakdown_overall": incident_breakdown(trials),
        "by_scenario": {sid: _scenario_summary(ts) for sid, ts in sorted(by_scenario.items())},
    }
    if selection is not None:
        summary["selection"] = selection
    return summary


def _fmt_pct(x: float | None) -> str:
    return f"{x * 100:.0f}%" if x is not None else "—"


def _fmt_ratio(x: float | None) -> str:
    return f"{x:.2f}" if x is not None else "—"


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Resumen de evaluación — M3",
        "",
        f"Generado: {summary['generated_at']}  ·  trials totales: {summary['n_trials_total']}",
    ]
    if "selection" in summary:
        lines += ["", f"Selección: `{json.dumps(summary['selection'], ensure_ascii=False)}`"]
    lines += [
        "",
        f"- **Success rate (micro)**: {_fmt_pct(summary['success_rate_micro'])}",
        f"- **Success rate (macro, promedio por escenario)**: {_fmt_pct(summary['success_rate_macro'])}",
        f"- **Eficiencia media** (óptimo/real, entre éxitos): {_fmt_ratio(summary['mean_efficiency_overall'])}",
        f"- **Rúbrica media**: {_fmt_ratio(summary['rubric_overall'].get('rubric_score_mean'))}",
        "",
        "## Por escenario",
        "",
        "| escenario | dificultad | n | success | eficiencia | rúbrica |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for sid, row in summary["by_scenario"].items():
        rubric_mean = row["rubric"].get("rubric_score_mean")
        lines.append(
            f"| {sid} | {row['difficulty']} | {row['n_trials']} | "
            f"{_fmt_pct(row['success_rate'])} | {_fmt_ratio(row['mean_efficiency'])} | "
            f"{_fmt_ratio(rubric_mean)} |"
        )

    lines += ["", "## Desglose de errores (global)", ""]
    breakdown = summary["error_breakdown_overall"]
    counts = breakdown.get("counts", {})
    if not counts:
        lines.append("(sin fallos)")
    else:
        lines.append(f"Fallos totales: {breakdown['total_failures']} / {breakdown['total_trials']} trials")
        lines.append("")
        lines.append("| categoría | n | % de los fallos |")
        lines.append("|---|---:|---:|")
        for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            pct = breakdown["pct_of_failures"].get(cat)
            lines.append(f"| {cat} | {n} | {_fmt_pct(pct)} |")

    lines += ["", "## Rúbrica — pass-rate por criterio (global)", ""]
    pass_rates = summary["rubric_overall"].get("criteria_pass_rate", {})
    applicable_n = summary["rubric_overall"].get("criteria_applicable_n", {})
    lines.append("| criterio | pass-rate | n aplicable |")
    lines.append("|---|---:|---:|")
    for crit, rate in pass_rates.items():
        lines.append(f"| {crit} | {_fmt_pct(rate)} | {applicable_n.get(crit, 0)} |")

    lines += ["", "## Incidencias observadas (no exclusivas)", ""]
    incidents = summary.get("incident_breakdown_overall", {})
    incident_counts = incidents.get("counts", {})
    if not incident_counts:
        lines.append("(sin incidencias adicionales)")
    else:
        lines.append("| incidencia | trials | % de trials |")
        lines.append("|---|---:|---:|")
        for category, count in sorted(incident_counts.items(), key=lambda kv: -kv[1]):
            rate = incidents.get("pct_of_trials", {}).get(category)
            lines.append(f"| {category} | {count} | {_fmt_pct(rate)} |")

    return "\n".join(lines) + "\n"


def write_summary(
    trials: list[TrialRecord],
    out_dir: Path,
    *,
    selection: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = build_summary(trials, selection=selection)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.report")
    parser.add_argument("--raw-dir", default=str(_REPO_ROOT / "eval" / "results" / "raw"))
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "eval" / "results"))
    parser.add_argument(
        "--cohort",
        choices=list(COHORTS),
        default="all",
        help="Corte reproducible de configuración (default: all).",
    )
    parser.add_argument(
        "--provider",
        choices=["all", *PROVIDERS],
        default="all",
        help="Filtrar trials por proveedor antes de agregar (default: todos).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filtrar además por modelo (p. ej. qwen2.5). Sin esto, se mezclan modelos del mismo proveedor.",
    )
    parser.add_argument("--module", default=None, help="Filtrar por módulo de agente.")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Filtrar por id de escenario (repetible).",
    )
    args = parser.parse_args(argv)

    trials = load_all_trial_records(Path(args.raw_dir))
    trials = select_cohort(trials, args.cohort)
    if args.provider != "all":
        trials = [t for t in trials if t.provider == args.provider]
    if args.model is not None:
        trials = [t for t in trials if t.model == args.model]
    if args.module is not None:
        trials = [t for t in trials if t.module == args.module]
    if args.scenario:
        trials = [t for t in trials if t.scenario_id in set(args.scenario)]

    if not trials:
        print(f"Sin trials en {args.raw_dir} (filtro provider={args.provider!r}).", file=sys.stderr)
        return 1

    selection = {
        "cohort": args.cohort,
        "provider": args.provider,
        "model": args.model,
        "module": args.module,
        "scenarios": args.scenario or "all",
    }
    summary = write_summary(trials, Path(args.out_dir), selection=selection)
    print(f"summary.json / summary.md escritos en {args.out_dir}")
    print(f"success_rate_micro={summary['success_rate_micro']}  n={summary['n_trials_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
