#!/usr/bin/env python3
"""CLI de evaluación: `python eval/run.py`.

Se invoca como script directo (no `-m`), así que `sys.path` no trae la
raíz del repo por defecto — sin el insert de abajo, `import mia_world`
fallaría. Tiene que ser la primera línea ejecutable, antes de cualquier
import de `mia_world`/`mia_agents`/`student_framework`/`eval.*`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mia_world import list_scenarios, load_scenario  # noqa: E402

from eval.prompts import SYSTEM_PROMPT_VARIANTS  # noqa: E402
from eval.report import write_summary  # noqa: E402
from eval.runner import run_suite  # noqa: E402

DEFAULT_SCENARIOS_DIR = _REPO_ROOT / "scenarios"
DEFAULT_OUT_DIR = _REPO_ROOT / "eval" / "results"


def _scenario_paths_for_spec(spec: str, scenarios_dir: Path) -> list[Path]:
    """Resuelve un spec a 1+ paths de escenario.

    Acepta: "all" (los 8), un path a un .json, un id exacto ("color-locks"),
    o una dificultad ("medium"). A diferencia de `mia_world.cli._resolve_scenario`
    (que para una dificultad devuelve solo el primero), acá una dificultad
    devuelve TODOS sus escenarios — para un sweep de evaluación eso es lo
    útil (p. ej. "medium" = color-locks + apartment-keys).
    """
    if spec == "all":
        return sorted(scenarios_dir.glob("*.json"))
    path = Path(spec)
    if path.is_file():
        return [path]

    all_files = sorted(scenarios_dir.glob("*.json"))
    for p in all_files:
        if load_scenario(p).id == spec:
            return [p]
    by_difficulty = [p for p in all_files if load_scenario(p).difficulty == spec]
    if by_difficulty:
        return by_difficulty

    available = ", ".join(sc.id for sc in list_scenarios(scenarios_dir))
    raise SystemExit(
        f"No se encontró el escenario o dificultad {spec!r} en {scenarios_dir}. "
        f"Usá 'all', un id ({available}), un path .json, o una dificultad "
        f"(easy/medium/hard/extreme)."
    )


def _resolve_scenario_paths(specs: list[str], scenarios_dir: Path) -> list[Path]:
    specs = specs or ["all"]
    seen: dict[Path, None] = {}
    for spec in specs:
        for p in _scenario_paths_for_spec(spec, scenarios_dir):
            seen[p] = None
    return list(seen.keys())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval.run", description="Evalúa el agente sobre mia_world.")
    parser.add_argument("--scenarios-dir", default=str(DEFAULT_SCENARIOS_DIR))
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Id, dificultad, path .json, o 'all' (repetible; default: all).",
    )
    parser.add_argument("--provider", choices=["auto", "ollama", "bedrock", "kimi"], default="auto")
    parser.add_argument("--ollama-model", default=None, help="Override de OLLAMA_MODEL para este run.")
    parser.add_argument("--ollama-host", default=None, help="Override de OLLAMA_HOST para este run.")
    parser.add_argument("--trials", type=int, default=1, help="Trials por escenario (default: 1).")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-history-messages", type=int, default=None)
    parser.add_argument(
        "--system-prompt-variant",
        choices=list(SYSTEM_PROMPT_VARIANTS.keys()),
        default="escape_room",
    )
    parser.add_argument("--module", default="student_framework")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--noop-m1-tools", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignora el resume; re-corre todo.")
    parser.add_argument("--dry-run", action="store_true", help="Lista la matriz de trials sin llamar al LLM.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenarios_dir = Path(args.scenarios_dir)
    scenario_paths = _resolve_scenario_paths(args.scenario, scenarios_dir)

    framework_config: dict[str, object] = {
        "system_prompt": SYSTEM_PROMPT_VARIANTS[args.system_prompt_variant]
    }
    if args.max_iterations is not None:
        framework_config["max_iterations"] = args.max_iterations
    if args.max_history_messages is not None:
        framework_config["max_history_messages"] = args.max_history_messages

    print(
        f"Evaluando {len(scenario_paths)} escenario(s) x {args.trials} trial(s) "
        f"— provider={args.provider} module={args.module}"
    )
    records = run_suite(
        scenario_paths,
        provider=args.provider,
        framework_config=framework_config,
        trials_per_scenario=args.trials,
        out_dir=Path(args.out_dir),
        module=args.module,
        noop_m1_tools=args.noop_m1_tools,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        force=args.force,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0
    if not records:
        print("Sin resultados.", file=sys.stderr)
        return 1

    summary = write_summary(records, Path(args.out_dir))
    print()
    print(f"success_rate_micro={summary['success_rate_micro']}  n={summary['n_trials_total']}")
    print(f"summary.json / summary.md escritos en {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
