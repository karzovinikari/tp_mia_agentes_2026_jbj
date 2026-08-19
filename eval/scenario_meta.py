"""Metadatos de los 8 escenarios que no están en el JSON: longitud óptima
de solución y peor caso de fuerza bruta, tal como los da `ENUNCIADO_M3.md`.

Se usan para la métrica de eficiencia (`metrics.efficiency_ratio`) y para
elegir qué escenarios entran en cada experimento. `tests/test_eval_harness.py`
valida que estos números coinciden con las soluciones óptimas codificadas
en `tests/conformance/test_m3_world.py`, para blindarse de un typo de
transcripción.
"""

from __future__ import annotations

OPTIMAL_TOOL_CALLS: dict[str, int] = {
    "study-with-key": 3,
    "color-locks": 11,
    "apartment-keys": 7,
    "library-search": 7,
    "office-sequence": 13,
    "extreme-archive": 4,
    "vault-combination": 21,
    "backtracking-vault": 18,
}

# None = "no cabe" (extreme-archive: el peor caso de fuerza bruta excede
# la ventana de contexto de la mayoría de modelos chicos, según el enunciado).
BRUTE_FORCE_WORST_CASE: dict[str, int | None] = {
    "study-with-key": 3,
    "color-locks": 11,
    "apartment-keys": 7,
    "library-search": 13,
    "office-sequence": 13,
    "extreme-archive": None,
    "vault-combination": 21,
    "backtracking-vault": 18,
}

# Escenarios multi-sala (registran el verbo `go`) — usados en el
# Experimento A (sensibilidad de memoria).
MULTI_ROOM_SCENARIOS: tuple[str, ...] = (
    "apartment-keys",
    "office-sequence",
    "vault-combination",
    "backtracking-vault",
)

# Escenarios cuya solución óptima se acerca o supera max_iterations=10
# (default) — usados en el Experimento B (sensibilidad de presupuesto de
# iteraciones).
LONG_SOLUTION_SCENARIOS: tuple[str, ...] = (
    "color-locks",
    "office-sequence",
    "vault-combination",
    "backtracking-vault",
)
