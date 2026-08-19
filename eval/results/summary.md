# Resumen de evaluación — M3

Generado: 2026-08-11T20:19:05  ·  trials totales: 24

- **Success rate (micro)**: 12%
- **Success rate (macro, promedio por escenario)**: 12%
- **Eficiencia media** (óptimo/real, entre éxitos): 0.75
- **Rúbrica media**: 0.88

## Por escenario

| escenario | dificultad | n | success | eficiencia | rúbrica |
|---|---|---:|---:|---:|---:|
| apartment-keys | medium | 3 | 0% | — | 0.89 |
| backtracking-vault | extreme | 3 | 0% | — | 0.78 |
| color-locks | medium | 3 | 0% | — | 1.00 |
| extreme-archive | extreme | 3 | 0% | — | 0.78 |
| library-search | hard | 3 | 0% | — | 0.89 |
| office-sequence | hard | 3 | 0% | — | 0.92 |
| study-with-key | easy | 3 | 100% | 0.75 | 0.92 |
| vault-combination | extreme | 3 | 0% | — | 0.89 |

## Desglose de errores (global)

Fallos totales: 21 / 24 trials

| categoría | n | % de los fallos |
|---|---:|---:|
| max_iterations_exhausted | 17 | 81% |
| unclassified | 3 | 14% |
| unrecovered_tool_error | 1 | 5% |

## Rúbrica — pass-rate por criterio (global)

| criterio | pass-rate | n aplicable |
|---|---:|---:|
| R1_look_first | 100% | 24 |
| R3_no_hallucination | 100% | 24 |
| R5_efficiency | 75% | 3 |
| R2_no_loop | 56% | 18 |
| R4_sequence_order | 100% | 3 |
