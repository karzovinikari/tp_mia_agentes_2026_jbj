# Resumen de evaluación — M3

Generado: 2026-08-23T16:55:21  ·  trials totales: 313

- **Success rate (micro)**: 20%
- **Success rate (macro, promedio por escenario)**: 25%
- **Eficiencia media** (óptimo/real, entre éxitos): 0.60
- **Rúbrica media**: 0.92

## Por escenario

| escenario | dificultad | n | success | eficiencia | rúbrica |
|---|---|---:|---:|---:|---:|
| apartment-keys | medium | 30 | 30% | 0.67 | 0.94 |
| backtracking-vault | extreme | 57 | 5% | 0.72 | 0.88 |
| color-locks | medium | 48 | 25% | 0.60 | 0.96 |
| extreme-archive | extreme | 21 | 0% | — | 0.87 |
| library-search | hard | 21 | 24% | 0.36 | 0.91 |
| office-sequence | hard | 57 | 19% | 0.58 | 0.96 |
| study-with-key | easy | 22 | 100% | 0.62 | 0.89 |
| vault-combination | extreme | 57 | 0% | — | 0.92 |

## Desglose de errores (global)

Fallos totales: 251 / 313 trials

| categoría | n | % de los fallos |
|---|---:|---:|
| max_iterations_exhausted | 220 | 88% |
| unclassified | 19 | 8% |
| unrecovered_tool_error | 12 | 5% |

## Rúbrica — pass-rate por criterio (global)

| criterio | pass-rate | n aplicable |
|---|---:|---:|
| R1_look_first | 100% | 313 |
| R2_no_loop | 76% | 235 |
| R3_no_hallucination | 100% | 313 |
| R5_efficiency | 60% | 62 |
| R4_sequence_order | 100% | 57 |
