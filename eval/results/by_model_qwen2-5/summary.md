# Resumen de evaluación — M3

Generado: 2026-08-23T16:09:28  ·  trials totales: 120

- **Success rate (micro)**: 6%
- **Success rate (macro, promedio por escenario)**: 13%
- **Eficiencia media** (óptimo/real, entre éxitos): 0.69
- **Rúbrica media**: 0.92

## Por escenario

| escenario | dificultad | n | success | eficiencia | rúbrica |
|---|---|---:|---:|---:|---:|
| apartment-keys | medium | 15 | 0% | — | 0.98 |
| backtracking-vault | extreme | 24 | 0% | — | 0.82 |
| color-locks | medium | 15 | 7% | 0.50 | 0.97 |
| extreme-archive | extreme | 6 | 0% | — | 0.72 |
| library-search | hard | 6 | 0% | — | 1.00 |
| office-sequence | hard | 24 | 0% | — | 0.98 |
| study-with-key | easy | 6 | 100% | 0.72 | 0.91 |
| vault-combination | extreme | 24 | 0% | — | 0.93 |

## Desglose de errores (global)

Fallos totales: 113 / 120 trials

| categoría | n | % de los fallos |
|---|---:|---:|
| max_iterations_exhausted | 94 | 83% |
| unclassified | 13 | 12% |
| unrecovered_tool_error | 6 | 5% |

## Rúbrica — pass-rate por criterio (global)

| criterio | pass-rate | n aplicable |
|---|---:|---:|
| R1_look_first | 100% | 120 |
| R3_no_hallucination | 100% | 120 |
| R2_no_loop | 74% | 103 |
| R5_efficiency | 69% | 7 |
| R4_sequence_order | 100% | 24 |
