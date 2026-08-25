# Resumen de evaluación — M3

Generado: 2026-08-23T16:09:29  ·  trials totales: 60

- **Success rate (micro)**: 13%
- **Success rate (macro, promedio por escenario)**: 21%
- **Eficiencia media** (óptimo/real, entre éxitos): 0.50
- **Rúbrica media**: 0.92

## Por escenario

| escenario | dificultad | n | success | eficiencia | rúbrica |
|---|---|---:|---:|---:|---:|
| apartment-keys | medium | 3 | 0% | — | 1.00 |
| backtracking-vault | extreme | 12 | 0% | — | 0.89 |
| color-locks | medium | 12 | 17% | 0.51 | 0.98 |
| extreme-archive | extreme | 3 | 0% | — | 1.00 |
| library-search | hard | 3 | 33% | 0.41 | 0.95 |
| office-sequence | hard | 12 | 17% | 0.49 | 0.92 |
| study-with-key | easy | 3 | 100% | 0.53 | 0.88 |
| vault-combination | extreme | 12 | 0% | — | 0.83 |

## Desglose de errores (global)

Fallos totales: 52 / 60 trials

| categoría | n | % de los fallos |
|---|---:|---:|
| max_iterations_exhausted | 52 | 100% |

## Rúbrica — pass-rate por criterio (global)

| criterio | pass-rate | n aplicable |
|---|---:|---:|
| R1_look_first | 100% | 60 |
| R2_no_loop | 75% | 51 |
| R3_no_hallucination | 100% | 60 |
| R5_efficiency | 50% | 8 |
| R4_sequence_order | 100% | 12 |
