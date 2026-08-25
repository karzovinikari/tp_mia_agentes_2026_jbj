# Resumen de evaluación — M3

Generado: 2026-08-20T20:34:13  ·  trials totales: 60

- **Success rate (micro)**: 27%
- **Success rate (macro, promedio por escenario)**: 35%
- **Eficiencia media** (óptimo/real, entre éxitos): 0.64
- **Rúbrica media**: 0.96

## Por escenario

| escenario | dificultad | n | success | eficiencia | rúbrica |
|---|---|---:|---:|---:|---:|
| apartment-keys | medium | 3 | 100% | 0.70 | 0.90 |
| backtracking-vault | extreme | 12 | 25% | 0.72 | 0.98 |
| color-locks | medium | 12 | 33% | 0.65 | 0.96 |
| extreme-archive | extreme | 3 | 0% | — | 0.89 |
| library-search | hard | 3 | 0% | — | 0.89 |
| office-sequence | hard | 12 | 25% | 0.62 | 0.98 |
| study-with-key | easy | 3 | 100% | 0.49 | 0.85 |
| vault-combination | extreme | 12 | 0% | — | 1.00 |

## Desglose de errores (global)

Fallos totales: 44 / 60 trials

| categoría | n | % de los fallos |
|---|---:|---:|
| max_iterations_exhausted | 44 | 100% |

## Rúbrica — pass-rate por criterio (global)

| criterio | pass-rate | n aplicable |
|---|---:|---:|
| R1_look_first | 100% | 60 |
| R3_no_hallucination | 100% | 60 |
| R5_efficiency | 64% | 16 |
| R2_no_loop | 94% | 31 |
| R4_sequence_order | 100% | 12 |
