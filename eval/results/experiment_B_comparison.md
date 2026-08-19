# Experimento B — comparación de arms (max_iterations)

**Hipótesis:** El default de 10 round-trips es insuficiente en soluciones largas porque el modelo no batchea varios tool_calls por respuesta.

| arm | n | success (micro) | success (macro) | eficiencia media | top error |
|---|---:|---:|---:|---:|---|
| 10 | 12 | 0% | 0% | — | max_iterations_exhausted |
| 15 | 12 | 0% | 0% | — | max_iterations_exhausted |
| 25 | 12 | 17% | 17% | 0.50 | unclassified |

## Desglose de errores por arm

### arm = 10
- max_iterations_exhausted: 12

### arm = 15
- max_iterations_exhausted: 11
- unclassified: 1

### arm = 25
- unclassified: 6
- max_iterations_exhausted: 3
- unrecovered_tool_error: 1

