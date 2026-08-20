# Experimento B — comparación de arms (max_iterations)

**Hipótesis:** El default de 10 round-trips es insuficiente en soluciones largas porque el modelo no batchea varios tool_calls por respuesta.

| arm | n | success (micro) | success (macro) | eficiencia media | top error |
|---|---:|---:|---:|---:|---|
| 10 | 12 | 0% | 0% | — | unrecovered_tool_error |
| 15 | 12 | 0% | 0% | — | unrecovered_tool_error |
| 25 | 12 | 0% | 0% | — | unrecovered_tool_error |

## Desglose de errores por arm

### arm = 10
- unrecovered_tool_error: 8
- max_iterations_exhausted: 4

### arm = 15
- unrecovered_tool_error: 7
- max_iterations_exhausted: 4
- unclassified: 1

### arm = 25
- unrecovered_tool_error: 8
- max_iterations_exhausted: 3
- unclassified: 1

