# Experimento A — comparación de arms (max_history_messages)

**Hipótesis:** Ventana chica degrada éxito/eficiencia en multi-sala porque el agente olvida el mapa o qué ítems ya tiene.

| arm | n | success (micro) | success (macro) | eficiencia media | top error |
|---|---:|---:|---:|---:|---|
| 8 | 12 | 0% | 0% | — | max_iterations_exhausted |
| 16 | 12 | 0% | 0% | — | max_iterations_exhausted |
| 50 | 12 | 0% | 0% | — | max_iterations_exhausted |

## Desglose de errores por arm

### arm = 8
- max_iterations_exhausted: 8
- unclassified: 4

### arm = 16
- max_iterations_exhausted: 11
- unclassified: 1

### arm = 50
- max_iterations_exhausted: 11
- unrecovered_tool_error: 1

