# Experimento A — comparación de arms (max_history_messages)

**Hipótesis:** Ventana chica degrada éxito/eficiencia en multi-sala porque el agente olvida el mapa o qué ítems ya tiene.

| arm | n | success (micro) | success (macro) | eficiencia media | top error |
|---|---:|---:|---:|---:|---|
| 8 | 12 | 0% | 0% | — | unrecovered_tool_error |
| 16 | 12 | 0% | 0% | — | max_iterations_exhausted |
| 50 | 12 | 0% | 0% | — | unrecovered_tool_error |

## Desglose de errores por arm

### arm = 8
- unrecovered_tool_error: 10
- max_iterations_exhausted: 2

### arm = 16
- max_iterations_exhausted: 9
- unrecovered_tool_error: 3

### arm = 50
- unrecovered_tool_error: 6
- max_iterations_exhausted: 5
- unclassified: 1

