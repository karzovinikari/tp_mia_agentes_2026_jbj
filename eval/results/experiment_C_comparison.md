# Experimento C — comparación de arms (m1_tools (real vs. noop))

**Hipótesis:** Tools de M1 irrelevantes en el prompt aumentan hallucinated_tool_or_args o bajan éxito.

| arm | n | success (micro) | success (macro) | eficiencia media | top error |
|---|---:|---:|---:|---:|---|
| real | 24 | 0% | 0% | — | unrecovered_tool_error |
| noop | 24 | 0% | 0% | — | max_iterations_exhausted |

## Desglose de errores por arm

### arm = real
- unrecovered_tool_error: 12
- max_iterations_exhausted: 10
- unclassified: 2

### arm = noop
- max_iterations_exhausted: 14
- unrecovered_tool_error: 10

