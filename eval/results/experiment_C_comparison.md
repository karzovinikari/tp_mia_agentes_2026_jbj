# Experimento C — comparación de arms (m1_tools (real vs. noop))

**Hipótesis:** Tools de M1 irrelevantes en el prompt aumentan hallucinated_tool_or_args o bajan éxito.

| arm | n | success (micro) | success (macro) | eficiencia media | top error |
|---|---:|---:|---:|---:|---|
| real | 24 | 12% | 12% | 0.75 | max_iterations_exhausted |
| noop | 24 | 12% | 12% | 0.70 | max_iterations_exhausted |

## Desglose de errores por arm

### arm = real
- max_iterations_exhausted: 20
- unclassified: 1

### arm = noop
- max_iterations_exhausted: 19
- unclassified: 2

