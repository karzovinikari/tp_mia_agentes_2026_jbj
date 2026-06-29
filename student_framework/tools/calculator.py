"""Herramienta de cálculo aritmético binario (cómputo puro).

Sigue el patrón de `example.py`: callable tipado con Annotated + Field,
docstring descriptivo y schema generado con ToolSchema.from_callable.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema


def calculator(
    a: Annotated[float, Field(description="Primer operando numérico.")],
    b: Annotated[float, Field(description="Segundo operando numérico.")],
    operator: Annotated[
        str,
        Field(description="Operador a aplicar: uno de '+', '-', '*', '/' o '%'."),
    ],
) -> str:
    """Calcula una operación aritmética binaria entre dos números.

    Soporta suma (+), resta (-), multiplicación (*), división (/) y
    módulo (%). Devuelve el resultado como cadena. No usa eval ni
    evalúa expresiones arbitrarias: solo aplica el operador indicado.
    """
    # match explícito en vez de un dict de lambdas o eval: queda auditable
    # y deja claro que solo estas operaciones están permitidas.
    match operator:
        case "+":
            return str(a + b)
        case "-":
            return str(a - b)
        case "*":
            return str(a * b)
        case "/":
            # La división por cero se maneja devolviendo un string de error
            # (no excepción): el agente nunca debe romperse por input del LLM.
            if b == 0:
                return "Error: división por cero"
            return str(a / b)
        case "%":
            if b == 0:
                return "Error: módulo por cero"
            return str(a % b)
        case _:
            return f"Error: operador '{operator}' no soportado (usá +, -, *, / o %)"


calculator_schema = ToolSchema.from_callable(calculator)
