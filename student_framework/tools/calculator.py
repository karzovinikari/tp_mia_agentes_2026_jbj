"""Herramienta de cálculo aritmético binario (cómputo puro).

M2: los errores recuperables devuelven mensajes accionables — dicen qué
parámetro falló, qué valor llegó y cómo corregirlo — para que el LLM
pueda reintentar con argumentos válidos.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

_ALLOWED_OPERATORS = ("+", "-", "*", "/", "%")


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
    # Validación de operandos (M2): el JSON del LLM puede traer strings u
    # otros tipos aunque el schema pida number. El mensaje nombra el
    # parámetro, el valor recibido y cómo corregir — error recuperable.
    # bool se excluye explícitamente: en Python bool es subclase de int.
    for pname, value in (("a", a), ("b", b)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return (
                f"Error recuperable: el parámetro '{pname}' debe ser numérico, "
                f"pero se recibió {value!r} (tipo {type(value).__name__}). "
                f"Reintentá pasando un número, por ejemplo 3 o 2.5."
            )

    if operator not in _ALLOWED_OPERATORS:
        return (
            f"Error recuperable: operador {operator!r} no soportado. "
            f"Operadores permitidos: {', '.join(_ALLOWED_OPERATORS)}. "
            f"Reintentá con uno de ellos."
        )

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
            if b == 0:
                return (
                    "Error recuperable: no se puede dividir por cero — el divisor "
                    "'b' vale 0 y la división por cero no está definida. "
                    "Reintentá con un divisor distinto de 0."
                )
            return str(a / b)
        case "%":
            if b == 0:
                return (
                    "Error recuperable: no se puede calcular módulo por cero — el "
                    "divisor 'b' vale 0 y el módulo por cero no está definido. "
                    "Reintentá con un divisor distinto de 0."
                )
            return str(a % b)

    # Inalcanzable: operator ya se validó contra _ALLOWED_OPERATORS.
    raise AssertionError("operador validado pero no manejado")


calculator_schema = ToolSchema.from_callable(calculator)
