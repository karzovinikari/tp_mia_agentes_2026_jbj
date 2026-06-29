"""Herramienta libre: contador de palabras.

Tercera herramienta obligatoria del M1 (de libre elección). Demuestra el
mismo patrón que las otras: callable tipado con Annotated + Field,
docstring descriptivo y schema generado con ToolSchema.from_callable.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema


def word_counter(
    text: Annotated[
        str, Field(description="Texto cuyas palabras se quieren contar.")
    ],
) -> str:
    """Cuenta la cantidad de palabras en el texto recibido.

    Una "palabra" es cualquier secuencia separada por espacios en blanco
    (str.split() colapsa espacios múltiples y saltos de línea). Devuelve
    el conteo como cadena.
    """
    return str(len(text.split()))


word_counter_schema = ToolSchema.from_callable(word_counter)
