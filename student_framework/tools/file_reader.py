"""Herramienta de lectura de archivos de texto (E/S restringida).

Sigue el patrón de `example.py`: callable tipado con Annotated + Field,
docstring descriptivo y schema generado con ToolSchema.from_callable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema


def file_reader(
    path: Annotated[
        str, Field(description="Ruta al archivo de texto a leer (UTF-8).")
    ],
) -> str:
    """Lee un archivo de texto y devuelve su contenido como cadena.

    Solo archivos de texto codificados en UTF-8. Si el archivo no existe
    o no es texto (p. ej. binario), devuelve un mensaje de error en lugar
    de lanzar una excepción.
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: el archivo '{path}' no existe"
    except IsADirectoryError:
        return f"Error: '{path}' es un directorio, no un archivo"
    except UnicodeDecodeError:
        # Archivo binario o con otra codificación: lo reportamos sin romper.
        return f"Error: '{path}' no es un archivo de texto UTF-8 (¿binario?)"
    except OSError as e:
        # Permisos, ruta inválida, etc. Capturamos OSError (no Exception
        # genérica) para no esconder bugs reales de programación.
        return f"Error al leer '{path}': {e}"


file_reader_schema = ToolSchema.from_callable(file_reader)
