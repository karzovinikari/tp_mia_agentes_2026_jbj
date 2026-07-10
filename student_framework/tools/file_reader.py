"""Herramienta de lectura de archivos de texto con sandbox (M2).

El acceso queda acotado a `SANDBOX_ROOT` (el directorio `sandbox/` dentro
de `student_framework/`). Los errores recuperables devuelven mensajes
accionables: qué regla se violó, cómo debe verse una ruta válida y —
cuando aplica — qué archivos hay disponibles para que el LLM elija bien
en el reintento.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

# Raíz del sandbox: student_framework/sandbox. Módulo-level para poder
# modificarla en tests (no es parámetro de la función: si lo fuera,
# aparecería en el schema y el LLM podría elegir otra raíz).
# parents[0]=tools, parents[1]=student_framework.
SANDBOX_ROOT = Path(__file__).resolve().parents[1] / "sandbox"


def _list_dir(directory: Path) -> str:
    """Lista el contenido de un directorio para un mensaje accionable."""
    try:
        entries = sorted(
            p.name + ("/" if p.is_dir() else "") for p in directory.iterdir()
        )
    except OSError as e:
        return f"(no se pudo listar: {e})"
    return ", ".join(entries) if entries else "(vacío)"


def file_reader(
    path: Annotated[
        str,
        Field(
            description=(
                "Ruta RELATIVA al sandbox del archivo de texto a leer (UTF-8). "
                "Sin '/' inicial y sin '..'. Ejemplos: 'notas.txt', 'datos/info.txt'."
            )
        ),
    ],
) -> str:
    """Lee un archivo de texto UTF-8 dentro del sandbox y devuelve su contenido.

    Solo acepta rutas relativas dentro del directorio sandbox del agente;
    las rutas absolutas o con '..' se rechazan. Si el archivo no existe,
    el error indica qué archivos hay disponibles.
    """
    # --- Validación de la ruta (reglas del sandbox) ---------------------
    if not path or not path.strip():
        return (
            "Error recuperable: la ruta está vacía. Pasá una ruta relativa "
            "dentro del sandbox, por ejemplo 'notas.txt' o 'datos/info.txt'."
        )
    candidate = Path(path)
    if candidate.is_absolute():
        return (
            f"Error recuperable: la ruta '{path}' es absoluta y eso viola la "
            f"regla del sandbox. Usá una ruta relativa sin '/' inicial, "
            f"por ejemplo 'notas.txt'."
        )
    if ".." in candidate.parts:
        return (
            f"Error recuperable: la ruta '{path}' contiene '..' y eso permitiría "
            f"escapar del sandbox. Usá una ruta relativa directa, "
            f"por ejemplo 'notas.txt' o 'datos/info.txt'."
        )

    root = SANDBOX_ROOT.resolve()
    if not root.is_dir():
        # No recuperable por el LLM: es configuración del entorno.
        return f"Error: el sandbox '{root}' no existe en este entorno."

    target = (root / candidate).resolve()
    # Defensa extra (symlinks): la ruta resuelta debe seguir bajo la raíz.
    if not target.is_relative_to(root):
        return (
            f"Error recuperable: la ruta '{path}' se resuelve fuera del sandbox. "
            f"Usá una ruta relativa que apunte a un archivo dentro del sandbox."
        )

    # --- Diagnóstico accionable según el estado del filesystem ----------
    if target.is_dir():
        return (
            f"Error recuperable: '{path}' es un directorio, no un archivo. "
            f"Contiene: {_list_dir(target)}. Elegí uno de esos archivos."
        )
    if not target.exists():
        parent = target.parent
        if parent.is_dir():
            rel_parent = parent.relative_to(root)
            shown = "." if str(rel_parent) == "." else str(rel_parent)
            return (
                f"Error recuperable: el archivo '{path}' no existe. "
                f"Archivos disponibles en '{shown}': {_list_dir(parent)}. "
                f"Reintentá con uno de esos nombres."
            )
        return (
            f"Error recuperable: ni el archivo '{path}' ni su directorio existen. "
            f"Disponible en la raíz del sandbox: {_list_dir(root)}."
        )

    # --- Lectura ---------------------------------------------------------
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return (
            f"Error recuperable: '{path}' no es un archivo de texto UTF-8 "
            f"(parece binario). Elegí un archivo de texto: {_list_dir(target.parent)}."
        )
    except OSError as e:
        # Permisos u otros errores de E/S: se reporta sin romper.
        return f"Error al leer '{path}': {e}"


file_reader_schema = ToolSchema.from_callable(file_reader)
