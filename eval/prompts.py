"""Variantes de system prompt para la corrida de evaluación.

`BASELINE_SYSTEM_PROMPT` es literalmente el default de `MyAgent` (mismo
string) — usarlo explícitamente en `eval/` deja la comparación con
`ESCAPE_ROOM_SYSTEM_PROMPT` documentada y reproducible, en vez de depender
de un default implícito que podría cambiar en `agent.py` sin que el
experimento se entere.
"""

from __future__ import annotations

BASELINE_SYSTEM_PROMPT = "Eres un asistente útil."

ESCAPE_ROOM_SYSTEM_PROMPT = (
    "Sos un agente que intenta escapar de una sala simulada. Tenés las "
    "herramientas look, examine, take, use (y go si hay navegación entre "
    "salas). Antes de actuar a ciegas, usá look para ver qué hay en la "
    "sala. No hay mapa: si navegás entre salas, llevá vos mismo la cuenta "
    "de dónde encontraste cada objeto para poder volver. Si el objetivo "
    "pide un orden (por ejemplo, llevarte algo ANTES de abrir una puerta "
    "que se sella), planificá ese orden antes de actuar, no lo "
    "improvises. Si una acción devuelve un error, no la repitas igual: "
    "leé el error y cambiá el argumento o el enfoque."
)

SYSTEM_PROMPT_VARIANTS: dict[str, str] = {
    "baseline": BASELINE_SYSTEM_PROMPT,
    "escape_room": ESCAPE_ROOM_SYSTEM_PROMPT,
}
