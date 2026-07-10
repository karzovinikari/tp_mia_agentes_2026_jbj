"""Implementación de su agente.

M1: bucle ReAct con registro de herramientas (`register_tool` + `run`).
M2: estado conversacional, sliding window acotada por `max_history_messages`,
tracking de tokens, reintentos ante fallos transitorios y salida
estructurada (`structured_call` con la tool sintética `final_result`).

Los tests de conformidad en `tests/conformance/test_m1.py` y
`test_m2.py` describen con precisión qué comportamientos deben funcionar
— léanlos antes de empezar.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from pydantic import ValidationError

from mia_agents.protocols import LLMClient
from mia_agents.tool_schema import FINAL_RESULT_TOOL_NAME, final_result_tool_schema
from mia_agents.types import AgentResult, AgentStep, LLMResponse, ToolSchema


class StructuredOutputError(RuntimeError):
    """`structured_call` agotó los reintentos sin lograr una salida válida."""


# --- Clasificación de fallos transitorios (M2) -----------------------------
# Dos criterios complementarios: tipos estándar de red/timeout de Python, y
# marcadores de texto que cubren los errores típicos de proveedor (throttling,
# 5xx, rate limit) sin acoplar el agente a las excepciones concretas de
# boto3/ollama (que no podemos importar sin arrastrar esas dependencias).
_TRANSIENT_EXC_TYPES = (TimeoutError, ConnectionError)
_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "rate",
    "throttl",
    "overload",
    "connection",
    "temporar",
    "unavailable",
    "429",
    "500",
    "502",
    "503",
)


def _is_transient(exc: Exception) -> bool:
    """True si el fallo es transitorio y vale la pena reintentar."""
    if isinstance(exc, _TRANSIENT_EXC_TYPES):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class MyAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = "Eres un asistente útil.",
        max_iterations: int = 10,
        max_history_messages: int = 50,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
    ) -> None:
        """Inicializa el agente.

        Parameters
        ----------
        llm_client : LLMClient
            Cliente LLM (real o mock) que el agente utilizará.
        system_prompt : str
            System prompt por defecto.
        max_iterations : int
            Tope de iteraciones del bucle del agente (M1).
        max_history_messages : int
            Número máximo de mensajes que se permiten en la lista
            `messages` enviada al LLM en una única llamada (M2). La
            sliding window de `_window()` garantiza este tope.
        max_retries : int
            Reintentos adicionales ante fallos transitorios del LLM o de
            una herramienta (M2). El total de intentos es 1 + max_retries.
        retry_base_delay : float
            Base del backoff exponencial entre reintentos, en segundos.
            Los tests lo bajan a 0 para no dormir.
        """
        self._llm = llm_client
        self._system = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max_history_messages
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        # Dos dicts paralelos keyed por schema.name.
        # _tools: el callable a invocar; _schemas: el ToolSchema para pasarle al LLM.
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: dict[str, ToolSchema] = {}
        # Historial conversacional (M2): persiste entre llamadas a run()
        # sobre la misma instancia. Crece sin límite acá; el tope se aplica
        # en _window() al armar cada llamada al LLM.
        self._history: list[dict[str, Any]] = []

    def register_tool(
        self,
        tool: Callable[..., str],
        schema: ToolSchema,
    ) -> None:
        """Registra una herramienta callable junto a su esquema.

        El esquema suele obtenerse con `ToolSchema.from_callable(fn)`. En
        `run`, pasá `tools=list(self._schemas.values())`; el cliente LLM
        aplica `to_llm_spec()` al llamar al proveedor.

        El callable se invoca con kwargs que coinciden con la firma.
        Debe devolver una cadena.
        """
        # Guardamos callable y schema con la misma clave (schema.name)
        # para poder buscarlos en O(1) durante el loop.
        self._tools[schema.name] = tool
        self._schemas[schema.name] = schema

    # ------------------------------------------------------------------
    # Memoria (M2): sliding window sobre el historial persistente
    # ------------------------------------------------------------------
    def _window(self) -> list[dict[str, Any]]:
        """Recorta el historial a `max_history_messages` para una llamada.

        Estrategia: sliding window por recencia — se conservan los últimos
        N mensajes y se descarta el contexto más viejo. Dos invariantes:

        1. Recencia: el mensaje de usuario más reciente SIEMPRE entra en la
           ventana, aunque un turno con muchos tool_calls lo hubiera
           empujado fuera de los últimos N.
        2. Sin tool huérfano: la ventana no puede empezar con un mensaje
           role="tool" cuyo turno assistant (con el tool_call) quedó
           recortado — los proveedores reales (Bedrock) rechazan un
           toolResult sin su toolUse previo.
        """
        budget = self._max_history_messages
        msgs = self._history
        if len(msgs) <= budget:
            window = list(msgs)
        else:
            window = msgs[-budget:]
            last_user = next(
                (m for m in reversed(msgs) if m.get("role") == "user"), None
            )
            if last_user is not None and last_user not in window:
                # Forzamos el último mensaje de usuario al inicio y
                # completamos con lo más reciente. (budget-1 puede ser 0:
                # en ese caso la ventana es solo el mensaje del usuario.)
                tail = window[-(budget - 1):] if budget > 1 else []
                window = [last_user] + tail
        while window and window[0].get("role") == "tool":
            window.pop(0)
        return window

    # ------------------------------------------------------------------
    # Resiliencia (M2): reintentos con backoff ante fallos transitorios
    # ------------------------------------------------------------------
    def _chat_with_retries(self, **chat_kwargs: Any) -> LLMResponse:
        """Llama a `chat` reintentando solo los fallos transitorios.

        Los errores no transitorios (bugs, credenciales, argumentos
        inválidos) afloran limpios en el primer intento: reintentarlos
        solo demoraría el mismo fallo.
        """
        attempts = 1 + self._max_retries
        for attempt in range(attempts):
            try:
                return self._llm.chat(**chat_kwargs)
            except Exception as exc:
                if not _is_transient(exc) or attempt == attempts - 1:
                    raise
                delay = self._retry_base_delay * (2**attempt)
                if delay > 0:
                    time.sleep(delay)
        raise RuntimeError("inalcanzable: el loop de reintentos siempre retorna o lanza")

    def _call_tool_with_retries(self, name: str, kwargs: dict[str, Any]) -> str:
        """Invoca una tool reintentando fallos transitorios (misma política)."""
        attempts = 1 + self._max_retries
        for attempt in range(attempts):
            try:
                return self._tools[name](**kwargs)
            except Exception as exc:
                if not _is_transient(exc) or attempt == attempts - 1:
                    raise
                delay = self._retry_base_delay * (2**attempt)
                if delay > 0:
                    time.sleep(delay)
        raise RuntimeError("inalcanzable: el loop de reintentos siempre retorna o lanza")

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------
    def run(self, user_message: str) -> AgentResult:
        """Ejecuta el bucle del agente hasta una respuesta final o hasta max_iterations.

        M1: bucle ReAct — chat con tools, ejecutar tool_calls, realimentar
        resultados, terminar cuando llega texto sin tool_calls.
        M2: además, el historial persiste entre llamadas (statefulness),
        cada llamada al LLM va acotada por `_window()`, y se acumulan los
        tokens reportados en `AgentResult.input_tokens/output_tokens`.
        """
        # Statefulness (M2): el turno nuevo se suma al historial persistente.
        self._history.append({"role": "user", "content": user_message})
        steps: list[AgentStep] = []

        # Tracking de tokens (M2): sumamos lo que reporte cada LLMResponse.
        # Si NINGUNA respuesta reporta tokens, ambos campos quedan None
        # (contrato del dataclass); si alguna reporta, None por respuesta
        # cuenta como 0.
        input_tokens = 0
        output_tokens = 0
        tokens_reported = False

        # Pasamos los ToolSchema tal cual; el LLMClient fijo aplica
        # to_llm_spec() y traduce al formato del proveedor.
        tools = list(self._schemas.values()) if self._schemas else None

        # Cada vuelta del for = exactamente UNA llamada al LLM. Acotar por
        # max_iterations garantiza terminación aunque el modelo entre en un
        # loop de tool_calls que nunca converge a texto final.
        for _ in range(self._max_iterations):
            resp = self._chat_with_retries(
                messages=self._window(),
                tools=tools,
                system=self._system,
            )
            if resp.input_tokens is not None or resp.output_tokens is not None:
                tokens_reported = True
                input_tokens += resp.input_tokens or 0
                output_tokens += resp.output_tokens or 0

            # Condición de parada: texto SIN tool_calls => respuesta final.
            if not resp.tool_calls:
                answer = resp.content or ""
                self._history.append({"role": "assistant", "content": answer})
                return AgentResult(
                    answer=answer,
                    steps=steps,
                    input_tokens=input_tokens if tokens_reported else None,
                    output_tokens=output_tokens if tokens_reported else None,
                )

            # El modelo pidió herramientas. Registramos el turno assistant
            # (con sus tool_calls) para que el LLM vea, en la próxima
            # llamada, qué pidió y qué resultado obtuvo.
            self._history.append(
                {
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in resp.tool_calls
                    ],
                }
            )

            # Ejecutamos cada tool_call y volcamos su resultado como mensaje
            # role="tool": realimentación al LLM en la siguiente llamada.
            for tc in resp.tool_calls:
                step = self._execute_tool_call(tc)
                steps.append(step)
                content = step.tool_output if step.error is None else step.error
                self._history.append(
                    {
                        "role": "tool",
                        "content": content or "",
                        "tool_call_id": tc.id,
                    }
                )

        # Se agotó max_iterations sin respuesta final. Devolvemos igual un
        # AgentResult válido con los steps y tokens acumulados.
        return AgentResult(
            answer="",
            steps=steps,
            error=f"Se alcanzó el límite de {self._max_iterations} iteraciones sin respuesta final.",
            input_tokens=input_tokens if tokens_reported else None,
            output_tokens=output_tokens if tokens_reported else None,
        )

    def _execute_tool_call(self, tool_call: Any) -> AgentStep:
        """Ejecuta un único tool_call y lo registra como AgentStep.

        Devuelve siempre un AgentStep (nunca lanza): una tool desconocida o
        un fallo de ejecución quedan registrados con `error` no nulo, sin
        romper el agente. Los fallos transitorios dentro de la tool se
        reintentan antes de darse por vencidos.
        """
        name = tool_call.name
        raw_args = tool_call.arguments  # string JSON según ToolCall.arguments

        # Tool alucinada: el LLM pidió un nombre que no registramos.
        if name not in self._tools:
            return AgentStep(
                tool_name=name,
                tool_input=raw_args,
                tool_output=None,
                error=f"Herramienta desconocida: '{name}'",
            )

        # Parseo de argumentos: ToolCall.arguments es SIEMPRE un string JSON;
        # lo convertimos a dict para invocar el callable con **kwargs.
        try:
            kwargs = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as e:
            return AgentStep(
                tool_name=name,
                tool_input=raw_args,
                tool_output=None,
                error=f"Argumentos JSON inválidos: {e}",
            )

        # Ejecución con reintentos transitorios. Cualquier excepción final
        # (TypeError por kwargs que no matchean la firma, errores internos)
        # se registra como error del step en vez de propagarse.
        try:
            output = self._call_tool_with_retries(name, kwargs)
            return AgentStep(
                tool_name=name,
                tool_input=raw_args,
                tool_output=output,
                error=None,
            )
        except Exception as e:  # noqa: BLE001 — robustez: ningún fallo de tool rompe run()
            return AgentStep(
                tool_name=name,
                tool_input=raw_args,
                tool_output=None,
                error=f"Error ejecutando '{name}': {e}",
            )

    # ------------------------------------------------------------------
    # Salida estructurada (M2)
    # ------------------------------------------------------------------
    def structured_call(
        self,
        prompt: str,
        schema: Any,
        max_repair_attempts: int = 2,
    ) -> Any:
        """Pide al LLM una respuesta validada contra `schema` (Pydantic).

        Contrato M2:
          - Cada llamada a chat ofrece la tool sintética `final_result`
            (schema derivado del modelo con `final_result_tool_schema`).
          - Termina solo cuando llega un tool_call a `final_result` cuyos
            argumentos validan con `schema.model_validate(...)`.
          - Si el modelo responde texto libre, o los argumentos no parsean
            o no validan, se reintenta con un mensaje de reparación que
            incluye el error concreto y el JSON Schema esperado.
          - Total de llamadas: 1 inicial + `max_repair_attempts`. Si se
            agotan, levanta `StructuredOutputError` (nunca devuelve None
            ni instancias parciales).

        Decisión de diseño: usa una conversación propia y aislada (no toca
        `self._history`): una extracción estructurada es una operación
        puntual, no un turno del diálogo con el usuario.
        """
        final_tool = final_result_tool_schema(schema)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        attempts = 1 + max_repair_attempts
        last_error = "no se realizó ningún intento"

        for _ in range(attempts):
            resp = self._chat_with_retries(
                messages=messages,
                tools=[final_tool],
                system=self._system,
            )

            call = next(
                (tc for tc in resp.tool_calls if tc.name == FINAL_RESULT_TOOL_NAME),
                None,
            )
            if call is None:
                # Modo de fallo 1: texto libre (o tool equivocada) en vez de
                # invocar final_result.
                last_error = (
                    f"respondió con texto libre en lugar de invocar la "
                    f"herramienta '{FINAL_RESULT_TOOL_NAME}'"
                )
                messages.append({"role": "assistant", "content": resp.content or ""})
                messages.append(
                    {"role": "user", "content": self._repair_prompt(last_error, schema)}
                )
                continue

            # Registramos el turno assistant con su tool_call antes de validar,
            # para que un eventual mensaje de reparación tenga el contexto.
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                    ],
                }
            )

            try:
                # Modo de fallo 2: arguments no es JSON válido.
                args = json.loads(call.arguments) if call.arguments else {}
                # Modo de fallo 3: JSON válido pero no valida contra el schema.
                return schema.model_validate(args)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = f"los argumentos no validan contra el schema: {exc}"
                messages.append(
                    {
                        "role": "tool",
                        "content": last_error,
                        "tool_call_id": call.id,
                    }
                )
                messages.append(
                    {"role": "user", "content": self._repair_prompt(last_error, schema)}
                )

        raise StructuredOutputError(
            f"structured_call agotó {attempts} intentos "
            f"(1 inicial + {max_repair_attempts} reparaciones). "
            f"Último error: {last_error}"
        )

    @staticmethod
    def _repair_prompt(error: str, schema: Any) -> str:
        """Mensaje de reparación: el error concreto + el schema esperado."""
        return (
            f"Tu respuesta anterior no sirvió: {error}. "
            f"Invocá la herramienta '{FINAL_RESULT_TOOL_NAME}' con argumentos "
            f"que cumplan exactamente este JSON Schema:\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        )
