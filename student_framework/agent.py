"""Implementación de su agente.

Completen `register_tool` y `run` para el Milestone 1.
En el Milestone 2 amplíen `MyAgent` para que sea estatal y respete
`max_history_messages`.

Los tests de conformidad en `tests/conformance/test_m1.py` y
`test_m2.py` describen con precisión qué comportamientos deben funcionar
— léanlos antes de empezar.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from mia_agents.protocols import LLMClient
from mia_agents.types import AgentResult, AgentStep, ToolSchema


class MyAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = "Eres un asistente útil.",
        max_iterations: int = 10,
        max_history_messages: int = 50,
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
            `messages` enviada al LLM en una única llamada. En M1 este
            valor es ignorado; el agente sólo necesita aceptarlo en su
            constructor. En M2 deben respetarlo: la longitud de la
            lista de mensajes pasada a `self._llm.chat(...)` no puede
            superar este número en ninguna llamada, sin importar la
            estrategia de memoria que elijan.
        """
        self._llm = llm_client
        self._system = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max_history_messages
        # Dos dicts paralelos keyed por schema.name.
        # _tools: el callable a invocar; _schemas: el ToolSchema para pasarle al LLM.
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: dict[str, ToolSchema] = {}
        # TODO (M2): inicializa la estructura de historial conversacional.

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

    def run(self, user_message: str) -> AgentResult:
        """Ejecuta el bucle del agente hasta una respuesta final o hasta max_iterations.

        Comportamiento esperado (consulta tests/conformance/test_m1.py
        para el contrato exacto del M1):
          - Llama a `self._llm.chat(..., tools=list(self._schemas.values()))`.
          - Si la respuesta contiene tool_calls, ejecuta cada uno y vuelca
            los resultados en la siguiente llamada al chat.
          - Si la respuesta solo contiene texto (sin `tool_calls`),
            devuélvelo en `AgentResult.answer`. En M1 no uses la tool
            sintética `final_result`; ese patrón es de M2 (ver README y
            ENUNCIADO_M2.md).
          - Limita el bucle a `self._max_iterations` y termina de forma
            limpia cuando se alcance.
          - Registra cada invocación de herramienta como un `AgentStep`
            dentro de `result.steps`.

        En el M2, además, llamadas sucesivas sobre la misma instancia
        deben continuar la conversación, y la longitud de la lista de
        mensajes enviada al LLM no debe superar `self._max_history_messages`.
        Acumula los tokens de entrada/salida reportados por los
        `LLMResponse` y exponlos en `AgentResult.input_tokens` /
        `AgentResult.output_tokens`.
        """
        # Historial local de la conversación (M1 es stateless: empieza vacío
        # en cada run y solo vive durante esta llamada).
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]
        steps: list[AgentStep] = []

        # Pasamos los ToolSchema tal cual; el LLMClient fijo aplica
        # to_llm_spec() y traduce al formato del proveedor. Si no hay
        # tools registradas, mandamos None (el contrato lo permite y evita
        # enviar una lista vacía).
        tools = list(self._schemas.values()) if self._schemas else None

        # Cada vuelta del for = exactamente UNA llamada al LLM. Acotar por
        # max_iterations garantiza finalización aunque el modelo entre en un
        # loop de tool_calls que nunca converge a texto final.
        for _ in range(self._max_iterations):
            resp = self._llm.chat(
                messages=messages,
                tools=tools,
                system=self._system,
            )

            # Condición de parada de M1: texto SIN tool_calls. Ese texto es
            # la respuesta final. steps queda como lo acumulado (vacío si
            # nunca se llamó a una tool).
            if not resp.tool_calls:
                return AgentResult(answer=resp.content or "", steps=steps)

            # El modelo pidió herramientas. Primero registramos el turno del
            # asistente en el historial (con sus tool_calls) para que el LLM
            # vea, en la próxima llamada, qué pidió y qué resultado obtuvo.
            messages.append(
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
            # role="tool". Ese mensaje aparecerá en los `messages` de la
            # siguiente llamada a chat (realimentación al LLM).
            for tc in resp.tool_calls:
                step = self._execute_tool_call(tc)
                steps.append(step)
                # El contenido del mensaje tool es el output, o el error si
                # falló: en ambos casos el LLM recibe una observación útil.
                content = step.tool_output if step.error is None else step.error
                messages.append(
                    {
                        "role": "tool",
                        "content": content or "",
                        "tool_call_id": tc.id,
                    }
                )

        # Se agotó max_iterations sin una respuesta de texto final. Devolvemos
        # igual un AgentResult válido (contrato de terminación): answer vacío,
        # pero con los steps acumulados y un error explicativo.
        return AgentResult(
            answer="",
            steps=steps,
            error=f"Se alcanzó el límite de {self._max_iterations} iteraciones sin respuesta final.",
        )

    def _execute_tool_call(self, tool_call: Any) -> AgentStep:
        """Ejecuta un único tool_call y lo registra como AgentStep.

        Devuelve siempre un AgentStep (nunca lanza): el contrato de M1 exige
        que una tool desconocida o un fallo de ejecución queden registrados
        con `error` no nulo, sin romper el agente.
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

        # Ejecución del callable. Capturamos cualquier excepción (TypeError
        # por kwargs que no matchean la firma, errores internos de la tool,
        # etc.) y la registramos como error del step en vez de propagarla.
        try:
            output = self._tools[name](**kwargs)
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

    def structured_call(
        self,
        prompt: str,
        schema: Any,
        max_repair_attempts: int = 2,
    ) -> Any:
        """Pide al LLM una respuesta validada contra `schema` (M2).

        Obligatorio: herramienta sintética `final_result` (ver
        `mia_agents.final_result_tool_schema` / `FINAL_RESULT_TOOL_NAME`).
        El agente ofrece esa tool al LLM, valida los `arguments` del
        `tool_call` y reintenta con contexto de reparación si el modelo
        responde con texto libre o con argumentos inválidos.

        Implementa esto en el M2:
          - Pasa `tools=[final_result_tool_schema(schema)]` en cada
            llamada a `chat` dentro de este método.
          - Termina solo cuando llega un `tool_call` a `final_result`
            cuyos argumentos validan con `schema.model_validate(...)`.
          - Reintenta hasta `max_repair_attempts` incluyendo el fallo en
            los mensajes (respuesta previa, mensaje `tool`, o user de
            reparación).
          - Si tras los reintentos sigue fallando, levanta una excepción
            limpia (no devuelvas valores parciales ni `None` sin avisar).

        El M1 deja esto como stub; los tests de M2 verifican el contrato.
        """
        raise NotImplementedError("M2: implementa salida estructurada con reparación")
