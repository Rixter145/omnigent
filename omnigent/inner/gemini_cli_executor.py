"""One-turn Executor adapter for the verified Gemini CLI transport."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping

from omnigent.gemini_cli import GeminiCliTransport, GeminiResult, redact_text

from .executor import (
    Executor,
    ExecutorConfig,
    ExecutorError,
    ExecutorEvent,
    Message,
    TextChunk,
    ToolSpec,
    TurnCancelled,
    TurnComplete,
)

MAX_HISTORY_MESSAGES = 32
MAX_MESSAGE_CHARS = 16_384
MAX_PROMPT_CHARS = 128 * 1024


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content[:64]:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text", item.get("content", ""))
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content) if content is not None else ""


def build_gemini_prompt(messages: list[Message], system_prompt: str) -> str:
    """Compose bounded system/history context for a fresh CLI invocation."""
    system_line = ""
    if system_prompt:
        system_line = f"system: {system_prompt[:MAX_MESSAGE_CHARS]}"
    history_lines: list[str] = []
    for message in messages[-MAX_HISTORY_MESSAGES:]:
        role = message.get("role")
        role_name = role if isinstance(role, str) and role else "user"
        text = _content_text(message.get("content"))[:MAX_MESSAGE_CHARS]
        if text:
            history_lines.append(f"{role_name}: {text}")
    history = "\n\n".join(history_lines)
    if system_line:
        available = MAX_PROMPT_CHARS - len(system_line) - 2
        return f"{system_line}\n\n{history[-max(0, available) :]}"[:MAX_PROMPT_CHARS]
    return history[-MAX_PROMPT_CHARS:]


def _session_key(messages: list[Message]) -> str:
    if messages:
        last = messages[-1]
        if last.get("session_id"):
            return str(last["session_id"])
        metadata = last.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("session_id"):
            return str(metadata["session_id"])
    return "__default__"


class GeminiCliExecutor(Executor):
    """Run exactly one bounded Gemini CLI subscription turn per request."""

    def __init__(
        self,
        transport: GeminiCliTransport | None = None,
        *,
        timeout_s: float = 120.0,
        model: str | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        self._transport = transport or GeminiCliTransport()
        self._timeout_s = timeout_s
        self._model = model
        self._cancel_events: dict[str, asyncio.Event] = {}

    def supports_streaming(self) -> bool:
        # The transport parses bounded JSONL after the process completes; it
        # does not yet expose token deltas through the Executor interface.
        return False

    def supports_tool_calling(self) -> bool:
        return False

    async def interrupt_session(self, session_key: str) -> bool:
        event = self._cancel_events.get(session_key)
        if event is None:
            return False
        event.set()
        return True

    async def close_session(self, session_key: str) -> None:
        await self.interrupt_session(session_key)
        self._cancel_events.pop(session_key, None)

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        del tools
        session_key = _session_key(messages)
        cancel_event = asyncio.Event()
        self._cancel_events[session_key] = cancel_event
        try:
            prompt = build_gemini_prompt(messages, system_prompt)
            if not prompt.strip():
                yield ExecutorError(message="Gemini CLI requires a non-empty prompt")
                return
            result: GeminiResult = await self._transport.run(
                prompt,
                timeout_s=self._timeout_s,
                cancel_event=cancel_event,
                model=(config.model if config is not None and config.model else self._model),
            )
            if result.cancelled:
                yield TurnCancelled(reason="Gemini CLI turn cancelled")
                return
            if result.error is not None or not result.completed:
                yield ExecutorError(
                    message=redact_text(result.error or "Gemini CLI ended without a completion")
                )
                return
            if result.text:
                yield TextChunk(text=result.text)
            yield TurnComplete(
                response=result.text,
                usage=dict(result.usage) if result.usage else None,
            )
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            yield ExecutorError(message=redact_text(exc))
        finally:
            self._cancel_events.pop(session_key, None)


GeminiCliSubscriptionExecutor = GeminiCliExecutor

__all__ = ["GeminiCliExecutor", "GeminiCliSubscriptionExecutor", "build_gemini_prompt"]
