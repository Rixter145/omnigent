"""One-turn implementation adapter for Cursor WSL's verified result stream."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

from omnigent.cursor_wsl import CursorWslError, CursorWslResult, CursorWslTransport

from .async_utils import run_sync_on_thread
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
MAX_ERROR_CHARS = 2_000

_TOKEN_RE = re.compile(
    r"(?:ya29\.[A-Za-z0-9._-]+|1//[A-Za-z0-9._-]+|AIza[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]+)"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)((?:\b|_)(?:access_token|refresh_token|id_token|authorization|api[_-]?key|token)\b\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)


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


def build_cursor_wsl_prompt(messages: list[Message], system_prompt: str) -> str:
    """Compose bounded context for Cursor's fresh one-turn process."""
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


def _redact_error(value: object) -> str:
    """Bound and redact provider-shaped values before exposing an error."""
    text = str(value)[:MAX_ERROR_CHARS]
    text = _SECRET_ASSIGNMENT_RE.sub(r"\1<redacted>", text)
    text = _TOKEN_RE.sub("<redacted>", text)
    return text[:MAX_ERROR_CHARS]


def _session_key(messages: list[Message]) -> str:
    if messages:
        last = messages[-1]
        if last.get("session_id"):
            return str(last["session_id"])
        metadata = last.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("session_id"):
            return str(metadata["session_id"])
    return "__default__"


class CursorWslExecutor(Executor):
    """Run one implementation turn; transport accepts one successful result terminal."""

    def __init__(
        self,
        distro: str | None,
        user: str | None,
        working_directory: str | Path | None,
        *,
        transport: CursorWslTransport | None = None,
        timeout_s: float = 120.0,
        model: str | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        self._distro = distro
        self._user = user
        self._working_directory = working_directory
        self._transport = transport
        self._timeout_s = timeout_s
        self._model = model
        self._cancel_events: dict[str, threading.Event] = {}

    def supports_streaming(self) -> bool:
        # Cursor's NDJSON is collected by the synchronous WSL transport before
        # this adapter yields, so claiming token streaming would be misleading.
        return False

    def supports_tool_calling(self) -> bool:
        return False

    def _get_transport(self) -> CursorWslTransport:
        if self._transport is not None:
            return self._transport
        if not self._distro:
            raise CursorWslError(
                "distro-missing",
                "Cursor WSL is unavailable: configure an explicit WSL distro.",
            )
        if not self._user:
            raise CursorWslError(
                "user-missing",
                "Cursor WSL is unavailable: configure an explicit non-root Linux user.",
            )
        if self._working_directory is None:
            raise CursorWslError(
                "workspace-missing",
                "Cursor WSL is unavailable: configure an absolute Windows workspace path.",
            )
        self._transport = CursorWslTransport(self._distro, self._user, self._working_directory)
        return self._transport

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
        if session_key in self._cancel_events:
            yield ExecutorError(message="Cursor WSL already has an active turn for this session")
            return
        cancel_event = threading.Event()
        self._cancel_events[session_key] = cancel_event
        try:
            prompt = build_cursor_wsl_prompt(messages, system_prompt)
            if not prompt.strip():
                yield ExecutorError(message="Cursor WSL requires a non-empty prompt")
                return
            result: CursorWslResult = await run_sync_on_thread(
                self._get_transport().run,
                prompt,
                timeout=self._timeout_s,
                cancel=cancel_event,
                model=(config.model if config is not None and config.model else self._model),
            )
            if result.cancelled:
                yield TurnCancelled(reason="Cursor WSL turn cancelled")
                return
            if not result.ok:
                yield ExecutorError(message=result.error or "Cursor WSL turn failed")
                return
            if result.text:
                yield TextChunk(text=result.text)
            yield TurnComplete(response=result.text)
        except asyncio.CancelledError:
            cancel_event.set()
            raise
        except (CursorWslError, OSError, RuntimeError, ValueError, TypeError) as exc:
            yield ExecutorError(message=_redact_error(exc))
        finally:
            if self._cancel_events.get(session_key) is cancel_event:
                self._cancel_events.pop(session_key, None)


CursorWslSubscriptionExecutor = CursorWslExecutor

__all__ = ["CursorWslExecutor", "CursorWslSubscriptionExecutor", "build_cursor_wsl_prompt"]
