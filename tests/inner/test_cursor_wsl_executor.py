from __future__ import annotations

import asyncio
import threading

import pytest

from omnigent.cursor_wsl import CursorWslResult
from omnigent.inner.cursor_wsl_executor import (
    MAX_PROMPT_CHARS,
    CursorWslExecutor,
    build_cursor_wsl_prompt,
)
from omnigent.inner.executor import ExecutorError, TextChunk, TurnComplete


class FakeCursorTransport:
    def __init__(self, result: CursorWslResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def run(self, prompt: str, **kwargs: object) -> CursorWslResult:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.result


@pytest.mark.asyncio
async def test_executor_maps_text_completion_and_preserves_context() -> None:
    transport = FakeCursorTransport(CursorWslResult(ok=True, text="done"))
    executor = CursorWslExecutor(
        "Ubuntu-24.04",
        "ricar",
        r"C:\Work Space\repo",
        transport=transport,
        model="auto-smart",
    )

    assert executor.supports_streaming() is False

    events = [
        event
        async for event in executor.run_turn(
            [
                {"role": "assistant", "content": "prior"},
                {"role": "user", "content": "hello", "session_id": "s1"},
            ],
            [],
            "system",
        )
    ]

    assert [type(event) for event in events] == [TextChunk, TurnComplete]
    assert "system: system" in str(transport.calls[0]["prompt"])
    assert "assistant: prior" in str(transport.calls[0]["prompt"])
    assert transport.calls[0]["cancel"].__class__ is threading.Event
    assert transport.calls[0]["model"] == "auto-smart"


@pytest.mark.asyncio
async def test_missing_distro_is_actionably_unavailable_without_transport() -> None:
    events = [
        event
        async for event in CursorWslExecutor(None, "ricar", r"C:\repo").run_turn(
            [{"role": "user", "content": "hello"}], [], ""
        )
    ]

    assert len(events) == 1 and isinstance(events[0], ExecutorError)
    assert "explicit WSL distro" in events[0].message


@pytest.mark.asyncio
async def test_missing_user_is_actionably_unavailable_without_transport() -> None:
    events = [
        event
        async for event in CursorWslExecutor("Ubuntu", None, r"C:\repo").run_turn(
            [{"role": "user", "content": "hello"}], [], ""
        )
    ]

    assert len(events) == 1 and isinstance(events[0], ExecutorError)
    assert "non-root Linux user" in events[0].message


@pytest.mark.asyncio
async def test_transport_error_is_mapped() -> None:
    events = [
        event
        async for event in CursorWslExecutor(
            "Ubuntu",
            "ricar",
            r"C:\repo",
            transport=FakeCursorTransport(
                CursorWslResult(ok=False, error="cursor-auth-unavailable")
            ),
        ).run_turn([{"role": "user", "content": "hello"}], [], "")
    ]

    assert len(events) == 1 and isinstance(events[0], ExecutorError)
    assert events[0].message == "cursor-auth-unavailable"


@pytest.mark.asyncio
async def test_interrupt_sets_thread_cancel_event() -> None:
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    class BlockingTransport(FakeCursorTransport):
        def run(self, prompt: str, **kwargs: object) -> CursorWslResult:
            del prompt
            cancel = kwargs["cancel"]
            assert isinstance(cancel, threading.Event)
            loop.call_soon_threadsafe(started.set)
            while not cancel.is_set():
                pass
            return CursorWslResult(ok=False, cancelled=True, error="cancelled")

    executor = CursorWslExecutor(
        "Ubuntu", "ricar", r"C:\repo", transport=BlockingTransport(CursorWslResult(ok=True))
    )
    iterator = executor.run_turn(
        [{"role": "user", "content": "hello", "session_id": "s1"}], [], ""
    )
    task = asyncio.create_task(iterator.__anext__())
    await started.wait()
    assert await executor.interrupt_session("s1")
    event = await task
    assert event.__class__.__name__ == "TurnCancelled"


@pytest.mark.asyncio
async def test_same_session_retry_does_not_start_a_duplicate_turn() -> None:
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    class BlockingTransport(FakeCursorTransport):
        def run(self, prompt: str, **kwargs: object) -> CursorWslResult:
            self.calls.append({"prompt": prompt, **kwargs})
            cancel = kwargs["cancel"]
            assert isinstance(cancel, threading.Event)
            loop.call_soon_threadsafe(started.set)
            cancel.wait()
            return CursorWslResult(ok=False, cancelled=True, error="cancelled")

    transport = BlockingTransport(CursorWslResult(ok=True))
    executor = CursorWslExecutor("Ubuntu", "ricar", r"C:\repo", transport=transport)
    first = executor.run_turn([{"role": "user", "content": "first", "session_id": "s1"}], [], "")
    first_event = asyncio.create_task(first.__anext__())
    first_result = None
    try:
        await asyncio.wait_for(started.wait(), timeout=1)

        retry_events = [
            event
            async for event in executor.run_turn(
                [{"role": "user", "content": "retry", "session_id": "s1"}], [], ""
            )
        ]

        assert len(retry_events) == 1 and isinstance(retry_events[0], ExecutorError)
        assert "active turn" in retry_events[0].message
        assert len(transport.calls) == 1
    finally:
        await executor.interrupt_session("s1")
        try:
            first_result = await asyncio.wait_for(first_event, timeout=1)
        finally:
            if not first_event.done():
                first_event.cancel()
                await asyncio.gather(first_event, return_exceptions=True)

    assert first_result.__class__.__name__ == "TurnCancelled"


def test_prompt_does_not_drop_system_context() -> None:
    assert build_cursor_wsl_prompt([{"role": "user", "content": "hello"}], "policy") == (
        "system: policy\n\nuser: hello"
    )


def test_prompt_reserves_front_budget_for_system_context() -> None:
    history = [{"role": "user", "content": "x" * 16_384} for _ in range(32)]
    prompt = build_cursor_wsl_prompt(history, "keep this system instruction")

    assert prompt.startswith("system: keep this system instruction\n\n")
    assert len(prompt) <= MAX_PROMPT_CHARS


@pytest.mark.asyncio
async def test_exception_error_redacts_credential_shaped_text() -> None:
    class FailingTransport(FakeCursorTransport):
        def run(self, prompt: str, **kwargs: object) -> CursorWslResult:
            del prompt, kwargs
            raise RuntimeError("CURSOR_API_KEY=secret sk-live-secret")

    events = [
        event
        async for event in CursorWslExecutor(
            "Ubuntu", "ricar", r"C:\repo", transport=FailingTransport(CursorWslResult(ok=True))
        ).run_turn([{"role": "user", "content": "hello"}], [], "")
    ]

    assert len(events) == 1 and isinstance(events[0], ExecutorError)
    assert "secret" not in events[0].message
    assert "<redacted>" in events[0].message
