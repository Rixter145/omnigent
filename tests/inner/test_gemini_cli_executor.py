from __future__ import annotations

import asyncio

import pytest

from omnigent.gemini_cli import GeminiResult
from omnigent.inner.executor import ExecutorConfig, ExecutorError, TextChunk, TurnComplete
from omnigent.inner.gemini_cli_executor import (
    MAX_HISTORY_MESSAGES,
    MAX_MESSAGE_CHARS,
    GeminiCliExecutor,
    build_gemini_prompt,
)


class FakeGeminiTransport:
    def __init__(self, result: GeminiResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def run(self, prompt: str, **kwargs: object) -> GeminiResult:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.result


def test_prompt_preserves_system_and_bounded_history() -> None:
    messages = [
        {"role": "user", "content": f"message-{index}"}
        for index in range(MAX_HISTORY_MESSAGES + 2)
    ]
    prompt = build_gemini_prompt(messages, "follow the policy")

    assert "system: follow the policy" in prompt
    assert "message-0" not in prompt
    assert f"message-{MAX_HISTORY_MESSAGES + 1}" in prompt
    assert len(prompt) <= (MAX_HISTORY_MESSAGES + 1) * (MAX_MESSAGE_CHARS + 16)


@pytest.mark.asyncio
async def test_executor_maps_text_completion_and_model() -> None:
    transport = FakeGeminiTransport(
        GeminiResult(text="answer", completed=True, usage={"total_tokens": 3})
    )
    executor = GeminiCliExecutor(transport=transport, timeout_s=7)

    assert executor.supports_streaming() is False

    events = [
        event
        async for event in executor.run_turn(
            [{"role": "user", "content": "hello", "session_id": "s1"}],
            [],
            "system",
            ExecutorConfig(model="gemini-2.5-pro"),
        )
    ]

    assert [type(event) for event in events] == [TextChunk, TurnComplete]
    assert events[0].text == "answer"
    assert events[1].response == "answer"
    assert events[1].usage == {"total_tokens": 3}
    assert transport.calls[0]["model"] == "gemini-2.5-pro"
    assert transport.calls[0]["timeout_s"] == 7


@pytest.mark.asyncio
async def test_executor_maps_transport_error_without_raw_provider_payload() -> None:
    transport = FakeGeminiTransport(GeminiResult(error="auth token=ya29.secret", completed=False))
    events = [
        event
        async for event in GeminiCliExecutor(transport=transport).run_turn(
            [{"role": "user", "content": "hello"}], [], "", None
        )
    ]

    assert len(events) == 1 and isinstance(events[0], ExecutorError)
    assert "ya29.secret" not in events[0].message


@pytest.mark.asyncio
async def test_interrupt_sets_transport_cancel_event() -> None:
    started = asyncio.Event()

    class BlockingTransport(FakeGeminiTransport):
        async def run(self, prompt: str, **kwargs: object) -> GeminiResult:
            del prompt
            cancel_event = kwargs["cancel_event"]
            assert isinstance(cancel_event, asyncio.Event)
            started.set()
            while not cancel_event.is_set():
                await asyncio.sleep(0)
            return GeminiResult(cancelled=True)

    executor = GeminiCliExecutor(transport=BlockingTransport(GeminiResult()))
    task = asyncio.create_task(
        executor.run_turn(
            [{"role": "user", "content": "hello", "session_id": "s1"}], [], ""
        ).__anext__()
    )
    await started.wait()
    assert await executor.interrupt_session("s1")
    event = await task
    assert event.__class__.__name__ == "TurnCancelled"
