"""Focused tests for the subscription-backed routing judge."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from omnigent.server.subscription_routing import (
    MAX_ERROR_CHARS,
    MAX_PROMPT_CHARS,
    MAX_RUBRIC_CHARS,
    ClaudeSubscriptionTransport,
    CodexSubscriptionTransport,
    SubscriptionRoutingClient,
    SubscriptionTransportError,
    build_bounded_rubric,
)

MENU = {
    "claude-sdk": ["claude-fast", "claude-strong"],
    "codex": ["codex-fast", "codex-strong"],
}


class FakeJudge:
    def __init__(self, provider: str, response: str | Exception) -> None:
        self.provider = provider
        self.judge_model = f"{provider}-test"
        self.response = response
        self.prompts: list[str] = []

    async def judge(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.asyncio
async def test_codex_is_first_and_claude_is_not_called_when_it_succeeds() -> None:
    codex = FakeJudge("codex", '{"harness":"codex","model":"codex-strong","rationale":"r"}')
    claude = FakeJudge(
        "claude", '{"harness":"claude-sdk","model":"claude-strong","rationale":"r"}'
    )

    result = await SubscriptionRoutingClient([codex, claude]).route("build it", MENU)

    assert result is not None and result.model == "codex-strong"
    assert len(codex.prompts) == 1
    assert claude.prompts == []


@pytest.mark.asyncio
async def test_failed_codex_falls_back_to_claude_in_order() -> None:
    codex = FakeJudge("codex", RuntimeError("offline"))
    claude = FakeJudge(
        "claude", '{"harness":"claude-sdk","model":"claude-strong","rationale":"r"}'
    )
    client = SubscriptionRoutingClient([codex, claude])

    result = await client.route("task", MENU)

    assert result is not None and result.model == "claude-strong"
    assert [attempt["provider"] for attempt in client.last_attempts] == ["codex", "claude"]
    assert client.last_attempts[0]["status"] == "failed"
    assert client.last_attempts[1]["status"] == "accepted"
    assert client.last_error is None


@pytest.mark.asyncio
async def test_timeout_falls_back_and_is_recorded() -> None:
    codex = FakeJudge("codex", asyncio.TimeoutError())
    claude = FakeJudge("claude", '{"harness":"claude-sdk","model":"claude-fast","rationale":"r"}')
    client = SubscriptionRoutingClient([codex, claude], timeout_s=0.1)

    result = await client.route("task", MENU)

    assert result is not None and result.model == "claude-fast"
    assert client.last_attempts[0]["status"] == "timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    ["not json", '```json\n{"model": "codex-fast"}\n```', '{"model":"codex-fast"}'],
)
async def test_malformed_output_falls_back(response: str) -> None:
    codex = FakeJudge("codex", response)
    claude = FakeJudge("claude", '{"harness":"claude-sdk","model":"claude-fast","rationale":"r"}')
    client = SubscriptionRoutingClient([codex, claude])

    result = await client.route("task", MENU)

    assert result is not None and result.model == "claude-fast"
    assert [attempt["status"] for attempt in client.last_attempts] == ["failed", "accepted"]


@pytest.mark.asyncio
async def test_codex_unknown_model_falls_back_to_claude() -> None:
    codex = FakeJudge("codex", '{"harness":"codex","model":"made-up","rationale":"r"}')
    claude = FakeJudge("claude", '{"harness":"claude-sdk","model":"claude-fast","rationale":"r"}')
    client = SubscriptionRoutingClient([codex, claude])

    result = await client.route("task", MENU)

    assert result is not None and result.model == "claude-fast"
    assert [attempt["provider"] for attempt in client.last_attempts] == ["codex", "claude"]
    assert client.last_attempts[0]["status"] == "failed"
    assert client.last_judge == {"provider": "claude", "judge_model": "claude-test"}


@pytest.mark.asyncio
async def test_exhausted_out_of_menu_judges_do_not_pick_first_candidate() -> None:
    codex = FakeJudge("codex", '{"harness":"codex","model":"not-offered","rationale":"r"}')
    claude = FakeJudge(
        "claude", '{"harness":"claude-sdk","model":"also-not-offered","rationale":"r"}'
    )
    client = SubscriptionRoutingClient([codex, claude])

    result = await client.route("task", MENU)

    assert result is None
    assert client.last_attempts[0]["status"] == "failed"
    assert client.last_attempts[1]["status"] == "failed"
    assert client.last_error is not None
    assert "outside the closed candidate menu" in client.last_error


@pytest.mark.asyncio
async def test_mismatched_harness_model_pair_falls_back() -> None:
    codex = FakeJudge("codex", '{"harness":"claude-sdk","model":"codex-strong","rationale":"r"}')
    claude = FakeJudge(
        "claude", '{"harness":"claude-sdk","model":"claude-fast","rationale":"fallback"}'
    )
    client = SubscriptionRoutingClient([codex, claude])

    result = await client.route("task", MENU)

    assert result is not None
    assert (result.harness, result.model) == ("claude-sdk", "claude-fast")
    assert [attempt["status"] for attempt in client.last_attempts] == ["failed", "accepted"]
    assert client.last_judge == {"provider": "claude", "judge_model": "claude-test"}
    assert client.last_validation is not None
    assert client.last_validation["in_candidate_menu"] is True
    assert client.last_validation["harness_reconciled"] is False


@pytest.mark.asyncio
async def test_empty_menu_fails_closed_without_calling_judges() -> None:
    judge = FakeJudge("codex", '{"harness":"codex","model":"codex-fast","rationale":"r"}')
    client = SubscriptionRoutingClient([judge])

    result = await client.route("task", {})

    assert result is None
    assert judge.prompts == []
    assert client.last_error == "no candidate models were available"
    assert client.last_metadata["candidate_count"] == 0


@pytest.mark.asyncio
async def test_exhaustion_returns_none_and_bounded_error_metadata() -> None:
    huge = "provider failure " + ("x" * 2_000)
    codex = FakeJudge("codex", SubscriptionTransportError(huge))
    claude = FakeJudge("claude", RuntimeError(huge))
    client = SubscriptionRoutingClient([codex, claude])

    assert await client.route("task", MENU) is None
    assert client.last_error is not None
    assert len(client.last_error) <= MAX_ERROR_CHARS
    assert client.last_metadata["error"] == client.last_error
    assert len(client.last_metadata["attempts"]) == 2


@pytest.mark.asyncio
async def test_prompt_contains_only_bounded_task_and_closed_menu() -> None:
    judge = FakeJudge("codex", '{"harness":"codex","model":"codex-fast","rationale":"r"}')
    message = "m" * 100_000

    await SubscriptionRoutingClient([judge]).route(message, MENU)

    assert len(judge.prompts[0]) <= MAX_PROMPT_CHARS
    assert "claude-fast" in judge.prompts[0]
    assert "codex-strong" in judge.prompts[0]
    assert "m" * 4_000 in judge.prompts[0]


@pytest.mark.asyncio
async def test_allowance_is_filtered_to_the_menu_and_only_described_as_tie_break() -> None:
    judge = FakeJudge("codex", '{"harness":"codex","model":"codex-fast","rationale":"r"}')
    client = SubscriptionRoutingClient(
        [judge], soft_allowance_models=("claude-strong", "not-offered")
    )

    await client.route("task", MENU)

    prompt = judge.prompts[0]
    assert "soft tie-break only" in prompt
    assert "after task capability and cross-vendor fit" in prompt
    assert "claude-strong" in prompt
    assert "not-offered" not in prompt
    assert client.last_metadata["allowance"] == {
        "policy": "soft tie-break after capability and cross-vendor fit",
        "preferred_models": ["claude-strong"],
    }


def test_rubric_is_bounded_for_large_catalog() -> None:
    menu = {"codex": [f"model-{i}-" + "x" * 180 for i in range(200)]}

    rubric = build_bounded_rubric(menu)

    assert len(rubric) <= MAX_RUBRIC_CHARS
    assert "strict JSON" in rubric


def test_subscription_commands_are_argument_arrays() -> None:
    assert CodexSubscriptionTransport().command_args() == [
        "codex",
        "-c",
        'model_provider="openai"',
        "exec",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--color",
        "never",
        "--json",
        "--skip-git-repo-check",
    ]
    assert ClaudeSubscriptionTransport().command_args() == [
        "claude",
        "--print",
        "--output-format",
        "text",
        "--setting-sources",
        "",
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--no-session-persistence",
    ]


class _FakeStream:
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def read(self, _size: int) -> bytes:
        data, self.data = self.data, b""
        return data


class _FakeStdin:
    def __init__(self) -> None:
        self.data = b""

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeProcess:
    def __init__(
        self,
        stdout: bytes = b'{"harness":"codex","model":"codex-fast","rationale":"r"}',
    ) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(b"diagnostic")
        self.returncode = 0
        self.killed = False

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


@pytest.mark.asyncio
async def test_command_transport_scrubs_provider_api_key_and_bounds_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    process = _FakeProcess()

    async def fake_exec(*args: str, **kwargs: Any) -> _FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-child")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "also-must-not-enter-child")
    monkeypatch.setenv("GEMINI_API_KEY", "also-must-not-enter-child")
    monkeypatch.setenv("SOME_FUTURE_API_KEY", "also-must-not-enter-child")
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "must-not-enter-child")
    monkeypatch.setenv("AWS_CONTAINER_AUTHORIZATION_TOKEN", "must-not-enter-child")
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", "must-not-enter-child")
    monkeypatch.setenv("CLOUDSDK_AUTH_ACCESS_TOKEN", "must-not-enter-child")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "must-not-enter-child")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "must-not-enter-child")
    monkeypatch.setenv("WSLENV", "must-not-enter-child")
    monkeypatch.setenv("CODEX_HOME", "selected-login-home")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    output = await CodexSubscriptionTransport(stdout_limit=512, stderr_limit=256).judge("prompt")

    assert output.startswith("{")
    assert captured["args"] == (
        "codex",
        "-c",
        'model_provider="openai"',
        "exec",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--color",
        "never",
        "--json",
        "--skip-git-repo-check",
    )
    assert captured["kwargs"]["env"].get("OPENAI_API_KEY") is None
    assert captured["kwargs"]["env"].get("ANTHROPIC_API_KEY") is None
    assert captured["kwargs"]["env"].get("GEMINI_API_KEY") is None
    assert captured["kwargs"]["env"].get("SOME_FUTURE_API_KEY") is None
    for name in (
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "AZURE_OPENAI_ENDPOINT",
        "WSLENV",
    ):
        assert name not in captured["kwargs"]["env"]
    assert captured["kwargs"]["env"]["CODEX_HOME"] == "selected-login-home"
    assert captured["kwargs"]["stdin"] is asyncio.subprocess.PIPE
    assert "shell" not in captured["kwargs"]
    assert Path(captured["kwargs"]["cwd"]).exists() is False


def test_codex_command_has_no_write_or_color_defaults() -> None:
    args = CodexSubscriptionTransport().command_args()

    assert "--sandbox" in args and args[args.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in args
    assert "--color" in args and args[args.index("--color") + 1] == "never"
    assert not set(args) & {
        "workspace-write",
        "danger-full-access",
        "--full-auto",
        "--dangerously-bypass-approvals-and-sandbox",
    }


@pytest.mark.asyncio
async def test_codex_jsonl_transport_returns_final_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(
        b'{"type":"thread.started"}\n'
        b'{"type":"item.completed","item":{"type":"agent_message",'
        b'"text":"{\\"harness\\":\\"codex\\",\\"model\\":\\"codex-fast\\",'
        b'\\"rationale\\":\\"r\\"}"}}\n'
    )

    async def fake_exec(*_args: str, **_kwargs: Any) -> _FakeProcess:
        return process

    # The implementation must consume the machine-readable stream without
    # passing the event wrapper to the strict verdict parser.
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    output = await CodexSubscriptionTransport().judge("prompt")

    assert output == '{"harness":"codex","model":"codex-fast","rationale":"r"}'


@pytest.mark.asyncio
async def test_claude_transport_scrubs_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_exec(*_args: str, **kwargs: Any) -> _FakeProcess:
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-enter-child")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    await ClaudeSubscriptionTransport().judge("prompt")

    assert captured["env"].get("ANTHROPIC_API_KEY") is None
    assert Path(captured["cwd"]).exists() is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conflict",
    [
        "managed api-key helper",
        "managed gateway",
        "managed Bedrock provider",
        "managed Vertex provider",
    ],
)
async def test_claude_transport_refuses_managed_redirect_before_launch(
    conflict: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed redirect policy must stop a subscription judge before spawning."""
    launched = False

    async def fake_exec(*_args: str, **_kwargs: Any) -> _FakeProcess:
        nonlocal launched
        launched = True
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        "omnigent.onboarding.ambient.claude_managed_subscription_conflicts",
        lambda: (conflict,),
    )

    with pytest.raises(SubscriptionTransportError, match=conflict):
        await ClaudeSubscriptionTransport().judge("prompt")
    assert launched is False


def test_claude_command_disables_tools_mcp_and_persistence() -> None:
    args = ClaudeSubscriptionTransport().command_args()

    assert "--tools" in args and args[args.index("--tools") + 1] == ""
    assert "--setting-sources" in args and args[args.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in args
    assert "--no-session-persistence" in args
    assert "--dangerously-skip-permissions" not in args


@pytest.mark.asyncio
async def test_rationale_and_error_receipts_redact_credential_shapes() -> None:
    secret = "sk-ant-api03-abcdefghijklmnop"
    accepted = FakeJudge(
        "codex",
        '{"harness":"codex","model":"codex-fast",'
        f'"rationale":"used {secret} Bearer eyJabcdefghijklmnop"}}',
    )
    client = SubscriptionRoutingClient([accepted])

    result = await client.route("task", MENU)

    assert result is not None
    assert secret not in result.rationale
    assert "[REDACTED]" in result.rationale

    failed = FakeJudge("codex", RuntimeError(f"api_key={secret}"))
    failed_client = SubscriptionRoutingClient([failed])
    assert await failed_client.route("task", MENU) is None
    assert failed_client.last_error is not None
    assert secret not in failed_client.last_error
    assert "[REDACTED]" in failed_client.last_error
