from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from omnigent.gemini_cli import (
    EventKind,
    GeminiCliTransport,
    GeminiExecutable,
    ReadinessState,
    UnsupportedAuthMode,
    UnsupportedLauncher,
    _default_process_factory,
    build_gemini_invocation,
    discover_gemini_cli,
    gemini_readiness,
    parse_gemini_stream,
    subscription_environment,
)


def test_discovery_finds_official_windows_launchers_and_rejects_agy() -> None:
    seen: list[str] = []

    def which(name: str) -> str | None:
        seen.append(name)
        return r"C:\Users\test\AppData\Roaming\npm\gemini.cmd" if name == "gemini.cmd" else None

    found = discover_gemini_cli(which=which, platform="nt")

    assert found == GeminiExecutable(r"C:\Users\test\AppData\Roaming\npm\gemini.cmd", "cmd")
    assert "agy" not in seen


def test_discovery_resolves_official_windows_npm_shim_without_cmd(
    tmp_path: Path,
) -> None:
    npm = tmp_path / "npm"
    shim = npm / "gemini.cmd"
    entrypoint = npm / "node_modules" / "@google" / "gemini-cli" / "bundle" / "gemini.js"
    node = tmp_path / "node.exe"
    entrypoint.parent.mkdir(parents=True)
    shim.write_text("npm wrapper must not be parsed", encoding="utf-8")
    entrypoint.write_text("#!/usr/bin/env node", encoding="utf-8")
    node.write_bytes(b"")

    def which(name: str) -> str | None:
        if name == "gemini":
            return str(shim)
        if name == "node.exe":
            return str(node)
        return None

    found = discover_gemini_cli(which=which, platform="nt")
    assert found == GeminiExecutable(str(entrypoint), "node", str(node))

    invocation = build_gemini_invocation(
        'quoted "prompt" & %PATH% !VAR!\nnext',
        executable=found,
        platform="nt",
    )
    assert invocation.shell is False
    assert invocation.argv[:2] == (str(node), str(entrypoint))
    assert invocation.argv[2:4] == ("-p", 'quoted "prompt" & %PATH% !VAR!\nnext')


def test_discovery_supports_posix_binary_and_override_cannot_be_agy() -> None:
    found = discover_gemini_cli(
        which=lambda name: "/usr/local/bin/gemini" if name == "gemini" else None,
        platform="posix",
    )
    assert found == GeminiExecutable("/usr/local/bin/gemini")
    assert (
        discover_gemini_cli(
            environ={"OMNIGENT_GEMINI_PATH": "/usr/bin/agy"},
            which=lambda _: "/usr/bin/agy",
            platform="posix",
        )
        is None
    )


def test_readiness_distinguishes_installed_auth_present_and_verified(tmp_path: Path) -> None:
    executable = GeminiExecutable("gemini")
    assert (
        gemini_readiness(executable=executable, auth_marker=tmp_path / "missing").state
        is ReadinessState.INSTALLED
    )
    marker = tmp_path / "oauth_creds.json"
    marker.write_text("opaque", encoding="utf-8")
    present = gemini_readiness(executable=executable, auth_marker=marker)
    assert present.state is ReadinessState.AUTH_PRESENT
    verified = gemini_readiness(
        executable=executable, auth_marker=marker, auth_probe=lambda _: True
    )
    assert verified.state is ReadinessState.AUTH_VERIFIED
    assert verified.auth_marker == str(marker)


def test_readiness_never_reads_oauth_contents(tmp_path: Path) -> None:
    marker = tmp_path / "oauth_creds.json"
    marker.write_text('{"refresh_token":"DO-NOT-READ"}', encoding="utf-8")
    readiness = gemini_readiness(executable=GeminiExecutable("gemini"), auth_marker=marker)
    assert readiness.auth_present
    assert "DO-NOT-READ" not in repr(readiness)


def test_subscription_env_scrubs_api_key_and_vertex_selectors() -> None:
    env = subscription_environment(
        {
            "PATH": "x",
            "GEMINI_API_KEY": "AIza-secret",
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GOOGLE_APPLICATION_CREDENTIALS": "secret.json",
            "VERTEXAI_PROJECT": "project",
            "aws_web_identity_token_file": "secret.json",
            "AWS_CONTAINER_AUTHORIZATION_TOKEN": "token",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI": "uri",
            "CLOUDSDK_AUTH_ACCESS_TOKEN": "token",
            "AZURE_OPENAI_ENDPOINT": "endpoint",
            "WSLENV": "GOOGLE_API_KEY/u",
            "FUTURE_VENDOR_API_KEY": "secret",
        }
    )
    assert env == {"PATH": "x"}


def test_invocation_is_shell_free_and_keeps_modes_explicit() -> None:
    invocation = build_gemini_invocation(
        "say hi & do not shell out", executable=GeminiExecutable("gemini")
    )
    assert invocation.argv == (
        "gemini",
        "-p",
        "say hi & do not shell out",
        "--approval-mode",
        "plan",
        "--output-format",
        "stream-json",
    )
    assert invocation.shell is False
    with pytest.raises(UnsupportedAuthMode):
        build_gemini_invocation(
            "hello", executable=GeminiExecutable("gemini"), auth_mode="api_key"
        )
    with pytest.raises(UnsupportedAuthMode):
        build_gemini_invocation("hello", executable=GeminiExecutable("gemini"), auth_mode="vertex")


@pytest.mark.parametrize("prompt", ["a & b", 'a "quoted"', "a %PATH%", "a !VAR!", "a\nwhoami"])
def test_cmd_launcher_fails_closed_without_prompt_command_string(prompt: str) -> None:
    with pytest.raises(UnsupportedLauncher):
        build_gemini_invocation(
            prompt,
            executable=GeminiExecutable(r"C:\tools\gemini.cmd", "cmd"),
            platform="nt",
            comspec=r"C:\Windows\System32\cmd.exe",
        )


def test_windows_powershell_launcher_remains_shell_false() -> None:
    ps = build_gemini_invocation(
        'quoted "prompt" & %PATH% !VAR!\nnext',
        executable=GeminiExecutable(r"C:\tools\gemini.ps1", "powershell"),
        platform="nt",
        powershell=r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    )
    assert ps.shell is False and ps.argv[:3] == (ps.argv[0], "-NoLogo", "-NoProfile")
    assert "-File" in ps.argv
    assert ps.argv[ps.argv.index("-p") + 1] == 'quoted "prompt" & %PATH% !VAR!\nnext'
    assert ps.argv[ps.argv.index("--approval-mode") + 1] == "plan"


def test_stream_parser_translates_text_tool_usage_error_and_completion() -> None:
    lines = [
        json.dumps({"type": "message", "role": "assistant", "content": "Hello "}),
        json.dumps(
            {
                "type": "tool_use",
                "tool_name": "read_file",
                "tool_id": "t1",
                "input": {"path": "a.txt"},
            }
        ),
        json.dumps(
            {
                "type": "tool_result",
                "tool_id": "t1",
                "status": "success",
                "output": "contents",
            }
        ),
        json.dumps(
            {
                "type": "usage",
                "usage": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 7,
                },
            }
        ),
        json.dumps({"type": "error", "error": "temporary token=ya29.secret"}),
        json.dumps(
            {
                "type": "result",
                "status": "success",
                "result": "done",
                "stats": {"total_tokens": 8},
            }
        ),
    ]
    parsed = parse_gemini_stream(lines)
    assert [event.kind for event in parsed.events] == [
        EventKind.TEXT,
        EventKind.TOOL,
        EventKind.TOOL,
        EventKind.USAGE,
        EventKind.ERROR,
        EventKind.COMPLETION,
    ]
    assert parsed.text == "done"
    assert parsed.usage == {"input_tokens": 4, "output_tokens": 3, "total_tokens": 8}
    assert parsed.completed and parsed.events[1].tool_name == "read_file"
    assert parsed.events[2].output == "contents"
    assert "ya29.secret" not in parsed.events[4].error


def test_stream_parser_bounds_malformed_and_oversized_lines() -> None:
    parsed = parse_gemini_stream([b"not json\n", b"{" + b"x" * 100 + b"}"], max_line_bytes=32)
    assert parsed.malformed_lines == 1
    assert parsed.oversized_lines == 1
    assert all(len(event.error or "") < 200 for event in parsed.events)


def test_stream_parser_caps_accumulated_text_and_redacts_secret_fields() -> None:
    lines = [
        json.dumps(
            {
                "type": "tool_use",
                "tool_name": "inspect",
                "parameters": {"access_token": "arbitrary-oauth-value"},
            }
        ),
        json.dumps({"type": "message", "role": "assistant", "content": "x" * 40_000}),
    ]
    parsed = parse_gemini_stream(lines)
    assert len(parsed.text) == 32_768
    assert parsed.events[0].input == {"access_token": "<redacted>"}


@pytest.mark.asyncio
async def test_default_process_factory_disables_shell_and_bounds_reader(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def spawn(*argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
    await _default_process_factory(("gemini", "-p", "hello"), {"PATH": "x"})
    assert captured["shell"] is False
    assert captured["env"] == {"PATH": "x"}
    assert captured["limit"] == 65_537


class _FakeStream:
    def __init__(self, lines: list[bytes] | None = None, stderr: bytes = b"") -> None:
        self.lines = iter(lines or [])
        self.stderr = stderr

    async def readline(self) -> bytes:
        try:
            return next(self.lines)
        except StopIteration:
            return b""

    async def read(self, _: int) -> bytes:
        return self.stderr


class _FakeProcess:
    def __init__(
        self, lines: list[bytes] | None = None, *, exit_code: int = 0, stderr: bytes = b""
    ) -> None:
        self.stdout = _FakeStream(lines)
        self.stderr = _FakeStream(stderr=stderr)
        self.returncode: int | None = None
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        self.returncode = self.exit_code
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


@pytest.mark.asyncio
async def test_transport_uses_injected_process_and_cleans_up() -> None:
    captured: dict[str, object] = {}
    process = _FakeProcess([b'{"type":"result","status":"success","result":"ok"}\n'])

    async def factory(argv, env):
        captured["argv"] = argv
        captured["env"] = env
        return process

    result = await GeminiCliTransport(
        executable="gemini",
        env={"GEMINI_API_KEY": "secret", "PATH": "x"},
        process_factory=factory,
    ).run("hello")
    assert result.completed and result.text == "ok" and result.exit_code == 0
    assert captured["argv"][1:5] == ("-p", "hello", "--approval-mode", "plan")
    assert captured["env"] == {"PATH": "x"}
    assert process.terminated is False


@pytest.mark.asyncio
async def test_transport_reports_unsupported_cmd_without_spawning() -> None:
    called = False

    async def factory(_, __):
        nonlocal called
        called = True
        raise AssertionError("unsupported launcher must not spawn")

    result = await GeminiCliTransport(
        executable=GeminiExecutable(r"C:\tools\gemini.cmd", "cmd"),
        process_factory=factory,
    ).run("a & b")
    assert result.unsupported_launcher
    assert "install gemini.exe" in result.error
    assert called is False


@pytest.mark.asyncio
async def test_transport_returns_bounded_redacted_failure() -> None:
    process = _FakeProcess([], exit_code=2, stderr=b"refresh_token=secret-value " + b"x" * 10000)

    async def factory(_, __):
        return process

    result = await GeminiCliTransport(executable="gemini", process_factory=factory).run("hello")
    assert result.error is not None
    assert "secret-value" not in result.error
    assert len(result.error) <= 2_000


@pytest.mark.asyncio
async def test_transport_timeout_terminates_child() -> None:
    process = _FakeProcess()

    async def factory(_, __):
        class HangingStream(_FakeStream):
            async def readline(self):
                await asyncio.sleep(10)
                return b""

        process.stdout = HangingStream()
        return process

    result = await GeminiCliTransport(executable="gemini", process_factory=factory).run(
        "hello", timeout_s=0.01
    )
    assert result.timed_out and process.terminated


@pytest.mark.asyncio
async def test_transport_cancel_event_terminates_child() -> None:
    process = _FakeProcess()
    cancel = asyncio.Event()

    async def factory(_, __):
        class HangingStream(_FakeStream):
            async def readline(self):
                await asyncio.sleep(10)
                return b""

        process.stdout = HangingStream()
        return process

    task = asyncio.create_task(
        GeminiCliTransport(executable="gemini", process_factory=factory).run(
            "hello", cancel_event=cancel, timeout_s=5
        )
    )
    await asyncio.sleep(0)
    cancel.set()
    result = await task
    assert result.cancelled and process.terminated


@pytest.mark.asyncio
async def test_external_task_cancellation_cleans_up_child() -> None:
    process = _FakeProcess()
    created = asyncio.Event()

    async def factory(_, __):
        class HangingStream(_FakeStream):
            async def readline(self):
                await asyncio.sleep(10)
                return b""

        process.stdout = HangingStream()
        created.set()
        return process

    task = asyncio.create_task(
        GeminiCliTransport(executable="gemini", process_factory=factory).run("hello", timeout_s=5)
    )
    await created.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated
