"""Focused tests for the shell-free Cursor WSL subscription transport."""

from __future__ import annotations

import io
import json
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from omnigent import cursor_wsl
from omnigent.cursor_wsl import (
    CommandResult,
    CursorWslError,
    CursorWslTransport,
    build_cursor_wsl_argv,
    parse_cursor_ndjson,
    parse_cursor_status,
    validate_distro_name,
    validate_linux_user,
    windows_path_to_wsl,
)


def test_distro_names_are_strict_and_shell_independent() -> None:
    assert validate_distro_name("Ubuntu-24.04") == "Ubuntu-24.04"
    for value in ("", " Ubuntu", "Ubuntu 24", "Ubuntu;id", "../x", "--help", "a/b"):
        with pytest.raises(CursorWslError, match="distro"):
            validate_distro_name(value)


def test_linux_user_is_strict_and_root_is_rejected() -> None:
    assert validate_linux_user("ricar") == "ricar"
    for value in ("", "Root", "root", "-ricar", "ricar;id", "ricar name", "a" * 33):
        with pytest.raises(CursorWslError, match=r"user|root"):
            validate_linux_user(value)


@pytest.mark.parametrize(
    ("windows", "expected"),
    [
        (r"C:\Work Space\repo", "/mnt/c/Work Space/repo"),
        ("D:/repo/child", "/mnt/d/repo/child"),
        ("C:\\", "/mnt/c"),
    ],
)
def test_absolute_drive_paths_translate_safely(windows: str, expected: str) -> None:
    assert windows_path_to_wsl(windows) == expected


@pytest.mark.parametrize(
    "path",
    [
        "relative/repo",
        r"\rooted-without-drive",
        "/already/posix",
        r"C:relative",
        r"C:\repo\..\secret",
        r"\\wsl$\Ubuntu\home\user",
        r"\\?\C:\repo",
        r"\\server\share\..\secret",
        r"C:\repo\file:name",
        r"C:\repo\\",
        r"\\server\share\folder with spaces",
    ],
)
def test_relative_or_ambiguous_paths_fail_closed(path: str) -> None:
    with pytest.raises(CursorWslError, match="path"):
        windows_path_to_wsl(path)


def test_build_argv_has_explicit_distro_cwd_and_no_shell_features() -> None:
    prompt = 'quote " ; $HOME && echo pwned\nsecond'
    argv = build_cursor_wsl_argv("Ubuntu", "ricar", r"C:\Work Space\repo", prompt)
    assert argv[:9] == [
        "wsl.exe",
        "--distribution",
        "Ubuntu",
        "--user",
        "ricar",
        "--cd",
        "/mnt/c/Work Space/repo",
        "--exec",
        "/usr/bin/env",
    ]
    assert argv[argv.index("cursor-agent") :] == [
        "cursor-agent",
        "--trust",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
    ]
    assert not any(value.lower() in {"--force", "-f", "--yolo", "yolo", "tmux"} for value in argv)
    assert "--mode" not in argv
    unset_names = argv[argv.index("/usr/bin/env") + 1 : argv.index("cursor-agent")]
    assert "WSLENV" in unset_names
    assert "AWS_WEB_IDENTITY_TOKEN_FILE" in unset_names
    assert "AZURE_OPENAI_ENDPOINT" in unset_names


def test_build_argv_threads_only_a_valid_model_as_data() -> None:
    argv = build_cursor_wsl_argv("Ubuntu", "ricar", r"C:\repo", "review", model="auto-smart")
    assert argv[argv.index("--model") + 1] == "auto-smart"
    assert "--force" not in argv
    with pytest.raises(CursorWslError, match="model"):
        build_cursor_wsl_argv("Ubuntu", "ricar", r"C:\repo", "review", model="--yolo")


def test_status_parser_reads_only_bounded_ordinary_auth_text() -> None:
    secret = "account@example.com"
    status = parse_cursor_status(f"Authenticated: yes\nAccount: {secret}\nEndpoint: private")
    assert status.authenticated is True
    assert status.known is True
    assert secret not in repr(status)
    assert parse_cursor_status(f"Not authenticated\nAccount: {secret}") == parse_cursor_status(
        "Not authenticated"
    )
    assert parse_cursor_status("Not authenticated").authenticated is False
    assert parse_cursor_status(f"Account: {secret}").known is False
    assert parse_cursor_status("Authenticated\nNot logged in").known is False
    assert parse_cursor_status("Authenticated: yes\n" + ("x" * 70_000)).known is False


def test_status_parser_accepts_current_logged_in_as_form_without_retaining_account() -> None:
    account = "synthetic-account@example.test"
    status = parse_cursor_status(f"Logged in as {account}")

    assert status == cursor_wsl.CursorWslStatus(authenticated=True, known=True)
    assert account not in repr(status)


def test_status_parser_normalizes_ansi_and_checkmark_for_current_login_form() -> None:
    assert parse_cursor_status("\x1b[32m✓ Logged in as synthetic-account@example.test\x1b[0m") == (
        cursor_wsl.CursorWslStatus(authenticated=True, known=True)
    )


@pytest.mark.parametrize(
    "output",
    [
        "Logged in as",
        "Logged in as ",
        "Logged in as \x1b[31m\x1b[0m",
        "Logged in as\nsynthetic-account@example.test",
        "Current user: synthetic-account@example.test",
    ],
)
def test_status_parser_rejects_malformed_or_unrelated_logged_in_as_text(output: str) -> None:
    assert parse_cursor_status(output) == cursor_wsl.CursorWslStatus(
        authenticated=False, known=False
    )


def test_status_parser_rejects_contradictory_current_login_form() -> None:
    output = "Logged in as synthetic-account@example.test\nNot authenticated"
    assert parse_cursor_status(output) == (
        cursor_wsl.CursorWslStatus(authenticated=False, known=False)
    )


def test_stream_projection_redacts_credential_shaped_text() -> None:
    parsed = parse_cursor_ndjson(
        '{"type":"error","text":"CURSOR_API_KEY=secret sk-live-secret"}\n'
    )
    assert parsed.events[0].text == "[redacted] [redacted]"


def test_readiness_uses_explicit_distro_and_never_returns_probe_output() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...], *, timeout: float) -> CommandResult:
        assert timeout == 5.0
        calls.append(argv)
        if argv[-2:] == ("uname", "-r"):
            return CommandResult(0, "5.15.153.1-microsoft-standard-WSL2\n")
        if argv[-3:] == ("-qx", "ID=ubuntu", "/etc/os-release"):
            return CommandResult(0)
        if argv[-2:] == ("id", "-u"):
            return CommandResult(0, "1000\n")
        if argv[-3:] == ("/usr/bin/test", "-d", "."):
            return CommandResult(0)
        if argv[-1] == "--version":
            return CommandResult(0)
        return CommandResult(0, "Authenticated: yes\nAccount: secret@example.com\n")

    transport = CursorWslTransport(
        "Ubuntu-24.04",
        "ricar",
        r"C:\Work Space\repo",
        which=lambda name: r"C:\Windows\System32\wsl.exe",
        command_runner=runner,
        clock=lambda: datetime(2026, 9, 4, tzinfo=timezone.utc),
    )
    readiness = transport.readiness()
    assert readiness.transport_ready is True
    assert readiness.auth_present is True
    assert readiness.last_verified == "2026-09-04T00:00:00+00:00"
    assert all(call[0].endswith("wsl.exe") for call in calls)
    assert all("Ubuntu-24.04" in call for call in calls)
    assert all(call[call.index("--user") + 1] == "ricar" for call in calls)
    assert [call[-2:] for call in calls[:3]] == [
        ("uname", "-r"),
        ("ID=ubuntu", "/etc/os-release"),
        ("id", "-u"),
    ]
    workspace_calls = [call for call in calls if call[-3:] == ("/usr/bin/test", "-d", ".")]
    assert len(workspace_calls) == 1
    workspace_call = workspace_calls[0]
    assert workspace_call[workspace_call.index("--cd") + 1] == "/mnt/c/Work Space/repo"
    cursor_calls = [call for call in calls if "cursor-agent" in call]
    assert all("/usr/bin/env" in call for call in cursor_calls)
    assert all(call[call.index("/usr/bin/env") + 1] == "-u" for call in cursor_calls)
    status_calls = [call for call in calls if call[-1] == "status"]
    assert len(status_calls) == 1
    assert "--format" not in status_calls[0]
    assert "secret" not in repr(readiness)


@pytest.mark.parametrize(
    ("failure_command", "error"),
    [
        (("uname", "-r"), "distro-unavailable"),
        (("uname", "-r"), "wsl2-required"),
        (("grep", "-qx", "ID=ubuntu", "/etc/os-release"), "ubuntu-required"),
        (("id", "-u"), "cursor-user-unavailable"),
        (("/usr/bin/test", "-d", "."), "workspace-unavailable"),
        ("--version", "cursor-agent-missing"),
        ("status", "cursor-auth-unavailable"),
    ],
)
def test_readiness_failures_are_truthful_and_fail_closed(
    failure_command: tuple[str, ...] | str, error: str
) -> None:
    def runner(argv: tuple[str, ...], *, timeout: float) -> CommandResult:
        del timeout
        command = tuple(argv[argv.index("--exec") + 1 :])
        if error == "wsl2-required" and command == ("uname", "-r"):
            return CommandResult(0, "5.15.0-microsoft-standard\n")
        if isinstance(failure_command, tuple) and command == failure_command:
            return CommandResult(1)
        if isinstance(failure_command, str) and command[-1] == failure_command:
            return CommandResult(1)
        if command == ("uname", "-r"):
            return CommandResult(0, "5.15.153.1-microsoft-standard-WSL2\n")
        if command == ("id", "-u"):
            return CommandResult(0, "1000\n")
        if command[-1] == "status":
            return CommandResult(0, "Not authenticated\nAccount: secret@example.com\n")
        return CommandResult(0)

    readiness = CursorWslTransport(
        "Ubuntu",
        "ricar",
        r"C:\repo",
        which=lambda _: "wsl.exe",
        command_runner=runner,
    ).readiness()
    assert readiness.transport_ready is False
    assert readiness.error == error


def test_missing_wsl_is_reported_without_process_launch() -> None:
    transport = CursorWslTransport("Ubuntu", "ricar", r"C:\repo", which=lambda _: None)
    readiness = transport.readiness()
    assert readiness.transport_ready is False
    assert readiness.error == "wsl-missing"
    assert transport.run("hello").error == "wsl-missing"


class FakeProcess:
    def __init__(
        self, lines: list[str], *, returncode: int | None = 0, block: bool = False
    ) -> None:
        self.stdout = io.StringIO("".join(lines)) if not block else BlockingStream()
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return (
            self.returncode
            if not isinstance(self.stdout, BlockingStream) or self.terminated
            else None
        )

    def terminate(self) -> None:
        self.terminated = True
        if isinstance(self.stdout, BlockingStream):
            self.stdout.release()
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self.returncode if self.returncode is not None else -15
        return self.returncode


class BlockingStream:
    def __init__(self) -> None:
        self.released = threading.Event()

    def readline(self) -> str:
        self.released.wait()
        return ""

    def release(self) -> None:
        self.released.set()


class UnknownExitProcess(FakeProcess):
    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> Any:
        del timeout
        return None


class EofBeforeExitProcess(FakeProcess):
    """Simulate wsl.exe closing stdout just before its exit status is available."""

    def __init__(self) -> None:
        super().__init__([json.dumps({"type": "result", "text": "ok"}) + "\n"], returncode=None)
        self.waited = False

    def poll(self) -> int | None:
        return 0 if self.waited else None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waited = True
        self.returncode = 0
        return 0


def test_run_parses_bounded_stream_into_stable_events() -> None:
    process = FakeProcess(
        [
            '{"type":"assistant","delta":"hello "}\n',
            '{"event":"result","result":"world"}\n',
            "not-json\n",
        ]
    )
    seen: list[tuple[str, ...]] = []
    transport = CursorWslTransport(
        "Ubuntu",
        "ricar",
        r"C:\repo",
        wsl_executable="wsl.exe",
        process_factory=lambda argv: seen.append(tuple(argv)) or process,
    )
    result = transport.run('"quoted"; $HOME', timeout=1)
    assert result.ok is True
    assert result.text == "hello world"
    assert [event.kind for event in result.events] == ["assistant", "result"]
    assert result.malformed_lines == 1
    assert seen[0][seen[0].index("-p") + 1] == '"quoted"; $HOME'


def test_run_allows_natural_exit_after_stdout_eof_before_poll_updates() -> None:
    process = EofBeforeExitProcess()
    transport = CursorWslTransport(
        "Ubuntu",
        "ricar",
        r"C:\repo",
        wsl_executable="wsl.exe",
        process_factory=lambda _: process,
    )

    result = transport.run("hello")

    assert result.ok is True
    assert result.exit_code == 0
    assert process.waited is True
    assert process.terminated is False


@pytest.mark.parametrize(
    ("lines", "error"),
    [
        ([], "missing-terminal-result"),
        ([json.dumps({"type": "error", "text": "failed"}) + "\n"], "missing-terminal-result"),
        (["not-json\n"], "invalid-ndjson"),
        (
            [
                json.dumps({"type": "result", "text": "first"}) + "\n",
                json.dumps({"type": "result", "text": "second"}) + "\n",
            ],
            "duplicate-terminal-result",
        ),
        ([json.dumps({"type": "assistant", "text": "partial"}) + "\n"], "missing-terminal-result"),
    ],
)
def test_run_requires_exactly_one_successful_terminal_result(lines: list[str], error: str) -> None:
    transport = CursorWslTransport(
        "Ubuntu",
        "ricar",
        r"C:\repo",
        wsl_executable="wsl.exe",
        process_factory=lambda _: FakeProcess(lines),
    )

    result = transport.run("hello")

    assert result.ok is False
    assert result.error == error


def test_result_terminal_with_explicit_failure_is_not_success() -> None:
    transport = CursorWslTransport(
        "Ubuntu",
        "ricar",
        r"C:\repo",
        wsl_executable="wsl.exe",
        process_factory=lambda _: FakeProcess(
            [json.dumps({"type": "result", "success": False, "text": "failed"}) + "\n"]
        ),
    )

    result = transport.run("hello")

    assert result.ok is False
    assert result.error == "missing-terminal-result"


def test_run_timeout_terminates_process_and_returns_bounded_error() -> None:
    process = FakeProcess([], block=True)
    transport = CursorWslTransport(
        "Ubuntu", "ricar", r"C:\repo", wsl_executable="wsl.exe", process_factory=lambda _: process
    )
    result = transport.run("hang", timeout=0.03)
    assert result.ok is False
    assert result.timed_out is True
    assert result.error == "timeout"
    assert process.terminated is True


def test_run_rejects_truncated_ndjson_instead_of_returning_partial_success() -> None:
    process = FakeProcess(['{"type":"result","text":"this is too large"}\n'])
    transport = CursorWslTransport(
        "Ubuntu",
        "ricar",
        r"C:\repo",
        wsl_executable="wsl.exe",
        output_limit=16,
        process_factory=lambda _: process,
    )
    result = transport.run("hello")
    assert result.ok is False
    assert result.error == "output-too-large"
    assert result.truncated is True


def test_run_rejects_unknown_exit_status_instead_of_returning_success() -> None:
    process = UnknownExitProcess(['{"type":"result","text":"ok"}\n'], returncode=None)
    transport = CursorWslTransport(
        "Ubuntu",
        "ricar",
        r"C:\repo",
        wsl_executable="wsl.exe",
        process_factory=lambda _: process,
    )
    result = transport.run("hello")
    assert result.ok is False
    assert result.error == "process-exit-unknown"
    assert result.exit_code is None
    assert process.terminated is True


def test_run_cancellation_terminates_process_without_leaking_prompt_or_output() -> None:
    process = FakeProcess([], block=True)
    cancel = threading.Event()
    transport = CursorWslTransport(
        "Ubuntu", "ricar", r"C:\repo", wsl_executable="wsl.exe", process_factory=lambda _: process
    )
    worker_result: list[Any] = []

    def invoke() -> None:
        worker_result.append(
            transport.run("cancel; CURSOR_API_KEY=do-not-leak", timeout=2, cancel=cancel)
        )

    worker = threading.Thread(target=invoke)
    worker.start()
    time.sleep(0.03)
    cancel.set()
    worker.join(timeout=1)
    assert worker_result[0].cancelled is True
    assert process.terminated is True
    assert "do-not-leak" not in repr(worker_result[0])


def test_nonzero_and_malformed_output_errors_are_bounded_and_redacted() -> None:
    secret = "access_token=should-not-appear"
    process = FakeProcess([json.dumps({"type": "error", "text": secret}) + "\n"], returncode=7)
    transport = CursorWslTransport(
        "Ubuntu", "ricar", r"C:\repo", wsl_executable="wsl.exe", process_factory=lambda _: process
    )
    result = transport.run("hello")
    assert result.ok is False
    assert result.error == "cursor-agent exited with status 7"
    assert secret not in repr(result)


def test_process_factory_receives_argv_not_shell_text_and_no_api_key_env_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "host-secret")
    captured: list[tuple[str, ...]] = []

    class DoneProcess(FakeProcess):
        pass

    process = DoneProcess(['{"type":"result","text":"ok"}\n'])
    transport = CursorWslTransport(
        "Ubuntu",
        "ricar",
        r"C:\repo",
        wsl_executable="wsl.exe",
        process_factory=lambda argv: captured.append(tuple(argv)) or process,
    )
    assert transport.run("ok").ok is True
    assert isinstance(captured[0], tuple)
    assert captured[0][captured[0].index("--exec") + 1] == "/usr/bin/env"
    assert captured[0][captured[0].index("cursor-agent") - 2] == "-u"
    assert "host-secret" not in repr(captured)


def test_default_probe_and_process_seams_strip_api_keys_and_disable_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("cursor_api_key", "lower-secret")
    monkeypatch.setenv("FUTURE_VENDOR_API_KEY", "future-secret")
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "token-file")
    monkeypatch.setenv("AWS_CONTAINER_AUTHORIZATION_TOKEN", "token")
    monkeypatch.setenv("CLOUDSDK_AUTH_ACCESS_TOKEN", "token")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "endpoint")
    monkeypatch.setenv("WSLENV", "OPENAI_API_KEY/u")
    probe_call: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        probe_call.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="should-not-be-captured")

    monkeypatch.setattr(cursor_wsl.subprocess, "run", fake_run)
    result = cursor_wsl._run_command(["wsl.exe", "--version"], timeout=1)
    assert result.returncode == 0
    assert probe_call["shell"] is False
    assert probe_call["stderr"] is cursor_wsl.subprocess.DEVNULL
    env = probe_call["env"]
    assert isinstance(env, dict)
    assert "cursor_api_key" not in env
    assert "CURSOR_API_KEY" not in env
    assert "FUTURE_VENDOR_API_KEY" not in env
    for name in (
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "AZURE_OPENAI_ENDPOINT",
        "WSLENV",
    ):
        assert name not in env


def test_default_process_seam_uses_no_shell_and_discards_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    process = FakeProcess([])

    def fake_popen(argv: list[str], **kwargs: object) -> FakeProcess:
        seen.update(kwargs)
        assert argv == ["wsl.exe", "--exec", "cursor-agent"]
        return process

    monkeypatch.setattr(cursor_wsl.subprocess, "Popen", fake_popen)
    cursor_wsl._spawn_process(["wsl.exe", "--exec", "cursor-agent"])
    assert seen["shell"] is False
    assert seen["stderr"] is cursor_wsl.subprocess.DEVNULL
