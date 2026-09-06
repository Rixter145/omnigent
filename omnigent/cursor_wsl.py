"""Fail-closed Windows-to-WSL transport for Cursor's subscription CLI.

This adapter deliberately does not inspect Cursor credential files.  The
official ``cursor-agent login`` state stays inside the selected WSL distro;
the Windows side only asks the CLI for a bounded status result and launches
the CLI with a sanitized environment.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, TypeAlias

from omnigent.subscription_security import (
    subscription_blocked_env_names,
    subscription_environment,
)

MAX_ERROR_LENGTH = 240
MAX_NDJSON_BYTES = 1_048_576
MAX_NDJSON_EVENTS = 4_096
MAX_STATUS_BYTES = 64_000
MAX_PROMPT_LENGTH = 128_000
MAX_MODEL_LENGTH = 256
_NATURAL_EXIT_GRACE_SECONDS = 0.25

_DISTRO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_LINUX_USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\[\]-]{0,255}$")
_DRIVE_RE = re.compile(r"^(?P<drive>[A-Za-z]):(?P<slash>[\\/])(?P<rest>.*)$")
_UNSAFE_PATH_CHARS = frozenset('\x00\r\n\t<>:"|?*')
_SENSITIVE_VALUE_RE = re.compile(
    r"(?i)(?:\b[A-Z][A-Z0-9_]*(?:API_KEY|TOKEN)\b|api[_ -]?key|access[_ -]?token|"
    r"refresh[_ -]?token|password)\s*[:=]\s*\S+|\bbearer\s+\S+|\bsk-[A-Za-z0-9._-]+\b"
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_AUTHENTICATED_LINE_RE = re.compile(
    r"^(?:authenticated|logged[ -]in|signed[ -]in)(?:[.!])?$|"
    r"^(?:status|authentication(?: status)?|authenticated|login(?: status)?)\s*[:=-]\s*"
    r"(?:authenticated|logged[ -]in|signed[ -]in|true|yes|active|ok)(?:[.!])?$"
)
_LOGGED_IN_AS_LINE_RE = re.compile(r"^logged[ -]in as \S(?:.*\S)?$")
_NOT_AUTHENTICATED_LINE_RE = re.compile(
    r"^(?:not authenticated|unauthenticated|not logged[ -]in|logged[ -]out|"
    r"signed[ -]out|authentication required)(?:[.!])?$|"
    r"^(?:status|authentication(?: status)?|authenticated|login(?: status)?)\s*[:=-]\s*"
    r"(?:unauthenticated|not authenticated|not logged[ -]in|logged[ -]out|"
    r"signed[ -]out|false|no|required)(?:[.!])?$"
)
_SUCCESSFUL_TERMINAL_RESULT_TYPES = frozenset({"result"})
_FAILURE_RESULT_STATUSES = frozenset({"error", "failed", "failure", "cancelled", "canceled"})


class CursorWslError(RuntimeError):
    """A bounded, user-actionable transport error."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = _bounded_error(message)
        super().__init__(self.message)


def _bounded_error(message: str) -> str:
    """Return an error safe to put in a receipt or UI message."""
    # Error text is generated from fixed messages in this module.  The
    # replacement also protects callers that pass command diagnostics into a
    # future seam from accidentally returning credential-shaped values.
    redacted = _SENSITIVE_VALUE_RE.sub("[redacted]", str(message))
    redacted = re.sub(r"(?i)\b(?:sess|token)_[A-Za-z0-9._-]+\b", "[redacted]", redacted)
    redacted = " ".join(redacted.split())
    return redacted[:MAX_ERROR_LENGTH]


def validate_distro_name(distro: str) -> str:
    """Validate and return an explicit WSL distro name.

    WSL accepts names containing spaces, but accepting them here makes config
    mistakes and command-line lookalikes difficult to distinguish.  The
    conservative allowlist is also independent of shell quoting rules.
    """
    if not isinstance(distro, str) or _DISTRO_RE.fullmatch(distro) is None:
        raise CursorWslError(
            "invalid-distro",
            "WSL distro must be an explicit alphanumeric name using '.', '_' or '-'.",
        )
    if distro in {".", ".."} or distro.startswith("-"):
        raise CursorWslError("invalid-distro", "WSL distro name is not valid.")
    return distro


def validate_linux_user(user: str) -> str:
    """Validate an explicit, non-root POSIX username for WSL execution."""

    if not isinstance(user, str) or _LINUX_USER_RE.fullmatch(user) is None:
        raise CursorWslError(
            "invalid-user",
            "Cursor WSL user must use lowercase POSIX letters, digits, '_' or '-'.",
        )
    if user == "root":
        raise CursorWslError("root-user-forbidden", "Cursor WSL must not execute as root.")
    return user


def _path_parts(value: str, *, label: str) -> list[str]:
    """Validate Windows path components and return them without separators."""
    if any(char in _UNSAFE_PATH_CHARS for char in value):
        raise CursorWslError("invalid-path", f"{label} contains an unsafe path character.")
    parts = value.replace("\\", "/").split("/")
    if parts and parts[-1] == "":
        parts.pop()
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise CursorWslError("invalid-path", f"{label} contains an ambiguous path component.")
    if any("/" in part or "\\" in part for part in parts):  # defensive for future callers
        raise CursorWslError("invalid-path", f"{label} contains an ambiguous separator.")
    return parts


def windows_path_to_wsl(path: str | os.PathLike[str]) -> str:
    """Translate a strict absolute Windows drive or UNC path to a WSL path.

    Drive paths map to ``/mnt/<lowercase-drive>/...``.  Generic UNC paths are
    rejected because WSL does not guarantee a stable mount for them.  POSIX
    paths, rooted paths without a drive, traversal, and device paths are also
    rejected.
    """
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or any(ord(char) < 32 for char in raw):
        raise CursorWslError("invalid-path", "working directory must be an absolute Windows path.")

    drive_match = _DRIVE_RE.fullmatch(raw)
    if drive_match is not None:
        if drive_match.group("rest").startswith(("/", "\\")):
            raise CursorWslError("invalid-path", "working directory has repeated separators.")
        parts = (
            _path_parts(drive_match.group("rest"), label="working directory")
            if drive_match.group("rest")
            else []
        )
        suffix = "/".join(parts)
        return f"/mnt/{drive_match.group('drive').lower()}" + (f"/{suffix}" if suffix else "")

    normalized = raw.replace("\\", "/")
    if normalized.startswith("//"):
        raise CursorWslError(
            "unc-unavailable",
            "generic UNC paths have no guaranteed WSL mapping.",
        )

    raise CursorWslError(
        "invalid-path", "working directory must be an absolute drive or UNC path."
    )


# Short aliases make the path contract easy to discover for callers.
translate_windows_path = windows_path_to_wsl


def _validate_prompt(prompt: str) -> str:
    if not isinstance(prompt, str) or "\x00" in prompt:
        raise CursorWslError("invalid-prompt", "prompt must be text without NUL characters.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise CursorWslError("prompt-too-large", "prompt exceeds the bounded transport limit.")
    return prompt


def _validate_model(model: str | None) -> str | None:
    """Validate an optional Cursor model id before it reaches WSL argv."""

    if model is None:
        return None
    normalized = model.strip()
    if not normalized or len(normalized) > MAX_MODEL_LENGTH or not _MODEL_RE.fullmatch(normalized):
        raise CursorWslError("invalid-model", "Cursor model id is not valid.")
    return normalized


def build_cursor_wsl_argv(
    distro: str,
    user: str,
    working_directory: str | os.PathLike[str],
    prompt: str,
    *,
    wsl_executable: str = "wsl.exe",
    model: str | None = None,
) -> list[str]:
    """Build a shell-free WSL argv for one headless subscription turn."""
    distro = validate_distro_name(distro)
    user = validate_linux_user(user)
    wsl_cwd = windows_path_to_wsl(working_directory)
    prompt = _validate_prompt(prompt)
    model = _validate_model(model)
    if not wsl_executable or any(ord(char) < 32 for char in wsl_executable):
        raise CursorWslError("invalid-wsl", "WSL executable is not valid.")
    argv = [
        wsl_executable,
        "--distribution",
        distro,
        "--user",
        user,
        "--cd",
        wsl_cwd,
        "--exec",
        "/usr/bin/env",
        *[part for name in sorted(subscription_blocked_env_names()) for part in ("-u", name)],
        "cursor-agent",
        "--trust",
        "-p",
        prompt,
    ]
    if model is not None:
        argv.extend(("--model", model))
    argv.extend(("--output-format", "stream-json"))
    return argv


build_cursor_argv = build_cursor_wsl_argv


@dataclass(frozen=True)
class CommandResult:
    """Small process result used by the injected probe seam."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner: TypeAlias = Callable[..., CommandResult]


def _subscription_environment() -> dict[str, str]:
    """Inherit normal host settings while excluding provider API keys."""
    return subscription_environment(os.environ, transport="cursor-wsl")


def _wsl_cursor_command(base: Sequence[str], *args: str) -> list[str]:
    """Build a probe argv with WSL-side billing selectors explicitly unset."""
    return [
        *base,
        "--exec",
        "/usr/bin/env",
        *[part for name in sorted(subscription_blocked_env_names()) for part in ("-u", name)],
        "cursor-agent",
        *args,
    ]


def _run_command(argv: Sequence[str], *, timeout: float) -> CommandResult:
    try:
        completed = subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            check=False,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_subscription_environment(),
        )
    except subprocess.TimeoutExpired:
        return CommandResult(returncode=124)
    except OSError:
        return CommandResult(returncode=127)
    return CommandResult(completed.returncode, completed.stdout or "")


class _OutputProcess(Protocol):
    stdout: io.TextIOBase | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


ProcessFactory: TypeAlias = Callable[[Sequence[str]], _OutputProcess]


def _spawn_process(argv: Sequence[str]) -> _OutputProcess:
    return subprocess.Popen(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        shell=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_subscription_environment(),
    )


@dataclass(frozen=True)
class CursorWslStatus:
    """Safe authentication signal from ordinary ``cursor-agent status`` text."""

    authenticated: bool
    known: bool


def parse_cursor_status(output: str) -> CursorWslStatus:
    """Parse only an unambiguous authentication signal from bounded text.

    Cursor documents ``status`` as a human-readable command, not a JSON
    contract.  Account, endpoint, and other lines are deliberately ignored.
    Unknown or contradictory output remains unknown so readiness cannot infer
    a subscription login from unrelated status details.
    """
    if not isinstance(output, str) or len(output.encode("utf-8", "replace")) > MAX_STATUS_BYTES:
        return CursorWslStatus(authenticated=False, known=False)
    signals: set[bool] = set()
    for line in output.splitlines():
        normalized = _ANSI_ESCAPE_RE.sub("", line).strip().lower()
        normalized = re.sub(r"^[^a-z0-9]+", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).rstrip(".! ")
        if _AUTHENTICATED_LINE_RE.fullmatch(normalized) or _LOGGED_IN_AS_LINE_RE.fullmatch(
            normalized
        ):
            signals.add(True)
        elif _NOT_AUTHENTICATED_LINE_RE.fullmatch(normalized):
            signals.add(False)
    if len(signals) != 1:
        return CursorWslStatus(authenticated=False, known=False)
    authenticated = signals.pop()
    return CursorWslStatus(authenticated=authenticated, known=True)


@dataclass(frozen=True)
class CursorWslReadiness:
    distro: str
    user: str
    wsl_installed: bool
    distro_ready: bool
    cursor_agent_installed: bool
    auth_present: bool
    transport_ready: bool
    last_verified: str | None
    error: str | None = None


@dataclass(frozen=True)
class CursorWslEvent:
    """Stable, credential-free projection of one Cursor stream event."""

    event_type: str
    text: str = ""
    final: bool = False

    @property
    def kind(self) -> str:
        return self.event_type

    @property
    def type(self) -> str:
        return self.event_type


@dataclass(frozen=True)
class NdjsonParseResult:
    events: tuple[CursorWslEvent, ...]
    malformed_lines: int = 0
    truncated: bool = False


def _bounded_value(value: object, *, limit: int = MAX_ERROR_LENGTH) -> str:
    return str(value)[:limit] if isinstance(value, str) else ""


def _safe_event_text(value: str) -> str:
    """Keep streamed text useful while removing credential-shaped values."""
    return _SENSITIVE_VALUE_RE.sub("[redacted]", value)


def _is_successful_terminal_result(record: dict[object, object], event_type: str) -> bool:
    """Recognize Cursor's successful ``result`` terminal without trusting errors."""
    if event_type.casefold() not in _SUCCESSFUL_TERMINAL_RESULT_TYPES:
        return False
    if any(record.get(name) is False for name in ("success", "ok")):
        return False
    if any(record.get(name) not in (None, False, "", [], {}) for name in ("error", "errors")):
        return False
    status = record.get("status")
    return not isinstance(status, str) or status.casefold() not in _FAILURE_RESULT_STATUSES


def parse_cursor_ndjson(output: str, *, max_bytes: int = MAX_NDJSON_BYTES) -> NdjsonParseResult:
    """Parse bounded Cursor NDJSON without retaining arbitrary provider fields."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")
    encoded = output.encode("utf-8", "replace")
    truncated = len(encoded) > max_bytes
    if truncated:
        output = encoded[:max_bytes].decode("utf-8", "ignore")

    events: list[CursorWslEvent] = []
    malformed = 0
    for line in output.splitlines():
        if len(events) >= MAX_NDJSON_EVENTS:
            truncated = True
            break
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            malformed += 1
            continue
        if not isinstance(record, dict):
            malformed += 1
            continue
        raw_type = record.get("type", record.get("event", record.get("kind", "unknown")))
        event_type = _bounded_value(raw_type) or "unknown"
        text = ""
        for key in ("text", "delta", "output", "result"):
            candidate = record.get(key)
            if isinstance(candidate, str):
                text = candidate
                break
        if not text and isinstance(record.get("content"), str):
            text = record["content"]
        message = record.get("message")
        if not text and isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
        text = _safe_event_text(text)
        final = _is_successful_terminal_result(record, event_type)
        events.append(CursorWslEvent(event_type=event_type, text=text, final=final))
    return NdjsonParseResult(tuple(events), malformed_lines=malformed, truncated=truncated)


@dataclass(frozen=True)
class CursorWslResult:
    ok: bool
    events: tuple[CursorWslEvent, ...] = ()
    text: str = ""
    error: str | None = None
    exit_code: int | None = None
    timed_out: bool = False
    cancelled: bool = False
    truncated: bool = False
    malformed_lines: int = 0


def _is_cancelled(cancel: threading.Event | Callable[[], bool] | None) -> bool:
    if cancel is None:
        return False
    if isinstance(cancel, threading.Event):
        return cancel.is_set()
    try:
        return bool(cancel())
    except (OSError, RuntimeError, TypeError, ValueError):
        return True


def _stop_process(process: _OutputProcess) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=0.25)
    except (OSError, RuntimeError, subprocess.SubprocessError, TimeoutError):
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=0.25)
        except (OSError, RuntimeError, subprocess.SubprocessError, TimeoutError):
            pass


class CursorWslTransport:
    """Run ``cursor-agent`` in one explicitly selected WSL distro."""

    def __init__(
        self,
        distro: str,
        user: str,
        working_directory: str | os.PathLike[str],
        *,
        wsl_executable: str | None = None,
        which: Callable[[str], str | None] | None = None,
        command_runner: CommandRunner | None = None,
        process_factory: ProcessFactory | None = None,
        clock: Callable[[], datetime] | None = None,
        probe_timeout: float = 5.0,
        output_limit: int = MAX_NDJSON_BYTES,
    ) -> None:
        self.distro = validate_distro_name(distro)
        self.user = validate_linux_user(user)
        self.working_directory = os.fspath(working_directory)
        self.wsl_cwd = windows_path_to_wsl(self.working_directory)
        self._which = shutil.which if which is None else which
        self._wsl_executable = wsl_executable
        self._command_runner = command_runner or _run_command
        self._process_factory = process_factory or _spawn_process
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        if probe_timeout <= 0 or output_limit <= 0:
            raise CursorWslError(
                "invalid-limits", "probe and output limits must be greater than zero."
            )
        self._probe_timeout = probe_timeout
        self._output_limit = output_limit

    def _resolve_wsl(self) -> str | None:
        if self._wsl_executable is not None:
            return self._wsl_executable if self._wsl_executable else None
        return self._which("wsl.exe")

    def _probe(self, argv: Sequence[str]) -> CommandResult:
        try:
            return self._command_runner(tuple(argv), timeout=self._probe_timeout)
        except (OSError, RuntimeError, TypeError, subprocess.SubprocessError):
            return CommandResult(returncode=127)

    def readiness(self) -> CursorWslReadiness:
        """Probe WSL, the distro, installation, and login without credentials."""
        executable = self._resolve_wsl()
        now = self._clock().astimezone(timezone.utc).isoformat()
        if executable is None:
            return CursorWslReadiness(
                self.distro, self.user, False, False, False, False, False, now, "wsl-missing"
            )
        base = [executable, "--distribution", self.distro, "--user", self.user]
        kernel = self._probe([*base, "--exec", "uname", "-r"])
        if kernel.returncode != 0:
            return CursorWslReadiness(
                self.distro,
                self.user,
                True,
                False,
                False,
                False,
                False,
                now,
                "distro-unavailable",
            )
        if "wsl2" not in kernel.stdout.lower():
            return CursorWslReadiness(
                self.distro, self.user, True, False, False, False, False, now, "wsl2-required"
            )
        ubuntu = self._probe([*base, "--exec", "grep", "-qx", "ID=ubuntu", "/etc/os-release"])
        if ubuntu.returncode != 0:
            return CursorWslReadiness(
                self.distro, self.user, True, False, False, False, False, now, "ubuntu-required"
            )
        uid = self._probe([*base, "--exec", "id", "-u"])
        if uid.returncode != 0:
            return CursorWslReadiness(
                self.distro,
                self.user,
                True,
                False,
                False,
                False,
                False,
                now,
                "cursor-user-unavailable",
            )
        selected_uid = uid.stdout.strip()
        if not selected_uid.isascii() or not selected_uid.isdecimal():
            return CursorWslReadiness(
                self.distro,
                self.user,
                True,
                False,
                False,
                False,
                False,
                now,
                "cursor-user-uid-invalid",
            )
        if int(selected_uid) == 0:
            return CursorWslReadiness(
                self.distro, self.user, True, False, False, False, False, now, "cursor-user-root"
            )
        workspace = self._probe(
            [*base, "--cd", self.wsl_cwd, "--exec", "/usr/bin/test", "-d", "."]
        )
        if workspace.returncode != 0:
            return CursorWslReadiness(
                self.distro,
                self.user,
                True,
                True,
                False,
                False,
                False,
                now,
                "workspace-unavailable",
            )
        version = self._probe(_wsl_cursor_command(base, "--version"))
        if version.returncode != 0:
            return CursorWslReadiness(
                self.distro,
                self.user,
                True,
                True,
                False,
                False,
                False,
                now,
                "cursor-agent-missing",
            )
        status = self._probe(_wsl_cursor_command(base, "status"))
        parsed = parse_cursor_status(status.stdout if status.returncode == 0 else "")
        ready = status.returncode == 0 and parsed.authenticated
        return CursorWslReadiness(
            self.distro,
            self.user,
            True,
            True,
            True,
            parsed.authenticated,
            ready,
            now,
            None if ready else "cursor-auth-unavailable",
        )

    def run(
        self,
        prompt: str,
        *,
        timeout: float = 120.0,
        cancel: threading.Event | Callable[[], bool] | None = None,
        model: str | None = None,
    ) -> CursorWslResult:
        """Run one bounded headless turn and clean up on every exit path."""
        if timeout <= 0:
            raise CursorWslError("invalid-timeout", "timeout must be greater than zero.")
        executable = self._resolve_wsl()
        if executable is None:
            return CursorWslResult(False, error="wsl-missing")
        argv = build_cursor_wsl_argv(
            self.distro,
            self.user,
            self.working_directory,
            prompt,
            wsl_executable=executable,
            model=model,
        )
        try:
            process = self._process_factory(tuple(argv))
        except (OSError, subprocess.SubprocessError):
            return CursorWslResult(False, error="cursor-process-start-failed")
        stream = process.stdout
        if stream is None:
            _stop_process(process)
            return CursorWslResult(False, error="cursor-process-output-unavailable")

        lines: queue.Queue[str] = queue.Queue()
        reader_done = threading.Event()
        output_truncated = threading.Event()

        def read_output() -> None:
            try:
                total = 0
                while True:
                    line = stream.readline()
                    if not line:
                        return
                    if not isinstance(line, str):
                        line = bytes(line).decode("utf-8", "replace")
                    remaining = self._output_limit - total
                    if remaining > 0:
                        encoded = line.encode("utf-8", "replace")
                        lines.put(encoded[:remaining].decode("utf-8", "ignore"))
                        total += len(encoded)
                        if len(encoded) > remaining:
                            output_truncated.set()
                    else:
                        output_truncated.set()
            finally:
                reader_done.set()

        reader = threading.Thread(target=read_output, name="cursor-wsl-output", daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        cancelled = False
        timed_out = False
        try:
            while not reader_done.wait(0.01):
                if _is_cancelled(cancel):
                    cancelled = True
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
            if reader_done.is_set() and not cancelled and not timed_out:
                grace_deadline = min(deadline, time.monotonic() + _NATURAL_EXIT_GRACE_SECONDS)
                while process.poll() is None:
                    if _is_cancelled(cancel):
                        cancelled = True
                        break
                    now = time.monotonic()
                    if now >= deadline:
                        timed_out = True
                        break
                    if now >= grace_deadline:
                        break
                    wait_timeout = min(0.01, deadline - now, grace_deadline - now)
                    try:
                        process.wait(timeout=wait_timeout)
                    except subprocess.TimeoutExpired:
                        pass
                    except (OSError, RuntimeError, subprocess.SubprocessError, TimeoutError):
                        break
                    if process.poll() is None:
                        time.sleep(wait_timeout)
        finally:
            if cancelled or timed_out or process.poll() is None:
                _stop_process(process)
            reader.join(timeout=0.5)
            close = getattr(stream, "close", None)
            if callable(close):
                with contextlib.suppress(OSError, ValueError):
                    close()

        output_parts: list[str] = []
        while True:
            try:
                output_parts.append(lines.get_nowait())
            except queue.Empty:
                break
        output = "".join(output_parts)
        parsed = parse_cursor_ndjson(output, max_bytes=self._output_limit)
        was_truncated = parsed.truncated or output_truncated.is_set()
        exit_code = process.poll()
        if cancelled:
            return CursorWslResult(
                False,
                parsed.events,
                "",
                "cancelled",
                exit_code,
                cancelled=True,
                truncated=was_truncated,
                malformed_lines=parsed.malformed_lines,
            )
        if timed_out:
            return CursorWslResult(
                False,
                parsed.events,
                "",
                "timeout",
                exit_code,
                timed_out=True,
                truncated=was_truncated,
                malformed_lines=parsed.malformed_lines,
            )
        if was_truncated:
            return CursorWslResult(
                False,
                parsed.events,
                "",
                "output-too-large",
                exit_code,
                truncated=True,
                malformed_lines=parsed.malformed_lines,
            )
        if exit_code is None:
            return CursorWslResult(
                False,
                parsed.events,
                "",
                "process-exit-unknown",
                None,
                malformed_lines=parsed.malformed_lines,
            )
        if exit_code not in (0, None):
            return CursorWslResult(
                False,
                parsed.events,
                "",
                _bounded_error(f"cursor-agent exited with status {exit_code}"),
                exit_code,
                truncated=was_truncated,
                malformed_lines=parsed.malformed_lines,
            )
        if parsed.malformed_lines and not parsed.events:
            return CursorWslResult(
                False,
                (),
                "",
                "invalid-ndjson",
                exit_code,
                truncated=was_truncated,
                malformed_lines=parsed.malformed_lines,
            )
        terminal_results = [event for event in parsed.events if event.final]
        if not terminal_results:
            return CursorWslResult(
                False,
                parsed.events,
                "",
                "missing-terminal-result",
                exit_code,
                malformed_lines=parsed.malformed_lines,
            )
        if len(terminal_results) != 1:
            return CursorWslResult(
                False,
                parsed.events,
                "",
                "duplicate-terminal-result",
                exit_code,
                malformed_lines=parsed.malformed_lines,
            )
        text = "".join(event.text for event in parsed.events if event.text)
        return CursorWslResult(
            True,
            parsed.events,
            text,
            None,
            exit_code,
            truncated=was_truncated,
            malformed_lines=parsed.malformed_lines,
        )

    dispatch = run


CursorWslSubscriptionTransport = CursorWslTransport


__all__ = [
    "MAX_NDJSON_BYTES",
    "CommandResult",
    "CursorWslError",
    "CursorWslEvent",
    "CursorWslReadiness",
    "CursorWslResult",
    "CursorWslStatus",
    "CursorWslSubscriptionTransport",
    "CursorWslTransport",
    "NdjsonParseResult",
    "build_cursor_argv",
    "build_cursor_wsl_argv",
    "parse_cursor_ndjson",
    "parse_cursor_status",
    "translate_windows_path",
    "validate_distro_name",
    "validate_linux_user",
    "windows_path_to_wsl",
]
