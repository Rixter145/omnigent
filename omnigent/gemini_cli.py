"""Safe, subscription-backed transport for Google's official Gemini CLI.

This module deliberately does not share the ``agy``/Antigravity path.  The
official ``gemini`` CLI owns Google-account OAuth and emits its own JSONL
stream; Omnigent only discovers the executable, removes competing billing
signals from the child environment, and translates bounded events.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import Any, Protocol

from omnigent.subscription_security import subscription_environment as _subscription_environment

MAX_JSON_LINE_BYTES = 64 * 1024
MAX_EVENTS = 512
MAX_EVENT_TEXT_CHARS = 32 * 1024
MAX_FAILURE_CHARS = 2_000
MAX_STDERR_BYTES = 16 * 1024
MAX_PROMPT_CHARS = 128 * 1024
SUBSCRIPTION_BLOCKED_ENV_VARS = frozenset()

_OAUTH_MARKER_RELATIVE_PATHS = (Path(".gemini") / "oauth_creds.json",)
_OFFICIAL_NAMES = frozenset({"gemini", "gemini.exe", "gemini.cmd", "gemini.bat", "gemini.ps1"})
_TOKEN_RE = re.compile(
    r"(?:ya29\.[A-Za-z0-9._-]+|1//[A-Za-z0-9._-]+|AIza[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]+)"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:access_token|refresh_token|id_token|authorization|api[_-]?key|token)\b\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)


class GeminiCliError(RuntimeError):
    """Base error for invalid or unavailable Gemini CLI transport choices."""


class UnsupportedAuthMode(GeminiCliError):
    """Raised when a non-subscription mode is sent to this transport."""


class UnsupportedLauncher(GeminiCliError):
    """Raised when a launcher cannot be invoked without a shell command string."""


class ReadinessState(StrEnum):
    NOT_INSTALLED = "not-installed"
    INSTALLED = "installed"
    AUTH_PRESENT = "auth-present"
    AUTH_VERIFIED = "auth-verified"


class EventKind(StrEnum):
    TEXT = "text"
    TOOL = "tool"
    USAGE = "usage"
    ERROR = "error"
    COMPLETION = "completion"


@dataclass(frozen=True)
class GeminiExecutable:
    """An official Gemini CLI executable and its required Windows launcher."""

    path: str
    launcher: str = "direct"
    runtime: str | None = None

    @property
    def is_wrapper(self) -> bool:
        return self.launcher != "direct"


@dataclass(frozen=True)
class GeminiReadiness:
    """Credential-blind readiness information for the subscription transport."""

    state: ReadinessState
    installed: bool
    auth_present: bool
    auth_verified: bool
    executable: GeminiExecutable | None = None
    auth_marker: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class GeminiEvent:
    """Stable, redacted representation of one meaningful JSONL event."""

    kind: EventKind
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    input: Any = None
    output: str | None = None
    usage: Mapping[str, int | float] = field(default_factory=dict)
    error: str | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe event mapping without provider raw payloads."""

        result: dict[str, Any] = {"kind": self.kind.value}
        for name in ("text", "tool_name", "tool_id", "input", "output", "error", "status"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.usage:
            result["usage"] = dict(self.usage)
        return result


@dataclass(frozen=True)
class ParsedGeminiStream:
    events: tuple[GeminiEvent, ...]
    text: str = ""
    usage: Mapping[str, int | float] = field(default_factory=dict)
    completed: bool = False
    error: str | None = None
    malformed_lines: int = 0
    oversized_lines: int = 0

    def __iter__(self):
        return iter(self.events)


@dataclass(frozen=True)
class GeminiResult:
    """Bounded result of one headless Gemini CLI turn."""

    events: tuple[GeminiEvent, ...] = ()
    text: str = ""
    usage: Mapping[str, int | float] = field(default_factory=dict)
    completed: bool = False
    exit_code: int | None = None
    error: str | None = None
    timed_out: bool = False
    cancelled: bool = False
    unsupported_launcher: bool = False
    malformed_lines: int = 0
    oversized_lines: int = 0

    @property
    def output(self) -> str:
        """Compatibility spelling for callers that use output instead of text."""

        return self.text


@dataclass(frozen=True)
class GeminiInvocation:
    argv: tuple[str, ...]
    env: Mapping[str, str]
    shell: bool = False


class _ReadableProcess(Protocol):
    stdout: Any
    stderr: Any
    returncode: int | None

    async def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


ProcessFactory = Callable[[Sequence[str], Mapping[str, str]], Any]
AuthProbe = Callable[[GeminiExecutable], bool]


def redact_text(value: object, *, limit: int = MAX_FAILURE_CHARS) -> str:
    """Return bounded text with common OAuth/API credential forms removed."""

    text = str(value)[:limit]
    text = _SECRET_ASSIGNMENT_RE.sub(r"\1<redacted>", text)
    text = _TOKEN_RE.sub("<redacted>", text)
    return text[:limit]


def subscription_environment(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy *env* while removing API-key and Vertex selectors.

    The OAuth credential remains owned by the user's Gemini CLI home.  This
    function never opens, parses, or returns credential files or values.
    """

    return _subscription_environment(env, transport="gemini-cli")


def _path_basename(path: str, *, windows: bool) -> str:
    return (PureWindowsPath(path).name if windows else Path(path).name).lower()


def _launcher_for_path(path: str, *, windows: bool) -> str:
    suffix = Path(PureWindowsPath(path).name if windows else path).suffix.lower()
    if windows and suffix in {".cmd", ".bat"}:
        return "cmd"
    if windows and suffix == ".ps1":
        return "powershell"
    return "direct"


def _windows_npm_node_executable(
    candidate: str,
    *,
    lookup: Callable[[str], str | None],
) -> GeminiExecutable | None:
    """Resolve npm's Windows shim to its official Node entrypoint.

    ``npm install -g @google/gemini-cli`` publishes ``gemini.cmd`` plus an
    extensionless POSIX shim.  Passing a prompt through either wrapper would
    require a command shell, so locate npm's fixed package entrypoint next to
    the shim and invoke it with Node directly instead.  No wrapper contents are
    parsed and no user-controlled command string is constructed.
    """

    shim = Path(candidate)
    entrypoint = shim.parent / "node_modules" / "@google" / "gemini-cli" / "bundle" / "gemini.js"
    if not entrypoint.is_file():
        return None
    adjacent_node = shim.parent / "node.exe"
    node = (
        str(adjacent_node) if adjacent_node.is_file() else (lookup("node.exe") or lookup("node"))
    )
    if not node:
        return None
    return GeminiExecutable(str(entrypoint), "node", node)


def discover_gemini_cli(
    *,
    which: Callable[[str], str | None] | None = None,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> GeminiExecutable | None:
    """Find the official ``gemini`` executable, never ``agy``/Antigravity."""

    env = os.environ if environ is None else environ
    windows = (os.name if platform is None else platform) == "nt"
    lookup = shutil.which if which is None else which
    override = env.get("OMNIGENT_GEMINI_PATH", "").strip()
    names = (
        (override,)
        if override
        else (
            ("gemini", "gemini.exe", "gemini.cmd", "gemini.bat", "gemini.ps1")
            if windows
            else ("gemini",)
        )
    )
    for name in names:
        candidate = lookup(name)
        if not candidate and override and (Path(name).is_file() or os.access(name, os.X_OK)):
            candidate = name
        if not candidate:
            continue
        basename = _path_basename(candidate, windows=windows)
        if basename not in _OFFICIAL_NAMES:
            continue
        if windows and basename in {"gemini", "gemini.cmd", "gemini.bat"}:
            npm_executable = _windows_npm_node_executable(candidate, lookup=lookup)
            if npm_executable is not None:
                return npm_executable
        return GeminiExecutable(candidate, _launcher_for_path(candidate, windows=windows))
    return None


def _default_auth_markers(home: Path | None = None) -> tuple[Path, ...]:
    root = Path.home() if home is None else home
    return tuple(root / relative for relative in _OAUTH_MARKER_RELATIVE_PATHS)


def gemini_readiness(
    *,
    executable: GeminiExecutable | None = None,
    which: Callable[[str], str | None] | None = None,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
    auth_marker: Path | None = None,
    auth_probe: AuthProbe | None = None,
) -> GeminiReadiness:
    """Report installed, OAuth-marker-present, and explicitly verified states.

    ``auth_probe`` is intentionally injected: the default readiness check is
    secret-blind and does not make a provider request or run a live prompt.
    """

    found = executable or discover_gemini_cli(which=which, environ=environ, platform=platform)
    if found is None:
        return GeminiReadiness(
            ReadinessState.NOT_INSTALLED,
            False,
            False,
            False,
            reason="gemini executable not found",
        )

    markers = (auth_marker,) if auth_marker is not None else _default_auth_markers(home)
    present_path = next((path for path in markers if path.is_file()), None)
    present = present_path is not None
    verified = False
    if auth_probe is not None:
        try:
            verified = bool(auth_probe(found))
        except (OSError, RuntimeError, TypeError, ValueError):
            verified = False
    state = (
        ReadinessState.AUTH_VERIFIED
        if verified
        else ReadinessState.AUTH_PRESENT
        if present
        else ReadinessState.INSTALLED
    )
    return GeminiReadiness(
        state,
        True,
        present,
        verified,
        found,
        str(present_path) if present_path is not None else None,
        None if present or verified else "Google-account OAuth marker not found",
    )


def build_gemini_invocation(
    prompt: str,
    *,
    executable: GeminiExecutable | str,
    env: Mapping[str, str] | None = None,
    model: str | None = None,
    auth_mode: str = "subscription",
    platform: str | None = None,
    comspec: str | None = None,
    powershell: str | None = None,
) -> GeminiInvocation:
    """Build a shell-free headless invocation for the subscription path."""

    # Kept as a compatibility keyword for callers that previously supplied a
    # COMSPEC override.  Batch launchers are now rejected before COMSPEC can
    # interpret any user-controlled data.
    del comspec
    if auth_mode != "subscription":
        raise UnsupportedAuthMode(
            f"Gemini CLI transport only supports subscription auth; refused {auth_mode!r}"
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Gemini prompt must be a non-empty string")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"Gemini prompt exceeds {MAX_PROMPT_CHARS} characters")
    item = (
        executable
        if isinstance(executable, GeminiExecutable)
        else GeminiExecutable(
            executable,
            _launcher_for_path(executable, windows=(platform or os.name) == "nt"),
        )
    )
    # Headless subscription dispatch has no interactive approval bridge yet.
    # Keep the transport useful for analysis/review while preventing an
    # unattended Gemini turn from mutating the workspace.
    args = [
        "-p",
        prompt,
        "--approval-mode",
        "plan",
        "--output-format",
        "stream-json",
    ]
    if model:
        args[0:0] = ["--model", model]
    if item.launcher == "cmd":
        raise UnsupportedLauncher(
            "Gemini .cmd/.bat launchers are unsupported safely; install gemini.exe "
            "or gemini.ps1 for subscription headless transport"
        )
    if item.launcher == "node":
        if not item.runtime:
            raise UnsupportedLauncher("Gemini Node launcher is missing its runtime")
        argv = (item.runtime, item.path, *args)
    elif item.launcher == "powershell":
        shell_path = (
            powershell
            or shutil.which("powershell.exe")
            or shutil.which("pwsh")
            or "powershell.exe"
        )
        argv = (
            shell_path,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            item.path,
            *args,
        )
    else:
        argv = (item.path, *args)
    return GeminiInvocation(tuple(argv), subscription_environment(env), shell=False)


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return "<nested value omitted>"
    if isinstance(value, str):
        return redact_text(value, limit=MAX_EVENT_TEXT_CHARS)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in list(value.items())[:64]:
            safe_key = redact_text(key, limit=128)
            if str(key).lower() in {
                "access_token",
                "refresh_token",
                "id_token",
                "authorization",
                "api_key",
                "apikey",
                "token",
            }:
                result[safe_key] = "<redacted>"
            else:
                result[safe_key] = _safe_value(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_value(item, depth=depth + 1) for item in list(value)[:64]]
    return redact_text(value, limit=256)


def _text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return redact_text(value, limit=MAX_EVENT_TEXT_CHARS)
    if isinstance(value, Mapping):
        for key in ("text", "content", "output"):
            text = _text_from_content(value.get(key))
            if text:
                return text
        return _text_from_content(value.get("parts"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "".join(_text_from_content(item) for item in list(value)[:128])[
            :MAX_EVENT_TEXT_CHARS
        ]
    return ""


def _usage_from(value: Any) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        return {}
    source = value.get("usage") if isinstance(value.get("usage"), Mapping) else value
    if source is value and isinstance(value.get("stats"), Mapping):
        source = value["stats"]
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens", "promptTokenCount"),
        "output_tokens": ("output_tokens", "completion_tokens", "candidatesTokenCount"),
        "total_tokens": ("total_tokens", "totalTokenCount"),
        "cached_input_tokens": ("cached_input_tokens", "cachedContentTokenCount"),
    }
    result: dict[str, int | float] = {}
    for normalized, keys in aliases.items():
        for key in keys:
            candidate = source.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                result[normalized] = candidate
                break
    return result


def _error_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return _error_text(value.get("message", value.get("error", "Gemini CLI error")))
    return redact_text(value)


def _event_from_payload(payload: Mapping[str, Any]) -> GeminiEvent | None:
    event_type = str(payload.get("type", "")).lower()
    if event_type in {"message", "assistant", "text", "content"}:
        message_value = payload.get("message")
        role = str(
            payload.get("role")
            or (message_value if isinstance(message_value, Mapping) else {}).get(
                "role", "assistant"
            )
        ).lower()
        message = message_value if isinstance(message_value, Mapping) else payload
        text = _text_from_content(message.get("content", message.get("text")))
        if text and role in {"assistant", "model", "gemini", ""}:
            return GeminiEvent(EventKind.TEXT, text=text)
        return None
    if event_type in {
        "tool",
        "tool_use",
        "tool_call",
        "function_call",
        "tool_result",
        "function_result",
    }:
        function = payload.get("function") if isinstance(payload.get("function"), Mapping) else {}
        name = payload.get("tool_name") or payload.get("name") or function.get("name")
        tool_id = payload.get("tool_id") or payload.get("id") or payload.get("call_id")
        arguments = payload.get(
            "input",
            payload.get("arguments", payload.get("parameters", function.get("arguments"))),
        )
        output = _text_from_content(payload.get("output", payload.get("result")))
        error = payload.get("error")
        return GeminiEvent(
            EventKind.TOOL,
            tool_name=redact_text(name, limit=256) if name is not None else None,
            tool_id=redact_text(tool_id, limit=256) if tool_id is not None else None,
            input=_safe_value(arguments),
            output=output or None,
            error=_error_text(error) if error is not None else None,
            status=(
                redact_text(payload.get("status"), limit=64)
                if payload.get("status") is not None
                else None
            ),
        )
    if event_type in {"usage", "stats"}:
        usage = _usage_from(payload)
        return GeminiEvent(EventKind.USAGE, usage=usage) if usage else None
    if (
        event_type in {"error", "failure"}
        or payload.get("error") is not None
        or str(payload.get("status", "")).lower() in {"error", "failed"}
    ):
        error = payload.get(
            "error", payload.get("message", payload.get("status", "Gemini CLI error"))
        )
        return GeminiEvent(EventKind.ERROR, error=_error_text(error))
    if event_type in {"result", "completion", "complete", "final"}:
        status = str(payload.get("status") or payload.get("subtype") or "success").lower()
        if status in {"error", "failed", "failure"}:
            return GeminiEvent(
                EventKind.ERROR,
                error=_error_text(payload.get("error", status)),
                status=status,
            )
        final_text = _text_from_content(
            payload.get("result", payload.get("response", payload.get("text")))
        )
        return GeminiEvent(
            EventKind.COMPLETION,
            text=final_text or None,
            usage=_usage_from(payload),
            status=status,
        )
    return None


def _parse_line(line: str | bytes, *, max_line_bytes: int) -> tuple[GeminiEvent | None, int, int]:
    if not line.strip():
        return None, 0, 0
    size = len(line) if isinstance(line, bytes) else len(line.encode("utf-8", errors="replace"))
    if size > max_line_bytes:
        return (
            GeminiEvent(
                EventKind.ERROR,
                error=f"ignored oversized JSON line (>{max_line_bytes} bytes)",
            ),
            0,
            1,
        )
    try:
        raw = line.decode("utf-8") if isinstance(line, bytes) else line
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return (
            GeminiEvent(EventKind.ERROR, error="ignored malformed Gemini JSON line"),
            1,
            0,
        )
    if not isinstance(payload, Mapping):
        return (
            GeminiEvent(EventKind.ERROR, error="ignored non-object Gemini JSON line"),
            1,
            0,
        )
    return _event_from_payload(payload), 0, 0


class _StreamAccumulator:
    """Incrementally collect semantic stream data without retaining raw JSON."""

    def __init__(self, *, max_line_bytes: int) -> None:
        self.max_line_bytes = max_line_bytes
        self.events: list[GeminiEvent] = []
        self.text = ""
        self.usage: dict[str, int | float] = {}
        self.completed = False
        self.first_error: str | None = None
        self.malformed_lines = 0
        self.oversized_lines = 0

    def add(self, line: str | bytes) -> None:
        event, malformed, oversized = _parse_line(line, max_line_bytes=self.max_line_bytes)
        self.malformed_lines += malformed
        self.oversized_lines += oversized
        if event is None:
            return
        if len(self.events) < MAX_EVENTS:
            self.events.append(event)
        if event.kind is EventKind.TEXT and event.text and len(self.text) < MAX_EVENT_TEXT_CHARS:
            remaining = MAX_EVENT_TEXT_CHARS - len(self.text)
            self.text += event.text[:remaining]
        if event.kind is EventKind.COMPLETION:
            self.completed = True
            if event.text:
                self.text = event.text[:MAX_EVENT_TEXT_CHARS]
        if event.usage:
            self.usage.update(event.usage)
        if event.kind is EventKind.ERROR and self.first_error is None:
            self.first_error = event.error

    def finish(self) -> ParsedGeminiStream:
        return ParsedGeminiStream(
            tuple(self.events),
            self.text,
            self.usage,
            self.completed,
            self.first_error,
            self.malformed_lines,
            self.oversized_lines,
        )


def parse_gemini_stream(
    lines: Iterable[str | bytes], *, max_line_bytes: int = MAX_JSON_LINE_BYTES
) -> ParsedGeminiStream:
    """Parse bounded JSONL without retaining raw provider responses."""

    accumulator = _StreamAccumulator(max_line_bytes=max_line_bytes)
    for line in lines:
        accumulator.add(line)
    return accumulator.finish()


parse_stream_json = parse_gemini_stream


async def _default_process_factory(
    argv: Sequence[str], env: Mapping[str, str]
) -> _ReadableProcess:
    return await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env),
        shell=False,
        limit=MAX_JSON_LINE_BYTES + 1,
    )


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _cleanup_process(process: _ReadableProcess) -> None:
    if getattr(process, "returncode", None) is None:
        with contextlib.suppress(OSError, ProcessLookupError):
            process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except (asyncio.TimeoutError, OSError, ProcessLookupError):
        if getattr(process, "returncode", None) is None:
            with contextlib.suppress(OSError, ProcessLookupError):
                process.kill()
        with contextlib.suppress(asyncio.TimeoutError, OSError, ProcessLookupError):
            await asyncio.wait_for(process.wait(), timeout=1.0)


class GeminiCliTransport:
    """Provider-neutral async Gemini CLI subscription transport."""

    transport_id = "gemini-cli"
    antigravity_id = "antigravity"

    def __init__(
        self,
        *,
        executable: GeminiExecutable | str | None = None,
        env: Mapping[str, str] | None = None,
        process_factory: ProcessFactory | None = None,
        platform: str | None = None,
        comspec: str | None = None,
        powershell: str | None = None,
        auth_mode: str = "subscription",
    ) -> None:
        if auth_mode != "subscription":
            raise UnsupportedAuthMode(
                f"Gemini CLI transport only supports subscription auth; refused {auth_mode!r}"
            )
        self.executable = executable
        self.env = env
        self.process_factory = process_factory or _default_process_factory
        self.platform = platform
        self.comspec = comspec
        self.powershell = powershell

    async def run(
        self,
        prompt: str,
        *,
        timeout_s: float = 120.0,
        cancel_event: asyncio.Event | None = None,
        model: str | None = None,
    ) -> GeminiResult:
        """Run one bounded headless turn; cancellation always cleans its child."""

        executable = self.executable
        if executable is None:
            executable = discover_gemini_cli(platform=self.platform)
        if executable is None:
            return GeminiResult(error="Gemini CLI is not installed")
        try:
            invocation = build_gemini_invocation(
                prompt,
                executable=executable,
                env=self.env,
                model=model,
                platform=self.platform,
                comspec=self.comspec,
                powershell=self.powershell,
            )
        except (GeminiCliError, ValueError) as exc:
            return GeminiResult(
                error=redact_text(exc),
                unsupported_launcher=isinstance(exc, UnsupportedLauncher),
            )
        worker = asyncio.create_task(self._execute(invocation))
        if cancel_event is None:
            try:
                return await asyncio.wait_for(worker, timeout=timeout_s)
            except asyncio.TimeoutError:
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
                return GeminiResult(
                    error=f"Gemini CLI timed out after {timeout_s:g}s", timed_out=True
                )
            except asyncio.CancelledError:
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
                raise
        cancel_waiter = asyncio.create_task(cancel_event.wait())
        worker_finished = False
        try:
            done, _ = await asyncio.wait(
                {worker, cancel_waiter},
                timeout=timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if worker in done:
                worker_finished = True
                return worker.result()
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            if cancel_waiter in done and cancel_event.is_set():
                return GeminiResult(error="Gemini CLI run cancelled", cancelled=True)
            return GeminiResult(error=f"Gemini CLI timed out after {timeout_s:g}s", timed_out=True)
        finally:
            cancel_waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_waiter
            if not worker_finished and not worker.done():
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker

    async def _execute(self, invocation: GeminiInvocation) -> GeminiResult:
        process: _ReadableProcess | None = None
        stderr_task: asyncio.Task[str] | None = None
        try:
            process = await _maybe_await(self.process_factory(invocation.argv, invocation.env))
            stderr_task = asyncio.create_task(self._read_stderr(process))
            accumulator = _StreamAccumulator(max_line_bytes=MAX_JSON_LINE_BYTES)
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                accumulator.add(line)
            exit_code = await process.wait()
            parsed = accumulator.finish()
            stderr = await stderr_task
            error = parsed.error
            if error is None and exit_code != 0:
                error = redact_text(stderr) or f"Gemini CLI exited with status {exit_code}"
            if error is None and not parsed.completed:
                error = "Gemini CLI stream ended without a completion event"
            return GeminiResult(
                parsed.events,
                parsed.text,
                parsed.usage,
                parsed.completed,
                exit_code,
                error,
                malformed_lines=parsed.malformed_lines,
                oversized_lines=parsed.oversized_lines,
            )
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            return GeminiResult(error=redact_text(exc))
        finally:
            if stderr_task is not None and not stderr_task.done():
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stderr_task
            if process is not None:
                await _cleanup_process(process)

    @staticmethod
    async def _read_stderr(process: _ReadableProcess) -> str:
        data = await process.stderr.read(MAX_STDERR_BYTES)
        return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)


async def run_gemini_cli(
    prompt: str,
    *,
    transport: GeminiCliTransport | None = None,
    timeout_s: float = 120.0,
    cancel_event: asyncio.Event | None = None,
    model: str | None = None,
) -> GeminiResult:
    """Convenience wrapper for a single subscription-backed headless turn."""

    runner = transport or GeminiCliTransport()
    return await runner.run(
        prompt,
        timeout_s=timeout_s,
        cancel_event=cancel_event,
        model=model,
    )


discover_gemini_executable = discover_gemini_cli
build_headless_invocation = build_gemini_invocation
GeminiTransport = GeminiCliTransport
check_gemini_readiness = gemini_readiness
scrub_subscription_env = subscription_environment

__all__ = [
    "MAX_JSON_LINE_BYTES",
    "SUBSCRIPTION_BLOCKED_ENV_VARS",
    "EventKind",
    "GeminiCliError",
    "GeminiCliTransport",
    "GeminiEvent",
    "GeminiExecutable",
    "GeminiInvocation",
    "GeminiReadiness",
    "GeminiResult",
    "GeminiTransport",
    "ParsedGeminiStream",
    "ReadinessState",
    "UnsupportedAuthMode",
    "UnsupportedLauncher",
    "build_gemini_invocation",
    "build_headless_invocation",
    "check_gemini_readiness",
    "discover_gemini_cli",
    "discover_gemini_executable",
    "gemini_readiness",
    "parse_gemini_stream",
    "parse_stream_json",
    "redact_text",
    "run_gemini_cli",
    "scrub_subscription_env",
    "subscription_environment",
]
