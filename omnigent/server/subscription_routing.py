"""Subscription-backed smart-routing judges.

This module deliberately treats Codex and Claude as opaque, already-authenticated
command transports.  The server supplies a closed candidate menu; the commands
only receive a bounded routing prompt and their response is accepted after strict
JSON parsing and candidate validation.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from omnigent.server.smart_routing import (
    ROUTING_REQUEST_TIMEOUT_S,
    RoutingResult,
    _build_rubric,
)

MAX_MESSAGE_CHARS = 4_000
MAX_RUBRIC_CHARS = 12_000
MAX_PROMPT_CHARS = 16_000
MAX_STDOUT_CHARS = 16_384
MAX_STDERR_CHARS = 4_096
MAX_ERROR_CHARS = 300
MAX_HARNESSES = 16
MAX_MODELS = 64
MAX_MODEL_ID_CHARS = 200

_CREDENTIAL_PATTERNS = (
    re.compile(r"\b(?:sk-(?:ant-)?|ghp_|github_pat_|xox[baprs]-|AIza)[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password)"
        r"\s*[:=]\s*[^\s,;]+",
        re.IGNORECASE,
    ),
)


class SubscriptionTransportError(RuntimeError):
    """A subscription CLI could not produce a judge response."""


class SubscriptionTransportTimeout(SubscriptionTransportError):
    """A subscription CLI exceeded its routing budget."""


class SubscriptionJudgeTransport(Protocol):
    """Minimal injected transport interface used by :class:`SubscriptionRoutingClient`."""

    provider: str
    judge_model: str

    async def judge(self, prompt: str) -> str:
        """Return the judge's response text without exposing provider state."""


@dataclass(frozen=True)
class JudgeAttempt:
    """Bounded, receipt-friendly metadata for one judge attempt."""

    provider: str
    judge_model: str
    status: str
    error: str | None = None
    duration_ms: int | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "provider": self.provider,
            "judge_model": self.judge_model,
            "status": self.status,
        }
        if self.error is not None:
            result["error"] = self.error
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        return result


def _bound_text(value: object, limit: int) -> str:
    """Return one-line text with a hard upper bound for logs and receipts."""

    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    for pattern in _CREDENTIAL_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text[:limit]


def _error_detail(exc: BaseException) -> str:
    """Describe an exception without retaining an unbounded provider message."""

    detail = _bound_text(exc, MAX_ERROR_CHARS)
    return detail or type(exc).__name__


def _safe_identity(value: object, fallback: str) -> str:
    """Keep provider/model labels safe for later JSON receipts."""

    text = _bound_text(value, 80)
    return text or fallback


def _normalize_menu(
    available_models: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Copy and cap the caller's candidate menu without inventing candidates."""

    menu: dict[str, list[str]] = {}
    model_count = 0
    for raw_harness, raw_models in available_models.items():
        if len(menu) >= MAX_HARNESSES or not isinstance(raw_harness, str):
            break
        harness = raw_harness.strip()
        if not harness or len(harness) > MAX_MODEL_ID_CHARS or not harness.isprintable():
            continue
        models: list[str] = []
        if not isinstance(raw_models, Sequence) or isinstance(raw_models, (str, bytes)):
            continue
        for raw_model in raw_models:
            if model_count >= MAX_MODELS or not isinstance(raw_model, str):
                break
            model = raw_model.strip()
            if (
                not model
                or len(model) > MAX_MODEL_ID_CHARS
                or not model.isprintable()
                or model in models
            ):
                continue
            models.append(model)
            model_count += 1
        if models:
            menu[harness] = models
        if model_count >= MAX_MODELS:
            break
    return menu


def build_bounded_rubric(
    available_models: Mapping[str, Sequence[str]],
    *,
    soft_allowance_models: Sequence[str] = (),
) -> str:
    """Build the routing rubric plus an optional, non-authoritative allowance hint."""

    menu = _normalize_menu(available_models)
    rubric = _build_rubric(menu)
    offered = {model for models in menu.values() for model in models}
    preferred = list(
        dict.fromkeys(
            model.strip()
            for model in soft_allowance_models
            if isinstance(model, str) and model.strip() in offered
        )
    )
    if preferred:
        rubric += (
            "\nSubscription allowance signal (soft tie-break only): after task "
            "capability and cross-vendor fit are satisfied, equally suitable routes "
            f"may prefer: {', '.join(preferred)}. Never choose a weaker or mismatched "
            "route solely because of this signal.\n"
        )
    if len(rubric) <= MAX_RUBRIC_CHARS:
        return rubric

    # Keep the schema and safety instruction even if a future model catalog grows.
    compact_menu = "\n".join(f"{harness}: {', '.join(models)}" for harness, models in menu.items())
    compact = (
        "Pick exactly one candidate from this closed menu. Return strict JSON only "
        'with keys "harness", "model", and "rationale"; do not add keys.\n'
        f"Closed menu:\n{compact_menu}\n"
        "The model and harness must be copied from the menu."
    )
    if preferred:
        compact += (
            "\nSoft allowance tie-break (only after capability and cross-vendor fit): "
            f"{', '.join(preferred)}.\n"
        )
    return compact[:MAX_RUBRIC_CHARS]


def _build_prompt(message: str, rubric: str) -> str:
    """Combine bounded user text and rubric without allowing prompt growth."""

    task = message[:MAX_MESSAGE_CHARS]
    task_marker = "\n\nUser task:\n"
    rubric_budget = MAX_PROMPT_CHARS - len(task_marker) - len(task)
    return f"{rubric[: max(0, rubric_budget)]}{task_marker}{task}"


def _strict_verdict(text: str) -> dict[str, str]:
    """Parse the exact three-field routing object, rejecting wrappers/fences."""

    if not isinstance(text, str) or len(text) > MAX_STDOUT_CHARS:
        raise ValueError("judge output exceeded the limit")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("judge output contains duplicate keys")
            result[key] = item
        return result

    value = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(value, dict) or set(value) != {"harness", "model", "rationale"}:
        raise ValueError("judge output must contain exactly the routing fields")
    if not all(isinstance(value[key], str) for key in ("harness", "model", "rationale")):
        raise ValueError("routing fields must be strings")
    verdict = cast(dict[str, str], value)
    if not verdict["harness"].strip() or not verdict["model"].strip():
        raise ValueError("judge returned an empty harness or model")
    return verdict


def _extract_command_text(stdout: str) -> str:
    """Extract the final assistant text from Codex JSONL or plain CLI output."""

    stripped = stdout.strip()
    if not stripped or not stripped.startswith("{"):
        return stripped
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        texts: list[str] = []
        for line in stripped.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            item = event.get("item")
            if (
                isinstance(item, dict)
                and item.get("type") in {"agent_message", "assistant_message"}
                and isinstance(item.get("text"), str)
            ):
                texts.append(item["text"])
            for key in ("output_text", "result"):
                if isinstance(event.get(key), str):
                    texts.append(event[key])
        return texts[-1].strip() if texts else stripped
    if isinstance(parsed, dict):
        item = parsed.get("item")
        if (
            isinstance(item, dict)
            and item.get("type") in {"agent_message", "assistant_message"}
            and isinstance(item.get("text"), str)
        ):
            return item["text"].strip()
        for key in ("output_text", "result"):
            if isinstance(parsed.get(key), str):
                return parsed[key].strip()
    return stripped


def _resolve_candidate(
    verdict: Mapping[str, str], menu: Mapping[str, Sequence[str]]
) -> tuple[str, str, dict[str, object]]:
    """Validate and resolve only against the copied closed candidate menu."""

    candidates = [model for models in menu.values() for model in models]
    if not candidates:
        raise ValueError("no candidate models were available")

    raw_model = verdict["model"].strip()
    raw_harness = verdict["harness"].strip()
    if raw_model not in candidates:
        raise ValueError("judge selected a model outside the closed candidate menu")
    model = raw_model
    if model not in menu.get(raw_harness, ()):
        raise ValueError("judge selected a harness/model pair outside the closed candidate menu")
    chosen_harness = raw_harness
    validation = {
        "raw_model": raw_model[:MAX_MODEL_ID_CHARS],
        "selected_model": model,
        "raw_harness": raw_harness[:MAX_MODEL_ID_CHARS],
        "selected_harness": chosen_harness,
        "in_candidate_menu": True,
        "harness_reconciled": False,
    }
    return chosen_harness, model, validation


async def _drain_stream(stream: asyncio.StreamReader, limit: int) -> str:
    """Drain a subprocess stream while retaining at most ``limit`` bytes."""

    chunks: list[bytes] = []
    retained = 0
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        if retained < limit:
            accepted = chunk[: limit - retained]
            chunks.append(accepted)
            retained += len(accepted)
    return b"".join(chunks).decode("utf-8", errors="replace")


async def _communicate_bounded(
    process: Any, prompt: str, *, stdout_limit: int, stderr_limit: int
) -> tuple[str, str]:
    """Write stdin and concurrently drain both streams to avoid pipe deadlocks."""

    stdout_task = asyncio.create_task(_drain_stream(process.stdout, stdout_limit))
    stderr_task = asyncio.create_task(_drain_stream(process.stderr, stderr_limit))
    try:
        if process.stdin is not None:
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
        await process.wait()
        return await stdout_task, await stderr_task
    except BaseException:
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise


async def _cleanup_process(process: Any | None) -> None:
    """Stop a command left behind by a cancelled or failed pipe operation."""

    if process is None or getattr(process, "returncode", None) is not None:
        return
    try:
        process.kill()
    except Exception:  # noqa: BLE001 — cleanup must not mask the original error
        return
    try:
        await process.wait()
    except Exception:  # noqa: BLE001 — cleanup must not mask the original error
        return


async def _invoke_transport(transport: SubscriptionJudgeTransport, prompt: str) -> str:
    """Invoke protocol transports and simple injected async callables alike."""

    callback = getattr(transport, "judge", None)
    if callback is None:
        callback = getattr(transport, "run", transport)
    response = callback(prompt)
    if inspect.isawaitable(response):
        response = await response
    return response


class CommandSubscriptionJudgeTransport:
    """Safe command transport for a vendor CLI's existing subscription login."""

    provider = "subscription"
    judge_model = "subscription-default"
    api_key_env = ""
    command: tuple[str, ...] = ()

    def __init__(
        self,
        executable: str | None = None,
        *,
        timeout_s: float = ROUTING_REQUEST_TIMEOUT_S,
        stdout_limit: int = MAX_STDOUT_CHARS,
        stderr_limit: int = MAX_STDERR_CHARS,
    ) -> None:
        self.executable = executable or self.command[0]
        self.timeout_s = max(0.1, min(float(timeout_s), ROUTING_REQUEST_TIMEOUT_S))
        self.stdout_limit = max(256, min(int(stdout_limit), MAX_STDOUT_CHARS))
        self.stderr_limit = max(256, min(int(stderr_limit), MAX_STDERR_CHARS))

    def command_args(self) -> list[str]:
        """Return an argument array suitable for ``create_subprocess_exec``."""

        return [self.executable, *self.command[1:]]

    async def judge(self, prompt: str) -> str:
        """Run the CLI with inherited login state and no developer API key."""

        from omnigent.onboarding.subscription_readiness import subscription_environment

        # Every API-key selector is scrubbed, not just this judge's vendor key:
        # the route must consume the already-authenticated subscription CLI.
        env = subscription_environment(os.environ, transport=self.provider)
        process: Any | None = None
        with tempfile.TemporaryDirectory(prefix="omnigent-routing-") as temp_dir:
            try:
                process = await asyncio.create_subprocess_exec(
                    *self.command_args(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=temp_dir,
                )
                stdout, stderr = await asyncio.wait_for(
                    _communicate_bounded(
                        process,
                        prompt[:MAX_PROMPT_CHARS],
                        stdout_limit=self.stdout_limit,
                        stderr_limit=self.stderr_limit,
                    ),
                    timeout=self.timeout_s,
                )
            except asyncio.TimeoutError as exc:
                await _cleanup_process(process)
                raise SubscriptionTransportTimeout("subscription judge timed out") from exc
            except FileNotFoundError as exc:
                raise SubscriptionTransportError("subscription judge command unavailable") from exc
            except (OSError, ValueError) as exc:
                await _cleanup_process(process)
                raise SubscriptionTransportError(_error_detail(exc)) from exc
            except BaseException:
                await _cleanup_process(process)
                raise
            if getattr(process, "returncode", 0) != 0:
                # Do not copy provider stderr into a receipt: CLIs own that stream
                # and may include account or auth details despite our env scrub.
                del stderr
                raise SubscriptionTransportError("subscription judge failed")
            return _extract_command_text(stdout)


class CodexSubscriptionTransport(CommandSubscriptionJudgeTransport):
    """Codex ChatGPT-login judge transport."""

    provider = "codex"
    judge_model = "codex-subscription"
    api_key_env = "OPENAI_API_KEY"
    command = (
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


class ClaudeSubscriptionTransport(CommandSubscriptionJudgeTransport):
    """Claude subscription-login judge transport."""

    provider = "claude"
    judge_model = "claude-subscription"
    api_key_env = "ANTHROPIC_API_KEY"
    command = (
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
    )

    async def judge(self, prompt: str) -> str:
        """Fail closed when non-disableable managed policy redirects Claude."""
        from omnigent.onboarding.ambient import claude_managed_subscription_conflicts

        conflicts = claude_managed_subscription_conflicts()
        if conflicts:
            raise SubscriptionTransportError(
                "Claude subscription isolation refused: managed settings configure "
                + ", ".join(conflicts)
            )
        return await super().judge(prompt)


# Descriptive aliases for callers that prefer the authentication mode in the name.
CodexChatGPTSubscriptionTransport = CodexSubscriptionTransport
ClaudeSubscriptionLoginTransport = ClaudeSubscriptionTransport


class SubscriptionRoutingClient:
    """RoutingClient using Codex subscription auth with Claude fallback."""

    def __init__(
        self,
        transports: Sequence[SubscriptionJudgeTransport] | None = None,
        *,
        timeout_s: float = ROUTING_REQUEST_TIMEOUT_S,
        transport_factory: Callable[[], Sequence[SubscriptionJudgeTransport]] | None = None,
        soft_allowance_models: Sequence[str] = (),
    ) -> None:
        if transports is None:
            transports = (
                transport_factory()
                if transport_factory is not None
                else (
                    CodexSubscriptionTransport(timeout_s=timeout_s),
                    ClaudeSubscriptionTransport(timeout_s=timeout_s),
                )
            )
        self._transports = tuple(transports)
        self._soft_allowance_models = tuple(soft_allowance_models)
        self.timeout_s = max(0.1, min(float(timeout_s), ROUTING_REQUEST_TIMEOUT_S))
        self.last_error: str | None = None
        self.last_attempts: list[dict[str, object]] = []
        self.last_judge: dict[str, str] | None = None
        self.last_validation: dict[str, object] | None = None
        self.last_metadata: dict[str, object] = {}
        # ``metadata`` is a convenient receipt seam for callers that retain the
        # client instance after route() returns.
        self.metadata: dict[str, object] = self.last_metadata

    @property
    def attempt_metadata(self) -> list[dict[str, object]]:
        """Alias used by receipt writers for the most recent attempts."""

        return self.last_attempts

    @property
    def judge_metadata(self) -> dict[str, str] | None:
        """Alias used by receipt writers for the accepted judge identity."""

        return self.last_judge

    async def route(
        self,
        message: str,
        available_models: dict[str, list[str]],
    ) -> RoutingResult | None:
        """Choose a candidate or fail closed after every injected judge fails."""

        menu = _normalize_menu(available_models)
        rubric = build_bounded_rubric(
            menu,
            soft_allowance_models=self._soft_allowance_models,
        )
        offered = {model for models in menu.values() for model in models}
        applied_allowance = [
            model for model in dict.fromkeys(self._soft_allowance_models) if model in offered
        ]
        self.last_error = None
        self.last_attempts = []
        self.last_judge = None
        self.last_validation = None
        self.last_metadata = {
            "attempts": self.last_attempts,
            "judge": None,
            "validation": None,
            "candidate_count": sum(len(models) for models in menu.values()),
            "allowance": {
                "policy": "soft tie-break after capability and cross-vendor fit",
                "preferred_models": applied_allowance,
            },
        }
        self.metadata = self.last_metadata
        if not menu:
            self.last_error = "no candidate models were available"
            self.last_metadata["error"] = self.last_error
            return None

        prompt = _build_prompt(message, rubric)
        for transport in self._transports:
            provider = _safe_identity(getattr(transport, "provider", None), "unknown")
            judge_model = _safe_identity(getattr(transport, "judge_model", None), "subscription")
            started = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    _invoke_transport(transport, prompt), timeout=self.timeout_s
                )
                verdict = _strict_verdict(response)
                chosen_harness, model, validation = _resolve_candidate(verdict, menu)
            except asyncio.TimeoutError:
                error = "subscription judge timed out"
                self.last_attempts.append(
                    JudgeAttempt(
                        provider, judge_model, "timeout", error, _elapsed_ms(started)
                    ).as_dict()
                )
                self.last_error = error
                continue
            except SubscriptionTransportTimeout as exc:
                error = _error_detail(exc)
                self.last_attempts.append(
                    JudgeAttempt(
                        provider, judge_model, "timeout", error, _elapsed_ms(started)
                    ).as_dict()
                )
                self.last_error = error
                continue
            except Exception as exc:  # noqa: BLE001 — every judge failure permits fallback
                error = _error_detail(exc)
                self.last_attempts.append(
                    JudgeAttempt(
                        provider, judge_model, "failed", error, _elapsed_ms(started)
                    ).as_dict()
                )
                self.last_error = f"{provider} judge failed: {error}"
                continue
            self.last_attempts.append(
                JudgeAttempt(
                    provider, judge_model, "accepted", duration_ms=_elapsed_ms(started)
                ).as_dict()
            )
            self.last_judge = {"provider": provider, "judge_model": judge_model}
            self.last_validation = validation
            self.last_error = None
            self.last_metadata.update(
                judge=self.last_judge,
                validation=validation,
                selected_harness=chosen_harness,
                selected_model=model,
            )
            return RoutingResult(
                model=model,
                rationale=_bound_text(verdict["rationale"], MAX_ERROR_CHARS),
                harness=chosen_harness,
                raw_model=verdict["model"].strip()[:MAX_MODEL_ID_CHARS],
            )

        if self.last_error is None:
            self.last_error = "no subscription routing judge was available"
        self.last_error = _bound_text(self.last_error, MAX_ERROR_CHARS)
        self.last_metadata["error"] = self.last_error
        return None


def _elapsed_ms(started: float) -> int:
    """Return a small integer duration suitable for a receipt."""

    return max(0, int((time.monotonic() - started) * 1000))


__all__ = [
    "ClaudeSubscriptionLoginTransport",
    "ClaudeSubscriptionTransport",
    "CodexChatGPTSubscriptionTransport",
    "CodexSubscriptionTransport",
    "CommandSubscriptionJudgeTransport",
    "JudgeAttempt",
    "SubscriptionJudgeTransport",
    "SubscriptionRoutingClient",
    "SubscriptionTransportError",
    "SubscriptionTransportTimeout",
    "build_bounded_rubric",
]
