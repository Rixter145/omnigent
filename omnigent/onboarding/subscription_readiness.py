"""Credential-blind readiness for the supported subscription transports.

This module is deliberately additive to the older boolean harness readiness
map.  It reports enough provenance for setup and diagnostics while never
loading an auth-file value or returning provider output.  A provider is only
``auth-verified`` after its own CLI status command or an explicit bounded
probe says so.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from omnigent._platform import resolve_cli_binary
from omnigent.cursor_wsl import CursorWslError, CursorWslTransport, validate_linux_user
from omnigent.gemini_cli import (
    GeminiCliTransport,
    GeminiExecutable,
    discover_gemini_cli,
    gemini_readiness,
)
from omnigent.onboarding.ambient import codex_auth_effective_mode
from omnigent.onboarding.harness_install import (
    ANTHROPIC_FAMILY,
    OPENAI_FAMILY,
    READINESS_CLI_PROBE_TIMEOUT_S,
    harness_install_spec,
)
from omnigent.onboarding.provider_config import load_config
from omnigent.subscription_defaults import GOOGLE_SUBSCRIPTION_POLICY_REASON
from omnigent.subscription_security import subscription_environment as _subscription_environment

CLAUDE_PROVIDER = "claude"
CODEX_PROVIDER = "codex"
GEMINI_CLI_PROVIDER = "gemini-cli"
CURSOR_WSL_PROVIDER = "cursor-wsl"
ANTIGRAVITY_PROVIDER = "antigravity"

SUPPORTED_PROVIDERS = (
    CLAUDE_PROVIDER,
    CODEX_PROVIDER,
    GEMINI_CLI_PROVIDER,
    CURSOR_WSL_PROVIDER,
)
# Compatibility export. The policy itself is owned by subscription_security.
COMPETING_API_KEY_ENV_VARS = frozenset()
PROVIDER_ALIASES = {"gemini": GEMINI_CLI_PROVIDER, "cursor": CURSOR_WSL_PROVIDER}
CURSOR_DISTRO_ENV = "OMNIGENT_CURSOR_WSL_DISTRO"
CURSOR_USER_ENV = "OMNIGENT_CURSOR_WSL_USER"
CURSOR_WORKING_DIRECTORY_ENV = "OMNIGENT_CURSOR_WSL_WORKING_DIRECTORY"
SAFE_PROBE_PROMPT = "Reply with exactly SUBSCRIPTION_MVP_PROBE_OK and nothing else."
MAX_PROBE_TIMEOUT_S = 30.0


class ReadinessState(StrEnum):
    NOT_READY = "not-ready"
    INSTALLED = "installed"
    AUTH_PRESENT = "auth-present"
    AUTH_VERIFIED = "auth-verified"


@dataclass(frozen=True)
class SubscriptionReadiness:
    """Safe, additive readiness state for one subscription provider."""

    provider: str
    state: ReadinessState
    installed: bool
    auth_present: bool
    auth_verified: bool
    transport_ready: bool
    provenance: tuple[str, ...] = ()
    last_verified: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-safe projection with no secret-shaped data."""

        return {
            "provider": self.provider,
            "state": self.state.value,
            "installed": self.installed,
            "auth_present": self.auth_present,
            "auth_verified": self.auth_verified,
            "transport_ready": self.transport_ready,
            "provenance": list(self.provenance),
            "last_verified": self.last_verified,
            "reason": self.reason,
        }


def canonical_provider(provider: str) -> str:
    """Normalize supported aliases without conflating Antigravity and Gemini."""

    normalized = provider.strip().lower()
    if normalized == ANTIGRAVITY_PROVIDER:
        raise ValueError("antigravity is a separate compatibility provider, not Gemini CLI")
    canonical = PROVIDER_ALIASES.get(normalized, normalized)
    if canonical not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unsupported subscription provider: {provider}")
    return canonical


def subscription_environment(
    environ: Mapping[str, str] | None = None,
    *,
    no_api_keys: bool = True,
    transport: str | None = None,
) -> dict[str, str]:
    """Copy an environment while banning competing API-key selectors.

    ``no_api_keys`` is retained as an explicit call-site control for the
    verifier.  Subscription transports always use the safe policy, even when
    a caller forgets to pass the CLI flag.
    """

    source = os.environ if environ is None else environ
    if not no_api_keys:
        # The subscription path is never allowed to inherit a competing key;
        # accepting this argument only keeps old embedding call sites readable.
        no_api_keys = True
    del no_api_keys
    return _subscription_environment(source, transport=transport)


def _now(clock: Callable[[], datetime] | None = None) -> str:
    current = (clock or (lambda: datetime.now(timezone.utc)))()
    return current.astimezone(timezone.utc).isoformat()


def _state(
    *, installed: bool, auth_present: bool, auth_verified: bool, not_ready: bool = False
) -> ReadinessState:
    if not installed or not_ready:
        return ReadinessState.NOT_READY
    if auth_verified:
        return ReadinessState.AUTH_VERIFIED
    if auth_present:
        return ReadinessState.AUTH_PRESENT
    return ReadinessState.INSTALLED


def _marker_exists(paths: Sequence[Path]) -> bool:
    try:
        return any(path.is_file() for path in paths)
    except OSError:
        return False


def _default_markers(provider: str, home: Path | None = None) -> tuple[Path, ...]:
    root = Path.home() if home is None else home
    if provider == CLAUDE_PROVIDER:
        return (root / ".claude" / ".credentials.json", root / ".claude.json")
    if provider == CODEX_PROVIDER:
        return (root / ".codex" / "auth.json",)
    return ()


def _codex_uses_chatgpt_auth(auth_path: Path) -> bool:
    """Return whether Codex's resolved local auth mode is ChatGPT OAuth."""

    return codex_auth_effective_mode(auth_path) == "chatgpt"


def _codex_probe_auth_path(environ: Mapping[str, str] | None) -> Path:
    """Locate the selected Codex auth file without loading credential values."""

    source = os.environ if environ is None else environ
    codex_home = source.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home) / "auth.json"
    return _default_markers(CODEX_PROVIDER)[0]


def resolve_cursor_distro(
    *,
    option: str | None = None,
    environ: Mapping[str, str] | None = None,
    config: Mapping[str, object] | None = None,
) -> tuple[str | None, str | None]:
    """Resolve an explicit Cursor WSL distro and its non-secret provenance.

    Precedence is option, environment, then config.  No installed distro is
    discovered or guessed.  Config accepts the additive ``cursor_wsl.distro``
    shape and the existing Cursor block's ``wsl_distro`` spelling.
    """

    env = os.environ if environ is None else environ
    if option is not None and option.strip():
        return option.strip(), "explicit-option"
    env_value = env.get(CURSOR_DISTRO_ENV, "").strip()
    if env_value:
        return env_value, "explicit-env"
    cfg = load_config() if config is None else config
    candidates: list[str] = []
    for block_name, field_names in (
        ("cursor_wsl", ("distro", "distribution")),
        ("cursor", ("wsl_distro", "wsl_distribution")),
    ):
        block = cfg.get(block_name)
        if not isinstance(block, Mapping):
            continue
        for field_name in field_names:
            value = block.get(field_name)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    if candidates:
        if len(set(candidates)) > 1:
            return None, "conflicting-config"
        return candidates[0], "explicit-config"
    return None, None


def resolve_cursor_user(
    *,
    option: str | None = None,
    environ: Mapping[str, str] | None = None,
    config: Mapping[str, object] | None = None,
) -> tuple[str | None, str | None]:
    """Resolve the explicit Cursor WSL user without guessing an identity."""

    env = os.environ if environ is None else environ
    if option is not None and option.strip():
        return option.strip(), "explicit-option"
    env_value = env.get(CURSOR_USER_ENV, "").strip()
    if env_value:
        return env_value, "explicit-env"
    cfg = load_config() if config is None else config
    block = cfg.get("cursor_wsl")
    if isinstance(block, Mapping):
        value = block.get("user")
        if isinstance(value, str) and value.strip():
            return value.strip(), "explicit-config"
    return None, None


def _readiness_for_cli(
    provider: str,
    *,
    verify: bool,
    home: Path | None,
    marker_paths: Sequence[Path] | None,
    which: Callable[[str], str | None] | None,
    environ: Mapping[str, str] | None,
    clock: Callable[[], datetime] | None,
) -> SubscriptionReadiness:
    key, binary, status_label = {
        CLAUDE_PROVIDER: (ANTHROPIC_FAMILY, "claude", "claude auth status"),
        CODEX_PROVIDER: (OPENAI_FAMILY, "codex", "codex login status"),
    }[provider]
    executable = (
        resolve_cli_binary(binary, which=which)
        if which is not None
        else resolve_cli_binary(binary)
    )
    installed = executable is not None
    markers = tuple(marker_paths) if marker_paths is not None else _default_markers(provider, home)
    codex_chatgpt_auth = provider != CODEX_PROVIDER or _codex_uses_chatgpt_auth(markers[0])
    auth_present = _marker_exists(markers) if provider != CODEX_PROVIDER else codex_chatgpt_auth
    auth_verified = False
    last_verified = None
    if verify and installed and codex_chatgpt_auth:
        try:
            spec = harness_install_spec(key)
            if spec is None or spec.status_args is None:
                auth_verified = False
            else:
                result = subprocess.run(
                    [executable, *spec.status_args],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    shell=False,
                    timeout=READINESS_CLI_PROBE_TIMEOUT_S,
                    text=True,
                    env=subscription_environment(environ, transport=provider),
                )
                auth_verified = result.returncode == 0
                if spec.login_status_key is not None:
                    try:
                        payload = json.loads(result.stdout or "")
                    except (TypeError, ValueError):
                        payload = None
                    auth_verified = (
                        result.returncode == 0
                        and isinstance(payload, dict)
                        and payload.get(spec.login_status_key) is True
                    )
        except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError):
            auth_verified = False
        last_verified = _now(clock)
    auth_present = auth_present or auth_verified
    return SubscriptionReadiness(
        provider,
        _state(installed=installed, auth_present=auth_present, auth_verified=auth_verified),
        installed,
        auth_present,
        auth_verified,
        installed and auth_verified,
        ("cli-binary", "auth-marker-presence")
        + ((f"cli-status:{status_label}",) if verify and installed else ()),
        last_verified,
        (
            f"{provider} CLI is not installed"
            if not installed
            else "Codex subscription routing requires a ChatGPT login"
            if provider == CODEX_PROVIDER and not codex_chatgpt_auth
            else f"{status_label} did not verify subscription auth"
            if verify and not auth_verified
            else None
        ),
    )


def _readiness_for_gemini(
    *,
    verify: bool,
    home: Path | None,
    executable: GeminiExecutable | None,
    which: Callable[[str], str | None] | None,
    environ: Mapping[str, str] | None,
    auth_probe: Callable[[GeminiExecutable], bool] | None,
    clock: Callable[[], datetime] | None,
) -> SubscriptionReadiness:
    """Report Gemini installation/auth markers, but enforce Google's policy gate.

    Individual subscription traffic moved from Gemini CLI to Antigravity, and
    Antigravity's terms prohibit third-party products from using its OAuth
    subscription. A marker or even an enterprise-capable CLI must therefore
    never become subscription-router eligibility.
    """

    raw = gemini_readiness(
        executable=executable,
        which=which,
        environ=environ,
        home=home,
        auth_probe=None,
    )
    del auth_probe
    return SubscriptionReadiness(
        GEMINI_CLI_PROVIDER,
        ReadinessState.NOT_READY,
        raw.installed,
        raw.auth_present,
        False,
        False,
        ("official-gemini-cli", "oauth-marker-presence", "google-policy-gate"),
        _now(clock) if verify and raw.installed else None,
        GOOGLE_SUBSCRIPTION_POLICY_REASON,
    )


def _cursor_reason(error: str | None) -> str | None:
    return {
        "wsl-missing": "WSL is unavailable; enable WSL and restart Windows, then retry.",
        "distro-unavailable": (
            "The selected WSL distro is unavailable; install/start it or correct "
            "the explicit distro setting."
        ),
        "wsl2-required": (
            "The selected distro must run under WSL2; convert or recreate it, then retry."
        ),
        "ubuntu-required": "The selected WSL distro must identify as Ubuntu in /etc/os-release.",
        "cursor-user-unavailable": (
            "The selected Cursor WSL user does not exist in the selected distro."
        ),
        "cursor-user-uid-invalid": (
            "The selected Cursor WSL user did not report a valid numeric UID."
        ),
        "cursor-user-root": "Cursor WSL must run as a non-root Linux user.",
        "invalid-user": "Cursor WSL user must be a safe lowercase POSIX username.",
        "root-user-forbidden": "Cursor WSL must run as a non-root Linux user.",
        "cursor-agent-missing": (
            "Install cursor-agent inside the selected WSL distro, then retry."
        ),
        "cursor-auth-unavailable": (
            "Run cursor-agent login inside the selected WSL distro, then retry."
        ),
    }.get(error, error)


def _readiness_for_cursor(
    *,
    distro: str | None,
    distro_source: str | None,
    user: str | None,
    user_source: str | None,
    working_directory: str,
    which: Callable[[str], str | None] | None,
    command_runner: Callable[..., object] | None,
    clock: Callable[[], datetime] | None,
) -> SubscriptionReadiness:
    if not distro:
        if distro_source == "conflicting-config":
            return SubscriptionReadiness(
                CURSOR_WSL_PROVIDER,
                ReadinessState.NOT_READY,
                False,
                False,
                False,
                False,
                ("conflicting-config",),
                None,
                "Cursor WSL distro settings conflict; keep one explicit distro setting.",
            )
        return SubscriptionReadiness(
            CURSOR_WSL_PROVIDER,
            ReadinessState.NOT_READY,
            False,
            False,
            False,
            False,
            ("explicit-distro-required",),
            None,
            "Cursor WSL distro is not configured; set --cursor-distro, "
            "OMNIGENT_CURSOR_WSL_DISTRO, or cursor_wsl.distro.",
        )
    if not user:
        return SubscriptionReadiness(
            CURSOR_WSL_PROVIDER,
            ReadinessState.NOT_READY,
            False,
            False,
            False,
            False,
            ("explicit-user-required",),
            None,
            "Cursor WSL user is not configured; set OMNIGENT_CURSOR_WSL_USER or cursor_wsl.user.",
        )
    try:
        transport = CursorWslTransport(
            distro,
            validate_linux_user(user),
            working_directory,
            which=which,
            command_runner=command_runner,
            clock=clock,
        )
        raw = transport.readiness()
    except CursorWslError as exc:
        return SubscriptionReadiness(
            CURSOR_WSL_PROVIDER,
            ReadinessState.NOT_READY,
            False,
            False,
            False,
            False,
            (distro_source or "explicit-distro", user_source or "explicit-user"),
            None,
            _cursor_reason(exc.code) or "Cursor WSL readiness could not be evaluated.",
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return SubscriptionReadiness(
            CURSOR_WSL_PROVIDER,
            ReadinessState.NOT_READY,
            False,
            False,
            False,
            False,
            (distro_source or "explicit-distro", user_source or "explicit-user"),
            None,
            "Cursor WSL readiness could not be evaluated.",
        )
    return SubscriptionReadiness(
        CURSOR_WSL_PROVIDER,
        _state(
            installed=raw.cursor_agent_installed,
            auth_present=raw.auth_present,
            auth_verified=raw.auth_present,
            not_ready=not raw.transport_ready,
        ),
        raw.cursor_agent_installed,
        raw.auth_present,
        raw.auth_present,
        raw.transport_ready,
        (distro_source or "explicit-distro", user_source or "explicit-user", "wsl-cli-status"),
        raw.last_verified if raw.cursor_agent_installed else None,
        _cursor_reason(raw.error),
    )


def subscription_readiness(
    provider: str,
    *,
    verify: bool = False,
    home: Path | None = None,
    marker_paths: Sequence[Path] | None = None,
    executable: GeminiExecutable | None = None,
    which: Callable[[str], str | None] | None = None,
    environ: Mapping[str, str] | None = None,
    auth_probe: Callable[[GeminiExecutable], bool] | None = None,
    cursor_distro: str | None = None,
    cursor_user: str | None = None,
    cursor_config: Mapping[str, object] | None = None,
    cursor_working_directory: str | None = None,
    command_runner: Callable[..., object] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> SubscriptionReadiness:
    """Return one provider's safe readiness state.

    ``verify=False`` performs no provider request.  Claude/Codex status and
    Cursor WSL status are used only when ``verify=True`` (Cursor's transport
    status itself remains credential-blind and never returns its output).
    """

    canonical = canonical_provider(provider)
    if canonical in {CLAUDE_PROVIDER, CODEX_PROVIDER}:
        return _readiness_for_cli(
            canonical,
            verify=verify,
            home=home,
            marker_paths=marker_paths,
            which=which,
            environ=environ,
            clock=clock,
        )
    if canonical == GEMINI_CLI_PROVIDER:
        return _readiness_for_gemini(
            verify=verify,
            home=home,
            executable=executable,
            which=which,
            environ=environ,
            auth_probe=auth_probe,
            clock=clock,
        )
    distro, source = resolve_cursor_distro(
        option=cursor_distro, environ=environ, config=cursor_config
    )
    user, user_source = resolve_cursor_user(
        option=cursor_user, environ=environ, config=cursor_config
    )
    workdir = cursor_working_directory or (
        (environ or os.environ).get(CURSOR_WORKING_DIRECTORY_ENV) or "C:\\"
    )
    return _readiness_for_cursor(
        distro=distro,
        distro_source=source,
        user=user,
        user_source=user_source,
        working_directory=workdir,
        which=which,
        command_runner=command_runner,
        clock=clock,
    )


def subscription_readiness_map(**kwargs: object) -> dict[str, SubscriptionReadiness]:
    """Return additive readiness for every supported diagnostic provider."""

    return {
        provider: subscription_readiness(provider, **kwargs) for provider in SUPPORTED_PROVIDERS
    }


def _run_cli_probe(provider: str, *, timeout_s: float, environ: Mapping[str, str] | None) -> bool:
    if provider == CODEX_PROVIDER and not _codex_uses_chatgpt_auth(
        _codex_probe_auth_path(environ)
    ):
        return False
    binary = "claude" if provider == CLAUDE_PROVIDER else "codex"
    executable = resolve_cli_binary(binary)
    if executable is None:
        return False
    argv = (
        [
            executable,
            "--print",
            "--output-format",
            "text",
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--no-session-persistence",
            SAFE_PROBE_PROMPT,
        ]
        if provider == CLAUDE_PROVIDER
        else [
            executable,
            "-c",
            'model_provider="openai"',
            "exec",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--color",
            "never",
            "--skip-git-repo-check",
            SAFE_PROBE_PROMPT,
        ]
    )
    try:
        with tempfile.TemporaryDirectory(prefix="omnigent-subscription-probe-") as temp_dir:
            result = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                shell=False,
                timeout=timeout_s,
                text=True,
                env=subscription_environment(environ, transport=provider),
                cwd=temp_dir,
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return result.returncode == 0 and "SUBSCRIPTION_MVP_PROBE_OK" in (result.stdout or "")


def _run_gemini_probe(*, timeout_s: float, environ: Mapping[str, str] | None) -> bool:
    executable = discover_gemini_cli(environ=environ)
    if executable is None:
        return False
    result = asyncio.run(
        GeminiCliTransport(executable=executable, env=subscription_environment(environ)).run(
            SAFE_PROBE_PROMPT, timeout_s=timeout_s
        )
    )
    return result.completed and "SUBSCRIPTION_MVP_PROBE_OK" in result.text


def _run_cursor_probe(
    *,
    readiness: SubscriptionReadiness,
    distro: str | None,
    user: str | None,
    working_directory: str,
    timeout_s: float,
    which: Callable[[str], str | None] | None,
    command_runner: Callable[..., object] | None,
) -> bool:
    if not readiness.transport_ready or not distro or not user:
        return False
    try:
        result = CursorWslTransport(
            distro,
            user,
            working_directory,
            which=which,
            command_runner=command_runner,
            probe_timeout=min(timeout_s, 5.0),
        ).run(SAFE_PROBE_PROMPT, timeout=timeout_s)
    except (CursorWslError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return result.ok and "SUBSCRIPTION_MVP_PROBE_OK" in result.text


def verify_subscription_provider(
    provider: str,
    *,
    timeout_s: float = 10.0,
    no_api_keys: bool = True,
    **kwargs: object,
) -> SubscriptionReadiness:
    """Run one explicit bounded live probe and return only its safe verdict."""

    if timeout_s <= 0 or timeout_s > MAX_PROBE_TIMEOUT_S:
        raise ValueError(f"timeout must be between 0 and {MAX_PROBE_TIMEOUT_S:g} seconds")
    del no_api_keys  # Subscription probes always use the scrubbed environment.
    canonical = canonical_provider(provider)
    environ = kwargs.get("environ")
    if environ is not None and not isinstance(environ, Mapping):
        raise TypeError("environ must be a mapping")
    readiness = subscription_readiness(canonical, verify=True, **kwargs)
    if canonical == CODEX_PROVIDER and not readiness.auth_present:
        return readiness
    if canonical in {CLAUDE_PROVIDER, CODEX_PROVIDER}:
        ok = _run_cli_probe(canonical, timeout_s=timeout_s, environ=environ)
    elif canonical == GEMINI_CLI_PROVIDER:
        # Never invoke the Google subscription from a third-party product.
        # The readiness object already carries the permanent policy reason.
        return readiness
    else:
        distro, _ = resolve_cursor_distro(
            option=(
                kwargs.get("cursor_distro")
                if isinstance(kwargs.get("cursor_distro"), str)
                else None
            ),
            environ=environ,
            config=kwargs.get("cursor_config")
            if isinstance(kwargs.get("cursor_config"), Mapping)
            else None,
        )
        user, _ = resolve_cursor_user(
            option=(
                kwargs.get("cursor_user") if isinstance(kwargs.get("cursor_user"), str) else None
            ),
            environ=environ,
            config=kwargs.get("cursor_config")
            if isinstance(kwargs.get("cursor_config"), Mapping)
            else None,
        )
        workdir = kwargs.get("cursor_working_directory")
        if not isinstance(workdir, str) or not workdir:
            workdir = (environ or os.environ).get(CURSOR_WORKING_DIRECTORY_ENV) or "C:\\"
        ok = _run_cursor_probe(
            readiness=readiness,
            distro=distro,
            user=user,
            working_directory=workdir,
            timeout_s=timeout_s,
            which=kwargs.get("which") if callable(kwargs.get("which")) else None,
            command_runner=(
                kwargs.get("command_runner") if callable(kwargs.get("command_runner")) else None
            ),
        )
    if not ok:
        return replace(
            readiness,
            state=ReadinessState.NOT_READY,
            auth_verified=False,
            transport_ready=False,
            provenance=(*readiness.provenance, "bounded-live-probe"),
            reason="bounded subscription probe did not succeed",
        )
    timestamp = _now(kwargs.get("clock") if callable(kwargs.get("clock")) else None)
    return replace(
        readiness,
        state=ReadinessState.AUTH_VERIFIED,
        auth_present=True,
        auth_verified=True,
        transport_ready=True,
        provenance=(*readiness.provenance, "bounded-live-probe"),
        last_verified=timestamp,
        reason=None,
    )


# Compatibility spellings for callers that prefer a map/getter vocabulary.
get_subscription_readiness = subscription_readiness
collect_subscription_readiness = subscription_readiness_map


__all__ = [
    "ANTIGRAVITY_PROVIDER",
    "CLAUDE_PROVIDER",
    "CODEX_PROVIDER",
    "COMPETING_API_KEY_ENV_VARS",
    "CURSOR_DISTRO_ENV",
    "CURSOR_USER_ENV",
    "CURSOR_WSL_PROVIDER",
    "GEMINI_CLI_PROVIDER",
    "SUPPORTED_PROVIDERS",
    "ReadinessState",
    "SubscriptionReadiness",
    "canonical_provider",
    "collect_subscription_readiness",
    "get_subscription_readiness",
    "resolve_cursor_distro",
    "resolve_cursor_user",
    "subscription_environment",
    "subscription_readiness",
    "subscription_readiness_map",
    "verify_subscription_provider",
]
