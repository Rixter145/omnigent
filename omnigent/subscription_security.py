"""Shared fail-closed environment policy for subscription CLI launches."""

from __future__ import annotations

import os
from collections.abc import Mapping

_TRANSPORT_ALIASES = {
    "claude": "claude-sdk",
    "claude-sdk": "claude-sdk",
    "codex": "codex",
    "gemini": "gemini-cli",
    "gemini-cli": "gemini-cli",
    "cursor": "cursor-wsl",
    "cursor-wsl": "cursor-wsl",
}
_SELECTED_AUTH = {
    "claude-sdk": "CLAUDE_CODE_OAUTH_TOKEN",
    "codex": "CODEX_HOME",
}
_VENDOR_PREFIXES = (
    "ANTHROPIC_",
    "OPENAI_",
    "DATABRICKS_",
    "AWS_",
    "BEDROCK_",
    "GOOGLE_",
    "GEMINI_",
    "VERTEX_",
    "VERTEXAI_",
    "CLOUDSDK_",
    "AZURE_",
    "AZURE_OPENAI_",
    "CURSOR_",
    "ANTIGRAVITY_",
)
_EXACT_BLOCKED = frozenset(
    {
        "WSLENV",
        "CODEX_ACCESS_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLOUD_ML_REGION",
        "GCLOUD_PROJECT",
    }
)
_SENSITIVE_SUFFIXES = (
    "_API_KEY",
    "_API_TOKEN",
    "_ACCESS_TOKEN",
    "_AUTH_TOKEN",
    "_BEARER_TOKEN",
    "_TOKEN_FILE",
    "_CREDENTIALS",
    "_CREDENTIALS_FILE",
    "_CREDENTIAL_FILE",
    "_AUTHORIZATION_TOKEN",
    "_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "_CONTAINER_CREDENTIALS_FULL_URI",
    "_BASE_URL",
    "_ENDPOINT",
    "_ENDPOINT_URL",
    "_MODEL_PROVIDER",
)
_HARNESS_REDIRECT_SUFFIXES = (
    "API_KEY",
    "API_KEY_HELPER",
    "AUTH_COMMAND",
    "BASE_URL",
    "DATABRICKS_PROFILE",
    "GATEWAY",
    "MODEL_PROVIDER",
)


def subscription_transport(marker: str | None) -> str | None:
    """Return the canonical selected transport encoded in a private marker."""

    if marker is None:
        return None
    return _TRANSPORT_ALIASES.get(marker.strip().lower())


def subscription_marker(transport: str) -> str:
    """Encode a supported transport for the private spawn-environment marker."""

    selected = subscription_transport(transport)
    if selected is None:
        raise ValueError(f"unsupported subscription transport: {transport}")
    return selected


def is_subscription_unsafe_env_name(name: str) -> bool:
    """Whether *name* can select credentials or a non-subscription transport."""

    key = name.upper()
    if key in _EXACT_BLOCKED or key.endswith(_SENSITIVE_SUFFIXES):
        return True
    if key.startswith(_VENDOR_PREFIXES):
        return True
    for prefix in ("HARNESS_CLAUDE_SDK_", "HARNESS_CODEX_"):
        if key.startswith(prefix):
            suffix = key.removeprefix(prefix)
            return suffix in _HARNESS_REDIRECT_SUFFIXES or suffix.startswith(
                ("AUTH_", "GATEWAY_", "DATABRICKS_")
            )
    return False


def subscription_blocked_env_names() -> frozenset[str]:
    """Known uppercase names to unset in an already-running WSL environment."""

    return frozenset(
        {
            *_EXACT_BLOCKED,
            "CURSOR_API_KEY",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_CONTAINER_AUTHORIZATION_TOKEN",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
            "AWS_DEFAULT_PROFILE",
            "AWS_BEARER_TOKEN_BEDROCK",
            "CLOUDSDK_AUTH_ACCESS_TOKEN",
            "GOOGLE_API_KEY",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_PROJECT_ID",
            "GOOGLE_CLOUD_LOCATION",
            "GEMINI_API_KEY",
            "VERTEXAI_API_KEY",
            "VERTEXAI_PROJECT",
            "VERTEXAI_LOCATION",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_AD_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_API_BASE",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BEDROCK_BASE_URL",
            "DATABRICKS_TOKEN",
            "DATABRICKS_CONFIG_PROFILE",
            "CODEX_HOME",
            "CLAUDE_CODE_OAUTH_TOKEN",
        }
    )


def subscription_environment(
    environ: Mapping[str, str] | None = None, *, transport: str | None = None
) -> dict[str, str]:
    """Copy an environment with only the selected subscription login preserved.

    Name matching deliberately uses uppercase so Windows' case-insensitive
    environment is handled before a child process is created.  Values are never
    inspected, logged, or returned except for the one login material required
    by the selected transport.
    """

    selected = subscription_transport(transport)
    allowed_auth = _SELECTED_AUTH.get(selected or "")
    source = os.environ if environ is None else environ
    clean: dict[str, str] = {}
    for key, value in source.items():
        normalized = key.upper()
        if normalized == allowed_auth:
            clean[normalized] = value
        elif normalized in _SELECTED_AUTH.values():
            continue
        elif not is_subscription_unsafe_env_name(normalized):
            clean[key] = value
    return clean
