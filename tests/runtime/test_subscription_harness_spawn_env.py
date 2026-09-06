"""Focused runner wiring checks for subscription-backed CLI harnesses."""

from pathlib import Path

import pytest

from omnigent.errors import OmnigentError
from omnigent.runner.app import _build_spawn_env_from_spec
from omnigent.runtime.harnesses import _HARNESS_MODULES
from omnigent.runtime.harnesses.process_manager import (
    _HARNESS_SUBSCRIPTION_AUTH_ENV,
    _build_harness_spawn_env,
    _resolve_module_path,
)
from omnigent.spec import AgentSpec, ExecutorSpec
from omnigent.subscription_defaults import (
    CLAUDE_SUBSCRIPTION_DEFAULT,
    CODEX_SUBSCRIPTION_DEFAULT,
    CURSOR_SUBSCRIPTION_DEFAULT,
    provider_default_transport_model,
)


def test_subscription_cli_spawn_envs_quarantine_google_consumer_oauth(monkeypatch) -> None:
    """Gemini consumer OAuth has no registered executable path."""
    with pytest.raises(RuntimeError, match="unknown harness"):
        _resolve_module_path("gemini-cli")
    assert _resolve_module_path("cursor-wsl") == "omnigent.inner.cursor_wsl_harness"
    assert "gemini" not in _HARNESS_MODULES
    assert "gemini-cli" not in _HARNESS_MODULES
    monkeypatch.setattr(
        "omnigent.runtime.workflow.load_config",
        lambda: {"cursor_wsl": {"distro": "Ubuntu-24.04", "user": "ricar"}},
    )
    workspace = Path(r"C:\Users\ricar\src\omnigent")

    gemini = AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(type="omnigent", model="gemini-2.5-pro"),
    )
    with pytest.raises(OmnigentError, match="consumer OAuth"):
        _build_spawn_env_from_spec(gemini, "gemini-cli", cwd=workspace)

    cursor = AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(type="omnigent", model="auto-smart"),
    )
    assert _build_spawn_env_from_spec(cursor, "cursor-wsl", cwd=workspace) == {
        _HARNESS_SUBSCRIPTION_AUTH_ENV: "cursor-wsl",
        "HARNESS_CURSOR_WSL_DISTRO": "Ubuntu-24.04",
        "HARNESS_CURSOR_WSL_USER": "ricar",
        "HARNESS_CURSOR_WSL_CWD": str(workspace),
        "HARNESS_CURSOR_WSL_MODEL": "auto-smart",
    }


@pytest.mark.parametrize(
    "config, cwd, message",
    [
        ({}, Path(r"C:\work"), "explicit `cursor_wsl.distro`"),
        ({"cursor_wsl": {"distro": "Ubuntu"}}, Path(r"C:\work"), "explicit `cursor_wsl.user`"),
        (
            {"cursor_wsl": {"distro": "Ubuntu", "user": "ricar"}},
            None,
            "Windows session working directory",
        ),
        (
            {"cursor_wsl": {"distro": "Ubuntu", "user": "ricar"}},
            Path(r"/tmp/work"),
            "absolute drive or UNC path",
        ),
    ],
)
def test_cursor_wsl_spawn_env_rejects_ambiguous_launch_inputs(
    monkeypatch, config: dict[str, object], cwd: Path | None, message: str
) -> None:
    """Cursor WSL never guesses a distro or translates a non-Windows cwd."""
    monkeypatch.setattr("omnigent.runtime.workflow.load_config", lambda: config)
    spec = AgentSpec(spec_version=1, executor=ExecutorSpec(type="omnigent"))

    with pytest.raises(OmnigentError, match=message):
        _build_spawn_env_from_spec(spec, "cursor-wsl", cwd=cwd)


@pytest.mark.parametrize(
    ("harness", "model", "model_key"),
    [
        ("claude-sdk", CLAUDE_SUBSCRIPTION_DEFAULT, "HARNESS_CLAUDE_SDK_MODEL"),
        ("codex", CODEX_SUBSCRIPTION_DEFAULT, "HARNESS_CODEX_MODEL"),
    ],
)
def test_provider_default_routing_sentinels_are_not_sent_to_vendor_clis(
    monkeypatch, tmp_path: Path, harness: str, model: str, model_key: str
) -> None:
    monkeypatch.setattr("omnigent.runtime.workflow.load_config", dict)
    if harness == "codex":
        monkeypatch.setattr(
            "omnigent.runtime.workflow.codex_cli_effective_auth_mode", lambda: "chatgpt"
        )
    spec = AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(type="omnigent", model=model),
    )

    env = _build_spawn_env_from_spec(
        spec,
        harness,
        cwd=tmp_path,
        model_override=model,
    )

    assert env is not None
    assert model_key not in env


def test_codex_subscription_spawn_requires_chatgpt_auth(monkeypatch) -> None:
    """A Codex API-key or PAT login cannot launch a subscription route."""
    monkeypatch.setattr("omnigent.runtime.workflow.load_config", dict)
    monkeypatch.setattr(
        "omnigent.runtime.workflow.codex_cli_effective_auth_mode", lambda: "apikey", raising=False
    )
    spec = AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(type="omnigent", model=CODEX_SUBSCRIPTION_DEFAULT),
    )

    with pytest.raises(OmnigentError, match="ChatGPT"):
        _build_spawn_env_from_spec(spec, "codex")


def test_cursor_auto_smart_is_forwarded_as_a_real_provider_model() -> None:
    assert (
        provider_default_transport_model("cursor-wsl", CURSOR_SUBSCRIPTION_DEFAULT)
        == CURSOR_SUBSCRIPTION_DEFAULT
    )


@pytest.mark.parametrize(
    ("harness", "sentinel", "hostile_env"),
    [
        (
            "claude-sdk",
            CLAUDE_SUBSCRIPTION_DEFAULT,
            {
                "HARNESS_CLAUDE_SDK_GATEWAY": "true",
                "HARNESS_CLAUDE_SDK_GATEWAY_BASE_URL": "https://metered.invalid/v1",
                "HARNESS_CLAUDE_SDK_GATEWAY_AUTH_COMMAND": "printf %s sk-ant-ambient",
                "HARNESS_CLAUDE_SDK_API_KEY_HELPER": "printf %s sk-ant-helper",
                "HARNESS_CLAUDE_SDK_DATABRICKS_PROFILE": "metered-profile",
                "ANTHROPIC_API_KEY": "sk-ant-ambient",
                "ANTHROPIC_AUTH_TOKEN": "sk-ant-token",
                "ANTHROPIC_BASE_URL": "https://anthropic-metered.invalid:8443",
                "ANTHROPIC_MODEL": "metered-claude",
                "ANTHROPIC_BEDROCK_BASE_URL": "https://bedrock.invalid:8443",
                "ANTHROPIC_VERTEX_PROJECT_ID": "metered-project",
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "CLAUDE_CODE_USE_VERTEX": "1",
                "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
                "AWS_ACCESS_KEY_ID": "aws-access-key",
                "AWS_SECRET_ACCESS_KEY": "aws-secret-key",
                "AWS_SESSION_TOKEN": "aws-session-token",
                "AWS_PROFILE": "metered-aws-profile",
                "AWS_DEFAULT_PROFILE": "metered-aws-default-profile",
                "AWS_WEB_IDENTITY_TOKEN_FILE": "metered-token-file",
                "AWS_CONTAINER_AUTHORIZATION_TOKEN": "metered-container-token",
                "AWS_CONTAINER_CREDENTIALS_FULL_URI": "https://metered.invalid/credentials",
                "CLOUDSDK_AUTH_ACCESS_TOKEN": "metered-cloud-token",
                "GOOGLE_APPLICATION_CREDENTIALS": "vertex-secret.json",
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
                "GOOGLE_API_KEY": "google-api-key",
                "GEMINI_API_KEY": "gemini-api-key",
                "AZURE_OPENAI_API_KEY": "azure-openai-api-key",
                "AZURE_OPENAI_AD_TOKEN": "azure-openai-ad-token",
                "AZURE_OPENAI_ENDPOINT": "https://metered.invalid/azure",
                "DATABRICKS_TOKEN": "dbx-token",
                "DATABRICKS_CONFIG_PROFILE": "metered-profile",
                "WSLENV": "OPENAI_API_KEY/u",
            },
        ),
        (
            "codex",
            CODEX_SUBSCRIPTION_DEFAULT,
            {
                "HARNESS_CODEX_GATEWAY": "true",
                "HARNESS_CODEX_GATEWAY_BASE_URL": "https://metered.invalid/v1",
                "HARNESS_CODEX_GATEWAY_AUTH_COMMAND": "printf %s sk-openai-ambient",
                "HARNESS_CODEX_API_KEY_HELPER": "printf %s sk-openai-helper",
                "HARNESS_CODEX_DATABRICKS_PROFILE": "metered-profile",
                "HARNESS_CODEX_MODEL_PROVIDER": "metered-provider",
                "OPENAI_API_KEY": "sk-openai-ambient",
                "OPENAI_BASE_URL": "https://openai-metered.invalid:8443",
                "AWS_ACCESS_KEY_ID": "aws-access-key",
                "AWS_SECRET_ACCESS_KEY": "aws-secret-key",
                "AWS_SESSION_TOKEN": "aws-session-token",
                "AWS_PROFILE": "metered-aws-profile",
                "AWS_DEFAULT_PROFILE": "metered-aws-default-profile",
                "AWS_WEB_IDENTITY_TOKEN_FILE": "metered-token-file",
                "AWS_CONTAINER_AUTHORIZATION_TOKEN": "metered-container-token",
                "AWS_CONTAINER_CREDENTIALS_FULL_URI": "https://metered.invalid/credentials",
                "CLOUDSDK_AUTH_ACCESS_TOKEN": "metered-cloud-token",
                "GOOGLE_APPLICATION_CREDENTIALS": "google-credentials.json",
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
                "GOOGLE_API_KEY": "google-api-key",
                "GEMINI_API_KEY": "gemini-api-key",
                "AZURE_OPENAI_API_KEY": "azure-openai-api-key",
                "AZURE_OPENAI_AD_TOKEN": "azure-openai-ad-token",
                "AZURE_OPENAI_ENDPOINT": "https://metered.invalid/azure",
                "DATABRICKS_TOKEN": "dbx-token",
                "DATABRICKS_CONFIG_PROFILE": "metered-profile",
                "WSLENV": "OPENAI_API_KEY/u",
            },
        ),
    ],
)
def test_subscription_spawn_neutralizes_hostile_ambient_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    harness: str,
    sentinel: str,
    hostile_env: dict[str, str],
) -> None:
    """Subscription CLI logins reject ambient and requested metered transport."""
    monkeypatch.setattr("omnigent.runtime.workflow.load_config", dict)
    if harness == "codex":
        monkeypatch.setattr(
            "omnigent.runtime.workflow.codex_cli_effective_auth_mode", lambda: "chatgpt"
        )
    intentional_env = {
        "CLAUDE_CODE_OAUTH_TOKEN": "intentional-claude-login",
        "CODEX_HOME": str(tmp_path / "codex-home"),
    }
    for key, value in hostile_env.items():
        monkeypatch.setenv(key, value)
    for key, value in intentional_env.items():
        monkeypatch.setenv(key, value)

    spec = AgentSpec(spec_version=1, executor=ExecutorSpec(type="omnigent", model=sentinel))
    spawn_env = _build_spawn_env_from_spec(spec, harness, cwd=tmp_path, model_override=sentinel)

    assert spawn_env is not None
    assert spawn_env[_HARNESS_SUBSCRIPTION_AUTH_ENV] == harness
    effective_env = _build_harness_spawn_env({**hostile_env, **spawn_env})
    assert _HARNESS_SUBSCRIPTION_AUTH_ENV not in effective_env
    for key, hostile_value in hostile_env.items():
        assert effective_env.get(key) != hostile_value
    selected_auth = "CLAUDE_CODE_OAUTH_TOKEN" if harness == "claude-sdk" else "CODEX_HOME"
    other_auth = "CODEX_HOME" if harness == "claude-sdk" else "CLAUDE_CODE_OAUTH_TOKEN"
    assert effective_env[selected_auth] == intentional_env[selected_auth]
    assert other_auth not in effective_env
    if harness == "codex":
        assert effective_env["HARNESS_CODEX_MODEL_PROVIDER"] == "openai"
