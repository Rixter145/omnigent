"""Runner-level proof that a routed subscription owns execution auth."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnigent.runner import create_runner_app
from omnigent.runner.app import _build_spawn_env_from_spec
from omnigent.runtime.harnesses.process_manager import _HARNESS_SUBSCRIPTION_AUTH_ENV
from omnigent.spec import AgentSpec, ExecutorSpec, ProviderAuth
from omnigent.subscription_defaults import (
    CLAUDE_SUBSCRIPTION_DEFAULT,
    CODEX_SUBSCRIPTION_DEFAULT,
    provider_default_transport_model,
)
from tests.runner.conftest import _FakeProcessManager, _runner_client, _ScriptedHarnessClient, _sse
from tests.runner.helpers import NullServerClient


@pytest.mark.parametrize(
    ("harness", "model", "expected"),
    [
        ("claude-sdk", CLAUDE_SUBSCRIPTION_DEFAULT, None),
        ("codex", CODEX_SUBSCRIPTION_DEFAULT, None),
        ("codex", "gpt-5.6", "gpt-5.6"),
    ],
)
def test_runner_in_band_transport_omits_only_matched_subscription_sentinels(
    harness: str,
    model: str,
    expected: str | None,
) -> None:
    """The runner's in-band helper strips valid sentinels but preserves models."""

    assert provider_default_transport_model(harness, model) == expected


@pytest.mark.parametrize(
    "harness",
    [None, "plugin-alias", "claude-sdk-v2", "codex"],
)
def test_runner_in_band_transport_rejects_unresolved_or_mismatched_sentinel(
    harness: str | None,
) -> None:
    """A sentinel cannot select an SDK transport through ambiguous metadata."""

    with pytest.raises(ValueError, match="sentinel"):
        provider_default_transport_model(harness, CLAUDE_SUBSCRIPTION_DEFAULT)


@pytest.mark.asyncio
async def test_executor_adapter_rejects_unresolved_subscription_sentinel() -> None:
    """No executor is constructed when a sentinel reaches the SDK boundary."""

    import asyncio

    from omnigent.runtime.harnesses._executor_adapter import ExecutorAdapter
    from omnigent.runtime.harnesses._scaffold import TurnContext
    from omnigent.server.schemas import CreateResponseRequest

    constructed = False

    def build_executor() -> Any:
        nonlocal constructed
        constructed = True
        raise AssertionError("subscription sentinel must not reach an executor")

    adapter = ExecutorAdapter(executor_factory=build_executor)
    request = CreateResponseRequest(
        model="subscription-agent",
        input="do not dispatch",
        model_override=CODEX_SUBSCRIPTION_DEFAULT,
    )
    ctx = TurnContext(
        response_id="resp_subscription_sentinel",
        event_queue=asyncio.Queue(),
        cancelled=asyncio.Event(),
    )

    with pytest.raises(ValueError, match="Subscription routing sentinel"):
        await adapter.run_turn(request, ctx)

    assert constructed is False


@pytest.mark.parametrize(
    ("harness", "sentinel", "provider_name"),
    [
        ("claude-sdk", CLAUDE_SUBSCRIPTION_DEFAULT, "metered-anthropic"),
        ("codex", CODEX_SUBSCRIPTION_DEFAULT, "metered-openai"),
    ],
)
def test_routed_subscription_model_replaces_source_spec_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    harness: str,
    sentinel: str,
    provider_name: str,
) -> None:
    """The persisted route sentinel survives spec copying and blocks metered auth."""

    monkeypatch.setenv("OMNIGENT_CONFIG_HOME", str(tmp_path))
    config = {
        "providers": {
            "metered-anthropic": {
                "kind": "key",
                "anthropic": {
                    "base_url": "https://metered-anthropic.invalid/v1",
                    "api_key": "sk-ant-must-not-run",
                    "models": {"default": "metered-claude"},
                },
            },
            "metered-openai": {
                "kind": "key",
                "openai": {
                    "base_url": "https://metered-openai.invalid/v1",
                    "api_key": "sk-openai-must-not-run",
                    "models": {"default": "metered-gpt"},
                },
            },
        }
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    source_spec = AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(
            type="omnigent",
            config={"harness": harness},
            model="source-model",
            auth=ProviderAuth(name=provider_name),
        ),
    )

    env = _build_spawn_env_from_spec(
        source_spec,
        harness,
        model_override=sentinel,
        cwd=tmp_path,
    )

    assert env is not None
    serialized = json.dumps(env, sort_keys=True)
    assert "must-not-run" not in serialized
    assert "metered-" not in serialized
    assert "GATEWAY" not in serialized
    if harness == "codex":
        assert env["HARNESS_CODEX_MODEL_PROVIDER"] == "openai"
        assert "HARNESS_CODEX_MODEL" not in env
    else:
        assert "HARNESS_CLAUDE_SDK_MODEL" not in env
        assert "HARNESS_CLAUDE_SDK_API_KEY_HELPER" not in env


class _LifecycleProcessManager(_FakeProcessManager):
    """Fake manager that records whether a routed turn rebinds before dispatch."""

    def __init__(self, client: _ScriptedHarnessClient) -> None:
        super().__init__(client)
        self.lifecycle: list[str] = []
        self.active_env: dict[str, str] | None = None

    async def get_client(
        self, conversation_id: str, harness: str, env: Any = None
    ) -> _ScriptedHarnessClient:
        next_env = dict(env or {})
        self.lifecycle.append("eager_spawn" if self.active_env is None else "replace")
        self.active_env = next_env
        return await super().get_client(conversation_id, harness, env)


class _LifecycleHarnessClient(_ScriptedHarnessClient):
    """Records the active launch contract at the instant prompt dispatch starts."""

    def __init__(self, frames: list[str]) -> None:
        super().__init__(frames)
        self.manager: _LifecycleProcessManager | None = None
        self.prompt_envs: list[dict[str, str]] = []

    def stream(self, method: str, url: str, *, json: dict[str, Any], timeout: Any) -> Any:
        assert self.manager is not None
        self.manager.lifecycle.append("prompt")
        self.prompt_envs.append(dict(self.manager.active_env or {}))
        return super().stream(method, url, json=json, timeout=timeout)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("harness_name", "provider_name", "provider_family", "source_model", "sentinel"),
    [
        (
            "claude-sdk",
            "metered-anthropic",
            "anthropic",
            "metered-claude",
            CLAUDE_SUBSCRIPTION_DEFAULT,
        ),
        ("codex", "metered-openai", "openai", "metered-gpt", CODEX_SUBSCRIPTION_DEFAULT),
    ],
)
async def test_subscription_first_turn_rebinds_eager_gateway_before_prompt_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    harness_name: str,
    provider_name: str,
    provider_family: str,
    source_model: str,
    sentinel: str,
) -> None:
    """Eager gateway state is replaced and no subscription sentinel is posted."""
    source_secret = "metered-secret-must-not-appear"
    source_url = (
        f"https://user:{source_secret}@metered.invalid/path/{source_secret}?token={source_secret}"
    )
    monkeypatch.setenv("OMNIGENT_CONFIG_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "providers": {
                    provider_name: {
                        "kind": "key",
                        provider_family: {
                            "base_url": source_url,
                            "api_key": source_secret,
                            "models": {"default": source_model},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    source_spec = AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(
            type="omnigent",
            config={"harness": harness_name},
            model=source_model,
            auth=ProviderAuth(name=provider_name),
        ),
    )

    async def resolve_spec(agent_id: str, session_id: str | None = None) -> AgentSpec:
        del agent_id, session_id
        return source_spec

    harness = _LifecycleHarnessClient(
        [
            _sse({"type": "response.created", "response": {"id": "resp_subscription"}}),
            _sse({"type": "response.completed", "response": {"id": "resp_subscription"}}),
        ]
    )
    manager = _LifecycleProcessManager(harness)
    harness.manager = manager
    app = create_runner_app(
        process_manager=manager,  # type: ignore[arg-type]
        spec_resolver=resolve_spec,
        server_client=NullServerClient(),  # type: ignore[arg-type]
    )

    caplog.set_level(logging.INFO, logger="omnigent.runner.app")
    async with _runner_client(app) as client:
        initialized = await client.post(
            "/v1/sessions",
            json={"session_id": "conv_subscription", "agent_id": "agent_subscription"},
        )
        assert initialized.status_code == 201, initialized.text

        routed_turn = await client.post(
            "/v1/sessions/conv_subscription/events",
            params={"stream": "true"},
            json={
                "type": "message",
                "role": "user",
                "agent_id": "agent_subscription",
                "model_override": sentinel,
                "content": [{"type": "input_text", "text": "use my subscription"}],
            },
        )
        assert routed_turn.status_code == 200, routed_turn.text
        _ = routed_turn.text

    eager_env = manager.get_client_calls[0][2]
    subscription_env = manager.get_client_calls[1][2]
    assert eager_env is not None
    assert subscription_env is not None
    prefix = f"HARNESS_{harness_name.upper().replace('-', '_')}"
    assert eager_env[f"{prefix}_GATEWAY"] == "true"
    assert subscription_env[_HARNESS_SUBSCRIPTION_AUTH_ENV] == harness_name
    assert f"{prefix}_GATEWAY" not in subscription_env
    assert f"{prefix}_GATEWAY_BASE_URL" not in subscription_env
    assert f"{prefix}_GATEWAY_AUTH_COMMAND" not in subscription_env
    assert manager.lifecycle == ["eager_spawn", "replace", "prompt"]
    assert harness.prompt_envs == [subscription_env]
    assert len(harness.posted_bodies) == 1
    assert "model_override" not in harness.posted_bodies[0]
    assert sentinel not in json.dumps(harness.posted_bodies[0], sort_keys=True)
    assert source_secret not in caplog.text
