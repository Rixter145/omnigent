"""Focused split-A tests for the subscription routing integration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from omnigent.cli import _build_routing_backends, parse_routing_settings
from omnigent.entities.conversation import RoutingDecisionData, parse_item_data
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.runner.turn_routing import TurnRouteRequest, resolve_turn_route
from omnigent.server.routes._sessions import orchestration as orchestration_module
from omnigent.server.routing_backend import RoutingBackends, route_with_fallback
from omnigent.server.schemas import SessionEventInput
from omnigent.server.smart_routing import (
    RequiredRoutingUnavailable,
    RoutingResult,
    SubscriptionRoutingPolicyClient,
    _redact_receipt_value,
    build_routing_receipt,
)
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from tests.server.helpers import create_test_agent, echo_runner_client


class FakeJudge:
    provider = "codex"
    judge_model = "subscription-test"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.prompts: list[str] = []

    async def judge(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self.response


def readiness(*ready_providers: str) -> dict[str, dict[str, object]]:
    """Build a credential-blind readiness snapshot for policy tests."""
    result: dict[str, dict[str, object]] = {}
    for provider in ("claude", "codex", "gemini-cli", "cursor-wsl"):
        ready = provider in ready_providers
        result[provider] = {
            "state": "auth-verified" if ready else "not-ready",
            "installed": ready,
            "auth_present": ready,
            "auth_verified": ready,
            "transport_ready": ready,
            "reason": None if ready else f"{provider} unavailable",
        }
    return result


def test_subscription_config_is_self_contained_and_defaults_safely() -> None:
    settings = parse_routing_settings({"provider": "subscription"})
    assert settings.provider == "subscription"
    assert settings.judge_order == ("codex", "claude")
    assert settings.required is False

    required = parse_routing_settings(
        {"provider": "subscription", "required": True, "judge_order": ["claude", "codex"]}
    )
    assert required.required is True
    assert required.judge_order == ("claude", "codex")

    backends = _build_routing_backends({"routing": {"provider": "subscription"}}, None, settings)
    assert backends.local is not None
    assert isinstance(backends.local, SubscriptionRoutingPolicyClient)


@pytest.mark.asyncio
async def test_zero_candidates_is_unavailable_and_one_is_deterministic() -> None:
    judge = FakeJudge('{"harness":"codex","model":"m","rationale":"judge"}')
    client = SubscriptionRoutingPolicyClient(
        transports=(judge,), readiness_provider=lambda: readiness("codex")
    )

    assert await client.route("task", {}) is None
    assert client.last_error == "no validated subscription candidates"
    result = await client.route("task", {"codex": ["subscription-codex"]})
    assert result == RoutingResult(
        model="subscription-codex",
        harness="codex",
        rationale="The only validated subscription candidate was selected deterministically.",
    )
    assert judge.calls == 0
    assert client.router_source == "deterministic"


@pytest.mark.asyncio
async def test_two_candidates_use_closed_menu_semantic_judge_and_dynamic_source() -> None:
    judge = FakeJudge('{"harness":"codex","model":"subscription-codex","rationale":"fits"}')
    client = SubscriptionRoutingPolicyClient(
        transports=(judge,), readiness_provider=lambda: readiness("claude", "codex")
    )
    result = await client.route(
        "task",
        {"claude": ["subscription-claude"], "codex": ["subscription-codex"]},
    )
    assert result is not None
    assert result.model == "subscription-codex"
    assert judge.calls == 1
    assert client.router_source == "subscription-codex"


@pytest.mark.asyncio
async def test_allowance_hint_is_a_soft_prompt_signal_and_never_reorders_candidates() -> None:
    judge = FakeJudge('{"harness":"claude","model":"subscription-claude","rationale":"tie"}')
    client = SubscriptionRoutingPolicyClient(
        transports=(judge,),
        allowance_hints={"preferred_models": ["subscription-claude", "not-offered"]},
        readiness_provider=lambda: readiness("claude", "codex"),
    )
    result = await client.route(
        "task",
        {"codex": ["subscription-codex"], "claude": ["subscription-claude"]},
    )
    assert result is not None
    assert result.model == "subscription-claude"
    assert set(client.last_metadata) >= {"selection_mode", "candidate_count"}
    prompt = judge.prompts[0]
    assert prompt.index("subscription-codex") < prompt.index("subscription-claude")
    assert "soft tie-break only" in prompt
    assert "after task capability and cross-vendor fit" in prompt
    assert "not-offered" not in prompt


@pytest.mark.asyncio
async def test_turn_route_pins_once_and_second_resolution_does_not_rejudge() -> None:
    """The persisted route decision is the once-only seam, not a hint."""
    from omnigent.runner.subagent_routing import ROUTING_DECISION_LABEL_KEY

    calls = 0
    conv = SimpleNamespace(cost_control_mode_override="on", labels={})

    async def route(_harness: str | None, _prompt: str) -> tuple[str, dict[str, object]]:
        nonlocal calls
        calls += 1
        return "subscription-codex", {"rationale": "deterministic"}

    async def pin(_model: str) -> bool:
        conv.labels[ROUTING_DECISION_LABEL_KEY] = "decision-1"
        return True

    first = await resolve_turn_route(
        "session",
        TurnRouteRequest(harness="codex", prompt="task"),
        conv=conv,
        route_turn=route,
        pin=pin,
    )
    second = await resolve_turn_route(
        "session",
        TurnRouteRequest(harness="codex", prompt="follow-up"),
        conv=conv,
        route_turn=route,
        pin=pin,
    )

    assert first.action == "route"
    assert second.action == "allow"
    assert calls == 1


@pytest.mark.asyncio
async def test_subscription_ignores_stale_gateway_catalog_and_uses_provider_default() -> None:
    """Subscription routing must offer a CLI default, never a Databricks id."""
    from omnigent.runtime import _globals
    from omnigent.server.smart_routing import route_session_harness

    settings = parse_routing_settings({"provider": "subscription"})
    caps = SimpleNamespace(
        routing_settings=settings,
        routing_backends=RoutingBackends(
            local=SubscriptionRoutingPolicyClient(
                transports=(), readiness_provider=lambda: readiness("codex")
            )
        ),
    )
    with patch.object(_globals, "_caps", caps):
        harness, model, verdict, error = await route_session_harness(
            "task",
            harness_candidates=("codex",),
            catalog={"codex": ["databricks-gpt-5-6-sol"]},
            gateway_backed=False,
            allow_static_fallback=False,
        )

    assert (harness, model, error) == ("codex", "gpt-subscription-default", None)
    assert verdict is not None
    assert verdict["receipt"]["selection_mode"] == "deterministic"
    assert verdict["receipt"]["candidates"] == {"codex": ["gpt-subscription-default"]}
    assert "databricks" not in json.dumps(verdict)


@pytest.mark.asyncio
async def test_subscription_turn_uses_fixed_harness_default_without_gateway_catalog() -> None:
    from omnigent.runtime import _globals
    from omnigent.server.smart_routing import route_turn

    client = SubscriptionRoutingPolicyClient(
        transports=(), readiness_provider=lambda: readiness("claude")
    )
    caps = SimpleNamespace(
        routing_settings=parse_routing_settings({"provider": "subscription"}),
        routing_backends=RoutingBackends(local=client),
    )
    with patch.object(_globals, "_caps", caps):
        model, verdict = await route_turn(
            "claude-sdk",
            "continue the implementation",
            catalog=["databricks-claude-opus-5"],
            gateway_backed=False,
            allow_static_fallback=False,
        )

    assert model == "claude-subscription-default"
    assert verdict is not None
    assert verdict["receipt"]["selection_mode"] == "deterministic"


@pytest.mark.asyncio
async def test_auto_subscription_menu_filters_unready_cursor_and_google_policy() -> None:
    from omnigent.runtime import _globals
    from omnigent.server.smart_routing import route_session_harness

    judge = FakeJudge(
        '{"harness":"codex","model":"gpt-subscription-default","rationale":"independent review"}'
    )
    client = SubscriptionRoutingPolicyClient(
        transports=(judge,),
        readiness_provider=lambda: readiness("claude", "codex", "gemini-cli"),
    )
    caps = SimpleNamespace(
        routing_settings=parse_routing_settings({"provider": "subscription"}),
        routing_backends=RoutingBackends(local=client),
    )
    with patch.object(_globals, "_caps", caps):
        harness, model, verdict, error = await route_session_harness(
            "Review this architecture independently",
            gateway_backed=False,
            allow_static_fallback=False,
        )

    assert (harness, model, error) == (
        "codex",
        "gpt-subscription-default",
        None,
    )
    assert verdict is not None
    receipt = verdict["receipt"]
    assert set(receipt["eligible_candidates"]) == {"claude-sdk", "codex"}
    assert receipt["excluded_candidates"]["cursor-wsl"] == "cursor-wsl unavailable"
    assert "third-party routing" in receipt["excluded_candidates"]["gemini-cli"]
    assert "antigravity-native" not in receipt["eligible_candidates"]
    assert "antigravity-native" not in receipt["excluded_candidates"]
    assert "pi" in receipt["excluded_candidates"] or "pi" not in receipt["candidates"]
    assert "harness: cursor-wsl" not in judge.prompts[0]


@pytest.mark.asyncio
async def test_tool_dependent_auto_agent_excludes_toolless_subscription_harnesses() -> None:
    """A Polly-like brain can never be handed to the one-turn CLI adapters."""
    from omnigent.runtime import _globals
    from omnigent.server.smart_routing import route_session_harness

    judge = FakeJudge(
        '{"harness":"codex","model":"gpt-subscription-default",'
        '"rationale":"tool-capable orchestrator"}'
    )
    client = SubscriptionRoutingPolicyClient(
        transports=(judge,),
        readiness_provider=lambda: readiness("claude", "codex", "gemini-cli", "cursor-wsl"),
    )
    caps = SimpleNamespace(
        routing_settings=parse_routing_settings({"provider": "subscription"}),
        routing_backends=RoutingBackends(local=client),
    )
    with patch.object(_globals, "_caps", caps):
        harness, model, verdict, error = await route_session_harness(
            "Coordinate implementation and dispatch workers",
            gateway_backed=False,
            allow_static_fallback=False,
            requires_tool_calling=True,
        )

    assert (harness, model, error) == ("codex", "gpt-subscription-default", None)
    assert verdict is not None
    receipt = verdict["receipt"]
    assert set(receipt["candidates"]) == {
        "claude-sdk",
        "codex",
        "cursor-wsl",
    }
    assert set(receipt["eligible_candidates"]) == {"claude-sdk", "codex"}
    assert "third-party routing" in receipt["excluded_candidates"]["gemini-cli"]
    assert "tool calling" in receipt["excluded_candidates"]["cursor-wsl"]
    assert "harness: gemini-cli" not in judge.prompts[0]
    assert "harness: cursor-wsl" not in judge.prompts[0]


def test_receipt_uses_leaf_judge_model_and_redacts_nested_untrusted_metadata() -> None:
    class UntrustedClient:
        last_metadata = {
            "judge": {"provider": "codex", "judge_model": "subscription-codex"},
            "attempts": [
                {
                    "provider": "codex",
                    "error": "authorization: Bearer TOP_SECRET_TOKEN",
                    "nested": {"api_key": "sk-live-should-not-appear"},
                }
            ],
            "readiness": {"auth": {"access_token": "opaque-secret"}},
            "validation": {"detail": "token=validation-secret"},
        }

    receipt = build_routing_receipt(
        task_summary="task token=summary-secret " + ("x" * 5000),
        candidates={"codex": ["subscription-codex"]},
        client=UntrustedClient(),
        result=RoutingResult(
            model="subscription-codex",
            harness="codex",
            rationale="rationale password=reason-secret",
        ),
        source="subscription-codex token=source-secret",
        user_override="Bearer override-secret",
    )
    encoded = json.dumps(receipt)

    assert receipt["judge_model"] == "subscription-codex"
    assert receipt["attempts"][0]["nested"]["api_key"] == "[REDACTED]"
    assert receipt["readiness"]["auth"]["access_token"] == "[REDACTED]"
    assert receipt["validation"]["detail"] == "[REDACTED]"
    assert "TOP_SECRET_TOKEN" not in encoded
    assert "sk-live-should-not-appear" not in encoded
    assert "opaque-secret" not in encoded
    assert "summary-secret" not in encoded
    assert "reason-secret" not in encoded
    assert "source-secret" not in encoded
    assert "override-secret" not in encoded
    assert len(encoded) < 20_000
    assert len(receipt["task_summary"]) <= 1000

    nested: object = "leaf"
    for _ in range(12):
        nested = {"next": nested}
    bounded = _redact_receipt_value(nested)
    assert bounded["next"]["next"]["next"]["next"]["next"]["next"] == "[TRUNCATED]"


@pytest.mark.asyncio
async def test_required_routing_exhaustion_is_actionable_503_at_message_orchestration(
    client: Any,
    db_uri: str,
) -> None:
    """Required exhaustion stops before runner forwarding with a structured error."""
    agent = await create_test_agent(client, include_llm=False)
    session_id = agent["_session_id"]
    patch_response = await client.patch(
        f"/v1/sessions/{session_id}",
        json={"cost_control_mode_override": "on"},
    )
    assert patch_response.status_code == 200, patch_response.text
    store = SqlAlchemyConversationStore(db_uri)
    conv = store.get_conversation(session_id)
    assert conv is not None
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "task"}]},
    )

    async def deny(*_args: object, **_kwargs: object) -> object:
        raise RequiredRoutingUnavailable("no validated subscription candidates")

    with patch("omnigent.server.smart_routing.route_turn_or_decline", deny):
        async with echo_runner_client() as runner:
            with pytest.raises(OmnigentError) as raised:
                await orchestration_module._forward_event_to_runner(
                    session_id,
                    conv,
                    body,
                    store,
                    runner,
                )

    assert raised.value.code == ErrorCode.RUNNER_UNAVAILABLE
    assert raised.value.http_status == 503
    assert "no message was sent" in raised.value.message.lower()
    assert "repair" in raised.value.message.lower()


@pytest.mark.asyncio
async def test_required_atomic_commit_failure_stops_before_runner_forward(
    client: Any,
    db_uri: str,
) -> None:
    """A required route exposes no pin, receipt, or forward on commit failure."""
    from omnigent.runner.subagent_routing import ROUTING_DECISION_LABEL_KEY

    agent = await create_test_agent(client, include_llm=False)
    session_id = agent["_session_id"]
    assert (
        await client.patch(
            f"/v1/sessions/{session_id}",
            json={"cost_control_mode_override": "on"},
        )
    ).status_code == 200
    store = SqlAlchemyConversationStore(db_uri)
    conv = store.get_conversation(session_id)
    assert conv is not None
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "task"}]},
    )
    model = "gpt-subscription-default"

    async def route(*_args: object, **_kwargs: object) -> tuple[str, dict[str, object], None]:
        return (
            model,
            {
                "rationale": "subscription judge pick",
                "routing_required": True,
                "receipt": {"selection_mode": "deterministic"},
            },
            None,
        )

    def fail_atomic_commit(*_args: object, **_kwargs: object) -> Any:
        raise OSError("store unavailable")

    forwarded_events: list[str] = []

    def handle_runner(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            forwarded_events.append(request.url.path)
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(
        base_url="http://runner.test", transport=httpx.MockTransport(handle_runner)
    ) as runner:
        with (
            patch("omnigent.server.smart_routing.route_turn_or_decline", route),
            patch.object(
                orchestration_module,
                "_routed_turn_model_spelling",
                return_value=model,
            ),
            patch.object(store, "commit_routing_decision", fail_atomic_commit),
        ):
            with pytest.raises(OmnigentError) as raised:
                await orchestration_module._forward_event_to_runner(
                    session_id,
                    conv,
                    body,
                    store,
                    runner,
                )

    assert raised.value.http_status == 503
    assert forwarded_events == []
    reloaded = store.get_conversation(session_id)
    assert reloaded is not None
    assert reloaded.model_override is None
    assert ROUTING_DECISION_LABEL_KEY not in reloaded.labels
    assert [
        item for item in store.list_items(session_id).data if item.type == "routing_decision"
    ] == []


@pytest.mark.asyncio
async def test_required_atomic_commit_failure_retry_routes_and_commits_once(
    client: Any,
    db_uri: str,
) -> None:
    """A failed required commit leaves the same turn routable on retry."""
    from omnigent.runner.subagent_routing import ROUTING_DECISION_LABEL_KEY

    agent = await create_test_agent(client, include_llm=False)
    session_id = agent["_session_id"]
    assert (
        await client.patch(
            f"/v1/sessions/{session_id}",
            json={"cost_control_mode_override": "on"},
        )
    ).status_code == 200
    store = SqlAlchemyConversationStore(db_uri)
    conv = store.get_conversation(session_id)
    assert conv is not None
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "task"}]},
    )
    model = "gpt-subscription-default"

    route_calls = 0

    async def route(*_args: object, **_kwargs: object) -> tuple[str, dict[str, object], None]:
        nonlocal route_calls
        route_calls += 1
        return (
            model,
            {
                "rationale": "subscription judge pick",
                "routing_required": True,
                "receipt": {"selection_mode": "deterministic"},
            },
            None,
        )

    real_commit = store.commit_routing_decision
    commit_calls = 0

    def fail_once_then_commit(*args: object, **kwargs: object) -> Any:
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 1:
            raise OSError("injected atomic commit failure")
        return real_commit(*args, **kwargs)

    forwarded_events: list[str] = []

    def handle_runner(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            forwarded_events.append(request.url.path)
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(
        base_url="http://runner.test", transport=httpx.MockTransport(handle_runner)
    ) as runner:
        with (
            patch("omnigent.server.smart_routing.route_turn_or_decline", route),
            patch.object(
                orchestration_module,
                "_routed_turn_model_spelling",
                return_value=model,
            ),
            patch.object(store, "commit_routing_decision", fail_once_then_commit),
        ):
            with pytest.raises(OmnigentError) as raised:
                await orchestration_module._forward_event_to_runner(
                    session_id,
                    conv,
                    body,
                    store,
                    runner,
                )

            after_failure = store.get_conversation(session_id)
            assert after_failure is not None
            assert after_failure.model_override is None
            assert ROUTING_DECISION_LABEL_KEY not in after_failure.labels
            assert forwarded_events == []
            assert [
                item
                for item in store.list_items(session_id).data
                if item.type == "message" and item.data.role == "user"
            ] == []

            await orchestration_module._forward_event_to_runner(
                session_id,
                after_failure,
                body,
                store,
                runner,
            )

    assert raised.value.http_status == 503
    assert route_calls == 2
    assert commit_calls == 2
    assert len(forwarded_events) == 1
    committed = store.get_conversation(session_id)
    assert committed is not None
    assert committed.model_override == model
    assert ROUTING_DECISION_LABEL_KEY in committed.labels
    receipts = [
        item for item in store.list_items(session_id).data if item.type == "routing_decision"
    ]
    assert len(receipts) == 1
    user_inputs = [
        item
        for item in store.list_items(session_id).data
        if item.type == "message" and item.data.role == "user"
    ]
    assert len(user_inputs) == 1


@pytest.mark.asyncio
async def test_required_title_failure_is_nonfatal_without_retry_duplicate(
    client: Any,
    db_uri: str,
) -> None:
    """A title write failure cannot make an accepted required turn retryable."""
    from omnigent.runner.subagent_routing import ROUTING_DECISION_LABEL_KEY

    agent = await create_test_agent(client, include_llm=False)
    session_id = agent["_session_id"]
    assert (
        await client.patch(
            f"/v1/sessions/{session_id}",
            json={"cost_control_mode_override": "on"},
        )
    ).status_code == 200
    store = SqlAlchemyConversationStore(db_uri)
    conv = store.get_conversation(session_id)
    assert conv is not None
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "task"}]},
    )
    model = "gpt-subscription-default"

    async def route(*_args: object, **_kwargs: object) -> tuple[str, dict[str, object], None]:
        return (
            model,
            {
                "rationale": "subscription judge pick",
                "routing_required": True,
                "receipt": {"selection_mode": "deterministic"},
            },
            None,
        )

    forwarded_events: list[str] = []

    def handle_runner(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            forwarded_events.append(request.url.path)
        return httpx.Response(202, json={"queued": True})

    async def fail_title_seed(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected title storage failure")

    async with httpx.AsyncClient(
        base_url="http://runner.test", transport=httpx.MockTransport(handle_runner)
    ) as runner:
        with (
            patch("omnigent.server.smart_routing.route_turn_or_decline", route),
            patch.object(
                orchestration_module,
                "_routed_turn_model_spelling",
                return_value=model,
            ),
            patch.object(
                orchestration_module,
                "_seed_missing_title_from_user_message",
                fail_title_seed,
            ),
        ):
            item_id = await orchestration_module._forward_event_to_runner(
                session_id,
                conv,
                body,
                store,
                runner,
            )

    assert item_id
    assert forwarded_events == [f"/v1/sessions/{session_id}/events"]
    committed = store.get_conversation(session_id)
    assert committed is not None
    assert committed.model_override == model
    assert ROUTING_DECISION_LABEL_KEY in committed.labels
    receipts = [
        item for item in store.list_items(session_id).data if item.type == "routing_decision"
    ]
    assert len(receipts) == 1
    user_inputs = [
        item
        for item in store.list_items(session_id).data
        if item.type == "message" and item.data.role == "user"
    ]
    # A successful return gives the caller no failed request to replay.
    assert len(user_inputs) == 1


@pytest.mark.asyncio
async def test_same_idempotency_key_reuses_the_persisted_input_and_delivery(
    client: Any,
    db_uri: str,
) -> None:
    """A retry after a lost response must not create or run another turn."""
    agent = await create_test_agent(client, include_llm=False)
    session_id = agent["_session_id"]
    store = SqlAlchemyConversationStore(db_uri)
    conv = store.get_conversation(session_id)
    assert conv is not None
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "task"}]},
        idempotency_key="retry-after-lost-response",
    )
    forwarded: list[dict[str, Any]] = []

    def handle_runner(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            forwarded.append(json.loads(request.content))
        return httpx.Response(202, json={"status": "accepted"})

    async with httpx.AsyncClient(
        base_url="http://runner.test", transport=httpx.MockTransport(handle_runner)
    ) as runner:
        first_item_id = await orchestration_module._forward_event_to_runner(
            session_id, conv, body, store, runner
        )
        retried_item_id = await orchestration_module._forward_event_to_runner(
            session_id, conv, body, store, runner
        )

    assert retried_item_id == first_item_id
    assert len(forwarded) == 1
    user_inputs = [
        item
        for item in store.list_items(session_id).data
        if item.type == "message" and item.data.role == "user"
    ]
    assert [item.id for item in user_inputs] == [first_item_id]


@pytest.mark.asyncio
async def test_lost_runner_response_keeps_same_key_single_dispatch(
    client: Any,
    db_uri: str,
) -> None:
    """An ambiguous accepted POST is never re-forwarded by a same-key retry."""
    agent = await create_test_agent(client, include_llm=False)
    session_id = agent["_session_id"]
    store = SqlAlchemyConversationStore(db_uri)
    conv = store.get_conversation(session_id)
    assert conv is not None
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "task"}]},
        idempotency_key="response-lost-after-accept",
    )
    accepted_payloads: list[dict[str, Any]] = []

    def accept_then_drop_response(request: httpx.Request) -> httpx.Response:
        accepted_payloads.append(json.loads(request.content))
        raise httpx.ReadTimeout("response lost after runner accepted")

    async with httpx.AsyncClient(
        base_url="http://runner.test", transport=httpx.MockTransport(accept_then_drop_response)
    ) as runner:
        with pytest.raises(OmnigentError, match="persisted"):
            await orchestration_module._forward_event_to_runner(
                session_id, conv, body, store, runner
            )
        retried_item_id = await orchestration_module._forward_event_to_runner(
            session_id, conv, body, store, runner
        )

    assert len(accepted_payloads) == 1
    assert retried_item_id == accepted_payloads[0]["persisted_item_id"]
    inputs = [
        item
        for item in store.list_items(session_id).data
        if item.type == "message" and item.data.role == "user"
    ]
    assert [item.id for item in inputs] == [retried_item_id]
    assert inputs[0].status == "in_progress"


@pytest.mark.asyncio
async def test_runner_rejection_retries_the_same_durable_input_once(
    client: Any,
    db_uri: str,
) -> None:
    """A definitive rejection may retry delivery, but may not append another input."""
    agent = await create_test_agent(client, include_llm=False)
    session_id = agent["_session_id"]
    store = SqlAlchemyConversationStore(db_uri)
    conv = store.get_conversation(session_id)
    assert conv is not None
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "task"}]},
        idempotency_key="retry-after-definitive-rejection",
    )
    payloads: list[dict[str, Any]] = []

    def reject_once_then_accept(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(503, json={"detail": "runner booting"})
        return httpx.Response(202, json={"status": "accepted"})

    async with httpx.AsyncClient(
        base_url="http://runner.test", transport=httpx.MockTransport(reject_once_then_accept)
    ) as runner:
        with pytest.raises(OmnigentError, match="rejected"):
            await orchestration_module._forward_event_to_runner(
                session_id, conv, body, store, runner
            )
        item_id = await orchestration_module._forward_event_to_runner(
            session_id, conv, body, store, runner
        )

    assert len(payloads) == 2
    assert payloads[0]["persisted_item_id"] == payloads[1]["persisted_item_id"] == item_id
    inputs = [
        item
        for item in store.list_items(session_id).data
        if item.type == "message" and item.data.role == "user"
    ]
    assert [item.id for item in inputs] == [item_id]
    assert inputs[0].status == "completed"


@pytest.mark.asyncio
async def test_post_accept_routing_card_failure_is_nonfatal_and_idempotent(
    client: Any,
    db_uri: str,
) -> None:
    """A routing-card or parent-mirror failure cannot make a turn replayable."""
    agent = await create_test_agent(client, include_llm=False)
    session_id = agent["_session_id"]
    assert (
        await client.patch(f"/v1/sessions/{session_id}", json={"cost_control_mode_override": "on"})
    ).status_code == 200
    store = SqlAlchemyConversationStore(db_uri)
    conv = store.get_conversation(session_id)
    assert conv is not None
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "task"}]},
        idempotency_key="card-write-fails-after-202",
    )
    model = "gpt-subscription-default"
    forwarded: list[dict[str, Any]] = []

    async def route(*_args: object, **_kwargs: object) -> tuple[str, dict[str, object], None]:
        return model, {"rationale": "optional card", "routing_required": False}, None

    async def fail_card(*_args: object, **_kwargs: object) -> str:
        raise OSError("injected card/parent metadata failure")

    def handle_runner(request: httpx.Request) -> httpx.Response:
        forwarded.append(json.loads(request.content))
        return httpx.Response(202, json={"status": "accepted"})

    async with httpx.AsyncClient(
        base_url="http://runner.test", transport=httpx.MockTransport(handle_runner)
    ) as runner:
        with (
            patch("omnigent.server.smart_routing.route_turn_or_decline", route),
            patch.object(
                orchestration_module,
                "_routed_turn_model_spelling",
                return_value=model,
            ),
            patch.object(orchestration_module, "_emit_server_routing_decision", fail_card),
        ):
            item_id = await orchestration_module._forward_event_to_runner(
                session_id, conv, body, store, runner
            )
            retried_item_id = await orchestration_module._forward_event_to_runner(
                session_id, conv, body, store, runner
            )

    assert retried_item_id == item_id
    assert len(forwarded) == 1
    inputs = [
        item
        for item in store.list_items(session_id).data
        if item.type == "message" and item.data.role == "user"
    ]
    assert [item.id for item in inputs] == [item_id]


@pytest.mark.asyncio
async def test_required_auto_route_ignores_cross_family_turn_override_atomically(
    client: Any,
    db_uri: str,
) -> None:
    """An unresolved auto route forwards and persists exactly one selected pair."""
    from omnigent.runner.subagent_routing import ROUTING_DECISION_LABEL_KEY

    agent = await create_test_agent(client, include_llm=False)
    session_id = agent["_session_id"]
    store = SqlAlchemyConversationStore(db_uri)
    updated = store.update_conversation(
        session_id,
        cost_control_mode_override="on",
        harness_override="auto",
    )
    assert updated is not None
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "task"}]},
        model_override="claude-opus-cross-family",
        idempotency_key="auto-route-cross-family-retry",
    )
    model = "gpt-subscription-codex"
    route_calls = 0

    async def route(*_args: object, **_kwargs: object) -> tuple[str, str, dict[str, Any], None]:
        nonlocal route_calls
        route_calls += 1
        return (
            "codex",
            model,
            {
                "rationale": "subscription judge pick",
                "routing_required": True,
                "receipt": {
                    "selection_mode": "deterministic",
                    "final_route": {"harness": "codex", "model": model},
                },
            },
            None,
        )

    forwarded: list[dict[str, Any]] = []

    def handle_runner(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            forwarded.append(json.loads(request.content))
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(
        base_url="http://runner.test", transport=httpx.MockTransport(handle_runner)
    ) as runner:
        with patch("omnigent.server.smart_routing.route_session_harness", route):
            first_item_id = await orchestration_module._forward_event_to_runner(
                session_id,
                updated,
                body,
                store,
                runner,
            )
            retried_item_id = await orchestration_module._forward_event_to_runner(
                session_id,
                updated,
                body,
                store,
                runner,
            )

    assert retried_item_id == first_item_id
    assert route_calls == 1
    assert len(forwarded) == 1
    assert forwarded[0]["harness_override"] == "codex"
    assert forwarded[0]["model_override"] == model
    committed = store.get_conversation(session_id)
    assert committed is not None
    assert committed.harness_override == "codex"
    assert committed.model_override == model
    assert ROUTING_DECISION_LABEL_KEY in committed.labels
    receipts = [
        item for item in store.list_items(session_id).data if item.type == "routing_decision"
    ]
    assert len(receipts) == 1
    receipt = receipts[0].data
    assert isinstance(receipt, RoutingDecisionData)
    assert receipt.harness == forwarded[0]["harness_override"]
    assert receipt.model == forwarded[0]["model_override"]
    assert receipt.receipt is not None
    assert receipt.receipt["final_route"] == {
        "harness": forwarded[0]["harness_override"],
        "model": forwarded[0]["model_override"],
    }


@pytest.mark.asyncio
async def test_required_named_cursor_child_keeps_its_pinned_subscription_route(
    client: Any,
    db_uri: str,
) -> None:
    """A named Cursor worker is a user pin, not an auto-routing candidate."""
    from omnigent.runner.subagent_routing import ROUTING_DECISION_LABEL_KEY
    from omnigent.runtime import _globals

    agent = await create_test_agent(
        client,
        name="required-pinned-cursor-child",
        executor={
            "type": "omnigent",
            "config": {"harness": "claude-sdk", "smart_routing_harness": "auto"},
        },
        sub_agents=[
            {
                "name": "cursor",
                "executor": {"type": "omnigent", "config": {"harness": "cursor-wsl"}},
            }
        ],
    )
    parent = await client.post(
        "/v1/sessions",
        json={
            "agent_id": agent["id"],
            "cost_control_mode_override": "on",
            "subagent_routing_override": "on",
        },
    )
    assert parent.status_code == 201, parent.text
    child = await client.post(
        "/v1/sessions",
        json={
            "agent_id": agent["id"],
            "parent_session_id": parent.json()["id"],
            "sub_agent_name": "cursor",
            "harness_override": "cursor-wsl",
            "title": "cursor:required-tooling",
        },
    )
    assert child.status_code == 201, child.text

    store = SqlAlchemyConversationStore(db_uri)
    child_conv = store.get_conversation(str(child.json()["id"]))
    assert child_conv is not None
    assert child_conv.harness_override == "cursor-wsl"

    settings = parse_routing_settings({"provider": "subscription", "required": True})
    caps = SimpleNamespace(
        routing_settings=settings,
        routing_backends=RoutingBackends(
            local=SubscriptionRoutingPolicyClient(
                transports=(), readiness_provider=lambda: readiness("cursor-wsl")
            )
        ),
    )
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "inspect this"}]},
        idempotency_key="required-pinned-cursor-child",
    )
    forwarded: list[dict[str, Any]] = []

    def handle_runner(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            forwarded.append(json.loads(request.content))
        return httpx.Response(202, json={"queued": True})

    async with httpx.AsyncClient(
        base_url="http://runner.test", transport=httpx.MockTransport(handle_runner)
    ) as runner:
        with (
            patch.object(_globals, "_caps", caps),
            patch.object(orchestration_module, "_gateway_backed", return_value=False),
        ):
            first_item_id = await orchestration_module._forward_event_to_runner(
                child_conv.id,
                child_conv,
                body,
                store,
                runner,
                # The events route gets this from the resolved named child spec.
                requires_tool_calling=True,
            )
            retried_item_id = await orchestration_module._forward_event_to_runner(
                child_conv.id,
                child_conv,
                body,
                store,
                runner,
                requires_tool_calling=True,
            )

    assert retried_item_id == first_item_id
    assert len(forwarded) == 1
    assert forwarded[0]["harness_override"] == "cursor-wsl"
    assert forwarded[0]["model_override"] == "auto-smart"
    committed = store.get_conversation(child_conv.id)
    assert committed is not None
    assert committed.harness_override == forwarded[0]["harness_override"]
    assert committed.model_override == forwarded[0]["model_override"]
    assert ROUTING_DECISION_LABEL_KEY in committed.labels
    receipts = [
        item for item in store.list_items(child_conv.id).data if item.type == "routing_decision"
    ]
    assert len(receipts) == 1
    receipt = receipts[0].data
    assert isinstance(receipt, RoutingDecisionData)
    assert receipt.receipt is not None
    assert receipt.receipt["final_route"] == {
        "harness": forwarded[0]["harness_override"],
        "model": forwarded[0]["model_override"],
    }


@pytest.mark.asyncio
async def test_required_turn_routing_denies_without_replay_route() -> None:
    async def unavailable(_harness: str | None, _prompt: str) -> tuple[None, None]:
        raise RequiredRoutingUnavailable("no validated subscription candidates")

    decision = await resolve_turn_route(
        "session",
        TurnRouteRequest(harness="codex", prompt="task"),
        conv=SimpleNamespace(cost_control_mode_override="on", labels={}),
        route_turn=unavailable,
    )
    assert decision.action == "deny"
    assert decision.terminal is True
    assert "required subscription routing" in decision.rationale.lower()
    assert "receipt" in decision.to_payload()


@pytest.mark.asyncio
async def test_legacy_turn_routing_remains_fail_open() -> None:
    async def unavailable(_harness: str | None, _prompt: str) -> tuple[None, None]:
        raise RuntimeError("router down")

    decision = await resolve_turn_route(
        "session",
        TurnRouteRequest(harness="codex", prompt="task"),
        conv=SimpleNamespace(cost_control_mode_override="on", labels={}),
        route_turn=unavailable,
    )
    assert decision.action == "allow"


def test_receipt_round_trips_as_an_additive_legacy_safe_field() -> None:
    receipt = {
        "task_summary": "bounded task",
        "candidates": {"codex": ["subscription-codex"]},
        "readiness": {},
        "judge_provider": "codex",
        "judge_model": "subscription-test",
        "attempts": [],
        "rationale": "deterministic",
        "validation": "validated candidate menu",
        "final_route": {"harness": "codex", "model": "subscription-codex"},
        "fallback_chain": ["deterministic"],
        "user_override": None,
    }
    data = parse_item_data(
        "routing_decision",
        {"model": "subscription-codex", "applied": True, "rationale": "ok", "receipt": receipt},
    )
    assert isinstance(data, RoutingDecisionData)
    assert data.receipt == receipt
    assert RoutingDecisionData(model="legacy", applied=False, rationale="old").receipt is None


@pytest.mark.asyncio
async def test_backend_preserves_subscription_source() -> None:
    @dataclass
    class Client:
        router_source: str = "subscription-claude"

        async def route(self, _message: str, _models: dict[str, list[str]]) -> RoutingResult:
            return RoutingResult(model="m", rationale="ok", harness="claude")

    call = await route_with_fallback(
        RoutingBackends(local=Client()), "task", {"claude": ["m"]}, gateway_backed=False
    )
    assert call is not None
    assert call.source == "subscription-claude"
