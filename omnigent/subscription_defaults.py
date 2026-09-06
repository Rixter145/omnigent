"""Stable candidate ids for subscription CLIs that expose no model catalog.

The Claude and Codex ids are routing/persistence sentinels, not vendor
model claims. Spawn builders translate those sentinels back to an omitted
``--model`` argument, leaving the authenticated CLI to use its own configured
default. Cursor's documented ``auto-smart`` id is a real model and is forwarded
directly.

Google is intentionally not in :data:`SUBSCRIPTION_HARNESSES`. Gemini CLI no
longer serves individual Google subscriptions, while Antigravity's terms bar a
third-party product from using Antigravity OAuth. The legacy Gemini sentinel is
kept non-transportable so stale persisted state cannot reach another vendor.
"""

from __future__ import annotations

CLAUDE_SUBSCRIPTION_DEFAULT = "claude-subscription-default"
CODEX_SUBSCRIPTION_DEFAULT = "gpt-subscription-default"
GEMINI_SUBSCRIPTION_DEFAULT = "gemini-subscription-default"
CURSOR_SUBSCRIPTION_DEFAULT = "auto-smart"

_ALIASES = {
    "claude": "claude-sdk",
    "claude_sdk": "claude-sdk",
}

_DEFAULTS = {
    "claude-sdk": CLAUDE_SUBSCRIPTION_DEFAULT,
    "codex": CODEX_SUBSCRIPTION_DEFAULT,
    "cursor-wsl": CURSOR_SUBSCRIPTION_DEFAULT,
}

# These values are durable routing state, never vendor model identifiers.  In
# particular, do not infer their meaning from a caller-provided harness: that
# metadata can be stale, aliased, or absent by the time a turn reaches a
# transport boundary.  Cursor's ``auto-smart`` remains a documented concrete
# model and is intentionally excluded.
_NON_TRANSPORTABLE_SENTINELS = frozenset(
    {
        CLAUDE_SUBSCRIPTION_DEFAULT,
        CODEX_SUBSCRIPTION_DEFAULT,
        GEMINI_SUBSCRIPTION_DEFAULT,
    }
)

_PROVIDERS = {
    "claude-sdk": "claude",
    "codex": "codex",
    "cursor-wsl": "cursor-wsl",
}

# Claude Agent SDK and Codex app-server can consume Omnigent's function-tool
# schemas. Cursor's transport is a one-turn CLI adapter that owns its workspace
# tools internally, so it cannot drive Omnigent tools such as
# ``sys_session_send``. Keep this as a launch fact, not a judge hint.
_OMNIGENT_TOOL_CALLING_HARNESSES = frozenset({"claude-sdk", "codex"})

GOOGLE_SUBSCRIPTION_POLICY_REASON = (
    "Google individual subscriptions are not eligible for third-party routing: "
    "Gemini CLI no longer serves them, and Antigravity terms prohibit a "
    "third-party product from using Antigravity OAuth. Use Google's API-key or "
    "enterprise path outside subscription routing."
)

SUBSCRIPTION_POLICY_EXCLUSIONS = {
    "gemini-cli": GOOGLE_SUBSCRIPTION_POLICY_REASON,
}

SUBSCRIPTION_HARNESSES = tuple(_DEFAULTS)


def canonical_subscription_harness(harness: str | None) -> str | None:
    """Return a recognized MVP subscription harness id, if applicable."""

    if not isinstance(harness, str):
        return None
    normalized = harness.strip().lower()
    canonical = _ALIASES.get(normalized, normalized)
    return canonical if canonical in _DEFAULTS else None


def subscription_default_model(harness: str | None) -> str | None:
    """Return the honest provider-default candidate for *harness*."""

    canonical = canonical_subscription_harness(harness)
    return _DEFAULTS.get(canonical) if canonical is not None else None


def subscription_provider(harness: str | None) -> str | None:
    """Return the readiness provider corresponding to *harness*."""

    canonical = canonical_subscription_harness(harness)
    return _PROVIDERS.get(canonical) if canonical is not None else None


def subscription_harness_supports_tool_calling(harness: str | None) -> bool:
    """Whether an MVP subscription harness can call Omnigent-owned tools."""

    canonical = canonical_subscription_harness(harness)
    return canonical in _OMNIGENT_TOOL_CALLING_HARNESSES


def is_subscription_default_sentinel(model: str | None) -> bool:
    """Whether *model* is routing state that must never reach a vendor SDK."""

    return model in _NON_TRANSPORTABLE_SENTINELS


def provider_default_transport_model(harness: str | None, model: str | None) -> str | None:
    """Resolve a model for transport without allowing routing sentinels through.

    A known sentinel is omitted only for its exact canonical harness.  A
    missing, unknown, or mismatched harness is unsafe: omitting the value there
    could silently select a paid/default vendor route, while forwarding it
    would leak routing state to the SDK.  Reject that boundary instead.
    """

    canonical = canonical_subscription_harness(harness)
    if not is_subscription_default_sentinel(model):
        return model
    if canonical is None:
        raise ValueError(
            f"Subscription routing sentinel {model!r} cannot be transported without "
            "a recognized subscription harness."
        )
    expected = subscription_default_model(canonical)
    if model != expected:
        raise ValueError(
            f"Subscription routing sentinel {model!r} is not valid for harness "
            f"{canonical!r}; refusing to select a transport model."
        )
    if canonical != "cursor-wsl":
        return None
    return model


__all__ = [
    "CLAUDE_SUBSCRIPTION_DEFAULT",
    "CODEX_SUBSCRIPTION_DEFAULT",
    "CURSOR_SUBSCRIPTION_DEFAULT",
    "GEMINI_SUBSCRIPTION_DEFAULT",
    "GOOGLE_SUBSCRIPTION_POLICY_REASON",
    "SUBSCRIPTION_HARNESSES",
    "SUBSCRIPTION_POLICY_EXCLUSIONS",
    "canonical_subscription_harness",
    "is_subscription_default_sentinel",
    "provider_default_transport_model",
    "subscription_default_model",
    "subscription_harness_supports_tool_calling",
    "subscription_provider",
]
