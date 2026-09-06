"""Focused tests for additive subscription readiness and safe verification."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from omnigent.onboarding import subscription_readiness as sr


def test_safe_environment_removes_competing_api_keys() -> None:
    env = sr.subscription_environment(
        {
            "PATH": "safe",
            "OPENAI_API_KEY": "secret",
            "CUSTOM_API_KEY": "secret",
            "GOOGLE_APPLICATION_CREDENTIALS": "file.json",
            "aws_web_identity_token_file": "file.json",
            "AWS_CONTAINER_AUTHORIZATION_TOKEN": "token",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI": "uri",
            "CLOUDSDK_AUTH_ACCESS_TOKEN": "token",
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "AZURE_OPENAI_ENDPOINT": "endpoint",
            "WSLENV": "OPENAI_API_KEY/u",
            "HOME": "home",
        }
    )
    assert env == {"PATH": "safe", "HOME": "home"}


def test_claude_and_codex_are_marker_only_until_explicit_status(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sr, "resolve_cli_binary", lambda name: f"/bin/{name}")
    marker = tmp_path / "opaque-auth"
    marker.write_text("DO-NOT-READ", encoding="utf-8")
    result = sr.subscription_readiness("claude", marker_paths=(marker,), verify=False)
    assert result.state is sr.ReadinessState.AUTH_PRESENT
    assert result.auth_present and not result.auth_verified
    assert result.transport_ready is False
    assert "DO-NOT-READ" not in repr(result)


def test_explicit_status_is_the_only_path_to_auth_verified(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sr, "resolve_cli_binary", lambda name: f"/bin/{name}")
    monkeypatch.setattr(sr, "codex_auth_effective_mode", lambda _path: "chatgpt", raising=False)
    calls: list[list[str]] = []

    def status(argv: list[str], **kwargs: object):
        calls.append(argv)
        assert kwargs["timeout"] == sr.READINESS_CLI_PROBE_TIMEOUT_S
        assert kwargs["env"] == {"PATH": "safe", "CODEX_HOME": "selected-login-home"}
        return type("Result", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(sr.subprocess, "run", status)
    result = sr.subscription_readiness(
        "codex",
        verify=True,
        environ={
            "PATH": "safe",
            "CODEX_HOME": "selected-login-home",
            "CLAUDE_CODE_OAUTH_TOKEN": "other-vendor-login",
            "OPENAI_API_KEY": "do-not-forward",
        },
        clock=lambda: datetime(2026, 9, 4, tzinfo=timezone.utc),
    )
    assert result.state is sr.ReadinessState.AUTH_VERIFIED
    assert result.auth_present
    assert result.transport_ready
    assert result.last_verified == "2026-09-04T00:00:00+00:00"
    assert calls


@pytest.mark.parametrize("mode", [None, "apikey", "pat"])
def test_codex_subscription_readiness_requires_chatgpt_effective_auth_mode(
    monkeypatch, tmp_path: Path, mode: str | None
) -> None:
    """Codex API-key and PAT logins must never become subscription-ready."""
    marker = tmp_path / ".codex" / "auth.json"
    marker.parent.mkdir()
    marker.write_text("credential material is never inspected by this test", encoding="utf-8")
    monkeypatch.setattr(sr, "resolve_cli_binary", lambda _name: "/bin/codex")
    monkeypatch.setattr(sr, "codex_auth_effective_mode", lambda _path: mode, raising=False)
    monkeypatch.setattr(
        sr.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": ""})(),
    )

    result = sr.subscription_readiness("codex", verify=True, marker_paths=(marker,))

    assert result.auth_present is False
    assert result.auth_verified is False
    assert result.transport_ready is False
    assert "credential material" not in repr(result)


@pytest.mark.parametrize(
    "stdout",
    ["", "not-json", "{}", '{"loggedIn": false}', '{"loggedIn": "true"}'],
)
def test_cli_status_key_requires_json_boolean_true(monkeypatch, stdout: str) -> None:
    """A keyed CLI status is authoritative only for a literal JSON ``true``."""
    monkeypatch.setattr(sr, "resolve_cli_binary", lambda _name: "/bin/claude")
    monkeypatch.setattr(
        sr.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": stdout})(),
    )

    result = sr.subscription_readiness("claude", verify=True, marker_paths=())

    assert result.auth_verified is False
    assert result.transport_ready is False


def test_codex_live_probe_does_not_run_without_chatgpt_auth(monkeypatch, tmp_path: Path) -> None:
    """A non-ChatGPT Codex login is rejected before a live CLI prompt."""
    marker = tmp_path / ".codex" / "auth.json"
    marker.parent.mkdir()
    marker.write_text("api-key-value-must-not-leak", encoding="utf-8")
    monkeypatch.setattr(sr, "resolve_cli_binary", lambda _name: "/bin/codex")
    monkeypatch.setattr(sr, "codex_auth_effective_mode", lambda _path: "apikey", raising=False)
    monkeypatch.setattr(
        sr,
        "_run_cli_probe",
        lambda *args, **kwargs: pytest.fail("non-subscription Codex auth must not probe"),
    )

    result = sr.verify_subscription_provider("codex", marker_paths=(marker,))

    assert result.transport_ready is False
    assert "api-key-value-must-not-leak" not in repr(result)


def test_gemini_cli_is_distinct_and_policy_blocked_from_subscription_routing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sr,
        "gemini_readiness",
        lambda **kwargs: type(
            "Raw",
            (),
            {
                "installed": True,
                "auth_present": True,
                "auth_verified": False,
                "reason": None,
            },
        )(),
    )
    result = sr.subscription_readiness("gemini-cli", verify=True, auth_probe=pytest.fail)
    assert result.provider == "gemini-cli"
    assert result.auth_present and not result.auth_verified
    assert result.transport_ready is False
    assert "Antigravity terms" in (result.reason or "")
    assert "google-policy-gate" in result.provenance
    assert "cli-probe" not in result.provenance
    with pytest.raises(ValueError, match="separate"):
        sr.subscription_readiness("antigravity")


def test_gemini_live_verifier_never_invokes_a_subscription_prompt(monkeypatch) -> None:
    monkeypatch.setattr(
        sr,
        "subscription_readiness",
        lambda *args, **kwargs: sr.SubscriptionReadiness(
            "gemini-cli",
            sr.ReadinessState.NOT_READY,
            True,
            True,
            False,
            False,
            reason=sr.GOOGLE_SUBSCRIPTION_POLICY_REASON,
        ),
    )
    monkeypatch.setattr(
        sr,
        "_run_gemini_probe",
        lambda **kwargs: pytest.fail("Google OAuth must not be invoked"),
    )

    result = sr.verify_subscription_provider("gemini-cli")

    assert result.state is sr.ReadinessState.NOT_READY
    assert result.transport_ready is False
    assert "third-party routing" in (result.reason or "")


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_live_cli_probe_is_read_only_ephemeral_and_key_free(monkeypatch, provider: str) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(sr, "resolve_cli_binary", lambda _name: f"{provider}.exe")
    if provider == "codex":
        monkeypatch.setattr(sr, "codex_auth_effective_mode", lambda _path: "chatgpt")

    def run(argv: list[str], **kwargs: object):
        captured["argv"] = argv
        captured.update(kwargs)
        return type(
            "Result",
            (),
            {"returncode": 0, "stdout": sr.SAFE_PROBE_PROMPT.split("exactly ")[1].split()[0]},
        )()

    monkeypatch.setattr(sr.subprocess, "run", run)
    assert sr._run_cli_probe(
        provider,
        timeout_s=3,
        environ={"PATH": "safe", "OPENAI_API_KEY": "secret"},
    )
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert captured["shell"] is False
    assert captured["env"] == {"PATH": "safe"}
    assert isinstance(captured["cwd"], str)
    if provider == "codex":
        assert argv == [
            "codex.exe",
            "-c",
            'model_provider="openai"',
            "exec",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--color",
            "never",
            "--skip-git-repo-check",
            sr.SAFE_PROBE_PROMPT,
        ]
    else:
        assert "--tools" in argv and "--no-session-persistence" in argv


def test_codex_live_probe_keeps_selected_home_but_blocks_hostile_provider_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The probe cannot inherit a custom provider from its selected login home."""
    captured: dict[str, object] = {}
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        'model_provider = "hostile-provider"\n', encoding="utf-8"
    )
    monkeypatch.setattr(sr, "resolve_cli_binary", lambda _name: "codex.exe")
    monkeypatch.setattr(sr, "codex_auth_effective_mode", lambda _path: "chatgpt")

    def run(argv: list[str], **kwargs: object):
        captured["argv"] = argv
        captured.update(kwargs)
        return type("Result", (), {"returncode": 0, "stdout": "SUBSCRIPTION_MVP_PROBE_OK"})()

    monkeypatch.setattr(sr.subprocess, "run", run)

    assert sr._run_cli_probe(
        "codex",
        timeout_s=3,
        environ={
            "PATH": "safe",
            "CODEX_HOME": str(codex_home),
            "OPENAI_API_KEY": "do-not-forward",
            "OPENAI_BASE_URL": "https://hostile.invalid/v1",
            "DATABRICKS_TOKEN": "do-not-forward",
            "HARNESS_CODEX_MODEL_PROVIDER": "hostile-provider",
        },
    )
    assert captured["argv"] == [
        "codex.exe",
        "-c",
        'model_provider="openai"',
        "exec",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--color",
        "never",
        "--skip-git-repo-check",
        sr.SAFE_PROBE_PROMPT,
    ]
    assert captured["env"] == {"PATH": "safe", "CODEX_HOME": str(codex_home)}


def test_cursor_requires_explicit_distro_and_reports_actionable_wsl_gate() -> None:
    result = sr.subscription_readiness("cursor-wsl", environ={}, cursor_config={})
    assert result.state is sr.ReadinessState.NOT_READY
    assert result.transport_ready is False
    assert "distro" in (result.reason or "")


def test_cursor_conflicting_config_is_not_silently_resolved() -> None:
    result = sr.subscription_readiness(
        "cursor-wsl",
        environ={},
        cursor_config={
            "cursor_wsl": {"distro": "Ubuntu"},
            "cursor": {"wsl_distro": "Debian"},
        },
    )
    assert result.transport_ready is False
    assert result.reason == (
        "Cursor WSL distro settings conflict; keep one explicit distro setting."
    )


def test_cursor_requires_explicit_non_root_user() -> None:
    result = sr.subscription_readiness(
        "cursor-wsl",
        environ={},
        cursor_config={"cursor_wsl": {"distro": "Ubuntu-24.04"}},
    )
    assert result.state is sr.ReadinessState.NOT_READY
    assert result.transport_ready is False
    assert "user" in (result.reason or "")


@pytest.mark.parametrize(
    ("user", "reason"),
    [("root", "non-root"), ("not safe", "lowercase POSIX")],
)
def test_cursor_rejects_invalid_or_root_configured_user(user: str, reason: str) -> None:
    result = sr.subscription_readiness(
        "cursor-wsl",
        environ={},
        cursor_config={"cursor_wsl": {"distro": "Ubuntu-24.04", "user": user}},
    )
    assert result.transport_ready is False
    assert reason in (result.reason or "")


def test_cursor_uses_explicit_config_identity_without_guessing(monkeypatch) -> None:
    class Raw:
        cursor_agent_installed = True
        auth_present = False
        transport_ready = False
        last_verified = "2026-09-04T00:00:00+00:00"
        error = "wsl-missing"

    class FakeTransport:
        def __init__(self, distro, user, workdir, **kwargs):
            assert distro == "Ubuntu-24.04"
            assert user == "ricar"
            assert workdir == r"C:\repo"

        def readiness(self):
            return Raw()

    monkeypatch.setattr(sr, "CursorWslTransport", FakeTransport)
    result = sr.subscription_readiness(
        "cursor-wsl",
        cursor_config={"cursor_wsl": {"distro": "Ubuntu-24.04", "user": "ricar"}},
        cursor_working_directory=r"C:\repo",
    )
    assert result.reason == "WSL is unavailable; enable WSL and restart Windows, then retry."
    assert "explicit-config" in result.provenance


def test_default_verifier_does_not_probe(monkeypatch, capsys) -> None:
    import dev.verify_subscription_mvp as verify

    monkeypatch.setattr(
        verify,
        "subscription_readiness_map",
        lambda **kwargs: {
            name: sr.SubscriptionReadiness(
                name, sr.ReadinessState.INSTALLED, True, False, False, False
            )
            for name in sr.SUPPORTED_PROVIDERS
        },
    )
    monkeypatch.setattr(
        verify,
        "verify_subscription_provider",
        lambda *args, **kwargs: pytest.fail("live verification was not requested"),
    )
    assert verify.main([]) == 0
    assert "SUBSCRIPTION_MVP_LIVE_OK" not in capsys.readouterr().out


def test_explicit_verifier_prints_success_token_only_when_all_providers_pass(
    monkeypatch, capsys
) -> None:
    import dev.verify_subscription_mvp as verify

    def result(provider: str, *, ok: bool) -> sr.SubscriptionReadiness:
        return sr.SubscriptionReadiness(
            provider,
            sr.ReadinessState.AUTH_VERIFIED if ok else sr.ReadinessState.NOT_READY,
            True,
            ok,
            ok,
            ok,
        )

    monkeypatch.setattr(
        verify,
        "verify_subscription_provider",
        lambda provider, **kwargs: result(provider, ok=provider == "claude"),
    )
    assert verify.main(["--providers", "claude", "codex", "--no-api-keys"]) == 1
    assert "SUBSCRIPTION_MVP_LIVE_OK" not in capsys.readouterr().out

    monkeypatch.setattr(
        verify,
        "verify_subscription_provider",
        lambda provider, **kwargs: result(provider, ok=True),
    )
    assert verify.main(["--providers", "claude", "codex", "--no-api-keys"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("SUBSCRIPTION_MVP_LIVE_OK")
