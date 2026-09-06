"""UI regression: antigravity-native worker reported as dead in ``sys_list_models``.

Journey (what the user does and sees in the SPA):

1. Install ``agy`` and sign in via its Google OAuth flow (token file on
   disk, no ``GEMINI_API_KEY``) — the state ``omnigent setup`` reports as
   "Antigravity sign-in ready".
2. Configure an orchestrator agent with an ``agy`` sub-agent on the
   ``antigravity-native`` harness.
3. Ask it for its workers' model availability — the brain calls
   ``sys_list_models`` and the tool call renders in the transcript.
4. Expand the tool call: the ``agy`` worker's row reads
   ``"source": "none"`` with an empty model list and the dead-worker note
   ("no usable model provider (…) — dispatches to this worker cannot run
   here"), even though agy brings its own Google OAuth login and a forced
   dispatch boots and completes normally.

The signed-in state is pinned before the suite's shared runner spawns: a
launchable ``agy`` stub goes on ``PATH`` and a Linux-shape OAuth token file
is written under ``~/.gemini/antigravity-cli/`` (only when absent; removed
on teardown), with any ambient ``GEMINI_API_KEY`` popped so the state is
OAuth-only — the default agy onboarding path from the report.

Sibling subscription-CLI workers (claude-native / codex-native / the fixed
cursor-native) report a usable ``source: "static"`` / ``"cli"`` row in the
same signed-in state; the agy row collapsing to ``"none"`` is the bug — an
orchestrator preflighting with ``sys_list_models`` drops a worker that runs
fine. The test FAILS on un-fixed code and must PASS once the antigravity
spellings degrade to a subscription-style readout the way cursor-native
does.

Run (spawns its own local server + runner; build the SPA first)::

    pytest tests/e2e_ui/chat/test_antigravity_native_list_models_row.py -v
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import tarfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Page, expect

from tests.e2e_ui.conftest import (
    _ensure_runner_online,
    _server_state,
    configure_mock_llm,
    reset_mock_llm,
)

# Model key routing the parent brain to its scripted mock queue.
_BRAIN_MODEL = "mock-agy-catalog-brain"
_PARENT_NAME = "agy_catalog_orch"

_COMPOSER = "Send a message…"
_ASSISTANT = '[data-testid="message-bubble"][data-role="assistant"]'

# One scripted tool turn + one text turn; catalog enumeration is local and
# fast, but CI boxes are slow, so waits get a generous budget.
_TURN_TIMEOUT_MS = 180_000

# Linux agy token shape: the OAuth object nested under ``token`` (see
# omnigent.onboarding.gemini_auth — a non-empty access/refresh token string
# counts as a completed login).
_FAKE_OAUTH_TOKEN: dict[str, object] = {
    "auth_method": "oauth",
    "token": {
        "access_token": "e2e-fake-access-token",
        "refresh_token": "e2e-fake-refresh-token",
        "token_type": "Bearer",
        "expiry": "2099-01-01T00:00:00Z",
    },
}


@pytest.fixture(scope="session", autouse=True)
def _agy_signed_in_state(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Pin the "agy installed and signed in via OAuth" state from the report.

    Prepends a launchable ``agy`` stub to ``PATH``, writes the Linux OAuth
    token file under the real home when absent (removed on teardown so no
    state leaks past the run), and pops any ambient ``GEMINI_API_KEY`` so
    the credential is OAuth-only. Runs before ``live_server`` first spawns
    so the suite's shared runner inherits the stubbed ``PATH``; the token
    file is read per-call by ``gemini_auth``, so it is visible even to a
    runner that spawned earlier (full-suite runs).

    :param tmp_path_factory: Pytest temp path factory for the stub dir.
    """
    if os.name == "nt":
        # antigravity-native is gated off Windows; the bash stub can't run there.
        yield
        return
    stub_dir = tmp_path_factory.mktemp("agy_signed_in_stub")
    stub = stub_dir / "agy"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "# Stub agy: launchable and signed in; models lists one id.\n"
        'if [ "$1" = "--version" ]; then\n'
        '  echo "1.1.26"\n'
        "  exit 0\n"
        "fi\n"
        'if [ "$1" = "models" ]; then\n'
        '  echo "gemini-3-pro"\n'
        "  exit 0\n"
        "fi\n"
        'echo "stub agy: $*"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{stub_dir}{os.pathsep}{old_path}"
    old_key = os.environ.pop("GEMINI_API_KEY", None)

    token_path = Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    created: list[Path] = []
    if not token_path.exists():
        for parent in (token_path.parent.parent, token_path.parent):
            if not parent.exists():
                created.append(parent)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps(_FAKE_OAUTH_TOKEN), encoding="utf-8")
        created.append(token_path)
    try:
        yield
    finally:
        os.environ["PATH"] = old_path
        if old_key is not None:
            os.environ["GEMINI_API_KEY"] = old_key
        for path in reversed(created):
            with contextlib.suppress(OSError):
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()


@dataclass(frozen=True)
class _CatalogSession:
    """Handle for the orchestrator-with-agy-worker session.

    :param base_url: Spawned server base URL.
    :param session_id: The runner-bound parent session id.
    :param routing_token: Per-run token that selects the brain's mock queue.
    """

    base_url: str
    session_id: str
    routing_token: str


def _orchestrator_yaml(mock_llm_server_url: str) -> str:
    """Build the orchestrator spec: an openai-agents brain + antigravity-native worker.

    Omnigent-flavored single-file YAML with an inline ``type: agent`` tool
    (the compat-adapter shape, same as the cursor catalog-row test), so the
    ``agy`` sub-agent registers ``sys_session_send`` / ``sys_list_models``
    on the brain. An explicit ``auth`` block pins the brain to the mock LLM
    server so an ambient provider config (e.g. a CI gateway in
    ``OMNIGENT_CONFIG_HOME``) can't shadow the mock routing.

    :param mock_llm_server_url: Mock LLM server base URL.
    :returns: YAML text ready for bundle upload.
    """
    return f"""\
name: {_PARENT_NAME}
prompt: |
  You are a coding orchestrator with one `agy` sub-agent. When the
  user asks which models your workers can run, call `sys_list_models`
  and then summarize the result.

executor:
  model: {_BRAIN_MODEL}
  harness: openai-agents
  auth:
    type: api_key
    api_key: mock-key
    base_url: {mock_llm_server_url}/v1

tools:
  agy:
    type: agent
    description: Google Antigravity coding sub-agent (native agy TUI).
    executor:
      model: gemini-3-pro
      harness: antigravity-native
    prompt: |
      You are the Antigravity coding sub-agent.
"""


@pytest.fixture
def catalog_session(
    live_server: str,
    mock_llm_server_url: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[_CatalogSession]:
    """Create a runner-bound session for the catalog journey.

    Same runner-respawn and bind contract as the suite's other agent
    fixtures. The brain's mock queue scripts one ``sys_list_models`` call
    followed by a closing text turn.

    :param live_server: Spawned server fixture.
    :param mock_llm_server_url: Mock LLM server used by credential-free runs.
    :param tmp_path_factory: Pytest temp path factory (for a respawn log).
    :yields: A :class:`_CatalogSession` handle.
    """
    routing_token = f"agy-catalog-{uuid.uuid4().hex[:10]}"
    configure_mock_llm(
        mock_llm_server_url,
        [
            {
                "tool_calls": [
                    {
                        "call_id": f"call-lm-{routing_token}",
                        "name": "sys_list_models",
                        "arguments": "{}",
                    }
                ]
            },
            {"text": f"Catalog reported. Marker: {routing_token}"},
        ],
        key=_BRAIN_MODEL,
        match=routing_token,
    )
    respawned_runner = _ensure_runner_online(live_server, tmp_path_factory)
    runner_id = str(_server_state["runner_id"])

    yaml_bytes = _orchestrator_yaml(mock_llm_server_url).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Non-config.yaml arcname routes the bundle through the omnigent
        # compat adapter, whose loader parses the inline `type: agent` tool.
        info = tarfile.TarInfo(name=f"{_PARENT_NAME}.yaml")
        info.size = len(yaml_bytes)
        tar.addfile(info, io.BytesIO(yaml_bytes))
    create_resp = httpx.post(
        f"{live_server}/v1/sessions",
        data={"metadata": json.dumps({})},
        files={"bundle": ("agent.tar.gz", buf.getvalue(), "application/gzip")},
        timeout=30.0,
    )
    create_resp.raise_for_status()
    session_id = create_resp.json()["session_id"]

    patch_resp = httpx.patch(
        f"{live_server}/v1/sessions/{session_id}",
        json={"runner_id": runner_id},
        timeout=10.0,
    )
    patch_resp.raise_for_status()

    try:
        yield _CatalogSession(
            base_url=live_server,
            session_id=session_id,
            routing_token=routing_token,
        )
    finally:
        try:
            httpx.delete(f"{live_server}/v1/sessions/{session_id}", timeout=10.0)
        finally:
            try:
                reset_mock_llm(mock_llm_server_url)
            finally:
                if respawned_runner is not None:
                    respawned_runner.terminate()
                    try:
                        respawned_runner.wait(timeout=5)
                    except Exception:
                        respawned_runner.kill()


def _expand_list_models_tool_call(page: Page) -> None:
    """Expand the completed turn, tool group, and model-listing output.

    :param page: The Playwright page, on the parent session.
    """
    worked = page.get_by_test_id("turn-worked-fold")
    expect(worked).to_be_visible(timeout=30_000)
    worked_trigger = worked.get_by_role("button", name=re.compile(r"^Worked"))
    # The fold mounts open to animate shut after idle; wait before opening it.
    expect(worked_trigger).to_have_attribute("aria-expanded", "false", timeout=30_000)
    worked_trigger.click()
    expect(worked_trigger).to_have_attribute("aria-expanded", "true")

    group = worked.get_by_role("button", name="Called 1 tool", exact=True)
    expect(group).to_be_visible(timeout=30_000)
    expect(group).to_have_attribute("aria-expanded", "false")
    group.click()
    expect(group).to_have_attribute("aria-expanded", "true")

    direct = worked.get_by_role("button", name=re.compile(r"^sys_list_models"))
    expect(direct).to_be_visible(timeout=30_000)
    direct.click()
    expect(direct).to_have_attribute("aria-expanded", "true")


def _agy_catalog_row(base_url: str, session_id: str) -> dict[str, object]:
    """Fetch the persisted ``sys_list_models`` result's ``agy`` row.

    :param base_url: Spawned server base URL.
    :param session_id: The parent session id.
    :returns: The agy worker's catalog row dict.
    """
    items_resp = httpx.get(f"{base_url}/v1/sessions/{session_id}/items", timeout=15.0)
    items_resp.raise_for_status()
    items = items_resp.json().get("data", [])
    call_ids = {
        item.get("call_id")
        for item in items
        if item.get("type") == "function_call" and item.get("name") == "sys_list_models"
    }
    assert call_ids, "no sys_list_models function_call found in the transcript"
    catalogs = [
        json.loads(item.get("output") or "{}")
        for item in items
        if item.get("type") == "function_call_output" and item.get("call_id") in call_ids
    ]
    assert catalogs, "no sys_list_models tool result found in the transcript"
    row = catalogs[-1].get("agy")
    assert isinstance(row, dict), f"catalog has no 'agy' row: {sorted(catalogs[-1])}"
    return row


@pytest.mark.timeout(600)
def test_antigravity_native_worker_row_not_source_none(
    page: Page,
    catalog_session: _CatalogSession,
) -> None:
    """The antigravity-native worker's catalog row must not be ``source: "none"``.

    Drives the reported journey in the SPA: ask the orchestrator for its
    workers' models, watch the ``sys_list_models`` tool call land in the
    transcript, expand it, and check the agy worker's row. On un-fixed code
    the row is the dead-worker shape (``source: "none"``, no models, the
    "no usable model provider … cannot run here" note) even though agy is
    signed in and dispatchable via its own Google OAuth login — this test
    fails there and passes once the row degrades to a usable source
    (``"static"`` / ``"cli"``) like the sibling subscription-CLI workers.

    :param page: pytest-playwright page fixture.
    :param catalog_session: The orchestrator session handle.
    """
    chat = catalog_session
    page.goto(f"{chat.base_url}/c/{chat.session_id}")

    composer = page.get_by_placeholder(_COMPOSER)
    expect(composer).to_be_visible(timeout=30_000)
    composer.fill(
        "Which models can each of your workers run? Call sys_list_models "
        f"and summarize. Routing marker: {chat.routing_token}"
    )
    page.get_by_role("button", name="Send", exact=True).click()

    # The closing text turn proves the tool call completed and persisted.
    expect(
        page.locator(_ASSISTANT, has_text=f"Catalog reported. Marker: {chat.routing_token}").first
    ).to_be_visible(timeout=_TURN_TIMEOUT_MS)
    # The completed-turn fold appears after the working indicator clears.
    expect(page.get_by_test_id("working-indicator")).to_be_hidden(timeout=30_000)

    # Put the catalog on screen the way a user reads it (and the video shows it).
    _expand_list_models_tool_call(page)
    agy_key = page.get_by_text(re.compile(r'"agy"')).first
    expect(agy_key).to_be_visible(timeout=30_000)
    with contextlib.suppress(Exception):
        agy_key.scroll_into_view_if_needed()
    # Let the expanded row linger on screen so the recording shows it.
    page.wait_for_timeout(1500)

    row = _agy_catalog_row(chat.base_url, chat.session_id)

    # THE BUG: a signed-in, dispatchable antigravity-native worker is
    # reported with the dead-worker source "none" (its note tells the
    # orchestrator dispatches "cannot run here"). Any usable provenance
    # passes.
    assert row.get("source") != "none", (
        "Bug reproduced: sys_list_models reports the signed-in "
        f"antigravity-native worker as source='none' — full row: {row}"
    )
