"""Mock-LLM e2e regression: a signed-in antigravity-native worker must not list as dead.

Bug: ``sys_list_models`` reports a fully dispatchable ``antigravity-native``
(``agy``) sub-agent worker as ``source: "none"`` — the dead-worker shape whose
note tells the driving agent the worker "cannot run here" — so orchestrators
skip a worker that would run fine. No configuration fixes it: the readout is
unchanged whether the install is OAuth-only (the default agy onboarding path)
or carries a ``GEMINI_API_KEY`` that the provider layer resolves correctly
under the ``antigravity-native`` harness name.

The user journey (from the report):

1. Install ``agy`` and sign in via its Google OAuth flow (or export a
   ``GEMINI_API_KEY``); ``omnigent setup`` reports "Antigravity sign-in ready".
2. Configure an orchestrator with an ``antigravity-native`` sub-agent (the
   shipped ``examples/polly`` bundle's ``agy`` worker).
3. Ask the orchestrator to call ``sys_list_models``.
4. The catalog row for the agy worker comes back ``source: "none"`` with an
   empty model list and the "no usable model provider (…) — dispatches to this
   worker cannot run here" note, even though a forced dispatch boots and
   completes normally.

This test drives the REAL chain — mock brain tool-call -> runner dispatch ->
``catalog_for_spec`` -> persisted transcript row — by booting a throwaway
local server from this working tree and running the polly orchestrator
headless with a mock LLM brain, exactly like its siblings in
``test_cursor_native_list_models_e2e.py``. A launchable ``agy`` stub is
prepended to the runner's ``PATH`` and the signed-in state is pinned per
case: an isolated ``HOME`` carrying the Linux OAuth token file (OAuth-only
case), or an ambient ``GEMINI_API_KEY`` with an empty ``HOME`` (API-key
case). Both env vars are forwarded CLI->runner by the runner env allowlist.

Both cases FAIL on un-fixed code (the agy row is ``source: "none"``) and must
PASS once the antigravity spellings degrade to a subscription-style readout
the way the sibling subscription CLIs (and the fixed cursor-native) do.

Run::

    pytest tests/e2e/test_antigravity_native_list_models_e2e.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.test_polly_e2e import (
    _MOCK_BRAIN_MODEL,
    _REPO,
    _SERVER_BOOT_TIMEOUT_SEC,
    _free_port,
    _mock_env,
    _mock_polly_spec_dir,
    _wait_for_health,
)

# Mock runs are fast (no real model inference) so a short timeout is enough.
_RUN_TIMEOUT_SEC = 300

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


def _api(base_url: str, path: str) -> dict[str, Any]:
    """GET a local-server API path and decode the JSON body.

    :param base_url: Server base URL, e.g. ``"http://127.0.0.1:8811"``.
    :param path: API path starting with ``/``, e.g. ``"/v1/sessions"``.
    :returns: Decoded JSON object.
    """
    with urllib.request.urlopen(f"{base_url}{path}", timeout=15) as resp:
        return json.load(resp)


def _write_agy_stub(tmp_path: Path) -> Path:
    """Write a launchable ``agy`` stub for the runner's ``PATH``.

    Mirrors the reported state: the worker is dispatchable (the binary
    resolves, reports a modern version, and launches). ``models`` lists one
    id so any post-fix listing probe finds something sane.

    :param tmp_path: Per-test temp dir to write the stub into.
    :returns: Absolute path to the executable stub.
    """
    stub = tmp_path / "agy"
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
    return stub


def _write_oauth_token(home: Path) -> None:
    """Write the Linux agy OAuth token file under an isolated ``HOME``.

    :param home: The directory the spawned processes will see as ``HOME``.
    """
    token_path = home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(json.dumps(_FAKE_OAUTH_TOKEN), encoding="utf-8")


@pytest.fixture
def local_polly_server(tmp_path: Path) -> Iterator[str]:
    """Start a throwaway local ``omnigent server`` from this working tree.

    Mirrors ``test_cursor_native_list_models_e2e.local_polly_server`` (own
    sqlite DB + artifact dir under ``tmp_path``); duplicated as a fixture
    because pytest fixtures don't import across modules without a conftest,
    and this file must stay droppable next to its siblings.

    :param tmp_path: pytest-provided per-test temp dir for the DB + artifacts.
    :yields: The base URL of the running server.
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    import os

    env = {
        **os.environ,
        "OMNIGENT_SKIP_ONBOARD": "1",
        "OMNIGENT_NO_UPDATE_CHECK": "1",
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "omnigent",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--database-uri",
            f"sqlite:///{tmp_path / 'agy_list_models_e2e.db'}",
            "--artifact-location",
            str(tmp_path / "artifacts"),
        ],
        cwd=str(_REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(base_url, time.monotonic() + _SERVER_BOOT_TIMEOUT_SEC)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _polly_parent_id(base_url: str) -> str:
    """Find the polly parent session on the throwaway server.

    The server DB is per-test, so the only polly session is ours.

    :param base_url: Local server base URL.
    :returns: The parent conversation id.
    """
    sessions = _api(base_url, "/v1/sessions").get("data", [])
    parents = [s["id"] for s in sessions if s.get("agent_name") == "polly"]
    assert parents, f"no polly session found among {len(sessions)} sessions"
    return parents[0]


def _agy_row_from_polly_run(
    local_polly_server: str,
    mock_llm_server_url: str,
    tmp_path: Path,
    extra_env: dict[str, str],
    drop_env: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run polly with a mock brain that calls ``sys_list_models``; return the agy row.

    Drives the full chain: ``omnigent run`` against the throwaway server,
    the scripted brain issues one ``sys_list_models`` call, the runner
    dispatches it, and the persisted transcript row is read back over the
    server API.

    :param local_polly_server: Base URL of the in-tree local server fixture.
    :param mock_llm_server_url: Mock LLM server base URL.
    :param tmp_path: Per-test temp dir for the spec copy and the CLI stub.
    :param extra_env: Env overrides for the ``omnigent run`` subprocess
        (e.g. an isolated ``HOME`` or a ``GEMINI_API_KEY``).
    :param drop_env: Env var names removed from the subprocess env.
    :returns: The agy worker's catalog row dict.
    """
    import os
    import uuid

    from tests.e2e.conftest import configure_mock_llm, reset_mock_llm

    reset_mock_llm(mock_llm_server_url)
    polly_dir = _mock_polly_spec_dir(tmp_path, mock_llm_server_url)
    stub = _write_agy_stub(tmp_path)
    tag = uuid.uuid4().hex[:8]

    configure_mock_llm(
        mock_llm_server_url,
        [
            # Step 1: the brain asks for the per-worker model catalog.
            {
                "tool_calls": [
                    {
                        "call_id": f"call-lm-{tag}",
                        "name": "sys_list_models",
                        "arguments": "{}",
                    }
                ]
            },
            # Step 2: end the turn after receiving the catalog.
            {"text": "Catalog received."},
        ],
        key=_MOCK_BRAIN_MODEL,
    )

    env = _mock_env(mock_llm_server_url)
    # The runner resolves agy from its own environment, and the CLI->runner
    # env strip only forwards allowlisted vars (PATH is one). Prepend the
    # stub's dir to PATH so the runner sees an "installed" agy regardless of
    # whether the host has a real one.
    env["PATH"] = f"{stub.parent}{os.pathsep}{env.get('PATH', '')}"
    # The spawned runner subprocess doesn't inherit sys.path[0] from the CLI's
    # cwd; make this working tree importable there (mirrors the e2e_ui
    # conftest's runner env) so worktree runs work without an editable install.
    env["PYTHONPATH"] = f"{_REPO}{os.pathsep}{env.get('PYTHONPATH', '')}"
    for stale in drop_env:
        env.pop(stale, None)
    env.update(extra_env)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnigent",
            "run",
            str(polly_dir),
            "--server",
            local_polly_server,
            "-p",
            "Call sys_list_models and report the catalog.",
        ],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SEC,
    )
    assert result.returncode == 0, (
        f"polly run exited {result.returncode}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    parent = _polly_parent_id(local_polly_server)
    items = _api(local_polly_server, f"/v1/sessions/{parent}/items").get("data", [])

    # The sys_list_models call ran and its result persisted in the transcript.
    call_ids = {
        item.get("call_id")
        for item in items
        if item.get("type") == "function_call" and item.get("name") == "sys_list_models"
    }
    assert call_ids, "no sys_list_models function_call found in the parent transcript"
    catalogs = [
        json.loads(item.get("output") or "{}")
        for item in items
        if item.get("type") == "function_call_output" and item.get("call_id") in call_ids
    ]
    assert catalogs, "no sys_list_models tool result found in the parent transcript"
    catalog = catalogs[-1]

    agy_row = catalog.get("agy")
    assert isinstance(agy_row, dict), f"sys_list_models result has no 'agy' row: {sorted(catalog)}"
    return agy_row


@pytest.mark.timeout(600)
def test_antigravity_native_worker_not_source_none_oauth_login(
    local_polly_server: str,
    mock_llm_server_url: str,
    tmp_path: Path,
) -> None:
    """OAuth-only signed-in agy must not list as ``source: "none"``.

    Pins the default agy onboarding state from the report: the Linux OAuth
    token file present under ``HOME``, no ``GEMINI_API_KEY`` anywhere. On
    un-fixed code the agy row is the dead-worker shape (``source: "none"``,
    empty models, "cannot run here" note) even though the CLI carries its
    own Google login; any usable provenance ("static" like the sibling
    subscription CLIs, or "cli" from a live probe) passes.

    :param local_polly_server: Base URL of the in-tree local server fixture.
    :param mock_llm_server_url: Mock LLM server base URL.
    :param tmp_path: Per-test temp dir for the spec copy, stub, and HOME.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    _write_oauth_token(fake_home)

    agy_row = _agy_row_from_polly_run(
        local_polly_server,
        mock_llm_server_url,
        tmp_path,
        extra_env={"HOME": str(fake_home)},
        drop_env=("GEMINI_API_KEY",),
    )

    # THE BUG: a signed-in, dispatchable antigravity-native worker
    # is reported with the dead-worker source "none".
    assert agy_row.get("source") != "none", (
        "Bug reproduced: sys_list_models reports the OAuth-signed-in "
        f"antigravity-native worker as source='none' — full row: {agy_row}"
    )


@pytest.mark.timeout(600)
def test_antigravity_native_worker_not_source_none_gemini_api_key(
    local_polly_server: str,
    mock_llm_server_url: str,
    tmp_path: Path,
) -> None:
    """A configured ``GEMINI_API_KEY`` must not leave agy listed as dead.

    The report's decisive evidence: the ambient key registers a
    gemini-family provider that resolves under the ``antigravity-native``
    harness name, yet the readout still reports ``source: "none"`` because
    it queries the collapsed ``"antigravity"`` spelling (mapped to the
    openai family). An isolated empty ``HOME`` pins the no-OAuth-token
    state so the key is the only credential.

    :param local_polly_server: Base URL of the in-tree local server fixture.
    :param mock_llm_server_url: Mock LLM server base URL.
    :param tmp_path: Per-test temp dir for the spec copy, stub, and HOME.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    agy_row = _agy_row_from_polly_run(
        local_polly_server,
        mock_llm_server_url,
        tmp_path,
        extra_env={
            "HOME": str(fake_home),
            "GEMINI_API_KEY": "AIza-e2e-not-a-real-key",
        },
    )

    # THE BUG: even with a resolvable gemini-family credential
    # configured, the readout still reports the dead-worker source "none".
    assert agy_row.get("source") != "none", (
        "Bug reproduced: sys_list_models reports the antigravity-native worker "
        f"as source='none' despite a configured GEMINI_API_KEY — full row: {agy_row}"
    )
