"""E2E regression: a failed native-terminal start must name a runner log the
reader can actually find.

Reported journey (Codex-native harness): a user starts a codex-native session
on a host whose runner cannot reach the ``codex`` CLI, opens the native Codex
terminal, and it fails to start with::

    Native Codex terminal failed to start; see the runner log for details:
    ~/.omnigent/logs/runner/runner-<ts>.log

They then go to that path on the host and the referenced runner log file is not
there. The ``~`` in the message is home-relative to the *runner* process
(``omnigent/process_logging.py`` collapses the path against the runner's own
``Path.home()``). For a host-launched / managed-sandbox runner — the normal
codex-native topology — the runner's home tree is not the interactive user's,
so expanding ``~`` from the reader's home lands on a file that does not exist.

This drives the real stack: a real ``omnigent`` server subprocess + a real
``omnigent.runner._entry`` runner with ``codex`` stripped from its ``PATH`` and
its own distinct ``HOME`` (standing in for the host-launched runner whose
filesystem is not the reader's). Both children run with ambient ``OMNIGENT*``
variables scrubbed: the repo test harness pins ``OMNIGENT_DATA_DIR`` to a
shared temp tree, which would move the runner's log *outside* its home and
mask the home-collapse under test — production runners default their data dir
to their own ``HOME``.

Two things are asserted:

* **Precondition (stable across a fix):** the terminal fails to start with the
  ``native_terminal_start_failed`` message naming a ``.log`` file — the
  reported symptom-1 failure.
* **The bug (fail while live, pass once fixed):** the runner log the message
  points at must be *findable on the host* — i.e. resolving the named path the
  way a user on the host would (the reader's ``HOME``) yields an existing file.
  While the bug is live the ``~/…`` path resolves under the reader's home and is
  absent (the log lives under the runner's home), so this assertion fails. A fix
  that surfaces a host-resolvable path makes it pass::

      .venv/bin/python -m pytest \
        tests/e2e/test_native_terminal_failure_names_findable_runner_log.py -v
"""

from __future__ import annotations

import io
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# A codex-native agent, reduced to the fields that pick the native-terminal
# launch path under test. spec_version 1 + executor.config.harness routes
# through the strict parser; the ``config.yaml`` arcname keeps it on that path.
_CODEX_NATIVE_YAML = """\
spec_version: 1
name: codex_native_repro
description: Codex-native agent for the failed-terminal log-reference regression.

executor:
  type: omnigent
  config:
    harness: codex-native
    yolo: true

prompt: |
  You are Codex.

os_env:
  type: caller_process
  cwd: .
  sandbox:
    type: none
"""

_HEALTH_TIMEOUT_S = 90.0
_POLL_INTERVAL_S = 0.5

# Proxy-blind client: CI can force an egress proxy that must not intercept
# loopback requests to the spawned server.
_client = httpx.Client(trust_env=False)
for _var in ("NO_PROXY", "no_proxy"):
    os.environ[_var] = ",".join(filter(None, [os.environ.get(_var, ""), "127.0.0.1,localhost"]))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _scrubbed_env() -> dict[str, str]:
    """Return the ambient env without ``OMNIGENT*`` keys, loopback proxy-free.

    The test harness (and a session-spawned shell) exports ``OMNIGENT_*``
    state — most importantly ``OMNIGENT_DATA_DIR`` / ``OMNIGENT_PROCESS_LOG_FILE``,
    which would relocate the children's logs and mask the home-anchored log
    layout production runners get by default. Each child re-adds exactly what
    it needs.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNIGENT")}
    for var in ("NO_PROXY", "no_proxy"):
        existing = env.get(var, "")
        env[var] = ",".join(filter(None, [existing, "127.0.0.1,localhost"]))
    return env


def _path_without_codex(env: dict[str, str]) -> str:
    """Return ``PATH`` with any ``codex`` bin dir removed.

    Drops the entry that resolves ``codex`` (CI ships it under
    ``.harness-clis/node_modules/.bin``), so the runner's native-Codex launch
    hits the ``ImportError: Native Codex requires the 'codex' CLI on PATH``
    path — the reported failure's root cause.
    """
    kept: list[str] = []
    for part in env.get("PATH", "").split(os.pathsep):
        if not part:
            continue
        if shutil.which("codex", path=part) is not None:
            continue
        kept.append(part)
    return os.pathsep.join(kept)


def _spec_bundle() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = _CODEX_NATIVE_YAML.encode()
        info = tarfile.TarInfo("config.yaml")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _resolve_as_host_reader(named: str, reader_home: Path) -> Path:
    """Resolve *named* the way a user on the host would read it.

    The message renders the path with a leading ``~`` (home-relative); a human
    on the host expands that against *their* home. An absolute path is taken
    as-is. This is exactly the resolution the reporter performed when they went
    looking for the log and found nothing.
    """
    if named.startswith("~"):
        return Path(str(reader_home) + named[1:])
    return Path(named)


@pytest.fixture
def codex_native_rig(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[str, str, Path, Path]]:
    """Server + runner whose runner cannot reach codex and lives under its own HOME.

    The runner's ``HOME`` is a distinct temp tree (models a host-launched /
    managed-sandbox runner whose filesystem is not the interactive user's), and
    ``codex`` is stripped from its ``PATH`` so the native-Codex terminal start
    fails at launch. The runner gets no ``OMNIGENT_DATA_DIR``: like a production
    runner, its logs default to ``$HOME/.omnigent/logs`` — the layout whose
    surfaced reference is under test. The server keeps its state in the test
    tree via an explicit data dir.

    :returns: ``(base_url, runner_id, runner_home, reader_home)``.
    """
    work = tmp_path_factory.mktemp("codex_native_log_ref")
    runner_home = work / "runner-home"
    artifacts = work / "artifacts"
    server_data = work / "server-data"
    for path in (runner_home, artifacts, server_data):
        path.mkdir(parents=True, exist_ok=True)

    # The reader is a user on the host, expanding '~' against their own home —
    # NOT the runner's. Use the ambient HOME, guaranteed distinct from
    # runner_home.
    reader_home = Path(os.path.expanduser("~"))
    assert reader_home != runner_home, "reader and runner must not share HOME"

    from omnigent.runner.identity import token_bound_runner_id

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    binding_token = secrets.token_urlsafe(32)
    runner_id = token_bound_runner_id(binding_token)

    pythonpath = os.pathsep.join(
        [
            str(_REPO_ROOT),
            str(_REPO_ROOT / "sdks" / "python-client"),
            str(_REPO_ROOT / "sdks" / "ui"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    shared_env = {**_scrubbed_env(), "PYTHONPATH": pythonpath}
    server_env = {
        **shared_env,
        "OMNIGENT_RUNNER_TUNNEL_TOKEN": binding_token,
        "OMNIGENT_DATA_DIR": str(server_data),
    }
    runner_env = {
        **shared_env,
        "HOME": str(runner_home),
        # Strip codex from PATH and clear any explicit override so the runner's
        # native-Codex launch cannot resolve the CLI.
        "PATH": _path_without_codex(shared_env),
        "OMNIGENT_CODEX_PATH": "",
        "OMNIGENT_RUNNER_ID": runner_id,
        "OMNIGENT_RUNNER_TUNNEL_BINDING_TOKEN": binding_token,
        "OMNIGENT_RUNNER_PARENT_PID": str(os.getpid()),
        "RUNNER_SERVER_URL": base_url,
    }

    server_log = work / "server.log"
    runner_log = work / "runner.log"
    server_handle = server_log.open("w")
    runner_handle = runner_log.open("w")
    server_proc: subprocess.Popen[bytes] | None = None
    runner_proc: subprocess.Popen[bytes] | None = None
    try:
        server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "omnigent.cli",
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--database-uri",
                f"sqlite:///{work}/test.db",
                "--artifact-location",
                str(artifacts),
            ],
            env=server_env,
            stdout=server_handle,
            stderr=subprocess.STDOUT,
            cwd=str(_REPO_ROOT),
        )
        runner_proc = subprocess.Popen(
            [sys.executable, "-m", "omnigent.runner._entry"],
            env=runner_env,
            stdout=runner_handle,
            stderr=subprocess.STDOUT,
            cwd=str(work),
        )

        deadline = time.monotonic() + _HEALTH_TIMEOUT_S
        online = False
        while time.monotonic() < deadline:
            if server_proc.poll() is not None or runner_proc.poll() is not None:
                break
            try:
                if _client.get(f"{base_url}/health", timeout=2).status_code == 200:
                    status = _client.get(f"{base_url}/v1/runners/{runner_id}/status", timeout=2)
                    if status.status_code == 200 and status.json().get("online"):
                        online = True
                        break
            except httpx.HTTPError:
                pass
            time.sleep(_POLL_INTERVAL_S)
        if not online:
            raise RuntimeError(
                "codex-native rig did not come online within "
                f"{_HEALTH_TIMEOUT_S:.0f}s.\nServer log:\n{server_log.read_text()[-3000:]}\n"
                f"Runner log:\n{runner_log.read_text()[-3000:]}"
            )
        yield (base_url, runner_id, runner_home, reader_home)
    finally:
        for proc in (runner_proc, server_proc):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in (runner_proc, server_proc):
            if proc is not None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        server_handle.close()
        runner_handle.close()


@pytest.mark.timeout(300)
def test_native_codex_terminal_failure_names_a_log_findable_on_host(
    codex_native_rig: tuple[str, str, Path, Path],
) -> None:
    """The failed-terminal message must point at a log the host user can find.

    Journey: create a codex-native session, bind the runner, and ensure the
    native Codex terminal. The terminal fails to start (codex CLI unreachable),
    and the error names a runner log. The regression: the named ``~/…`` path is
    home-relative to the runner, so a user on the host cannot find it.
    """
    base_url, runner_id, runner_home, reader_home = codex_native_rig

    create = _client.post(
        f"{base_url}/v1/sessions",
        data={"metadata": json.dumps({"workspace": str(runner_home)})},
        files={"bundle": ("codex.tar.gz", _spec_bundle(), "application/gzip")},
        timeout=30.0,
    )
    create.raise_for_status()
    session_id = str(create.json()["session_id"])

    _client.patch(
        f"{base_url}/v1/sessions/{session_id}",
        json={"runner_id": runner_id},
        timeout=60.0,
    ).raise_for_status()

    ensure = _client.post(
        f"{base_url}/v1/sessions/{session_id}/resources/terminals",
        json={"terminal": "codex", "session_key": "main", "ensure_native_terminal": True},
        timeout=120.0,
    )

    # --- Symptom 1 (stable precondition): the terminal fails to start. ---
    assert ensure.status_code >= 400, (
        f"expected the codex terminal start to fail, got {ensure.status_code}: {ensure.text}"
    )
    error = ensure.json()["error"]
    assert error["code"] == "native_terminal_start_failed", error
    message = error["message"]
    assert "Native Codex terminal failed to start" in message, message

    # The message must name a runner log file (not just a directory).
    named = message.rsplit(": ", 1)[-1].strip()
    assert named.endswith(".log"), f"message names no log file: {message!r}"

    # Confirm the runner actually wrote the log — it exists under the RUNNER's
    # home. The bug is not a missing log, it is an unfindable *reference* to it.
    runner_side = Path(str(runner_home) + named[1:]) if named.startswith("~") else Path(named)
    assert runner_side.exists(), (
        f"expected the runner to have written its log under its own home "
        f"({runner_side}); the rig may be misconfigured"
    )

    # --- The bug (fails while live, passes once fixed): the referenced log ---
    # --- must be findable on the host by a user reading the error.        ---
    host_side = _resolve_as_host_reader(named, reader_home)
    assert host_side.exists(), (
        "The failed-terminal message names a runner log that does not exist "
        f"on the host. The message shows {named!r}, which a user on the host "
        f"resolves to {host_side} — no such file (the log is under the "
        f"runner's own home at {runner_side}). The surfaced path must be "
        "resolvable on the host that reads the error."
    )
