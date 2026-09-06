"""E2E regression: a badge in the composer's opening rule drops the first message.

The reported journey: with ``fastMode: true`` in
``~/.claude/settings.json``, Claude Code >= 2.1.261 running Opus 5 renders a
fast-mode badge *inside* the composer's opening rule::

    ──────────────────────────────────────────────────────── ⇯ ─
    ❯
    ────────────────────────────────────────────────────────────
      ⏵⏵ bypass permissions on (shift+tab to cycle)

``_is_box_rule`` requires every character of a rule to be a box glyph, so the
badged opening rule stops counting as a rule, ``_composer_row`` anchors on the
closing rule instead and lands on the statusline, and the prompt-readiness
gate never turns true. Every first web-UI message to such a session dies with
``ClaudePromptTimeout`` while the pane shows a perfectly healthy TUI.

This test drives the real delivery transport end-to-end: a real tmux server on
a private socket, advertised through the production ``write_tmux_target``, and
a delivery through the public ``inject_user_message`` entry point — the exact
call the runner makes for a web-UI message. The TUI in the pane is a scripted
composer that renders, byte-for-byte, the frame a real Claude Code 2.1.261
(Opus 5, fast mode on) was observed to draw; scripting it keeps the test
independent of the installed Claude Code version and of the fast-mode org
entitlement (CI's pinned CLI predates the badge), while everything on the
omnigent side of the pane is production code.

Before the fix: the readiness gate never sees the composer under the badged
opening rule, ``inject_user_message`` raises ``ClaudePromptTimeout``, and
``test_first_message_delivered_with_badged_opening_rule`` FAILS.
After a fix (a rule that is *predominantly* box glyphs still counts):
delivery proceeds and both tests PASS. The plain-rule control pins that the
badge is the only discriminator, so the harness cannot pass vacuously.

Runs with no LLM, no claude binary, and no server — only ``tmux``::

    pytest tests/e2e/test_claude_native_fast_mode_badge_e2e.py -v
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from omnigent.claude_native_bridge import (
    _BRIDGE_ROOT,
    inject_user_message,
    write_tmux_target,
)

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="requires tmux on PATH")

#: The opening rule exactly as a real Claude Code 2.1.261 (Opus 5, fastMode on,
#: 60-column pane) draws it: box glyphs with the fast-mode badge set into the
#: rule near its right edge.
BADGED_OPENING_RULE = "─" * 56 + " ⇯ ─"

#: The same rule without the badge — what every pre-fast-mode Claude Code
#: version draws, and what the same CLI draws with ``fastMode: false``.
PLAIN_OPENING_RULE = "─" * 60

#: The message "typed" into the web UI composer.
MESSAGE = "hello from the omnigent web UI"

#: Delivery budget. Generous against slow CI, small enough that the
#: before-fix failure (which burns the whole budget polling) stays quick.
DELIVERY_TIMEOUT_S = 12.0

#: How long to wait for the scripted TUI to draw its first frame.
TUI_BOOT_TIMEOUT_S = 15.0

#: A minimal interactive composer that renders the observed Claude Code frame
#: and behaves like the real input box as far as the delivery path probes it:
#: the pasted draft becomes visible on the prompt row (the paste-commit poll),
#: Ctrl-K kills the draft (the stale-input clear), and Enter submits — the row
#: clears (the submit-verification poll) and the draft is appended to the
#: delivered-file, which is this test's proof the message reached the TUI.
FAKE_TUI = r'''
import os
import sys
import termios

opening_rule = sys.argv[1]
delivered_path = sys.argv[2]

# Keep output processing (\n -> \r\n) but take input raw and unechoed, like a
# real TUI: the tty driver must not echo injected keystrokes into the frame.
fd = sys.stdin.fileno()
attrs = termios.tcgetattr(fd)
attrs[3] &= ~(termios.ECHO | termios.ICANON)  # lflag
termios.tcsetattr(fd, termios.TCSANOW, attrs)

MARK_OPEN = b"\x1b[200~"
MARK_CLOSE = b"\x1b[201~"


def render(draft: str) -> None:
    row = draft.replace("\r", "⏎")
    lines = [
        " ▐▛███▛█   Claude Code v2.1.261",
        "▝▜██████▀  Opus 5 · API Usage Billing",
        "  ▝▝ ▝▝    " + os.getcwd(),
        opening_rule,
        "❯ " + row,
        "─" * 60,
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)",
    ]
    sys.stdout.write("\x1b[2J\x1b[H" + "\n".join(lines))
    sys.stdout.flush()


buf = b""
draft = ""
in_paste = False
render("")
while True:
    try:
        data = os.read(fd, 4096)
    except OSError:
        break
    if not data:
        break
    buf += data
    while buf:
        if in_paste:
            idx = buf.find(MARK_CLOSE)
            if idx >= 0:
                draft += buf[:idx].decode("utf-8", "replace")
                buf = buf[idx + len(MARK_CLOSE):]
                in_paste = False
            else:
                # Hold back a possible split close-marker prefix at the tail.
                hold = 0
                for k in range(len(MARK_CLOSE) - 1, 0, -1):
                    if buf.endswith(MARK_CLOSE[:k]):
                        hold = k
                        break
                cut = len(buf) - hold
                draft += buf[:cut].decode("utf-8", "replace")
                buf = buf[cut:]
                render(draft)
                break  # need more bytes
            render(draft)
            continue
        head = buf[0:1]
        if head == b"\x1b":
            if buf.startswith(MARK_OPEN):
                in_paste = True
                buf = buf[len(MARK_OPEN):]
            elif MARK_OPEN.startswith(buf):
                break  # partial marker; wait for more bytes
            else:
                buf = buf[1:]  # a lone Escape (dismiss attempt) — ignore
            continue
        if head in (b"\r", b"\n"):
            if draft.strip():
                with open(delivered_path, "a", encoding="utf-8") as fh:
                    fh.write(draft.replace("\r", "\n") + "\n")
            draft = ""
            render("")
            buf = buf[1:]
            continue
        if head == b"\x0b":  # Ctrl-K: kill to end of line (stale-input clear)
            draft = ""
            render("")
            buf = buf[1:]
            continue
        if head < b" ":  # other control bytes (Ctrl-A home, etc.) — no-op
            buf = buf[1:]
            continue
        draft += head.decode("utf-8", "replace")
        render(draft)
        buf = buf[1:]
'''


def _tmux(socket_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run one tmux command against the private test server.

    :param socket_path: The test's private tmux socket.
    :param args: tmux subcommand and arguments.
    :returns: The completed process (checked).
    """
    return subprocess.run(
        ["tmux", "-S", str(socket_path), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30.0,
    )


def _pane_with_opening_rule(tmp_path: Path, opening_rule: str) -> Iterator[tuple[Path, Path]]:
    """A claude-native bridge dir advertising a scripted composer pane.

    Boots the scripted TUI (rendering *opening_rule* above the prompt row) in
    a real tmux pane, advertises it through the production
    ``write_tmux_target``, and waits for the first frame.

    :param tmp_path: Per-test temp dir for the socket, script, delivered file.
    :param opening_rule: The composer's opening rule line to render.
    :yields: ``(bridge_dir, delivered_file)``.
    """
    socket_path = tmp_path / "tmux.sock"
    script_path = tmp_path / "fake_tui.py"
    delivered_file = tmp_path / "delivered.txt"
    script_path.write_text(FAKE_TUI, encoding="utf-8")
    subprocess.run(
        [
            "tmux",
            "-S",
            str(socket_path),
            "new-session",
            "-d",
            "-s",
            "claude",
            "-x",
            "60",
            "-y",
            "24",
            f"{sys.executable} {script_path} '{opening_rule}' {delivered_file}",
        ],
        check=True,
        timeout=30.0,
    )

    # The bridge validates its dir sits under the trusted claude-native root,
    # so the fixture cannot use tmp_path for it.
    bridge_dir = _BRIDGE_ROOT / f"badge-test-{uuid.uuid4().hex}"
    write_tmux_target(bridge_dir, socket_path=socket_path, tmux_target="claude")

    deadline = time.monotonic() + TUI_BOOT_TIMEOUT_S
    while True:
        pane = _tmux(socket_path, "capture-pane", "-p", "-t", "claude").stdout
        if opening_rule in pane and "❯" in pane:
            break
        if time.monotonic() >= deadline:
            raise AssertionError(f"scripted TUI never rendered its frame; pane:\n{pane}")
        time.sleep(0.1)

    try:
        yield bridge_dir, delivered_file
    finally:
        subprocess.run(
            ["tmux", "-S", str(socket_path), "kill-server"],
            check=False,
            timeout=30.0,
        )
        shutil.rmtree(bridge_dir, ignore_errors=True)


@pytest.fixture
def badged_pane(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """The reported pane: fast-mode badge inside the composer's opening rule."""
    yield from _pane_with_opening_rule(tmp_path, BADGED_OPENING_RULE)


@pytest.fixture
def plain_pane(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """The control pane: the same composer framed by an unbadged rule."""
    yield from _pane_with_opening_rule(tmp_path, PLAIN_OPENING_RULE)


def test_first_message_delivered_with_badged_opening_rule(
    badged_pane: tuple[Path, Path],
) -> None:
    """The first message must reach a composer whose opening rule carries a badge.

    Before the fix this raises ``ClaudePromptTimeout`` ("input prompt never
    rendered … The message was not delivered.") out of the production
    delivery path, because the badge stops the opening rule from counting as
    a rule and the readiness gate never sees the input box.
    """
    bridge_dir, delivered_file = badged_pane
    inject_user_message(bridge_dir, content=MESSAGE, timeout_s=DELIVERY_TIMEOUT_S)
    assert delivered_file.exists() and MESSAGE in delivered_file.read_text(encoding="utf-8"), (
        "inject_user_message returned but the message never reached the TUI"
    )


def test_first_message_delivered_with_plain_opening_rule(
    plain_pane: tuple[Path, Path],
) -> None:
    """Control: the identical journey with an unbadged rule delivers.

    This pins the badge as the only discriminator: if this control ever
    fails, the scripted composer (not the badge handling) is broken, and the
    badged test's failure would prove nothing.
    """
    bridge_dir, delivered_file = plain_pane
    inject_user_message(bridge_dir, content=MESSAGE, timeout_s=DELIVERY_TIMEOUT_S)
    assert delivered_file.exists() and MESSAGE in delivered_file.read_text(encoding="utf-8"), (
        "inject_user_message returned but the message never reached the TUI"
    )
