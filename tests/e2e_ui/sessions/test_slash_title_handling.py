"""Browser e2e: '/' in session titles must be surfaced or sanitized, never swallowed.

Two user journeys, two facets of the same bug:

1. **Manual rename with '/' flickers and is silently rejected.** The sidebar's
   inline rename fires ``PATCH /v1/sessions/{id}`` optimistically: the row
   paints the new name immediately, and ``useRenameConversation.onError`` rolls
   it back *without any toast, alert, or validation message* (contrast with the
   delete/archive mutations, which toast on failure). Against managed Omnigent
   the WHS-homed storage adapter rejects titles containing '/', so the user
   sees the row flicker to the new name and back with zero feedback. The local
   SQLAlchemy store accepts slashes, so the managed backend's rejection is
   simulated here by intercepting the rename PATCH — the UI contract under
   test ("a rejected rename must surface visible feedback") is backend-agnostic.

2. **Auto-naming passes an unsanitized URL title to the storage adapter.** The
   first user message seeds the session title verbatim via
   ``synthesize_conversation_title`` (no sanitization), so a prompt that begins
   with a URL produces a title containing ``https://…``. The managed WHS-homed
   store subclass consumes that title unsanitized, producing a production
   storage error. This test drives the real journey (composer send) and asserts
   the persisted auto-title carries no raw '/' path separators to the store.

Both tests FAIL on the unfixed build — that failure is the reproduction.
"""

from __future__ import annotations

import json
import time

import httpx
from playwright.sync_api import Locator, Page, Route, expect

from tests.e2e_ui.conftest import configure_mock_llm

_COMPOSER = "Send a message…"

# The slash-y name a user might reasonably type when renaming.
_SLASH_TITLE = "docs/setup notes"

# Any user-visible feedback surface the app could use for a failed/blocked
# rename: a sonner toast (all `showToast` call sites carry testId "toast",
# and sonner stamps `data-sonner-toast` on every toast it renders), an ARIA
# alert, or the rename input flagged invalid by inline validation.
_FEEDBACK_SELECTOR = ", ".join(
    [
        "[data-sonner-toast]",
        '[data-testid="toast"]',
        '[role="alert"]',
        '[data-testid="rename-conversation-input"][aria-invalid="true"]',
    ]
)

# Records the row's rendered title once per animation frame (same sampler as
# test_sidebar_rename.py), so the optimistic paint + silent revert — each only
# a few frames long — are observable as evidence in the failure message.
_FRAME_SAMPLER = """
(sessionId) => {
  window.__renameSamples = [];
  const tick = () => {
    const link = document.querySelector(`a[href="/c/${sessionId}"]`);
    const span = link && (link.querySelector('span.relative') || link.querySelector('span'));
    window.__renameSamples.push({
      text: span
        ? Array.from(span.childNodes)
            .filter((n) => n.nodeType === 3)
            .map((n) => n.textContent)
            .join('')
            .trim()
        : null,
      editing: !!document.querySelector('[data-testid="rename-conversation-input"]'),
    });
    window.__renameRaf = requestAnimationFrame(tick);
  };
  window.__renameRaf = requestAnimationFrame(tick);
}
"""


def _row(page: Page, session_id: str) -> Locator:
    """Locate the sidebar row (``<li>``) for *session_id* by its href."""
    return page.locator("li").filter(has=page.locator(f'a[href="/c/{session_id}"]'))


def _visible_feedback_count(page: Page) -> int:
    """Count visible rename-feedback elements (toast / alert / invalid input).

    Text-bearing surfaces must have non-empty text to count; the flagged
    rename input counts on visibility alone (inputs carry no inner text).
    """
    count = 0
    feedback = page.locator(_FEEDBACK_SELECTOR)
    for i in range(feedback.count()):
        el = feedback.nth(i)
        if not el.is_visible():
            continue
        tag = el.evaluate("(n) => n.tagName.toLowerCase()")
        if tag in ("input", "textarea") or (el.inner_text() or "").strip():
            count += 1
    return count


def test_rejected_slash_rename_surfaces_feedback(
    page: Page,
    seeded_session: tuple[str, str],
) -> None:
    """A rename rejected by the backend must not be swallowed silently.

    Reproduces the first facet: rename a session to a name containing '/'
    against a backend that rejects it (managed Omnigent's WHS-homed store
    does; simulated here by rejecting the PATCH like that deployment's
    storage adapter). Today the row flickers to the new name, silently
    reverts, and the app shows **no** validation message, toast, or alert —
    so this test fails on the unfixed build.

    It passes once the app gives the user *any* visible feedback, whichever
    fix shape lands: client-side validation that blocks the '/' before the
    PATCH (inline message / invalid input), or an error toast/alert when the
    rename PATCH fails.

    :param page: Playwright page fixture (fresh context per test).
    :param seeded_session: ``(base_url, session_id)`` for a pre-created
        runner-bound session.
    """
    base_url, session_id = seeded_session

    # Reject slash-titled rename PATCHes exactly where the managed WHS-homed
    # storage adapter does. Everything else (GETs, the fixture's own PATCH
    # traffic) passes through to the real server.
    rejections = [0]

    def _reject_slash_rename(route: Route) -> None:
        request = route.request
        body = request.post_data_json if request.post_data else None
        title = (body or {}).get("title")
        if request.method == "PATCH" and isinstance(title, str) and "/" in title:
            rejections[0] += 1
            route.fulfill(
                status=500,
                content_type="application/json",
                body=json.dumps(
                    {
                        "error": {
                            "code": "internal_error",
                            "message": "Storage error: title contains an invalid path separator",
                        }
                    }
                ),
            )
            return
        route.continue_()

    page.route(f"**/v1/sessions/{session_id}", _reject_slash_rename)

    page.goto(f"{base_url}/c/{session_id}")
    row = _row(page, session_id)
    expect(row).to_be_visible()
    link = page.locator(f'a[href="/c/{session_id}"]')
    old_title = link.inner_text().strip()

    assert _visible_feedback_count(page) == 0, (
        "precondition: the page already shows a toast/alert before the rename — "
        "the feedback assertion below would not be attributable to the rename"
    )

    # Rename via the row kebab, typing a name with '/' like a user would.
    row.hover()
    row.get_by_test_id("conversation-actions").click()
    page.get_by_test_id("rename-conversation").click()
    edit = page.get_by_test_id("rename-conversation-input")
    expect(edit).to_be_visible()

    page.evaluate(_FRAME_SAMPLER, session_id)
    edit.fill(_SLASH_TITLE)
    edit.press("Enter")

    # Wait for the commit to settle: either feedback appears (fixed behavior,
    # whichever shape), or the PATCH was rejected and the row silently
    # reverted to the old name (the bug). Poll via wait_for_timeout — the
    # sync API dispatches the route handler cooperatively on this thread.
    silently_reverted = False
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if _visible_feedback_count(page) > 0:
            break
        current = page.evaluate("() => window.__renameSamples[window.__renameSamples.length - 1]")
        if (
            rejections[0] > 0
            and current
            and not current["editing"]
            and current["text"] == old_title
        ):
            # The rejected rename already rolled back; give any late feedback
            # (toast animation, error banner) a grace period to appear.
            page.wait_for_timeout(1500)
            silently_reverted = _visible_feedback_count(page) == 0
            break
        page.wait_for_timeout(100)

    samples = page.evaluate(
        "() => { cancelAnimationFrame(window.__renameRaf); return window.__renameSamples; }"
    )
    flicker_frames = [s for s in samples if not s["editing"] and s["text"] == _SLASH_TITLE]

    assert _visible_feedback_count(page) > 0, (
        f"rename to {_SLASH_TITLE!r} was rejected by the backend "
        f"(PATCH rejected {rejections[0]}x) and swallowed silently: the row "
        f"painted the new name for {len(flicker_frames)} frame(s), reverted to "
        f"{old_title!r} (silently_reverted={silently_reverted}), and no "
        "validation message, toast, or alert ever appeared"
    )


def test_auto_title_from_url_prompt_is_sanitized_for_storage(
    page: Page,
    seeded_session: tuple[str, str],
    mock_llm_server_url: str,
) -> None:
    """A URL-first prompt must not push raw '/' into the stored auto-title.

    Reproduces the second facet: start a session whose first prompt begins
    with a URL. The deterministic seed title is the prompt text verbatim
    (``synthesize_conversation_title`` does no sanitization), so the title
    handed to the conversation-store adapter contains ``https://…`` with raw
    slashes — which the managed deployment's WHS-homed store rejects with a
    production storage error. This asserts the first persisted auto-title is
    storage-safe (no raw '/'), so it fails on the unfixed build, where the
    stored title is the URL itself.

    :param page: Playwright page fixture (fresh context per test).
    :param seeded_session: ``(base_url, session_id)`` for a pre-created
        runner-bound session (created without a title).
    :param mock_llm_server_url: Mock LLM base URL for the turn's scripted reply.
    """
    base_url, session_id = seeded_session
    prompt = "https://example.com/docs/setup review this page for me"
    configure_mock_llm(
        mock_llm_server_url,
        [{"text": "Reviewed the page."}],
        key="url-first-auto-title",
        match="review this page",
    )

    # Real journey: the user's first message in a fresh session is a URL.
    page.goto(f"{base_url}/c/{session_id}")
    composer = page.get_by_placeholder(_COMPOSER)
    expect(composer).to_be_visible()
    composer.fill(prompt)
    page.get_by_role("button", name="Send", exact=True).click()

    # The seed title is written when the user message persists — before any
    # background title generation can replace it — so the first non-null
    # title the server reports is exactly what reached the storage adapter.
    title: str | None = None
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        snap = httpx.get(f"{base_url}/v1/sessions/{session_id}", timeout=10.0)
        snap.raise_for_status()
        title = snap.json().get("title")
        if title:
            break
        page.wait_for_timeout(200)

    assert title, "session never received an auto-generated title from its first prompt"
    assert "/" not in title, (
        f"auto-naming passed an unsanitized title {title!r} to the storage "
        "adapter — on managed Omnigent the WHS-homed store rejects the raw "
        "'/' with a production storage error"
    )
