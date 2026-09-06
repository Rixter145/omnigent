# Windows subscription-routing MVP and continuation spec

## Local consolidation (2026-09-06)

The Windows MVP and repairs are consolidated on
`codex/windows-subscription-routing-mvp`. Git reported one registered worktree;
`codex/gpt6where` had no unique commits to merge. The consolidation is local and
does not publish to the remote. Continue from this branch and the continuation
prompt in this directory.

The user approved committing with documented pre-commit limitations: the full
hook suite assumes Unix `.venv/bin` paths and includes unavailable Android/iOS
tooling. Applicable Windows checks include pinned Ruff lint/format, web
TypeScript, Electron tests, focused subscription tests, and staged whitespace
validation. No claim is made that the full cross-platform hook suite passed.

Consolidation verification: 153 focused Python tests and 439 Electron tests
passed. The Python run clears `OMNIGENT_CURSOR_WSL_DISTRO` and
`OMNIGENT_CURSOR_WSL_USER` in its child process to isolate test configuration
from machine-specific overrides. Pinned Ruff checks passed for all 76 staged
Python files.

## Status — accepted MVP (2026-09-06)

### Runtime repair verified (2026-09-06)

- Windows subscription readiness now reports `claude-native`, `native-claude`,
  `codex-native`, and `native-codex` as unavailable. These tmux/PTY wrappers
  cannot run on Windows; select Polly (`claude-sdk`) to orchestrate work.
- Root verification: 49 readiness tests passed, including Windows and
  non-Windows behavior. The restarted local host reports native routes false
  and `claude-sdk` / `codex` true.
- Polly session `c78ec50623dd4218874c42717cfa4fb1` completed a real SDK request
  with `WINDOWS_SDK_OK` and no error. No continuation implementation was started.
- Current limitation: the fresh host probe reports `cursor-wsl` as
  `binary-missing`; the earlier Cursor proof below is historical and must be
  revalidated before claiming current three-provider readiness.

### Previously accepted evidence

The Windows MVP is implemented and accepted from
`C:\Users\ricar\OneDrive\Documents\Omnigent` on branch
`codex/windows-subscription-routing-mvp`.

- The installed launcher resolves the OneDrive checkout.
- The local server and Windows host pass `WINDOWS_RUNTIME_VERIFIED`.
- Claude, Codex, and Cursor-in-WSL pass `SUBSCRIPTION_MVP_LIVE_OK` using
  their existing subscriptions and no Omnigent-managed API keys.
- Automatic Smart Routing produced a durable route decision and matching final
  receipt in parent session `78c17f4f0e314b98b4723d289699a687`.
- Explicit Cursor dispatch completed through `cursor-wsl` in parent session
  `37ec1d6271814112bc455bcbe7be7ad7`, child session
  `889528577253418591a2d34a549be093`, with marker
  `OMNI_CURSOR_E2E_F2A91C7D` and no task error.
- The packaged Electron executable passes `WINDOWS_DESKTOP_MVP_VERIFIED`.
- A cold packaged launch now detects the installed Windows CLI on its initial
  setup probe without requiring **Re-detect**. The bounded regression task was
  `OMNI-FIX-WINDOWS-CLI-COLD-DETECT-20`.
- Root verification passed 288 Python tests and 439 Electron tests. The final
  two Cursor remediations independently passed 53 and 96 focused tests.
- The single final read-only Sol Ultra review was
  `OMNI-WIN-MVP-FINAL-SOL-ULTRA-10`; its critical/high findings were
  remediated and independently reverified. Do not repeat that review merely to
  reconfirm the accepted MVP.

The accepted gate ledger is
`.unlazy/omnigent-windows-mvp/GATES.md`.

## Shipped boundary

Windows subscription routing is shipped for the locally authenticated Claude,
Codex, and Cursor CLI routes. Claude and Codex run through their existing
subscription-backed harnesses. Cursor remains a WSL-backed route with an
explicit distro and non-root user; it is unavailable to agents that require
Omnigent-owned tools.

Polly is the existing dispatch surface, not a second provider system. Smart
Routing selects an eligible subscription harness before execution, pins that
route for the session, and fails closed if its bounded candidate menu has no
ready route. Explicit Polly workers retain their requested provider; no worker
is silently replayed on another provider after it has begun.

The Electron app is a packaged native shell over the server-served Omnigent
web UI. When the endpoint is absent, its setup screen detects the installed
Windows CLI and can start the local backend through **Start locally**; the
remaining P0 lifecycle item is to make that recovery automatic on cold launch
instead of requiring the explicit click. Hosting remains an explicit UI flow.

The Windows NSIS artifact is currently unsigned. It is an MVP test artifact,
not a claim of SmartScreen- or enterprise-trusted distribution.

## Frozen product contract

The continuation must preserve these behaviors:

1. Omnigent remains the single user-facing dispatch and routing interface.
2. Polly uses Omnigent Smart Routing; it is not a separate model provider or a
   second router.
3. Automatic routes consider only ready, eligible harnesses. The selected
   route is pinned before execution and recorded truthfully in a durable
   receipt.
4. Explicit provider requests remain pinned to that provider. If the provider
   is unavailable or incompatible, execution fails closed with a useful error;
   it is never silently replayed elsewhere.
5. Claude, Codex, and Cursor authenticate through their own installed consumer
   subscriptions. Omnigent must not request, copy, persist, or synthesize their
   API keys or browser tokens.
6. Existing routing determinism and capability gates must not be weakened to
   add UI convenience or another provider.

## Reproducible acceptance

From the repository root, after creating the OneDrive copy and synchronizing
its editable environment, run the safe offline checks first:

```powershell
python -m pytest tests/test_verify_windows_app_mvp.py -q
python -m py_compile dev/verify_windows_app_mvp.py

python dev/verify_windows_app_mvp.py relocation `
  --original C:\src\omnigent `
  --destination C:\Users\<user>\OneDrive\Documents\Omnigent
python dev/verify_windows_app_mvp.py launcher `
  --destination C:\Users\<user>\OneDrive\Documents\Omnigent `
  --python .\.venv\Scripts\python.exe --uv uv
```

To enable subscription routing, bind a Windows host that has authenticated
Claude and/or Codex harnesses, then use Polly's `subscription_worker` (or an
explicit supported worker). Confirm the host session's authoritative
`configured_harnesses` report marks the selected harness exactly `true`.
Neither a binary on `PATH` nor a CLI version check establishes readiness.

The remaining commands are opt-in live checks. They can contact an already
running local service or the selected subscription provider; do not run them
without that intent. First send one small real request through
`subscription_worker` and retain its routing receipt, including the selected
harness and event identifier. Then inspect the durable receipt store and verify
that its selected harness, event identifier, and route decision exactly equal
the returned receipt. Finally, the no-API-key runtime verifier may confirm the
bound host is online:

```powershell

# This performs GET requests only. Start neither service nor host for this check.
python dev/verify_windows_app_mvp.py runtime --host-id <online-host-id>

pnpm --dir web/electron run build:win
python dev/verify_windows_app_mvp.py desktop `
  --executable .\web\electron\dist\win-unpacked\Omnigent.exe
```

`desktop` launches exactly the supplied unpacked executable, requires a visible
top-level window titled Omnigent from its process tree, and then terminates only
that launched tree. The verifier prints one of its success tokens only after
the requested check completes; any missing file, receipt mismatch, import from
another checkout, unhealthy backend, offline host, or absent window fails the
command.

The Electron package remains an unsigned MVP artifact. Manual UI acceptance is
therefore required: acknowledge any Windows warning as appropriate for the
test environment, launch the unpacked app, confirm its first setup probe finds
the CLI, use **Start locally** when the backend is absent, and verify the
existing UI hosting flow. Do not treat this as a
SmartScreen, enterprise-distribution, or embedded-backend claim.

## Remaining work, in priority order

### P0 — One-click Windows lifecycle

Opening the packaged app must be sufficient after a normal Windows login or
reboot. The Electron shell should check the configured local endpoint, reuse a
healthy existing backend and host, or start them non-interactively from the
installed OneDrive-backed launcher when absent.

Acceptance:

- A cold launch with nothing listening on port 6767 reaches a usable Omnigent
  window without requiring PowerShell.
- Startup propagates the Windows UTF-8 requirement and configured Cursor WSL
  distro/user without exposing credentials.
- A healthy pre-existing backend is reused, not replaced.
- Failure presents a bounded recovery screen with the failing component and
  log location; restart fails closed instead of claiming success.
- The app never kills a backend or host process it did not start unless the
  user explicitly invokes a Stop action.

### P0 — Cursor transcript normalization

The Cursor adapter currently proves dispatch and execution, but its persisted
raw response may echo framework/user prompt text and repeat the final answer.

Acceptance:

- The conversation stores and displays one normalized assistant answer.
- Framework instructions and the user's prompt are not reproduced as assistant
  output.
- Raw transport text may be retained only in an explicit diagnostic log.
- Normal process exit, cancellation, timeout, and malformed-output paths remain
  deterministic and covered by regression tests.

### P1 — Routing and authentication truth in the desktop UI

Expose the existing backend truth instead of adding a second routing engine.

Acceptance:

- Smart Routing is the default and shows the selected harness/model, decision
  identifier, reason, and final-route receipt.
- The user can explicitly pin Claude, Codex, or Cursor for a session.
- Readiness distinguishes installed, authenticated, transport-ready, and
  unavailable states, with a concrete repair action.
- The UI never labels a binary-on-PATH check as authenticated readiness and
  never claims a fallback that did not execute.

### P2 — Gemini and Antigravity subscription support

Add either provider only through an official installed CLI/SDK flow that owns
its own consumer login. Do not extract browser cookies, reuse private token
files outside the provider's supported interface, or turn this work into API
key brokerage.

Acceptance:

- Document the official authentication mechanism and credential owner first.
- Detect readiness with a bounded no-op or minimal provider-supported probe.
- Add an explicit harness and capability declaration before admitting it to
  automatic Smart Routing.
- Pin the selected route and produce the same durable receipt as existing
  providers.
- If no supported consumer-subscription interface exists, report the provider
  as unsupported and stop; do not weaken the credential policy.

### P3 — Distribution hardening

- Add a reproducible signed NSIS build once the user supplies an appropriate
  Windows code-signing identity. Functional development must not fabricate or
  bypass this external requirement.
- Add installer upgrade/uninstall coverage while preserving local history and
  configuration.

## Continuation non-goals

- Replacing the existing Omnigent web UI or Smart Routing algorithm.
- Rewriting the app in Tauri, WinUI, or another desktop framework.
- Cloud hosting, account synchronization, billing, or API-key management.
- Adding providers before P0 lifecycle and Cursor-output gates pass.
- Broad refactors unrelated to the acceptance criteria above.

## Continuation definition of done

For each priority, use a bounded worker with non-overlapping owned paths, then
independently rerun its verification from the root checkout. The continuation
is complete only when a clean Windows cold start, subscription readiness,
automatic routing with a truthful receipt, explicit provider routing, desktop
packaging, and focused regression suites all pass. Use lower-cost workers for
implementation and reserve GPT-5.6 Sol Ultra for one final read-only review.
