# Omnigent Windows continuation prompt

Paste the prompt below into Omnigent/Polly to continue from the accepted MVP.

Create the session with the **Polly** agent (`claude-sdk`). The **Claude Code**
terminal wrapper (`claude-native-ui`) requires tmux/PTY and cannot run on Windows.

```text
Continue the Omnigent Windows subscription-routing project from the accepted
MVP in C:\Users\ricar\OneDrive\Documents\Omnigent on branch
codex/windows-subscription-routing-mvp.

Read these first:
- AGENTS.md
- docs/windows-subscription-routing.md
- .unlazy/omnigent-windows-mvp/GATES.md

Do not rebuild the MVP or redesign Smart Routing. Treat the shipped behavior
and frozen product contract in the spec as authoritative. Preserve all
unrelated tracked and untracked user work.

Current verified baseline:
- OneDrive launcher, local server, Windows host, and packaged Electron app pass.
- Claude, Codex, and Cursor-in-WSL authenticate through existing subscriptions
  with no Omnigent-managed API keys.
- Automatic Smart Routing has a durable matching route receipt.
- Explicit Cursor dispatch works through cursor-wsl.
- A cold packaged launch detects the installed Windows CLI on its initial
  setup probe without requiring Re-detect (`OMNI-FIX-WINDOWS-CLI-COLD-DETECT-20`).
- Python: 288 passed. Electron: 439 passed. Cursor regressions: 53 and 96 passed.
- One final Sol Ultra review already covered the accepted MVP; do not repeat it.

Objective: complete the remaining work in strict priority order:

1. P0 one-click Windows lifecycle. The setup screen can already detect the CLI
   and start the backend with **Start locally**; now make a packaged-app cold
   start reuse or safely start the local backend and host automatically without
   requiring PowerShell or that extra click. Never kill processes the app does
   not own except through an explicit user Stop.
2. P0 Cursor transcript normalization. Persist/display one assistant answer;
   remove framework/user prompt echo and duplicate final output while keeping
   raw transport data only in diagnostic logs.
3. P1 desktop routing/auth truth. Show provider readiness and the actual Smart
   Routing decision/final receipt; support explicit Claude, Codex, and Cursor
   pins without silent fallback.
4. P2 Gemini/Antigravity. Proceed only through an official CLI/SDK-owned
   consumer subscription login. Never scrape browser cookies, copy private
   tokens, or substitute API keys. If no supported flow exists, document that
   result and fail closed.
5. P3 signing/installer hardening. Complete everything not requiring an
   external certificate; stop and identify the exact user-supplied signing
   prerequisite when reached.

Execution rules:
- Use Polly to dispatch bounded implementation tasks to subscription-backed
  workers. Give each task explicit owned paths and acceptance commands; do not
  run workers with overlapping write scopes.
- Use lower-cost workers for implementation. Use GPT-5.6 Sol Ultra only once,
  as the final read-only review after all implementation and root verification.
- Investigate and reproduce each defect before patching it.
- Keep automatic Smart Routing capability-gated, deterministic, pinned, and
  receipt-backed. Explicit routes must fail closed rather than silently switch.
- Do not push, open a PR, delete files, or discard dirty-worktree changes unless
  I explicitly authorize it.
- After every worker, inspect its diff and independently rerun its verification
  from the root checkout. A worker's success claim is not acceptance evidence.
- Update docs/windows-subscription-routing.md and a continuation gate ledger as
  behavior becomes verified. Separate DONE, PARTIAL, and BLOCKED honestly.
- Do not expand into a UI rewrite, cloud service, billing, or provider-token
  brokerage.

Start now with a read-only preflight, convert the first P0 lifecycle gap into a
bounded task, dispatch it, and continue autonomously until a real credential,
certificate, authority, or integrity gate requires my input.
```
