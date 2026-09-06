// Focused contract tests for the bundled setup page. It has no Node access;
// platform-sensitive copy must remain renderer-only and leave the IPC bridge
// unchanged.

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const page = fs.readFileSync(path.join(__dirname, "..", "setup", "index.html"), "utf8");

describe("Windows CLI setup guidance", () => {
  it("uses Electron's platform information and names the Windows uv-tool path", () => {
    assert.match(page, /navigator\.userAgentData\?\.platform \|\| navigator\.platform/);
    assert.match(page, /%USERPROFILE%\\\\\.local\\\\bin\\\\omnigent\.exe/);
    assert.match(page, /or omni\.exe/);
  });

  it("keeps the setup page on the existing narrow preload bridge", () => {
    assert.match(page, /const setup = window\.omnigentSetup;/);
    assert.doesNotMatch(page, /require\s*\(/);
    assert.doesNotMatch(page, /ipcRenderer/);
  });
});
