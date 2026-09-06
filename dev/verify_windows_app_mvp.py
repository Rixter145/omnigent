"""Fail-closed checks for the Windows desktop MVP.

Each subcommand is read-only except ``desktop``.  That command starts the
explicitly supplied unpacked executable and terminates only that process tree
after observing its window.  A success token is emitted only after every
assertion for the requested subcommand passes.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit
from urllib.request import urlopen

RELOCATION_VERIFIED = "RELOCATION_VERIFIED"
ONEDRIVE_LAUNCHER_VERIFIED = "ONEDRIVE_LAUNCHER_VERIFIED"
WINDOWS_RUNTIME_VERIFIED = "WINDOWS_RUNTIME_VERIFIED"
WINDOWS_DESKTOP_MVP_VERIFIED = "WINDOWS_DESKTOP_MVP_VERIFIED"

CONTINUATION_PATHS = frozenset(
    {
        "dev/verify_windows_app_mvp.py",
        "docs/windows-subscription-routing.md",
        "examples/polly/agents/cursor/config.yaml",
        "examples/polly/config.yaml",
        "omnigent/cursor_wsl.py",
        "omnigent/harness_plugins.py",
        "omnigent/inner/cursor_wsl_executor.py",
        "omnigent/model_catalog.py",
        "omnigent/onboarding/harness_readiness.py",
        "omnigent/onboarding/subscription_readiness.py",
        "omnigent/runtime/harnesses/_runner.py",
        "omnigent/runtime/harnesses/process_manager.py",
        "omnigent/runtime/workflow.py",
        "omnigent/server/routes/_sessions/orchestration.py",
        "omnigent/subscription_defaults.py",
        "tests/inner/test_cursor_wsl_executor.py",
        "tests/onboarding/test_harness_readiness.py",
        "tests/onboarding/test_subscription_readiness.py",
        "tests/runtime/harnesses/test_process_manager.py",
        "tests/runtime/harnesses/test_runner.py",
        "tests/runtime/test_provider_spawn_env.py",
        "tests/runtime/test_subscription_harness_spawn_env.py",
        "tests/server/integration/test_subscription_routing_mvp.py",
        "tests/test_cursor_wsl.py",
        "tests/test_harness_plugins.py",
        "tests/test_model_catalog.py",
        "tests/test_verify_windows_app_mvp.py",
        "web/electron/src/omnigent_cli.js",
        "web/electron/src/server_manager.js",
        "web/electron/setup/index.html",
        "web/electron/test/omnigent_cli.test.js",
        "web/electron/test/server_manager.test.js",
        "web/electron/test/setup.test.js",
    }
)
PACKAGE_TARGETS = {
    "omnigent": Path("."),
    "omnigent-client": Path("sdks/python-client"),
    "omnigent-ui-sdk": Path("sdks/ui"),
}
IMPORT_TARGETS = {
    "omnigent": Path("."),
    "omnigent_client": Path("sdks/python-client"),
    "omnigent_ui_sdk": Path("sdks/ui"),
}
ORCHESTRATION_METADATA_DIRECTORIES = frozenset({".codex-tasks", ".unlazy"})


class VerificationError(RuntimeError):
    """Raised when MVP evidence is missing or inconsistent."""


@dataclass(frozen=True)
class GitStatusEntry:
    """One complete `git status --porcelain=v1 -z` record."""

    xy: str
    path: str
    source_path: str | None = None


def _normal_path(value: str) -> str:
    return value.replace("\\", "/").strip("/")


def _resolved_directory(value: Path, label: str) -> Path:
    resolved = value.expanduser().resolve()
    if not resolved.is_dir():
        raise VerificationError(f"{label} is not a directory: {resolved}")
    return resolved


def _is_allowed(relative: str, allowed_paths: Iterable[str]) -> bool:
    return relative in allowed_paths


def _is_orchestration_metadata(relative: str) -> bool:
    return relative.split("/", 1)[0] in ORCHESTRATION_METADATA_DIRECTORIES


def _safe_relative_path(value: str) -> str:
    """Validate a Git porcelain path before resolving it under a checkout."""
    if not value or "\\" in value or value.startswith("/"):
        raise VerificationError(f"unsafe Git status path: {value!r}")
    if len(value) >= 2 and value[1] == ":":
        raise VerificationError(f"unsafe Git status path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise VerificationError(f"unsafe Git status path: {value!r}")
    return value


def _parse_porcelain_entries(output: bytes) -> set[GitStatusEntry]:
    """Parse complete v1 -z status records without walking the checkout."""
    fields = output.split(b"\0")
    entries: set[GitStatusEntry] = set()
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        if len(field) < 4 or field[2:3] != b" ":
            raise VerificationError("invalid Git status porcelain entry")
        status = os.fsdecode(field[:2])
        path = _safe_relative_path(os.fsdecode(field[3:]))
        source_path: str | None = None
        if "R" in status or "C" in status:
            if index >= len(fields) or not fields[index]:
                raise VerificationError("truncated Git rename/copy porcelain entry")
            source_path = _safe_relative_path(os.fsdecode(fields[index]))
            index += 1
        entries.add(GitStatusEntry(status, path, source_path))
    return entries


def _parse_porcelain_paths(output: bytes) -> set[str]:
    """Return every endpoint reported by v1 -z status records."""
    return {
        path
        for entry in _parse_porcelain_entries(output)
        for path in (entry.path, entry.source_path)
        if path is not None
    }


def _git_output(root: Path, arguments: Sequence[str]) -> bytes:
    git_root = root.as_posix()
    command = ["git", "-c", f"safe.directory={git_root}", "-C", git_root, *arguments]
    try:
        completed = subprocess.run(command, capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VerificationError(f"could not run git: {exc}") from exc
    if completed.returncode != 0:
        detail = os.fsdecode(completed.stderr).strip() or "no diagnostic"
        raise VerificationError(f"git command failed: {detail}")
    return completed.stdout


def _git_text(root: Path, arguments: Sequence[str]) -> str:
    return os.fsdecode(_git_output(root, arguments)).strip()


def _checkout_identity(root: Path) -> tuple[str, str]:
    branch = _git_text(root, ["branch", "--show-current"])
    head = _git_text(root, ["rev-parse", "HEAD"])
    if not branch or not head:
        raise VerificationError(f"checkout has no branch or HEAD: {root}")
    return branch, head


def _git_scope(root: Path) -> set[GitStatusEntry]:
    status = _git_output(root, ["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    return {
        entry
        for entry in _parse_porcelain_entries(status)
        if not _is_orchestration_metadata(entry.path)
        and (entry.source_path is None or not _is_orchestration_metadata(entry.source_path))
    }


def _path_identity(path: Path) -> tuple[int, bool]:
    """Return the lstat file type and Windows reparse-point flag for one path."""
    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_point = bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_IFMT(metadata.st_mode), reparse_point


def verify_relocation(
    original: Path,
    destination: Path,
    *,
    expected_count: int = 88,
    allowed_paths: Iterable[str] = CONTINUATION_PATHS,
) -> None:
    """Prove the Git-reported migration baseline is preserved without tree walks."""
    original_root = _resolved_directory(original, "original checkout")
    destination_root = _resolved_directory(destination, "OneDrive destination")
    allowed = {_normal_path(path) for path in allowed_paths}
    if not allowed:
        raise VerificationError("continuation paths must be relative, non-empty paths")
    allowed = {_safe_relative_path(path) for path in allowed}
    if expected_count < 0:
        raise VerificationError("expected baseline count must be non-negative")

    original_branch, original_head = _checkout_identity(original_root)
    destination_branch, destination_head = _checkout_identity(destination_root)
    failures: list[str] = []
    if original_branch != destination_branch:
        failures.append(f"branch mismatch: {original_branch} != {destination_branch}")
    if original_head != destination_head:
        failures.append(f"HEAD mismatch: {original_head} != {destination_head}")

    baseline = _git_scope(original_root)
    if len(baseline) != expected_count:
        failures.append(f"baseline scope count: expected {expected_count}, got {len(baseline)}")
    destination_scope = _git_scope(destination_root)
    baseline_paths = {entry.path for entry in baseline}
    destination_paths = {entry.path for entry in destination_scope}
    for relative in sorted(destination_paths - baseline_paths - allowed):
        failures.append(f"unexpected destination dirty path: {relative}")
    for relative in sorted(baseline_paths - destination_paths):
        failures.append(f"missing destination dirty path: {relative}")
    for relative in sorted(baseline_paths & destination_paths):
        baseline_entries = {entry for entry in baseline if entry.path == relative}
        destination_entries = {entry for entry in destination_scope if entry.path == relative}
        if baseline_entries != destination_entries:
            failures.append(f"Git state mismatch: {relative}")
    for entry in sorted(
        baseline, key=lambda value: (value.path, value.source_path or "", value.xy)
    ):
        relative = entry.path
        source = original_root / relative
        copied = destination_root / relative
        try:
            source_identity = _path_identity(source)
        except FileNotFoundError:
            failures.append(f"missing baseline file: {relative}")
            continue
        try:
            copied_identity = _path_identity(copied)
        except FileNotFoundError:
            failures.append(f"missing baseline file: {relative}")
            continue
        if source_identity != copied_identity:
            failures.append(f"file type or reparse mismatch: {relative}")
        elif (
            source_identity[0] == stat.S_IFREG
            and not source_identity[1]
            and not _is_allowed(relative, allowed)
            and source.read_bytes() != copied.read_bytes()
        ):
            failures.append(f"byte mismatch: {relative}")
    if failures:
        raise VerificationError("relocation verification failed: " + "; ".join(failures[:10]))


def _run_capture(argv: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(argv), capture_output=True, check=False, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VerificationError(f"could not run {argv[0]}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise VerificationError(f"command failed ({argv[0]}): {detail}")
    return completed.stdout


def _receipt_locations(receipt: str) -> dict[str, Path]:
    locations: dict[str, Path] = {}
    current_name: str | None = None
    for raw_line in receipt.splitlines():
        line = raw_line.strip()
        if not line:
            current_name = None
            continue
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key.casefold() == "name":
            current_name = value.casefold()
        elif key.casefold() == "editable project location" and current_name:
            locations[current_name] = Path(value).expanduser().resolve()
    return locations


def validate_editable_receipt(receipt: str, destination: Path) -> None:
    """Validate uv's editable-install receipt against the copied checkout."""
    target_root = _resolved_directory(destination, "OneDrive destination")
    locations = _receipt_locations(receipt)
    failures: list[str] = []
    for package, relative_target in PACKAGE_TARGETS.items():
        actual = locations.get(package)
        expected = (target_root / relative_target).resolve()
        if actual is None:
            failures.append(f"missing editable receipt for {package}")
        elif actual != expected:
            failures.append(f"editable receipt for {package} is {actual}, expected {expected}")
    if failures:
        raise VerificationError("launcher receipt verification failed: " + "; ".join(failures))


def _under(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def validate_import_origins(origins: dict[str, str], destination: Path) -> None:
    """Require all three Python packages to execute from the copied checkout."""
    target_root = _resolved_directory(destination, "OneDrive destination")
    failures: list[str] = []
    for module, relative_target in IMPORT_TARGETS.items():
        origin = origins.get(module)
        if not isinstance(origin, str) or not origin:
            failures.append(f"missing import origin for {module}")
        elif not _under(Path(origin), target_root / relative_target):
            failures.append(f"{module} imported outside destination: {origin}")
    if failures:
        raise VerificationError("launcher import verification failed: " + "; ".join(failures))


def verify_launcher(destination: Path, python: str, uv: str) -> None:
    """Validate uv's receipt and a fresh interpreter's import provenance."""
    receipt = _run_capture(
        [
            uv,
            "pip",
            "show",
            "--python",
            python,
            "omnigent",
            "omnigent-client",
            "omnigent-ui-sdk",
        ]
    )
    validate_editable_receipt(receipt, destination)
    import_code = (
        "import json, omnigent, omnigent_client, omnigent_ui_sdk; "
        "print(json.dumps({'omnigent': omnigent.__file__, "
        "'omnigent_client': omnigent_client.__file__, "
        "'omnigent_ui_sdk': omnigent_ui_sdk.__file__}))"
    )
    try:
        origins = json.loads(_run_capture([python, "-c", import_code]))
    except json.JSONDecodeError as exc:
        raise VerificationError("import provenance command did not return JSON") from exc
    if not isinstance(origins, dict):
        raise VerificationError("import provenance command returned a non-object JSON value")
    validate_import_origins(origins, destination)


def _http_json(url: str, timeout: float) -> tuple[int, dict[str, Any]]:
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationError(f"GET {url} failed: {exc}") from exc


def _local_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise VerificationError(
            "runtime URL must be a local http://127.0.0.1 or localhost endpoint"
        )
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise VerificationError("runtime URL must be an origin without a path, query, or fragment")
    return value.rstrip("/")


def verify_runtime(
    base_url: str,
    host_id: str,
    *,
    timeout: float = 5.0,
    get_json: Callable[[str, float], tuple[int, dict[str, Any]]] = _http_json,
) -> None:
    """Read only the health and host APIs; neither call changes service state."""
    if not host_id.strip():
        raise VerificationError("host id is required")
    if timeout <= 0:
        raise VerificationError("runtime timeout must be positive")
    base = _local_base_url(base_url)
    health_status, health = get_json(f"{base}/health", timeout)
    if health_status != 200 or health.get("status") != "ok":
        raise VerificationError("local health endpoint is not healthy")
    host_status, host = get_json(f"{base}/v1/hosts/{quote(host_id, safe='')}", timeout)
    if host_status != 200 or host.get("host_id") != host_id or host.get("status") != "online":
        raise VerificationError(f"host {host_id!r} is not online")


def _is_windows() -> bool:
    return sys.platform == "win32"


def _validate_unpacked_executable(executable: Path) -> Path:
    resolved = executable.expanduser().resolve()
    if resolved.name != "Omnigent.exe" or not resolved.is_file():
        raise VerificationError("desktop requires the exact built unpacked Omnigent.exe")
    if not (resolved.parent / "resources").is_dir():
        raise VerificationError(
            "desktop executable is not an unpacked Electron build (resources missing)"
        )
    return resolved


def _windows_process_tree(root_pid: int) -> set[int]:
    """Return the root and current descendants without invoking a shell."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        raise VerificationError("could not enumerate launched desktop process tree")

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        parents: dict[int, int] = {}
        if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            while True:
                parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
    finally:
        kernel32.CloseHandle(snapshot)

    process_ids = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid in parents.items():
            if parent_pid in process_ids and pid not in process_ids:
                process_ids.add(pid)
                changed = True
    return process_ids


def _visible_window_titles(process_ids: set[int]) -> list[tuple[int, str]]:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    windows: list[tuple[int, str]] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    @callback_type
    def visit(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if int(pid.value) not in process_ids:
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, length + 1)
        windows.append((int(pid.value), title.value))
        return True

    if not user32.EnumWindows(visit, 0):
        raise VerificationError("could not enumerate desktop windows")
    return windows


def _terminate_launched_tree(root_pid: int) -> None:
    """Ask Windows to terminate only the process tree rooted at our Popen PID."""
    completed = subprocess.run(
        ["taskkill", "/PID", str(root_pid), "/T", "/F"],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    if completed.returncode not in {0, 128}:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise VerificationError(f"could not clean up launched desktop process tree: {detail}")


def verify_desktop(
    executable: Path,
    *,
    timeout: float = 20.0,
    is_windows: Callable[[], bool] = _is_windows,
    popen: Callable[..., Any] = subprocess.Popen,
    process_tree: Callable[[int], set[int]] = _windows_process_tree,
    windows: Callable[[set[int]], list[tuple[int, str]]] = _visible_window_titles,
    terminate_tree: Callable[[int], None] = _terminate_launched_tree,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Launch exactly one unpacked executable and prove its visible window."""
    if not is_windows():
        raise VerificationError("desktop verification is Windows-only")
    if timeout <= 0:
        raise VerificationError("desktop timeout must be positive")
    target = _validate_unpacked_executable(executable)
    try:
        launched = popen([str(target)])
    except OSError as exc:
        raise VerificationError(f"could not launch Omnigent.exe: {exc}") from exc
    try:
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if launched.poll() is not None:
                raise VerificationError(
                    "launched Omnigent.exe exited before a visible window appeared"
                )
            process_ids = process_tree(int(launched.pid))
            for _, title in windows(process_ids):
                if "omnigent" in title.casefold():
                    return
            sleep(0.1)
        raise VerificationError("no visible Omnigent window owned by the launched process tree")
    finally:
        terminate_tree(int(launched.pid))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    relocation = commands.add_parser("relocation", help="compare preserved checkout bytes")
    relocation.add_argument("--original", required=True, type=Path)
    relocation.add_argument("--destination", required=True, type=Path)
    relocation.add_argument("--expected-count", type=int, default=88)

    launcher = commands.add_parser("launcher", help="validate uv editable-install provenance")
    launcher.add_argument("--destination", required=True, type=Path)
    launcher.add_argument("--python", default=sys.executable)
    launcher.add_argument("--uv", default="uv")

    runtime = commands.add_parser("runtime", help="validate local server and host liveness")
    runtime.add_argument("--host-id", required=True)
    runtime.add_argument("--base-url", default="http://127.0.0.1:6767")
    runtime.add_argument("--timeout", type=float, default=5.0)

    desktop = commands.add_parser(
        "desktop", help="launch and observe unpacked Electron executable"
    )
    desktop.add_argument("--executable", required=True, type=Path)
    desktop.add_argument("--timeout", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "relocation":
            verify_relocation(args.original, args.destination, expected_count=args.expected_count)
            print(RELOCATION_VERIFIED)
        elif args.command == "launcher":
            verify_launcher(args.destination, args.python, args.uv)
            print(ONEDRIVE_LAUNCHER_VERIFIED)
        elif args.command == "runtime":
            verify_runtime(args.base_url, args.host_id, timeout=args.timeout)
            print(WINDOWS_RUNTIME_VERIFIED)
        elif args.command == "desktop":
            verify_desktop(args.executable, timeout=args.timeout)
            print(WINDOWS_DESKTOP_MVP_VERIFIED)
        else:  # pragma: no cover - argparse makes this unreachable.
            raise VerificationError(f"unknown command: {args.command}")
    except VerificationError as exc:
        print(f"VERIFICATION_FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
