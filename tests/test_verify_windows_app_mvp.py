"""Deterministic unit tests for the fail-closed Windows MVP verifier."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path, PureWindowsPath
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dev import verify_windows_app_mvp as verify


class GitOutputTests(unittest.TestCase):
    def test_git_output_uses_forward_slash_windows_checkout_path(self) -> None:
        root = PureWindowsPath(r"C:\Users\ricar\OneDrive\Documents\Omnigent")
        expected_root = "C:/Users/ricar/OneDrive/Documents/Omnigent"

        with patch.object(
            verify.subprocess,
            "run",
            return_value=CompletedProcess([], 0, stdout=b"", stderr=b""),
        ) as run:
            verify._git_output(root, ["status", "--short"])

        self.assertEqual(
            run.call_args.args[0],
            [
                "git",
                "-c",
                f"safe.directory={expected_root}",
                "-C",
                expected_root,
                "status",
                "--short",
            ],
        )


class RelocationTests(unittest.TestCase):
    def _checkout(self, root: Path, files: dict[str, bytes]) -> None:
        root.mkdir()
        for relative, content in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def _git_outputs(
        self,
        original: Path,
        destination: Path,
        original_status: bytes,
        destination_status: bytes,
        *,
        original_branch: str = "main",
        destination_branch: str = "main",
        original_head: str = "abc",
        destination_head: str = "abc",
    ):
        def output(root: Path, arguments: list[str]) -> bytes:
            if arguments == ["branch", "--show-current"]:
                return (original_branch if root == original else destination_branch).encode()
            if arguments == ["rev-parse", "HEAD"]:
                return (original_head if root == original else destination_head).encode()
            self.assertEqual(
                arguments, ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
            )
            return original_status if root == original else destination_status

        return output

    def test_relocation_allows_declared_continuation_file(self) -> None:
        with TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            destination = Path(temp) / "destination"
            self._checkout(
                original, {"kept.txt": b"same", "dev/verify_windows_app_mvp.py": b"old"}
            )
            self._checkout(
                destination, {"kept.txt": b"same", "dev/verify_windows_app_mvp.py": b"new"}
            )
            output = self._git_outputs(
                original,
                destination,
                b" M kept.txt\0 M dev/verify_windows_app_mvp.py\0",
                b" M kept.txt\0 M dev/verify_windows_app_mvp.py\0",
            )
            with patch.object(verify, "_git_output", output):
                verify.verify_relocation(original, destination, expected_count=2)

    def test_relocation_allows_declared_destination_only_continuation_path(self) -> None:
        with TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            destination = Path(temp) / "destination"
            continuation = "web/electron/src/server_manager.js"
            self._checkout(original, {"kept.txt": b"same"})
            self._checkout(destination, {"kept.txt": b"same", continuation: b"new"})
            output = self._git_outputs(
                original,
                destination,
                b" M kept.txt\0",
                f" M kept.txt\0?? {continuation}\0".encode(),
            )
            with patch.object(verify, "_git_output", output):
                verify.verify_relocation(original, destination, expected_count=1)

    def test_relocation_rejects_undeclared_destination_only_path(self) -> None:
        with TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            destination = Path(temp) / "destination"
            self._checkout(original, {"kept.txt": b"same"})
            self._checkout(destination, {"kept.txt": b"same", "unexpected.txt": b"new"})
            output = self._git_outputs(
                original,
                destination,
                b" M kept.txt\0",
                b" M kept.txt\0?? unexpected.txt\0",
            )
            with (
                patch.object(verify, "_git_output", output),
                self.assertRaisesRegex(
                    verify.VerificationError, "unexpected destination dirty path: unexpected.txt"
                ),
            ):
                verify.verify_relocation(original, destination, expected_count=1)

    def test_relocation_rejects_baseline_byte_mismatch_not_in_allowlist(self) -> None:
        with TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            destination = Path(temp) / "destination"
            self._checkout(original, {"kept.txt": b"old"})
            self._checkout(destination, {"kept.txt": b"new"})
            output = self._git_outputs(original, destination, b" M kept.txt\0", b" M kept.txt\0")
            with (
                patch.object(verify, "_git_output", output),
                self.assertRaisesRegex(verify.VerificationError, "byte mismatch: kept.txt"),
            ):
                verify.verify_relocation(original, destination, expected_count=1)

    def test_relocation_rejects_count_branch_and_head_failures(self) -> None:
        with TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            destination = Path(temp) / "destination"
            self._checkout(original, {"kept.txt": b"same"})
            self._checkout(destination, {"kept.txt": b"same"})
            output = self._git_outputs(
                original,
                destination,
                b" M kept.txt\0",
                b" M kept.txt\0",
                destination_branch="other",
                destination_head="def",
            )
            with (
                patch.object(verify, "_git_output", output),
                self.assertRaisesRegex(
                    verify.VerificationError,
                    "branch mismatch.*HEAD mismatch.*baseline scope count",
                ),
            ):
                verify.verify_relocation(original, destination, expected_count=2)

    def test_relocation_rejects_missing_mismatched_and_unexpected_paths(self) -> None:
        with TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            destination = Path(temp) / "destination"
            self._checkout(
                original, {"same.txt": b"same", "different.txt": b"old", "missing.txt": b"gone"}
            )
            self._checkout(
                destination, {"same.txt": b"same", "different.txt": b"new", "extra.txt": b"extra"}
            )
            output = self._git_outputs(
                original,
                destination,
                b" M same.txt\0 M different.txt\0 M missing.txt\0",
                b" M same.txt\0 M different.txt\0?? extra.txt\0",
            )
            with (
                patch.object(verify, "_git_output", output),
                self.assertRaisesRegex(
                    verify.VerificationError,
                    r"unexpected destination dirty path: extra\.txt; "
                    r"missing destination dirty path: missing\.txt; "
                    r"byte mismatch: different\.txt; missing baseline file: missing\.txt",
                ),
            ):
                verify.verify_relocation(original, destination, expected_count=3)

    def test_relocation_parses_renames_and_never_scans_unreported_generated_data(self) -> None:
        with TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            destination = Path(temp) / "destination"
            self._checkout(original, {"renamed.txt": b"same", ".uv-cache/huge.bin": b"old"})
            self._checkout(
                destination, {"renamed.txt": b"same", ".uv-cache/huge.bin": b"different"}
            )
            output = self._git_outputs(
                original,
                destination,
                b"R  renamed.txt\0old-name.txt\0",
                b"R  renamed.txt\0old-name.txt\0",
            )
            with patch.object(verify, "_git_output", output):
                verify.verify_relocation(original, destination, expected_count=1)

    def test_relocation_rejects_staged_vs_unstaged_state(self) -> None:
        with TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            destination = Path(temp) / "destination"
            self._checkout(original, {"kept.txt": b"same"})
            self._checkout(destination, {"kept.txt": b"same"})
            output = self._git_outputs(original, destination, b"M  kept.txt\0", b" M kept.txt\0")
            with (
                patch.object(verify, "_git_output", output),
                self.assertRaisesRegex(verify.VerificationError, "Git state mismatch: kept.txt"),
            ):
                verify.verify_relocation(original, destination, expected_count=1)

    def test_relocation_rejects_rename_source_mismatch(self) -> None:
        with TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            destination = Path(temp) / "destination"
            self._checkout(original, {"renamed.txt": b"same"})
            self._checkout(destination, {"renamed.txt": b"same"})
            output = self._git_outputs(
                original,
                destination,
                b"R  renamed.txt\0old-name.txt\0",
                b"R  renamed.txt\0other-old-name.txt\0",
            )
            with (
                patch.object(verify, "_git_output", output),
                self.assertRaisesRegex(
                    verify.VerificationError, "Git state mismatch: renamed.txt"
                ),
            ):
                verify.verify_relocation(original, destination, expected_count=1)

    def test_relocation_rejects_file_type_mismatch(self) -> None:
        with TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            destination = Path(temp) / "destination"
            self._checkout(original, {"kept.txt": b"same"})
            self._checkout(destination, {})
            (destination / "kept.txt").mkdir()
            output = self._git_outputs(original, destination, b" M kept.txt\0", b" M kept.txt\0")
            with (
                patch.object(verify, "_git_output", output),
                self.assertRaisesRegex(
                    verify.VerificationError, "file type or reparse mismatch: kept.txt"
                ),
            ):
                verify.verify_relocation(original, destination, expected_count=1)

    def test_relocation_rejects_unsafe_porcelain_paths(self) -> None:
        with self.assertRaisesRegex(verify.VerificationError, "unsafe Git status path"):
            verify._parse_porcelain_paths(b"?? ../escape.txt\0")


class LauncherTests(unittest.TestCase):
    def _destination(self, temp: str) -> Path:
        root = Path(temp) / "OneDriveCopy"
        (root / "sdks" / "python-client").mkdir(parents=True)
        (root / "sdks" / "ui").mkdir(parents=True)
        return root

    def test_receipt_requires_each_exact_editable_target(self) -> None:
        with TemporaryDirectory() as temp:
            destination = self._destination(temp)
            receipt = "\n\n".join(
                (
                    f"Name: omnigent\nEditable project location: {destination}",
                    "Name: omnigent-client\nEditable project location: "
                    f"{destination / 'sdks/python-client'}",
                    f"Name: omnigent-ui-sdk\nEditable project location: {destination / 'sdks/ui'}",
                )
            )
            verify.validate_editable_receipt(receipt, destination)
            with self.assertRaisesRegex(verify.VerificationError, "missing editable receipt"):
                verify.validate_editable_receipt(
                    receipt.replace("omnigent-ui-sdk", "other"), destination
                )

    def test_launcher_checks_uv_receipt_and_import_provenance(self) -> None:
        with TemporaryDirectory() as temp:
            destination = self._destination(temp)
            receipt = "\n\n".join(
                (
                    f"Name: omnigent\nEditable project location: {destination}",
                    "Name: omnigent-client\nEditable project location: "
                    f"{destination / 'sdks/python-client'}",
                    f"Name: omnigent-ui-sdk\nEditable project location: {destination / 'sdks/ui'}",
                )
            )
            origins = {
                "omnigent": str(destination / "omnigent" / "__init__.py"),
                "omnigent_client": str(
                    destination / "sdks/python-client/omnigent_client/__init__.py"
                ),
                "omnigent_ui_sdk": str(destination / "sdks/ui/omnigent_ui_sdk/__init__.py"),
            }
            calls: list[list[str]] = []

            def run_capture(argv: list[str]) -> str:
                calls.append(argv)
                return (
                    receipt if argv[1:3] == ["pip", "show"] else __import__("json").dumps(origins)
                )

            with patch.object(verify, "_run_capture", run_capture):
                verify.verify_launcher(destination, "python.exe", "uv.exe")
            self.assertEqual(calls[0][:3], ["uv.exe", "pip", "show"])
            self.assertEqual(calls[1][:2], ["python.exe", "-c"])


class RuntimeTests(unittest.TestCase):
    def test_runtime_requires_health_and_exact_online_host(self) -> None:
        calls: list[str] = []

        def get_json(url: str, timeout: float):
            calls.append(url)
            if url.endswith("/health"):
                return 200, {"status": "ok"}
            return 200, {"host_id": "host local/1", "status": "online"}

        verify.verify_runtime("http://127.0.0.1:6767", "host local/1", get_json=get_json)
        self.assertEqual(calls[-1], "http://127.0.0.1:6767/v1/hosts/host%20local%2F1")
        with self.assertRaisesRegex(verify.VerificationError, "not online"):
            verify.verify_runtime(
                "http://127.0.0.1:6767",
                "host",
                get_json=lambda *_: (200, {"status": "ok"}),
            )


class DesktopTests(unittest.TestCase):
    def _executable(self, temp: str) -> Path:
        executable = Path(temp) / "win-unpacked" / "Omnigent.exe"
        executable.parent.mkdir(parents=True)
        (executable.parent / "resources").mkdir()
        executable.write_bytes(b"fixture")
        return executable

    def test_desktop_rejects_non_windows_without_launching(self) -> None:
        with TemporaryDirectory() as temp:
            executable = self._executable(temp)
            with self.assertRaisesRegex(verify.VerificationError, "Windows-only"):
                verify.verify_desktop(executable, is_windows=lambda: False)

    def test_desktop_observes_owned_visible_window_and_cleans_tree(self) -> None:
        class Process:
            pid = 9123

            @staticmethod
            def poll() -> None:
                return None

        with TemporaryDirectory() as temp:
            executable = self._executable(temp)
            launched: list[list[str]] = []
            cleaned: list[int] = []
            verify.verify_desktop(
                executable,
                is_windows=lambda: True,
                popen=lambda argv: launched.append(argv) or Process(),
                process_tree=lambda pid: {pid, 9124},
                windows=lambda pids: [(9124, "Omnigent - local")],
                terminate_tree=cleaned.append,
            )
            self.assertEqual(launched, [[str(executable.resolve())]])
            self.assertEqual(cleaned, [9123])


class MainTests(unittest.TestCase):
    def test_main_prints_a_success_token_only_after_check_returns(self) -> None:
        output = io.StringIO()
        with patch.object(verify, "verify_runtime") as runtime, redirect_stdout(output):
            self.assertEqual(verify.main(["runtime", "--host-id", "host-1"]), 0)
        runtime.assert_called_once()
        self.assertEqual(output.getvalue().strip(), verify.WINDOWS_RUNTIME_VERIFIED)


if __name__ == "__main__":
    unittest.main()
