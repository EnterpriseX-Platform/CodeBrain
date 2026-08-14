from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from codebrain import cli
from codebrain.store import BRAIN_DIR


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def make_repo(tmp: str) -> Path:
    root = Path(tmp)
    (root / "src").mkdir()
    (root / "src" / "api.py").write_text("def charge():\n    return 1\n", encoding="utf-8")
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    return root


class TestTerminalDegradation(unittest.TestCase):
    """A hook that raises breaks the user's session. Rendering degrades instead."""

    def test_ascii_fallback_when_the_console_cannot_encode(self):
        class Cp1252Stream:
            encoding = "cp1252"

        real = sys.stdout
        sys.stdout = Cp1252Stream()  # type: ignore[assignment]
        try:
            self.assertFalse(cli._unicode_ok())
        finally:
            sys.stdout = real

    def test_unicode_is_used_when_available(self):
        class Utf8Stream:
            encoding = "utf-8"

        real = sys.stdout
        sys.stdout = Utf8Stream()  # type: ignore[assignment]
        try:
            self.assertTrue(cli._unicode_ok())
        finally:
            sys.stdout = real

    def test_bar_never_exceeds_its_width(self):
        for count in (0, 1, 7, 999):
            self.assertEqual(len(cli._bar(count, 999, width=22)), 22)

    def test_bar_shows_something_for_a_nonzero_count(self):
        self.assertIn(cli.FULL, cli._bar(1, 10_000))


class TestCliEndToEnd(unittest.TestCase):
    def test_build_status_validate_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            brain_dir = str(root / BRAIN_DIR)

            code, out, _ = run("build", str(root))
            self.assertEqual(code, 0)
            self.assertIn("Brain built", out)

            code, out, _ = run("status", brain_dir, "--json")
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertGreater(payload["stats"]["total"], 0)
            self.assertIn("census", payload["manifest"]["providers"])

            code, _, _ = run("validate", brain_dir)
            self.assertEqual(code, 0)

            code, out, _ = run("diff", brain_dir, brain_dir)
            self.assertEqual(code, 0)
            self.assertIn("No change.", out)

    def test_status_renders_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            run("build", str(root))
            code, out, _ = run("status", str(root / BRAIN_DIR))
        self.assertEqual(code, 0)
        self.assertIn("L0 corpus", out)

    def test_drift_gate_exits_nonzero_on_substantive_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            first = str(root / "brain-a")
            run("build", str(root), "--out", first)

            (root / "src" / "extra.py").write_text("x = 1\n", encoding="utf-8")
            second = str(root / "brain-b")
            run("build", str(root), "--out", second)

            code, out, _ = run("diff", first, second, "--check")

        self.assertEqual(code, 1)
        self.assertIn("extra.py", out)

    def test_diff_without_check_reports_but_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            first = str(root / "brain-a")
            run("build", str(root), "--out", first)
            (root / "src" / "extra.py").write_text("x = 1\n", encoding="utf-8")
            second = str(root / "brain-b")
            run("build", str(root), "--out", second)
            code, _, _ = run("diff", first, second)
        self.assertEqual(code, 0)

    def test_only_restricts_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            run("build", str(root), "--only", "census")
            code, out, _ = run("status", str(root / BRAIN_DIR), "--json")
        self.assertEqual(json.loads(out)["manifest"]["providers"], ["census"])

    def test_unknown_provider_is_an_error_not_a_silent_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = run("build", tmp, "--only", "nonexistent")
        self.assertEqual(code, 2)
        self.assertIn("no such provider", err)

    def test_status_without_a_brain_explains_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = run("status", str(Path(tmp) / "nope"))
        self.assertEqual(code, 2)
        self.assertIn("codebrain build", err)

    def test_build_on_a_missing_directory(self):
        code, _, err = run("build", "/definitely/not/here")
        self.assertEqual(code, 2)
        self.assertIn("not a directory", err)


class TestBrainPathSpellings(unittest.TestCase):
    """Half the commands took the Brain path positionally and half as --brain.
    Both spellings work now, everywhere."""

    def test_status_accepts_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            run("build", str(root))
            brain = str(root / BRAIN_DIR)
            positional, _, _ = run("status", brain, "--json")
            keyword, _, _ = run("status", "--brain", brain, "--json")
        self.assertEqual((positional, keyword), (0, 0))

    def test_validate_and_atlas_accept_keyword_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            run("build", str(root))
            brain = str(root / BRAIN_DIR)
            self.assertEqual(run("validate", "--brain", brain)[0], 0)
            self.assertEqual(run("atlas", "--brain", brain, "--out", "-")[0], 0)


class TestVerifySyncDrift(unittest.TestCase):
    # sync and drift are defined relative to HEAD, so these fixtures must be
    # real git repositories or the commands can only ever say "cannot tell".
    TASK = "change charge_endpoint in payments.py"

    def _repo(self, test_recipe: str | None = None) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "payments.py").write_text(
            "def charge_endpoint():\n    return 1\n", encoding="utf-8")
        recipe = test_recipe or f'"{sys.executable}" -c "pass"'
        (root / "Makefile").write_text(f"test:\n\t{recipe}\n", encoding="utf-8")
        for args in (("init", "-b", "main"),
                     ("config", "user.email", "t@example.com"),
                     ("config", "user.name", "T"),
                     ("add", "-A"),
                     ("-c", "commit.gpgsign=false", "commit", "-m", "initial")):
            subprocess.run(("git", *args), cwd=str(root), capture_output=True,
                           timeout=30)
        run("build", str(root))
        return root

    def test_verify_is_a_dry_run_by_default(self):
        root = self._repo()
        code, out, _ = run("verify", "--brain", str(root / BRAIN_DIR),
                           "--root", str(root))
        self.assertEqual(code, 0)
        self.assertIn("Dry run", out)
        self.assertIn("--yes", out)

    def test_verify_yes_promotes_and_persists(self):
        root = self._repo()
        brain_dir = str(root / BRAIN_DIR)
        code, out, _ = run("verify", "--brain", brain_dir, "--root", str(root), "--yes")
        self.assertEqual(code, 0)
        self.assertTrue("promoted" in out or "refuted" in out)
        # Reload from disk: the verdict must have been saved, not just held.
        _, packed, _ = run("pack", self.TASK, "--brain", brain_dir, "--root", str(root))
        self.assertNotIn("never executed", packed)
        self.assertTrue("OBSERVED" in packed or "REFUTED" in packed)

    def test_verify_check_fails_on_a_refuted_claim(self):
        root = self._repo(f'"{sys.executable}" -c "import sys; sys.exit(1)"')
        code, out, _ = run("verify", "--brain", str(root / BRAIN_DIR),
                           "--root", str(root), "--yes", "--check")
        self.assertEqual(code, 1)
        self.assertIn("refuted", out)

    def test_sync_reports_up_to_date_when_nothing_moved(self):
        root = self._repo()
        code, out, _ = run("sync", "--brain", str(root / BRAIN_DIR), "--root", str(root))
        self.assertEqual(code, 0)
        self.assertIn("Up to date", out)

    def test_sync_carries_a_verification_forward(self):
        root = self._repo()
        brain_dir = str(root / BRAIN_DIR)
        run("verify", "--brain", brain_dir, "--root", str(root), "--yes")
        code, out, _ = run("sync", "--brain", brain_dir, "--root", str(root), "--force")
        self.assertEqual(code, 0)
        self.assertIn("carried forward", out)
        _, packed, _ = run("pack", self.TASK, "--brain", brain_dir,
                           "--root", str(root))
        self.assertNotIn("never executed", packed)

    def test_drift_is_clean_right_after_a_build(self):
        root = self._repo()
        code, out, _ = run("drift", "--brain", str(root / BRAIN_DIR),
                           "--root", str(root), "--check")
        self.assertEqual(code, 0)
        self.assertIn("No drift", out)

    def test_drift_check_fails_once_the_code_moves(self):
        root = self._repo()
        (root / "a.py").write_text("def f():\n    return 1\n\ndef g():\n    return 2\n",
                                   encoding="utf-8")
        code, out, _ = run("drift", "--brain", str(root / BRAIN_DIR),
                           "--root", str(root), "--check")
        self.assertEqual(code, 1)
        self.assertIn("DRIFT", out)
        self.assertIn("codebrain sync", out)

    def test_drift_without_check_reports_but_succeeds(self):
        root = self._repo()
        (root / "b.py").write_text("def h():\n    return 3\n", encoding="utf-8")
        code, _, _ = run("drift", "--brain", str(root / BRAIN_DIR), "--root", str(root))
        self.assertEqual(code, 0)


class TestHooksFailOpen(unittest.TestCase):
    """A hook that errors breaks the user's session. Every one of these must
    exit 0 and print nothing when the Brain is unavailable."""

    def _missing(self) -> str:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return str(Path(tmp.name) / "no-brain-here")

    def test_pack_without_a_brain(self):
        code, out, err = run("pack", "do a thing", "--brain", self._missing())
        self.assertEqual((code, out, err), (0, "", ""))

    def test_brief_without_a_brain(self):
        code, out, err = run("brief", "--brain", self._missing())
        self.assertEqual((code, out, err), (0, "", ""))

    def test_guard_without_a_brain(self):
        code, out, err = run("guard", "--brain", self._missing(), "--path", "a.py")
        self.assertEqual((code, out, err), (0, "", ""))

    def test_pack_with_an_empty_task(self):
        code, out, _ = run("pack", "", "--brain", self._missing())
        self.assertEqual((code, out), (0, ""))

    def test_guard_with_no_path(self):
        code, out, _ = run("guard", "--brain", self._missing())
        self.assertEqual((code, out), (0, ""))

    def test_touch_without_a_brain_directory(self):
        code, _, _ = run("touch", "--brain", self._missing(), "--path", "a.py")
        self.assertEqual(code, 0)

    def test_corrupt_brain_does_not_break_the_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain_dir = Path(tmp) / ".brain"
            (brain_dir / "layers").mkdir(parents=True)
            (brain_dir / "manifest.json").write_text("{not json", encoding="utf-8")
            code, out, err = run("pack", "anything", "--brain", str(brain_dir))
        self.assertEqual((code, out, err), (0, "", ""))


class TestHooksWithABrain(unittest.TestCase):
    def _repo(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "payments").mkdir()
        (root / "payments" / "api.py").write_text(
            "def charge_endpoint():\n    return 1\n", encoding="utf-8")
        (root / "CODEOWNERS").write_text("/payments/ @risk-eng\n", encoding="utf-8")
        (root / "Makefile").write_text("test:\n\tpytest -q\n", encoding="utf-8")
        run("build", str(root))
        return root

    def test_pack_produces_a_pack(self):
        root = self._repo()
        code, out, _ = run("pack", "fix charge_endpoint", "--brain",
                           str(root / BRAIN_DIR), "--root", str(root))
        self.assertEqual(code, 0)
        self.assertIn("CONTEXT PACK", out)
        self.assertIn("charge_endpoint", out)

    def test_brief_mentions_the_test_command(self):
        root = self._repo()
        code, out, _ = run("brief", "--brain", str(root / BRAIN_DIR))
        self.assertEqual(code, 0)
        self.assertIn("make test", out)

    def test_guard_warns_on_a_codeowners_path(self):
        root = self._repo()
        code, out, _ = run("guard", "--brain", str(root / BRAIN_DIR),
                           "--root", str(root), "--path", "payments/api.py")
        self.assertEqual(code, 0)
        payload = json.loads(out)["hookSpecificOutput"]
        self.assertEqual(payload["hookEventName"], "PreToolUse")
        self.assertIn("@risk-eng", payload["permissionDecisionReason"])

    def test_guard_allows_by_default_and_asks_only_when_told_to(self):
        # Blocking an edit on inference alone gets the hook uninstalled.
        root = self._repo()
        _, allowed, _ = run("guard", "--brain", str(root / BRAIN_DIR),
                            "--root", str(root), "--path", "payments/api.py")
        _, asked, _ = run("guard", "--brain", str(root / BRAIN_DIR), "--root", str(root),
                          "--path", "payments/api.py", "--deny-guarded")
        self.assertEqual(json.loads(allowed)["hookSpecificOutput"]
                         ["permissionDecision"], "allow")
        self.assertEqual(json.loads(asked)["hookSpecificOutput"]
                         ["permissionDecision"], "ask")

    def test_guard_is_silent_on_an_unconstrained_path(self):
        root = self._repo()
        code, out, _ = run("guard", "--brain", str(root / BRAIN_DIR),
                           "--root", str(root), "--path", "Makefile")
        self.assertEqual((code, out), (0, ""))

    def test_touch_marks_records_stale_for_later_packs(self):
        root = self._repo()
        brain_dir = str(root / BRAIN_DIR)
        run("touch", "--brain", brain_dir, "--root", str(root),
            "--path", "payments/api.py")
        code, out, _ = run("pack", "fix charge_endpoint", "--brain", brain_dir,
                           "--root", str(root))
        self.assertEqual(code, 0)
        self.assertIn("stale", out)

    def test_build_clears_the_touched_list(self):
        root = self._repo()
        brain_dir = str(root / BRAIN_DIR)
        run("touch", "--brain", brain_dir, "--root", str(root),
            "--path", "payments/api.py")
        run("build", str(root))
        _, out, _ = run("pack", "fix charge_endpoint", "--brain", brain_dir,
                        "--root", str(root))
        self.assertNotIn("stale", out)


if __name__ == "__main__":
    unittest.main()
