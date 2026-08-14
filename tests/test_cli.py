from __future__ import annotations

import io
import json
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


if __name__ == "__main__":
    unittest.main()
