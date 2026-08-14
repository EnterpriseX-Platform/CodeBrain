from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Method
from codebrain.model import REPO, Layer
from codebrain.providers import BuildContext, build
from codebrain.extractors.census import CensusProvider
from codebrain.extractors.constraints import ConstraintsProvider, matches
from codebrain.extractors.operations import OperationsProvider


class TestPatternMatching(unittest.TestCase):
    def test_directory_rule(self):
        self.assertTrue(matches("/payments/", "payments/api.py"))
        self.assertFalse(matches("/payments/", "billing/api.py"))

    def test_glob_on_bare_name(self):
        self.assertTrue(matches("*.md", "docs/readme.md"))
        self.assertFalse(matches("*.md", "docs/readme.rst"))

    def test_path_glob(self):
        self.assertTrue(matches("src/*", "src/a.py"))

    def test_empty_pattern_matches_nothing(self):
        self.assertFalse(matches("", "a.py"))


class TestConstraintsProvider(unittest.TestCase):
    def _build(self, files: dict[str, str]):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return build(BuildContext(root=root),
                     [CensusProvider(), OperationsProvider(),
                      ConstraintsProvider()]).brain

    def test_codeowners_becomes_a_review_requirement(self):
        brain = self._build({
            "CODEOWNERS": "/payments/ @risk-eng\n",
            "payments/api.py": "x = 1\n",
            "docs/readme.md": "# hi\n",
        })
        guarded = brain.fact("L0:file:payments/api.py", "requires_review", Layer.L6)
        self.assertIsNotNone(guarded)
        self.assertEqual(guarded.value["owners"], ["@risk-eng"])
        self.assertIsNone(brain.fact("L0:file:docs/readme.md", "requires_review",
                                     Layer.L6))

    def test_later_codeowners_rules_win(self):
        brain = self._build({
            "CODEOWNERS": "/src/ @a\n/src/pay/ @b\n",
            "src/pay/x.py": "x = 1\n",
        })
        found = brain.fact("L0:file:src/pay/x.py", "requires_review", Layer.L6)
        self.assertEqual(found.value["owners"], ["@b"])

    def test_review_requirements_are_derived_and_say_the_matching_is_approximate(self):
        brain = self._build({"CODEOWNERS": "/p/ @a\n", "p/x.py": "x = 1\n"})
        found = brain.fact("L0:file:p/x.py", "requires_review", Layer.L6)
        self.assertIs(found.env.method, Method.DERIVED)
        self.assertIn("approximate", found.env.note)

    def test_guarded_count_is_reported(self):
        brain = self._build({"CODEOWNERS": "* @all\n", "a.py": "x = 1\n"})
        self.assertGreaterEqual(brain.fact(REPO, "guarded_file_count", Layer.L6).value, 1)

    def test_no_codeowners_means_no_review_constraints(self):
        brain = self._build({"a.py": "x = 1\n"})
        self.assertIsNone(brain.fact(REPO, "guarded_file_count", Layer.L6))

    def test_provider_declines_without_a_brain_to_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(root=Path(tmp))
            self.assertFalse(ConstraintsProvider().applies(ctx))

    def test_constraints_are_derived_from_evidence_not_invented(self):
        brain = self._build({"CODEOWNERS": "/p/ @a\n", "p/x.py": "x = 1\n"})
        found = brain.fact("L0:file:p/x.py", "requires_review", Layer.L6)
        self.assertTrue(found.env.evidence)
        self.assertEqual(found.env.source, "constraints")


if __name__ == "__main__":
    unittest.main()
