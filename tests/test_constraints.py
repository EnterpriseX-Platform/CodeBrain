from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Method
from codebrain.model import REPO, Layer
from codebrain.providers import BuildContext, build
from codebrain.extractors.behavior import BehaviorProvider
from codebrain.extractors.census import CensusProvider
from codebrain.extractors.constraints import ConstraintsProvider, matches
from codebrain.extractors.operations import OperationsProvider
from codebrain.extractors.structure_py import PythonStructureProvider


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


class TestPolicyZones(unittest.TestCase):
    """The one place a human states a constraint outright. Everything else in
    L6 is inferred, so only this may stop an agent on its own."""

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

    POLICY = (
        '[[zone]]\n'
        'name = "pci"\n'
        'paths = ["payments/"]\n'
        'reason = "PCI scope — cardholder data"\n'
        'requires = ["@risk-eng"]\n'
        'block_agents = true\n'
    )

    def test_declared_zones_become_constraints(self):
        brain = self._build({".codebrain.toml": self.POLICY,
                             "payments/settle.py": "x = 1\n",
                             "docs/readme.md": "# hi\n"})
        found = brain.fact("L0:file:payments/settle.py", "policy_zone", Layer.L6)
        self.assertIsNotNone(found)
        self.assertEqual(found.value["zone"], "pci")
        self.assertTrue(found.value["block_agents"])
        self.assertIsNone(brain.fact("L0:file:docs/readme.md", "policy_zone", Layer.L6))

    def test_a_declared_zone_is_asserted_not_derived(self):
        # A human said so. It must outrank anything the machinery infers.
        brain = self._build({".codebrain.toml": self.POLICY,
                             "payments/settle.py": "x = 1\n"})
        found = brain.fact("L0:file:payments/settle.py", "policy_zone", Layer.L6)
        self.assertIs(found.env.method, Method.ASSERTED)

    def test_the_reason_travels_with_the_constraint(self):
        brain = self._build({".codebrain.toml": self.POLICY,
                             "payments/settle.py": "x = 1\n"})
        value = brain.fact("L0:file:payments/settle.py", "policy_zone", Layer.L6).value
        self.assertIn("PCI", value["reason"])
        self.assertEqual(value["requires"], ["@risk-eng"])

    def test_table_form_is_accepted(self):
        brain = self._build({
            ".codebrain.toml": '[zone.secrets]\npaths = ["config/"]\n',
            "config/keys.py": "x = 1\n"})
        found = brain.fact("L0:file:config/keys.py", "policy_zone", Layer.L6)
        self.assertEqual(found.value["zone"], "secrets")

    def test_a_malformed_policy_file_is_ignored_not_fatal(self):
        brain = self._build({".codebrain.toml": "this is not toml [[[",
                             "payments/settle.py": "x = 1\n"})
        self.assertIsNone(brain.fact(REPO, "policy_zone_count", Layer.L6))
        self.assertIsNotNone(brain.get("L0:file:payments/settle.py"))

    def test_no_policy_file_means_no_zones(self):
        brain = self._build({"payments/settle.py": "x = 1\n"})
        self.assertIsNone(brain.fact("L0:file:payments/settle.py", "policy_zone",
                                     Layer.L6))


class TestPublicContracts(unittest.TestCase):
    def _build(self, files: dict[str, str]):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return build(BuildContext(root=root),
                     [CensusProvider(), PythonStructureProvider(),
                      BehaviorProvider(), ConstraintsProvider()]).brain

    def _shared(self) -> dict[str, str]:
        return {
            "core/__init__.py": "",
            "core/api.py": "def charge():\n    return 1\n",
            "a/__init__.py": "",
            "a/one.py": "from core.api import charge\ndef go():\n    charge()\n",
            "b/__init__.py": "",
            "b/two.py": "from core.api import charge\ndef run():\n    charge()\n",
        }

    def test_a_symbol_two_modules_call_is_a_contract(self):
        brain = self._build(self._shared())
        found = brain.fact("L1:symbol:core/api.py#charge", "public_contract", Layer.L6)
        self.assertIsNotNone(found)
        self.assertEqual(found.value["consumer_count"], 2)

    def test_one_caller_is_coupling_not_a_contract(self):
        files = self._shared()
        del files["b/two.py"]
        brain = self._build(files)
        self.assertIsNone(brain.fact("L1:symbol:core/api.py#charge",
                                     "public_contract", Layer.L6))

    def test_test_callers_do_not_create_a_contract(self):
        files = self._shared()
        del files["b/two.py"]
        files["tests/test_api.py"] = ("from core.api import charge\n"
                                      "def test_charge():\n    charge()\n")
        brain = self._build(files)
        self.assertIsNone(brain.fact("L1:symbol:core/api.py#charge",
                                     "public_contract", Layer.L6))

    def test_an_http_route_is_always_a_contract(self):
        brain = self._build({"api.py": "@app.get('/v1/charges')\n"
                                       "def charge():\n    return 1\n"})
        found = brain.fact("L2:route:GET /v1/charges", "public_contract", Layer.L6)
        self.assertIsNotNone(found)
        self.assertIn("callable by anyone", found.value["reason"])


if __name__ == "__main__":
    unittest.main()
