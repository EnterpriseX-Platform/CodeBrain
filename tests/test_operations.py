from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Method
from codebrain.model import REPO
from codebrain.providers import BuildContext, build
from codebrain.extractors.operations import OperationsProvider


class TestOperations(unittest.TestCase):
    def _build(self, files: dict[str, str]):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return build(BuildContext(root=root), [OperationsProvider()]).brain

    # -- package.json ------------------------------------------------------

    def test_npm_scripts_become_commands(self):
        brain = self._build({"package.json": json.dumps({
            "name": "demo", "scripts": {"test": "jest", "build": "tsc"},
            "dependencies": {"react": "^18"},
        })})
        node = brain.get("L5:command:package.json:test")
        self.assertIsNotNone(node)
        self.assertEqual(node.attrs["command"], "jest")
        self.assertEqual(brain.fact(REPO, "test_command").value, "npm run test")

    def test_malformed_package_json_does_not_crash(self):
        brain = self._build({"package.json": "{not json"})
        self.assertIsNone(brain.fact(REPO, "node_package"))

    # -- pyproject ---------------------------------------------------------

    def test_pyproject_metadata(self):
        brain = self._build({"pyproject.toml":
                             '[project]\nname = "demo"\nversion = "1.0"\n'
                             'dependencies = ["httpx"]\n'})
        value = brain.fact(REPO, "python_package").value
        self.assertEqual(value["name"], "demo")
        self.assertEqual(value["dependencies"], 1)

    def test_pytest_dependency_suggests_the_test_command(self):
        brain = self._build({"pyproject.toml":
                             '[project]\nname = "d"\ndependencies = ["pytest"]\n'})
        self.assertEqual(brain.fact(REPO, "test_command").value, "pytest")

    def test_a_tests_directory_is_the_fallback(self):
        brain = self._build({"pyproject.toml": '[project]\nname = "d"\n',
                             "tests/test_x.py": "pass\n"})
        self.assertIn("unittest", brain.fact(REPO, "test_command").value)

    # -- makefile ----------------------------------------------------------

    def test_makefile_targets_become_commands(self):
        brain = self._build({"Makefile": "test:\n\tpytest -q\n\nbuild:\n\ttsc\n"})
        node = brain.get("L5:command:make:test")
        self.assertIsNotNone(node)
        self.assertEqual(node.attrs["recipe"], ["pytest -q"])

    def test_variable_assignments_are_not_targets(self):
        brain = self._build({"Makefile": "PYTHON := python3\ntest:\n\tpytest\n"})
        self.assertIsNone(brain.get("L5:command:make:PYTHON"))
        self.assertIsNotNone(brain.get("L5:command:make:test"))

    def test_phony_is_skipped(self):
        brain = self._build({"Makefile": ".PHONY: test\ntest:\n\tpytest\n"})
        self.assertIsNone(brain.get("L5:command:make:.PHONY"))

    def test_makefile_outranks_package_json(self):
        # A Makefile target usually wraps the underlying tool, and is what a
        # human would actually type.
        brain = self._build({
            "Makefile": "test:\n\tpytest -q\n",
            "package.json": json.dumps({"scripts": {"test": "jest"}}),
        })
        found = brain.fact(REPO, "test_command")
        self.assertEqual(found.value, "make test")
        self.assertIn("npm run test", [a["command"] for a in found.attrs["alternatives"]])

    # -- the envelope story ------------------------------------------------

    def test_the_chosen_command_is_derived_not_extracted(self):
        # That a `test` script exists is written down; that it is *the* test
        # command is a judgement, and stays one until P3 executes it.
        brain = self._build({"package.json": json.dumps({"scripts": {"test": "jest"}})})
        found = brain.fact(REPO, "test_command")
        self.assertIs(found.env.method, Method.DERIVED)
        self.assertIn("unverified", found.env.note)

    def test_the_script_itself_is_extracted(self):
        brain = self._build({"package.json": json.dumps({"scripts": {"test": "jest"}})})
        self.assertIs(brain.get("L5:command:package.json:test").env.method,
                      Method.EXTRACTED)

    # -- CI, containers, ownership ----------------------------------------

    def test_workflow_steps_are_scanned(self):
        brain = self._build({".github/workflows/ci.yml":
                             "jobs:\n  build:\n    steps:\n"
                             "      - uses: actions/checkout@v4\n"
                             "      - run: pytest -q\n"})
        node = brain.get("L5:pipeline:.github/workflows/ci.yml")
        self.assertIsNotNone(node)
        self.assertIn("pytest -q", node.attrs["steps"])
        self.assertIn("actions/checkout@v4", node.attrs["actions"])

    def test_workflow_parsing_admits_it_is_not_yaml(self):
        brain = self._build({".github/workflows/ci.yml": "jobs:\n  x:\n    steps:\n"
                                                          "      - run: make test\n"})
        node = brain.get("L5:pipeline:.github/workflows/ci.yml")
        self.assertIs(node.env.method, Method.DERIVED)
        self.assertIn("not YAML-parsed", node.env.note)

    def test_dockerfile(self):
        brain = self._build({"Dockerfile":
                             "FROM python:3.12-slim\nEXPOSE 8080\nCMD [\"app\"]\n"})
        node = brain.get("L5:container:Dockerfile")
        self.assertEqual(node.attrs["base_image"], "python:3.12-slim")
        self.assertEqual(node.attrs["exposed_ports"], ["8080"])

    def test_codeowners(self):
        brain = self._build({"CODEOWNERS":
                             "# comment\n/payments/ @risk-eng @sec\n*.md @docs\n"})
        rules = brain.fact(REPO, "codeowners").value
        self.assertEqual(rules[0]["pattern"], "/payments/")
        self.assertEqual(rules[0]["owners"], ["@risk-eng", "@sec"])
        self.assertEqual(len(rules), 2)

    def test_a_bare_repo_yields_nothing_rather_than_guessing(self):
        brain = self._build({"README.md": "# hi\n"})
        self.assertIsNone(brain.fact(REPO, "test_command"))


if __name__ == "__main__":
    unittest.main()
