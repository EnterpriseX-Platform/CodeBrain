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

    # -- rust ---------------------------------------------------------------

    def test_cargo_toml_yields_cargo_test(self):
        brain = self._build({"Cargo.toml":
                             '[package]\nname = "spike-shell"\nversion = "0.4.194"\n'})
        found = brain.fact(REPO, "test_command")
        self.assertEqual(found.value, "cargo test")
        self.assertEqual(found.attrs["source"], "cargo.toml")

    def test_a_nested_manifest_gets_a_manifest_path_flag(self):
        # Every Tauri app keeps Cargo.toml under src-tauri/, not the root. And
        # verify.py always runs commands with cwd at the repo root, so without
        # --manifest-path `cargo` finds nothing there and wrongly refutes a
        # test command that was actually fine.
        brain = self._build({"spike-shell/src-tauri/Cargo.toml":
                             '[package]\nname = "spike-shell"\n'})
        found = brain.fact(REPO, "test_command")
        self.assertEqual(found.value,
                         "cargo test --manifest-path spike-shell/src-tauri/Cargo.toml")

    def test_nested_manifest_lint_also_gets_the_flag(self):
        brain = self._build({"app/Cargo.toml": '[package]\nname = "x"\n'})
        self.assertEqual(brain.fact(REPO, "lint_command").value,
                         "cargo clippy --manifest-path app/Cargo.toml")

    def test_cargo_test_is_offered_even_with_no_test_functions(self):
        # A crate with zero #[test] fns still runs `cargo test` successfully and
        # reports zero — same posture as an empty pytest suite.
        brain = self._build({"Cargo.toml": '[package]\nname = "x"\n'})
        self.assertIsNotNone(brain.fact(REPO, "test_command"))

    def test_cargo_clippy_is_the_lint_command(self):
        brain = self._build({"Cargo.toml": '[package]\nname = "x"\n'})
        self.assertEqual(brain.fact(REPO, "lint_command").value, "cargo clippy")

    def test_cargo_never_offers_build_or_run(self):
        # In a Tauri-shaped repo `cargo build`/`cargo run` compile the crate
        # alone, not the packaged app — offering them would be actively wrong.
        brain = self._build({"Cargo.toml": '[package]\nname = "x"\n'})
        self.assertIsNone(brain.fact(REPO, "build_command"))
        self.assertIsNone(brain.fact(REPO, "run_command"))

    def test_a_workflow_build_command_is_not_displaced_by_cargo(self):
        # This is the regression that matters: Cargo contributes no build
        # candidate at all, so whatever the workflow found stays the pick.
        brain = self._build({
            "spike-shell/src-tauri/Cargo.toml": '[package]\nname = "spike-shell"\n',
            ".github/workflows/build-macos.yml":
                "jobs:\n  macos-dmg:\n    steps:\n"
                "      - run: npx tauri build --bundles dmg\n",
        })
        found = brain.fact(REPO, "build_command")
        self.assertEqual(found.value, "npx tauri build --bundles dmg")

    def test_a_makefile_test_target_still_outranks_cargo(self):
        brain = self._build({
            "Cargo.toml": '[package]\nname = "x"\n',
            "Makefile": "test:\n\tcargo test --all-features\n",
        })
        found = brain.fact(REPO, "test_command")
        self.assertEqual(found.value, "make test")
        self.assertIn("cargo test", [a["command"] for a in found.attrs["alternatives"]])

    def test_rust_package_metadata(self):
        brain = self._build({"Cargo.toml":
                             '[package]\nname = "spike-shell"\nversion = "0.4.194"\n'
                             'edition = "2021"\n[dependencies]\ntauri = "2"\nserde = "1"\n'})
        value = brain.fact(REPO, "rust_package").value
        self.assertEqual(value["name"], "spike-shell")
        self.assertEqual(value["version"], "0.4.194")
        self.assertEqual(value["dependencies"], 2)
        self.assertFalse(value["workspace"])

    def test_a_cargo_workspace_is_recognised_without_a_package_table(self):
        brain = self._build({"Cargo.toml": '[workspace]\nmembers = ["crate-a", "crate-b"]\n'})
        self.assertTrue(brain.fact(REPO, "rust_package").value["workspace"])

    def test_malformed_cargo_toml_does_not_crash(self):
        brain = self._build({"Cargo.toml": "this is not toml [[["})
        self.assertIsNone(brain.fact(REPO, "test_command"))

    def test_multiple_manifests_each_get_their_own_fact(self):
        brain = self._build({
            "crates/a/Cargo.toml": '[package]\nname = "a"\n',
            "crates/b/Cargo.toml": '[package]\nname = "b"\n',
        })
        a = brain.fact(REPO, "rust_package:crates/a/Cargo.toml")
        b = brain.fact(REPO, "rust_package:crates/b/Cargo.toml")
        self.assertEqual((a.value["name"], b.value["name"]), ("a", "b"))

    def test_vendored_cargo_manifests_are_ignored(self):
        # DEFAULT_IGNORE already excludes vendor/target/node_modules; a Cargo
        # provider that walked past that would offer a test command for code
        # that isn't even this repo's own.
        brain = self._build({
            "Cargo.toml": '[package]\nname = "real"\n',
            "vendor/dep/Cargo.toml": '[package]\nname = "vendored"\n',
        })
        found = brain.fact(REPO, "test_command")
        self.assertEqual(found.value, "cargo test")

    def test_a_bare_repo_yields_nothing_rather_than_guessing(self):
        brain = self._build({"README.md": "# hi\n"})
        self.assertIsNone(brain.fact(REPO, "test_command"))


if __name__ == "__main__":
    unittest.main()
