"""The P0 gate: two extractors coexist behind the provider interface, and their
disagreement is resolved by the envelope rather than by whoever ran last."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Iterable

from codebrain.envelope import Envelope, Evidence, Method
from codebrain.model import REPO, Fact, Layer, Node, Record
from codebrain.providers import BuildContext, Provider, Registry, build
from codebrain.extractors.census import CensusProvider, classify
from codebrain.extractors.gitmeta import GitMetaProvider


def make_repo(tmp: str) -> Path:
    root = Path(tmp)
    (root / "src").mkdir()
    (root / "src" / "api.py").write_text("def charge():\n    return 1\n", encoding="utf-8")
    (root / "src" / "app.ts").write_text("export const x = 1;\n", encoding="utf-8")
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.js").write_text("junk\n", encoding="utf-8")
    return root


class Weak(Provider):
    id = "weak"
    layers = (Layer.L0,)
    order = 1

    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        yield Fact(layer=Layer.L0, subject=REPO, predicate="repo_name", value="guess",
                   env=Envelope.make(Method.DERIVED, source=self.id, confidence=0.5,
                                     ts="2026-01-01T00:00:00+00:00",
                                     evidence=(Evidence(path="."),)))


class Strong(Provider):
    id = "strong"
    layers = (Layer.L0,)
    order = 2

    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        yield Fact(layer=Layer.L0, subject=REPO, predicate="repo_name", value="truth",
                   env=Envelope.make(Method.EXTRACTED, source=self.id,
                                     ts="2026-01-01T00:00:00+00:00",
                                     evidence=(Evidence(path=".git/config"),)))


class Broken(Provider):
    id = "broken"
    layers = (Layer.L0,)
    order = 3

    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        raise RuntimeError("tree-sitter grammar missing")


class Inapplicable(Provider):
    id = "inapplicable"
    layers = (Layer.L2,)

    def applies(self, ctx: BuildContext) -> bool:
        return False

    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        raise AssertionError("must not run")


class TestRegistry(unittest.TestCase):
    def test_duplicate_ids_are_rejected(self):
        reg = Registry()
        reg.register(Weak())
        with self.assertRaises(ValueError):
            reg.register(Weak())

    def test_ordering_is_by_order_then_id(self):
        reg = Registry()
        reg.register(Strong())
        reg.register(Weak())
        self.assertEqual([p.id for p in reg.all()], ["weak", "strong"])


class TestBuild(unittest.TestCase):
    def test_two_providers_coexist_and_the_envelope_settles_the_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(root=Path(tmp))
            result = build(ctx, [Weak(), Strong()])

        self.assertEqual(sorted(result.ran), ["strong", "weak"])
        fact = result.brain.fact(REPO, "repo_name")
        self.assertEqual(fact.value, "truth")
        self.assertEqual(fact.env.source, "strong")
        self.assertEqual(result.report.conflicts, 1)

    def test_conflict_resolution_is_independent_of_provider_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(root=Path(tmp))
            forward = build(ctx, [Weak(), Strong()]).brain.fact(REPO, "repo_name")
            backward = build(ctx, [Strong(), Weak()]).brain.fact(REPO, "repo_name")
        self.assertEqual(forward.value, backward.value)

    def test_a_broken_provider_does_not_lose_the_build(self):
        # Principle vi: a partial Brain beats no Brain.
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(root=Path(tmp))
            result = build(ctx, [Strong(), Broken()])
        self.assertEqual(result.ran, ["strong"])
        self.assertIn("broken", result.failed)
        self.assertIn("tree-sitter", result.failed["broken"])
        self.assertIsNotNone(result.brain.fact(REPO, "repo_name"))

    def test_inapplicable_providers_are_skipped_not_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = build(BuildContext(root=Path(tmp)), [Inapplicable()])
        self.assertEqual(result.skipped, ["inapplicable"])

    def test_manifest_records_what_actually_ran(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = build(BuildContext(root=Path(tmp)), [Strong(), Broken()])
        self.assertEqual(result.brain.manifest.providers, ["strong"])


class TestBuildContext(unittest.TestCase):
    def test_ignores_are_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            names = {Path(p).name for p in BuildContext(root=root).iter_files()}
        self.assertIn("api.py", names)
        self.assertNotIn("junk.js", names)

    def test_relative_paths_are_posix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            ctx = BuildContext(root=root)
            rels = {ctx.rel(p) for p in ctx.iter_files()}
        self.assertIn("src/api.py", rels)
        self.assertFalse(any("\\" in r for r in rels))


class TestCensus(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify("api.py", ".py"), "Python")
        self.assertEqual(classify("Dockerfile", ""), "Dockerfile")
        self.assertEqual(classify("mystery.zzz", ".zzz"), "unknown")

    def test_census_counts_and_classifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            result = build(BuildContext(root=root), [CensusProvider()])

        brain = result.brain
        self.assertEqual(brain.fact(REPO, "file_count").value, 3)
        self.assertEqual(brain.fact(REPO, "language_mix").value["Python"], 1)
        self.assertIsNotNone(brain.get("L0:file:src/api.py"))
        self.assertEqual(brain.get("L0:file:src/api.py").attrs["language"], "Python")

    def test_primary_language_is_derived_not_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            brain = build(BuildContext(root=root), [CensusProvider()]).brain
        primary = brain.fact(REPO, "primary_language")
        self.assertIs(primary.env.method, Method.DERIVED)

    def test_census_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            one = build(BuildContext(root=root), [CensusProvider()]).brain
            two = build(BuildContext(root=root), [CensusProvider()]).brain
        self.assertEqual({r.id for r in one.records()}, {r.id for r in two.records()})


def _git_available() -> bool:
    try:
        return subprocess.run(("git", "--version"), capture_output=True,
                              timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@unittest.skipUnless(_git_available(), "git not available")
class TestGitMeta(unittest.TestCase):
    def _init_repo(self, root: Path) -> None:
        run = lambda *a: subprocess.run(("git", *a), cwd=str(root),  # noqa: E731
                                        capture_output=True, timeout=20)
        run("init", "-b", "main")
        run("config", "user.email", "t@example.com")
        run("config", "user.name", "Test")
        run("config", "commit.gpgsign", "false")
        run("add", "-A")
        run("-c", "commit.gpgsign=false", "commit", "-m", "initial")

    def test_declines_outside_a_git_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(GitMetaProvider().applies(BuildContext(root=Path(tmp))))

    def test_extracts_identity_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            self._init_repo(root)
            ctx = BuildContext(root=root)
            if not GitMetaProvider().applies(ctx):
                self.skipTest("git repo could not be created here")
            brain = build(ctx, [GitMetaProvider()]).brain

        self.assertIsNotNone(brain.fact(REPO, "head"))
        self.assertEqual(brain.fact(REPO, "commit_count").value, 1)
        self.assertEqual(len(brain.fact(REPO, "head").value), 40)
        self.assertTrue([r for r in brain.by_layer(Layer.L4) if isinstance(r, Node)])


if __name__ == "__main__":
    unittest.main()
