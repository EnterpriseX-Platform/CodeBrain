from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Envelope, Evidence, Method, Status
from codebrain.model import REPO, Brain, Fact, Layer, Node, new_brain
from codebrain.sync import (
    carry_forward,
    drift,
    needs_rebuild,
    working_tree_paths,
)


def env(method=Method.EXTRACTED, **kw):
    return Envelope.make(method, source="t", evidence=(Evidence(path="a.py"),), **kw)


def fact(predicate: str, value, method=Method.EXTRACTED, **kw) -> Fact:
    return Fact(layer=Layer.L5, subject=REPO, predicate=predicate, value=value,
                env=env(method, **kw))


def git(root: Path, *args: str):
    return subprocess.run(("git", *args), cwd=str(root), capture_output=True, timeout=30)


def make_repo(tmp: str) -> Path:
    root = Path(tmp)
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    git(root, "init", "-b", "main")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "T")
    git(root, "add", "-A")
    git(root, "-c", "commit.gpgsign=false", "commit", "-m", "initial")
    return root


class TestCarryForward(unittest.TestCase):
    def test_observed_claims_survive_a_rebuild(self):
        # Without this, every build erases every verification and P3 is
        # pointless by the next commit.
        old = new_brain()
        proven = fact("test_command", "make test", Method.OBSERVED,
                      status=Status.FRESH, note="verified: passed in 3.9s")
        old.add(proven)

        fresh = new_brain()
        fresh.add(fact("test_command", "make test", Method.DERIVED))

        carried, invalidated = carry_forward(old, fresh)
        self.assertEqual((carried, invalidated), (1, 0))
        self.assertIs(fresh.fact(REPO, "test_command").env.method, Method.OBSERVED)

    def test_human_assertions_survive_a_rebuild(self):
        old, fresh = new_brain(), new_brain()
        old.add(fact("test_command", "make test", Method.ASSERTED))
        fresh.add(fact("test_command", "make test", Method.DERIVED))
        self.assertEqual(carry_forward(old, fresh)[0], 1)
        self.assertIs(fresh.fact(REPO, "test_command").env.method, Method.ASSERTED)

    def test_a_verification_of_a_changed_claim_is_invalidated(self):
        # The evidence proved `make test`. The command is now `pytest`. Carrying
        # the receipt across would be the Brain lying with a real receipt.
        old, fresh = new_brain(), new_brain()
        old.add(fact("test_command", "make test", Method.OBSERVED))
        fresh.add(fact("test_command", "pytest", Method.DERIVED))

        carried, invalidated = carry_forward(old, fresh)
        self.assertEqual((carried, invalidated), (0, 1))
        self.assertIs(fresh.fact(REPO, "test_command").env.method, Method.DERIVED)

    def test_reproducible_claims_are_not_carried(self):
        old, fresh = new_brain(), new_brain()
        old.add(fact("test_command", "make test", Method.EXTRACTED, confidence=0.99))
        fresh.add(fact("test_command", "make test", Method.EXTRACTED, confidence=0.5))
        self.assertEqual(carry_forward(old, fresh)[0], 0)
        self.assertEqual(fresh.fact(REPO, "test_command").env.confidence, 0.5)

    def test_refutations_survive_a_rebuild(self):
        # `demote` leaves the method alone, so a refuted claim is not OBSERVED.
        # If it is not carried, the rebuild regenerates it as "never executed"
        # and the next agent runs a command already proved broken.
        old, fresh = new_brain(), new_brain()
        refuted = fact("test_command", "make test", Method.DERIVED)
        refuted.env = refuted.env.demote("execution failed — exit 2")
        old.add(refuted)
        fresh.add(fact("test_command", "make test", Method.DERIVED))

        carried, _ = carry_forward(old, fresh)
        self.assertEqual(carried, 1)
        self.assertIs(fresh.fact(REPO, "test_command").env.status, Status.REFUTED)

    def test_a_refutation_of_a_changed_command_is_invalidated(self):
        old, fresh = new_brain(), new_brain()
        refuted = fact("test_command", "make test", Method.DERIVED)
        refuted.env = refuted.env.demote("failed")
        old.add(refuted)
        fresh.add(fact("test_command", "pytest -q", Method.DERIVED))

        carried, invalidated = carry_forward(old, fresh)
        self.assertEqual((carried, invalidated), (0, 1))
        self.assertIs(fresh.fact(REPO, "test_command").env.status, Status.UNVERIFIED)

    def test_a_claim_that_no_longer_exists_is_dropped(self):
        old, fresh = new_brain(), new_brain()
        old.add(fact("test_command", "make test", Method.OBSERVED))
        self.assertEqual(carry_forward(old, fresh), (0, 0))
        self.assertEqual(len(fresh), 0)


class TestWorkingTreePaths(unittest.TestCase):
    def test_unstaged_modification_keeps_its_full_path(self):
        # `git status --porcelain` writes " M path"; stripping the payload eats
        # the leading space and the fixed slice then bites the filename.
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            (root / "a.py").write_text("x = 2\n", encoding="utf-8")
            paths = working_tree_paths(root)
        self.assertEqual(paths, {"a.py"})

    def test_untracked_files_are_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            (root / "b.py").write_text("y = 1\n", encoding="utf-8")
            self.assertIn("b.py", working_tree_paths(root))

    def test_clean_tree_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(working_tree_paths(make_repo(tmp)), set())

    def test_not_a_repo_is_empty_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(working_tree_paths(Path(tmp)), set())


class TestNeedsRebuild(unittest.TestCase):
    def test_clean_repo_at_head_needs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=str(root),
                                  capture_output=True, text=True).stdout.strip()
            brain = new_brain(as_of=head)
            rebuild, reason, _ = needs_rebuild(brain, root, root / ".brain")
        self.assertFalse(rebuild)
        self.assertIn("at HEAD", reason)

    def test_uncommitted_changes_trigger_a_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=str(root),
                                  capture_output=True, text=True).stdout.strip()
            (root / "a.py").write_text("x = 99\n", encoding="utf-8")
            rebuild, reason, changed = needs_rebuild(new_brain(as_of=head), root,
                                                     root / ".brain")
        self.assertTrue(rebuild)
        self.assertIn("uncommitted", reason)
        self.assertEqual(changed, {"a.py"})

    def test_a_moved_head_triggers_a_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            stale = "0" * 40
            rebuild, reason, _ = needs_rebuild(new_brain(as_of=stale), root,
                                               root / ".brain")
        self.assertTrue(rebuild)
        self.assertIn("HEAD moved", reason)

    def test_a_non_repo_always_rebuilds(self):
        with tempfile.TemporaryDirectory() as tmp:
            rebuild, reason, _ = needs_rebuild(new_brain(), Path(tmp),
                                               Path(tmp) / ".brain")
        self.assertTrue(rebuild)
        self.assertIn("not a git repository", reason)


class TestDrift(unittest.TestCase):
    def _brain(self, *records) -> Brain:
        brain = new_brain(as_of="abc1234")
        brain.extend(records)
        return brain

    def test_identical_brains_do_not_drift(self):
        node = Node(layer=Layer.L1, kind="symbol", key="a.py#f", env=env())
        self.assertFalse(drift(self._brain(node), self._brain(node)).substantive)

    def test_a_new_symbol_is_drift(self):
        old = self._brain(Node(layer=Layer.L1, kind="symbol", key="a.py#f", env=env()))
        new = self._brain(Node(layer=Layer.L1, kind="symbol", key="a.py#f", env=env()),
                          Node(layer=Layer.L1, kind="symbol", key="a.py#g", env=env()))
        self.assertTrue(drift(old, new).substantive)

    def test_a_verified_claim_is_not_reported_as_drift(self):
        # Drift must carry forward first, or every verified Brain fails its own
        # gate the moment it is verified.
        old = self._brain(fact("test_command", "make test", Method.OBSERVED))
        fresh = self._brain(fact("test_command", "make test", Method.DERIVED))
        self.assertFalse(drift(old, fresh).substantive)


if __name__ == "__main__":
    unittest.main()
