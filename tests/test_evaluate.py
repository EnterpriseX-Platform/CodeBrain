from __future__ import annotations

import unittest

from codebrain.envelope import Envelope, Evidence, Method
from codebrain.evaluate import (
    MAX_FILES_PER_CASE,
    Case,
    CaseResult,
    Report,
    is_automated,
    record_path,
    recall,
)
from codebrain.model import Brain, Fact, Layer, Node


def env():
    return Envelope.make(Method.EXTRACTED, source="t",
                         evidence=(Evidence(path="a.py"),))


class TestAutomatedFilter(unittest.TestCase):
    def test_dependency_bumps_are_excluded(self):
        for subject in (
            "Bump the actions group with 2 updates (#7596)",
            "Bump actions/setup-python from 6.2.0 to 6.3.0",
            "chore(deps): update foo",
            "[pre-commit.ci] pre-commit autoupdate",
            "Merge pull request #12 from x/y",
        ):
            self.assertTrue(is_automated(subject), subject)

    def test_real_engineering_commits_are_kept(self):
        for subject in (
            "Add hasattr checks for remaining protocol isinstance checks",
            "Fix cookie handling on redirect",
            "Improve static typing of 3rd party imports",
        ):
            self.assertFalse(is_automated(subject), subject)

    def test_bumper_named_in_a_real_subject_is_not_over_matched(self):
        self.assertFalse(is_automated("Fix crash when the bumper cache is cold"))


class TestRecall(unittest.TestCase):
    def test_full_recall(self):
        self.assertEqual(recall(["a.py", "b.py"], ["a.py", "b.py"], 2), 1.0)

    def test_partial_recall(self):
        self.assertEqual(recall(["a.py", "z.py"], ["a.py", "b.py"], 2), 0.5)

    def test_only_the_first_k_count(self):
        self.assertEqual(recall(["z.py", "a.py"], ["a.py"], 1), 0.0)

    def test_suffix_matching_handles_absolute_paths(self):
        self.assertEqual(recall(["/tmp/repo/src/a.py"], ["src/a.py"], 1), 1.0)

    def test_empty_truth(self):
        self.assertEqual(recall(["a.py"], [], 1), 0.0)


class TestRecordPath(unittest.TestCase):
    def setUp(self):
        self.brain = Brain()
        self.brain.add(Node(layer=Layer.L1, kind="symbol", key="pay/api.py#charge",
                            env=env(), attrs={"module": "pay/api.py"}))
        self.brain.add(Node(layer=Layer.L0, kind="file", key="pay/api.py", env=env()))
        self.brain.add(Fact(layer=Layer.L4, subject="L0:file:pay/api.py",
                            predicate="churn", value={}, env=env()))

    def test_symbol_resolves_to_its_module(self):
        self.assertEqual(record_path(self.brain, "L1:symbol:pay/api.py#charge"),
                         "pay/api.py")

    def test_file_node_resolves_to_itself(self):
        self.assertEqual(record_path(self.brain, "L0:file:pay/api.py"), "pay/api.py")

    def test_fact_resolves_through_its_subject(self):
        self.assertEqual(record_path(self.brain, "L4:fact:L0:file:pay/api.py|churn"),
                         "pay/api.py")

    def test_unknown_id_with_a_file_segment_still_resolves(self):
        self.assertEqual(record_path(self.brain, "L6:fact:L0:file:gone.py|x"), "gone.py")

    def test_no_id(self):
        self.assertIsNone(record_path(self.brain, ""))

    def test_a_confidence_value_is_not_mistaken_for_a_file(self):
        # The bug this guards: `0.80` inside a `[DERIVED 0.80]` tag was being
        # counted as a predicted filename, burning a slot per pack item.
        self.assertIsNone(record_path(self.brain, "0.80"))


class TestReport(unittest.TestCase):
    def _result(self, pack: float, grep: float) -> CaseResult:
        case = Case(sha="x", subject="s", changed=["a.py"])
        return CaseResult(case=case, pack_files=[], grep_files=[],
                          pack_recall=pack, grep_recall=grep, tokens=100, k=5)

    def test_win_loss_tie_counting(self):
        report = Report(results=[self._result(1.0, 0.5), self._result(0.5, 1.0),
                                 self._result(0.5, 0.5)])
        self.assertEqual((report.wins, report.losses, report.ties), (1, 1, 1))

    def test_means(self):
        report = Report(results=[self._result(1.0, 0.0), self._result(0.0, 1.0)])
        self.assertEqual(report.mean("pack_recall"), 0.5)

    def test_empty_report_does_not_divide_by_zero(self):
        self.assertEqual(Report().mean("pack_recall"), 0.0)


class TestRigorousSetup(unittest.TestCase):
    def test_extractors_are_registered_for_programmatic_callers(self):
        """The registry is populated by importing codebrain.extractors. When
        run_rigorous relied on the caller having done that, every worktree Brain
        came out empty and every case was reported "skipped" — a silent nothing
        that reads exactly like a real result."""
        import importlib

        import codebrain.evaluate as ev
        importlib.reload(ev)
        from codebrain.providers import REGISTRY

        # Simulate a caller that never imported the extractors package.
        self.assertGreater(len(REGISTRY), 0)
        self.assertIn("census", [p.id for p in REGISTRY.all()])


class TestCaseSelection(unittest.TestCase):
    def test_bulk_commits_are_out_of_scope(self):
        # A 300-file reformat has no meaningful "correct answer" and would
        # flatter any retrieval method.
        self.assertLess(MAX_FILES_PER_CASE, 20)


if __name__ == "__main__":
    unittest.main()
