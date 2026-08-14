from __future__ import annotations

import unittest

from codebrain.diff import diff, render
from codebrain.envelope import Envelope, Evidence, Method, Status
from codebrain.model import REPO, Brain, Fact, Layer, Node


def node(key, value="v", method=Method.EXTRACTED, conf=None, status=Status.FRESH):
    return Node(layer=Layer.L1, kind="symbol", key=key,
                env=Envelope.make(method, source="t", ts="2026-01-01T00:00:00+00:00",
                                  confidence=conf, status=status,
                                  evidence=(Evidence(path="a.py"),)),
                attrs={"value": value})


def brain(*nodes) -> Brain:
    b = Brain()
    b.extend(nodes)
    return b


class TestDiff(unittest.TestCase):
    def test_identical_brains_are_empty(self):
        self.assertTrue(diff(brain(node("a")), brain(node("a"))).empty)

    def test_detects_added_and_removed(self):
        d = diff(brain(node("a")), brain(node("b")))
        self.assertEqual([r.id for r in d.added], ["L1:symbol:b"])
        self.assertEqual([r.id for r in d.removed], ["L1:symbol:a"])
        self.assertTrue(d.substantive)

    def test_detects_payload_change(self):
        d = diff(brain(node("a", value="old")), brain(node("a", value="new")))
        self.assertEqual(len(d.changed), 1)
        self.assertTrue(d.changed[0].payload_changed)
        self.assertTrue(d.substantive)

    def test_envelope_only_change_is_not_substantive(self):
        # A rebuild re-derives confidence and status. The drift gate must fire
        # on claims moving, not on the Brain being rebuilt.
        d = diff(brain(node("a", conf=0.8)), brain(node("a", conf=0.9)))
        self.assertEqual(len(d.changed), 1)
        self.assertFalse(d.changed[0].payload_changed)
        self.assertTrue(d.changed[0].envelope_changed)
        self.assertFalse(d.substantive)

    def test_verification_shows_up_as_a_method_transition(self):
        before = brain(node("a", method=Method.DERIVED))
        after = brain(node("a", method=Method.OBSERVED))
        d = diff(before, after)
        self.assertIn("DERIVED→OBSERVED", d.changed[0].summary)

    def test_refutation_is_visible_in_the_summary(self):
        before = brain(node("a"))
        refuted = node("a")
        refuted.env = refuted.env.demote("command failed")
        d = diff(before, brain(refuted))
        self.assertIn("refuted", d.changed[0].summary)

    def test_diff_is_deterministic_and_sorted(self):
        old = brain(node("b"), node("a"))
        new = brain(node("d"), node("c"))
        d = diff(old, new)
        self.assertEqual([r.id for r in d.added], sorted(r.id for r in d.added))
        self.assertEqual([r.id for r in d.removed], sorted(r.id for r in d.removed))

    def test_render_marks_metadata_only_changes(self):
        d = diff(brain(node("a", conf=0.8)), brain(node("a", conf=0.9)))
        self.assertIn("metadata only", render(d))

    def test_render_of_empty_diff(self):
        self.assertEqual(render(diff(brain(), brain())), "No change.")

    def test_facts_participate_in_diffs(self):
        def fact(value):
            b = Brain()
            b.add(Fact(layer=Layer.L5, subject=REPO, predicate="test_command", value=value,
                       env=Envelope.make(Method.EXTRACTED, source="t",
                                         evidence=(Evidence(path="Makefile"),))))
            return b

        d = diff(fact("pytest"), fact("make test"))
        self.assertTrue(d.substantive)
        self.assertEqual(len(d.changed), 1)


if __name__ == "__main__":
    unittest.main()
