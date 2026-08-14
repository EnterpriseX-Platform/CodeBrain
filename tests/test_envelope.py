from __future__ import annotations

import unittest

from codebrain.envelope import Envelope, Evidence, Method, Status


def env(method, conf=None, ts="2026-01-01T00:00:00+00:00", status=Status.UNVERIFIED):
    return Envelope.make(method, source="t", ts=ts, confidence=conf, status=status,
                         evidence=(Evidence(path="a.py", start_line=1),))


class TestEnvelope(unittest.TestCase):
    def test_confidence_is_clamped(self):
        self.assertEqual(env(Method.EXTRACTED, 5.0).confidence, 1.0)
        self.assertEqual(env(Method.EXTRACTED, -3.0).confidence, 0.0)

    def test_status_decays_confidence(self):
        fresh = env(Method.EXTRACTED, 1.0, status=Status.FRESH)
        stale = env(Method.EXTRACTED, 1.0, status=Status.STALE)
        self.assertEqual(fresh.effective_confidence(), 1.0)
        self.assertEqual(stale.effective_confidence(), 0.5)

    def test_refuted_is_never_usable(self):
        refuted = env(Method.OBSERVED, 1.0).demote("command not found")
        self.assertEqual(refuted.effective_confidence(), 0.0)
        self.assertFalse(refuted.usable())
        # ...and the reason survives, because a refutation is information
        self.assertIn("not found", refuted.note)

    def test_higher_effective_confidence_wins(self):
        strong = env(Method.INFERRED, 0.9, status=Status.FRESH)
        weak = env(Method.EXTRACTED, 0.2, status=Status.FRESH)
        self.assertTrue(strong.outranks(weak))
        self.assertFalse(weak.outranks(strong))

    def test_method_trust_breaks_confidence_ties(self):
        extracted = env(Method.EXTRACTED, 0.8, status=Status.FRESH)
        derived = env(Method.DERIVED, 0.8, status=Status.FRESH)
        self.assertTrue(extracted.outranks(derived))
        self.assertFalse(derived.outranks(extracted))

    def test_human_assertion_overrules_machinery(self):
        # Lower confidence, lower trust rank — a human still wins, provided they
        # are not speaking about an older state of the world.
        human = env(Method.ASSERTED, 0.5, ts="2026-02-01T00:00:00+00:00")
        machine = env(Method.OBSERVED, 1.0, ts="2026-01-01T00:00:00+00:00",
                      status=Status.FRESH)
        self.assertTrue(human.outranks(machine))
        self.assertFalse(machine.outranks(human))

    def test_stale_human_assertion_does_not_overrule_newer_machine(self):
        human = env(Method.ASSERTED, 0.95, ts="2026-01-01T00:00:00+00:00")
        machine = env(Method.OBSERVED, 1.0, ts="2026-06-01T00:00:00+00:00",
                      status=Status.FRESH)
        self.assertFalse(human.outranks(machine))

    def test_fresh_refutation_beats_belief(self):
        belief = env(Method.EXTRACTED, 1.0, ts="2026-01-01T00:00:00+00:00",
                     status=Status.FRESH)
        refutation = env(Method.OBSERVED, 1.0, ts="2026-02-01T00:00:00+00:00").demote("no")
        self.assertTrue(refutation.outranks(belief))

    def test_outranks_is_a_strict_order(self):
        # A true tie must not come out both ways, or merge results would depend
        # on provider ordering and two identical builds could differ.
        a = env(Method.EXTRACTED, 0.8, status=Status.FRESH)
        b = env(Method.EXTRACTED, 0.8, status=Status.FRESH)
        self.assertFalse(a.outranks(b))
        self.assertFalse(b.outranks(a))

    def test_promote_and_demote_transitions(self):
        e = env(Method.DERIVED, 0.8)
        promoted = e.promote("abc1234")
        self.assertIs(promoted.method, Method.OBSERVED)
        self.assertIs(promoted.status, Status.FRESH)
        self.assertEqual(promoted.verified_at, "abc1234")

    def test_mark_stale_leaves_refuted_alone(self):
        refuted = env(Method.EXTRACTED, 1.0).demote("wrong")
        self.assertIs(refuted.mark_stale().status, Status.REFUTED)

    def test_roundtrip(self):
        original = env(Method.EXTRACTED, 0.77, status=Status.FRESH)
        restored = Envelope.from_json(original.to_json())
        self.assertEqual(restored.method, original.method)
        self.assertEqual(restored.status, original.status)
        self.assertEqual(restored.confidence, original.confidence)
        self.assertEqual(restored.evidence, original.evidence)

    def test_evidence_renders_readably(self):
        self.assertEqual(
            str(Evidence(path="pay/api.py", start_line=44, end_line=51, commit="9f21abcd")),
            "pay/api.py:44-51@9f21abc",
        )


if __name__ == "__main__":
    unittest.main()
