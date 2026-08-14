from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Envelope, Evidence, Method
from codebrain.model import REPO, Fact, Layer, new_brain
from codebrain.trial import (
    CONTROL,
    FAILURE,
    MIN_PER_ARM,
    SUCCESS,
    TREATMENT,
    UNKNOWN,
    Event,
    Trace,
    assign_arm,
    clear_active,
    get_active,
    list_trials,
    load_traces,
    render,
    save_trace,
    set_active,
    summarise,
    verification_command,
)


def trace(arm=TREATMENT, outcome=SUCCESS, edits=(), baseline=None, verdict=None,
          session="s1", trial="t") -> Trace:
    return Trace(trial=trial, session=session, arm=arm, task="do a thing",
                 outcome=outcome,
                 events=[Event(kind="edit", path=p) for p in edits],
                 baseline=baseline or {}, verdict=verdict or {})


class TestArmAssignment(unittest.TestCase):
    def test_deterministic(self):
        # Re-derivable, so a recorded trace can be audited afterwards.
        self.assertEqual(assign_arm("t", "abc"), assign_arm("t", "abc"))

    def test_depends_on_the_trial_as_well_as_the_session(self):
        arms = {assign_arm(f"trial-{i}", "abc") for i in range(20)}
        self.assertEqual(arms, {TREATMENT, CONTROL})

    def test_roughly_balanced(self):
        arms = [assign_arm("t", f"session-{i}") for i in range(1000)]
        share = arms.count(TREATMENT) / len(arms)
        self.assertGreater(share, 0.44)
        self.assertLess(share, 0.56)

    def test_split_is_respected(self):
        arms = [assign_arm("t", f"s{i}", split=0.2) for i in range(1000)]
        self.assertLess(arms.count(TREATMENT) / len(arms), 0.26)

    def test_always_returns_a_known_arm(self):
        for i in range(50):
            self.assertIn(assign_arm("t", str(i)), (TREATMENT, CONTROL))


class TestTrace(unittest.TestCase):
    def test_edits_are_deduplicated_in_order(self):
        t = trace(edits=("a.py", "b.py", "a.py"))
        self.assertEqual(t.edits, ["a.py", "b.py"])

    def test_failed_commands_counted(self):
        t = trace()
        t.events += [Event(kind="command", command="x", exit_code=1),
                     Event(kind="command", command="y", exit_code=0)]
        self.assertEqual(t.failed_commands, 1)

    def test_regression_needs_a_passing_baseline(self):
        broke = trace(baseline={"exit_code": 0}, verdict={"exit_code": 1})
        self.assertTrue(broke.broke_a_passing_build)

    def test_an_already_failing_build_is_not_a_regression(self):
        # It was broken before the session; blaming the session would be wrong.
        already = trace(baseline={"exit_code": 1}, verdict={"exit_code": 1})
        self.assertFalse(already.broke_a_passing_build)

    def test_no_baseline_means_no_regression_claim(self):
        self.assertFalse(trace(verdict={"exit_code": 1}).broke_a_passing_build)

    def test_roundtrip(self):
        original = trace(edits=("a.py",), baseline={"exit_code": 0})
        restored = Trace.from_json(original.to_json())
        self.assertEqual(restored.arm, original.arm)
        self.assertEqual(restored.edits, original.edits)
        self.assertEqual(restored.baseline, original.baseline)


class TestSummarise(unittest.TestCase):
    def test_unknowns_are_excluded_from_rates_not_counted_as_failures(self):
        summaries = summarise([trace(outcome=SUCCESS), trace(outcome=UNKNOWN)])
        arm = summaries[TREATMENT]
        self.assertEqual((arm.n, arm.decided, arm.unknown), (2, 1, 1))
        self.assertEqual(arm.success_rate, 1.0)

    def test_no_decided_sessions_gives_no_rate_rather_than_zero(self):
        arm = summarise([trace(outcome=UNKNOWN)])[TREATMENT]
        self.assertIsNone(arm.success_rate)

    def test_both_arms_always_present(self):
        self.assertEqual(set(summarise([]).keys()), {TREATMENT, CONTROL})

    def test_context_tokens_only_tracked_for_treatment(self):
        t = trace(arm=TREATMENT)
        t.context_tokens = 900
        c = trace(arm=CONTROL)
        c.context_tokens = 900
        summaries = summarise([t, c])
        self.assertEqual(summaries[TREATMENT].context_tokens, [900])
        self.assertEqual(summaries[CONTROL].context_tokens, [])


class TestRender(unittest.TestCase):
    def _many(self, arm, outcome, n):
        return [trace(arm=arm, outcome=outcome, session=f"{arm}-{outcome}-{i}")
                for i in range(n)]

    def test_small_samples_are_refused_not_reported(self):
        text = render("t", self._many(TREATMENT, SUCCESS, 1)
                      + self._many(CONTROL, FAILURE, 1))
        self.assertIn("NOT YET A RESULT", text)
        self.assertIn("not for quoting", text)
        self.assertNotIn("success delta", text)

    def test_a_sufficient_sample_reports_a_delta(self):
        text = render("t", self._many(TREATMENT, SUCCESS, MIN_PER_ARM)
                      + self._many(CONTROL, FAILURE, MIN_PER_ARM))
        self.assertIn("success delta", text)
        self.assertNotIn("NOT YET A RESULT", text)

    def test_the_delta_is_printed_with_its_n(self):
        text = render("t", self._many(TREATMENT, SUCCESS, MIN_PER_ARM)
                      + self._many(CONTROL, FAILURE, MIN_PER_ARM))
        self.assertIn(f"{MIN_PER_ARM} vs {MIN_PER_ARM} decided", text)

    def test_unknowns_are_declared(self):
        text = render("t", self._many(TREATMENT, UNKNOWN, 2))
        self.assertIn("no verdict", text)

    def test_empty_trial_explains_how_to_start(self):
        self.assertIn("codebrain trial start", render("t", []))


class TestStorage(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_trace(tmp, trace(session="a"))
            save_trace(tmp, trace(session="b", arm=CONTROL))
            loaded = load_traces(tmp, "t")
        self.assertEqual({t.session for t in loaded}, {"a", "b"})

    def test_trials_are_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_trace(tmp, trace(trial="one"))
            save_trace(tmp, trace(trial="two"))
            self.assertEqual(list_trials(tmp), ["one", "two"])

    def test_awkward_session_ids_are_made_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_trace(tmp, trace(session="../../etc/passwd"))
            self.assertEqual(len(load_traces(tmp, "t")), 1)

    def test_active_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(get_active(tmp))
            set_active(tmp, "t", "s1")
            self.assertEqual(get_active(tmp), ("t", "s1"))
            clear_active(tmp)
            self.assertIsNone(get_active(tmp))

    def test_a_corrupt_trace_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_trace(tmp, trace(session="good"))
            bad = Path(tmp) / "trials" / "t" / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertEqual([t.session for t in load_traces(tmp, "t")], ["good"])


class TestVerificationSource(unittest.TestCase):
    def test_the_repos_command_is_used(self):
        """The harness must not be scored by the system under test."""
        brain = new_brain()
        brain.add(Fact(layer=Layer.L5, subject=REPO, predicate="test_command",
                       value="make test",
                       env=Envelope.make(Method.OBSERVED, source="verify",
                                         evidence=(Evidence(path="Makefile"),))))
        self.assertEqual(verification_command(brain), ("make test", "OBSERVED"))

    def test_no_command_means_no_verification(self):
        self.assertIsNone(verification_command(new_brain()))

    def test_the_method_travels_so_the_report_can_say_how_solid_it_is(self):
        brain = new_brain()
        brain.add(Fact(layer=Layer.L5, subject=REPO, predicate="test_command",
                       value="pytest",
                       env=Envelope.make(Method.DERIVED, source="operations",
                                         evidence=(Evidence(path="pyproject.toml"),))))
        self.assertEqual(verification_command(brain)[1], "DERIVED")


if __name__ == "__main__":
    unittest.main()
