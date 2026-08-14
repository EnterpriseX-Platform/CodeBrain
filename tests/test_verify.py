from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Envelope, Evidence, Method, Status
from codebrain.model import REPO, Brain, Fact, Layer, new_brain
from codebrain.verify import (
    NON_TERMINATING,
    VERIFIABLE_INTENTS,
    Candidate,
    Outcome,
    apply_outcome,
    candidates,
    execute,
    render,
    verify,
)


def command_fact(intent: str, command: str, source: str = "makefile") -> Fact:
    return Fact(
        layer=Layer.L5, subject=REPO, predicate=f"{intent}_command", value=command,
        attrs={"source": source},
        env=Envelope.make(Method.DERIVED, source="operations", as_of="abc1234",
                          evidence=(Evidence(path="Makefile", start_line=1),),
                          note="chosen from 2 candidate(s); unverified until executed"),
    )


def brain_with(*facts: Fact) -> Brain:
    brain = new_brain(repo="demo", as_of="abc1234")
    brain.extend(facts)
    return brain


class TestCandidates(unittest.TestCase):
    def test_finds_executable_claims(self):
        brain = brain_with(command_fact("test", "pytest -q"),
                           command_fact("build", "make build"))
        found = {c.intent: c.command for c in candidates(brain)}
        self.assertEqual(found["test"], "pytest -q")
        self.assertEqual(found["build"], "make build")

    def test_servers_are_never_candidates(self):
        # `npm start` does not return. A verifier that hangs is worse than one
        # that never ran, so long-running intents are excluded by construction.
        brain = brain_with(command_fact("run", "npm start"))
        self.assertEqual(candidates(brain, ("run",)), [])
        for intent in NON_TERMINATING:
            self.assertNotIn(intent, VERIFIABLE_INTENTS)

    def test_empty_command_is_not_a_candidate(self):
        self.assertEqual(candidates(brain_with(command_fact("test", "   "))), [])

    def test_missing_command_is_not_a_candidate(self):
        self.assertEqual(candidates(Brain()), [])


class TestExecute(unittest.TestCase):
    def test_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = execute(f'"{sys.executable}" -c "print(1)"', Path(tmp))
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.exit_code, 0)

    def test_failure_captures_exit_code_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = execute(
                f'"{sys.executable}" -c "import sys; print(\'boom\'); sys.exit(3)"',
                Path(tmp))
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.exit_code, 3)
        self.assertIn("boom", outcome.output)

    def test_timeout_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = execute(f'"{sys.executable}" -c "import time; time.sleep(30)"',
                              Path(tmp), timeout=1)
        self.assertFalse(outcome.ok)
        self.assertIn("timed out", outcome.error)

    def test_a_nonexistent_command_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = execute("definitely-not-a-real-binary-xyz", Path(tmp))
        self.assertFalse(outcome.ok)

    def test_stdin_is_closed_so_a_prompt_cannot_hang(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = execute(f'"{sys.executable}" -c "import sys; sys.stdin.read()"',
                              Path(tmp), timeout=10)
        self.assertTrue(outcome.ok)


class TestDryRunIsTheDefault(unittest.TestCase):
    def test_verify_does_not_execute_without_an_explicit_opt_in(self):
        marker = None
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "ran"
            brain = brain_with(command_fact(
                "test", f'"{sys.executable}" -c "open(r\'{marker}\',\'w\').close()"'))
            report = verify(brain, Path(tmp))
            self.assertFalse(marker.exists())
        self.assertTrue(report.dry_run)
        self.assertEqual((report.promoted, report.refuted), (0, 0))

    def test_dry_run_shows_exactly_what_would_run(self):
        brain = brain_with(command_fact("test", "make test"))
        with tempfile.TemporaryDirectory() as tmp:
            text = render(verify(brain, Path(tmp)))
        self.assertIn("Dry run", text)
        self.assertIn("make test", text)
        self.assertIn("--yes", text)

    def test_dry_run_leaves_the_claim_untouched(self):
        brain = brain_with(command_fact("test", "make test"))
        with tempfile.TemporaryDirectory() as tmp:
            verify(brain, Path(tmp))
        self.assertIs(brain.fact(REPO, "test_command", Layer.L5).env.method,
                      Method.DERIVED)


class TestPromotionAndRefutation(unittest.TestCase):
    def test_a_passing_command_is_promoted_to_observed(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = brain_with(command_fact("test", f'"{sys.executable}" -c "pass"'))
            report = verify(brain, Path(tmp), execute_commands=True, commit="abc1234")

        fact = brain.fact(REPO, "test_command", Layer.L5)
        self.assertEqual(report.promoted, 1)
        self.assertIs(fact.env.method, Method.OBSERVED)
        self.assertIs(fact.env.status, Status.FRESH)
        self.assertEqual(fact.env.verified_at, "abc1234")
        self.assertEqual(fact.env.confidence, 1.0)

    def test_promotion_records_the_execution_as_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = brain_with(command_fact("test", f'"{sys.executable}" -c "pass"'))
            verify(brain, Path(tmp), execute_commands=True)
        refs = [e.ref or "" for e in brain.fact(REPO, "test_command", Layer.L5).env.evidence]
        self.assertTrue(any("executed:" in r for r in refs))

    def test_promotion_does_not_change_the_claim_payload(self):
        # If verifying a claim altered its payload, carry-forward would decide
        # the claim had moved and discard the evidence on the next sync.
        with tempfile.TemporaryDirectory() as tmp:
            fact = command_fact("test", f'"{sys.executable}" -c "pass"')
            before = dict(fact.attrs)
            brain = brain_with(fact)
            verify(brain, Path(tmp), execute_commands=True)
        self.assertEqual(brain.fact(REPO, "test_command", Layer.L5).attrs, before)

    def test_a_failing_command_is_refuted_and_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = brain_with(command_fact(
                "test", f'"{sys.executable}" -c "import sys; sys.exit(1)"'))
            report = verify(brain, Path(tmp), execute_commands=True)

        fact = brain.fact(REPO, "test_command", Layer.L5)
        self.assertEqual(report.refuted, 1)
        self.assertIs(fact.env.status, Status.REFUTED)
        self.assertIsNotNone(fact)  # kept, not deleted
        self.assertIn("failed", fact.env.note)

    def test_a_refuted_claim_can_never_reach_a_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = brain_with(command_fact(
                "test", f'"{sys.executable}" -c "import sys; sys.exit(1)"'))
            verify(brain, Path(tmp), execute_commands=True)
        fact = brain.fact(REPO, "test_command", Layer.L5)
        self.assertEqual(fact.env.effective_confidence(), 0.0)
        self.assertFalse(fact.env.usable())

    def test_apply_outcome_on_a_missing_fact(self):
        outcome = Outcome(Candidate("test", "nope", "x"), True, 0, 0.1)
        self.assertEqual(apply_outcome(Brain(), outcome, "abc"), "missing")


class TestRender(unittest.TestCase):
    def test_nothing_to_verify(self):
        self.assertIn("no executable claims", render(verify(Brain(), Path("."))))

    def test_failure_output_is_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = brain_with(command_fact(
                "test",
                f'"{sys.executable}" -c "import sys; print(\'kaput\'); sys.exit(2)"'))
            text = render(verify(brain, Path(tmp), execute_commands=True))
        self.assertIn("FAIL", text)
        self.assertIn("kaput", text)
        self.assertIn("refuted", text)


if __name__ == "__main__":
    unittest.main()
