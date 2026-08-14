from __future__ import annotations

import unittest

from codebrain.envelope import Envelope, Evidence, Method, Status
from codebrain.memory import (
    HALF_LIFE_COMMITS,
    MEMORY_FLOOR,
    Session,
    dispute,
    disputes_for,
    effective_memory_confidence,
    from_session,
    lessons_for,
    memory_weight,
    remember,
    slug,
    stats,
)
from codebrain.model import REPO, Brain, Fact, Layer, Node, new_brain


def session(**kw) -> Session:
    base = dict(session_id="s1", task="add rate limiting to payments",
                files=("payments/api.py",), commit="abc1234", commits_now=100)
    base.update(kw)
    return Session(**base)


class TestSlug(unittest.TestCase):
    def test_stable_and_readable(self):
        self.assertEqual(slug("The limiter store is Redis"),
                         "the-limiter-store-is-redis")

    def test_same_text_gives_the_same_id(self):
        # So recording a lesson twice updates it instead of duplicating it.
        self.assertEqual(slug("a lesson"), slug("a lesson"))

    def test_long_text_is_truncated_but_still_unique(self):
        a, b = slug("x" * 200 + "one"), slug("x" * 200 + "two")
        self.assertNotEqual(a, b)
        self.assertLessEqual(len(a), 60)


class TestSessionIngestion(unittest.TestCase):
    def test_what_the_session_did_is_extracted(self):
        records = from_session(session())
        outcome = next(r for r in records if isinstance(r, Fact)
                       and r.predicate.startswith("task_outcome"))
        # The files an agent edited are a fact about the session, not an opinion.
        self.assertIs(outcome.env.method, Method.EXTRACTED)
        self.assertEqual(outcome.value["files"], ["payments/api.py"])

    def test_what_the_session_concluded_is_inferred(self):
        records = from_session(session(lessons=("the limiter store is Redis",)))
        lesson = next(r for r in records if isinstance(r, Fact)
                      and r.predicate.startswith("lesson"))
        self.assertIs(lesson.env.method, Method.INFERRED)
        self.assertIn("not verified", lesson.env.note)

    def test_a_session_node_is_recorded(self):
        node = next(r for r in from_session(session()) if isinstance(r, Node))
        self.assertEqual(node.key, "s1")
        self.assertEqual(node.layer, Layer.L7)

    def test_resolved_unknowns_and_open_questions(self):
        records = from_session(session(
            resolved=(("redis or in-process?", "Redis"),),
            questions=("does the limiter need to be distributed?",)))
        kinds = {r.predicate.split(":", 1)[0] for r in records if isinstance(r, Fact)}
        self.assertIn("resolved_unknown", kinds)
        self.assertIn("open_question", kinds)

    def test_everything_lands_in_l7(self):
        for record in from_session(session(lessons=("x",), questions=("y",))):
            self.assertIs(record.layer, Layer.L7)

    def test_recording_commit_count_enables_decay(self):
        outcome = next(r for r in from_session(session()) if isinstance(r, Fact))
        self.assertEqual(outcome.attrs["recorded_at_commits"], 100)


class TestDecay(unittest.TestCase):
    def _fact(self, recorded: int) -> Fact:
        return Fact(layer=Layer.L7, subject=REPO, predicate="lesson:x", value="x",
                    env=Envelope.make(Method.INFERRED, source="agent",
                                      evidence=(Evidence(path="."),)),
                    attrs={"recorded_at_commits": recorded})

    def test_fresh_memory_is_undecayed(self):
        self.assertEqual(memory_weight(self._fact(100), 100), 1.0)

    def test_one_half_life_halves_it(self):
        self.assertAlmostEqual(
            memory_weight(self._fact(0), HALF_LIFE_COMMITS), 0.5, places=6)

    def test_decay_is_read_time_and_does_not_mutate(self):
        # Decay that wrote back would make two reads of the same Brain disagree.
        fact = self._fact(0)
        before = fact.env.confidence
        memory_weight(fact, HALF_LIFE_COMMITS * 3)
        self.assertEqual(fact.env.confidence, before)

    def test_missing_stamp_does_not_decay(self):
        fact = self._fact(0)
        fact.attrs = {}
        self.assertEqual(memory_weight(fact, 5000), 1.0)

    def test_effective_confidence_combines_status_and_age(self):
        fact = self._fact(0)
        self.assertLess(effective_memory_confidence(fact, HALF_LIFE_COMMITS),
                        fact.env.confidence)


class TestRelevance(unittest.TestCase):
    def _brain(self) -> Brain:
        brain = new_brain(as_of="abc1234")
        brain.extend(from_session(session(lessons=("the limiter store is Redis",))))
        return brain

    def test_repo_wide_lessons_are_always_relevant(self):
        found = lessons_for(self._brain(), {"anything.py"}, 100)
        self.assertTrue(any("Redis" in str(f.value) for f, _ in found))

    def test_outcomes_are_relevant_to_the_files_they_touched(self):
        found = lessons_for(self._brain(), {"payments/api.py"}, 100)
        self.assertTrue(any(f.predicate.startswith("task_outcome") for f, _ in found))

    def test_faded_memory_is_dropped(self):
        brain = self._brain()
        far = 100 + HALF_LIFE_COMMITS * 8
        self.assertEqual(lessons_for(brain, {"payments/api.py"}, far), [])

    def test_refuted_memory_never_surfaces(self):
        brain = self._brain()
        for fact in brain.facts.values():
            fact.env = fact.env.demote("wrong")
        self.assertEqual(lessons_for(brain, {"payments/api.py"}, 100), [])

    def test_strongest_first(self):
        found = lessons_for(self._brain(), {"payments/api.py"}, 100)
        weights = [w for _, w in found]
        self.assertEqual(weights, sorted(weights, reverse=True))


class TestRemember(unittest.TestCase):
    def test_agent_memory_is_inferred(self):
        self.assertIs(remember("x", commits_now=1).env.method, Method.INFERRED)

    def test_human_memory_is_asserted(self):
        self.assertIs(remember("x", commits_now=1, human=True).env.method,
                      Method.ASSERTED)

    def test_a_lesson_can_be_about_a_file(self):
        fact = remember("careful here", about="L0:file:pay/api.py", commits_now=1)
        self.assertEqual(fact.subject, "L0:file:pay/api.py")


class TestDispute(unittest.TestCase):
    def _brain(self) -> Brain:
        brain = new_brain(as_of="abc1234")
        brain.add(Fact(layer=Layer.L5, subject=REPO, predicate="test_command",
                       value="make test",
                       env=Envelope.make(Method.EXTRACTED, source="operations",
                                         status=Status.FRESH,
                                         evidence=(Evidence(path="Makefile"),))))
        return brain

    def test_an_agent_may_dispute_but_not_demote(self):
        # A model that can overrule the AST by asserting harder can poison
        # every session downstream, confidently.
        brain = self._brain()
        record, outcome = dispute(brain, "L5:fact:|test_command", "it does not work")
        self.assertEqual(outcome, "disputed")
        self.assertIs(brain.fact(REPO, "test_command", Layer.L5).env.status,
                      Status.FRESH)
        self.assertIs(record.env.method, Method.INFERRED)

    def test_a_human_may_overrule(self):
        brain = self._brain()
        record, outcome = dispute(brain, "L5:fact:|test_command", "wrong",
                                  human=True)
        self.assertEqual(outcome, "overruled")
        self.assertIs(brain.fact(REPO, "test_command", Layer.L5).env.status,
                      Status.REFUTED)
        self.assertIs(record.env.method, Method.ASSERTED)

    def test_disputing_a_missing_record(self):
        _, outcome = dispute(Brain(), "L5:fact:|nope", "x")
        self.assertEqual(outcome, "missing")

    def test_disputes_are_findable_by_target(self):
        brain = self._brain()
        record, _ = dispute(brain, "L5:fact:|test_command", "broken")
        brain.add(record)
        self.assertEqual([f.id for f in disputes_for(brain, ["L5:fact:|test_command"])],
                         [record.id])


class TestStats(unittest.TestCase):
    def test_counts_by_kind(self):
        brain = new_brain()
        brain.extend(from_session(session(lessons=("a",), questions=("b",))))
        summary = stats(brain, 100)
        self.assertEqual(summary["sessions"], 1)
        self.assertIn("lesson", summary["by_kind"])

    def test_faded_are_counted(self):
        brain = new_brain()
        brain.extend(from_session(session()))
        self.assertGreater(stats(brain, 100 + HALF_LIFE_COMMITS * 9)["faded"], 0)

    def test_floor_is_meaningful(self):
        self.assertGreater(MEMORY_FLOOR, 0.0)
        self.assertLess(MEMORY_FLOOR, 0.5)


if __name__ == "__main__":
    unittest.main()
