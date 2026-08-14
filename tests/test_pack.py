from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Envelope, Evidence, Method, Status
from codebrain.model import REPO, Brain, Edge, Fact, Layer, Node, new_brain
from codebrain.pack import (
    DEFAULT_BUDGET,
    Compiler,
    Item,
    brief,
    compile_pack,
    estimate_tokens,
    is_test_path,
    mentioned_paths,
    split_identifier,
    tokenize,
)


def env(method=Method.EXTRACTED, path="a.py", line=None, **kw):
    return Envelope.make(method, source="t", status=Status.FRESH,
                         evidence=(Evidence(path=path, start_line=line),), **kw)


def sym(module: str, name: str, **attrs) -> Node:
    return Node(layer=Layer.L1, kind="symbol", key=f"{module}#{name}", name=name,
                env=env(path=module, line=10),
                attrs={"symbol_kind": "function", "module": module, **attrs})


def demo_brain() -> Brain:
    brain = new_brain(repo="demo", as_of="abc1234", branch="main")
    brain.extend([
        Node(layer=Layer.L0, kind="file", key="pay/api.py", env=env(path="pay/api.py"),
             attrs={"text": True, "language": "Python"}),
        Node(layer=Layer.L0, kind="file", key="pay/settle.py",
             env=env(path="pay/settle.py"), attrs={"text": True}),
        Node(layer=Layer.L0, kind="file", key="tests/test_pay.py",
             env=env(path="tests/test_pay.py"), attrs={"text": True}),
        sym("pay/api.py", "charge_endpoint"),
        sym("pay/api.py", "refund_endpoint"),
        sym("pay/settle.py", "settle"),
        sym("tests/test_pay.py", "test_charge"),
        # Deliberately not named after anything in the task: a test symbol that
        # matches the task text becomes an anchor, and anchors are excluded from
        # blast radius, so it could never exercise the collapsing behaviour.
        sym("tests/test_pay.py", "test_flow"),
    ])
    brain.extend([
        Edge(layer=Layer.L1, kind="calls", src="L1:symbol:pay/settle.py#settle",
             dst="L1:symbol:pay/api.py#charge_endpoint", env=env()),
        Edge(layer=Layer.L1, kind="calls", src="L1:symbol:tests/test_pay.py#test_charge",
             dst="L1:symbol:pay/api.py#charge_endpoint", env=env()),
        Edge(layer=Layer.L1, kind="calls", src="L1:symbol:tests/test_pay.py#test_flow",
             dst="L1:symbol:pay/api.py#charge_endpoint", env=env()),
        Edge(layer=Layer.L4, kind="changed_with", src="L0:file:pay/api.py",
             dst="L0:file:pay/settle.py", env=env(Method.DERIVED),
             attrs={"commits": 7, "strength": 0.7}),
    ])
    brain.extend([
        Fact(layer=Layer.L5, subject=REPO, predicate="test_command", value="make test",
             env=env(Method.DERIVED, path="Makefile"), attrs={"source": "makefile"}),
        Fact(layer=Layer.L6, subject="L0:file:pay/api.py", predicate="requires_review",
             value={"owners": ["@risk-eng"], "pattern": "/pay/"},
             env=env(Method.DERIVED, path="pay/api.py")),
        Fact(layer=Layer.L6, subject="L0:file:pay/api.py", predicate="danger_zone",
             value={"rank": 1, "commits": 31, "reason": "high churn"},
             env=env(Method.DERIVED, path="pay/api.py")),
        Fact(layer=Layer.L4, subject="L0:file:pay/api.py", predicate="churn",
             value={"commits": 31, "insertions": 400, "deletions": 120,
                    "window_commits": 500},
             env=env(path="pay/api.py")),
        Fact(layer=Layer.L1, subject=REPO, predicate="typescript_coverage_gap",
             value={"call_graph": False, "impact": "blast radius incomplete"},
             env=env(path=".")),
    ])
    return brain


class TestTokenizing(unittest.TestCase):
    def test_split_identifier(self):
        self.assertEqual(split_identifier("charge_endpoint"), {"charge", "endpoint"})
        self.assertIn("endpoint", split_identifier("chargeEndpoint"))
        self.assertIn("http", split_identifier("HTTPServer"))

    def test_tokenize_drops_filler(self):
        terms = tokenize("please add rate limiting to the payments API")
        self.assertIn("rate", terms)
        self.assertIn("payments", terms)
        for filler in ("please", "add", "the", "to"):
            self.assertNotIn(filler, terms)

    def test_tokenize_splits_identifiers_in_the_task(self):
        self.assertIn("charge", tokenize("fix charge_endpoint"))

    def test_mentioned_paths(self):
        self.assertIn("pay/api.py", mentioned_paths("edit pay/api.py please"))

    def test_is_test_path(self):
        for path in ("tests/test_x.py", "src/__tests__/a.ts", "a/b_test.py",
                     "web/Button.spec.tsx"):
            self.assertTrue(is_test_path(path), path)
        for path in ("src/pay/api.py", "latest/contest.py"):
            self.assertFalse(is_test_path(path), path)

    def test_estimate_tokens_is_never_zero(self):
        self.assertGreaterEqual(estimate_tokens("x"), 1)


class TestAnchors(unittest.TestCase):
    def setUp(self):
        self.brain = demo_brain()
        self.compiler = Compiler(self.brain)

    def test_finds_the_named_symbol(self):
        top = self.compiler.score_anchors("fix charge_endpoint", limit=3)
        self.assertEqual(top[0][0].id, "L1:symbol:pay/api.py#charge_endpoint")

    def test_explicit_path_wins(self):
        top = self.compiler.score_anchors("change something in pay/settle.py", limit=3)
        self.assertIn("settle", top[0][0].id)

    def test_tests_are_deprioritised_unless_asked_for(self):
        ranked = self.compiler.score_anchors("charge", limit=8)
        ids = [n.id for n, _ in ranked]
        self.assertLess(ids.index("L1:symbol:pay/api.py#charge_endpoint"),
                        ids.index("L1:symbol:tests/test_pay.py#test_charge"))

    def test_no_terms_no_anchors(self):
        self.assertEqual(self.compiler.score_anchors("the and of"), [])

    def test_scoring_is_deterministic(self):
        a = [n.id for n, _ in self.compiler.score_anchors("charge endpoint")]
        b = [n.id for n, _ in self.compiler.score_anchors("charge endpoint")]
        self.assertEqual(a, b)


class TestFacets(unittest.TestCase):
    def setUp(self):
        self.brain = demo_brain()
        self.pack = compile_pack(self.brain, "fix charge_endpoint rate limiting")
        self.facets = self.pack.by_facet()

    def test_anchors_present(self):
        self.assertTrue(self.facets["anchors"])

    def test_blast_radius_finds_the_production_caller(self):
        text = " ".join(i.text for i in self.facets["blast_radius"])
        self.assertIn("settle", text)

    def test_tests_are_collapsed_not_enumerated(self):
        text = " ".join(i.text for i in self.facets["blast_radius"])
        self.assertNotIn("test_charge", text)
        self.assertIn("test symbol(s)", text)

    def test_precedent_from_coupling(self):
        text = " ".join(i.text for i in self.facets.get("precedent", []))
        self.assertIn("settle.py", text)

    def test_constraints_surface_owners_and_danger(self):
        text = " ".join(i.text for i in self.facets["constraints"])
        self.assertIn("@risk-eng", text)
        self.assertIn("hotspot", text)

    def test_runbook_flags_that_the_command_was_never_run(self):
        text = " ".join(i.text for i in self.facets["runbook"])
        self.assertIn("make test", text)
        self.assertIn("never executed", text)

    def test_unknowns_are_included(self):
        text = " ".join(i.text for i in self.facets["unknowns"])
        self.assertIn("call graph", text)


class TestBudget(unittest.TestCase):
    def test_small_budget_keeps_the_critical_facets(self):
        # The failure this guards against: spending the whole budget on blast
        # radius and dropping the "this file is frozen" line.
        pack = compile_pack(demo_brain(), "fix charge_endpoint", budget=120)
        facets = pack.by_facet()
        self.assertTrue(facets.get("constraints") or facets.get("runbook"))
        self.assertLessEqual(pack.tokens, 400)

    def test_truncation_is_declared(self):
        pack = compile_pack(demo_brain(), "fix charge_endpoint", budget=60)
        self.assertTrue(pack.dropped)
        self.assertIn("omitted", pack.render())

    def test_generous_budget_is_not_exceeded(self):
        pack = compile_pack(demo_brain(), "fix charge_endpoint", budget=DEFAULT_BUDGET)
        self.assertLessEqual(pack.tokens, DEFAULT_BUDGET)


class TestRender(unittest.TestCase):
    def test_headline_carries_task_and_budget(self):
        text = compile_pack(demo_brain(), "fix charge_endpoint").render()
        self.assertIn("CONTEXT PACK", text)
        self.assertIn("charge_endpoint", text)
        self.assertIn("abc1234", text)

    def test_no_match_says_so_rather_than_pretending(self):
        text = compile_pack(demo_brain(), "quantum flux capacitor").render()
        self.assertIn("no anchors matched", text)
        self.assertIn("Fall back", text)

    def test_json_round_trips(self):
        payload = compile_pack(demo_brain(), "fix charge_endpoint").to_json()
        self.assertIn("facets", payload)
        self.assertIn("anchors", payload["facets"])
        self.assertGreater(payload["tokens"], 0)

    def test_deterministic(self):
        one = compile_pack(demo_brain(), "fix charge_endpoint").render()
        two = compile_pack(demo_brain(), "fix charge_endpoint").render()
        self.assertEqual(one, two)


class TestContentAnchors(unittest.TestCase):
    def test_behaviour_described_tasks_find_files_by_content(self):
        # No symbol is called "isinstance"; only reading the file finds it.
        # This is the case that made packs lose to plain search.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pay").mkdir()
            (root / "pay" / "models.py").write_text(
                "def check(x):\n    return isinstance(x, dict)\n", encoding="utf-8")
            brain = Brain()
            brain.add(Node(layer=Layer.L0, kind="file", key="pay/models.py",
                           env=env(path="pay/models.py"), attrs={"text": True}))
            compiler = Compiler(brain, root=root)
            hits = compiler.score_anchors("add hasattr checks for isinstance", limit=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0].key, "pay/models.py")

    def test_no_root_means_no_content_scan(self):
        compiler = Compiler(demo_brain(), root=None)
        self.assertEqual(compiler.content_anchors({"isinstance"}, False, 5), [])


class TestBrief(unittest.TestCase):
    def test_brief_is_short_and_useful(self):
        text = brief(demo_brain())
        self.assertIn("demo", text)
        self.assertIn("make test", text)
        self.assertLessEqual(estimate_tokens(text), 500)

    def test_brief_on_an_empty_brain_does_not_raise(self):
        self.assertIn("BRAIN", brief(Brain()))


class TestMemoryFacet(unittest.TestCase):
    def _with_memory(self, commit_count: int = 100, **kw):
        from codebrain.memory import Session, from_session

        brain = demo_brain()
        # Set once: a second commit_count fact would lose the envelope merge to
        # the first, and the test would be measuring the wrong number.
        brain.add(Fact(layer=Layer.L4, subject=REPO, predicate="commit_count",
                       value=commit_count, env=env()))
        defaults = dict(session_id="s1", task="add rate limiting to charge_endpoint",
                        files=("pay/settle.py",), commit="abc1234",
                        commits_now=100, succeeded=True)
        defaults.update(kw)
        brain.extend(from_session(Session(**defaults)))
        return brain

    def test_a_previous_session_appears_in_the_pack(self):
        pack = compile_pack(self._with_memory(), "fix charge_endpoint rate limiting")
        text = " ".join(i.text for i in pack.by_facet().get("memory", []))
        self.assertIn("previous session", text)

    def test_lessons_appear(self):
        brain = self._with_memory(lessons=("the limiter store is Redis",))
        pack = compile_pack(brain, "fix charge_endpoint rate limiting")
        text = " ".join(i.text for i in pack.by_facet().get("memory", []))
        self.assertIn("Redis", text)

    def test_agent_memory_is_labelled_inferred(self):
        brain = self._with_memory(lessons=("the limiter store is Redis",))
        pack = compile_pack(brain, "fix charge_endpoint rate limiting")
        text = " ".join(i.text for i in pack.by_facet().get("memory", []))
        self.assertIn("INFERRED", text)

    def test_memory_survives_a_tight_budget(self):
        # It is the only facet nobody can re-derive; dropping it to fit blast
        # radius would throw away what a previous session paid for.
        brain = self._with_memory(lessons=("the limiter store is Redis",))
        pack = compile_pack(brain, "fix charge_endpoint", budget=160)
        self.assertTrue(pack.by_facet().get("memory"))

    def test_faded_memory_is_not_shown(self):
        from codebrain.memory import HALF_LIFE_COMMITS

        brain = self._with_memory(commit_count=HALF_LIFE_COMMITS * 9,
                                  lessons=("ancient advice",), commits_now=0)
        pack = compile_pack(brain, "fix charge_endpoint")
        text = " ".join(i.text for i in pack.by_facet().get("memory", []))
        self.assertNotIn("ancient advice", text)


class TestMemoryBoost(unittest.TestCase):
    """Memory has to change the answer, or write-back is a note in the margin."""

    def _brain(self, past_task: str, past_files: tuple[str, ...],
               commit_count: int = 100):
        from codebrain.memory import Session, from_session

        brain = demo_brain()
        brain.add(Fact(layer=Layer.L4, subject=REPO, predicate="commit_count",
                       value=commit_count, env=env()))
        brain.extend(from_session(Session(
            session_id="s1", task=past_task, files=past_files,
            commit="abc1234", commits_now=100, succeeded=True)))
        return brain

    def test_a_prior_session_lifts_the_files_it_touched(self):
        task = "add rate limiting to the charge endpoint"
        cold = Compiler(demo_brain()).score_anchors(task, limit=8)
        warm = Compiler(self._brain(task, ("pay/settle.py",))).score_anchors(task,
                                                                             limit=8)
        cold_rank = [n.id for n, _ in cold]
        warm_rank = [n.id for n, _ in warm]
        target = "L0:file:pay/settle.py"
        self.assertIn(target, warm_rank)
        if target in cold_rank:
            self.assertLess(warm_rank.index(target), cold_rank.index(target))

    def test_an_unrelated_prior_session_does_not_boost(self):
        brain = self._brain("update the deployment documentation",
                            ("pay/settle.py",))
        boosts = Compiler(brain).memory_boost(tokenize("fix charge_endpoint"))
        self.assertEqual(boosts, {})

    def test_boost_scales_with_task_overlap(self):
        exact = Compiler(self._brain("fix charge_endpoint",
                                     ("pay/settle.py",))).memory_boost(
            tokenize("fix charge_endpoint"))
        partial = Compiler(self._brain("fix charge_endpoint in the billing service",
                                       ("pay/settle.py",))).memory_boost(
            tokenize("fix charge_endpoint"))
        self.assertGreater(exact["pay/settle.py"], partial["pay/settle.py"])

    def test_faded_memory_boosts_less(self):
        from codebrain.memory import HALF_LIFE_COMMITS

        fresh = Compiler(self._brain("fix charge_endpoint", ("pay/settle.py",))
                         ).memory_boost(tokenize("fix charge_endpoint"))
        faded = Compiler(self._brain("fix charge_endpoint", ("pay/settle.py",),
                                     commit_count=100 + HALF_LIFE_COMMITS)
                         ).memory_boost(tokenize("fix charge_endpoint"))
        self.assertLess(faded["pay/settle.py"], fresh["pay/settle.py"])
        self.assertAlmostEqual(faded["pay/settle.py"] / fresh["pay/settle.py"],
                               0.5, places=6)


class TestDisputesSurface(unittest.TestCase):
    def test_a_disputed_anchor_is_flagged_as_an_unknown(self):
        from codebrain.memory import dispute

        brain = demo_brain()
        record, _ = dispute(brain, "L1:symbol:pay/api.py#charge_endpoint",
                            "this was moved last week")
        brain.add(record)
        pack = compile_pack(brain, "fix charge_endpoint")
        text = " ".join(i.text for i in pack.by_facet().get("unknowns", []))
        self.assertIn("disputed", text)
        self.assertIn("moved last week", text)


class TestStaleness(unittest.TestCase):
    def test_stale_records_are_announced(self):
        brain = demo_brain()
        brain.touch("pay/", reason="edited")
        pack = compile_pack(brain, "fix charge_endpoint")
        self.assertTrue(any("stale" in note for note in pack.notes))


if __name__ == "__main__":
    unittest.main()
