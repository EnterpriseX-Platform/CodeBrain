from __future__ import annotations

import unittest

from codebrain.atlas import human_bytes, provenance, render
from codebrain.envelope import Envelope, Evidence, Method, Status
from codebrain.model import REPO, Brain, Fact, Layer, Node, new_brain


def env(method=Method.EXTRACTED, **kw):
    return Envelope.make(method, source="t", evidence=(Evidence(path="."),), **kw)


def fact(layer, predicate, value, method=Method.EXTRACTED, attrs=None, **kw):
    return Fact(layer=layer, subject=REPO, predicate=predicate, value=value,
                env=env(method, **kw), attrs=attrs or {})


def populated() -> Brain:
    brain = new_brain(repo="demo", as_of="abc1234def", branch="main")
    brain.extend([
        fact(Layer.L0, "repo_name", "demo"),
        fact(Layer.L0, "file_count", 42),
        fact(Layer.L0, "total_bytes", 2048),
        fact(Layer.L0, "language_mix", {"Python": 30, "Markdown": 12}),
        fact(Layer.L0, "primary_language", "Python", Method.DERIVED),
        fact(Layer.L5, "test_command", "make test", Method.DERIVED,
             attrs={"source": "makefile",
                    "alternatives": [{"command": "pytest", "source": "pyproject.toml"}]},
             note="chosen from 2 candidate(s); unverified until executed"),
        fact(Layer.L1, "python_summary", {"modules": 5, "symbols": 40, "call_edges": 12}),
        fact(Layer.L4, "hotspots", [{"path": "a.py", "commits": 9,
                                     "lines_changed": 300, "score": 155.9}]),
        fact(Layer.L4, "top_authors", [{"author": "Ada", "commits": 30}]),
    ])
    brain.add(Fact(layer=Layer.L4, subject="L0:file:a.py", predicate="ownership",
                   value={"primary_author": "Ada", "share": 1.0, "distinct_authors": 1},
                   env=env(Method.DERIVED)))
    brain.add(Node(layer=Layer.L1, kind="module", key="a.py", env=env()))
    return brain


class TestHelpers(unittest.TestCase):
    def test_human_bytes(self):
        self.assertEqual(human_bytes(512), "512 B")
        self.assertEqual(human_bytes(2048), "2.0 KB")

    def test_provenance_distinguishes_verified_from_guessed(self):
        self.assertIn("OBSERVED", provenance(env().promote("abc1234")))
        self.assertIn("inferred, not verified", provenance(env(Method.DERIVED)))
        self.assertIn("REFUTED", provenance(env().demote("no")))
        self.assertIn("by a human", provenance(env(Method.ASSERTED)))


class TestAtlas(unittest.TestCase):
    def setUp(self):
        self.text = render(populated())

    def test_names_the_repo_and_commit(self):
        self.assertIn("# demo", self.text)
        self.assertIn("abc1234d", self.text)

    def test_says_it_is_generated(self):
        self.assertIn("Do not edit", self.text)

    def test_shows_how_to_run_the_tests(self):
        self.assertIn("make test", self.text)

    def test_flags_unverified_commands_rather_than_stating_them_as_fact(self):
        self.assertIn("never been executed", self.text)
        self.assertIn("inferred, not verified", self.text)

    def test_keeps_the_rejected_candidates_visible(self):
        self.assertIn("pytest", self.text)
        self.assertIn("considered", self.text)

    def test_surfaces_hotspots(self):
        self.assertIn("Hotspots", self.text)
        self.assertIn("a.py", self.text)

    def test_surfaces_bus_factor_risk(self):
        self.assertIn("Single-author files", self.text)
        self.assertIn("Ada", self.text)

    def test_states_its_own_gaps(self):
        # The section that stops the Atlas being trusted where it is weakest.
        self.assertIn("does not know", self.text)
        self.assertIn("Layers not yet populated", self.text)

    def test_counts_inferred_claims_in_the_gaps(self):
        self.assertIn("inferred rather than extracted", self.text)

    def test_is_valid_markdown_structure(self):
        headings = [ln for ln in self.text.splitlines() if ln.startswith("#")]
        self.assertEqual(headings[0], "# demo")
        self.assertTrue(any(h.startswith("## ") for h in headings))

    def test_deterministic(self):
        self.assertEqual(render(populated()), render(populated()))


def all_layers() -> Brain:
    """A Brain with something in every layer the Atlas is meant to show."""
    brain = populated()
    brain.extend([
        # L2 behavior
        Node(layer=Layer.L2, kind="route", key="POST /v1/charges",
             name="POST /v1/charges", env=env(),
             attrs={"handler": "charge", "module": "pay/api.py", "method": "POST"}),
        Node(layer=Layer.L2, kind="entrypoint", key="cli.py", name="cli.py",
             env=env(), attrs={"kind": "__main__"}),
        Node(layer=Layer.L2, kind="job", key="jobs.py#nightly", name="nightly",
             env=env(), attrs={"trigger": "task", "module": "jobs.py"}),
        fact(Layer.L2, "environment_variables", ["DB_URL", "STRIPE_KEY"]),
        fact(Layer.L2, "data_stores", ["sqlalchemy"], Method.DERIVED),
        fact(Layer.L2, "outbound_network", ["httpx"], Method.DERIVED),
        # L3 semantics
        Node(layer=Layer.L3, kind="context", key="payments", name="payments",
             env=env(Method.DERIVED), attrs={"modules": 5, "cohesion": 0.82}),
        fact(Layer.L3, "ubiquitous_language",
             [{"term": "invoice", "modules": 4, "contexts": ["payments"]}],
             Method.DERIVED),
        fact(Layer.L3, "entity_candidates",
             [{"name": "Invoice", "module": "payments/invoice.py",
               "shared_across_modules": 4}], Method.DERIVED),
    ])
    brain.add(Fact(layer=Layer.L6, subject="L1:symbol:pay/api.py#charge",
                   predicate="public_contract",
                   value={"consumers": ["a", "b"], "consumer_count": 2,
                          "reason": "called from other modules"},
                   env=env(Method.DERIVED)))
    brain.add(Fact(layer=Layer.L6, subject="L0:file:pay/settle.py",
                   predicate="policy_zone",
                   value={"zone": "pci", "reason": "cardholder data",
                          "requires": ["@risk-eng"], "block_agents": True},
                   env=env(Method.ASSERTED)))
    brain.add(Fact(layer=Layer.L6, subject="L0:file:pay/api.py",
                   predicate="untested_churn",
                   value={"commits": 9, "reason": "no test reaches it"},
                   env=env(Method.DERIVED)))
    return brain


class TestAllLayersReachTheAtlas(unittest.TestCase):
    """The Atlas was written at P1 and three layers were added after it. A
    human-facing artifact that silently omits half the Brain is worse than one
    that admits it knows nothing."""

    def setUp(self):
        self.text = render(all_layers())

    def test_routes_are_shown(self):
        self.assertIn("What it does when it runs", self.text)
        self.assertIn("POST /v1/charges", self.text)

    def test_entrypoints_and_jobs_are_shown(self):
        self.assertIn("Entrypoints", self.text)
        self.assertIn("Background work", self.text)
        self.assertIn("nightly", self.text)

    def test_configuration_and_external_surface(self):
        self.assertIn("STRIPE_KEY", self.text)
        self.assertIn("sqlalchemy", self.text)
        self.assertIn("httpx", self.text)

    def test_bounded_contexts_with_cohesion(self):
        self.assertIn("What it means", self.text)
        self.assertIn("payments", self.text)
        self.assertIn("82%", self.text)

    def test_contexts_are_labelled_as_candidates_not_a_domain_model(self):
        self.assertIn("not a domain model anyone has agreed to", self.text)

    def test_ubiquitous_language_and_entities(self):
        self.assertIn("Ubiquitous language", self.text)
        self.assertIn("invoice", self.text)
        self.assertIn("Invoice", self.text)

    def test_public_contracts(self):
        self.assertIn("What must not break", self.text)
        self.assertIn("Public contracts", self.text)

    def test_declared_policy_zones_lead_and_say_they_block(self):
        self.assertIn("Declared policy zones", self.text)
        self.assertIn("pci", self.text)
        self.assertIn("block", self.text)
        self.assertLess(self.text.index("Declared policy zones"),
                        self.text.index("Public contracts"))

    def test_untested_churn_is_declared_as_a_gap(self):
        self.assertIn("no test reaches them", self.text)

    def test_behavior_and_semantics_gaps_are_declared(self):
        brain = all_layers()
        brain.extend([
            fact(Layer.L2, "behavior_coverage_gap",
                 {"misses": ["Django urlpatterns"],
                  "impact": "a route count of zero is not proof of no HTTP surface"}),
            fact(Layer.L3, "semantics_coverage_gap",
                 {"impact": "contexts are candidates read off imports",
                  "reason": "no language-model pass has run"}),
        ])
        text = render(brain)
        self.assertIn("Django urlpatterns", text)
        self.assertIn("no language-model pass", text)

    def test_sections_are_omitted_when_a_layer_is_empty(self):
        # A repo with no routes must not get an empty "What it does" heading.
        text = render(populated())
        self.assertNotIn("What it does when it runs", text)
        self.assertNotIn("What must not break", text)

    def test_still_deterministic(self):
        self.assertEqual(render(all_layers()), render(all_layers()))


class TestEmptyBrain(unittest.TestCase):
    def test_renders_without_raising(self):
        # A Brain from a repo with nothing recognisable must still produce a
        # readable page rather than a traceback.
        text = render(Brain())
        self.assertIn("#", text)
        self.assertIn("does not know", text)


if __name__ == "__main__":
    unittest.main()
