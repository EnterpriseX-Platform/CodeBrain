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


class TestEmptyBrain(unittest.TestCase):
    def test_renders_without_raising(self):
        # A Brain from a repo with nothing recognisable must still produce a
        # readable page rather than a traceback.
        text = render(Brain())
        self.assertIn("#", text)
        self.assertIn("does not know", text)


if __name__ == "__main__":
    unittest.main()
