from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Method
from codebrain.model import REPO, Layer
from codebrain.providers import BuildContext, build
from codebrain.extractors.semantics import (
    SemanticsProvider,
    context_of,
    terms_of,
)
from codebrain.extractors.structure_py import PythonStructureProvider


class TestHelpers(unittest.TestCase):
    def test_terms_splits_identifiers(self):
        self.assertEqual(terms_of("ChargeEndpoint"), {"charge", "endpoint"})
        self.assertEqual(terms_of("settle_invoice"), {"settle", "invoice"})

    def test_programming_vocabulary_is_dropped(self):
        self.assertEqual(terms_of("get_value"), set())
        self.assertEqual(terms_of("build_handler"), set())

    def test_english_filler_is_dropped(self):
        # Test names are sentences; without this the "domain language" of a
        # well-tested repo is "not, and, are, the, does".
        self.assertEqual(terms_of("does_not_reach_the_pack") & {"does", "not", "the"},
                         set())

    def test_context_of(self):
        self.assertEqual(context_of("payments/api.py"), "payments")
        self.assertEqual(context_of("src/payments/api.py"), "src/payments")
        self.assertIsNone(context_of("main.py"))


class TestSemantics(unittest.TestCase):
    def _build(self, files: dict[str, str]):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return build(BuildContext(root=root),
                     [PythonStructureProvider(), SemanticsProvider()]).brain

    def _domain(self) -> dict[str, str]:
        return {
            "payments/__init__.py": "",
            "payments/invoice.py": "class Invoice:\n    pass\n",
            "payments/settlement.py": (
                "from payments.invoice import Invoice\n"
                "def settle_invoice(invoice):\n    return Invoice()\n"),
            "shipping/__init__.py": "",
            "shipping/parcel.py": "class Parcel:\n    pass\n",
            "shipping/dispatch.py": (
                "from shipping.parcel import Parcel\n"
                "def dispatch_parcel(parcel):\n    return Parcel()\n"),
        }

    def test_contexts_come_from_cohesion(self):
        brain = self._build(self._domain())
        keys = {n.key for n in brain.nodes.values() if n.layer is Layer.L3}
        self.assertIn("payments", keys)
        self.assertIn("shipping", keys)

    def test_cohesion_is_recorded_and_drives_confidence(self):
        brain = self._build(self._domain())
        node = brain.get("L3:context:payments")
        self.assertEqual(node.attrs["cohesion"], 1.0)
        self.assertGreater(node.env.confidence, 0.9)

    def test_contexts_are_derived_never_extracted(self):
        # A directory that coheres may cohere for non-domain reasons.
        brain = self._build(self._domain())
        self.assertIs(brain.get("L3:context:payments").env.method, Method.DERIVED)

    def test_a_lone_module_is_not_a_context(self):
        brain = self._build({"solo/only.py": "def f():\n    pass\n"})
        self.assertIsNone(brain.get("L3:context:solo"))

    def test_test_directories_are_not_bounded_contexts(self):
        brain = self._build({**self._domain(),
                             "tests/test_a.py": "def test_one():\n    pass\n",
                             "tests/test_b.py": "def test_two():\n    pass\n"})
        self.assertIsNone(brain.get("L3:context:tests"))

    def test_vocabulary_excludes_test_prose(self):
        brain = self._build({
            **self._domain(),
            "tests/test_x.py": (
                "def test_a_refuted_claim_can_never_reach_a_pack():\n    pass\n"),
            "tests/test_y.py": (
                "def test_a_refuted_claim_can_never_reach_a_store():\n    pass\n"),
        })
        terms = {t["term"] for t in
                 brain.fact(REPO, "ubiquitous_language", Layer.L3).value}
        self.assertIn("invoice", terms)
        for noise in ("refuted", "claim", "never", "reach"):
            self.assertNotIn(noise, terms)

    def test_entity_candidates_exclude_test_classes(self):
        brain = self._build({**self._domain(),
                             "tests/test_a.py": "class TestInvoice:\n    pass\n"})
        names = {e["name"] for e in
                 brain.fact(REPO, "entity_candidates", Layer.L3).value}
        self.assertNotIn("TestInvoice", names)

    def test_cross_context_dependencies_are_recorded(self):
        files = self._domain()
        files["shipping/dispatch.py"] += (
            "from payments.invoice import Invoice\n"
            "from payments.settlement import settle_invoice\n")
        brain = self._build(files)
        found = brain.fact("L3:context:shipping", "depends_on:payments", Layer.L3)
        self.assertIsNotNone(found)
        self.assertGreaterEqual(found.value["imports"], 2)

    def test_the_llm_gap_is_declared(self):
        brain = self._build(self._domain())
        gap = brain.fact(REPO, "semantics_coverage_gap", Layer.L3)
        self.assertIsNotNone(gap)
        self.assertIn("business rules", gap.value["missing"])
        self.assertIn("no language-model pass", gap.value["reason"])

    def test_provider_declines_without_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(root=Path(tmp))
            self.assertFalse(SemanticsProvider().applies(ctx))

    def test_provider_is_marked_derivative(self):
        # Otherwise it is asked whether it applies against an empty Brain,
        # answers no, and the whole layer silently disappears.
        self.assertTrue(SemanticsProvider().derivative)

    def test_reproducible(self):
        files = self._domain()
        one, two = self._build(files), self._build(files)
        self.assertEqual({r.id for r in one.records()}, {r.id for r in two.records()})


if __name__ == "__main__":
    unittest.main()
