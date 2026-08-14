from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Envelope, Evidence, Method, Status
from codebrain.model import REPO, Brain, Edge, Fact, Layer, Node, new_brain
from codebrain.store import BrainNotFound, append_memory, exists, layer_path, load, save


def node(key, method=Method.EXTRACTED, conf=None, ts="2026-01-01T00:00:00+00:00", **attrs):
    return Node(layer=Layer.L1, kind="symbol", key=key, name=key.rsplit("#", 1)[-1],
                env=Envelope.make(method, source="t", ts=ts, confidence=conf,
                                  evidence=(Evidence(path=key.split("#")[0]),)),
                attrs=attrs)


def sample_brain() -> Brain:
    b = new_brain(repo="demo", as_of="abc1234", branch="main")
    b.add(node("pay/api.py#charge", visibility="public"))
    b.add(node("pay/api.py#refund"))
    b.add(Edge(layer=Layer.L1, kind="calls",
               src="L1:symbol:pay/api.py#charge", dst="L1:symbol:pay/api.py#refund",
               env=Envelope.make(Method.EXTRACTED, source="t",
                                 ts="2026-01-01T00:00:00+00:00")))
    b.add(Fact(layer=Layer.L5, subject=REPO, predicate="test_command", value="make test",
               env=Envelope.make(Method.OBSERVED, source="t",
                                 ts="2026-01-01T00:00:00+00:00", status=Status.FRESH,
                                 evidence=(Evidence(path="Makefile", start_line=3),))))
    return b


class TestIds(unittest.TestCase):
    def test_ids_are_readable_and_stable(self):
        self.assertEqual(node("pay/api.py#charge").id, "L1:symbol:pay/api.py#charge")
        self.assertEqual(node("pay/api.py#charge").id, node("pay/api.py#charge").id)

    def test_ids_reject_newlines(self):
        with self.assertRaises(ValueError):
            _ = node("pay/api.py\n#charge").id


class TestBrain(unittest.TestCase):
    def test_add_is_idempotent(self):
        b = Brain()
        self.assertEqual(b.add(node("a.py#x")), "added")
        self.assertEqual(b.add(node("a.py#x")), "kept")
        self.assertEqual(len(b), 1)

    def test_better_evidence_replaces_worse(self):
        b = Brain()
        b.add(node("a.py#x", Method.INFERRED, 0.6))
        self.assertEqual(b.add(node("a.py#x", Method.EXTRACTED, 0.98)), "replaced")
        self.assertIs(b.nodes["L1:symbol:a.py#x"].env.method, Method.EXTRACTED)

    def test_merge_order_does_not_matter(self):
        strong = node("a.py#x", Method.EXTRACTED, 0.98)
        weak = node("a.py#x", Method.INFERRED, 0.6)
        one, two = Brain(), Brain()
        one.extend([strong, weak])
        two.extend([weak, strong])
        self.assertEqual(one.nodes["L1:symbol:a.py#x"].env.method,
                         two.nodes["L1:symbol:a.py#x"].env.method)

    def test_merge_two_brains(self):
        a, b = Brain(), Brain()
        a.add(node("a.py#x"))
        b.add(node("b.py#y"))
        report = a.merge(b)
        self.assertEqual(report.added, 1)
        self.assertEqual(len(a), 2)

    def test_usable_excludes_refuted(self):
        b = Brain()
        good = node("a.py#x")
        bad = node("b.py#y")
        bad.env = bad.env.demote("proved wrong")
        b.extend([good, bad])
        self.assertEqual({r.id for r in b.usable()}, {good.id})

    def test_touch_marks_neighbourhood_stale(self):
        b = sample_brain()
        touched = b.touch("pay/", reason="edited by agent")
        self.assertEqual(touched, 2)
        self.assertIs(b.nodes["L1:symbol:pay/api.py#charge"].env.status, Status.STALE)

    def test_validate_flags_dangling_edges(self):
        b = Brain()
        b.add(Edge(layer=Layer.L1, kind="calls", src="L1:symbol:nope", dst="L1:symbol:nah",
                   env=Envelope.make(Method.EXTRACTED, source="t")))
        problems = b.validate()
        self.assertTrue(any("unknown src" in p for p in problems))

    def test_validate_flags_unattributed_records(self):
        b = Brain()
        b.add(Node(layer=Layer.L0, kind="file", key="a.py",
                   env=Envelope.make(Method.EXTRACTED, source="")))
        self.assertTrue(any("unattributable" in p for p in b.validate()))


class TestStore(unittest.TestCase):
    def test_roundtrip(self):
        original = sample_brain()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / ".brain"
            save(original, out)
            self.assertTrue(exists(out))
            restored = load(out)

        self.assertEqual(len(restored), len(original))
        self.assertEqual(restored.manifest.repo, "demo")
        self.assertEqual(restored.manifest.as_of, "abc1234")
        self.assertEqual({r.id for r in restored.records()},
                         {r.id for r in original.records()})
        fact = restored.fact(REPO, "test_command")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.value, "make test")
        self.assertIs(fact.env.method, Method.OBSERVED)

    def test_output_is_byte_identical_across_saves(self):
        # Principle v: the Brain is reviewed on pull requests. If a rebuild
        # reorders lines, every diff is noise and nobody reads them.
        brain = sample_brain()
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            save(brain, a)
            save(load(a), b)
            for layer in (Layer.L1, Layer.L5):
                self.assertEqual(layer_path(a, layer).read_bytes(),
                                 layer_path(b, layer).read_bytes())

    def test_records_are_one_per_line_and_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / ".brain"
            save(sample_brain(), out)
            lines = layer_path(out, Layer.L1).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)  # 2 nodes + 1 edge
            # Ordered by record id, not by raw line text: the id is the stable
            # thing a reviewer reads down the left of a diff.
            ids = [json.loads(line)["id"] for line in lines]
            self.assertEqual(ids, sorted(ids))

    def test_empty_layers_leave_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / ".brain"
            save(sample_brain(), out)
            self.assertFalse(layer_path(out, Layer.L3).exists())

    def test_derived_indexes_are_gitignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / ".brain"
            save(sample_brain(), out)
            ignored = (out / ".gitignore").read_text(encoding="utf-8")
            self.assertIn("claims.db", ignored)

    def test_missing_brain_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(BrainNotFound):
                load(Path(tmp) / "nope")

    def test_corrupt_line_names_the_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / ".brain"
            save(sample_brain(), out)
            path = layer_path(out, Layer.L1)
            path.write_text(path.read_text(encoding="utf-8") + "{not json\n",
                            encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load(out)
            self.assertIn("l1.jsonl:4", str(caught.exception))

    def test_memory_append_does_not_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / ".brain"
            save(sample_brain(), out)
            lesson = Fact(layer=Layer.L7, subject=REPO, predicate="session:1",
                          value="limiter store is Redis",
                          env=Envelope.make(Method.ASSERTED, source="agent"))
            self.assertEqual(append_memory(out, [lesson]), 1)
            self.assertEqual(len(load(out).by_layer(Layer.L7)), 1)

    def test_memory_append_rejects_other_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / ".brain"
            save(sample_brain(), out)
            with self.assertRaises(ValueError):
                append_memory(out, [node("a.py#x")])


if __name__ == "__main__":
    unittest.main()
