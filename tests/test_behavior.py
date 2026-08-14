from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Method
from codebrain.model import REPO, Layer
from codebrain.providers import BuildContext, build
from codebrain.extractors.behavior import BehaviorProvider


class TestBehavior(unittest.TestCase):
    def _build(self, files: dict[str, str]):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return build(BuildContext(root=root), [BehaviorProvider()]).brain

    # -- routes ------------------------------------------------------------

    def test_flask_style_route(self):
        brain = self._build({"api.py":
                             "@app.route('/v1/charges', methods=['POST', 'GET'])\n"
                             "def charge():\n    pass\n"})
        post = brain.get("L2:route:POST /v1/charges")
        self.assertIsNotNone(post)
        self.assertIsNotNone(brain.get("L2:route:GET /v1/charges"))
        self.assertEqual(post.attrs["handler"], "charge")

    def test_fastapi_style_verb_decorator(self):
        brain = self._build({"api.py": "@router.get('/health')\ndef health():\n    pass\n"})
        self.assertIsNotNone(brain.get("L2:route:GET /health"))

    def test_route_is_linked_to_its_handler(self):
        brain = self._build({"api.py": "@router.post('/x')\ndef handler():\n    pass\n"})
        edges = {(e.src, e.dst) for e in brain.edges.values() if e.kind == "handled_by"}
        self.assertIn(("L2:route:POST /x", "L1:symbol:api.py#handler"), edges)

    def test_a_decorator_with_no_path_is_not_a_route(self):
        brain = self._build({"api.py": "@app.route\ndef charge():\n    pass\n"})
        self.assertEqual([n for n in brain.nodes.values() if n.kind == "route"], [])

    def test_express_style_route(self):
        brain = self._build({"server.js":
                             "app.get('/users', (req, res) => res.send(1));\n"})
        node = brain.get("L2:route:GET /users")
        self.assertIsNotNone(node)
        # A scanner is not a parser, and the envelope says so.
        self.assertIs(node.env.method, Method.DERIVED)

    def test_a_route_written_inside_a_string_is_not_a_route(self):
        brain = self._build({"server.js":
                             "const doc = \"app.get('/fake', h)\";\n"})
        self.assertEqual([n for n in brain.nodes.values() if n.kind == "route"], [])

    # -- entrypoints and jobs ----------------------------------------------

    def test_main_guard_is_an_entrypoint(self):
        brain = self._build({"cli.py": "def main():\n    pass\n\n"
                                       "if __name__ == '__main__':\n    main()\n"})
        self.assertIsNotNone(brain.get("L2:entrypoint:cli.py"))

    def test_a_module_without_a_main_guard_is_not_an_entrypoint(self):
        brain = self._build({"lib.py": "def helper():\n    pass\n"})
        self.assertEqual([n for n in brain.nodes.values()
                          if n.kind == "entrypoint"], [])

    def test_celery_task(self):
        brain = self._build({"jobs.py": "@celery.task\ndef nightly():\n    pass\n"})
        node = brain.get("L2:job:jobs.py#nightly")
        self.assertIsNotNone(node)
        self.assertEqual(node.attrs["trigger"], "task")

    # -- config and data ---------------------------------------------------

    def test_environment_reads_are_captured(self):
        brain = self._build({"conf.py":
                             "import os\n"
                             "def load():\n"
                             "    a = os.getenv('STRIPE_KEY')\n"
                             "    b = os.environ['DB_URL']\n"
                             "    c = os.environ.get('TIMEOUT')\n"})
        names = brain.fact(REPO, "environment_variables", Layer.L2).value
        self.assertEqual(names, ["DB_URL", "STRIPE_KEY", "TIMEOUT"])

    def test_env_reads_are_attributed_to_the_function(self):
        brain = self._build({"conf.py":
                             "import os\ndef load():\n    return os.getenv('KEY')\n"})
        found = brain.fact("L1:symbol:conf.py#load", "reads_env", Layer.L2)
        self.assertEqual(found.value, ["KEY"])

    def test_data_stores_and_network_from_imports(self):
        brain = self._build({"db.py": "import sqlalchemy\nimport requests\n"})
        self.assertIn("sqlalchemy", brain.fact(REPO, "data_stores", Layer.L2).value)
        self.assertIn("requests", brain.fact(REPO, "outbound_network", Layer.L2).value)

    def test_data_store_claims_are_derived_not_extracted(self):
        # An import proves the dependency exists, not that it is used at runtime.
        brain = self._build({"db.py": "import sqlalchemy\n"})
        self.assertIs(brain.fact(REPO, "data_stores", Layer.L2).env.method,
                      Method.DERIVED)

    # -- honesty -----------------------------------------------------------

    def test_the_coverage_gap_is_declared(self):
        brain = self._build({"a.py": "x = 1\n"})
        gap = brain.fact(REPO, "behavior_coverage_gap", Layer.L2)
        self.assertIsNotNone(gap)
        self.assertIn("Django urlpatterns", gap.value["misses"])
        self.assertIn("not proof", gap.value["impact"])

    def test_summary_counts(self):
        brain = self._build({"api.py": "@app.get('/a')\ndef a():\n    pass\n",
                             "m.py": "if __name__ == '__main__':\n    pass\n"})
        summary = brain.fact(REPO, "runtime_summary", Layer.L2).value
        self.assertEqual(summary["routes"], 1)
        self.assertEqual(summary["entrypoints"], 1)

    def test_unparsable_python_does_not_break_the_provider(self):
        brain = self._build({"broken.py": "def (((\n", "ok.py": "x = 1\n"})
        self.assertIsNotNone(brain.fact(REPO, "runtime_summary", Layer.L2))

    def test_reproducible(self):
        files = {"api.py": "@app.get('/a')\ndef a():\n    pass\n"}
        one, two = self._build(files), self._build(files)
        self.assertEqual({r.id for r in one.records()}, {r.id for r in two.records()})


if __name__ == "__main__":
    unittest.main()
