from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Method
from codebrain.model import REPO, Edge, Layer
from codebrain.providers import BuildContext, build
from codebrain.extractors.structure_py import (
    PythonStructureProvider,
    dotted,
    resolve_relative,
    symbol_defs,
)
from codebrain.extractors.structure_ts import (
    TypeScriptStructureProvider,
    mask,
    resolve_specifier,
)

import ast


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def edges(brain, kind: str) -> set[tuple[str, str]]:
    return {(e.src, e.dst) for e in brain.edges.values()
            if isinstance(e, Edge) and e.kind == kind}


# ============================== Python ==============================


class TestPythonHelpers(unittest.TestCase):
    def test_dotted(self):
        self.assertEqual(dotted("pkg/mod.py"), "pkg.mod")
        self.assertEqual(dotted("pkg/__init__.py"), "pkg")
        self.assertEqual(dotted("pkg/sub/thing.pyi"), "pkg.sub.thing")

    def test_resolve_relative_from_a_module(self):
        # inside pkg.mod:  from .other import x  ->  pkg.other
        self.assertEqual(resolve_relative("pkg.mod", False, 1, "other"), "pkg.other")

    def test_resolve_relative_from_a_package_init(self):
        # inside pkg/__init__.py:  from .mod import x  ->  pkg.mod
        self.assertEqual(resolve_relative("pkg", True, 1, "mod"), "pkg.mod")

    def test_resolve_relative_two_levels_up(self):
        self.assertEqual(resolve_relative("a.b.c", False, 2, "d"), "a.d")

    def test_resolve_bare_relative_import(self):
        self.assertEqual(resolve_relative("pkg.mod", False, 1, None), "pkg")

    def test_symbol_defs_finds_nested_names(self):
        tree = ast.parse(
            "class A:\n"
            "    def m(self):\n"
            "        def inner(): pass\n"
            "def top(): pass\n"
        )
        found = {q: k for q, k, _ in symbol_defs(tree)}
        self.assertEqual(found["A"], "class")
        self.assertEqual(found["A.m"], "method")
        self.assertEqual(found["A.m.inner"], "function")
        self.assertEqual(found["top"], "function")


class TestPythonStructure(unittest.TestCase):
    def _build_result(self, files: dict[str, str]):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, text in files.items():
            write(root, rel, text)
        return build(BuildContext(root=root), [PythonStructureProvider()])

    def _build(self, files: dict[str, str]):
        return self._build_result(files).brain

    def test_symbols_and_defines(self):
        brain = self._build({"a.py": "def foo():\n    pass\n\nclass Bar:\n    def baz(self): pass\n"})
        self.assertIsNotNone(brain.get("L1:symbol:a.py#foo"))
        self.assertIsNotNone(brain.get("L1:symbol:a.py#Bar.baz"))
        self.assertEqual(brain.get("L1:symbol:a.py#Bar.baz").attrs["symbol_kind"], "method")
        self.assertIn(("L1:module:a.py", "L1:symbol:a.py#foo"), edges(brain, "defines"))

    def test_same_module_calls_are_extracted(self):
        brain = self._build({"a.py": "def helper(): pass\ndef caller():\n    helper()\n"})
        call = ("L1:symbol:a.py#caller", "L1:symbol:a.py#helper")
        self.assertIn(call, edges(brain, "calls"))
        edge = next(e for e in brain.edges.values()
                    if (e.src, e.dst) == call and e.kind == "calls")
        self.assertIs(edge.env.method, Method.EXTRACTED)

    def test_self_method_calls_are_resolved(self):
        brain = self._build({
            "a.py": "class S:\n    def helper(self): pass\n"
                    "    def run(self):\n        self.helper()\n"
        })
        self.assertIn(("L1:symbol:a.py#S.run", "L1:symbol:a.py#S.helper"),
                      edges(brain, "calls"))

    def test_cross_module_calls_resolve_through_imports(self):
        brain = self._build({
            "pkg/__init__.py": "",
            "pkg/core.py": "def work(): pass\n",
            "pkg/app.py": "from .core import work\ndef run():\n    work()\n",
        })
        call = ("L1:symbol:pkg/app.py#run", "L1:symbol:pkg/core.py#work")
        self.assertIn(call, edges(brain, "calls"))
        edge = next(e for e in brain.edges.values()
                    if (e.src, e.dst) == call and e.kind == "calls")
        # Python names can be rebound at runtime, so this is a reading, not a fact.
        self.assertIs(edge.env.method, Method.DERIVED)

    def test_module_attribute_calls_resolve(self):
        brain = self._build({
            "pkg/__init__.py": "",
            "pkg/core.py": "def work(): pass\n",
            "pkg/app.py": "from pkg import core\ndef run():\n    core.work()\n",
        })
        self.assertIn(("L1:symbol:pkg/app.py#run", "L1:symbol:pkg/core.py#work"),
                      edges(brain, "calls"))

    def test_imports_produce_module_edges(self):
        brain = self._build({
            "pkg/__init__.py": "",
            "pkg/core.py": "def work(): pass\n",
            "pkg/app.py": "from .core import work\n",
        })
        self.assertIn(("L1:module:pkg/app.py", "L1:module:pkg/core.py"),
                      edges(brain, "imports"))

    def test_third_party_calls_are_not_claimed(self):
        # Nothing in the repo defines `requests.get`, so no edge may be invented.
        brain = self._build({"a.py": "import requests\ndef f():\n    requests.get('x')\n"})
        self.assertEqual(edges(brain, "calls"), set())

    def test_unparsable_files_are_reported_not_swallowed(self):
        brain = self._build({"broken.py": "def (((\n", "ok.py": "def f(): pass\n"})
        gap = brain.fact(REPO, "python_unparsed_files")
        self.assertIsNotNone(gap)
        self.assertEqual(gap.value[0]["path"], "broken.py")
        self.assertIsNotNone(brain.get("L1:symbol:ok.py#f"))

    def test_third_party_imports_exclude_stdlib_and_local(self):
        brain = self._build({"a.py": "import os\nimport json\nimport requests\n"})
        third = brain.fact(REPO, "python_third_party_imports")
        self.assertIn("requests", third.value)
        self.assertNotIn("os", third.value)

    def test_property_pairs_keep_both_definitions(self):
        # A getter and its setter share a qualname. Neither may silently
        # overwrite the other, or the Brain quietly loses a definition.
        brain = self._build({"a.py": (
            "class C:\n"
            "    @property\n"
            "    def x(self): return 1\n"
            "    @x.setter\n"
            "    def x(self, v): pass\n"
        )})
        first = brain.get("L1:symbol:a.py#C.x")
        second = brain.get("L1:symbol:a.py#C.x~2")
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(second.attrs["redefinition"], 2)
        self.assertEqual(second.attrs["qualname"], "C.x")

    def test_redefinitions_do_not_collide_in_the_merge(self):
        result = self._build_result({"a.py": (
            "class C:\n"
            "    @property\n"
            "    def x(self): return 1\n"
            "    @x.setter\n"
            "    def x(self, v): pass\n"
        )})
        self.assertEqual(result.report.kept, 0)

    def test_self_recursion_produces_no_edge(self):
        brain = self._build({"a.py": "def f():\n    f()\n"})
        self.assertEqual(edges(brain, "calls"), set())

    def test_reproducible(self):
        files = {"a.py": "def f(): pass\ndef g():\n    f()\n"}
        one, two = self._build(files), self._build(files)
        self.assertEqual({r.id for r in one.records()}, {r.id for r in two.records()})


# ============================ TypeScript ============================


class TestMask(unittest.TestCase):
    def test_line_numbers_survive(self):
        source = "const a = 1;\n/* two\n   lines */\nconst b = 2;\n"
        self.assertEqual(len(mask(source)[0].splitlines()), len(source.splitlines()))

    def test_multiline_template_preserves_line_count(self):
        source = "const a = `one\ntwo\nthree`;\nconst b = 2;\n"
        self.assertEqual(len(mask(source)[0].splitlines()), len(source.splitlines()))

    def test_line_comment_content_is_removed(self):
        self.assertNotIn("secret", mask("const a = 1; // secret\n")[0])

    def test_string_bodies_leave_the_source_but_stay_recoverable(self):
        masked, values = mask("""const a = "import x from 'y'";\n""")
        self.assertNotIn("import x", masked)
        self.assertEqual(values, ["import x from 'y'"])

    def test_escapes_do_not_end_a_string_early(self):
        masked, values = mask(r'const a = "he said \"hi\""; const b = 2;')
        self.assertIn("const b", masked)
        self.assertEqual(values, ['he said "hi"'])

    def test_template_literals_are_masked(self):
        masked, values = mask("const a = `nope`;\nconst b = 1;\n")
        self.assertNotIn("nope", masked)
        self.assertIn("nope", values)


class TestResolveSpecifier(unittest.TestCase):
    def setUp(self):
        self.known = {"src/a.ts", "src/util/index.ts", "src/b.tsx"}

    def test_sibling_with_extension_added(self):
        self.assertEqual(resolve_specifier("./a", "src/main.ts", self.known), "src/a.ts")

    def test_directory_index(self):
        self.assertEqual(resolve_specifier("./util", "src/main.ts", self.known),
                         "src/util/index.ts")

    def test_parent_traversal(self):
        self.assertEqual(resolve_specifier("../a", "src/deep/main.ts", self.known),
                         "src/a.ts")

    def test_bare_specifier_is_external(self):
        self.assertIsNone(resolve_specifier("react", "src/main.ts", self.known))

    def test_unresolvable_relative_import(self):
        self.assertIsNone(resolve_specifier("./nope", "src/main.ts", self.known))


class TestTypeScriptStructure(unittest.TestCase):
    def _build(self, files: dict[str, str]):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, text in files.items():
            write(root, rel, text)
        return build(BuildContext(root=root), [TypeScriptStructureProvider()]).brain

    def test_declarations_are_found(self):
        brain = self._build({"a.ts": (
            "export class Widget {}\n"
            "export function build() {}\n"
            "export interface Shape {}\n"
            "export const run = async () => {};\n"
            "type Id = string;\n"
        )})
        kinds = {n.key.split("#")[-1]: n.attrs["symbol_kind"]
                 for n in brain.nodes.values() if n.kind == "symbol"}
        self.assertEqual(kinds["Widget"], "class")
        self.assertEqual(kinds["build"], "function")
        self.assertEqual(kinds["Shape"], "interface")
        self.assertEqual(kinds["run"], "const")
        self.assertEqual(kinds["Id"], "type")

    def test_declarations_are_derived_not_extracted(self):
        # A scanner is not a parser, and the envelope has to say so.
        brain = self._build({"a.ts": "export class Widget {}\n"})
        node = brain.get("L1:symbol:a.ts#Widget")
        self.assertIs(node.env.method, Method.DERIVED)

    def test_relative_imports_become_edges(self):
        brain = self._build({
            "src/a.ts": "export const x = 1;\n",
            "src/main.ts": "import { x } from './a';\n",
        })
        self.assertIn(("L1:module:src/main.ts", "L1:module:src/a.ts"),
                      edges(brain, "imports"))

    def test_require_and_dynamic_import_are_seen(self):
        brain = self._build({
            "src/a.js": "module.exports = 1;\n",
            "src/main.js": "const a = require('./a');\nimport('./a');\n",
        })
        self.assertIn(("L1:module:src/main.js", "L1:module:src/a.js"),
                      edges(brain, "imports"))

    def test_imports_inside_strings_are_ignored(self):
        brain = self._build({
            "src/a.ts": "export const x = 1;\n",
            "src/main.ts": "const doc = \"import { x } from './a'\";\n",
        })
        self.assertEqual(edges(brain, "imports"), set())

    def test_external_imports_are_recorded(self):
        brain = self._build({"a.ts": "import React from 'react';\n"
                                     "import { z } from '@scope/pkg';\n"})
        external = brain.fact(REPO, "typescript_external_imports").value
        self.assertIn("react", external)
        self.assertIn("@scope/pkg", external)

    def test_the_call_graph_gap_is_declared(self):
        brain = self._build({"a.ts": "export const x = 1;\n"})
        gap = brain.fact(REPO, "typescript_coverage_gap")
        self.assertIsNotNone(gap)
        self.assertFalse(gap.value["call_graph"])


if __name__ == "__main__":
    unittest.main()
