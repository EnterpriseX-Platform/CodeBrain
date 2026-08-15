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
    find_body,
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
        self.assertFalse(gap.value["cross_file_calls"])
        self.assertIn("more than once", gap.value["impact"])

    def test_a_name_redeclared_in_separate_scopes_keeps_both(self):
        # A closure reusing a local name (e.g. two IIFEs each defining their own
        # `close`) must not have the second silently overwrite the first.
        brain = self._build({"main.js": (
            "function outer1() {\n"
            "  function close() { return 1; }\n"
            "}\n"
            "function outer2() {\n"
            "  function close() { return 2; }\n"
            "}\n"
        )})
        first = brain.get("L1:symbol:main.js#close")
        second = brain.get("L1:symbol:main.js#close~2")
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(second.attrs["qualname"], "close")
        self.assertEqual(second.attrs["redefinition"], 2)
        self.assertNotIn("redefinition", first.attrs)

    def test_redefinitions_do_not_collide_in_the_merge(self):
        result = build(BuildContext(root=self._root({"main.js": (
            "function close() { return 1; }\n"
            "function close() { return 2; }\n"
        )})), [TypeScriptStructureProvider()])
        self.assertEqual(result.report.kept, 0)

    # -- same-file call graph -------------------------------------------------

    def test_find_body_single_line(self):
        lines = ["function foo() { return bar(); }"]
        self.assertEqual(find_body(lines, 0), (0, 0))

    def test_find_body_multi_line(self):
        lines = ["function foo() {", "  doWork();", "}"]
        self.assertEqual(find_body(lines, 0), (0, 2))

    def test_find_body_handles_a_same_line_object_literal_default(self):
        # The default-parameter object's own braces must net to zero on that
        # line so they do not get mistaken for the function body closing.
        lines = ["function foo(opts = {debug: true}) {", "  bar();", "}"]
        self.assertEqual(find_body(lines, 0), (0, 2))

    def test_find_body_handles_a_wrapped_signature(self):
        lines = ["function foo(", "  a: number,", "  b: string,", ") {",
                 "  bar();", "}"]
        self.assertEqual(find_body(lines, 0), (3, 5))

    def test_find_body_none_for_a_braceless_arrow(self):
        self.assertIsNone(find_body(["const f = (x) => x + 1;"], 0))

    def test_find_body_is_not_limited_to_a_short_body(self):
        # The bug this guards against: an early version bounded the whole
        # body to the same short window used to search for the opening
        # brace, and silently dropped most real, ordinary functions in a
        # large real codebase as a result.
        lines = (["function foo() {"] + [f"  step{i}();" for i in range(40)]
                 + ["}"])
        self.assertEqual(find_body(lines, 0), (0, len(lines) - 1))

    def test_find_body_none_when_nothing_is_within_reach(self):
        lines = ["declare function f(): void;"] + ["x;"] * 10 + ["function g() {}"]
        self.assertIsNone(find_body(lines, 0))

    def test_a_same_file_call_is_resolved(self):
        brain = self._build({"a.ts": (
            "function helper() { return 1; }\n"
            "function main() {\n"
            "  return helper();\n"
            "}\n"
        )})
        self.assertIn(("L1:symbol:a.ts#main", "L1:symbol:a.ts#helper"),
                      edges(brain, "calls"))

    def test_a_call_is_derived_not_extracted(self):
        brain = self._build({"a.ts": (
            "function helper() { return 1; }\n"
            "function main() {\n  return helper();\n}\n"
        )})
        edge = next(e for e in brain.edges.values()
                    if e.kind == "calls" and e.src.endswith("#main"))
        self.assertIs(edge.env.method, Method.DERIVED)
        self.assertIn("same-file only", edge.env.note)

    def test_a_call_from_an_arrow_const_is_resolved(self):
        brain = self._build({"a.ts": (
            "function helper() { return 1; }\n"
            "const main = () => {\n"
            "  return helper();\n"
            "};\n"
        )})
        self.assertIn(("L1:symbol:a.ts#main", "L1:symbol:a.ts#helper"),
                      edges(brain, "calls"))

    def test_self_recursion_produces_no_ts_edge(self):
        brain = self._build({"a.ts": "function f() {\n  return f();\n}\n"})
        self.assertEqual(edges(brain, "calls"), set())

    def test_a_method_call_is_not_attributed_to_a_same_named_top_level_function(self):
        # this.helper()/obj.helper() might resolve to a class method this pass
        # does not model — attributing it to an unrelated top-level function of
        # the same name would be a wrong edge, worse than no edge.
        brain = self._build({"a.ts": (
            "function helper() { return 1; }\n"
            "class C {\n"
            "  helper() { return 2; }\n"
            "  run() {\n"
            "    return this.helper();\n"
            "  }\n"
            "}\n"
        )})
        self.assertEqual(edges(brain, "calls"), set())

    def test_a_name_declared_twice_is_never_resolved(self):
        # A scanner has no real scope analysis; a wrong guess between two
        # candidates is worse than declining to pick one.
        brain = self._build({"a.ts": (
            "function target() { return 1; }\n"
            "function target() { return 2; }\n"
            "function main() {\n  return target();\n}\n"
        )})
        self.assertEqual(edges(brain, "calls"), set())

    def test_a_call_to_an_undeclared_name_is_not_claimed(self):
        # `fetch` is a global, not declared anywhere in this file.
        brain = self._build({"a.ts": "function main() {\n  return fetch('/x');\n}\n"})
        self.assertEqual(edges(brain, "calls"), set())

    def test_a_braceless_arrow_body_is_not_scanned(self):
        brain = self._build({"a.ts": (
            "function helper() { return 1; }\n"
            "const main = () => helper();\n"
        )})
        self.assertEqual(edges(brain, "calls"), set())

    def test_call_count_is_summarised(self):
        brain = self._build({"a.ts": (
            "function helper() { return 1; }\n"
            "function main() {\n  return helper();\n}\n"
        )})
        self.assertEqual(brain.fact(REPO, "typescript_summary").value["call_edges"], 1)

    def test_the_call_graph_is_deterministic(self):
        files = {"a.ts": (
            "function helper() { return 1; }\n"
            "function main() {\n  return helper();\n}\n"
        )}
        one = self._build(files)
        two = self._build(files)
        self.assertEqual({e.id for e in one.edges.values() if e.kind == "calls"},
                         {e.id for e in two.edges.values() if e.kind == "calls"})

    def _root(self, files: dict[str, str]) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, text in files.items():
            write(root, rel, text)
        return root


if __name__ == "__main__":
    unittest.main()
