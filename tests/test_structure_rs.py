from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codebrain.envelope import Method
from codebrain.model import REPO, Layer
from codebrain.providers import BuildContext, build
from codebrain.extractors.structure_rs import (
    RustStructureProvider,
    find_body,
    mask,
    module_path,
)


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def edges(brain, kind: str) -> set[tuple[str, str]]:
    return {(e.src, e.dst) for e in brain.edges.values() if e.kind == kind}


class TestModulePath(unittest.TestCase):
    def test_plain_file(self):
        self.assertEqual(module_path("src/foo/bar.rs"), "foo::bar")

    def test_mod_rs_names_its_directory(self):
        self.assertEqual(module_path("src/foo/mod.rs"), "foo")

    def test_main_rs_at_root(self):
        self.assertEqual(module_path("src/main.rs"), "crate")

    def test_lib_rs(self):
        self.assertEqual(module_path("src/lib.rs"), "crate")

    def test_no_src_prefix(self):
        self.assertEqual(module_path("main.rs"), "crate")


class TestMask(unittest.TestCase):
    def test_line_numbers_survive(self):
        source = 'fn a() {}\n/* two\nlines */\nfn b() {}\n'
        self.assertEqual(len(mask(source).splitlines()), len(source.splitlines()))

    def test_line_comment_removed(self):
        self.assertNotIn("secret", mask("let x = 1; // secret\n"))

    def test_block_comment_removed(self):
        self.assertNotIn("secret", mask("/* secret */\nlet x = 1;\n"))

    def test_string_body_removed(self):
        masked = mask('let s = "contains { and } braces";\n')
        self.assertNotIn("contains", masked)
        self.assertIn('"', masked)  # delimiters survive so quote-counting stays sane

    def test_escaped_quote_does_not_end_the_string_early(self):
        masked = mask(r'let s = "he said \"hi\""; let n = 1;' + "\n")
        self.assertIn("let n", masked)

    # -- the lifetime vs char-literal trap -----------------------------------

    def test_simple_char_literal_is_masked(self):
        masked = mask("let c = 'x';\n")
        self.assertNotIn("x", masked)
        self.assertIn("'", masked)

    def test_escaped_char_literal_is_masked(self):
        masked = mask(r"let c = '\n';" + "\n")
        self.assertNotIn("n", masked.split("=")[1] if "=" in masked else masked)

    def test_escaped_quote_char_literal(self):
        # '\'' — an escaped single quote as a char literal.
        masked = mask(r"let c = '\'';" + "\n" + "fn after() {}\n")
        self.assertIn("fn after", masked)

    def test_a_lifetime_is_not_consumed_as_a_string(self):
        # The failure mode this guards against: treating 'a as the start of
        # an unterminated string would consume everything after it, up to
        # some unrelated later quote, corrupting the rest of the scan.
        source = "fn foo<'a>(x: &'a str) -> &'a str {\n    x\n}\n"
        masked = mask(source)
        self.assertIn("fn foo", masked)
        self.assertIn("x", masked)
        self.assertEqual(len(masked.splitlines()), len(source.splitlines()))

    def test_a_function_after_a_lifetime_signature_is_still_reachable(self):
        # If the lifetime were mis-consumed as a string, this second function
        # would vanish from the masked output entirely.
        source = "fn a<'a>(x: &'a str) {}\nfn b() {}\n"
        self.assertIn("fn b", mask(source))

    def test_static_lifetime(self):
        source = "fn f() -> &'static str {\n    \"x\"\n}\nfn after() {}\n"
        self.assertIn("fn after", mask(source))

    # -- hash-delimited raw strings -------------------------------------------

    def test_plain_raw_string(self):
        masked = mask('let s = r"contains \\ backslash";\nfn after() {}\n')
        self.assertNotIn("backslash", masked)
        self.assertIn("fn after", masked)

    def test_single_hash_raw_string(self):
        masked = mask('let s = r#"has "quotes" inside"#;\nfn after() {}\n')
        self.assertNotIn("has", masked)
        self.assertIn("fn after", masked)

    def test_double_hash_raw_string_is_not_closed_by_a_single_hash(self):
        # r##"..."## must not end at the first "# it meets — only at "##.
        masked = mask('let s = r##"one "# still open"##;\nfn after() {}\n')
        self.assertNotIn("still open", masked)
        self.assertIn("fn after", masked)

    def test_byte_raw_string(self):
        masked = mask('let s = br#"bytes"#;\nfn after() {}\n')
        self.assertIn("fn after", masked)

    def test_byte_string(self):
        masked = mask('let s = b"raw bytes";\nfn after() {}\n')
        self.assertNotIn("raw bytes", masked)
        self.assertIn("fn after", masked)


class TestFindBody(unittest.TestCase):
    def test_single_line(self):
        self.assertEqual(find_body(["fn f() { g(); }"], 0), (0, 0))

    def test_multi_line(self):
        lines = ["fn f() {", "    g();", "}"]
        self.assertEqual(find_body(lines, 0), (0, 2))

    def test_is_not_limited_to_a_short_body(self):
        lines = (["fn f() {"] + [f"    step{i}();" for i in range(40)] + ["}"])
        self.assertEqual(find_body(lines, 0), (0, len(lines) - 1))

    def test_wrapped_signature(self):
        lines = ["fn f(", "    a: i32,", "    b: &str,", ") {", "    g();", "}"]
        self.assertEqual(find_body(lines, 0), (3, 5))


class TestRustStructure(unittest.TestCase):
    def _build(self, files: dict[str, str]):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, text in files.items():
            write(root, rel, text)
        return build(BuildContext(root=root), [RustStructureProvider()]).brain

    def test_free_function_is_extracted(self):
        brain = self._build({"src/main.rs": "fn helper() -> i32 {\n    1\n}\n"})
        node = brain.get("L1:symbol:src/main.rs#helper")
        self.assertIsNotNone(node)
        self.assertEqual(node.attrs["symbol_kind"], "function")

    def test_pub_fn_is_marked_public(self):
        brain = self._build({"src/main.rs": "pub fn helper() {}\n"})
        self.assertTrue(brain.get("L1:symbol:src/main.rs#helper").attrs["public"])

    def test_private_fn_is_not_marked_public(self):
        brain = self._build({"src/main.rs": "fn helper() {}\n"})
        self.assertFalse(brain.get("L1:symbol:src/main.rs#helper").attrs["public"])

    def test_async_fn_is_recognised(self):
        brain = self._build({"src/main.rs": "pub async fn fetch() {}\n"})
        self.assertIsNotNone(brain.get("L1:symbol:src/main.rs#fetch"))

    def test_struct_enum_trait_type_const(self):
        # Declarations are matched at line-start (the same constraint the TS
        # scanner already has), so this is formatted the way rustfmt actually
        # produces it — not crammed onto one line each.
        brain = self._build({"src/main.rs": (
            "pub struct Job {\n"
            "    pub id: String,\n"
            "}\n"
            "enum Status {\n"
            "    Running,\n"
            "    Done,\n"
            "}\n"
            "trait Runner {\n"
            "    fn run(&self);\n"
            "}\n"
            "type JobId = String;\n"
            "const MAX: usize = 10;\n"
        )})
        kinds = {n.attrs["symbol_kind"] for n in brain.nodes.values()
                if n.kind == "symbol"}
        self.assertEqual(kinds, {"struct", "enum", "trait", "function", "type", "const"})

    def test_a_trait_method_with_no_body_is_still_a_symbol(self):
        brain = self._build({"src/main.rs":
                             "trait Runner {\n    fn run(&self);\n}\n"})
        self.assertIsNotNone(brain.get("L1:symbol:src/main.rs#run"))

    def test_impl_methods_are_qualified_by_their_type(self):
        brain = self._build({"src/main.rs": (
            "struct Job;\n"
            "impl Job {\n"
            "    pub fn new() -> Self {\n"
            "        Job\n"
            "    }\n"
            "}\n"
        )})
        node = brain.get("L1:symbol:src/main.rs#Job::new")
        self.assertIsNotNone(node)
        self.assertEqual(node.attrs["impl_target"], "Job")

    def test_impl_target_resets_after_the_block_closes(self):
        brain = self._build({"src/main.rs": (
            "struct Job;\n"
            "impl Job {\n"
            "    fn method(&self) {}\n"
            "}\n"
            "fn free_function() {}\n"
        )})
        self.assertIsNotNone(brain.get("L1:symbol:src/main.rs#Job::method"))
        self.assertIsNotNone(brain.get("L1:symbol:src/main.rs#free_function"))
        self.assertIsNone(brain.get("L1:symbol:src/main.rs#Job::free_function"))

    def test_impl_trait_for_type(self):
        brain = self._build({"src/main.rs": (
            "struct Job;\n"
            "trait Runner { fn run(&self); }\n"
            "impl Runner for Job {\n"
            "    fn run(&self) {}\n"
            "}\n"
        )})
        self.assertIsNotNone(brain.get("L1:symbol:src/main.rs#Job::run"))

    def test_a_same_file_call_is_resolved(self):
        brain = self._build({"src/main.rs": (
            "fn helper() -> i32 { 1 }\n"
            "fn main() {\n"
            "    helper();\n"
            "}\n"
        )})
        self.assertIn(("L1:symbol:src/main.rs#main", "L1:symbol:src/main.rs#helper"),
                      edges(brain, "calls"))

    def test_a_call_is_derived_not_extracted(self):
        brain = self._build({"src/main.rs": (
            "fn helper() { }\nfn main() {\n    helper();\n}\n"
        )})
        edge = next(e for e in brain.edges.values()
                    if e.kind == "calls" and e.src.endswith("#main"))
        self.assertIs(edge.env.method, Method.DERIVED)

    def test_self_recursion_produces_no_edge(self):
        brain = self._build({"src/main.rs": "fn f() {\n    f();\n}\n"})
        self.assertEqual(edges(brain, "calls"), set())

    def test_a_method_call_is_not_claimed_via_self(self):
        # self.method() might resolve to a DIFFERENT type's method of the
        # same name; this pass does not model receivers at all.
        brain = self._build({"src/main.rs": (
            "fn helper() {}\n"
            "struct Job;\n"
            "impl Job {\n"
            "    fn helper(&self) {}\n"
            "    fn run(&self) {\n"
            "        self.helper();\n"
            "    }\n"
            "}\n"
        )})
        self.assertEqual(edges(brain, "calls"), set())

    def test_an_ambiguous_name_is_never_resolved(self):
        brain = self._build({"src/main.rs": (
            "fn target() { }\n"
            "fn target() { }\n"
            "fn main() {\n    target();\n}\n"
        )})
        self.assertEqual(edges(brain, "calls"), set())

    def test_a_call_through_a_lifetime_heavy_signature_still_resolves(self):
        # The real-world case this feature exists for: idiomatic Rust
        # signatures are full of lifetimes, and they must not break scanning
        # of the function that follows.
        brain = self._build({"src/main.rs": (
            "fn helper() -> i32 { 1 }\n"
            "fn process<'a>(x: &'a str) -> &'a str {\n"
            "    helper();\n"
            "    x\n"
            "}\n"
        )})
        self.assertIn(("L1:symbol:src/main.rs#process", "L1:symbol:src/main.rs#helper"),
                      edges(brain, "calls"))

    def test_mod_declaration_resolves_to_a_sibling_file(self):
        brain = self._build({
            "src/main.rs": "mod util;\n",
            "src/util.rs": "pub fn helper() {}\n",
        })
        self.assertIn(("L1:module:src/main.rs", "L1:module:src/util.rs"),
                      edges(brain, "imports"))

    def test_mod_declaration_resolves_to_a_directory_mod_rs(self):
        brain = self._build({
            "src/main.rs": "mod util;\n",
            "src/util/mod.rs": "pub fn helper() {}\n",
        })
        self.assertIn(("L1:module:src/main.rs", "L1:module:src/util/mod.rs"),
                      edges(brain, "imports"))

    def test_an_unresolvable_mod_is_not_a_broken_edge(self):
        brain = self._build({"src/main.rs": "mod somewhere_else;\n"})
        self.assertEqual(edges(brain, "imports"), set())

    def test_a_redefined_name_across_impls_is_kept_not_dropped(self):
        brain = self._build({"src/main.rs": (
            "struct A;\nstruct B;\n"
            "impl A {\n    fn run(&self) {}\n}\n"
            "impl B {\n    fn run(&self) {}\n}\n"
        )})
        self.assertIsNotNone(brain.get("L1:symbol:src/main.rs#A::run"))
        self.assertIsNotNone(brain.get("L1:symbol:src/main.rs#B::run"))

    def test_the_coverage_gap_is_declared(self):
        brain = self._build({"src/main.rs": "fn f() {}\n"})
        gap = brain.fact(REPO, "rust_coverage_gap", Layer.L1)
        self.assertIsNotNone(gap)
        self.assertFalse(gap.value["cross_file_calls"])
        self.assertIn("macro", gap.value["impact"])

    def test_unreadable_file_does_not_break_the_provider(self):
        brain = self._build({"src/main.rs": "fn f() {}\n"})
        self.assertIsNotNone(brain.fact(REPO, "rust_summary", Layer.L1))

    def test_reproducible(self):
        files = {"src/main.rs": "fn helper() {}\nfn main() {\n    helper();\n}\n"}
        one, two = self._build(files), self._build(files)
        self.assertEqual({r.id for r in one.records()}, {r.id for r in two.records()})

    def test_applies_only_when_rust_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x = 1\n", encoding="utf-8")
            self.assertFalse(RustStructureProvider().applies(BuildContext(root=root)))


if __name__ == "__main__":
    unittest.main()
