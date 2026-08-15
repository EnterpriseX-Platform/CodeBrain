"""L1 structure for Rust.

The same scanner architecture as structure_ts.py — no Rust parser in the
dependency-free core, so comments and literals are stripped first, then
declarations, module structure and same-file calls are read off what remains.

Rust's lexical grammar is a harder scanning target than TypeScript's, and two
traps are worth naming because getting them wrong is not "a bit less
accurate" but actively corrupting.

**Lifetimes versus char literals.** `'a` (a lifetime, appearing in nearly
every function signature with a reference) and `'a'` (a one-character string)
share an opening quote and nothing else. A scanner that treats every `'` as
"start of a string, consume to the next `'`" will, on hitting a lifetime,
consume everything up to some unrelated later quote — silently corrupting the
rest of the file's scan. Disambiguated here by a one-character lookahead: a
real char literal closes within a character or an escape sequence; a lifetime
does not, and is left untouched as ordinary text.

**Raw strings are hash-counted, and the count is not decorative.**
`r#"..."#` closes only at a `"` followed by exactly one `#`; `r##"..."##`
needs two. Measured against a real 29,000-line Rust file before trusting a
simpler design: hash-delimited raw strings were the *majority* of genuine raw
strings in it, not a rare form worth ignoring.

What is deliberately not attempted: nested block comments (rare in idiomatic
Rust; the first `*/` closes it, stated rather than guessed at), macro bodies
(a macro_rules! arm is not scanned for declarations or calls it does not
literally contain), and cross-file call resolution — the same same-file-only,
ambiguous-name-skipped posture already used for TypeScript, for the same
reason: a scanner has no real scope analysis, and a wrong edge is worse than
a missing one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ..envelope import Envelope, Evidence, Method, Status
from ..model import REPO, Edge, Fact, Layer, Node, Record
from ..providers import BuildContext, Provider, register

SUFFIXES = (".rs",)

DECLARATIONS = (
    ("function", re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+"
        r"\"[^\"]*\"\s+)?fn\s+([A-Za-z_][\w]*)")),
    ("struct", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+([A-Za-z_][\w]*)")),
    ("enum", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+([A-Za-z_][\w]*)")),
    ("trait", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?trait\s+([A-Za-z_][\w]*)")),
    ("type", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?type\s+([A-Za-z_][\w]*)")),
    ("const", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+([A-Za-z_][\w]*)")),
)

#: `impl Foo { ... }` and `impl Trait for Foo { ... }` — not a symbol in its
#: own right (there is nothing to call), but its target type is recorded so
#: methods declared inside can be attributed to the type they extend.
IMPL_BLOCK = re.compile(
    r"^\s*(?:unsafe\s+)?impl(?:\s*<[^>]*>)?\s+(?:[\w:]+(?:<[^>]*>)?\s+for\s+)?"
    r"([A-Za-z_][\w]*)")

MOD_DECL = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_][\w]*)\s*;")

CALL = re.compile(r"(?<!\.)(?<!::)\b([A-Za-z_][\w]*)\s*\(")

OPEN_BRACE_LOOKAHEAD_LINES = 6
MAX_BODY_LINES = 4000


def find_body(lines: list[str], start_idx: int) -> tuple[int, int] | None:
    """Identical in spirit to structure_ts.find_body: bound only the search
    for the *opening* brace, never how long the body runs once found — that
    conflation silently dropped most real multi-line functions the first time
    it was tried, for TypeScript, and the fix carries over unchanged."""
    depth = 0
    body_start: int | None = None
    open_search_limit = min(len(lines), start_idx + OPEN_BRACE_LOOKAHEAD_LINES)
    i = start_idx
    while i < len(lines):
        if body_start is None and i >= open_search_limit:
            return None
        opens, closes = lines[i].count("{"), lines[i].count("}")
        if body_start is None:
            if opens == 0:
                i += 1
                continue
            body_start = i
        depth += opens - closes
        if depth <= 0:
            return body_start, i
        if i - body_start > MAX_BODY_LINES:
            return None
        i += 1
    return None


def module_path(rel: str) -> str:
    """The Rust module path implied by a file's location, mirroring how
    structure_py derives a dotted path from a file path: `src/foo/bar.rs`
    becomes `foo::bar`; `mod.rs`/`lib.rs`/`main.rs` name the *containing*
    directory's module, the same role `__init__.py` plays for Python.
    """
    parts = list(Path(rel).parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if not parts:
        return rel
    stem = Path(parts[-1]).stem
    if stem in ("mod", "lib", "main"):
        parts = parts[:-1]
    else:
        parts[-1] = stem
    return "::".join(parts) if parts else "crate"


def mask(source: str) -> str:
    """Blank comments and string/char literal bodies, preserving line count so
    evidence line numbers stay truthful. Unlike structure_ts's mask(), string
    content is discarded rather than kept recoverable — nothing here needs to
    read a Rust string's contents back out, only to stop it from polluting
    brace-depth tracking and declaration matching.
    """
    out: list[str] = []
    i, n = 0, len(source)

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if ch == "/" and nxt == "/":
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
            continue

        if ch == "/" and nxt == "*":
            # Not nesting-aware: the first `*/` closes it. Rare in idiomatic
            # Rust even though the grammar allows nesting; stated, not hidden.
            out.append("  ")
            i += 2
            while i < n and not (source[i] == "*" and source[i + 1: i + 2] == "/"):
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
            continue

        # Raw strings: r"...", r#"...#"..."#, br#"..."#, with the hash count
        # on the close required to match the open exactly. Checked first,
        # since a plain string branch below would misparse the r-prefix.
        raw_match = re.match(r'b?r(#*)"', source[i:i + 64])
        if raw_match:
            hashes = raw_match.group(1)
            close = f'"{hashes}'
            start = i + raw_match.end()
            end = source.find(close, start)
            body = source[start:] if end == -1 else source[start:end]
            out.append(" " * raw_match.end())
            out.append("\n" * body.count("\n"))
            out.append(" " * (len(body) - body.count("\n")))
            if end == -1:
                i = n
            else:
                out.append(" " * len(close))
                i = end + len(close)
            continue

        if ch == '"' or (ch == "b" and nxt == '"'):
            i += 1 if ch == '"' else 2
            out.append(source[i - (1 if ch == '"' else 2):i])
            while i < n:
                if source[i] == "\\":
                    out.append("  ")
                    i += 2
                    continue
                if source[i] == '"':
                    out.append('"')
                    i += 1
                    break
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            continue

        if ch == "'":
            # The lifetime/char-literal trap. A real char literal closes
            # within an escape or a single character; a lifetime does not.
            # Getting this wrong means treating a lifetime as an unterminated
            # string and consuming the rest of the file looking for a
            # closing quote that was never a string to begin with.
            if nxt == "\\":
                j = i + 2
                while j < n and source[j] not in ("'", "\n") and j - i < 12:
                    j += 1
                if j < n and source[j] == "'":
                    out.append("'" + " " * (j - i - 1) + "'")
                    i = j + 1
                    continue
            elif nxt and source[i + 2: i + 3] == "'":
                out.append("'" + " " + "'")
                i += 3
                continue
            # Not a char literal: a lifetime, or something stranger. Leave the
            # quote as ordinary text and move on one character at a time.
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


class RustStructureProvider(Provider):
    id = "structure-rs"
    layers = (Layer.L1,)
    description = "Rust modules, symbols, impls and same-file calls (scanner)."
    order = 46

    def applies(self, ctx: BuildContext) -> bool:
        return any(p.suffix in SUFFIXES for p in ctx.iter_files())

    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        files = [p for p in ctx.iter_files() if p.suffix in SUFFIXES and ctx.readable(p)]
        if not files:
            return

        def env(method: Method, path: str, line: int | None = None,
                note: str = "", confidence: float | None = None) -> Envelope:
            return Envelope.make(
                method, source=self.id, as_of=ctx.commit, ts=ctx.ts,
                status=Status.FRESH if ctx.commit else Status.UNVERIFIED,
                confidence=confidence,
                evidence=(Evidence(path=path, start_line=line,
                                   commit=ctx.commit or None),),
                note=note,
            )

        module_count = symbol_count = call_count = 0
        mod_edges: list[tuple[str, str, int]] = []  # (from_rel, mod_name, line)
        known_files = {ctx.rel(p) for p in files}

        for path in sorted(files):
            rel = ctx.rel(path)
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            source = mask(raw)
            lines = source.splitlines()
            module_count += 1
            module_id = f"{Layer.L1}:module:{rel}"
            modpath = module_path(rel)

            yield Node(layer=Layer.L1, kind="module", key=rel, name=modpath,
                       env=env(Method.EXTRACTED, rel),
                       attrs={"path": rel, "lines": len(lines), "language": "Rust",
                              "modpath": modpath})
            yield Edge(layer=Layer.L1, kind="contains",
                       src=f"{Layer.L0}:file:{rel}", dst=module_id,
                       env=env(Method.EXTRACTED, rel))

            occurrences: dict[str, int] = {}
            bare_name_keys: dict[str, list[str]] = {}
            callable_syms: list[tuple[str, int]] = []
            impl_target: str | None = None
            impl_depth = 0
            depth = 0

            for lineno, line in enumerate(lines, 1):
                stripped = line.strip()

                impl_match = IMPL_BLOCK.match(line) if impl_target is None else None
                if impl_match:
                    impl_target = impl_match.group(1)
                    impl_depth = depth

                mod_match = MOD_DECL.match(line)
                if mod_match:
                    mod_edges.append((rel, mod_match.group(1), lineno))

                for kind, pattern in DECLARATIONS:
                    match = pattern.match(line)
                    if not match:
                        continue
                    name = match.group(1)
                    qual = f"{impl_target}::{name}" if (kind == "function"
                                                        and impl_target) else name
                    occurrences[qual] = occurrences.get(qual, 0) + 1
                    nth = occurrences[qual]
                    unique = qual if nth == 1 else f"{qual}~{nth}"
                    symbol_count += 1
                    yield Node(
                        layer=Layer.L1, kind="symbol", key=f"{rel}#{unique}", name=name,
                        env=env(Method.DERIVED, rel, lineno,
                                note="scanned, not parsed", confidence=0.85),
                        attrs={"symbol_kind": kind, "module": rel, "qualname": qual,
                               "impl_target": impl_target,
                               "public": "pub" in line[:match.start(1)],
                               **({"redefinition": nth} if nth > 1 else {})},
                    )
                    yield Edge(layer=Layer.L1, kind="defines", src=module_id,
                               dst=f"{Layer.L1}:symbol:{rel}#{unique}",
                               env=env(Method.DERIVED, rel, lineno, confidence=0.85))
                    bare_name_keys.setdefault(name, []).append(unique)
                    if kind == "function":
                        callable_syms.append((unique, lineno - 1))
                    break

                depth += line.count("{") - line.count("}")
                if impl_target is not None and depth <= impl_depth:
                    impl_target = None

            file_calls = list(self._calls(rel, lines, callable_syms, bare_name_keys, env))
            call_count += len(file_calls)
            yield from file_calls

        for from_rel, mod_name, lineno in mod_edges:
            # `mod foo;` loads foo.rs or foo/mod.rs sitting beside this file —
            # both are tried, first match wins; neither found means the module
            # lives somewhere this simple sibling rule does not reach (a
            # workspace member, a path attribute), left unresolved rather than
            # guessed at.
            base = Path(from_rel).parent
            for candidate in (base / f"{mod_name}.rs", base / mod_name / "mod.rs"):
                target = candidate.as_posix()
                if target in known_files:
                    yield Edge(layer=Layer.L1, kind="imports",
                               src=f"{Layer.L1}:module:{from_rel}",
                               dst=f"{Layer.L1}:module:{target}",
                               env=env(Method.EXTRACTED, from_rel, lineno))
                    break

        yield Fact(layer=Layer.L1, subject=REPO, predicate="rust_summary",
                   value={"modules": module_count, "symbols": symbol_count,
                          "call_edges": call_count},
                   env=env(Method.EXTRACTED, "."))

        yield Fact(
            layer=Layer.L1, subject=REPO, predicate="rust_coverage_gap",
            value={"call_graph": "same-file only", "cross_file_calls": False,
                   "cross_crate_resolution": False, "macro_expansion": False,
                   "reason": "scanner-based extraction; no Rust parser in the "
                             "dependency-free core",
                   "impact": "blast radius across Rust call sites is incomplete "
                             "outside the declaring file, and code produced by "
                             "a macro is invisible unless it is written out "
                             "literally in the source"},
            env=env(Method.EXTRACTED, "."),
        )

    @staticmethod
    def _calls(rel: str, lines: list[str], callable_syms: list[tuple[str, int]],
              bare_name_keys: dict[str, list[str]], env) -> Iterable[Record]:
        unambiguous = {name: keys[0] for name, keys in bare_name_keys.items()
                       if len(keys) == 1}
        seen: set[tuple[str, str]] = set()

        for unique_key, start_idx in callable_syms:
            body = find_body(lines, start_idx)
            if body is None:
                continue
            body_start, body_end = body
            src_id = f"{Layer.L1}:symbol:{rel}#{unique_key}"

            for lineno in range(body_start, body_end + 1):
                for match in CALL.finditer(lines[lineno]):
                    name = match.group(1)
                    target_key = unambiguous.get(name)
                    if target_key is None or target_key == unique_key:
                        continue
                    dst_id = f"{Layer.L1}:symbol:{rel}#{target_key}"
                    if (src_id, dst_id) in seen:
                        continue
                    seen.add((src_id, dst_id))
                    yield Edge(
                        layer=Layer.L1, kind="calls", src=src_id, dst=dst_id,
                        env=env(Method.DERIVED, rel, lineno + 1, confidence=0.7,
                                note="scanned via brace-depth tracking, not "
                                     "parsed; same-file only"),
                    )


register(RustStructureProvider())
