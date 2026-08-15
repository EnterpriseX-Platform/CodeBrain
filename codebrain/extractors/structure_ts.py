"""L1 structure for TypeScript and JavaScript.

There is no TS parser in the standard library and the deterministic core takes
no third-party dependencies, so this is a scanner: comments and string literals
are stripped first, then declarations, imports, and same-file calls are read
off the remaining source.

That constraint is stated in the output rather than hidden. Import specifiers
are EXTRACTED — the text is unambiguous once strings are gone. Declarations and
calls are DERIVED, because a scanner will miss forms a parser would catch.

Call resolution is same-file only, and deliberately conservative about it. A
symbol's body is found by counting net brace depth from its declaration —
sound for standard K&R-style bodies, blind to a body opened more than a few
lines after its signature and to braceless single-expression arrows
(`x => x + 1`), both stated as gaps rather than guessed at. A bare name that is
declared more than once in the same file is never resolved to either
occurrence: a scanner has no real scope analysis, and a wrong edge is worse
than a missing one. Method calls (`this.helper()`, `obj.method()`) are not
attempted at all — this pass only tracks top-level function and arrow-const
symbols, not class methods, so resolving a `.`-qualified call would mean
guessing at a receiver this pass cannot see.

Replacing this with a real parser is a provider swap, not a rewrite — which is
the point of the plugin contract.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ..envelope import Envelope, Evidence, Method, Status
from ..model import REPO, Edge, Fact, Layer, Node, Record
from ..providers import BuildContext, Provider, register

SUFFIXES = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")

#: Tried in order when resolving a relative import with no extension.
RESOLUTION_ORDER = (
    "", ".ts", ".tsx", ".d.ts", ".js", ".jsx", ".mts", ".cts", ".mjs", ".cjs",
    "/index.ts", "/index.tsx", "/index.js", "/index.jsx",
)

#: A masked string literal: the delimiters survive, the body becomes an index
#: into the recovered-values table (see `mask`).
LITERAL = r"""["'`]\x01(\d+)\x01["'`]"""

IMPORT_FROM = re.compile(r"\b(?:import|export)\b[^;\n]*?\bfrom\s*" + LITERAL)
IMPORT_BARE = re.compile(r"\bimport\s*" + LITERAL)
REQUIRE = re.compile(r"\brequire\s*\(\s*" + LITERAL + r"\s*\)")
DYNAMIC_IMPORT = re.compile(r"\bimport\s*\(\s*" + LITERAL + r"\s*\)")

DECLARATIONS = (
    ("class", re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)")),
    ("interface", re.compile(r"^\s*(?:export\s+(?:default\s+)?)?interface\s+([A-Za-z_$][\w$]*)")),
    ("type", re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*=")),
    ("enum", re.compile(r"^\s*(?:export\s+)?(?:const\s+)?enum\s+([A-Za-z_$][\w$]*)")),
    ("function", re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)")),
    ("const", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>")),
)

EXPORTED = re.compile(r"^\s*export\b")

#: A bare call: an identifier directly followed by `(`, not reached through a
#: `.` — `helper(x)` is a candidate, `this.helper(x)` and `obj.helper(x)` are
#: not, because a receiver this pass does not model could resolve to a method
#: that only coincidentally shares a top-level function's name.
CALL = re.compile(r"(?<!\.)\b([A-Za-z_$][\w$]*)\s*\(")

#: How many lines past a declaration to search for its OPENING brace before
#: giving up. This bounds only where a body may *start* — long enough for a
#: signature wrapped across a few lines of typed parameters, short enough
#: that a bodyless declaration (an ambient `declare function f(): void;`)
#: never gets misattributed to some later, unrelated block's brace. It does
#: not bound how long the body may run once found: an early version of this
#: function applied the same six-line limit to the whole body and silently
#: dropped almost three-quarters of real, ordinary multi-line functions in a
#: large real codebase — found by measuring the hit rate, not by inspection.
OPEN_BRACE_LOOKAHEAD_LINES = 6

#: A hard ceiling on total body length, once an opening brace is found —
#: protection against a genuinely unbalanced brace count (the regex-literal
#: blind spot documented at the top of this module) turning into an unbounded
#: scan, not a limit real function bodies are expected to hit.
MAX_BODY_LINES = 2000


def find_body(lines: list[str], start_idx: int) -> tuple[int, int] | None:
    """The 0-based (start, end) line range of a braced body beginning at or
    after `start_idx`, by net brace depth over already-masked lines — string
    and comment content can never contribute a stray brace, because mask()
    has already removed it. None if no `{` appears within the open-brace
    lookahead window at all.
    """
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


def mask(source: str) -> tuple[str, list[str]]:
    """Blank comments and replace string bodies with recoverable placeholders.

    Import specifiers *are* string literals, so a scanner cannot simply delete
    string contents — it would destroy the thing it came to read. Instead each
    literal becomes `"\\x01<index>\\x01"` and its real value is returned
    alongside. That kills the false positive (an `import` written inside a
    string is now invisible to the pattern) while keeping the specifier
    recoverable.

    Line count is preserved so evidence line numbers stay truthful.
    """
    out: list[str] = []
    values: list[str] = []
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
            while i < n and not (source[i] == "*" and source[i + 1: i + 2] == "/"):
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
            continue

        if ch in "\"'`":
            quote = ch
            i += 1
            body: list[str] = []
            newlines = 0
            while i < n:
                if source[i] == "\\":
                    body.append(source[i + 1: i + 2])
                    i += 2
                    continue
                if source[i] == quote:
                    i += 1
                    break
                if source[i] == "\n":
                    newlines += 1
                body.append(source[i])
                i += 1
            out.append(f"{quote}\x01{len(values)}\x01{quote}")
            out.append("\n" * newlines)
            values.append("".join(body))
            continue

        out.append(ch)
        i += 1

    return "".join(out), values


def resolve_specifier(spec: str, from_rel: str, known: set[str]) -> str | None:
    """Resolve a relative import to a repo file, or None if it leaves the repo."""
    if not spec.startswith("."):
        return None
    base = Path(from_rel).parent
    target = (base / spec).as_posix()
    # Normalise ./ and ../ without touching the filesystem.
    parts: list[str] = []
    for part in target.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    stem = "/".join(parts)
    for suffix in RESOLUTION_ORDER:
        if (candidate := stem + suffix) in known:
            return candidate
    return None


class TypeScriptStructureProvider(Provider):
    id = "structure-ts"
    layers = (Layer.L1,)
    description = "TypeScript/JavaScript modules, declarations and imports (scanner)."
    order = 45

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

        known = {ctx.rel(p) for p in files}
        edges: list[tuple[str, str, str, int]] = []   # (from_rel, to_rel, from_rel, line)
        external: dict[str, int] = {}
        symbol_count = 0
        module_count = 0
        call_count = 0

        for path in sorted(files):
            rel = ctx.rel(path)
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            source, literals = mask(raw)
            lines = source.splitlines()
            module_count += 1
            module_id = f"{Layer.L1}:module:{rel}"

            yield Node(layer=Layer.L1, kind="module", key=rel, name=Path(rel).stem,
                       env=env(Method.EXTRACTED, rel),
                       attrs={"path": rel, "lines": len(lines),
                              "language": "TypeScript" if path.suffix.startswith(".ts")
                              or path.suffix == ".tsx" else "JavaScript"})
            yield Edge(layer=Layer.L1, kind="contains",
                       src=f"{Layer.L0}:file:{rel}", dst=module_id,
                       env=env(Method.EXTRACTED, rel))

            # A name can legitimately recur in one file: closures over the same
            # local name in separate scopes, or a small handler reused per
            # view/component. Without tracking occurrences, each redeclaration
            # silently overwrote the last — 90 lost declarations in one real
            # UI file was the signal that surfaced this. Later ones are
            # suffixed by source order rather than dropped, mirroring the same
            # fix already made for the Python extractor.
            occurrences: dict[str, int] = {}

            # Collected for the call-resolution pass below, once every
            # declaration in the file is known. bare_name_keys tracks every
            # unique key sharing a bare name, so a call to an ambiguous name
            # can be recognised and deliberately left unresolved rather than
            # guessed at.
            bare_name_keys: dict[str, list[str]] = {}
            callable_syms: list[tuple[str, str, int]] = []  # (unique_key, kind, line_idx)

            for lineno, line in enumerate(lines, 1):
                for spec in self._specifiers(line, literals):
                    target = resolve_specifier(spec, rel, known)
                    if target:
                        edges.append((rel, target, rel, lineno))
                    elif not spec.startswith("."):
                        root = spec.split("/")[0] if not spec.startswith("@") \
                            else "/".join(spec.split("/")[:2])
                        external[root] = external.get(root, 0) + 1

                for kind, pattern in DECLARATIONS:
                    match = pattern.match(line)
                    if not match:
                        continue
                    name = match.group(1)
                    occurrences[name] = occurrences.get(name, 0) + 1
                    nth = occurrences[name]
                    unique = name if nth == 1 else f"{name}~{nth}"
                    symbol_count += 1
                    yield Node(
                        layer=Layer.L1, kind="symbol", key=f"{rel}#{unique}", name=name,
                        env=env(Method.DERIVED, rel, lineno,
                                note="scanned, not parsed", confidence=0.85),
                        attrs={"symbol_kind": kind, "module": rel, "qualname": name,
                               "exported": bool(EXPORTED.match(line)),
                               **({"redefinition": nth} if nth > 1 else {})},
                    )
                    yield Edge(layer=Layer.L1, kind="defines", src=module_id,
                               dst=f"{Layer.L1}:symbol:{rel}#{unique}",
                               env=env(Method.DERIVED, rel, lineno, confidence=0.85))
                    bare_name_keys.setdefault(name, []).append(unique)
                    if kind in ("function", "const"):
                        callable_syms.append((unique, kind, lineno - 1))
                    break

            file_calls = list(self._calls(rel, lines, callable_syms, bare_name_keys, env))
            call_count += len(file_calls)
            yield from file_calls

        seen: set[tuple[str, str]] = set()
        for from_rel, to_rel, path, line in edges:
            if from_rel == to_rel or (from_rel, to_rel) in seen:
                continue
            seen.add((from_rel, to_rel))
            yield Edge(layer=Layer.L1, kind="imports",
                       src=f"{Layer.L1}:module:{from_rel}",
                       dst=f"{Layer.L1}:module:{to_rel}",
                       env=env(Method.EXTRACTED, path, line))

        yield Fact(layer=Layer.L1, subject=REPO, predicate="typescript_summary",
                   value={"modules": module_count, "symbols": symbol_count,
                          "import_edges": len(seen), "call_edges": call_count},
                   env=env(Method.EXTRACTED, "."))

        if external:
            yield Fact(
                layer=Layer.L1, subject=REPO, predicate="typescript_external_imports",
                value=dict(sorted(external.items(), key=lambda kv: (-kv[1], kv[0]))[:50]),
                env=env(Method.EXTRACTED, "."),
            )

        # State the gap rather than letting absence read as evidence of absence.
        yield Fact(
            layer=Layer.L1, subject=REPO, predicate="typescript_coverage_gap",
            value={"call_graph": "same-file only", "cross_file_calls": False,
                   "type_resolution": False,
                   "reason": "scanner-based extraction; no TS parser in the "
                             "dependency-free core",
                   "impact": "blast radius across TS call sites is incomplete "
                             "outside the declaring file, and a name declared "
                             "more than once in one file is never resolved"},
            env=env(Method.EXTRACTED, "."),
        )

    @staticmethod
    def _specifiers(line: str, literals: list[str]) -> Iterable[str]:
        for pattern in (IMPORT_FROM, IMPORT_BARE, REQUIRE, DYNAMIC_IMPORT):
            for match in pattern.finditer(line):
                index = int(match.group(1))
                if index < len(literals) and literals[index]:
                    yield literals[index]

    @staticmethod
    def _calls(rel: str, lines: list[str], callable_syms: list[tuple[str, str, int]],
               bare_name_keys: dict[str, list[str]], env) -> Iterable[Record]:
        # A name resolved only when it is declared exactly once in the file —
        # a scanner has no real scope analysis, so an ambiguous name is left
        # unresolved rather than guessed at.
        unambiguous = {name: keys[0] for name, keys in bare_name_keys.items()
                       if len(keys) == 1}
        seen: set[tuple[str, str]] = set()

        for unique_key, _kind, start_idx in callable_syms:
            body = find_body(lines, start_idx)
            if body is None:
                continue  # braceless arrow, or no body within the lookahead
            body_start, body_end = body
            src_id = f"{Layer.L1}:symbol:{rel}#{unique_key}"

            for lineno in range(body_start, body_end + 1):
                for match in CALL.finditer(lines[lineno]):
                    name = match.group(1)
                    target_key = unambiguous.get(name)
                    if target_key is None or target_key == unique_key:
                        continue  # unresolved (ambiguous/unknown), or self-recursion
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


register(TypeScriptStructureProvider())
