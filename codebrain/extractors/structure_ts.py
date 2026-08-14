"""L1 structure for TypeScript and JavaScript.

There is no TS parser in the standard library and the deterministic core takes
no third-party dependencies, so this is a scanner: comments and string literals
are stripped first, then declarations and import specifiers are read off the
remaining source.

That constraint is stated in the output rather than hidden. Import specifiers
are EXTRACTED — the text is unambiguous once strings are gone. Declarations are
DERIVED, because a scanner will miss forms a parser would catch. And there is no
call graph at all for TS in P1; that gap is emitted as a fact so an agent can
see the edge of the map instead of concluding these files call nothing
(principle vi).

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
                    break

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
                          "import_edges": len(seen)},
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
            value={"call_graph": False, "type_resolution": False,
                   "reason": "scanner-based extraction; no TS parser in the "
                             "dependency-free core",
                   "impact": "blast radius across TS call sites is incomplete"},
            env=env(Method.EXTRACTED, "."),
        )

    @staticmethod
    def _specifiers(line: str, literals: list[str]) -> Iterable[str]:
        for pattern in (IMPORT_FROM, IMPORT_BARE, REQUIRE, DYNAMIC_IMPORT):
            for match in pattern.finditer(line):
                index = int(match.group(1))
                if index < len(literals) and literals[index]:
                    yield literals[index]


register(TypeScriptStructureProvider())
