"""L1 structure for Python, from the standard library's own AST.

Deterministic and exact for what it claims. Where Python is genuinely dynamic —
a name that could be rebound at runtime, a call through an attribute chain we
cannot follow — this extractor either downgrades the claim to DERIVED or emits
nothing at all. It never guesses to look thorough (principle i, and the reason
`method` exists on the envelope at all).

Two passes, each file parsed exactly once. The first walks every file emitting
modules and symbols while recording imports and unresolved call candidates as
small tuples; the second resolves those against the now-complete module and
symbol tables. Parsed trees are never retained, so memory stays flat on large
repositories and the cold build keeps to its budget.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from typing import Iterable

from ..envelope import Envelope, Evidence, Method, Status
from ..model import REPO, Edge, Fact, Layer, Node, Record
from ..providers import BuildContext, Provider, register

SUFFIXES = (".py", ".pyi")

STDLIB: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ())) or frozenset(
    {"os", "sys", "json", "re", "typing", "pathlib", "dataclasses", "collections"}
)


def dotted(rel: str) -> str:
    """Repo-relative path to importable module path."""
    for suffix in SUFFIXES:
        if rel.endswith(suffix):
            rel = rel[: -len(suffix)]
            break
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")]
    return rel.replace("/", ".")


def resolve_relative(current: str, is_package: bool, level: int, module: str | None) -> str:
    """Resolve `from ..pkg import x` against the importing module."""
    base = current.split(".") if is_package else current.split(".")[:-1]
    if level > 1:
        base = base[: -(level - 1)] if (level - 1) <= len(base) else []
    parts = [p for p in base if p]
    if module:
        parts.extend(module.split("."))
    return ".".join(parts)


def call_target(node: ast.Call) -> tuple[str | None, str | None]:
    """(name, prefix) for a call, or (None, None) when it cannot be followed."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id, None
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.attr, func.value.id
    return None, None


@dataclass(slots=True)
class Candidate:
    """A call seen in pass one, resolvable only once every file has been read."""

    src: str
    module: str | None   # dotted module of the callee; None means same file
    name: str            # symbol qualname within that module
    method: Method
    path: str
    line: int


@dataclass(slots=True)
class ModuleInfo:
    rel: str
    dotted: str
    is_package: bool
    symbols: set[str] = field(default_factory=set)
    imports: dict[str, str] = field(default_factory=dict)                    # alias -> dotted
    from_imports: dict[str, tuple[str, str]] = field(default_factory=dict)   # alias -> (mod, orig)
    sites: list[tuple[str, int]] = field(default_factory=list)               # (dotted, line)

    @property
    def node_id(self) -> str:
        return f"{Layer.L1}:module:{self.rel}"


def symbol_defs(tree: ast.Module) -> list[tuple[str, str, ast.AST]]:
    """(qualname, kind, node) for every def and class, nested names included."""
    found: list[tuple[str, str, ast.AST]] = []

    def walk(node: ast.AST, prefix: str, inside_class: bool) -> None:
        for child in getattr(node, "body", ()):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = prefix + child.name
                found.append((qual, "method" if inside_class else "function", child))
                walk(child, qual + ".", False)
            elif isinstance(child, ast.ClassDef):
                qual = prefix + child.name
                found.append((qual, "class", child))
                walk(child, qual + ".", True)

    walk(tree, "", False)
    return found


def read_imports(info: ModuleInfo, tree: ast.Module, external: dict[str, int]) -> None:
    """Populate the module's import tables. Mutates `info` and `external`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                info.imports[alias.asname or alias.name.split(".")[0]] = alias.name
                info.sites.append((alias.name, node.lineno))
                root = alias.name.split(".")[0]
                external[root] = external.get(root, 0) + 1
        elif isinstance(node, ast.ImportFrom):
            target = (
                resolve_relative(info.dotted, info.is_package, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            if target:
                info.sites.append((target, node.lineno))
            for alias in node.names:
                info.from_imports[alias.asname or alias.name] = (target, alias.name)
            if not node.level and node.module:
                root = node.module.split(".")[0]
                external[root] = external.get(root, 0) + 1


def collect_calls(info: ModuleInfo, qual: str, sym_id: str,
                  node: ast.AST, rel: str) -> list[Candidate]:
    out: list[Candidate] = []
    class_scope = qual.rsplit(".", 1)[0] if "." in qual else ""

    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        name, prefix = call_target(sub)
        if not name:
            continue

        # `self.helper()` — the most reliable cross-symbol edge in Python, and
        # the one a purely name-based extractor misses entirely.
        if prefix == "self":
            if class_scope:
                out.append(Candidate(sym_id, None, f"{class_scope}.{name}",
                                     Method.EXTRACTED, rel, sub.lineno))
            continue

        if prefix is None:
            if name in info.symbols:
                out.append(Candidate(sym_id, None, name, Method.EXTRACTED, rel, sub.lineno))
            elif name in info.from_imports:
                mod, original = info.from_imports[name]
                out.append(Candidate(sym_id, mod, original, Method.DERIVED, rel, sub.lineno))
            continue

        # `mod.func()` where `mod` names something imported from this repo.
        if prefix in info.imports:
            out.append(Candidate(sym_id, info.imports[prefix], name,
                                 Method.DERIVED, rel, sub.lineno))
        elif prefix in info.from_imports:
            mod, original = info.from_imports[prefix]
            out.append(Candidate(sym_id, f"{mod}.{original}" if mod else original, name,
                                 Method.DERIVED, rel, sub.lineno))
    return out


class PythonStructureProvider(Provider):
    id = "structure-py"
    layers = (Layer.L1,)
    description = "Python modules, symbols, imports and resolvable calls (stdlib ast)."
    order = 40

    def applies(self, ctx: BuildContext) -> bool:
        return any(p.suffix in SUFFIXES for p in ctx.iter_files())

    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        files = [p for p in ctx.iter_files() if p.suffix in SUFFIXES and ctx.readable(p)]
        if not files:
            return

        def env(method: Method, path: str, line: int | None = None,
                end: int | None = None, note: str = "") -> Envelope:
            return Envelope.make(
                method, source=self.id, as_of=ctx.commit, ts=ctx.ts,
                status=Status.FRESH if ctx.commit else Status.UNVERIFIED,
                evidence=(Evidence(path=path, start_line=line, end_line=end,
                                   commit=ctx.commit or None),),
                note=note,
            )

        modules: dict[str, ModuleInfo] = {}
        all_symbols: set[str] = set()
        candidates: list[Candidate] = []
        unparsed: list[dict[str, str]] = []
        external: dict[str, int] = {}

        # -- pass one -----------------------------------------------------
        for path in sorted(files):
            rel = ctx.rel(path)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except (SyntaxError, UnicodeDecodeError, OSError, ValueError) as exc:
                unparsed.append({"path": rel, "reason": f"{type(exc).__name__}: {exc}"})
                continue

            info = ModuleInfo(rel=rel, dotted=dotted(rel),
                              is_package=rel.endswith("__init__.py"))
            modules[info.dotted] = info
            read_imports(info, tree, external)

            source_lines = getattr(tree, "end_lineno", None)
            yield Node(
                layer=Layer.L1, kind="module", key=rel, name=info.dotted,
                env=env(Method.EXTRACTED, rel),
                attrs={"path": rel, "package": info.is_package, "lines": source_lines,
                       "docstring": bool(ast.get_docstring(tree))},
            )
            yield Edge(layer=Layer.L1, kind="contains",
                       src=f"{Layer.L0}:file:{rel}", dst=info.node_id,
                       env=env(Method.EXTRACTED, rel))

            defs = symbol_defs(tree)
            for qual, _kind, _node in defs:
                info.symbols.add(qual)

            # A qualname can legitimately appear more than once in one module:
            # property getter/setter pairs, @overload stubs, and definitions
            # branched on platform or version. Suffixing the later ones keeps
            # every definition instead of letting one silently overwrite
            # another. The suffix is source order, so it does not churn when
            # unrelated lines move.
            occurrences: dict[str, int] = {}

            for qual, kind, node in defs:
                occurrences[qual] = occurrences.get(qual, 0) + 1
                nth = occurrences[qual]
                unique = qual if nth == 1 else f"{qual}~{nth}"
                sym_id = f"{Layer.L1}:symbol:{rel}#{unique}"
                all_symbols.add(sym_id)
                end = getattr(node, "end_lineno", None)
                leaf = qual.rsplit(".", 1)[-1]
                yield Node(
                    layer=Layer.L1, kind="symbol", key=f"{rel}#{unique}", name=leaf,
                    env=env(Method.EXTRACTED, rel, node.lineno, end),
                    attrs={"symbol_kind": kind, "module": rel, "qualname": qual,
                           "lines": (end - node.lineno + 1) if end else None,
                           "docstring": bool(ast.get_docstring(node)),
                           "private": leaf.startswith("_"),
                           **({"redefinition": nth} if nth > 1 else {})},
                )
                yield Edge(layer=Layer.L1, kind="defines", src=info.node_id, dst=sym_id,
                           env=env(Method.EXTRACTED, rel, node.lineno))
                candidates.extend(collect_calls(info, qual, sym_id, node, rel))

        # -- pass two: imports --------------------------------------------
        seen_imports: set[tuple[str, str]] = set()
        for info in modules.values():
            for target_dotted, line in info.sites:
                target = modules.get(target_dotted)
                if target is None or target.rel == info.rel:
                    continue  # external, or a package importing itself
                key = (info.node_id, target.node_id)
                if key in seen_imports:
                    continue
                seen_imports.add(key)
                yield Edge(layer=Layer.L1, kind="imports", src=info.node_id,
                           dst=target.node_id, env=env(Method.EXTRACTED, info.rel, line))

        # -- pass two: calls ----------------------------------------------
        seen_calls: set[tuple[str, str]] = set()
        for cand in candidates:
            if cand.module is None:
                target_id = f"{Layer.L1}:symbol:{cand.path}#{cand.name}"
            else:
                target_module = modules.get(cand.module)
                if target_module is None:
                    continue  # third-party or stdlib — not ours to claim
                target_id = f"{Layer.L1}:symbol:{target_module.rel}#{cand.name}"

            if target_id not in all_symbols or target_id == cand.src:
                continue
            key = (cand.src, target_id)
            if key in seen_calls:
                continue
            seen_calls.add(key)
            yield Edge(
                layer=Layer.L1, kind="calls", src=cand.src, dst=target_id,
                env=env(cand.method, cand.path, cand.line,
                        note="" if cand.method is Method.EXTRACTED
                        else "resolved through an import; Python names can be rebound"),
            )

        # -- repo level ----------------------------------------------------
        third_party = {name: n for name, n in external.items()
                       if name not in modules and name not in STDLIB}

        yield Fact(layer=Layer.L1, subject=REPO, predicate="python_summary",
                   value={"modules": len(modules), "symbols": len(all_symbols),
                          "packages": sum(1 for m in modules.values() if m.is_package),
                          "call_edges": len(seen_calls),
                          "import_edges": len(seen_imports)},
                   env=env(Method.EXTRACTED, "."))

        if third_party:
            yield Fact(
                layer=Layer.L1, subject=REPO, predicate="python_third_party_imports",
                value=dict(sorted(third_party.items(), key=lambda kv: (-kv[1], kv[0]))[:50]),
                env=env(Method.DERIVED, ".",
                        "top-level import names that are neither stdlib nor in-repo"),
            )

        if unparsed:
            # Principle vi: unknowns are first-class. An agent must be able to
            # see the edge of the map, not assume these files are empty.
            yield Fact(layer=Layer.L1, subject=REPO, predicate="python_unparsed_files",
                       value=unparsed[:50], env=env(Method.EXTRACTED, "."))


register(PythonStructureProvider())
