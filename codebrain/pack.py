"""The context pack compiler.

Given a task, a token budget and a role, produce the smallest bundle of cited
facts sufficient to do the job correctly.

This is graph traversal with typed expansion and a budget fit — not similarity
search over chunks. Lexical scoring resolves a fuzzy task description to
concrete anchors, and everything after that is structural, which is why the
result is explainable and reproducible: the same task against the same Brain
produces the same pack, and every line in it can be traced to a record.

Six facets, because six different kinds of ignorance cause six different
failures:

    anchors        touching the wrong code
    blast radius   breaking callers you did not know existed
    contracts      changing a shape someone else depends on
    precedent      inventing a pattern the repo already has
    constraints    violating an invariant or editing frozen code
    runbook        being unable to check your own work
    unknowns       guessing confidently

The budget fit guarantees the small critical facets — constraints, runbook,
unknowns — before letting the large ones compete, because a pack that spends
its whole budget on blast radius and drops the "this file is frozen" line has
failed at exactly the moment it mattered.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .envelope import Method, Status
from .model import REPO, Brain, Edge, Layer, Node

#: Rough characters-per-token. Deliberately a heuristic — the real tokenizer
#: belongs to whichever model consumes the pack, and being slightly
#: conservative is the safe direction to be wrong in.
CHARS_PER_TOKEN = 4

DEFAULT_BUDGET = 6000

#: Facets that are small, critical, and reserved before the large ones compete.
GUARANTEED = ("anchors", "constraints", "runbook", "unknowns")

#: Share of the budget the guaranteed facets may claim before yielding.
GUARANTEE_SHARE = 0.45

STOPWORDS = frozenset("""
a an the and or but if then else for to from in on at by of with without into
is are was were be been being do does did doing have has had having i we you
it its this that these those add adds added new make makes making use using
used need needs want wants please can could should would will shall may might
implement implementing support fix fixes fixing update updating change changing
""".split())

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
_PATHISH = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]{1,6}")


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def split_identifier(name: str) -> set[str]:
    """`charge_endpoint` and `chargeEndpoint` both yield {charge, endpoint}."""
    parts: set[str] = set()
    for chunk in name.replace("-", "_").split("_"):
        if not chunk:
            continue
        parts.add(chunk.lower())
        for piece in _CAMEL.findall(chunk):
            if len(piece) > 1:
                parts.add(piece.lower())
    return parts


def tokenize(task: str) -> set[str]:
    terms: set[str] = set()
    for word in _WORD.findall(task):
        low = word.lower()
        if low in STOPWORDS or len(low) < 2:
            continue
        terms.add(low)
        terms |= split_identifier(word)
    return {t for t in terms if t not in STOPWORDS and len(t) > 1}


def is_test_path(path: str) -> bool:
    lowered = path.lower()
    parts = lowered.replace("\\", "/").split("/")
    name = parts[-1] if parts else lowered
    return (any(p in ("test", "tests", "spec", "specs", "__tests__") for p in parts[:-1])
            or name.startswith(("test_", "spec_"))
            or name.endswith(("_test.py", "_spec.py", ".test.ts", ".spec.ts",
                              ".test.js", ".spec.js", ".test.tsx", ".spec.tsx")))


def mentioned_paths(task: str) -> set[str]:
    """File paths written out in the task are the strongest signal available."""
    return {m.group(0).replace("\\", "/").lstrip("./") for m in _PATHISH.finditer(task)}


def provenance_tag(env) -> str:
    if env.status is Status.REFUTED:
        return "REFUTED"
    if env.method is Method.OBSERVED:
        return "OBSERVED"
    if env.method is Method.DERIVED:
        return f"DERIVED {env.confidence:.2f}"
    if env.method is Method.INFERRED:
        return f"INFERRED {env.confidence:.2f}"
    if env.method is Method.ASSERTED:
        return "ASSERTED"
    return "EXTRACTED"


@dataclass(slots=True)
class Item:
    facet: str
    text: str
    score: float
    record_id: str = ""
    tokens: int = 0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = estimate_tokens(self.text)

    @property
    def density(self) -> float:
        return self.score / max(1, self.tokens)


@dataclass(slots=True)
class Pack:
    task: str
    budget: int
    commit: str = ""
    repo: str = ""
    items: list[Item] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(i.tokens for i in self.items)

    def by_facet(self) -> dict[str, list[Item]]:
        grouped: dict[str, list[Item]] = defaultdict(list)
        for item in self.items:
            grouped[item.facet].append(item)
        return grouped

    # -- output ------------------------------------------------------------

    LABELS = (
        ("anchors", "ANCHORS"),
        ("blast_radius", "BLAST RADIUS"),
        ("contracts", "CONTRACTS"),
        ("precedent", "PRECEDENT"),
        ("constraints", "CONSTRAINTS"),
        ("runbook", "RUNBOOK"),
        ("unknowns", "UNKNOWNS"),
    )

    def render(self) -> str:
        if not self.items:
            return (f"CONTEXT PACK · no anchors matched \"{self.task}\"\n\n"
                    "The Brain has nothing specific for this task. Fall back to "
                    "ordinary search — and treat that as a gap worth recording.")

        head = (f"CONTEXT PACK · task: {self.task} · "
                f"{self.tokens}/{self.budget} tokens")
        if self.commit:
            head += f" · brain @{self.commit[:8]}"

        lines = [head, ""]
        grouped = self.by_facet()
        for key, label in self.LABELS:
            entries = grouped.get(key)
            if not entries:
                continue
            lines.append(label)
            for item in entries:
                lines.append(f"  {item.text}")
            lines.append("")

        if self.dropped:
            summary = ", ".join(f"{n} {facet.replace('_', ' ')}"
                                for facet, n in sorted(self.dropped.items()))
            # Never let truncation read as "there was nothing more".
            lines.append(f"[budget] {summary} omitted to fit {self.budget} tokens")
        for note in self.notes:
            lines.append(f"[note] {note}")
        return "\n".join(lines).rstrip() + "\n"

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task, "budget": self.budget, "tokens": self.tokens,
            "commit": self.commit, "repo": self.repo, "anchors": self.anchors,
            "dropped": self.dropped, "notes": self.notes,
            "facets": {facet: [{"text": i.text, "score": round(i.score, 2),
                                "tokens": i.tokens, "record": i.record_id}
                               for i in items]
                       for facet, items in self.by_facet().items()},
        }


#: Bounds on the content sweep. Measured rather than guessed: 6,932 files and
#: 42.5 MB of Django cost 1.4s, so the limits sit well above a large real
#: repository and only a monorepo trips them. An earlier, arbitrary cap of 4,000
#: files disabled content matching on exactly the codebases that needed it most,
#: which cost more in retrieval quality than it ever saved in time.
CONTENT_SCAN_LIMIT = 30_000
CONTENT_SCAN_BYTES = 250_000_000

#: Content matches are weaker evidence than a symbol whose name matches, but
#: they are the only way to find code described by behaviour rather than named.
CONTENT_WEIGHT = 2.2


class Compiler:
    def __init__(self, brain: Brain, root: Any = None) -> None:
        self.brain = brain
        self.root = Path(root) if root is not None else None
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)
        for edge in brain.edges.values():
            self._out[edge.src].append(edge)
            self._in[edge.dst].append(edge)
        self.content_skipped = False

    # -- anchors -----------------------------------------------------------

    def content_anchors(self, terms: set[str], wants_tests: bool,
                        limit: int) -> list[tuple[Node, float]]:
        """Score files by what they contain, not by what they are called.

        Structural matching cannot find code described by behaviour — "add
        hasattr checks for protocol isinstance" names no symbol in the
        repository. Plain search can, which is precisely why a pack that only
        matched names lost to grep on those tasks. Reading content makes the
        pack a superset of the search result rather than an alternative to it.
        """
        if self.root is None or not terms:
            return []

        files = [n for n in self.brain.nodes.values()
                 if n.layer is Layer.L0 and n.kind == "file" and n.attrs.get("text")]
        if len(files) > CONTENT_SCAN_LIMIT:
            self.content_skipped = True
            return []

        scored: list[tuple[Node, float]] = []
        budget_bytes = CONTENT_SCAN_BYTES
        for node in files:
            path = self.root / node.key
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except (OSError, ValueError):
                continue
            budget_bytes -= len(text)
            if budget_bytes < 0:
                self.content_skipped = True
                break
            score = 0.0
            for term in terms:
                hits = text.count(term)
                if hits:
                    score += 1.0 + min(hits, 20) * 0.1
            if score <= 0:
                continue
            score *= CONTENT_WEIGHT
            if not wants_tests and is_test_path(node.key):
                score *= 0.45
            scored.append((node, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:limit]

    def score_anchors(self, task: str, limit: int = 8) -> list[tuple[Node, float]]:
        terms = tokenize(task)
        paths = mentioned_paths(task)
        if not terms and not paths:
            return []

        wants_tests = bool({"test", "tests", "testing"} & terms)
        scored: list[tuple[Node, float]] = []

        for node in self.brain.nodes.values():
            if node.layer not in (Layer.L0, Layer.L1):
                continue
            if not node.env.usable():
                continue

            path = str(node.attrs.get("module") or node.attrs.get("path") or node.key)
            path = path.split("#", 1)[0]
            name = node.name or node.key.rsplit("#", 1)[-1]

            score = 0.0
            if paths and any(path.endswith(p) or p in path for p in paths):
                score += 15.0

            name_terms = split_identifier(name)
            exact = {name.lower()} & terms
            if exact:
                score += 12.0
            overlap = name_terms & terms
            score += 4.0 * len(overlap)

            path_terms: set[str] = set()
            for part in path.replace("/", " ").replace(".", " ").split():
                path_terms |= split_identifier(part)
            score += 2.0 * len(path_terms & terms)

            if score <= 0:
                continue

            # A symbol is a more precise place to start work than a whole file.
            if node.kind == "symbol":
                score *= 1.25
            elif node.kind == "file":
                score *= 0.8
            if not wants_tests and is_test_path(path):
                score *= 0.45
            if node.attrs.get("private"):
                score *= 0.85

            scored.append((node, score))

        # Union, not fallback: a file whose *contents* match is evidence even
        # when a symbol name also matched, and the two signals reinforce.
        merged: dict[str, tuple[Node, float]] = {}
        for node, value in scored:
            merged[node.id] = (node, value)
        for node, value in self.content_anchors(terms, wants_tests, limit * 2):
            existing = merged.get(node.id)
            merged[node.id] = (node, (existing[1] if existing else 0.0) + value)

        ranked = sorted(merged.values(), key=lambda pair: (-pair[1], pair[0].id))
        return ranked[:limit]

    # -- facets ------------------------------------------------------------

    def _file_id_of(self, node: Node) -> str:
        path = str(node.attrs.get("module") or node.attrs.get("path") or node.key)
        return f"{Layer.L0}:file:{path.split('#', 1)[0]}"

    def _location(self, node: Node) -> str:
        evidence = node.env.evidence[0] if node.env.evidence else None
        if evidence and evidence.start_line:
            return f"{evidence.path}:{evidence.start_line}"
        return str(node.attrs.get("path") or node.key.split("#", 1)[0])

    def anchors(self, scored: list[tuple[Node, float]]) -> list[Item]:
        items = []
        for node, score in scored:
            kind = node.attrs.get("symbol_kind") or node.kind
            items.append(Item("anchors", f"{self._location(node)}  {node.name} ({kind})",
                              score, node.id))
        return items

    def blast_radius(self, scored: list[tuple[Node, float]], depth: int = 2) -> list[Item]:
        """Who depends on the anchors. Breadth-first, decaying with distance.

        Test code is collapsed into a single count rather than enumerated. Tests
        depend on everything, so listing them crowds out the production callers
        that are the actual reason to look — but "and 40 tests cover this" is
        itself worth one line.
        """
        seen: dict[str, int] = {}
        anchor_ids = {n.id for n, _ in scored}
        frontier = list(anchor_ids)
        for level in range(1, depth + 1):
            nxt: list[str] = []
            for node_id in frontier:
                for edge in self._in.get(node_id, ()):
                    if edge.kind not in ("calls", "imports"):
                        continue
                    if edge.src in seen or edge.src in anchor_ids:
                        continue
                    seen[edge.src] = level
                    nxt.append(edge.src)
            frontier = nxt
            if not frontier:
                break

        items: list[Item] = []
        test_files: set[str] = set()
        test_count = 0

        for node_id, level in seen.items():
            node = self.brain.nodes.get(node_id)
            if node is None:
                continue
            path = str(node.attrs.get("module") or node.attrs.get("path")
                       or node.key).split("#", 1)[0]
            if is_test_path(path):
                test_count += 1
                test_files.add(path)
                continue
            items.append(Item(
                "blast_radius",
                f"{self._location(node)}  {node.name}  "
                f"({'direct' if level == 1 else f'{level} hops'}) "
                f"[{provenance_tag(node.env)}]",
                10.0 / level, node_id,
            ))

        items.sort(key=lambda i: (-i.score, i.text))
        if test_count:
            items.append(Item(
                "blast_radius",
                f"+ {test_count} test symbol(s) across {len(test_files)} test file(s) "
                f"also depend on this — they will fail loudly if you break it",
                3.0,
            ))
        return items

    def contracts(self, scored: list[tuple[Node, float]]) -> list[Item]:
        """Public surface of the anchor modules: symbols other modules reach into."""
        anchor_modules = {self._file_id_of(node).split(":file:", 1)[-1]
                          for node, _ in scored}
        items: list[Item] = []
        for node in self.brain.nodes.values():
            if node.kind != "symbol" or node.layer is not Layer.L1:
                continue
            module = str(node.attrs.get("module", ""))
            if module not in anchor_modules or node.attrs.get("private"):
                continue
            external = [e for e in self._in.get(node.id, ())
                        if e.kind == "calls"
                        and str(self.brain.nodes.get(e.src, node).attrs.get("module", ""))
                        != module]
            if not external:
                continue
            items.append(Item(
                "contracts",
                f"{self._location(node)}  {node.name}  "
                f"— {len(external)} external caller(s); signature is load-bearing "
                f"[{provenance_tag(node.env)}]",
                6.0 + len(external), node.id,
            ))
        items.sort(key=lambda i: (-i.score, i.text))
        return items

    def precedent(self, scored: list[tuple[Node, float]]) -> list[Item]:
        """How this part of the repo has changed before."""
        anchor_files = {self._file_id_of(node) for node, _ in scored}
        items: list[Item] = []

        for edge in self.brain.edges.values():
            if edge.kind != "changed_with":
                continue
            for near, far in ((edge.src, edge.dst), (edge.dst, edge.src)):
                if near not in anchor_files:
                    continue
                strength = float(edge.attrs.get("strength", 0))
                items.append(Item(
                    "precedent",
                    f"{far.split(':file:', 1)[-1]}  changes with "
                    f"{near.split(':file:', 1)[-1]} "
                    f"({edge.attrs.get('commits', 0)} shared commits, "
                    f"{strength:.0%}) [{provenance_tag(edge.env)}]",
                    4.0 + 6.0 * strength, edge.id,
                ))

        for file_id in sorted(anchor_files):
            churn = self.brain.fact(file_id, "churn", Layer.L4)
            if churn is None:
                continue
            value = churn.value or {}
            items.append(Item(
                "precedent",
                f"{file_id.split(':file:', 1)[-1]}  {value.get('commits', 0)} commits, "
                f"{value.get('insertions', 0)}+/{value.get('deletions', 0)}- "
                f"in the last {value.get('window_commits', '?')} commits "
                f"[{provenance_tag(churn.env)}]",
                5.0, churn.id,
            ))
        items.sort(key=lambda i: (-i.score, i.text))
        return items

    def constraints(self, scored: list[tuple[Node, float]]) -> list[Item]:
        anchor_files = {self._file_id_of(node) for node, _ in scored}
        items: list[Item] = []
        for file_id in sorted(anchor_files):
            path = file_id.split(":file:", 1)[-1]

            review = self.brain.fact(file_id, "requires_review", Layer.L6)
            if review:
                owners = " ".join((review.value or {}).get("owners", []))
                items.append(Item("constraints",
                                  f"{path}  needs review from {owners} "
                                  f"[{provenance_tag(review.env)}]",
                                  20.0, review.id))

            danger = self.brain.fact(file_id, "danger_zone", Layer.L6)
            if danger:
                value = danger.value or {}
                items.append(Item("constraints",
                                  f"{path}  hotspot #{value.get('rank')} — "
                                  f"{value.get('commits')} commits; "
                                  f"{value.get('reason')} "
                                  f"[{provenance_tag(danger.env)}]",
                                  18.0, danger.id))

            bus = self.brain.fact(file_id, "bus_factor_risk", Layer.L6)
            if bus:
                value = bus.value or {}
                items.append(Item("constraints",
                                  f"{path}  only {value.get('primary_author')} has "
                                  f"ever changed this [{provenance_tag(bus.env)}]",
                                  14.0, bus.id))
        return items

    def runbook(self) -> list[Item]:
        items: list[Item] = []
        for intent in ("test", "build", "lint", "run"):
            found = self.brain.fact(REPO, f"{intent}_command", Layer.L5)
            if found is None:
                continue
            tag = provenance_tag(found.env)
            caveat = "" if found.env.method is Method.OBSERVED else "  (never executed)"
            items.append(Item("runbook",
                              f"{intent:<6} {found.value}  [{tag}]{caveat}",
                              20.0 if intent == "test" else 12.0, found.id))
        return items

    def unknowns(self, scored: list[tuple[Node, float]]) -> list[Item]:
        """The edge of the map. Cheap, and the difference between a careful
        agent and a confident wrong one."""
        items: list[Item] = []
        for predicate in ("typescript_coverage_gap", "python_unparsed_files",
                          "history_coverage_gap"):
            for layer in (Layer.L1, Layer.L4):
                found = self.brain.fact(REPO, predicate, layer)
                if found is None:
                    continue
                value = found.value
                if predicate == "typescript_coverage_gap":
                    text = f"TS/JS has no call graph — {value.get('impact')}"
                elif predicate == "python_unparsed_files":
                    text = (f"{len(value)} Python file(s) could not be parsed and are "
                            "missing from the structure layer")
                else:
                    text = (f"{value.get('files_without_history')} file(s) have no "
                            f"history signal ({value.get('reason')})")
                items.append(Item("unknowns", text, 10.0, found.id))
                break

        inferred = sum(1 for node, _ in scored
                       if node.env.method in (Method.DERIVED, Method.INFERRED))
        if inferred:
            items.append(Item("unknowns",
                              f"{inferred} of the anchors are inferred rather than "
                              "extracted — verify before relying on them", 8.0))
        return items

    # -- budget ------------------------------------------------------------

    def fit(self, pack: Pack, groups: dict[str, list[Item]]) -> None:
        """Reserve the small critical facets, then let the rest compete."""
        budget = pack.budget
        used = 0
        taken: list[Item] = []
        dropped: dict[str, int] = defaultdict(int)

        reserve = int(budget * GUARANTEE_SHARE)
        for facet in GUARANTEED:
            for item in groups.get(facet, []):
                if used + item.tokens <= reserve or not taken:
                    taken.append(item)
                    used += item.tokens
                else:
                    dropped[facet] += 1

        rest: list[Item] = []
        for facet, items in groups.items():
            if facet not in GUARANTEED:
                rest.extend(items)
        rest.sort(key=lambda i: (-i.density, i.text))

        for item in rest:
            if used + item.tokens <= budget:
                taken.append(item)
                used += item.tokens
            else:
                dropped[item.facet] += 1

        order = {key: n for n, (key, _) in enumerate(Pack.LABELS)}
        taken.sort(key=lambda i: (order.get(i.facet, 99), -i.score, i.text))
        pack.items = taken
        pack.dropped = dict(dropped)


def compile_pack(brain: Brain, task: str, budget: int = DEFAULT_BUDGET,
                 max_anchors: int = 8, root: Any = None) -> Pack:
    compiler = Compiler(brain, root=root)
    pack = Pack(task=task.strip(), budget=budget,
                commit=brain.manifest.as_of, repo=brain.manifest.repo)

    scored = compiler.score_anchors(task, limit=max_anchors)
    if not scored:
        return pack

    pack.anchors = [node.id for node, _ in scored]
    groups = {
        "anchors": compiler.anchors(scored),
        "blast_radius": compiler.blast_radius(scored),
        "contracts": compiler.contracts(scored),
        "precedent": compiler.precedent(scored),
        "constraints": compiler.constraints(scored),
        "runbook": compiler.runbook(),
        "unknowns": compiler.unknowns(scored),
    }

    stale = sum(1 for r in brain.records() if r.env.status is Status.STALE)
    if stale:
        pack.notes.append(f"{stale} record(s) in this Brain are stale — "
                          "run `codebrain build` if something looks wrong")
    if compiler.content_skipped:
        pack.notes.append("corpus too large for a content sweep; anchors came from "
                          "names and structure only, so behaviour-described work may "
                          "be under-served")

    compiler.fit(pack, groups)
    return pack


def brief(brain: Brain, budget: int = 500) -> str:
    """Session-start orientation. Small on purpose: it is paid for on every
    session, whether or not it is used."""
    manifest = brain.manifest
    name = (brain.fact(REPO, "repo_name") or {}) and brain.fact(REPO, "repo_name")
    lines = [f"BRAIN · {(name.value if name else None) or manifest.repo or 'this repo'}"
             f" @{(manifest.as_of or '?')[:8]}"
             + (f" on {manifest.branch}" if manifest.branch else "")]

    primary = brain.fact(REPO, "primary_language")
    files = brain.fact(REPO, "file_count")
    py = brain.fact(REPO, "python_summary", Layer.L1)
    ts = brain.fact(REPO, "typescript_summary", Layer.L1)
    shape = []
    if primary:
        shape.append(str(primary.value))
    if files:
        shape.append(f"{files.value} files")
    if py and py.value.get("modules"):
        shape.append(f"{py.value['modules']} Python modules")
    if ts and ts.value.get("modules"):
        shape.append(f"{ts.value['modules']} TS/JS modules")
    if shape:
        lines.append("  " + " · ".join(shape))

    test = brain.fact(REPO, "test_command", Layer.L5)
    if test:
        tag = provenance_tag(test.env)
        lines.append(f"  tests: {test.value}  [{tag}]")

    guarded = brain.fact(REPO, "guarded_file_count", Layer.L6)
    if guarded:
        lines.append(f"  {guarded.value} file(s) require a named reviewer "
                     "— check constraints before editing")

    stale = sum(1 for r in brain.records() if r.env.status is Status.STALE)
    if stale:
        lines.append(f"  {stale} record(s) stale — this Brain may lag HEAD")

    lines.append("  Ask for a context pack before searching: it is cheaper and cited.")
    text = "\n".join(lines)
    return text if estimate_tokens(text) <= budget else text[: budget * CHARS_PER_TOKEN]
