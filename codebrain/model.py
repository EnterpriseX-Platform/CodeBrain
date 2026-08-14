"""The Brain data model.

A Brain is one repository's understanding, expressed as three record types over
eight layers. Everything downstream — packs, MCP tools, the Atlas, the drift
gate — reads this and nothing else.

Record ids are readable rather than hashed, on purpose. The Brain is committed
to the repository and reviewed on pull requests (principle v), so a diff has to
be legible to a human: `L1:symbol:payments/api.py#charge_endpoint` tells a
reviewer what changed, `a3f9c2e1` does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Iterator

from .envelope import SCHEMA_VERSION, Envelope, Method, utc_now


class Layer(StrEnum):
    L0 = "L0"  # corpus     — what is this thing, and what is in it
    L1 = "L1"  # structure  — what exists and how is it wired
    L2 = "L2"  # behavior   — what it does when it runs
    L3 = "L3"  # semantics  — what it means in the business
    L4 = "L4"  # intent     — why it is like this
    L5 = "L5"  # operations — how to build, test, run, ship it
    L6 = "L6"  # constraints— what must not break
    L7 = "L7"  # memory     — what we have learned since


LAYER_NAMES: dict[Layer, str] = {
    Layer.L0: "corpus",
    Layer.L1: "structure",
    Layer.L2: "behavior",
    Layer.L3: "semantics",
    Layer.L4: "intent",
    Layer.L5: "operations",
    Layer.L6: "constraints",
    Layer.L7: "memory",
}

REPO = ""  # the subject id meaning "the repository as a whole"

_FORBIDDEN = ("\n", "\r", "\t")


def _clean(part: str, what: str) -> str:
    """Ids end up as JSONL keys and as git diff lines; they must stay one line."""
    if any(c in part for c in _FORBIDDEN):
        raise ValueError(f"{what} may not contain newlines or tabs: {part!r}")
    return part


@dataclass(slots=True)
class Node:
    """A thing that exists: a file, a symbol, a route, a domain entity, a command."""

    layer: Layer
    kind: str
    key: str
    env: Envelope
    name: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.layer}:{_clean(self.kind, 'kind')}:{_clean(self.key, 'key')}"

    def to_json(self) -> dict[str, Any]:
        d = {"t": "node", "id": self.id, "layer": str(self.layer),
             "kind": self.kind, "key": self.key, "env": self.env.to_json()}
        if self.name:
            d["name"] = self.name
        if self.attrs:
            d["attrs"] = self.attrs
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Node":
        return cls(layer=Layer(d["layer"]), kind=d["kind"], key=d["key"],
                   env=Envelope.from_json(d["env"]), name=d.get("name", ""),
                   attrs=d.get("attrs", {}))


@dataclass(slots=True)
class Edge:
    """A typed relationship: imports, calls, defines, depends_on, changed_with, constrains."""

    layer: Layer
    kind: str
    src: str
    dst: str
    env: Envelope
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.layer}:{_clean(self.kind, 'kind')}:{_clean(self.src, 'src')}->{_clean(self.dst, 'dst')}"

    def to_json(self) -> dict[str, Any]:
        d = {"t": "edge", "id": self.id, "layer": str(self.layer), "kind": self.kind,
             "src": self.src, "dst": self.dst, "env": self.env.to_json()}
        if self.attrs:
            d["attrs"] = self.attrs
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Edge":
        return cls(layer=Layer(d["layer"]), kind=d["kind"], src=d["src"], dst=d["dst"],
                   env=Envelope.from_json(d["env"]), attrs=d.get("attrs", {}))


@dataclass(slots=True)
class Fact:
    """A claim that is not graph-shaped.

    "The test command is `make test`" is a fact about the repository, not an
    edge between two things. Facts are also the unit of verification: an
    executable claim is a Fact whose `verify` attr says how to test it.
    """

    layer: Layer
    subject: str          # a node id, or REPO for the repository itself
    predicate: str
    value: Any
    env: Envelope
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.layer}:fact:{_clean(self.subject, 'subject')}|{_clean(self.predicate, 'predicate')}"

    def to_json(self) -> dict[str, Any]:
        d = {"t": "fact", "id": self.id, "layer": str(self.layer), "subject": self.subject,
             "predicate": self.predicate, "value": self.value, "env": self.env.to_json()}
        if self.attrs:
            d["attrs"] = self.attrs
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Fact":
        return cls(layer=Layer(d["layer"]), subject=d["subject"], predicate=d["predicate"],
                   value=d["value"], env=Envelope.from_json(d["env"]), attrs=d.get("attrs", {}))


Record = Node | Edge | Fact

_LOADERS = {"node": Node.from_json, "edge": Edge.from_json, "fact": Fact.from_json}


def record_from_json(d: dict[str, Any]) -> Record:
    t = d.get("t")
    if t not in _LOADERS:
        raise ValueError(f"unknown record type {t!r}")
    return _LOADERS[t](d)


@dataclass(slots=True)
class Manifest:
    schema_version: str = SCHEMA_VERSION
    codebrain_version: str = "0.1.0"
    repo: str = ""          # remote url or directory name
    as_of: str = ""         # commit the Brain was built at
    branch: str = ""
    built_at: str = ""
    providers: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "codebrain_version": self.codebrain_version,
            "repo": self.repo,
            "as_of": self.as_of,
            "branch": self.branch,
            "built_at": self.built_at,
            "providers": sorted(self.providers),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Manifest":
        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            codebrain_version=d.get("codebrain_version", "0.1.0"),
            repo=d.get("repo", ""), as_of=d.get("as_of", ""), branch=d.get("branch", ""),
            built_at=d.get("built_at", ""), providers=list(d.get("providers", [])),
        )


@dataclass(slots=True)
class MergeReport:
    added: int = 0
    replaced: int = 0
    kept: int = 0       # incoming record lost to the incumbent

    @property
    def conflicts(self) -> int:
        return self.replaced + self.kept

    def __str__(self) -> str:
        return f"+{self.added} ~{self.replaced} kept {self.kept}"


class Brain:
    """One repository's understanding.

    Records are keyed by id, so re-extraction is idempotent: a provider that
    runs twice over unchanged source produces the same ids and the same Brain.
    """

    def __init__(self, manifest: Manifest | None = None) -> None:
        self.manifest = manifest or Manifest()
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.facts: dict[str, Fact] = {}

    # -- population --------------------------------------------------------

    def _bucket(self, rec: Record) -> dict[str, Any]:
        if isinstance(rec, Node):
            return self.nodes
        if isinstance(rec, Edge):
            return self.edges
        return self.facts

    def add(self, rec: Record) -> str:
        """Insert a record, resolving collisions by envelope rank.

        Returns "added", "replaced" or "kept" so callers can report honestly
        instead of silently dropping one of two competing beliefs.
        """
        bucket = self._bucket(rec)
        rid = rec.id
        incumbent = bucket.get(rid)
        if incumbent is None:
            bucket[rid] = rec
            return "added"
        if rec.env.outranks(incumbent.env):
            bucket[rid] = rec
            return "replaced"
        return "kept"

    def extend(self, records: Iterable[Record]) -> MergeReport:
        report = MergeReport()
        for rec in records:
            outcome = self.add(rec)
            setattr(report, outcome, getattr(report, outcome) + 1)
        return report

    def merge(self, other: "Brain") -> MergeReport:
        """Fold another Brain in. Used for multi-provider builds and, later, Cortex."""
        return self.extend(other.records())

    # -- access ------------------------------------------------------------

    def records(self) -> Iterator[Record]:
        yield from self.nodes.values()
        yield from self.edges.values()
        yield from self.facts.values()

    def __len__(self) -> int:
        return len(self.nodes) + len(self.edges) + len(self.facts)

    def get(self, rid: str) -> Record | None:
        return self.nodes.get(rid) or self.edges.get(rid) or self.facts.get(rid)

    def by_layer(self, layer: Layer) -> list[Record]:
        return [r for r in self.records() if r.layer is layer]

    def fact(self, subject: str, predicate: str, layer: Layer | None = None) -> Fact | None:
        layers = [layer] if layer else list(Layer)
        for lyr in layers:
            hit = self.facts.get(f"{lyr}:fact:{subject}|{predicate}")
            if hit is not None:
                return hit
        return None

    def usable(self, floor: float = 0.0) -> Iterator[Record]:
        """Every record fit to reach a context pack. Refuted claims never appear."""
        return (r for r in self.records() if r.env.usable(floor))

    # -- reporting ---------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        by_layer: dict[str, int] = {}
        by_method: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for r in self.records():
            by_layer[str(r.layer)] = by_layer.get(str(r.layer), 0) + 1
            by_method[str(r.env.method)] = by_method.get(str(r.env.method), 0) + 1
            by_status[str(r.env.status)] = by_status.get(str(r.env.status), 0) + 1
        return {
            "total": len(self), "nodes": len(self.nodes), "edges": len(self.edges),
            "facts": len(self.facts), "by_layer": by_layer, "by_method": by_method,
            "by_status": by_status,
        }

    def validate(self) -> list[str]:
        """Structural problems, worst first. `codebrain validate` prints these."""
        problems: list[str] = []
        if self.manifest.schema_version != SCHEMA_VERSION:
            problems.append(
                f"schema version mismatch: brain is {self.manifest.schema_version}, "
                f"this codebrain speaks {SCHEMA_VERSION}"
            )
        known = set(self.nodes)
        for e in self.edges.values():
            # Dangling edges are expected mid-build across providers, but a
            # persisted Brain with them will mislead traversal.
            if e.src not in known:
                problems.append(f"edge {e.id} has unknown src {e.src}")
            if e.dst not in known:
                problems.append(f"edge {e.id} has unknown dst {e.dst}")
        for r in self.records():
            if not r.env.source:
                problems.append(f"{r.id} has no provider source — unattributable")
            if r.env.method is not Method.ASSERTED and not r.env.evidence and not isinstance(r, Edge):
                problems.append(f"{r.id} is {r.env.method} but cites no evidence")
        return problems

    def touch(self, predicate: str, reason: str = "") -> int:
        """Mark every record whose evidence lives under `predicate` (a path prefix) stale.

        This is what the PostToolUse hook calls after an agent edits a file.
        """
        n = 0
        for bucket in (self.nodes, self.edges, self.facts):
            for rid, rec in bucket.items():
                if any(ev.path.startswith(predicate) for ev in rec.env.evidence):
                    rec.env = rec.env.mark_stale(reason)
                    n += 1
        return n


def new_brain(repo: str = "", as_of: str = "", branch: str = "",
              providers: Iterable[str] = ()) -> Brain:
    return Brain(Manifest(repo=repo, as_of=as_of, branch=branch,
                          built_at=utc_now(), providers=list(providers)))
