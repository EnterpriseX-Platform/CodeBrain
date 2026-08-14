"""Cortex — a higher-order index over several Brains.

Real organisations are systems of systems: no single repository's Brain can
answer "which twelve services break if this field changes type?" A Cortex
composes multiple already-built Brains to reach across that boundary, on one
rule that is not negotiable — **a Cortex is composed, never re-extracted.**
Every member Brain stays exactly as its own `codebrain build` left it; Cortex
never writes to one, never merges their graphs into a new on-disk artifact, and
recomputes everything at query time from whatever is on disk right now. A
member repository that has drifted is a member repository's problem, caught by
its own drift gate — Cortex is not a second copy of that job.

What this delivers is deliberately narrower than the vision's long-run target.
Resolving "a field's type changed, who calls it" needs schema and type
extraction this project does not have. What is buildable without inventing
facts: for a route or symbol known to one member, search every other member's
*own source* for the literal string and report file:line evidence. That is
honest textual matching, not a resolved import graph — a hit is something to
go look at, never a proof of a real dependency, and every result says so.
"""

from __future__ import annotations

import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model import Layer
from .store import BRAIN_DIR, BrainNotFound
from .store import load as load_brain

CONFIG_FILE = ".codebrain-cortex.toml"

#: Bytes read per member before a scan gives up and says so. Smaller than
#: pack.py's single-repo budget because impact queries fan out across many
#: members in one call — the same total-cost discipline, spent differently.
SCAN_BYTES_PER_MEMBER = 50_000_000

#: Cross-repo hits shown per member before the rest are collapsed to a count.
HITS_SHOWN = 10


@dataclass(slots=True)
class Member:
    name: str
    root: Path
    brain_dir: Path
    brain: Any = None  # Brain | None — Any avoids importing Brain just for the hint
    error: str = ""

    @property
    def loaded(self) -> bool:
        return self.brain is not None


@dataclass(slots=True)
class Cortex:
    config_path: Path
    members: list[Member] = field(default_factory=list)

    def get(self, name: str) -> Member | None:
        return next((m for m in self.members if m.name == name), None)

    @property
    def loaded(self) -> list[Member]:
        return [m for m in self.members if m.loaded]

    @property
    def failed(self) -> list[Member]:
        return [m for m in self.members if not m.loaded]


def parse_config(text: str) -> list[dict[str, str]]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    members = data.get("member")
    if not isinstance(members, list):
        return []
    out = []
    for entry in members:
        if not isinstance(entry, dict) or "name" not in entry or "root" not in entry:
            continue
        out.append({"name": str(entry["name"]), "root": str(entry["root"]),
                   "brain": str(entry["brain"]) if entry.get("brain") else ""})
    return out


def load_cortex(config_path: Path | str) -> Cortex:
    """Load every member's Brain from disk. A member that fails to load is kept
    in the roster with its error, not silently dropped — a Cortex that quietly
    ignores a broken member gives a confidently incomplete answer, which is
    worse than an answer that admits a gap."""
    config_path = Path(config_path)
    cortex = Cortex(config_path=config_path)
    if not config_path.is_file():
        return cortex

    base = config_path.parent
    for entry in parse_config(config_path.read_text(encoding="utf-8")):
        root = (base / entry["root"]).resolve()
        brain_dir = (base / entry["brain"]).resolve() if entry["brain"] else root / BRAIN_DIR
        try:
            brain = load_brain(brain_dir)
            cortex.members.append(Member(entry["name"], root, brain_dir, brain))
        except (BrainNotFound, ValueError, OSError) as exc:
            cortex.members.append(Member(entry["name"], root, brain_dir, None, str(exc)))
    return cortex


# -- cross-repo impact -------------------------------------------------------


@dataclass(slots=True)
class Hit:
    member: str
    path: str
    line: int
    snippet: str
    score: int


@dataclass(slots=True)
class ImpactReport:
    target: str
    needle: str
    origin_member: str | None
    origin_kind: str            # route | symbol | unknown
    known_consumers: int | None  # from the origin's own L6 public_contract, if any
    hits: list[Hit] = field(default_factory=list)
    scanned: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)   # failed to load
    truncated: list[str] = field(default_factory=list)     # too large to fully scan

    def by_member(self) -> dict[str, list[Hit]]:
        grouped: dict[str, list[Hit]] = defaultdict(list)
        for hit in self.hits:
            grouped[hit.member].append(hit)
        return grouped


def _find_origin(cortex: Cortex, target: str) -> tuple[str | None, str, int | None]:
    for member in cortex.loaded:
        brain = member.brain
        route = brain.get(f"{Layer.L2}:route:{target}")
        if route is not None:
            contract = brain.fact(route.id, "public_contract", Layer.L6)
            count = (contract.value or {}).get("consumer_count") if contract else None
            return member.name, "route", count
        for node in brain.nodes.values():
            if (node.layer is Layer.L1 and node.kind == "symbol"
                    and (node.name == target or node.key.endswith(f"#{target}"))):
                return member.name, "symbol", None
    return None, "unknown", None


def _search_member(member: Member, needle: str) -> tuple[list[Hit], bool]:
    """Evidence-backed textual search over one member's own source. Mirrors
    pack.py's content_anchors: read what census already marked as text, stop
    within a byte budget, and say so rather than silently scanning less."""
    hits: list[Hit] = []
    if not member.loaded or not needle:
        return hits, False

    budget = SCAN_BYTES_PER_MEMBER
    truncated = False
    for node in member.brain.nodes.values():
        if node.layer is not Layer.L0 or node.kind != "file" or not node.attrs.get("text"):
            continue
        path = member.root / node.key
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, ValueError):
            continue
        budget -= len(text)
        if budget < 0:
            truncated = True
            break
        count = text.count(needle)
        if not count:
            continue
        first = text.find(needle)
        line = text.count("\n", 0, first) + 1
        snippet = text.splitlines()[line - 1].strip()[:140]
        hits.append(Hit(member.name, node.key, line, snippet, count))

    hits.sort(key=lambda h: (-h.score, h.path))
    return hits, truncated


def cross_repo_impact(cortex: Cortex, target: str) -> ImpactReport:
    origin_member, kind, consumers = _find_origin(cortex, target)
    # A route target looks like "POST /v1/charges" in the Brain but only the
    # path appears in a caller's source (`fetch("/v1/charges")`), so the
    # method verb is stripped before searching.
    needle = target.split(" ", 1)[1] if kind == "route" and " " in target else target

    report = ImpactReport(target=target, needle=needle, origin_member=origin_member,
                          origin_kind=kind, known_consumers=consumers)
    for member in cortex.members:
        if not member.loaded:
            report.unreachable.append(member.name)
            continue
        if member.name == origin_member:
            continue  # a repo referencing its own route is not a cross-repo hit
        report.scanned.append(member.name)
        hits, truncated = _search_member(member, needle)
        report.hits.extend(hits)
        if truncated:
            report.truncated.append(member.name)
    return report


# -- rendering ---------------------------------------------------------------


def render_roster(cortex: Cortex) -> str:
    if not cortex.members:
        return (f"No members configured. Create {CONFIG_FILE} with:\n\n"
               '  [[member]]\n  name = "payments-api"\n  root = "../payments-api"\n')

    lines = [f"Cortex - {len(cortex.members)} member(s) declared in "
            f"{cortex.config_path.name}", ""]
    for member in cortex.members:
        if not member.loaded:
            lines.append(f"  [FAIL] {member.name:<20} {member.error}")
            continue
        stats = member.brain.stats()
        routes = sum(1 for n in member.brain.nodes.values()
                    if n.layer is Layer.L2 and n.kind == "route")
        lines.append(f"  [ok]   {member.name:<20} {stats['total']:>6} records"
                     f"{f', {routes} route(s)' if routes else ''}"
                     f", {member.brain.manifest.as_of[:8] or '?'}")
    if cortex.failed:
        lines.append("")
        lines.append(f"{len(cortex.failed)} member(s) could not be loaded - "
                     "run `codebrain build` in each before querying impact.")
    return "\n".join(lines)


def render_impact(report: ImpactReport) -> str:
    lines: list[str] = []
    if report.origin_member:
        lines.append(f"{report.target}  -  defined in {report.origin_member} "
                     f"({report.origin_kind})")
        if report.known_consumers is not None:
            lines.append(f"  {report.known_consumers} known consumer(s) inside "
                         f"{report.origin_member} itself (L6 public_contract)")
    else:
        lines.append(f'{report.target!r} was not found as a route or symbol in any '
                     "loaded member -- searching every member for the literal text "
                     "instead.")
    lines.append("")

    if not report.hits:
        lines.append(f"No reference to {report.needle!r} found in any other member's "
                     "source.")
    else:
        for member, hits in sorted(report.by_member().items()):
            lines.append(f"{member} - {len(hits)} reference(s)")
            for hit in hits[:HITS_SHOWN]:
                lines.append(f"  {hit.path}:{hit.line}  {hit.snippet}")
            if len(hits) > HITS_SHOWN:
                lines.append(f"  … {len(hits) - HITS_SHOWN} more")

    lines.append("")
    lines.append(f"Scanned: {', '.join(report.scanned) or 'none'}")
    if report.unreachable:
        lines.append(f"Could not load: {', '.join(report.unreachable)}")
    if report.truncated:
        lines.append(f"Too large to fully scan: {', '.join(report.truncated)}")
    lines.append("")
    lines.append("This is textual matching over each member's own source, not a")
    lines.append("resolved import graph or a type-checked call -- a hit is evidence")
    lines.append("to go look at, not proof of a real dependency.")
    return "\n".join(lines)
