"""L6 constraints — what must not break, and where it is dangerous.

A thin, deterministic slice. The full layer is P4 (invariants, public contracts,
compliance zones); what is here is everything derivable from what P1 already
knows, because the context pack compiler needs a constraints facet now and
inventing one from a language model would be exactly the wrong way to get it.

Every constraint carries the evidence that produced it, so an agent that is
blocked can see *why* and argue with the reason rather than the verdict.
"""

from __future__ import annotations

import fnmatch
from typing import Iterable

from ..envelope import Envelope, Evidence, Method, Status
from ..model import REPO, Fact, Layer, Record
from ..providers import BuildContext, Provider, register

#: A file in the top slice of the hotspot ranking is where defects land.
HOTSPOT_SLICE = 10

#: Below this share of commits, "one person owns this" is not a useful warning.
LONE_AUTHOR_MIN_COMMITS = 2


def matches(pattern: str, path: str) -> bool:
    """CODEOWNERS-style matching, close enough to be useful and honest about it.

    Full gitignore semantics are not implemented; the constraint records the
    pattern that matched so a human can check the call.
    """
    pattern = pattern.strip()
    if not pattern:
        return False
    if pattern.startswith("/"):
        pattern = pattern[1:]
    if pattern.endswith("/"):
        return path.startswith(pattern) or f"/{pattern}" in f"/{path}"
    if "/" not in pattern:
        return fnmatch.fnmatch(path.rsplit("/", 1)[-1], pattern)
    return (fnmatch.fnmatch(path, pattern)
            or fnmatch.fnmatch(path, pattern + "/*")
            or path.startswith(pattern.rstrip("*")))


class ConstraintsProvider(Provider):
    id = "constraints"
    layers = (Layer.L6,)
    description = "Review requirements, danger zones and bus-factor risk."
    # Last: reads L4 ownership and L5 CODEOWNERS off the partially built Brain.
    order = 90

    def applies(self, ctx: BuildContext) -> bool:
        return ctx.brain is not None

    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        brain = ctx.brain
        if brain is None:
            return

        def env(method: Method, path: str, note: str = "",
                confidence: float | None = None) -> Envelope:
            return Envelope.make(
                method, source=self.id, as_of=ctx.commit, ts=ctx.ts,
                status=Status.FRESH if ctx.commit else Status.UNVERIFIED,
                confidence=confidence,
                evidence=(Evidence(path=path, commit=ctx.commit or None),),
                note=note,
            )

        yield from self._review_rules(brain, env)
        yield from self._danger_zones(brain, env)
        yield from self._lone_authors(brain, env)

    # -- who must approve --------------------------------------------------

    def _review_rules(self, brain, env) -> Iterable[Record]:
        found = brain.fact(REPO, "codeowners", Layer.L5)
        if not found:
            return
        rules = found.value or []
        if not rules:
            return

        files = [n.key for n in brain.nodes.values()
                 if n.layer is Layer.L0 and n.kind == "file"]
        guarded = 0

        for path in sorted(files):
            owners: list[str] = []
            matched: list[str] = []
            for rule in rules:
                if matches(rule["pattern"], path):
                    # Later rules win in CODEOWNERS, so keep overwriting.
                    owners = list(rule["owners"])
                    matched = [rule["pattern"]]
            if not owners:
                continue
            guarded += 1
            yield Fact(
                layer=Layer.L6, subject=f"{Layer.L0}:file:{path}",
                predicate="requires_review",
                value={"owners": owners, "pattern": matched[0]},
                env=env(Method.DERIVED, path,
                        "CODEOWNERS match; pattern semantics are approximate"),
            )

        if guarded:
            yield Fact(layer=Layer.L6, subject=REPO, predicate="guarded_file_count",
                       value=guarded, env=env(Method.DERIVED, "."))

    # -- where it is dangerous --------------------------------------------

    def _danger_zones(self, brain, env) -> Iterable[Record]:
        found = brain.fact(REPO, "hotspots", Layer.L4)
        if not found:
            return
        hotspots = (found.value or [])[:HOTSPOT_SLICE]
        for rank, spot in enumerate(hotspots, 1):
            path = spot.get("path")
            if not path:
                continue
            yield Fact(
                layer=Layer.L6, subject=f"{Layer.L0}:file:{path}",
                predicate="danger_zone",
                value={"rank": rank, "commits": spot.get("commits"),
                       "lines_changed": spot.get("lines_changed"),
                       "reason": "high churn — historically where defects land"},
                env=env(Method.DERIVED, path,
                        "derived from git churn, not from a defect record"),
            )

    # -- who is the single point of failure --------------------------------

    def _lone_authors(self, brain, env) -> Iterable[Record]:
        for fact in list(brain.facts.values()):
            if fact.predicate != "ownership" or fact.layer is not Layer.L4:
                continue
            value = fact.value or {}
            if value.get("distinct_authors") != 1:
                continue
            path = fact.subject.split(":file:", 1)[-1]
            churn = brain.fact(fact.subject, "churn", Layer.L4)
            commits = (churn.value or {}).get("commits", 0) if churn else 0
            if commits < LONE_AUTHOR_MIN_COMMITS:
                continue
            yield Fact(
                layer=Layer.L6, subject=fact.subject, predicate="bus_factor_risk",
                value={"primary_author": value.get("primary_author"),
                       "commits": commits,
                       "reason": "only one person has ever changed this file"},
                env=env(Method.DERIVED, path),
            )


register(ConstraintsProvider())
