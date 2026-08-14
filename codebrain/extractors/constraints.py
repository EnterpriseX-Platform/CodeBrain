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
import tomllib
from pathlib import Path
from typing import Any, Iterable

from ..envelope import Envelope, Evidence, Method, Status
from ..model import REPO, Fact, Layer, Record
from ..pack import is_test_path
from ..providers import BuildContext, Provider, register

#: Where a team declares policy zones. The one place in L6 where a human states
#: a constraint outright instead of it being inferred from evidence.
POLICY_FILE = ".codebrain.toml"


def load_policy(root: Path) -> list[dict[str, Any]]:
    """Read declared policy zones. A malformed file is ignored, never fatal —
    a broken config must not take the whole Brain down with it."""
    path = Path(root) / POLICY_FILE
    if not path.is_file():
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return []

    zones: list[dict[str, Any]] = []
    declared = data.get("zone")
    if isinstance(declared, dict):  # [zone.payments] table form
        declared = [{"name": name, **body} for name, body in declared.items()
                    if isinstance(body, dict)]
    if not isinstance(declared, list):
        return []

    for entry in declared:
        if not isinstance(entry, dict):
            continue
        paths = entry.get("paths") or entry.get("path")
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            continue
        zones.append({
            "name": str(entry.get("name", "unnamed")),
            "paths": [str(p) for p in paths],
            "reason": str(entry.get("reason", "")),
            "requires": [str(r) for r in (entry.get("requires") or [])],
            "block_agents": bool(entry.get("block_agents", False)),
        })
    return zones

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
    derivative = True

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
        yield from self._public_contracts(brain, env)
        yield from self._untested(ctx, brain, env)
        yield from self._policy_zones(ctx, brain, env)

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


    # -- what may not change shape ----------------------------------------

    def _public_contracts(self, brain, env) -> Iterable[Record]:
        """Symbols other modules reach into, and routes anyone can call.

        A contract is not "this is public" in the language's sense; it is "some
        other code depends on this exact shape". That is a fact about the graph,
        not about a keyword.
        """
        callers: dict[str, set[str]] = {}
        for edge in brain.edges.values():
            if edge.kind != "calls":
                continue
            target = brain.nodes.get(edge.dst)
            source = brain.nodes.get(edge.src)
            if target is None or source is None:
                continue
            here = str(target.attrs.get("module", ""))
            there = str(source.attrs.get("module", ""))
            if here and there and here != there and not is_test_path(there):
                callers.setdefault(edge.dst, set()).add(there)

        for symbol_id, modules in sorted(callers.items()):
            if len(modules) < 2:
                continue  # one caller is coupling, not a contract
            node = brain.nodes.get(symbol_id)
            if node is None or node.attrs.get("private"):
                continue
            yield Fact(
                layer=Layer.L6, subject=symbol_id, predicate="public_contract",
                value={"consumers": sorted(modules)[:20], "consumer_count": len(modules),
                       "reason": "called from other modules; the signature is "
                                 "load-bearing"},
                env=env(Method.DERIVED, str(node.attrs.get("module", "."))),
            )

        for node in brain.nodes.values():
            if node.layer is Layer.L2 and node.kind == "route":
                yield Fact(
                    layer=Layer.L6, subject=node.id, predicate="public_contract",
                    value={"reason": "an HTTP route is callable by anyone; the "
                                     "request and response shapes are external",
                           "handler": node.attrs.get("handler"),
                           "module": node.attrs.get("module")},
                    env=env(Method.DERIVED, str(node.attrs.get("module", "."))),
                )

    # -- where a mistake will not be caught --------------------------------

    def _untested(self, ctx, brain, env) -> Iterable[Record]:
        """Churning files that no test appears to exercise.

        Churn alone says "defects land here". Churn with no test says "defects
        land here and nothing will tell you". That combination is the single
        most useful thing L6 can hand an agent.
        """
        exercised: set[str] = set()
        for edge in brain.edges.values():
            if edge.kind != "calls":
                continue
            source = brain.nodes.get(edge.src)
            target = brain.nodes.get(edge.dst)
            if source is None or target is None:
                continue
            if is_test_path(str(source.attrs.get("module", ""))):
                module = str(target.attrs.get("module", ""))
                if module:
                    exercised.add(module)

        if not exercised:
            return  # no test call graph at all: saying "untested" would be noise

        for fact in list(brain.facts.values()):
            if fact.predicate != "churn" or fact.layer is not Layer.L4:
                continue
            path = fact.subject.split(":file:", 1)[-1]
            if is_test_path(path) or path in exercised:
                continue
            if not any(path.endswith(suffix) for suffix in (".py", ".ts", ".tsx",
                                                            ".js", ".jsx")):
                continue
            commits = (fact.value or {}).get("commits", 0)
            if commits < 2:
                continue
            yield Fact(
                layer=Layer.L6, subject=fact.subject, predicate="untested_churn",
                value={"commits": commits,
                       "reason": "changes often and no test reaches it; a mistake "
                                 "here fails silently"},
                env=env(Method.DERIVED, path,
                        "absence of a test edge, not proof of no coverage"),
            )

    # -- declared policy ----------------------------------------------------

    def _policy_zones(self, ctx, brain, env) -> Iterable[Record]:
        """Zones a human declared off-limits, from .codebrain.toml.

        Everything else in L6 is inferred from evidence. This is the one place a
        human states policy directly, so these are ASSERTED and outrank anything
        the machinery derives.
        """
        zones = load_policy(ctx.root)
        if not zones:
            return

        files = [n.key for n in brain.nodes.values()
                 if n.layer is Layer.L0 and n.kind == "file"]
        guarded = 0
        for path in sorted(files):
            for zone in zones:
                if not any(matches(pattern, path) for pattern in zone["paths"]):
                    continue
                guarded += 1
                yield Fact(
                    layer=Layer.L6, subject=f"{Layer.L0}:file:{path}",
                    predicate="policy_zone",
                    value={"zone": zone["name"], "reason": zone.get("reason", ""),
                           "requires": zone.get("requires", []),
                           "block_agents": bool(zone.get("block_agents"))},
                    env=Envelope.make(
                        Method.ASSERTED, source=self.id, as_of=ctx.commit, ts=ctx.ts,
                        status=Status.FRESH,
                        evidence=(Evidence(path=POLICY_FILE,
                                           commit=ctx.commit or None),),
                        note=f"declared in {POLICY_FILE}"),
                )
                break

        if guarded:
            yield Fact(layer=Layer.L6, subject=REPO, predicate="policy_zone_count",
                       value=guarded, env=env(Method.DERIVED, POLICY_FILE))


register(ConstraintsProvider())
