"""L3 semantics — what this code means, as far as structure can tell.

This is the layer people expect a language model to write, and the temptation is
to have one narrate the domain and call the result knowledge. That fails
principle i twice over: it is expensive on every build, and it produces
confident prose no one can check.

So the deterministic part is done deterministically. Bounded-context candidates
come from module cohesion — a directory whose modules import each other far more
than they import outward *is* a boundary, whatever anyone calls it. Ubiquitous
language comes from the vocabulary the code actually uses, weighted by how
widely each term is shared. Entity candidates come from classes that recur
across contexts.

What is left — business rules, invariants in prose, why a boundary sits where it
does — needs a model, and this provider does not pretend otherwise. It records
the gap as a fact so the absence is visible instead of being mistaken for
"this repository has no domain". An LLM-backed provider can register alongside
this one and its claims will be marked INFERRED, ranked below these, and
overridden by any human who disagrees.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from ..envelope import Envelope, Evidence, Method, Status
from ..model import REPO, Fact, Layer, Node, Record
from ..pack import is_test_path
from ..providers import BuildContext, Provider, register

#: A context needs enough modules to be a boundary rather than a coincidence.
MIN_CONTEXT_MODULES = 2

#: Vocabulary shared by fewer modules than this is local jargon, not domain
#: language.
MIN_TERM_MODULES = 2

#: Programming vocabulary carries no domain meaning. Excluded so the result is
#: about the business, not about Python.
NOISE = frozenset("""
get set add remove delete update create new init main run test tests self cls
value values key keys item items name names list dict set str int bool float
data info obj object type kind base impl util utils helper helpers common core
lib libs src main app config settings const args kwargs param params arg
result results out output input inputs res req err error errors exception
handler handle handles process processed make build parse format render load
save read write open close start stop begin end first last next prev current
temp tmp foo bar baz qux num count total index idx iter next node file path
class def return yield import from none true false null undefined
and the not are was were has have had for with without into onto only just
does did doing can cannot could should would when where which what who why how
its their there here then than that this these those some any all both each
own same too very will still ever never always also because while about after
before between during through against above below over under again once
""".split())

#: Test names are English sentences, not domain language. A method called
#: `test_a_refuted_claim_can_never_reach_a_pack` contributes "refuted" and
#: "claim" — but also "never", "reach", "a" — and there are far more test
#: symbols than domain symbols in a well-tested repository, so left in they
#: drown the vocabulary and fill the entity list with `TestFoo` classes.
SKIP_TEST_CODE = True

WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def terms_of(name: str) -> set[str]:
    out: set[str] = set()
    for chunk in name.replace("-", "_").split("_"):
        for piece in CAMEL.findall(chunk):
            lowered = piece.lower()
            if len(lowered) > 2 and lowered not in NOISE and not lowered.isdigit():
                out.add(lowered)
    return out


def context_of(module_path: str) -> str | None:
    """The boundary a module sits in: its owning directory."""
    parts = module_path.split("/")
    if len(parts) < 2:
        return None
    # Skip a single top-level source wrapper so `src/payments/api.py` reads as
    # `payments`, not `src`.
    if parts[0] in ("src", "lib", "app", "pkg", "internal") and len(parts) > 2:
        return "/".join(parts[:2])
    return parts[0]


class SemanticsProvider(Provider):
    id = "semantics"
    layers = (Layer.L3,)
    description = "Bounded-context candidates and ubiquitous language, from structure."
    order = 70  # needs L1 modules and imports
    derivative = True

    def applies(self, ctx: BuildContext) -> bool:
        brain = ctx.brain
        return brain is not None and any(
            n.kind == "module" and n.layer is Layer.L1 for n in brain.nodes.values())

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

        modules = {n.key: n for n in brain.nodes.values()
                   if n.layer is Layer.L1 and n.kind == "module"}
        if not modules:
            return

        # -- contexts, from import cohesion -------------------------------
        members: dict[str, list[str]] = defaultdict(list)
        for path in modules:
            if SKIP_TEST_CODE and is_test_path(path):
                continue  # a test directory is not a bounded context
            context = context_of(path)
            if context:
                members[context].append(path)

        internal: Counter[str] = Counter()
        outward: Counter[str] = Counter()
        crossings: Counter[tuple[str, str]] = Counter()
        for edge in brain.edges.values():
            if edge.kind != "imports":
                continue
            src = edge.src.split(":module:", 1)[-1]
            dst = edge.dst.split(":module:", 1)[-1]
            a, b = context_of(src), context_of(dst)
            if not a or not b:
                continue
            if a == b:
                internal[a] += 1
            else:
                outward[a] += 1
                crossings[(a, b)] += 1

        contexts = []
        for context, paths in sorted(members.items()):
            if len(paths) < MIN_CONTEXT_MODULES:
                continue
            inside, outside = internal[context], outward[context]
            total = inside + outside
            cohesion = inside / total if total else 0.0
            contexts.append(context)
            yield Node(
                layer=Layer.L3, kind="context", key=context, name=context,
                # A boundary read off the import graph is a reading, not a fact:
                # the code may cohere for reasons that are not domain reasons.
                env=env(Method.DERIVED, context,
                        "cohesion = internal imports / all imports from this directory",
                        confidence=round(0.55 + 0.4 * cohesion, 2)),
                attrs={"modules": len(paths), "internal_imports": inside,
                       "outward_imports": outside, "cohesion": round(cohesion, 3)},
            )

        for (a, b), count in sorted(crossings.items(), key=lambda kv: (-kv[1], kv[0])):
            if a in contexts and b in contexts and count >= 2:
                yield Fact(
                    layer=Layer.L3, subject=f"{Layer.L3}:context:{a}",
                    predicate=f"depends_on:{b}", value={"imports": count},
                    env=env(Method.DERIVED, a),
                )

        # -- vocabulary ----------------------------------------------------
        term_modules: dict[str, set[str]] = defaultdict(set)
        term_contexts: dict[str, set[str]] = defaultdict(set)
        for node in brain.nodes.values():
            if node.layer is not Layer.L1 or node.kind != "symbol":
                continue
            module = str(node.attrs.get("module", ""))
            if SKIP_TEST_CODE and is_test_path(module):
                continue
            context = context_of(module) or ""
            for term in terms_of(node.name or ""):
                term_modules[term].add(module)
                if context:
                    term_contexts[term].add(context)

        shared = {t: m for t, m in term_modules.items() if len(m) >= MIN_TERM_MODULES}
        ranked = sorted(shared.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        if ranked:
            yield Fact(
                layer=Layer.L3, subject=REPO, predicate="ubiquitous_language",
                value=[{"term": term, "modules": len(mods),
                        "contexts": sorted(term_contexts[term])[:5]}
                       for term, mods in ranked[:40]],
                env=env(Method.DERIVED, ".",
                        "identifier vocabulary shared across modules"),
            )

        for context in contexts:
            local = [(t, m) for t, m in shared.items()
                     if context in term_contexts[t] and len(term_contexts[t]) == 1]
            local.sort(key=lambda kv: (-len(kv[1]), kv[0]))
            if local:
                yield Fact(
                    layer=Layer.L3, subject=f"{Layer.L3}:context:{context}",
                    predicate="vocabulary",
                    value=[t for t, _ in local[:15]],
                    env=env(Method.DERIVED, context,
                            "terms used only inside this context"),
                )

        # -- entity candidates ---------------------------------------------
        entities = []
        for node in brain.nodes.values():
            if (node.layer is not Layer.L1 or node.kind != "symbol"
                    or node.attrs.get("symbol_kind") != "class"):
                continue
            if SKIP_TEST_CODE and is_test_path(str(node.attrs.get("module", ""))):
                continue
            name = node.name or ""
            spread = max((len(term_modules.get(t, ())) for t in terms_of(name)),
                         default=0)
            if spread >= MIN_TERM_MODULES:
                entities.append({"name": name,
                                 "module": str(node.attrs.get("module", "")),
                                 "shared_across_modules": spread})
        entities.sort(key=lambda e: (-e["shared_across_modules"], e["name"]))
        if entities:
            yield Fact(
                layer=Layer.L3, subject=REPO, predicate="entity_candidates",
                value=entities[:30],
                env=env(Method.DERIVED, ".",
                        "classes whose vocabulary recurs across modules"),
            )

        # -- the honest gap -------------------------------------------------
        yield Fact(
            layer=Layer.L3, subject=REPO, predicate="semantics_coverage_gap",
            value={"derived_from": "module cohesion and identifier vocabulary",
                   "missing": ["business rules", "entity relationships",
                               "why a boundary sits where it does",
                               "domain meaning of terms"],
                   "reason": "no language-model pass has run; the deterministic "
                             "core takes no API key",
                   "impact": "contexts here are candidates read off imports, not "
                             "a domain model anyone has agreed to"},
            env=env(Method.EXTRACTED, "."),
        )


register(SemanticsProvider())
