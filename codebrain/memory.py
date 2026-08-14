"""L7 memory — what we have learned since the last build.

The layer that makes the asset compound. Everything below L7 is re-derived from
the code on every build; this is the only layer that accumulates. A session that
resolves an unknown, discovers the Brain was wrong, or finds out which file the
work actually lived in should leave that behind, or the next agent pays for it
again.

Three rules shape it.

**An agent's claim is not a human's claim.** What an agent *did* — the files it
edited — is EXTRACTED, because it is a fact about the session. What an agent
*concluded* is INFERRED, because it is a reading. Only a person gets ASSERTED.

**An agent may dispute, not overrule.** If extraction says one thing and an
agent says another, the agent records a dispute that packs surface; it does not
silently demote an extracted fact. A model that can overwrite the AST by
asserting harder is a model that can poison every session downstream. A human
can overrule, explicitly.

**Memory ages.** A lesson from four hundred commits ago is about a different
codebase. Decay is computed at read time from the distance between when the
lesson was recorded and now — never by mutating stored confidence, which would
make two reads of the same Brain disagree.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .envelope import Envelope, Evidence, Method, Status, utc_now
from .model import REPO, Brain, Fact, Layer, Node, Record

#: Commits after which a lesson is worth half what it was. Chosen so that on an
#: active repository memory stays useful for a few months and then fades, rather
#: than accumulating into folklore nobody dares delete.
HALF_LIFE_COMMITS = 400

#: Below this, a lesson is too faded to be worth a line in a pack.
MEMORY_FLOOR = 0.15

LESSON = "lesson"
RESOLVED = "resolved_unknown"
DISPUTE = "dispute"
OUTCOME = "task_outcome"
QUESTION = "open_question"

_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str, length: int = 40) -> str:
    """A stable, readable id fragment. Same text always produces the same id, so
    recording the same lesson twice updates it instead of duplicating it."""
    cleaned = _SLUG.sub("-", text.lower()).strip("-")
    if len(cleaned) <= length:
        return cleaned or "note"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:length].rstrip('-')}-{digest}"


def memory_weight(fact: Fact, commits_now: int) -> float:
    """How much a memory is still worth, from its age in commits.

    Computed at read time and never written back: decay that mutates stored
    confidence would make a Brain change every time it is read, and two reads of
    the same commit would disagree.
    """
    recorded = fact.attrs.get("recorded_at_commits")
    if not isinstance(recorded, int) or not isinstance(commits_now, int):
        return 1.0
    elapsed = max(0, commits_now - recorded)
    return 0.5 ** (elapsed / HALF_LIFE_COMMITS)


def effective_memory_confidence(fact: Fact, commits_now: int) -> float:
    return fact.env.effective_confidence() * memory_weight(fact, commits_now)


@dataclass(slots=True)
class Session:
    """One agent session's worth of write-back."""

    session_id: str
    task: str = ""
    files: tuple[str, ...] = ()
    lessons: tuple[str, ...] = ()
    resolved: tuple[tuple[str, str], ...] = ()   # (question, answer)
    questions: tuple[str, ...] = ()
    succeeded: bool | None = None
    commit: str = ""
    commits_now: int = 0


def _env(method: Method, source: str, commit: str, evidence: tuple[Evidence, ...],
         confidence: float | None = None, note: str = "") -> Envelope:
    return Envelope.make(method, source=source, as_of=commit, ts=utc_now(),
                         status=Status.UNVERIFIED, confidence=confidence,
                         evidence=evidence, note=note)


def from_session(session: Session, source: str = "agent") -> list[Record]:
    """Turn a finished session into L7 records."""
    records: list[Record] = []
    commit = session.commit
    stamp = {"recorded_at_commits": session.commits_now,
             "session": session.session_id}
    if session.task:
        stamp["task"] = session.task

    evidence = (Evidence(path=".", commit=commit or None,
                         ref=f"session:{session.session_id}"),)

    records.append(Node(
        layer=Layer.L7, kind="session", key=session.session_id,
        name=session.task[:80] or session.session_id,
        env=_env(Method.EXTRACTED, source, commit, evidence),
        attrs={**stamp, "files": list(session.files),
               "succeeded": session.succeeded},
    ))

    # What the session *did* is a fact about the session, whoever ran it.
    if session.files:
        records.append(Fact(
            layer=Layer.L7, subject=REPO, predicate=f"{OUTCOME}:{session.session_id}",
            value={"task": session.task, "files": sorted(session.files),
                   "succeeded": session.succeeded},
            env=_env(Method.EXTRACTED, source, commit, evidence),
            attrs=stamp,
        ))

    # What the session *concluded* is a reading, and is marked as one.
    for lesson in session.lessons:
        records.append(Fact(
            layer=Layer.L7, subject=REPO, predicate=f"{LESSON}:{slug(lesson)}",
            value=lesson,
            env=_env(Method.INFERRED, source, commit,
                     (Evidence(path=".", commit=commit or None,
                               ref=f"session:{session.session_id}"),),
                     note="recorded by an agent, not verified"),
            attrs=stamp,
        ))

    for question, answer in session.resolved:
        records.append(Fact(
            layer=Layer.L7, subject=REPO, predicate=f"{RESOLVED}:{slug(question)}",
            value={"question": question, "answer": answer},
            env=_env(Method.INFERRED, source, commit, evidence,
                     note="resolved during a session; not independently verified"),
            attrs=stamp,
        ))

    for question in session.questions:
        records.append(Fact(
            layer=Layer.L7, subject=REPO, predicate=f"{QUESTION}:{slug(question)}",
            value=question,
            env=_env(Method.INFERRED, source, commit, evidence,
                     confidence=0.5, note="raised during a session, unanswered"),
            attrs=stamp,
        ))

    return records


def remember(text: str, *, about: str = REPO, task: str = "", commit: str = "",
             commits_now: int = 0, human: bool = False,
             source: str = "agent") -> Fact:
    """A single lesson, from an agent or a person."""
    method = Method.ASSERTED if human else Method.INFERRED
    attrs: dict[str, Any] = {"recorded_at_commits": commits_now}
    if task:
        attrs["task"] = task
    return Fact(
        layer=Layer.L7, subject=about, predicate=f"{LESSON}:{slug(text)}", value=text,
        env=_env(method, "human" if human else source, commit,
                 (Evidence(path=about.split(":file:")[-1] if ":file:" in about else ".",
                           commit=commit or None),),
                 note="" if human else "recorded by an agent, not verified"),
        attrs=attrs,
    )


def dispute(brain: Brain, record_id: str, reason: str, *, commit: str = "",
            commits_now: int = 0, human: bool = False) -> tuple[Fact | None, str]:
    """Contest an existing claim.

    An agent files a dispute the packs will surface. A human overrules outright.
    The asymmetry is the point: a model that can demote an AST fact by asserting
    harder can poison every session downstream, and it will do so confidently.
    """
    target = brain.get(record_id)
    if target is None:
        return None, "missing"

    note = f"disputed: {reason}"
    record = Fact(
        layer=Layer.L7, subject=record_id, predicate=f"{DISPUTE}:{slug(reason)}",
        value={"reason": reason, "target": record_id,
               "target_method": str(target.env.method)},
        env=_env(Method.ASSERTED if human else Method.INFERRED,
                 "human" if human else "agent", commit,
                 (Evidence(path=".", commit=commit or None),), note=note),
        attrs={"recorded_at_commits": commits_now},
    )

    if human:
        target.env = target.env.demote(note)
        return record, "overruled"
    return record, "disputed"


def lessons_for(brain: Brain, paths: Iterable[str], commits_now: int,
                floor: float = MEMORY_FLOOR) -> list[tuple[Fact, float]]:
    """Memory relevant to a set of files, strongest first.

    Faded and refuted memories are dropped here rather than in the renderer, so
    every consumer gets the same answer.
    """
    wanted = {p for p in paths if p}
    out: list[tuple[Fact, float]] = []

    for fact in brain.facts.values():
        if fact.layer is not Layer.L7:
            continue
        if not fact.env.usable():
            continue
        weight = effective_memory_confidence(fact, commits_now)
        if weight < floor:
            continue

        subject_path = (fact.subject.split(":file:", 1)[-1]
                        if ":file:" in fact.subject else "")
        relevant = fact.subject == REPO or subject_path in wanted
        if not relevant and fact.predicate.startswith(OUTCOME):
            touched = set((fact.value or {}).get("files", ()))
            relevant = bool(touched & wanted)
        if relevant:
            out.append((fact, weight))

    out.sort(key=lambda pair: (-pair[1], pair[0].id))
    return out


def disputes_for(brain: Brain, record_ids: Iterable[str]) -> list[Fact]:
    wanted = set(record_ids)
    return sorted(
        (f for f in brain.facts.values()
         if f.layer is Layer.L7 and f.predicate.startswith(DISPUTE)
         and f.subject in wanted and f.env.usable()),
        key=lambda f: f.id,
    )


def stats(brain: Brain, commits_now: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    faded = 0
    for fact in brain.facts.values():
        if fact.layer is not Layer.L7:
            continue
        kind = fact.predicate.split(":", 1)[0]
        counts[kind] = counts.get(kind, 0) + 1
        if effective_memory_confidence(fact, commits_now) < MEMORY_FLOOR:
            faded += 1
    sessions = sum(1 for n in brain.nodes.values()
                   if n.layer is Layer.L7 and n.kind == "session")
    return {"sessions": sessions, "by_kind": counts, "faded": faded}
