"""The provenance envelope.

Every record in a Brain carries one of these. It is the mechanism behind
principle ii — *provenance or it didn't happen* — and behind the trust tiering
that lets an agent say "extracted facts only" before a risky refactor.

The envelope is also where the verification loop (P3) lands: `promote` and
`demote` are how an executed claim changes its own standing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


SCHEMA_VERSION = "0.1"


def utc_now() -> str:
    """ISO-8601 UTC, second precision. Stable enough to sort on, short enough to read."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Method(StrEnum):
    """How a claim came to be believed."""

    EXTRACTED = "EXTRACTED"  # deterministic, from a source of truth (AST, manifest, CI config)
    DERIVED = "DERIVED"      # computed from other facts (churn, coupling, blast radius)
    INFERRED = "INFERRED"    # language model or heuristic — a reading, not a reading-off
    OBSERVED = "OBSERVED"    # proven by execution (the test command actually ran)
    ASSERTED = "ASSERTED"    # a human stated it


#: Tie-break ranking when two records collide. Execution beats reading the
#: source; a human beats a computation; a guess loses to everything.
TRUST: dict[Method, int] = {
    Method.OBSERVED: 5,
    Method.EXTRACTED: 4,
    Method.ASSERTED: 3,
    Method.DERIVED: 2,
    Method.INFERRED: 1,
}

#: Starting confidence for each method, before verification moves it.
DEFAULT_CONFIDENCE: dict[Method, float] = {
    Method.OBSERVED: 1.00,
    Method.EXTRACTED: 0.98,
    Method.ASSERTED: 0.95,
    Method.DERIVED: 0.80,
    Method.INFERRED: 0.60,
}


class Status(StrEnum):
    FRESH = "fresh"            # re-established at the current commit
    UNVERIFIED = "unverified"  # believed, never tested
    STALE = "stale"            # its evidence moved or the neighbourhood changed
    REFUTED = "refuted"        # verification proved it wrong


#: Multiplier on raw confidence. A refuted claim is worth nothing and must never
#: reach a context pack; a stale one is worth keeping but not acting on.
DECAY: dict[Status, float] = {
    Status.FRESH: 1.0,
    Status.UNVERIFIED: 0.9,
    Status.STALE: 0.5,
    Status.REFUTED: 0.0,
}


@dataclass(frozen=True, slots=True)
class Evidence:
    """A pointer an agent (or a reviewer) can follow back to the source."""

    path: str = ""
    start_line: int | None = None
    end_line: int | None = None
    commit: str | None = None
    ref: str | None = None  # PR url, CI log, ADR id, incident id

    def __str__(self) -> str:
        s = self.path or (self.ref or "?")
        if self.start_line is not None:
            s += f":{self.start_line}"
            if self.end_line is not None and self.end_line != self.start_line:
                s += f"-{self.end_line}"
        if self.commit:
            s += f"@{self.commit[:7]}"
        return s

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"path": self.path}
        for k in ("start_line", "end_line", "commit", "ref"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Evidence":
        return cls(
            path=d.get("path", ""),
            start_line=d.get("start_line"),
            end_line=d.get("end_line"),
            commit=d.get("commit"),
            ref=d.get("ref"),
        )


@dataclass(frozen=True, slots=True)
class Envelope:
    """Trust metadata attached to every node, edge and fact."""

    method: Method
    confidence: float = 1.0
    as_of: str = ""                     # the commit the claim was true at
    ts: str = ""                        # when the claim was recorded (orders claims)
    status: Status = Status.UNVERIFIED
    evidence: tuple[Evidence, ...] = ()
    source: str = ""                    # provider id that produced it
    verified_at: str | None = None      # commit at which execution confirmed it
    note: str = ""

    def __post_init__(self) -> None:
        # frozen dataclass: clamp through the back door rather than trusting callers
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        if isinstance(self.evidence, list):
            object.__setattr__(self, "evidence", tuple(self.evidence))

    # -- construction ------------------------------------------------------

    @classmethod
    def make(
        cls,
        method: Method,
        *,
        source: str = "",
        as_of: str = "",
        ts: str | None = None,
        evidence: tuple[Evidence, ...] | list[Evidence] = (),
        confidence: float | None = None,
        status: Status = Status.UNVERIFIED,
        note: str = "",
    ) -> "Envelope":
        return cls(
            method=method,
            confidence=DEFAULT_CONFIDENCE[method] if confidence is None else confidence,
            as_of=as_of,
            ts=ts or utc_now(),
            status=status,
            evidence=tuple(evidence),
            source=source,
            note=note,
        )

    # -- trust -------------------------------------------------------------

    @property
    def trust(self) -> int:
        return TRUST[self.method]

    def effective_confidence(self) -> float:
        """Confidence after status decay. This is what ranking uses, never the raw value."""
        return self.confidence * DECAY[self.status]

    def usable(self, floor: float = 0.0) -> bool:
        """Whether this claim may enter a context pack at all."""
        return self.status is not Status.REFUTED and self.effective_confidence() > floor

    def outranks(self, other: "Envelope") -> bool:
        """Whether this record should displace `other` on merge.

        Deliberately total and deterministic: two builds that see the same
        inputs must produce byte-identical Brains, so every tier ends in a
        tie-break that cannot come out both ways.
        """
        # A human overrules machinery, provided they are not speaking about an
        # older state of the world than the machine is.
        self_human = self.method is Method.ASSERTED
        other_human = other.method is Method.ASSERTED
        if self_human != other_human:
            if self_human and self.ts >= other.ts:
                return True
            if other_human and other.ts >= self.ts:
                return False

        # A fresh refutation beats a stale belief.
        if (self.status is Status.REFUTED) != (other.status is Status.REFUTED):
            if self.status is Status.REFUTED:
                return self.ts >= other.ts
            return not (other.ts >= self.ts)

        a, b = self.effective_confidence(), other.effective_confidence()
        if a != b:
            return a > b
        if self.trust != other.trust:
            return self.trust > other.trust
        if self.ts != other.ts:
            return self.ts > other.ts
        return False  # true tie: the incumbent stays, so merge order cannot matter

    # -- verification transitions (used from P3 onward) --------------------

    def promote(self, commit: str, confidence: float = 1.0) -> "Envelope":
        """Execution confirmed this claim."""
        return replace(
            self,
            method=Method.OBSERVED,
            status=Status.FRESH,
            confidence=confidence,
            verified_at=commit,
            ts=utc_now(),
        )

    def demote(self, reason: str) -> "Envelope":
        """Execution contradicted this claim. Kept, not deleted — a refutation is information."""
        return replace(self, status=Status.REFUTED, note=reason, ts=utc_now())

    def mark_stale(self, reason: str = "") -> "Envelope":
        if self.status is Status.REFUTED:
            return self
        return replace(self, status=Status.STALE, note=reason or self.note, ts=utc_now())

    def refresh(self, commit: str) -> "Envelope":
        return replace(self, status=Status.FRESH, as_of=commit, ts=utc_now())

    # -- serialisation -----------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "method": str(self.method),
            "confidence": round(self.confidence, 4),
            "status": str(self.status),
        }
        if self.as_of:
            d["as_of"] = self.as_of
        if self.ts:
            d["ts"] = self.ts
        if self.evidence:
            d["evidence"] = [e.to_json() for e in self.evidence]
        if self.source:
            d["source"] = self.source
        if self.verified_at:
            d["verified_at"] = self.verified_at
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Envelope":
        return cls(
            method=Method(d["method"]),
            confidence=d.get("confidence", 1.0),
            as_of=d.get("as_of", ""),
            ts=d.get("ts", ""),
            status=Status(d.get("status", "unverified")),
            evidence=tuple(Evidence.from_json(e) for e in d.get("evidence", ())),
            source=d.get("source", ""),
            verified_at=d.get("verified_at"),
            note=d.get("note", ""),
        )
