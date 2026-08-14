"""Diffing two Brains.

Used three ways: to review what a build changed, to show a reviewer what a pull
request did to the repository's understanding, and — from P3 — as the drift
gate, where a non-empty diff between the committed Brain and a fresh extraction
at HEAD fails CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import Brain, Record


def _payload(rec: Record) -> dict[str, Any]:
    """The claim itself, without its trust metadata."""
    d = dict(rec.to_json())
    d.pop("env", None)
    return d


@dataclass(slots=True)
class Change:
    id: str
    before: Record
    after: Record
    payload_changed: bool
    envelope_changed: bool

    @property
    def summary(self) -> str:
        bits = []
        if self.payload_changed:
            bits.append("value")
        if self.envelope_changed:
            b, a = self.before.env, self.after.env
            if b.method is not a.method:
                bits.append(f"{b.method}→{a.method}")
            if b.status is not a.status:
                bits.append(f"{b.status}→{a.status}")
            if round(b.confidence, 3) != round(a.confidence, 3):
                bits.append(f"conf {b.confidence:.2f}→{a.confidence:.2f}")
        return ", ".join(bits) or "no-op"


@dataclass(slots=True)
class BrainDiff:
    added: list[Record] = field(default_factory=list)
    removed: list[Record] = field(default_factory=list)
    changed: list[Change] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    @property
    def substantive(self) -> bool:
        """Whether anything a consumer would act on moved.

        A rebuild always shifts timestamps and re-derives confidences. The drift
        gate must fire on *claims* changing, not on the Brain being rebuilt.
        """
        return bool(self.added or self.removed or any(c.payload_changed for c in self.changed))

    def counts(self) -> dict[str, int]:
        return {"added": len(self.added), "removed": len(self.removed),
                "changed": len(self.changed)}


def diff(old: Brain, new: Brain) -> BrainDiff:
    before = {r.id: r for r in old.records()}
    after = {r.id: r for r in new.records()}

    result = BrainDiff()
    for rid in sorted(after.keys() - before.keys()):
        result.added.append(after[rid])
    for rid in sorted(before.keys() - after.keys()):
        result.removed.append(before[rid])
    for rid in sorted(before.keys() & after.keys()):
        b, a = before[rid], after[rid]
        payload_changed = _payload(b) != _payload(a)
        envelope_changed = (
            b.env.method is not a.env.method
            or b.env.status is not a.env.status
            or round(b.env.confidence, 3) != round(a.env.confidence, 3)
        )
        if payload_changed or envelope_changed:
            result.changed.append(Change(rid, b, a, payload_changed, envelope_changed))
    return result


def render(d: BrainDiff, limit: int = 40) -> str:
    if d.empty:
        return "No change."
    lines: list[str] = []
    for rec in d.added[:limit]:
        lines.append(f"  + {rec.id}")
    if len(d.added) > limit:
        lines.append(f"    … {len(d.added) - limit} more added")
    for rec in d.removed[:limit]:
        lines.append(f"  - {rec.id}")
    if len(d.removed) > limit:
        lines.append(f"    … {len(d.removed) - limit} more removed")
    for ch in d.changed[:limit]:
        lines.append(f"  ~ {ch.id}  ({ch.summary})")
    if len(d.changed) > limit:
        lines.append(f"    … {len(d.changed) - limit} more changed")
    c = d.counts()
    lines.append("")
    lines.append(f"  {c['added']} added · {c['removed']} removed · {c['changed']} changed"
                 + ("" if d.substantive else "  (metadata only)"))
    return "\n".join(lines)
