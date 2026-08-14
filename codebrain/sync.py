"""Keeping a Brain true: sync, carry-forward, and drift.

Three related jobs.

**Carry-forward** is the load-bearing one. Extraction is deterministic, so a
rebuild reproduces every EXTRACTED and DERIVED claim exactly — but it also
overwrites the two kinds of claim that a rebuild cannot regenerate: what
execution proved (OBSERVED) and what a human stated (ASSERTED). Without
carry-forward, every `codebrain build` silently erases every verification, and
P3 would be pointless by the next commit.

**Sync** is build plus carry-forward, skipped entirely when nothing has changed.

**Drift** rebuilds into memory and compares against the committed Brain without
writing anything. A substantive difference means the Brain no longer describes
HEAD, which is the CI gate: a stale Brain misleads every agent downstream, and
that is worse than having no Brain at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .diff import BrainDiff
from .diff import diff as diff_brains
from .envelope import Method, Status
from .gitutil import git, git_stripped, is_repo
from .model import Brain, Record
from .store import BRAIN_DIR, read_touched

#: Claims a rebuild cannot reproduce, because they were established by running
#: something or by a person saying so.
UNREPRODUCIBLE = (Method.OBSERVED, Method.ASSERTED)


def _payload(record: Record) -> object:
    """The claim itself, ignoring trust metadata."""
    data = dict(record.to_json())
    data.pop("env", None)
    return data


def carry_forward(old: Brain, new: Brain) -> tuple[int, int]:
    """Move OBSERVED and ASSERTED envelopes from `old` onto matching records in
    `new`. Returns (carried, invalidated).

    A verification is only carried when the claim it verified is unchanged. If
    the test command was `make test` when it was proven and is now `pytest`, the
    old evidence is about a different claim and must not travel with the new
    one — that would be the Brain lying with a real receipt attached.
    """
    carried = invalidated = 0
    new_by_id = {r.id: r for r in new.records()}

    for record in old.records():
        # A refutation is as unreproducible as a promotion — it was produced by
        # running something. `demote` leaves the method alone, so refuted claims
        # must be matched on status or they are silently regenerated as
        # "never executed", and the next agent is sent to run a command
        # CodeBrain already proved broken.
        if (record.env.method not in UNREPRODUCIBLE
                and record.env.status is not Status.REFUTED):
            continue
        replacement = new_by_id.get(record.id)
        if replacement is None:
            continue
        if _payload(record) != _payload(replacement):
            invalidated += 1
            continue
        replacement.env = record.env
        carried += 1
    return carried, invalidated


def working_tree_paths(root: Path, exclude: str = BRAIN_DIR) -> set[str]:
    """Paths with uncommitted changes, excluding the Brain's own directory.

    Two traps here. First, parse raw output rather than stripped output: `git
    status --porcelain` encodes status in the first two columns, and an unstaged
    modification looks like " M path" — stripping the whole payload eats that
    leading space and the fixed-width slice then bites a character off the front
    of the filename.

    Second, `.brain/` is itself a working-tree change after every build. Left in,
    the Brain would permanently observe its own shadow and conclude the
    repository had changed, so sync could never report "up to date".
    """
    raw = git(root, "status", "--porcelain")
    if not raw:
        return set()

    paths: set[str] = set()
    for line in raw.split("\n"):
        if len(line) < 4:
            continue
        candidate = line[3:].strip().replace("\\", "/")
        if " -> " in candidate:  # rename
            candidate = candidate.split(" -> ", 1)[1]
        candidate = candidate.strip('"')
        if not candidate:
            continue
        if candidate == exclude or candidate.startswith(exclude.rstrip("/") + "/"):
            continue
        paths.add(candidate)
    return paths


def changed_files(root: Path, since: str) -> set[str] | None:
    """Paths that moved between `since` and HEAD, or None if it cannot be known."""
    if not since or not is_repo(root):
        return None
    out = git_stripped(root, "diff", "--name-only", f"{since}..HEAD")
    if out is None:
        return None
    paths = {line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()}
    return paths | working_tree_paths(root)


@dataclass(slots=True)
class SyncReport:
    rebuilt: bool = False
    reason: str = ""
    changed: set[str] = field(default_factory=set)
    carried: int = 0
    invalidated: int = 0
    delta: BrainDiff | None = None

    def render(self) -> str:
        if not self.rebuilt:
            return f"Up to date — {self.reason}"

        lines = [f"Rebuilt — {self.reason}"]
        if self.changed:
            shown = sorted(self.changed)[:8]
            lines.append(f"  {len(self.changed)} file(s) changed: "
                         + ", ".join(shown)
                         + (f", +{len(self.changed) - len(shown)} more"
                            if len(self.changed) > len(shown) else ""))
        if self.carried:
            lines.append(f"  {self.carried} verified/asserted claim(s) carried forward")
        if self.invalidated:
            lines.append(f"  {self.invalidated} verification(s) invalidated — the claim "
                         "they proved has changed; re-run `codebrain verify`")
        if self.delta is not None and not self.delta.empty:
            counts = self.delta.counts()
            lines.append(f"  +{counts['added']} -{counts['removed']} "
                         f"~{counts['changed']} records")
        return "\n".join(lines)


def needs_rebuild(brain: Brain, root: Path, brain_dir: Path) -> tuple[bool, str, set[str]]:
    head = git_stripped(root, "rev-parse", "HEAD") if is_repo(root) else ""
    touched = read_touched(brain_dir)

    if head and brain.manifest.as_of and head != brain.manifest.as_of:
        changed = changed_files(root, brain.manifest.as_of) or set()
        return True, f"HEAD moved {brain.manifest.as_of[:8]} → {head[:8]}", changed
    if touched:
        return True, f"{len(touched)} file(s) edited since the last build", set(touched)
    if not head:
        return True, "not a git repository — cannot tell what changed", set()

    dirty = working_tree_paths(root)
    if dirty:
        return True, f"{len(dirty)} uncommitted change(s)", dirty
    return False, f"Brain is at HEAD ({head[:8]}) and nothing is edited", set()


def drift(old: Brain, fresh: Brain) -> BrainDiff:
    """What the committed Brain gets wrong about the current code.

    Carry-forward runs first so that a verification the rebuild could not
    reproduce is not itself reported as drift — otherwise every verified Brain
    would fail its own gate.
    """
    carry_forward(old, fresh)
    return diff_brains(old, fresh)
