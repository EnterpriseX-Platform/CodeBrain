"""Session instrumentation — measuring what the git benchmark cannot see.

`codebrain eval` asks one question: did the pack name the right files? That is
the single facet keyword search also covers, and it saturates on small
repositories. It cannot see whether the constraints stopped a bad edit, whether
the verified runbook saved a guess, or whether the blast radius prevented a
regression — five of the six facets, and the reason the pack exists.

Those only show up when an agent actually does the work. So this records real
sessions and scores them.

Three rules the design turns on.

**CodeBrain does not judge its own trial.** Success is the repository's own test
command, executed before and after the session. A claim by the Brain about the
Brain is worth nothing here, and a harness that scores itself will always find
that it is doing well.

**Arms are assigned before the work, deterministically.** A session's arm is a
hash of the trial name and session id, fixed at `start`. Deciding afterwards
which sessions "count" is how honest people produce dishonest numbers.

**A trace is evidence, not a verdict.** Recording is cheap and always on;
interpretation happens at report time, where the sample size and the arm balance
are printed next to the effect. A harness that reports a delta without an n is
not reporting a result.

This module collects. It produces no numbers of its own until real sessions run
through it — which is the honest cost of measuring the thing that actually
matters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .envelope import utc_now

TRIALS_DIR = "trials"

#: The two arms. `treatment` receives a context pack; `control` works the way an
#: agent does today. Named rather than "on"/"off" so the report reads as an
#: experiment and not as a demo.
TREATMENT = "treatment"
CONTROL = "control"
ARMS = (TREATMENT, CONTROL)

#: Session outcomes. `unknown` is a first-class value: a session whose
#: verification never ran must not be silently counted as either.
SUCCESS, FAILURE, UNKNOWN = "success", "failure", "unknown"


def assign_arm(trial: str, session_id: str, split: float = 0.5) -> str:
    """Deterministic, unbiased, and fixed before any work happens.

    Deterministic so a trace can be re-derived and audited; before the work so
    nobody can look at how a session went and then decide which arm it was in.
    """
    digest = hashlib.sha256(f"{trial}:{session_id}".encode("utf-8")).digest()
    # First 8 bytes as a fraction of the range: uniform, stable across platforms.
    fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return TREATMENT if fraction < split else CONTROL


@dataclass(slots=True)
class Event:
    kind: str                      # edit | command | context | note
    at: str = field(default_factory=utc_now)
    path: str = ""
    command: str = ""
    exit_code: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "at": self.at}
        if self.path:
            out["path"] = self.path
        if self.command:
            out["command"] = self.command
        if self.exit_code is not None:
            out["exit_code"] = self.exit_code
        if self.detail:
            out["detail"] = self.detail
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Event":
        return cls(kind=data.get("kind", "note"), at=data.get("at", ""),
                   path=data.get("path", ""), command=data.get("command", ""),
                   exit_code=data.get("exit_code"), detail=data.get("detail", {}))


@dataclass(slots=True)
class Trace:
    """One recorded session."""

    trial: str
    session: str
    arm: str
    task: str = ""
    started_at: str = field(default_factory=utc_now)
    ended_at: str = ""
    brain_commit: str = ""
    context_tokens: int = 0
    context_facets: list[str] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    baseline: dict[str, Any] = field(default_factory=dict)   # verification before
    verdict: dict[str, Any] = field(default_factory=dict)    # verification after
    outcome: str = UNKNOWN
    notes: list[str] = field(default_factory=list)

    # -- derived ----------------------------------------------------------

    @property
    def edits(self) -> list[str]:
        seen: list[str] = []
        for event in self.events:
            if event.kind == "edit" and event.path and event.path not in seen:
                seen.append(event.path)
        return seen

    @property
    def commands(self) -> list[Event]:
        return [e for e in self.events if e.kind == "command"]

    @property
    def failed_commands(self) -> int:
        return sum(1 for e in self.commands if e.exit_code not in (None, 0))

    @property
    def broke_a_passing_build(self) -> bool:
        """The regression case: it worked before this session and does not now."""
        return (self.baseline.get("exit_code") == 0
                and self.verdict.get("exit_code") not in (None, 0))

    def to_json(self) -> dict[str, Any]:
        return {
            "trial": self.trial, "session": self.session, "arm": self.arm,
            "task": self.task, "started_at": self.started_at,
            "ended_at": self.ended_at, "brain_commit": self.brain_commit,
            "context_tokens": self.context_tokens,
            "context_facets": self.context_facets,
            "events": [e.to_json() for e in self.events],
            "baseline": self.baseline, "verdict": self.verdict,
            "outcome": self.outcome, "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Trace":
        return cls(
            trial=data.get("trial", ""), session=data.get("session", ""),
            arm=data.get("arm", CONTROL), task=data.get("task", ""),
            started_at=data.get("started_at", ""), ended_at=data.get("ended_at", ""),
            brain_commit=data.get("brain_commit", ""),
            context_tokens=int(data.get("context_tokens", 0) or 0),
            context_facets=list(data.get("context_facets", ())),
            events=[Event.from_json(e) for e in data.get("events", ())],
            baseline=data.get("baseline", {}), verdict=data.get("verdict", {}),
            outcome=data.get("outcome", UNKNOWN), notes=list(data.get("notes", ())),
        )


# -- storage ---------------------------------------------------------------


def trial_dir(brain_dir: Path | str, trial: str) -> Path:
    return Path(brain_dir) / TRIALS_DIR / trial


def trace_path(brain_dir: Path | str, trial: str, session: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in session)[:80]
    return trial_dir(brain_dir, trial) / f"{safe or 'session'}.json"


def save_trace(brain_dir: Path | str, trace: Trace) -> Path:
    path = trace_path(brain_dir, trace.trial, trace.session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace.to_json(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def load_trace(path: Path) -> Trace | None:
    try:
        return Trace.from_json(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def load_traces(brain_dir: Path | str, trial: str) -> list[Trace]:
    directory = trial_dir(brain_dir, trial)
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        trace = load_trace(path)
        if trace is not None:
            out.append(trace)
    return out


ACTIVE = "ACTIVE.json"


def set_active(brain_dir: Path | str, trial: str, session: str) -> None:
    """Point at the session currently being recorded.

    The PostToolUse hook fires with no idea which trial is running, so the
    pointer is what lets a plain `codebrain touch` land in the right trace
    without every hook having to be re-wired per experiment.
    """
    root = Path(brain_dir) / TRIALS_DIR
    root.mkdir(parents=True, exist_ok=True)
    (root / ACTIVE).write_text(
        json.dumps({"trial": trial, "session": session}) + "\n", encoding="utf-8")


def get_active(brain_dir: Path | str) -> tuple[str, str] | None:
    path = Path(brain_dir) / TRIALS_DIR / ACTIVE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data["trial"]), str(data["session"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None


def clear_active(brain_dir: Path | str) -> None:
    (Path(brain_dir) / TRIALS_DIR / ACTIVE).unlink(missing_ok=True)


def list_trials(brain_dir: Path | str) -> list[str]:
    root = Path(brain_dir) / TRIALS_DIR
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def verification_command(brain) -> tuple[str, str] | None:
    """The repository's own test command, and how well established it is.

    Deliberately the repo's command rather than anything CodeBrain asserts about
    itself: a harness scored by the system under test will always find that the
    system under test is doing well.
    """
    from .model import REPO, Layer

    found = brain.fact(REPO, "test_command", Layer.L5) if brain else None
    if found is None or not isinstance(found.value, str) or not found.value.strip():
        return None
    return found.value.strip(), str(found.env.method)


# -- scoring ---------------------------------------------------------------


@dataclass(slots=True)
class ArmSummary:
    arm: str
    n: int = 0
    successes: int = 0
    failures: int = 0
    unknown: int = 0
    regressions: int = 0
    edits: list[int] = field(default_factory=list)
    failed_commands: list[int] = field(default_factory=list)
    context_tokens: list[int] = field(default_factory=list)

    @property
    def decided(self) -> int:
        """Sessions with a real verdict. Unknowns are excluded from rates rather
        than being quietly counted as failures."""
        return self.successes + self.failures

    @property
    def success_rate(self) -> float | None:
        return self.successes / self.decided if self.decided else None

    @property
    def regression_rate(self) -> float | None:
        return self.regressions / self.decided if self.decided else None

    def mean(self, values: list[int]) -> float | None:
        return sum(values) / len(values) if values else None


def summarise(traces: Iterable[Trace]) -> dict[str, ArmSummary]:
    summaries = {arm: ArmSummary(arm=arm) for arm in ARMS}
    for trace in traces:
        summary = summaries.setdefault(trace.arm, ArmSummary(arm=trace.arm))
        summary.n += 1
        if trace.outcome == SUCCESS:
            summary.successes += 1
        elif trace.outcome == FAILURE:
            summary.failures += 1
        else:
            summary.unknown += 1
        if trace.broke_a_passing_build:
            summary.regressions += 1
        summary.edits.append(len(trace.edits))
        summary.failed_commands.append(trace.failed_commands)
        if trace.arm == TREATMENT:
            summary.context_tokens.append(trace.context_tokens)
    return summaries


#: Below this many decided sessions per arm, a delta is noise and the report
#: says so instead of printing a percentage that invites belief.
MIN_PER_ARM = 10


def render(trial: str, traces: list[Trace]) -> str:
    if not traces:
        return (f"No sessions recorded for trial {trial!r}.\n"
                "Run `codebrain trial start` from a session, or wire the hooks "
                "with `codebrain init`.")

    summaries = summarise(traces)
    treatment, control = summaries[TREATMENT], summaries[CONTROL]

    lines = [f"Trial {trial} — {len(traces)} recorded session(s)", ""]
    lines.append(f"  {'':<22}{'with pack':>12}{'control':>12}")
    lines.append(f"  {'sessions':<22}{treatment.n:>12}{control.n:>12}")
    lines.append(f"  {'decided':<22}{treatment.decided:>12}{control.decided:>12}")

    def row(label: str, a: float | None, b: float | None, pct: bool = True) -> None:
        fmt = (lambda v: "—" if v is None else (f"{v:.0%}" if pct else f"{v:.1f}"))
        lines.append(f"  {label:<22}{fmt(a):>12}{fmt(b):>12}")

    row("task success", treatment.success_rate, control.success_rate)
    row("broke the build", treatment.regression_rate, control.regression_rate)
    row("mean files edited", treatment.mean(treatment.edits),
        control.mean(control.edits), pct=False)
    row("mean failed commands", treatment.mean(treatment.failed_commands),
        control.mean(control.failed_commands), pct=False)

    if treatment.context_tokens:
        mean_tokens = treatment.mean(treatment.context_tokens) or 0
        lines.append(f"  {'mean pack tokens':<22}{mean_tokens:>12.0f}{'—':>12}")

    lines.append("")
    thin = [s for s in (treatment, control) if s.decided < MIN_PER_ARM]
    if thin:
        need = ", ".join(f"{s.arm} has {s.decided}" for s in thin)
        lines.append(f"  NOT YET A RESULT — fewer than {MIN_PER_ARM} decided "
                     f"sessions per arm ({need}).")
        lines.append("  The rates above are printed for inspection, not for "
                     "quoting.")
    else:
        a, b = treatment.success_rate, control.success_rate
        if a is not None and b is not None:
            lines.append(f"  success delta: {a - b:+.0%} "
                         f"({treatment.decided} vs {control.decided} decided)")

    unknown = treatment.unknown + control.unknown
    if unknown:
        lines.append(f"  {unknown} session(s) had no verdict and are excluded "
                     "from every rate — a session whose verification never ran "
                     "is not a failure.")
    return "\n".join(lines)
