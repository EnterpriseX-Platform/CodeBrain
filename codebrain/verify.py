"""Verification by execution — the line between a Brain and documentation.

"The test command is `make test`" is a hypothesis. CodeBrain runs it. If it
succeeds the claim is promoted to OBSERVED with the exit code and duration as
evidence; if it fails the claim is REFUTED, kept, and annotated with why. That
loop is what stops a Brain drifting into confident fiction, and it is the one
thing a wiki cannot do.

Safety
------
This module runs commands that came out of a repository — a Makefile, a
package.json, a CI config. On an untrusted repository that is arbitrary code
execution, so:

* verification never runs from a hook, and never as part of `build`;
* `codebrain verify` is a dry run by default and prints exactly what it would
  execute — running requires an explicit `--yes`;
* only commands CodeBrain itself extracted are candidates, never free text;
* long-running intents are excluded by construction (see NON_TERMINATING) —
  `npm start` does not return, and a verifier that hangs is worse than one that
  never ran;
* every execution is bounded by a timeout.

The rule to hold on to: a human opts in once per run, with the command list in
front of them.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from .envelope import Evidence
from .model import REPO, Brain, Layer

#: Intents whose commands are expected to terminate. `run`/`start`/`serve`
#: deliberately never appear here: they are servers, and "it did not exit" is
#: not evidence of anything.
VERIFIABLE_INTENTS = ("test", "build", "lint")
NON_TERMINATING = ("run", "serve", "start", "dev", "watch")

DEFAULT_TIMEOUT = 600

#: How much command output to keep as evidence. Enough to see a failure, not so
#: much that a Brain becomes a log store.
OUTPUT_TAIL = 2000


@dataclass(slots=True)
class Candidate:
    intent: str
    fact_id: str
    command: str
    source: str = ""


@dataclass(slots=True)
class Outcome:
    candidate: Candidate
    ok: bool
    exit_code: int | None
    duration: float
    output: str = ""
    error: str = ""

    @property
    def summary(self) -> str:
        if self.ok:
            return f"passed in {self.duration:.1f}s"
        if self.error:
            return self.error
        return f"exit {self.exit_code} after {self.duration:.1f}s"


@dataclass(slots=True)
class VerifyReport:
    outcomes: list[Outcome] = field(default_factory=list)
    promoted: int = 0
    refuted: int = 0
    dry_run: bool = False

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if not o.ok)


def candidates(brain: Brain, intents: Iterable[str] = VERIFIABLE_INTENTS) -> list[Candidate]:
    """The claims in this Brain that can be settled by running something."""
    found: list[Candidate] = []
    for intent in intents:
        if intent in NON_TERMINATING:
            continue
        fact = brain.fact(REPO, f"{intent}_command", Layer.L5)
        if fact is None or not isinstance(fact.value, str) or not fact.value.strip():
            continue
        found.append(Candidate(intent=intent, fact_id=fact.id, command=fact.value.strip(),
                               source=str(fact.attrs.get("source", ""))))
    return found


def execute(command: str, root: Path, timeout: int = DEFAULT_TIMEOUT) -> Outcome:
    """Run one command. Never raises — a verifier that crashes verifies nothing."""
    stub = Candidate(intent="", fact_id="", command=command)
    env = dict(os.environ)
    # Keep the child from opening a pager or asking a question it will never
    # get an answer to: nothing is attached to its stdin.
    env.update({"CI": "1", "PYTHONUNBUFFERED": "1", "PAGER": "cat",
                "GIT_PAGER": "cat", "NO_COLOR": "1"})

    started = time.monotonic()
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(root), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL, env=env,
        )
    except subprocess.TimeoutExpired:
        return Outcome(stub, False, None, time.monotonic() - started,
                       error=f"timed out after {timeout}s")
    except (OSError, ValueError) as exc:
        return Outcome(stub, False, None, time.monotonic() - started,
                       error=f"{type(exc).__name__}: {exc}")

    duration = time.monotonic() - started
    combined = (proc.stdout or "") + (proc.stderr or "")
    return Outcome(stub, proc.returncode == 0, proc.returncode, duration,
                   output=combined[-OUTPUT_TAIL:])


def apply_outcome(brain: Brain, outcome: Outcome, commit: str) -> str:
    """Move the claim's envelope to match what actually happened."""
    fact = brain.facts.get(outcome.candidate.fact_id)
    if fact is None:
        return "missing"

    evidence = Evidence(path=".", commit=commit or None,
                        ref=f"executed: {outcome.candidate.command}")
    if outcome.ok:
        promoted = fact.env.promote(commit or fact.env.as_of)
        # Everything the verification learned goes in the envelope, never in
        # attrs. attrs are part of the claim's payload, and a payload that
        # changes when it is verified makes carry-forward decide the claim
        # itself moved — so the next sync would throw away the very evidence
        # this function just produced.
        fact.env = replace(
            promoted,
            evidence=promoted.evidence + (evidence,),
            note=f"verified: {outcome.summary} (exit {outcome.exit_code})",
        )
        return "promoted"

    # Same rule as promotion: nothing about the verification may touch attrs.
    # attrs are the claim's payload, and a payload that shifts when the claim is
    # tested makes carry-forward conclude the claim itself moved — which would
    # discard this refutation on the very next sync.
    tail = " / ".join(ln.strip() for ln in outcome.output.strip().splitlines()
                      if ln.strip())[-240:]
    demoted = fact.env.demote(f"execution failed — {outcome.summary}"
                              + (f" | {tail}" if tail else ""))
    fact.env = replace(demoted, evidence=demoted.evidence + (evidence,))
    return "refuted"


def verify(brain: Brain, root: Path, *, intents: Iterable[str] = VERIFIABLE_INTENTS,
           timeout: int = DEFAULT_TIMEOUT, execute_commands: bool = False,
           commit: str = "") -> VerifyReport:
    """Settle every executable claim. Dry by default: nothing runs unless asked."""
    report = VerifyReport(dry_run=not execute_commands)
    for candidate in candidates(brain, intents):
        if not execute_commands:
            report.outcomes.append(Outcome(candidate, True, None, 0.0,
                                           error="not executed (dry run)"))
            continue

        outcome = execute(candidate.command, root, timeout)
        outcome.candidate = candidate
        report.outcomes.append(outcome)

        result = apply_outcome(brain, outcome, commit or brain.manifest.as_of)
        if result == "promoted":
            report.promoted += 1
        elif result == "refuted":
            report.refuted += 1
    return report


def render(report: VerifyReport) -> str:
    if not report.outcomes:
        return ("Nothing to verify — this Brain has no executable claims. "
                "L5 found no build, test or lint command.")

    lines: list[str] = []
    if report.dry_run:
        lines.append("Dry run — nothing was executed. These commands came from this "
                     "repository's own manifests and would run as-is:")
        lines.append("")
        for outcome in report.outcomes:
            c = outcome.candidate
            lines.append(f"  {c.intent:<6} {c.command}"
                         + (f"   (from {c.source})" if c.source else ""))
        lines.append("")
        lines.append("Read them, then re-run with --yes to execute and settle the "
                     "claims. CodeBrain will not run repository commands without "
                     "an explicit opt-in.")
        return "\n".join(lines)

    for outcome in report.outcomes:
        mark = "ok  " if outcome.ok else "FAIL"
        lines.append(f"  {mark} {outcome.candidate.intent:<6} "
                     f"{outcome.candidate.command}")
        lines.append(f"       {outcome.summary}")
        if not outcome.ok and outcome.output:
            tail = [ln for ln in outcome.output.strip().splitlines() if ln.strip()][-4:]
            for line in tail:
                lines.append(f"       | {line[:110]}")

    lines.append("")
    lines.append(f"  {report.promoted} claim(s) promoted to OBSERVED · "
                 f"{report.refuted} refuted")
    if report.refuted:
        lines.append("  Refuted claims are kept, not deleted — a refutation is "
                     "information, and packs will stop offering them.")
    return "\n".join(lines)
