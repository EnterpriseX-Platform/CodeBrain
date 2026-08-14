"""Does the pack actually help?

P2's gate is a measurement, not an assertion, so the benchmark has to be real
and it has to be cheap enough to run on every change. This one generates itself
from the repository's own git history: each past commit is a task whose correct
answer is already known — the files that commit changed.

    task          the commit subject, as the developer wrote it
    ground truth  the files that commit touched
    prediction    the files a context pack surfaces for that task
    baseline      the files a keyword search surfaces for the same task

Two modes. The fast one builds a single Brain from HEAD, which has therefore
seen the finished state of the code these commits produced. That inflates both
arms — the pack *and* the baseline read the same post-change repository — so the
comparison stays fair while the absolute numbers read high.

`--rigorous` removes the leakage: each case gets its own Brain, built from that
commit's parent in a detached worktree, so nothing in it can know the answer. It
costs a full extraction per case, which is why it is opt-in rather than the
default — but it is the number to quote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .gitutil import (
    REC,
    REC_FMT,
    SEP,
    SEP_FMT,
    git,
    git_stripped,
    is_repo,
    normalise_rename,
)
from .model import Brain
from .pack import compile_pack, is_test_path, tokenize

#: Commits touching more than this are bulk moves, reformats or merges. They
#: have no meaningful "correct answer" and would flatter any retrieval method.
MAX_FILES_PER_CASE = 8

MIN_FILES_PER_CASE = 1

DEFAULT_CASES = 40


#: Dependency-bump and release-automation commits. Their subjects carry no
#: engineering intent to retrieve — "Bump the actions group with 2 updates"
#: describes no behaviour and names no code — so they measure nothing about
#: either arm. Excluded as a stated rule, not tuned away case by case.
AUTOMATED = re.compile(
    r"^(bump\b|chore\(deps|build\(deps|update dependenc|merge (pull request|branch)\b"
    r"|\[?(pre-commit\.ci|dependabot|renovate)\b)|\[bot\]|\bdependabot\b",
    re.IGNORECASE,
)


def is_automated(subject: str) -> bool:
    return bool(AUTOMATED.search(subject))


@dataclass(slots=True)
class Case:
    sha: str
    subject: str
    changed: list[str]


@dataclass(slots=True)
class CaseResult:
    case: Case
    pack_files: list[str]
    grep_files: list[str]
    pack_recall: float
    grep_recall: float
    tokens: int
    k: int


@dataclass(slots=True)
class Report:
    results: list[CaseResult] = field(default_factory=list)
    skipped: int = 0

    @property
    def n(self) -> int:
        return len(self.results)

    def mean(self, attr: str) -> float:
        if not self.results:
            return 0.0
        return sum(getattr(r, attr) for r in self.results) / len(self.results)

    @property
    def wins(self) -> int:
        return sum(1 for r in self.results if r.pack_recall > r.grep_recall)

    @property
    def losses(self) -> int:
        return sum(1 for r in self.results if r.pack_recall < r.grep_recall)

    @property
    def ties(self) -> int:
        return self.n - self.wins - self.losses


def collect_cases(root: Path, limit: int = DEFAULT_CASES, skip: int = 0) -> list[Case]:
    raw = git(root, "log", "--no-merges", "--name-only",
              f"--max-count={limit + skip + 40}",
              f"--format={REC_FMT}%H{SEP_FMT}%s")
    if not raw:
        return []

    cases: list[Case] = []
    for chunk in raw.split(REC):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        lines = chunk.split("\n")
        header = lines[0].split(SEP)
        if len(header) < 2:
            continue
        sha, subject = header[0], header[1]
        changed = [normalise_rename(ln.strip()) for ln in lines[1:] if ln.strip()]
        changed = [c for c in changed if c]
        if not (MIN_FILES_PER_CASE <= len(changed) <= MAX_FILES_PER_CASE):
            continue
        if len(tokenize(subject)) < 2:
            continue  # "wip", "fix" — no retrievable signal, unfair to both arms
        if is_automated(subject):
            continue
        cases.append(Case(sha=sha, subject=subject, changed=changed))

    return cases[skip: skip + limit]


def record_path(brain: Brain, record_id: str) -> str | None:
    """The repo-relative file a record belongs to."""
    if not record_id:
        return None
    record = brain.get(record_id)
    if record is None:
        if ":file:" in record_id:
            # A fact id carries its predicate after a pipe: strip it, or the
            # "path" gains a `|churn` tail and matches nothing.
            return record_id.split(":file:", 1)[-1].split("|", 1)[0]
        return None
    subject = getattr(record, "subject", None)
    if subject and ":file:" in subject:
        return subject.split(":file:", 1)[-1]
    attrs = getattr(record, "attrs", {}) or {}
    candidate = attrs.get("module") or attrs.get("path")
    if candidate:
        return str(candidate).split("#", 1)[0]
    key = getattr(record, "key", None)
    if key:
        return str(key).split("#", 1)[0]
    src = getattr(record, "src", "")
    if ":file:" in src:
        return src.split(":file:", 1)[-1]
    return None


def pack_prediction(brain: Brain, task: str, budget: int,
                    root: Path | None = None) -> tuple[list[str], int]:
    """Files the pack points at, in the order it presents them.

    Resolved through record ids rather than by scraping the rendered text. The
    text carries things that merely look like paths — a confidence value such
    as `0.80` inside a `[DERIVED 0.80]` tag parses as a filename — and each
    phantom burns a slot that a real candidate needed.
    """
    pack = compile_pack(brain, task, budget=budget, root=root)
    ordered: list[str] = []
    seen: set[str] = set()

    for item in pack.items:
        path = record_path(brain, item.record_id)
        if path is None:
            # Items without a record (summaries) may still name real files.
            for match in re.findall(r"[\w][\w./-]*\.[A-Za-z][A-Za-z0-9]{0,5}\b",
                                    item.text):
                if "/" in match and match not in seen:
                    seen.add(match)
                    ordered.append(match)
            continue
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered, pack.tokens


def grep_prediction(root: Path, task: str, files: Iterable[Path],
                    limit: int) -> list[str]:
    """The honest baseline: keyword search, ranked by match count.

    This is what an agent does today without a Brain, and it is a real
    baseline — not a straw man. Identifiers in the task are strong signal, and
    grep finds them.
    """
    terms = tokenize(task)
    if not terms:
        return []

    scored: list[tuple[float, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except (OSError, ValueError):
            continue
        score = 0.0
        for term in terms:
            hits = text.count(term)
            if hits:
                score += 1.0 + min(hits, 20) * 0.1
        rel = path.as_posix()
        if score:
            if is_test_path(rel):
                score *= 0.45  # same handicap the pack applies, so it is like-for-like
            scored.append((score, rel))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [rel for _, rel in scored[:limit]]


def recall(predicted: Iterable[str], truth: Iterable[str], k: int) -> float:
    """Share of the truly-changed files that appear in the first k predictions."""
    truth_set = {t for t in truth}
    if not truth_set:
        return 0.0
    head = list(predicted)[:k]
    found = sum(1 for t in truth_set if any(p.endswith(t) or t.endswith(p)
                                            for p in head))
    return found / len(truth_set)


def run(brain: Brain, root: Path, cases: list[Case], budget: int = 6000) -> Report:
    from .providers import BuildContext

    ctx = BuildContext(root=root)
    corpus = [p for p in ctx.iter_files() if ctx.readable(p)]
    relative = []
    for path in corpus:
        try:
            relative.append(path.relative_to(root))
        except ValueError:
            continue

    report = Report()
    for case in cases:
        predicted, tokens = pack_prediction(brain, case.subject, budget, root=root)
        if not predicted:
            report.skipped += 1
            continue
        # Judge both arms at identical k, and let k be what the pack actually
        # offered. Any k larger than the pack's own list hands the baseline
        # extra guesses the pack cannot use — which is a property of the
        # scoring, not of the retrieval, and would make the comparison
        # meaningless in exactly the flattering direction for grep.
        k = min(len(predicted), 25)
        baseline = grep_prediction(root, case.subject,
                                   (root / r for r in relative), k)
        report.results.append(CaseResult(
            case=case,
            pack_files=predicted[:k],
            grep_files=baseline,
            pack_recall=recall(predicted, case.changed, k),
            grep_recall=recall(baseline, case.changed, k),
            tokens=tokens, k=k,
        ))
    return report


def render(report: Report, verbose: bool = False, rigorous: bool = False,
           memory: bool = False) -> str:
    if not report.n:
        return ("No usable evaluation cases. This needs a repository with git "
                "history whose commit subjects carry at least two meaningful words.")

    left, right = ("with memory", "without memory") if memory else ("pack", "search")
    if memory:
        heading = (f"Write-back effect — {report.n} case(s) where a later commit "
                   "touches files a previous one did")
    else:
        heading = (f"Context pack vs keyword search — {report.n} case(s) "
                   "from git history")
    lines = [
        heading,
        "",
        f"  {left:<17} {report.mean('pack_recall'):.1%}",
        f"  {right:<17} {report.mean('grep_recall'):.1%}",
        f"  delta             {report.mean('pack_recall') - report.mean('grep_recall'):+.1%}",
        "",
        f"  better {report.wins} · same {report.ties} · worse {report.losses}",
        f"  mean pack size    {report.mean('tokens'):.0f} tokens",
    ]
    if report.skipped:
        lines.append(f"  {report.skipped} case(s) produced no pack at all "
                     "(no anchors matched)")

    if verbose:
        lines += ["", "  case                                      pack   search"]
        for result in report.results:
            subject = result.case.subject[:38].ljust(38)
            lines.append(f"  {subject}  {result.pack_recall:>5.0%}   "
                         f"{result.grep_recall:>5.0%}")

    if memory:
        lines += [
            "",
            "Both arms use the same Brain, built from the later commit's parent.",
            "The only difference is whether the earlier session was written back,",
            "so this delta is the value of memory and nothing else.",
        ]
    elif rigorous:
        lines += [
            "",
            "Each case used a Brain built from that commit's parent, in a detached",
            "worktree. Nothing in it had seen the change being asked about, so",
            "these numbers carry no leakage — and read lower than the fast mode's",
            "for exactly that reason.",
        ]
    else:
        lines += [
            "",
            "Caveat: the Brain is built from HEAD, so it has seen the finished state",
            "of the code these commits produced. Both arms read the same post-change",
            "repository, so the comparison is fair, but absolute recall reads high.",
            "Run with --rigorous for a Brain per commit that cannot know the answer.",
        ]
    return "\n".join(lines)


def run_rigorous(root: Path, cases: list[Case], budget: int = 6000,
                 log=None) -> Report:
    """The honest version: a Brain per commit, built from that commit's parent.

    The fast benchmark builds one Brain from HEAD, so it has already seen the
    finished state of the code these commits produced. Here each case gets a
    Brain that has not: a detached worktree is checked out at `sha^`, extracted,
    and asked the task. Nothing in it can know the answer.

    Slow — a full extraction per case — so it is opt-in. Worktrees are always
    removed, including when a case fails, and none of this touches the caller's
    working tree.
    """
    import shutil
    import tempfile

    from .providers import BuildContext
    from .providers import build as run_build

    # Extractors register themselves on import. Without this the registry is
    # empty for any caller that has not imported them, every Brain comes out
    # with zero records, and the run reports "skipped" for every case instead
    # of failing — a silent nothing that looks like a result.
    from . import extractors  # noqa: F401

    report = Report()
    for index, case in enumerate(cases):
        parent = git_stripped(root, "rev-parse", f"{case.sha}^")
        if not parent:
            report.skipped += 1  # a root commit has no "before"
            continue

        workdir = Path(tempfile.mkdtemp(prefix="codebrain-eval-"))
        checkout = workdir / "tree"
        try:
            added = git(root, "worktree", "add", "--detach", str(checkout), parent)
            if added is None:
                report.skipped += 1
                continue

            ctx = BuildContext(root=checkout, commit=parent)
            past = run_build(ctx).brain

            predicted, tokens = pack_prediction(past, case.subject, budget,
                                                root=checkout)
            if not predicted:
                report.skipped += 1
                continue

            k = min(len(predicted), 25)
            corpus = [p for p in ctx.iter_files() if ctx.readable(p)]
            baseline = grep_prediction(checkout, case.subject, corpus, k)
            report.results.append(CaseResult(
                case=case, pack_files=predicted[:k], grep_files=baseline,
                pack_recall=recall(predicted, case.changed, k),
                grep_recall=recall(baseline, case.changed, k),
                tokens=tokens, k=k,
            ))
            if log:
                log(f"  [{index + 1}/{len(cases)}] {case.subject[:48]}")
        finally:
            git(root, "worktree", "remove", "--force", str(checkout))
            shutil.rmtree(workdir, ignore_errors=True)

    return report


def run_memory_effect(root: Path, cases: list[Case], budget: int = 6000,
                      log=None) -> Report:
    """Does write-back make the *next* task easier?

    P5's gate, measured without a live agent. Git history supplies the pair: for
    two commits A then B that touch overlapping files, A is what a previous
    session did and B is the task now being asked. The Brain is built at B's
    parent, so it cannot know B — and A is genuinely in the past, so ingesting
    it leaks nothing.

    `pack_recall` is the arm with A's session in memory; `grep_recall` carries
    the same pack *without* memory, so the report's delta reads as the effect of
    write-back rather than as a comparison against search.
    """
    import shutil
    import tempfile

    from .memory import Session, from_session
    from .providers import BuildContext
    from .providers import build as run_build

    from . import extractors  # noqa: F401

    ordered = list(reversed(cases))  # git log is newest-first; walk forward
    report = Report()

    for index in range(1, len(ordered)):
        earlier, later = ordered[index - 1], ordered[index]
        if not (set(earlier.changed) & set(later.changed)):
            continue  # unrelated work: memory has nothing to offer
        parent = git_stripped(root, "rev-parse", f"{later.sha}^")
        if not parent:
            report.skipped += 1
            continue

        workdir = Path(tempfile.mkdtemp(prefix="codebrain-mem-"))
        checkout = workdir / "tree"
        try:
            if git(root, "worktree", "add", "--detach", str(checkout), parent) is None:
                report.skipped += 1
                continue

            ctx = BuildContext(root=checkout, commit=parent)
            brain = run_build(ctx).brain

            cold, tokens = pack_prediction(brain, later.subject, budget,
                                           root=checkout)
            brain.extend(from_session(Session(
                session_id=f"eval-{earlier.sha[:8]}", task=earlier.subject,
                files=tuple(earlier.changed), commit=parent, succeeded=True)))
            warm, warm_tokens = pack_prediction(brain, later.subject, budget,
                                                root=checkout)
            if not warm and not cold:
                report.skipped += 1
                continue

            k = min(max(len(warm), len(cold)), 25)
            report.results.append(CaseResult(
                case=later, pack_files=warm[:k], grep_files=cold[:k],
                pack_recall=recall(warm, later.changed, k),
                grep_recall=recall(cold, later.changed, k),
                tokens=warm_tokens, k=k,
            ))
            if log:
                log(f"  [{len(report.results)}] {later.subject[:46]}")
        finally:
            git(root, "worktree", "remove", "--force", str(checkout))
            shutil.rmtree(workdir, ignore_errors=True)

    return report


def evaluate(brain: Brain, root: Path, limit: int = DEFAULT_CASES,
             skip: int = 0, budget: int = 6000, rigorous: bool = False,
             memory: bool = False, log=None) -> Report:
    if not is_repo(root):
        return Report()
    cases = collect_cases(root, limit, skip)
    if memory:
        return run_memory_effect(root, cases, budget, log=log)
    if rigorous:
        return run_rigorous(root, cases, budget, log=log)
    return run(brain, root, cases, budget)
