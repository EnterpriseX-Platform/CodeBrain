"""L4 intent — git archaeology.

The layer that makes legacy tractable. Structure tells you a function exists;
only history tells you it was rewritten three times, that one person wrote all
of it and left, and that it never changes without another file changing too.
That is the knowledge that walks out of the building.

Everything here comes from a single `git log --numstat` pass, because the cold
build has a wall-clock budget and history is the most expensive honest layer.

Reproducibility: windows are counted in commits, never in days. A Brain built
from the same HEAD twice must be byte-identical, or the drift gate becomes
noise (see gitutil).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

from ..envelope import Envelope, Evidence, Method, Status
from ..gitutil import REC, REC_FMT, SEP, SEP_FMT, git, is_repo, normalise_rename
from ..model import REPO, Edge, Fact, Layer, Record
from ..providers import BuildContext, Provider, register

#: How far back to look, in commits. Bounds the cold build on repos with deep
#: history without making the result depend on the calendar.
DEFAULT_MAX_COMMITS = 3000

#: Commits touching more than this are excluded from co-change coupling. A
#: sweeping reformat or a vendored-dependency bump couples everything to
#: everything and drowns the real signal.
COUPLING_MAX_FILES = 40

#: Minimum co-occurrences before a coupling edge is worth asserting.
COUPLING_MIN_COMMITS = 3

#: Cap on emitted coupling edges, strongest first.
COUPLING_MAX_EDGES = 2000

HOTSPOT_COUNT = 25


class Commit:
    __slots__ = ("sha", "when", "author", "files")

    def __init__(self, sha: str, when: str, author: str) -> None:
        self.sha = sha
        self.when = when
        self.author = author
        self.files: list[tuple[str, int, int]] = []


def parse_log(raw: str) -> list[Commit]:
    """Parse `git log --numstat` output into commits.

    Binary files report `-` for added/deleted; they are counted as touched but
    contribute no line churn.
    """
    commits: list[Commit] = []
    for chunk in raw.split(REC):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        lines = chunk.split("\n")
        header = lines[0].split(SEP)
        if len(header) < 3:
            continue
        commit = Commit(header[0], header[1], header[2])
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            added, deleted, path = parts
            commit.files.append((
                normalise_rename(path),
                int(added) if added.isdigit() else 0,
                int(deleted) if deleted.isdigit() else 0,
            ))
        commits.append(commit)
    return commits


class HistoryProvider(Provider):
    id = "history"
    layers = (Layer.L4,)
    description = "Git archaeology: churn, hotspots, ownership, co-change coupling."
    order = 30  # after census — history attaches to the file nodes census creates

    def applies(self, ctx: BuildContext) -> bool:
        return is_repo(ctx.root)

    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        max_commits = int(ctx.config.get("max_commits", DEFAULT_MAX_COMMITS))
        raw = git(
            ctx.root, "log", "--no-merges", "--numstat", "--find-renames",
            f"--max-count={max_commits}",
            f"--format={REC_FMT}%H{SEP_FMT}%aI{SEP_FMT}%aN",
        )
        if not raw:
            return

        commits = parse_log(raw)
        if not commits:
            return

        # The Brain describes HEAD. History for files that no longer exist would
        # dangle off nodes census never created.
        live = {ctx.rel(p) for p in ctx.iter_files()}

        commits_touching: dict[str, int] = defaultdict(int)
        insertions: dict[str, int] = defaultdict(int)
        deletions: dict[str, int] = defaultdict(int)
        authors: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        last_seen: dict[str, str] = {}
        first_seen: dict[str, str] = {}
        pairs: dict[tuple[str, str], int] = defaultdict(int)

        for commit in commits:
            touched = [f for f in commit.files if f[0] in live]
            for path, added, removed in touched:
                commits_touching[path] += 1
                insertions[path] += added
                deletions[path] += removed
                authors[path][commit.author] += 1
                # git log walks newest first, so the first sighting is the latest.
                last_seen.setdefault(path, commit.when)
                first_seen[path] = commit.when

            if 1 < len(touched) <= COUPLING_MAX_FILES:
                for a, b in combinations(sorted({f[0] for f in touched}), 2):
                    pairs[(a, b)] += 1

        window = len(commits)
        head_sha = commits[0].sha

        def env(method: Method, path: str, note: str = "") -> Envelope:
            return Envelope.make(
                method, source=self.id, as_of=ctx.commit or head_sha, ts=ctx.ts,
                status=Status.FRESH,
                evidence=(Evidence(path=path, commit=head_sha, ref="git log --numstat"),),
                note=note,
            )

        yield from self._per_file(ctx, live, commits_touching, insertions, deletions,
                                  authors, first_seen, last_seen, window, env)
        yield from self._coupling(pairs, commits_touching, env)
        yield from self._repo_level(commits_touching, insertions, deletions, live,
                                    window, env)

    # -- per file ----------------------------------------------------------

    def _per_file(self, ctx, live, commits_touching, insertions, deletions, authors,
                  first_seen, last_seen, window, env) -> Iterable[Record]:
        for path, count in commits_touching.items():
            file_id = f"{Layer.L0}:file:{path}"

            yield Fact(
                layer=Layer.L4, subject=file_id, predicate="churn",
                value={"commits": count, "insertions": insertions[path],
                       "deletions": deletions[path], "window_commits": window},
                env=env(Method.EXTRACTED, path),
            )
            yield Fact(
                layer=Layer.L4, subject=file_id, predicate="last_changed_at",
                value=last_seen[path], env=env(Method.EXTRACTED, path),
            )
            yield Fact(
                layer=Layer.L4, subject=file_id, predicate="first_changed_at",
                value=first_seen[path], env=env(Method.EXTRACTED, path),
            )

            ranked = sorted(authors[path].items(), key=lambda kv: (-kv[1], kv[0]))
            yield Fact(
                layer=Layer.L4, subject=file_id, predicate="authors",
                value=[{"author": a, "commits": n} for a, n in ranked[:10]],
                env=env(Method.EXTRACTED, path),
            )

            # Ownership concentration. A file one person has ever touched is a
            # bus-factor-of-one risk; L6 turns this into a constraint later.
            top_share = ranked[0][1] / count if count else 0.0
            yield Fact(
                layer=Layer.L4, subject=file_id, predicate="ownership",
                value={"primary_author": ranked[0][0],
                       "share": round(top_share, 3),
                       "distinct_authors": len(ranked)},
                env=env(Method.DERIVED, path,
                        "share of commits by the most frequent author"),
            )

    # -- coupling ----------------------------------------------------------

    def _coupling(self, pairs, commits_touching, env) -> Iterable[Record]:
        scored: list[tuple[float, int, str, str]] = []
        for (a, b), together in pairs.items():
            if together < COUPLING_MIN_COMMITS:
                continue
            # Jaccard: co-changes over the union of each file's changes. Raw
            # co-change count alone just re-reports which files churn most.
            union = commits_touching[a] + commits_touching[b] - together
            if union <= 0:
                continue
            scored.append((together / union, together, a, b))

        scored.sort(key=lambda s: (-s[0], -s[1], s[2], s[3]))
        for strength, together, a, b in scored[:COUPLING_MAX_EDGES]:
            yield Edge(
                layer=Layer.L4, kind="changed_with",
                src=f"{Layer.L0}:file:{a}", dst=f"{Layer.L0}:file:{b}",
                env=env(Method.DERIVED, a,
                        "co-change coupling, Jaccard over commit sets"),
                attrs={"commits": together, "strength": round(strength, 3)},
            )

    # -- repo level --------------------------------------------------------

    def _repo_level(self, commits_touching, insertions, deletions, live, window,
                    env) -> Iterable[Record]:
        # Hotspot = churn weighted by how much of the file moves. A config file
        # edited constantly one line at a time is not the same risk as a module
        # rewritten repeatedly.
        hotspots = []
        for path, count in commits_touching.items():
            volume = insertions[path] + deletions[path]
            hotspots.append({
                "path": path, "commits": count, "lines_changed": volume,
                "score": round(count * (volume ** 0.5), 1),
            })
        hotspots.sort(key=lambda h: (-h["score"], h["path"]))

        yield Fact(
            layer=Layer.L4, subject=REPO, predicate="hotspots",
            value=hotspots[:HOTSPOT_COUNT],
            env=env(Method.DERIVED, ".", "commits x sqrt(lines changed)"),
        )
        yield Fact(
            layer=Layer.L4, subject=REPO, predicate="history_window",
            value={"commits_scanned": window, "files_with_history": len(commits_touching),
                   "files_at_head": len(live)},
            env=env(Method.EXTRACTED, "."),
        )

        # An honest gap rather than a silent one: files present at HEAD that the
        # scanned window never touched have no history signal at all.
        untouched = len(live) - len(commits_touching)
        if untouched > 0:
            yield Fact(
                layer=Layer.L4, subject=REPO, predicate="history_coverage_gap",
                value={"files_without_history": untouched,
                       "reason": f"not touched within the last {window} commits"},
                env=env(Method.DERIVED, "."),
            )


register(HistoryProvider())
