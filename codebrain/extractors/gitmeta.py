"""L0/L4 git metadata — identity, and the first thin slice of history.

This is the seed of the intent layer. P1 grows it into real archaeology (churn,
hotspots, co-change coupling, ownership); for now it establishes that a provider
can write two layers, can decline to run, and can outrank another provider's
belief with better evidence.
"""

from __future__ import annotations

import subprocess
from typing import Iterable

from ..envelope import Envelope, Evidence, Method, Status
from ..model import REPO, Fact, Layer, Node, Record
from ..providers import BuildContext, Provider, register

TIMEOUT = 20


def git(ctx: BuildContext, *args: str) -> str | None:
    """Run a git command, or return None. Never raises: a shallow clone, a
    missing binary and a corrupt object store all mean the same thing here —
    this provider has nothing to say."""
    try:
        proc = subprocess.run(
            ("git", *args), cwd=str(ctx.root), capture_output=True, text=True,
            timeout=TIMEOUT, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


class GitMetaProvider(Provider):
    id = "gitmeta"
    layers = (Layer.L0, Layer.L4)
    description = "Git identity and top-level history signals."
    order = 20

    def applies(self, ctx: BuildContext) -> bool:
        return git(ctx, "rev-parse", "--git-dir") is not None

    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        head = git(ctx, "rev-parse", "HEAD") or ctx.commit
        if not head:
            return

        def env(method: Method = Method.EXTRACTED, ref: str = "git", **kw) -> Envelope:
            return Envelope.make(
                method, source=self.id, as_of=head, ts=ctx.ts,
                evidence=(Evidence(path=".git", commit=head, ref=ref),),
                status=Status.FRESH, **kw,
            )

        yield Fact(layer=Layer.L0, subject=REPO, predicate="head", value=head,
                   env=env(ref="rev-parse HEAD"))

        branch = git(ctx, "rev-parse", "--abbrev-ref", "HEAD")
        if branch and branch != "HEAD":
            yield Fact(layer=Layer.L0, subject=REPO, predicate="branch", value=branch,
                       env=env(ref="rev-parse --abbrev-ref HEAD"))

        remote = git(ctx, "config", "--get", "remote.origin.url")
        if remote:
            yield Fact(layer=Layer.L0, subject=REPO, predicate="remote", value=remote,
                       env=env(ref="config remote.origin.url"))
            # The authoritative repo name — from the remote, not a guess at the
            # directory. This deliberately collides with census's DERIVED claim.
            name = remote.rstrip("/").rsplit("/", 1)[-1]
            if name.endswith(".git"):
                name = name[:-4]
            if name:
                yield Fact(layer=Layer.L0, subject=REPO, predicate="repo_name", value=name,
                           env=env(ref="config remote.origin.url"))

        count = git(ctx, "rev-list", "--count", "HEAD")
        if count and count.isdigit():
            yield Fact(layer=Layer.L4, subject=REPO, predicate="commit_count",
                       value=int(count), env=env(ref="rev-list --count HEAD"))

        first = git(ctx, "log", "--reverse", "--format=%aI", "--max-parents=0")
        if first:
            yield Fact(layer=Layer.L4, subject=REPO, predicate="first_commit_at",
                       value=first.splitlines()[0], env=env(ref="log --reverse"))

        last = git(ctx, "log", "-1", "--format=%aI")
        if last:
            yield Fact(layer=Layer.L4, subject=REPO, predicate="last_commit_at",
                       value=last, env=env(ref="log -1"))

        # Authorship. Ownership is an L4 signal that L6 later turns into a
        # constraint ("this file has one author and they left").
        shortlog = git(ctx, "shortlog", "-sne", "--all", "HEAD")
        if shortlog:
            authors: list[dict[str, object]] = []
            for line in shortlog.splitlines():
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                commits, who = line.split("\t", 1)
                if not commits.strip().isdigit():
                    continue
                handle = who.strip()
                authors.append({"author": handle, "commits": int(commits.strip())})
                yield Node(
                    layer=Layer.L4, kind="author", key=handle, name=handle,
                    env=env(ref="shortlog -sne"),
                    attrs={"commits": int(commits.strip())},
                )
            if authors:
                yield Fact(
                    layer=Layer.L4, subject=REPO, predicate="top_authors",
                    value=authors[:10], env=env(Method.DERIVED, ref="shortlog -sne"),
                )


register(GitMetaProvider())
