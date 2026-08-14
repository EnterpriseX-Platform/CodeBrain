"""The extractor plugin contract.

Principle viii: pluggable, not language-locked. Every layer is populated by
providers behind this one interface, so adding Go support or swapping in a
third-party graph tool never touches the core.

A provider must be honest about two things: which layers it writes, and whether
it applies at all. A provider that cannot run returns False from `applies` — it
does not raise, and it does not emit low-confidence guesses to look busy.
"""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .envelope import utc_now
from .model import Brain, Layer, MergeReport, Record, new_brain

#: Directories never worth walking. Vendored and generated trees produce
#: enormous, uninformative Brains.
DEFAULT_IGNORE = (
    ".git", ".hg", ".svn", ".brain", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".env", "dist", "build", "target", "out", ".next", ".nuxt", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "vendor", "third_party",
    "site-packages", ".gradle", ".idea", ".vscode", "coverage", ".cache",
)

#: Extensions we will not read as text under any circumstances.
BINARY_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tiff", ".pdf",
    ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".jar", ".war", ".class",
    ".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a", ".pyc", ".pyo",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp3", ".mp4", ".mov", ".avi",
    ".wav", ".webm", ".db", ".sqlite", ".sqlite3", ".lock",
})


@dataclass(slots=True)
class BuildContext:
    """Everything a provider is allowed to know about the build."""

    root: Path
    commit: str = ""
    branch: str = ""
    repo: str = ""
    ts: str = field(default_factory=utc_now)
    config: dict[str, Any] = field(default_factory=dict)
    ignore: tuple[str, ...] = DEFAULT_IGNORE
    max_file_bytes: int = 2_000_000

    #: The Brain as it stands so far, set by `build` before providers run.
    #: Some layers are inherently derivative — L6 constraints are computed from
    #: L4 ownership and L5 CODEOWNERS, not read from disk — so a late provider
    #: must be able to see what earlier ones established. Only providers with a
    #: higher `order` than their inputs may rely on this; anything else is
    #: reading a half-built Brain and will produce order-dependent output.
    brain: Any = None

    def rel(self, path: Path) -> str:
        """Repo-relative, forward-slashed. Ids must not differ between Windows and CI."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def ignored(self, path: Path) -> bool:
        parts = set(path.parts)
        if parts & set(self.ignore):
            return True
        return any(fnmatch.fnmatch(path.name, pat) for pat in self.ignore if "*" in pat)

    def iter_files(self) -> Iterator[Path]:
        """Every candidate file, ignores applied, pruned as we descend."""
        stack = [self.root]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(current.iterdir())
            except (PermissionError, OSError):
                continue
            for entry in entries:
                if entry.name in self.ignore:
                    continue
                if entry.is_symlink():
                    continue  # symlinks invite cycles and double-counting
                if entry.is_dir():
                    stack.append(entry)
                elif entry.is_file():
                    yield entry

    def readable(self, path: Path) -> bool:
        if path.suffix.lower() in BINARY_SUFFIXES:
            return False
        try:
            return path.stat().st_size <= self.max_file_bytes
        except OSError:
            return False


class Provider(ABC):
    """One extractor. Deterministic providers must produce identical output for
    identical input — that property is what makes the drift gate trustworthy."""

    id: str = "unnamed"
    layers: tuple[Layer, ...] = ()
    description: str = ""
    #: Providers with lower order run first, so later ones can rely on earlier
    #: nodes existing (edges need their endpoints).
    order: int = 100
    #: True when `applies` inspects `ctx.brain` rather than the filesystem. Such
    #: a provider cannot be asked whether it applies before the build starts,
    #: because the answer depends on what earlier providers found.
    derivative: bool = False

    def applies(self, ctx: BuildContext) -> bool:
        return True

    @abstractmethod
    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id!r} layers={[str(l) for l in self.layers]}>"


class Registry:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> Provider:
        if provider.id in self._providers:
            raise ValueError(f"provider id {provider.id!r} is already registered")
        self._providers[provider.id] = provider
        return provider

    def unregister(self, provider_id: str) -> None:
        self._providers.pop(provider_id, None)

    def get(self, provider_id: str) -> Provider | None:
        return self._providers.get(provider_id)

    def all(self) -> list[Provider]:
        return sorted(self._providers.values(), key=lambda p: (p.order, p.id))

    def applicable(self, ctx: BuildContext) -> list[Provider]:
        return [p for p in self.all() if p.applies(ctx)]

    def __len__(self) -> int:
        return len(self._providers)


REGISTRY = Registry()


def register(provider: Provider) -> Provider:
    return REGISTRY.register(provider)


@dataclass(slots=True)
class BuildResult:
    brain: Brain
    report: MergeReport
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def build(ctx: BuildContext, providers: Iterable[Provider] | None = None) -> BuildResult:
    """Run providers and fold their output into one Brain.

    A provider that raises is recorded and skipped, never fatal. Principle vi:
    a partial Brain beats no Brain, and a broken Go extractor must not cost you
    the Python one.
    """
    # The Brain must exist before anyone is asked whether they apply: a
    # derivative provider decides by looking at what is already known, and
    # asking it against a context with no Brain silently excludes it from the
    # build entirely.
    brain = new_brain(repo=ctx.repo, as_of=ctx.commit, branch=ctx.branch)
    ctx.brain = brain
    # Every registered provider is a candidate; `applies` is asked inside the
    # loop, when that provider's turn actually comes. Filtering up front asks a
    # derivative provider whether it applies against an empty Brain, which it
    # never does — that silently dropped L6 once and L3 again.
    chosen = list(providers) if providers is not None else REGISTRY.all()
    result = BuildResult(brain=brain, report=MergeReport())

    for provider in sorted(chosen, key=lambda p: (p.order, p.id)):
        if not provider.applies(ctx):
            result.skipped.append(provider.id)
            continue
        try:
            report = brain.extend(provider.extract(ctx))
        except Exception as exc:  # noqa: BLE001 — a bad provider must not kill the build
            result.failed[provider.id] = f"{type(exc).__name__}: {exc}"
            continue
        result.report.added += report.added
        result.report.replaced += report.replaced
        result.report.kept += report.kept
        result.ran.append(provider.id)

    brain.manifest.providers = list(result.ran)
    return result
