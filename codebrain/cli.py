"""The CodeBrain command line.

P0 surface only: build, status, diff, validate, providers. The agent-facing
commands (pack, brief, guard, touch, learn, serve) arrive with the context pack
compiler in P2 — this is deliberately the part a human uses to see whether the
Brain is any good before any agent depends on it.

Exit codes matter here: `validate` and `diff` are meant to run in CI, so a
problem is a non-zero exit, not a printed warning.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from .diff import diff as diff_brains
from .diff import render as render_diff
from .model import LAYER_NAMES, Layer
from .providers import REGISTRY, BuildContext
from .providers import build as run_build
from .store import BRAIN_DIR, BrainNotFound, exists, load, save

# Importing the extractors registers them.
from . import extractors  # noqa: F401


def _init_streams() -> None:
    """Never let output encoding kill a command.

    CodeBrain runs inside hooks, and a hook that raises is a hook that breaks
    someone's session. A legacy Windows console cannot encode box-drawing
    characters, so we degrade the rendering rather than the run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def _unicode_ok() -> bool:
    encoding = getattr(sys.stdout, "encoding", "") or "ascii"
    try:
        "█·—".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


_init_streams()
_UNI = _unicode_ok()

FULL = "█" if _UNI else "#"
EMPTY = "·" if _UNI else "."
DOT = " · " if _UNI else " | "
DASH = "—" if _UNI else "-"


def _git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(("git", *args), cwd=str(root), capture_output=True,
                              text=True, timeout=15, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _context(root: Path) -> BuildContext:
    return BuildContext(
        root=root.resolve(),
        commit=_git(root, "rev-parse", "HEAD"),
        branch=_git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        repo=_git(root, "config", "--get", "remote.origin.url") or root.resolve().name,
    )


def _bar(count: int, total: int, width: int = 22) -> str:
    if total <= 0:
        return " " * width
    filled = max(1, round(width * count / total)) if count else 0
    return FULL * filled + EMPTY * (width - filled)


# -- commands --------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.is_dir():
        print(f"codebrain: not a directory: {root}", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else root / BRAIN_DIR
    ctx = _context(root)
    only = set(args.only or ())
    providers = [p for p in REGISTRY.all() if p.id in only] if only else None
    if only and not providers:
        print(f"codebrain: no such provider: {', '.join(sorted(only))}", file=sys.stderr)
        return 2

    previous = load(out) if exists(out) else None
    result = run_build(ctx, providers)
    save(result.brain, out)

    stats = result.brain.stats()
    print(f"Brain built at {out}")
    print(f"  {stats['total']} records{DOT}{stats['nodes']} nodes{DOT}"
          f"{stats['edges']} edges{DOT}{stats['facts']} facts")
    print(f"  providers: {', '.join(result.ran) or 'none'}")
    if result.skipped:
        print(f"  skipped:   {', '.join(result.skipped)} (did not apply)")
    for pid, err in result.failed.items():
        print(f"  FAILED:    {pid} — {err}", file=sys.stderr)
    if result.report.kept:
        print(f"  {result.report.kept} lower-ranked claim(s) lost a conflict "
              f"and were discarded")

    if previous is not None:
        delta = diff_brains(previous, result.brain)
        if delta.substantive:
            c = delta.counts()
            print(f"  changed since last build: +{c['added']} -{c['removed']} "
                  f"~{c['changed']}")
    return 1 if result.failed else 0


def cmd_status(args: argparse.Namespace) -> int:
    try:
        brain = load(args.brain)
    except BrainNotFound as exc:
        print(f"codebrain: {exc}", file=sys.stderr)
        return 2

    stats = brain.stats()
    if args.json:
        print(json.dumps({"manifest": brain.manifest.to_json(), "stats": stats}, indent=2))
        return 0

    m = brain.manifest
    print(f"{m.repo or '(unknown repo)'}{DOT}schema {m.schema_version}{DOT}"
          f"codebrain {m.codebrain_version}")
    print(f"built {m.built_at or '?'}   at {m.as_of[:8] or '?'}"
          f"{' on ' + m.branch if m.branch else ''}")
    print(f"providers: {', '.join(m.providers) or 'none'}")
    print()
    print(f"{stats['total']} records {DASH} {stats['nodes']} nodes, "
          f"{stats['edges']} edges, {stats['facts']} facts")
    print()

    total = stats["total"]
    print("  layer                         records")
    for layer in Layer:
        n = stats["by_layer"].get(str(layer), 0)
        marker = " " if n else EMPTY
        print(f"  {marker} {layer} {LAYER_NAMES[layer]:<12} {_bar(n, total)} {n:>6}")
    print()
    print("  method        " + "   ".join(f"{k} {v}" for k, v in
                                          sorted(stats["by_method"].items())))
    print("  status        " + "   ".join(f"{k} {v}" for k, v in
                                          sorted(stats["by_status"].items())))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        brain = load(args.brain)
    except (BrainNotFound, ValueError) as exc:
        print(f"codebrain: {exc}", file=sys.stderr)
        return 2

    problems = brain.validate()
    if not problems:
        print(f"OK {DASH} {len(brain)} records, no structural problems.")
        return 0

    shown = problems[: args.limit]
    for p in shown:
        print(f"  ! {p}")
    if len(problems) > len(shown):
        print(f"  … {len(problems) - len(shown)} more")
    print(f"\n{len(problems)} problem(s).", file=sys.stderr)
    return 1


def cmd_diff(args: argparse.Namespace) -> int:
    try:
        old = load(args.old)
        new = load(args.new)
    except (BrainNotFound, ValueError) as exc:
        print(f"codebrain: {exc}", file=sys.stderr)
        return 2

    delta = diff_brains(old, new)
    print(render_diff(delta, limit=args.limit))
    # Non-zero only on substantive change, so a rebuild's timestamp churn does
    # not fail CI. This is the seed of the P3 drift gate.
    return 1 if (delta.substantive and args.check) else 0


def cmd_providers(args: argparse.Namespace) -> int:
    root = Path(args.root)
    ctx = _context(root) if root.is_dir() else None
    for p in REGISTRY.all():
        state = ""
        if ctx is not None:
            state = "  applies" if p.applies(ctx) else "  does not apply here"
        layers = " ".join(str(l) for l in p.layers)
        print(f"  {p.id:<12} {layers:<8}{state}")
        if p.description:
            print(f"  {'':<12} {p.description}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root)
    out = root / BRAIN_DIR
    if exists(out) and not args.force:
        print(f"codebrain: a Brain already exists at {out} (use --force to rebuild)",
              file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    print(f"Initialised {out}")
    print("Next: codebrain build")
    return 0


# -- entry point -----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codebrain",
        description="Compile a repository into a Brain.",
    )
    parser.add_argument("--version", action="version", version=f"codebrain {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create an empty .brain/ directory")
    p.add_argument("root", nargs="?", default=".")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("build", help="run extractors and write the Brain")
    p.add_argument("root", nargs="?", default=".")
    p.add_argument("--out", default=None, help="output directory (default <root>/.brain)")
    p.add_argument("--only", action="append", metavar="PROVIDER",
                   help="run only this provider (repeatable)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("status", help="summarise a Brain")
    p.add_argument("brain", nargs="?", default=BRAIN_DIR)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("validate", help="check a Brain for structural problems")
    p.add_argument("brain", nargs="?", default=BRAIN_DIR)
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("diff", help="compare two Brains")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--check", action="store_true",
                   help="exit non-zero on substantive change (CI drift gate)")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("providers", help="list registered extractors")
    p.add_argument("root", nargs="?", default=".")
    p.set_defaults(func=cmd_providers)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
