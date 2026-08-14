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
from .atlas import render as render_atlas
from .diff import diff as diff_brains
from .diff import render as render_diff
from .model import LAYER_NAMES, REPO as REPO_SUBJECT, Layer
from .pack import DEFAULT_BUDGET, brief as render_brief, compile_pack
from .providers import REGISTRY, BuildContext
from .providers import build as run_build
from .store import (
    BRAIN_DIR,
    BrainNotFound,
    append_memory,
    apply_touched,
    clear_touched,
    exists,
    load,
    read_touched,
    record_touch,
    save,
)

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

    atlas_path = out / "ATLAS.md"
    atlas_path.write_text(render_atlas(result.brain), encoding="utf-8", newline="\n")
    clear_touched(out)  # everything just re-extracted is fresh again

    stats = result.brain.stats()
    print(f"Brain built at {out}")
    print(f"  {stats['total']} records{DOT}{stats['nodes']} nodes{DOT}"
          f"{stats['edges']} edges{DOT}{stats['facts']} facts")
    print(f"  atlas:     {atlas_path}")
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
    args.brain = _resolve_brain(args)
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
    args.brain = _resolve_brain(args)
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


# -- agent-facing commands -------------------------------------------------
#
# These run inside hooks. Every one of them fails open: if the Brain is missing,
# stale or corrupt, they print nothing and exit 0, and the session proceeds
# exactly as it would without CodeBrain installed. A Brain that can break
# someone's session gets uninstalled the first time it does.


def _hook_input() -> dict:
    """Read the hook payload from stdin. Absent or malformed is not an error."""
    if sys.stdin is None or sys.stdin.isatty():
        return {}
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_quietly(brain_dir: str):
    """Load a Brain with edits-since-build applied, or None."""
    try:
        brain = load(brain_dir)
    except (BrainNotFound, ValueError, OSError):
        return None
    try:
        apply_touched(brain, read_touched(brain_dir))
    except Exception:  # noqa: BLE001 — staleness is a nicety, never a failure
        pass
    return brain


def cmd_pack(args: argparse.Namespace) -> int:
    payload = _hook_input() if args.stdin else {}
    task = args.task or payload.get("prompt") or ""
    if not task.strip():
        return 0

    brain = _load_quietly(args.brain)
    if brain is None:
        return 0

    try:
        pack = compile_pack(brain, task, budget=args.budget,
                            max_anchors=args.anchors, root=args.root)
    except Exception:  # noqa: BLE001
        return 0

    if args.json:
        print(json.dumps(pack.to_json(), indent=2))
    elif pack.items or not args.quiet:
        print(pack.render())
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    brain = _load_quietly(args.brain)
    if brain is None:
        return 0
    try:
        print(render_brief(brain, budget=args.budget))
    except Exception:  # noqa: BLE001
        pass
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    """PreToolUse: check a pending edit against L6 before it happens."""
    payload = _hook_input()
    tool_input = payload.get("tool_input") or {}
    target = tool_input.get("file_path") or args.path or ""
    if not target:
        return 0

    brain = _load_quietly(args.brain)
    if brain is None:
        return 0

    rel = str(target).replace("\\", "/")
    try:
        root = str(Path(args.root).resolve()).replace("\\", "/")
        if rel.replace("\\", "/").startswith(root):
            rel = rel[len(root):].lstrip("/")
    except OSError:
        pass

    file_id = f"{Layer.L0}:file:{rel}"
    reasons: list[str] = []
    blocking = False
    declared_block = False

    # A policy zone is the one constraint a human stated outright rather than
    # the machinery inferring it, so it is the only one allowed to stop an edit
    # on its own. Everything else warns.
    zone = brain.fact(file_id, "policy_zone", Layer.L6)
    if zone:
        value = zone.value or {}
        detail = f"{rel} is in the '{value.get('zone')}' policy zone"
        if value.get("reason"):
            detail += f" — {value['reason']}"
        if value.get("requires"):
            detail += f" (requires {', '.join(value['requires'])})"
        reasons.append(detail)
        blocking = True
        declared_block = bool(value.get("block_agents"))

    review = brain.fact(file_id, "requires_review", Layer.L6)
    if review:
        owners = " ".join((review.value or {}).get("owners", []))
        reasons.append(f"{rel} requires review from {owners} (CODEOWNERS)")
        blocking = True

    danger = brain.fact(file_id, "danger_zone", Layer.L6)
    if danger:
        value = danger.value or {}
        reasons.append(f"{rel} is hotspot #{value.get('rank')} — "
                       f"{value.get('commits')} commits, {value.get('reason')}")

    bus = brain.fact(file_id, "bus_factor_risk", Layer.L6)
    if bus:
        reasons.append(f"{rel} has only ever been changed by "
                       f"{(bus.value or {}).get('primary_author')}")

    untested = brain.fact(file_id, "untested_churn", Layer.L6)
    if untested:
        value = untested.value or {}
        reasons.append(f"{rel} changes often ({value.get('commits')} commits) and "
                       "no test reaches it — a mistake here fails silently")

    if not reasons:
        return 0

    # Inferred constraints warn; they never block, because denying an edit on
    # churn would get the hook uninstalled by lunchtime. A policy zone a human
    # declared with block_agents is different: that is a stated decision, and
    # it is the only thing here allowed to stop an edit on its own.
    if declared_block:
        decision = "deny"
    elif blocking and args.deny_guarded:
        decision = "ask"
    else:
        decision = "allow"
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": "CodeBrain: " + "; ".join(reasons),
    }}))
    return 0


def cmd_touch(args: argparse.Namespace) -> int:
    """PostToolUse: note an edit so later packs do not serve stale facts."""
    payload = _hook_input()
    tool_input = payload.get("tool_input") or {}
    target = tool_input.get("file_path") or args.path or ""
    if not target:
        return 0

    rel = str(target).replace("\\", "/")
    try:
        root = str(Path(args.root).resolve()).replace("\\", "/")
        if rel.startswith(root):
            rel = rel[len(root):].lstrip("/")
    except OSError:
        pass
    try:
        record_touch(args.brain, [rel])
    except OSError:
        pass
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .verify import VERIFIABLE_INTENTS, render as render_verify, verify

    try:
        brain = load(args.brain)
    except (BrainNotFound, ValueError) as exc:
        print(f"codebrain: {exc}", file=sys.stderr)
        return 2

    intents = args.intent or list(VERIFIABLE_INTENTS)
    report = verify(brain, Path(args.root), intents=intents, timeout=args.timeout,
                    execute_commands=args.yes, commit=brain.manifest.as_of)
    print(render_verify(report))

    if args.yes and (report.promoted or report.refuted):
        save(brain, args.brain)
        atlas = Path(args.brain) / "ATLAS.md"
        atlas.write_text(render_atlas(brain), encoding="utf-8", newline="\n")
    # A refuted claim is a finding, not a crash: report it, and let CI decide.
    return 1 if (args.check and report.failed) else 0


def cmd_sync(args: argparse.Namespace) -> int:
    from .sync import SyncReport, carry_forward, needs_rebuild

    brain_dir = Path(args.brain)
    root = Path(args.root)
    try:
        previous = load(brain_dir)
    except (BrainNotFound, ValueError):
        print("No Brain yet — running a full build.")
        return cmd_build(argparse.Namespace(root=args.root, out=args.brain, only=None))

    rebuild, reason, changed = needs_rebuild(previous, root, brain_dir)
    if not rebuild and not args.force:
        print(SyncReport(rebuilt=False, reason=reason, changed=changed).render())
        return 0
    # Past this point we are definitely rebuilding, so the report must say so —
    # a forced sync that reports "up to date" describes the opposite of what
    # just happened.
    report = SyncReport(rebuilt=True, changed=changed,
                        reason=reason if rebuild else "forced")

    result = run_build(_context(root))
    report.carried, report.invalidated = carry_forward(previous, result.brain)
    report.delta = diff_brains(previous, result.brain)
    save(result.brain, brain_dir)
    (brain_dir / "ATLAS.md").write_text(render_atlas(result.brain),
                                        encoding="utf-8", newline="\n")
    clear_touched(brain_dir)
    print(report.render())
    return 1 if result.failed else 0


def cmd_drift(args: argparse.Namespace) -> int:
    from .sync import drift

    try:
        committed = load(args.brain)
    except (BrainNotFound, ValueError) as exc:
        print(f"codebrain: {exc}", file=sys.stderr)
        return 2

    fresh = run_build(_context(Path(args.root))).brain
    delta = drift(committed, fresh)

    if not delta.substantive:
        print(f"No drift — the Brain still describes HEAD ({len(committed)} records).")
        return 0

    counts = delta.counts()
    print(f"DRIFT: the committed Brain no longer describes the code.")
    print(f"  +{counts['added']} -{counts['removed']} ~{counts['changed']} records")
    print()
    print(render_diff(delta, limit=args.limit))
    print()
    print("  Run `codebrain sync` and commit the result.")
    return 1 if args.check else 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, render as render_eval

    try:
        brain = load(args.brain)
    except (BrainNotFound, ValueError) as exc:
        print(f"codebrain: {exc}", file=sys.stderr)
        return 2

    if args.rigorous:
        print(f"Building a Brain per commit (up to {args.cases}) "
              "— this is slow by design.")
    report = evaluate(brain, Path(args.root), limit=args.cases, skip=args.skip,
                      budget=args.budget, rigorous=args.rigorous,
                      memory=args.memory,
                      log=print if args.verbose else None)
    print(render_eval(report, verbose=args.verbose, rigorous=args.rigorous,
                      memory=args.memory))
    if not report.n:
        return 0
    # A pack that does not beat plain search has not earned its place in the
    # session, so CI can gate on it.
    delta = report.mean("pack_recall") - report.mean("grep_recall")
    return 1 if (args.check and delta <= 0) else 0


def cmd_learn(args: argparse.Namespace) -> int:
    """Stop hook: fold what a session did and found into L7."""
    from .memory import Session, from_session

    payload = _hook_input() if not sys.stdin.isatty() else {}
    brain_dir = Path(args.brain)
    if not brain_dir.is_dir():
        return 0

    touched = sorted(read_touched(brain_dir))
    task = args.task or payload.get("prompt") or ""
    lessons = tuple(args.lesson or ())
    questions = tuple(args.question or ())
    if not (touched or lessons or questions):
        return 0  # nothing happened worth remembering

    try:
        brain = load(brain_dir)
    except (BrainNotFound, ValueError, OSError):
        return 0

    commits = brain.fact(REPO_SUBJECT, "commit_count", Layer.L4)
    session = Session(
        session_id=str(payload.get("session_id") or args.session or "local"),
        task=task, files=tuple(touched), lessons=lessons, questions=questions,
        commit=brain.manifest.as_of,
        commits_now=commits.value if commits and isinstance(commits.value, int) else 0,
        succeeded=None if args.outcome is None else args.outcome == "success",
    )

    try:
        written = append_memory(brain_dir, from_session(session))
    except (OSError, ValueError):
        return 0
    if not args.quiet:
        print(f"Recorded {written} memory record(s) from session "
              f"{session.session_id}.")
    return 0


def cmd_remember(args: argparse.Namespace) -> int:
    from .memory import remember

    try:
        brain = load(args.brain)
    except (BrainNotFound, ValueError) as exc:
        print(f"codebrain: {exc}", file=sys.stderr)
        return 2

    commits = brain.fact(REPO_SUBJECT, "commit_count", Layer.L4)
    about = f"{Layer.L0}:file:{args.about}" if args.about else REPO_SUBJECT
    fact = remember(args.text, about=about, task=args.task or "",
                    commit=brain.manifest.as_of,
                    commits_now=commits.value if commits and
                    isinstance(commits.value, int) else 0,
                    human=args.human)
    append_memory(args.brain, [fact])
    who = "human" if args.human else "agent"
    print(f"Remembered ({who}, {fact.env.method}): {args.text}")
    return 0


def cmd_dispute(args: argparse.Namespace) -> int:
    from .memory import dispute

    try:
        brain = load(args.brain)
    except (BrainNotFound, ValueError) as exc:
        print(f"codebrain: {exc}", file=sys.stderr)
        return 2

    commits = brain.fact(REPO_SUBJECT, "commit_count", Layer.L4)
    record, outcome = dispute(
        brain, args.record, args.reason, commit=brain.manifest.as_of,
        commits_now=commits.value if commits and isinstance(commits.value, int) else 0,
        human=args.human)

    if outcome == "missing":
        print(f"codebrain: no record {args.record!r} in this Brain", file=sys.stderr)
        return 2

    append_memory(args.brain, [record])
    if outcome == "overruled":
        save(brain, args.brain)  # a human demotion changes the claim itself
        print(f"Overruled {args.record} — demoted and recorded.")
    else:
        print(f"Dispute recorded against {args.record}. Packs will surface it; "
              "the claim itself is unchanged.\n"
              "Only a human (--human) may demote an extracted fact.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .mcp_server import serve

    return serve(args.brain, root=args.root)


def cmd_atlas(args: argparse.Namespace) -> int:
    args.brain = _resolve_brain(args)
    try:
        brain = load(args.brain)
    except (BrainNotFound, ValueError) as exc:
        print(f"codebrain: {exc}", file=sys.stderr)
        return 2

    text = render_atlas(brain)
    if args.out == "-":
        # May contain characters a legacy console cannot encode; streams are
        # already configured to replace rather than raise.
        print(text)
    else:
        target = Path(args.out) if args.out else Path(args.brain) / "ATLAS.md"
        target.write_text(text, encoding="utf-8", newline="\n")
        print(f"Atlas written to {target} ({len(text.splitlines())} lines)")
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    root = Path(args.root)
    ctx = _context(root) if root.is_dir() else None
    for p in REGISTRY.all():
        state = ""
        if p.derivative:
            state = "  decided during the build (reads earlier layers)"
        elif ctx is not None:
            state = "  applies" if p.applies(ctx) else "  does not apply here"
        layers = " ".join(str(l) for l in p.layers)
        print(f"  {p.id:<12} {layers:<8}{state}")
        if p.description:
            print(f"  {'':<12} {p.description}")
    return 0


MCP_CONFIG = {"mcpServers": {"codebrain": {"command": "codebrain",
                                           "args": ["serve", "--mcp"]}}}

HOOKS = {
    "SessionStart": [{"hooks": [{"type": "command", "command": "codebrain brief"}]}],
    "UserPromptSubmit": [{"hooks": [{"type": "command",
                                     "command": "codebrain pack --stdin --quiet"}]}],
    "PreToolUse": [{"matcher": "Edit|Write|MultiEdit",
                    "hooks": [{"type": "command", "command": "codebrain guard"}]}],
    "PostToolUse": [{"matcher": "Edit|Write|MultiEdit",
                     "hooks": [{"type": "command", "command": "codebrain touch"}]}],
    "Stop": [{"hooks": [{"type": "command",
                         "command": "codebrain learn --quiet"}]}],
}

CLAUDE_STANZA = """\
<!-- codebrain:start -->
## This repository has a Brain

Before searching the codebase, ask for a context pack — it is cheaper than
grepping and every line in it is cited:

    codebrain pack "<what you are about to do>"

Claims are tagged `EXTRACTED`, `DERIVED`, `INFERRED`, `OBSERVED` or `ASSERTED`.
Prefer `EXTRACTED` and `OBSERVED` for risky changes. Treat the `UNKNOWNS`
section as the edge of what is known, not as nothing to worry about.

`.brain/ATLAS.md` is the generated overview. Do not edit anything under
`.brain/` by hand — regenerate with `codebrain build`.
<!-- codebrain:end -->
"""


def _merge_json(path: Path, addition: dict) -> str:
    """Merge into an existing JSON config without discarding what is there."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(addition, indent=2) + "\n", encoding="utf-8")
        return "created"
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "left alone (unreadable)"
    if not isinstance(current, dict):
        return "left alone (unexpected shape)"

    changed = False
    for top, block in addition.items():
        section = current.setdefault(top, {})
        if not isinstance(section, dict):
            continue
        for key, value in block.items():
            if key not in section:
                section[key] = value
                changed = True
    if not changed:
        return "already configured"
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return "updated"


def _merge_hooks(path: Path) -> str:
    if path.is_file():
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return "left alone (unreadable)"
        if not isinstance(settings, dict):
            return "left alone (unexpected shape)"
    else:
        settings = {}
        path.parent.mkdir(parents=True, exist_ok=True)

    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        return "left alone (unexpected shape)"

    added = []
    for event, entries in HOOKS.items():
        existing = hooks.setdefault(event, [])
        if not isinstance(existing, list):
            continue
        # Never duplicate ours, never touch anyone else's.
        already = json.dumps(existing)
        if "codebrain" in already:
            continue
        existing.extend(entries)
        added.append(event)

    if not added:
        return "already configured"
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return f"added {', '.join(added)}"


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.is_dir():
        print(f"codebrain: not a directory: {root}", file=sys.stderr)
        return 2

    out = root / BRAIN_DIR
    out.mkdir(parents=True, exist_ok=True)
    print(f"Brain directory   {out}")

    if not args.no_mcp:
        print(f"MCP server        .mcp.json — {_merge_json(root / '.mcp.json', MCP_CONFIG)}")

    if not args.no_hooks:
        state = _merge_hooks(root / ".claude" / "settings.json")
        print(f"Hooks             .claude/settings.json — {state}")

    if not args.no_claude_md:
        claude_md = root / "CLAUDE.md"
        current = claude_md.read_text(encoding="utf-8") if claude_md.is_file() else ""
        if "codebrain:start" in current:
            state = "already present"
        else:
            separator = "\n\n" if current.strip() else ""
            claude_md.write_text(current + separator + CLAUDE_STANZA, encoding="utf-8")
            state = "appended" if current.strip() else "created"
        print(f"Instructions      CLAUDE.md — {state}")

    print()
    print("Next: codebrain build")
    return 0


# -- entry point -----------------------------------------------------------


def _brain_arg(parser: argparse.ArgumentParser) -> None:
    """Accept the Brain path positionally *and* as --brain.

    Half the commands took it one way and half the other, which is the kind of
    inconsistency that costs a minute every time and is never worth anyone's
    while to fix later.
    """
    parser.add_argument("brain_pos", nargs="?", default=None, metavar="brain")
    parser.add_argument("--brain", dest="brain_opt", default=None)


def _resolve_brain(args: argparse.Namespace) -> str:
    return getattr(args, "brain_opt", None) or getattr(args, "brain_pos", None) \
        or BRAIN_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codebrain",
        description="Compile a repository into a Brain.",
    )
    parser.add_argument("--version", action="version", version=f"codebrain {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="wire CodeBrain into this repository")
    p.add_argument("root", nargs="?", default=".")
    p.add_argument("--no-mcp", action="store_true", help="skip .mcp.json")
    p.add_argument("--no-hooks", action="store_true",
                   help="skip .claude/settings.json")
    p.add_argument("--no-claude-md", action="store_true", help="skip CLAUDE.md")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("build", help="run extractors and write the Brain")
    p.add_argument("root", nargs="?", default=".")
    p.add_argument("--out", default=None, help="output directory (default <root>/.brain)")
    p.add_argument("--only", action="append", metavar="PROVIDER",
                   help="run only this provider (repeatable)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("status", help="summarise a Brain")
    _brain_arg(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("validate", help="check a Brain for structural problems")
    _brain_arg(p)
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("diff", help="compare two Brains")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--check", action="store_true",
                   help="exit non-zero on substantive change (CI drift gate)")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("pack", help="compile a context pack for a task")
    p.add_argument("task", nargs="?", default=None)
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--root", default=".")
    p.add_argument("--stdin", action="store_true",
                   help="read the task from a hook payload on stdin")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true",
                   help="print nothing when no anchors match (hook default)")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("brief", help="session-start orientation")
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--budget", type=int, default=500)
    p.set_defaults(func=cmd_brief)

    p = sub.add_parser("guard", help="check a pending edit against L6 constraints")
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--root", default=".")
    p.add_argument("--path", default=None, help="check this path instead of reading stdin")
    p.add_argument("--deny-guarded", action="store_true",
                   help="ask for confirmation on CODEOWNERS-guarded paths")
    p.set_defaults(func=cmd_guard)

    p = sub.add_parser("touch", help="note that a file was edited")
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--root", default=".")
    p.add_argument("--path", default=None)
    p.set_defaults(func=cmd_touch)

    p = sub.add_parser("verify", help="execute this repo's claimed commands to "
                                      "settle them (dry run unless --yes)")
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--root", default=".")
    p.add_argument("--intent", action="append",
                   help="verify only this intent (test/build/lint; repeatable)")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--yes", action="store_true",
                   help="actually run the commands listed by the dry run")
    p.add_argument("--check", action="store_true",
                   help="exit non-zero if any claim was refuted")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("sync", help="rebuild if anything moved, preserving "
                                    "verified and asserted claims")
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--root", default=".")
    p.add_argument("--force", action="store_true", help="rebuild even if unchanged")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("drift", help="check whether the committed Brain still "
                                     "describes HEAD (CI gate)")
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--root", default=".")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--check", action="store_true",
                   help="exit non-zero when the Brain has drifted")
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("eval", help="measure pack retrieval against keyword search")
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--root", default=".")
    p.add_argument("--cases", type=int, default=40)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--memory", action="store_true",
                   help="measure the write-back effect: same Brain, with "
                        "and without a previous session in memory")
    p.add_argument("--rigorous", action="store_true",
                   help="build a Brain per commit from its parent, so the "
                        "Brain cannot have seen the answer (slow)")
    p.add_argument("--check", action="store_true",
                   help="exit non-zero if packs do not beat keyword search")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("learn", help="fold a finished session into memory (Stop hook)")
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--session", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--lesson", action="append", help="something learned (repeatable)")
    p.add_argument("--question", action="append", help="an unanswered question")
    p.add_argument("--outcome", choices=("success", "failure"), default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("remember", help="record one lesson")
    p.add_argument("text")
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--about", default=None, help="a file this is about")
    p.add_argument("--task", default=None)
    p.add_argument("--human", action="store_true",
                   help="record as a human assertion rather than an agent reading")
    p.set_defaults(func=cmd_remember)

    p = sub.add_parser("dispute", help="contest a claim in the Brain")
    p.add_argument("record", help="the record id being disputed")
    p.add_argument("--reason", required=True)
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--human", action="store_true",
                   help="overrule outright; agents may only dispute")
    p.set_defaults(func=cmd_dispute)

    p = sub.add_parser("serve", help="run the MCP server on stdio")
    p.add_argument("--brain", default=BRAIN_DIR)
    p.add_argument("--root", default=".")
    p.add_argument("--mcp", action="store_true", help="accepted for symmetry; implied")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("atlas", help="regenerate the human-readable Atlas")
    _brain_arg(p)
    p.add_argument("--out", default=None, help="output path, or - for stdout")
    p.set_defaults(func=cmd_atlas)

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
