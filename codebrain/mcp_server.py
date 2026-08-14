"""MCP server — the Brain's agent-facing surface.

Newline-delimited JSON-RPC 2.0 over stdio, implemented directly. The protocol
subset an agent host actually uses is small (initialize, tools/list,
tools/call), and hand-rolling it keeps the deterministic core dependency-free —
which is the difference between `pip install codebrain` working on a locked-down
build machine and not.

The tools mirror the pack facets, because the facets are the questions. `pack`
is the one to reach for by default; the others exist for the mid-task moment
when an agent already has a pack and needs one more thing.

Every handler catches its own errors and returns them as text. A tool that
raises across the transport kills the session's connection to the Brain, which
is exactly the fail-open violation the whole integration is built to avoid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .envelope import Method, Status
from .model import REPO, Layer
from .pack import DEFAULT_BUDGET, brief, compile_pack, provenance_tag, tokenize
from .store import BrainNotFound, apply_touched, load, read_touched

PROTOCOL_VERSION = "2025-06-18"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "brain_pack",
        "description": (
            "THE DEFAULT FIRST CALL for any coding task. Compiles a task-scoped, "
            "cited context pack: the symbols the work touches, what depends on them, "
            "the contracts that must not change shape, how this pattern was done here "
            "before, the constraints in play, the verified build/test commands, and "
            "the known unknowns. Cheaper and more accurate than searching."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string",
                         "description": "What you are about to do, in a sentence."},
                "budget": {"type": "integer",
                           "description": f"Token budget (default {DEFAULT_BUDGET})."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "brain_locate",
        "description": "Find the symbols, modules and files that match a description. "
                       "Use when you need a location, not a whole pack.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"},
                           "limit": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "brain_explain",
        "description": "Everything the Brain knows about one symbol, module or file, "
                       "with the provenance of each claim.",
        "inputSchema": {
            "type": "object",
            "properties": {"target": {"type": "string",
                                      "description": "A record id, path, or symbol name."}},
            "required": ["target"],
        },
    },
    {
        "name": "brain_impact",
        "description": "What breaks if you change this: callers and importers, "
                       "transitively. Call before editing anything shared.",
        "inputSchema": {
            "type": "object",
            "properties": {"target": {"type": "string"},
                           "depth": {"type": "integer"}},
            "required": ["target"],
        },
    },
    {
        "name": "brain_runbook",
        "description": "How to build, test, lint and run this repository, with "
                       "whether each command has actually been executed or is "
                       "only inferred.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "brain_constraints",
        "description": "What must not break for a given path: required reviewers, "
                       "danger zones, single-author risk.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]


class Server:
    def __init__(self, brain_dir: str, root: str = ".") -> None:
        self.brain_dir = brain_dir
        self.root = Path(root)
        self._brain = None
        self._stamp: tuple[float, int] | None = None

    # -- brain access ------------------------------------------------------

    def brain(self):
        """Reload when the Brain on disk changes, so a long-lived agent session
        does not keep answering from a Brain that was rebuilt underneath it."""
        manifest = Path(self.brain_dir) / "manifest.json"
        try:
            stat = manifest.stat()
            stamp = (stat.st_mtime, stat.st_size)
        except OSError:
            return None
        if self._brain is None or stamp != self._stamp:
            try:
                self._brain = load(self.brain_dir)
                self._stamp = stamp
            except (BrainNotFound, ValueError, OSError):
                return None
        try:
            apply_touched(self._brain, read_touched(self.brain_dir))
        except Exception:  # noqa: BLE001
            pass
        return self._brain

    # -- tools -------------------------------------------------------------

    def tool_brain_pack(self, args: dict[str, Any], brain) -> str:
        task = str(args.get("task", "")).strip()
        if not task:
            return "No task given."
        budget = int(args.get("budget") or DEFAULT_BUDGET)
        return compile_pack(brain, task, budget=budget).render()

    def tool_brain_locate(self, args: dict[str, Any], brain) -> str:
        from .pack import Compiler

        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit") or 12)
        if not query:
            return "No query given."
        hits = Compiler(brain).score_anchors(query, limit=limit)
        if not hits:
            return f"Nothing in the Brain matches {query!r}."
        lines = [f"{len(hits)} match(es) for {query!r}:"]
        for node, score in hits:
            evidence = node.env.evidence[0] if node.env.evidence else None
            where = (f"{evidence.path}:{evidence.start_line}"
                     if evidence and evidence.start_line else node.key)
            kind = node.attrs.get("symbol_kind") or node.kind
            lines.append(f"  {where}  {node.name} ({kind})  "
                         f"score {score:.1f}  [{provenance_tag(node.env)}]")
        return "\n".join(lines)

    def _resolve(self, brain, target: str):
        direct = brain.get(target)
        if direct is not None:
            return direct
        for node in brain.nodes.values():
            if node.key == target or node.key.endswith(f"#{target}"):
                return node
        terms = tokenize(target)
        for node in brain.nodes.values():
            if node.name and node.name.lower() in terms:
                return node
        return None

    def tool_brain_explain(self, args: dict[str, Any], brain) -> str:
        target = str(args.get("target", "")).strip()
        node = self._resolve(brain, target)
        if node is None:
            return f"The Brain has no record for {target!r}."

        lines = [f"{node.id}", f"  kind: {node.attrs.get('symbol_kind') or node.kind}"]
        for key, value in sorted(node.attrs.items()):
            if key != "symbol_kind":
                lines.append(f"  {key}: {value}")
        lines.append(f"  provenance: {provenance_tag(node.env)} "
                     f"via {node.env.source} at {(node.env.as_of or '?')[:8]}")
        for evidence in node.env.evidence:
            lines.append(f"  evidence: {evidence}")

        related = [f for f in brain.facts.values() if f.subject == node.id]
        related += [f for f in brain.facts.values()
                    if f.subject == f"{Layer.L0}:file:{node.attrs.get('module', '')}"]
        if related:
            lines.append("  facts:")
            for fact in sorted(related, key=lambda f: f.predicate)[:15]:
                lines.append(f"    {fact.predicate} = {fact.value}  "
                             f"[{provenance_tag(fact.env)}]")
        return "\n".join(lines)

    def tool_brain_impact(self, args: dict[str, Any], brain) -> str:
        from .pack import Compiler

        target = str(args.get("target", "")).strip()
        depth = max(1, min(4, int(args.get("depth") or 2)))
        node = self._resolve(brain, target)
        if node is None:
            return f"The Brain has no record for {target!r}."

        items = Compiler(brain).blast_radius([(node, 1.0)], depth=depth)
        if not items:
            return (f"Nothing in the Brain depends on {node.id}. Note this may mean "
                    "no dependents, or that this language's call graph is incomplete "
                    "— check brain_pack's UNKNOWNS.")
        lines = [f"{len(items)} dependent(s) of {node.id} within {depth} hop(s):"]
        lines.extend(f"  {item.text}" for item in items[:60])
        if len(items) > 60:
            lines.append(f"  … {len(items) - 60} more")
        return "\n".join(lines)

    def tool_brain_runbook(self, args: dict[str, Any], brain) -> str:
        lines = []
        for intent in ("test", "build", "lint", "run"):
            found = brain.fact(REPO, f"{intent}_command", Layer.L5)
            if found is None:
                continue
            if found.env.status is Status.REFUTED:
                caveat = "  — CodeBrain ran this and it FAILED; do not rely on it"
            elif found.env.method is Method.OBSERVED:
                caveat = ""
            else:
                caveat = "  — inferred, never executed by CodeBrain"
            lines.append(f"{intent:<6} {found.value}  "
                         f"[{provenance_tag(found.env)}]{caveat}")
        if not lines:
            return "The Brain found no build, test or run commands for this repository."
        return "\n".join(lines)

    def tool_brain_constraints(self, args: dict[str, Any], brain) -> str:
        path = str(args.get("path", "")).replace("\\", "/").strip()
        if not path:
            return "No path given."
        file_id = f"{Layer.L0}:file:{path}"
        lines = []
        for predicate in ("requires_review", "danger_zone", "bus_factor_risk"):
            found = brain.fact(file_id, predicate, Layer.L6)
            if found:
                lines.append(f"  {predicate}: {found.value}  "
                             f"[{provenance_tag(found.env)}]")
        if not lines:
            return (f"No constraints recorded for {path}. That is not a guarantee — "
                    "L6 currently covers CODEOWNERS, churn hotspots and single-author "
                    "files only.")
        return f"Constraints on {path}:\n" + "\n".join(lines)

    # -- dispatch ----------------------------------------------------------

    def handlers(self) -> dict[str, Callable]:
        return {
            "brain_pack": self.tool_brain_pack,
            "brain_locate": self.tool_brain_locate,
            "brain_explain": self.tool_brain_explain,
            "brain_impact": self.tool_brain_impact,
            "brain_runbook": self.tool_brain_runbook,
            "brain_constraints": self.tool_brain_constraints,
        }

    def call_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        handler = self.handlers().get(name)
        if handler is None:
            return f"Unknown tool {name!r}.", True
        brain = self.brain()
        if brain is None:
            return ("No Brain is available at "
                    f"{self.brain_dir}. Run `codebrain build` first."), True
        try:
            return handler(args, brain), False
        except Exception as exc:  # noqa: BLE001 — never break the transport
            return f"{type(exc).__name__}: {exc}", True

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        msg_id = message.get("id")

        if method == "initialize":
            return self._ok(msg_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "codebrain", "version": __version__},
                "instructions": (
                    "This repository has a Brain. Call brain_pack with the task "
                    "before searching the codebase — it returns a cited, "
                    "budget-bounded context pack including constraints and the "
                    "verified test command. Claims are tagged EXTRACTED, DERIVED, "
                    "INFERRED, OBSERVED or ASSERTED; prefer EXTRACTED and OBSERVED "
                    "for risky changes, and treat the UNKNOWNS section as the edge "
                    "of what is known rather than as nothing to worry about."
                ),
            })

        if method in ("notifications/initialized", "initialized"):
            return None  # notification: no reply

        if method == "ping":
            return self._ok(msg_id, {})

        if method == "tools/list":
            return self._ok(msg_id, {"tools": TOOLS})

        if method == "tools/call":
            params = message.get("params") or {}
            text, is_error = self.call_tool(params.get("name", ""),
                                            params.get("arguments") or {})
            return self._ok(msg_id, {"content": [{"type": "text", "text": text}],
                                     "isError": is_error})

        if msg_id is None:
            return None  # unknown notification
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}

    @staticmethod
    def _ok(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def serve(brain_dir: str, root: str = ".", stdin=None, stdout=None) -> int:
    server = Server(brain_dir, root)
    source = stdin or sys.stdin
    sink = stdout or sys.stdout

    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            sink.write(json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "Parse error"}}) + "\n")
            sink.flush()
            continue

        response = server.handle(message)
        if response is not None:
            sink.write(json.dumps(response) + "\n")
            sink.flush()
    return 0
