"""L5 operations — how to build, test, run and ship this repository.

The layer an agent needs most often and finds least reliably. "Run the tests"
is a different command in every repo, and guessing between `pytest`, `make test`
and `npm test` is exactly the kind of avoidable failure that burns a session.

Note what is EXTRACTED and what is DERIVED here. That a `test` script exists in
package.json is extracted — it is written down. That it is *the* test command
for this repository is a judgement across competing candidates, so it is
DERIVED, and stays DERIVED until P3 actually runs it and promotes it to
OBSERVED. This is the clearest example in the codebase of why the envelope
exists.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Iterable

from ..envelope import Envelope, Evidence, Method, Status
from ..model import REPO, Fact, Layer, Node, Record
from ..providers import BuildContext, Provider, register

#: Which script names count as which intent, best candidate first. Order is the
#: ranking used when several sources offer a command for the same intent.
INTENTS: dict[str, tuple[str, ...]] = {
    "test": ("test", "tests", "check", "pytest", "unit", "test:unit"),
    "build": ("build", "compile", "dist", "package"),
    "lint": ("lint", "ruff", "eslint", "format:check", "typecheck"),
    "run": ("start", "dev", "serve", "run", "up"),
}

#: Source precedence when the same intent is claimed twice. A Makefile target
#: usually wraps the underlying tool, and is what a human would actually type.
SOURCE_RANK = {"makefile": 3, "package.json": 2, "pyproject.toml": 2, "workflow": 1}

MAKE_TARGET = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.\-/]*)\s*:(?!=)")
WORKFLOW_RUN = re.compile(r"^\s*-?\s*run:\s*(.+?)\s*$")
WORKFLOW_USES = re.compile(r"^\s*-?\s*uses:\s*(\S+)")


class OperationsProvider(Provider):
    id = "operations"
    layers = (Layer.L5,)
    description = "Build, test and run commands, CI pipelines, containers, ownership."
    order = 50

    def applies(self, ctx: BuildContext) -> bool:
        return True  # every repo has *some* operational surface, even if only files

    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        def env(method: Method, path: str, line: int | None = None,
                note: str = "", confidence: float | None = None) -> Envelope:
            return Envelope.make(
                method, source=self.id, as_of=ctx.commit, ts=ctx.ts,
                status=Status.FRESH if ctx.commit else Status.UNVERIFIED,
                confidence=confidence,
                evidence=(Evidence(path=path, start_line=line,
                                   commit=ctx.commit or None),),
                note=note,
            )

        # (intent, command, source, path, line)
        candidates: list[tuple[str, str, str, str, int | None]] = []

        yield from self._package_json(ctx, env, candidates)
        yield from self._pyproject(ctx, env, candidates)
        yield from self._makefile(ctx, env, candidates)
        yield from self._workflows(ctx, env, candidates)
        yield from self._containers(ctx, env)
        yield from self._codeowners(ctx, env)

        # -- pick the best command per intent -----------------------------
        for intent in INTENTS:
            options = [c for c in candidates if c[0] == intent]
            if not options:
                continue
            options.sort(key=lambda c: (-SOURCE_RANK.get(c[2], 0), c[3], c[1]))
            _, command, source, path, line = options[0]
            yield Fact(
                layer=Layer.L5, subject=REPO, predicate=f"{intent}_command",
                value=command,
                attrs={"source": source,
                       "alternatives": [{"command": o[1], "source": o[2]}
                                        for o in options[1:5]]},
                # A choice between competing candidates, not a reading. P3 runs
                # it and promotes to OBSERVED, or refutes it.
                env=env(Method.DERIVED, path, line,
                        note=f"chosen from {len(options)} candidate(s); unverified until executed"),
            )

    # -- node ---------------------------------------------------------------

    def _package_json(self, ctx, env, candidates) -> Iterable[Record]:
        path = ctx.root / "package.json"
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return

        scripts = data.get("scripts")
        if isinstance(scripts, dict):
            for name, command in scripts.items():
                if not isinstance(command, str):
                    continue
                yield Node(
                    layer=Layer.L5, kind="command", key=f"package.json:{name}", name=name,
                    env=env(Method.EXTRACTED, "package.json"),
                    attrs={"command": command, "source": "package.json", "runner": "npm"},
                )
                for intent, names in INTENTS.items():
                    if name in names:
                        candidates.append((intent, f"npm run {name}", "package.json",
                                           "package.json", None))

        deps = {}
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            value = data.get(key)
            if isinstance(value, dict):
                deps[key] = len(value)
        if deps:
            yield Fact(layer=Layer.L5, subject=REPO, predicate="node_package",
                       value={"name": data.get("name"), "version": data.get("version"),
                              "dependency_counts": deps,
                              "package_manager": data.get("packageManager")},
                       env=env(Method.EXTRACTED, "package.json"))

    # -- python -------------------------------------------------------------

    def _pyproject(self, ctx, env, candidates) -> Iterable[Record]:
        path = ctx.root / "pyproject.toml"
        if not path.is_file():
            return
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError):
            return

        project = data.get("project", {})
        if project:
            yield Fact(
                layer=Layer.L5, subject=REPO, predicate="python_package",
                value={"name": project.get("name"), "version": project.get("version"),
                       "requires_python": project.get("requires-python"),
                       "dependencies": len(project.get("dependencies") or []),
                       "optional_groups": sorted(
                           (project.get("optional-dependencies") or {}).keys())},
                env=env(Method.EXTRACTED, "pyproject.toml"),
            )

        for name, target in (project.get("scripts") or {}).items():
            yield Node(layer=Layer.L5, kind="command", key=f"pyproject:{name}", name=name,
                       env=env(Method.EXTRACTED, "pyproject.toml"),
                       attrs={"command": name, "entrypoint": target,
                              "source": "pyproject.toml", "runner": "console_script"})

        # A test runner declared as a dependency is weak evidence of how tests
        # run — enough to offer, not enough to assert.
        blob = json.dumps(data)
        for runner, command in (("pytest", "pytest"), ("tox", "tox")):
            if f'"{runner}' in blob:
                candidates.append(("test", command, "pyproject.toml", "pyproject.toml", None))
                break
        else:
            if (ctx.root / "tests").is_dir():
                candidates.append(("test", "python -m unittest discover -s tests -t .",
                                   "pyproject.toml", "pyproject.toml", None))

    # -- make ---------------------------------------------------------------

    def _makefile(self, ctx, env, candidates) -> Iterable[Record]:
        for filename in ("Makefile", "makefile", "GNUmakefile"):
            path = ctx.root / filename
            if path.is_file():
                break
        else:
            return

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return

        for lineno, line in enumerate(lines, 1):
            if line.startswith("\t") or line.startswith("#"):
                continue
            match = MAKE_TARGET.match(line)
            if not match:
                continue
            target = match.group(1)
            if target.startswith(".") or "$" in target:
                continue
            recipe = []
            for follow in lines[lineno:]:
                if not follow.startswith("\t"):
                    break
                recipe.append(follow.strip())

            yield Node(
                layer=Layer.L5, kind="command", key=f"make:{target}", name=target,
                env=env(Method.EXTRACTED, filename, lineno),
                attrs={"command": f"make {target}", "source": "makefile",
                       "recipe": recipe[:10], "runner": "make"},
            )
            for intent, names in INTENTS.items():
                if target in names:
                    candidates.append((intent, f"make {target}", "makefile", filename, lineno))

    # -- CI -----------------------------------------------------------------

    def _workflows(self, ctx, env, candidates) -> Iterable[Record]:
        workflows = ctx.root / ".github" / "workflows"
        if not workflows.is_dir():
            return

        for path in sorted(workflows.iterdir()):
            if path.suffix.lower() not in (".yml", ".yaml") or not path.is_file():
                continue
            rel = ctx.rel(path)
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            steps: list[str] = []
            actions: list[str] = []
            for lineno, line in enumerate(lines, 1):
                run = WORKFLOW_RUN.match(line)
                if run:
                    command = run.group(1).strip().strip("'\"")
                    if command and command not in ("|", ">", "|-", ">-"):
                        steps.append(command)
                        for intent, names in INTENTS.items():
                            head = command.split()[0] if command.split() else ""
                            if any(n in command.split() for n in names) or head in names:
                                candidates.append((intent, command, "workflow", rel, lineno))
                    continue
                uses = WORKFLOW_USES.match(line)
                if uses:
                    actions.append(uses.group(1))

            yield Node(
                layer=Layer.L5, kind="pipeline", key=rel, name=path.stem,
                # Line-scanned, not YAML-parsed: no third-party dependency is
                # allowed in the deterministic core, and a wrong YAML parser is
                # worse than an honest partial read.
                env=env(Method.DERIVED, rel, note="line-scanned, not YAML-parsed",
                        confidence=0.75),
                attrs={"steps": steps[:40], "actions": sorted(set(actions))[:20],
                       "step_count": len(steps)},
            )

    # -- containers ---------------------------------------------------------

    def _containers(self, ctx, env) -> Iterable[Record]:
        for path in sorted(ctx.iter_files()):
            if path.name != "Dockerfile" and not path.name.startswith("Dockerfile."):
                continue
            rel = ctx.rel(path)
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            base, ports, entry = None, [], None
            for line in lines:
                stripped = line.strip()
                upper = stripped.upper()
                if upper.startswith("FROM ") and base is None:
                    base = stripped[5:].strip()
                elif upper.startswith("EXPOSE "):
                    ports.extend(stripped[7:].split())
                elif upper.startswith(("ENTRYPOINT ", "CMD ")):
                    entry = stripped

            yield Node(layer=Layer.L5, kind="container", key=rel, name=path.name,
                       env=env(Method.EXTRACTED, rel),
                       attrs={"base_image": base, "exposed_ports": ports, "entrypoint": entry})

    # -- ownership ----------------------------------------------------------

    def _codeowners(self, ctx, env) -> Iterable[Record]:
        for candidate in ("CODEOWNERS", ".github/CODEOWNERS", "docs/CODEOWNERS"):
            path = ctx.root / candidate
            if path.is_file():
                break
        else:
            return

        rules = []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return

        for lineno, line in enumerate(lines, 1):
            stripped = line.split("#", 1)[0].strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            rules.append({"pattern": parts[0], "owners": parts[1:], "line": lineno})

        if rules:
            # Ownership is operational fact here; L6 turns it into a constraint
            # ("this path needs @risk-eng before an agent may touch it").
            yield Fact(layer=Layer.L5, subject=REPO, predicate="codeowners",
                       value=rules[:200], env=env(Method.EXTRACTED, ctx.rel(path)))


register(OperationsProvider())
