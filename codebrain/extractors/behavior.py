"""L2 behavior — what this code does when it runs.

Structure says a function exists. Behavior says it answers `POST /v1/charges`,
reads `STRIPE_KEY` out of the environment, writes to a database and calls an
external host. That is the layer an agent needs before it can reason about
blast radius beyond the call graph: changing a function is local, changing the
route it serves is not.

Deterministic throughout. Routes, entrypoints and jobs come from decorators and
call syntax read off the AST, not from a model's opinion about what a function
looks like. Where a framework expresses routing in a way this cannot see, the
gap is recorded rather than papered over.
"""

from __future__ import annotations

import ast
import re
from typing import Iterable

from ..envelope import Envelope, Evidence, Method, Status
from ..model import REPO, Edge, Fact, Layer, Node, Record
from ..providers import BuildContext, Provider, register

HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head",
                          "options", "trace"})

#: Decorator attributes that mount a handler on a path.
ROUTE_ATTRS = HTTP_METHODS | {"route", "websocket", "api_route"}

#: Decorator names that mean "this runs on a schedule or a queue".
JOB_MARKERS = frozenset({"task", "shared_task", "periodic_task", "scheduled",
                         "cron", "job", "on_event", "subscribe", "listener"})

#: Modules whose import implies the process talks to a database.
DB_MODULES = frozenset({"sqlalchemy", "psycopg", "psycopg2", "asyncpg", "sqlite3",
                        "pymongo", "redis", "mysql", "pymysql", "aiomysql",
                        "motor", "peewee", "tortoise", "databases", "duckdb"})

#: Modules whose import implies the process makes outbound network calls.
NET_MODULES = frozenset({"requests", "httpx", "aiohttp", "urllib", "urllib3",
                         "http", "grpc", "websockets", "boto3", "botocore"})

#: Express/Fastify style route calls in JS/TS: app.get("/x", handler)
TS_ROUTE = re.compile(
    r"\b(?:app|router|server|api)\s*\.\s*(get|post|put|patch|delete|all|use)\s*\(\s*"
    r"[\"'`]([^\"'`]+)[\"'`]"
)


def literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def decorator_parts(node: ast.AST) -> tuple[str | None, str | None, list[ast.AST], list[ast.keyword]]:
    """(receiver, attribute, args, keywords) for a decorator, best effort."""
    call_args: list[ast.AST] = []
    keywords: list[ast.keyword] = []
    target = node
    if isinstance(node, ast.Call):
        call_args = list(node.args)
        keywords = list(node.keywords)
        target = node.func

    if isinstance(target, ast.Attribute):
        receiver = target.value.id if isinstance(target.value, ast.Name) else None
        return receiver, target.attr, call_args, keywords
    if isinstance(target, ast.Name):
        return None, target.id, call_args, keywords
    return None, None, call_args, keywords


def methods_from_keywords(keywords: list[ast.keyword]) -> list[str]:
    for keyword in keywords:
        if keyword.arg != "methods":
            continue
        if isinstance(keyword.value, (ast.List, ast.Tuple)):
            found = [literal(el) for el in keyword.value.elts]
            return [m.upper() for m in found if m]
    return []


class BehaviorProvider(Provider):
    id = "behavior"
    layers = (Layer.L2,)
    description = "Entrypoints, HTTP routes, jobs, data access, config reads."
    order = 60  # after structure, so handlers resolve to real symbol ids

    def applies(self, ctx: BuildContext) -> bool:
        return any(p.suffix in (".py", ".ts", ".tsx", ".js", ".jsx", ".mjs")
                   for p in ctx.iter_files())

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

        env_vars: dict[str, list[str]] = {}
        db_users: set[str] = set()
        net_users: set[str] = set()
        route_count = 0
        entrypoints = 0

        for path in sorted(ctx.iter_files()):
            if not ctx.readable(path):
                continue
            rel = ctx.rel(path)

            if path.suffix in (".py", ".pyi"):
                produced = list(self._python(ctx, path, rel, env, env_vars,
                                             db_users, net_users))
                for record in produced:
                    if isinstance(record, Node) and record.kind == "route":
                        route_count += 1
                    if isinstance(record, Node) and record.kind == "entrypoint":
                        entrypoints += 1
                yield from produced
            elif path.suffix in (".ts", ".tsx", ".js", ".jsx", ".mjs"):
                for record in self._typescript(ctx, path, rel, env):
                    if isinstance(record, Node) and record.kind == "route":
                        route_count += 1
                    yield record

        for symbol_id, names in sorted(env_vars.items()):
            yield Fact(layer=Layer.L2, subject=symbol_id, predicate="reads_env",
                       value=sorted(set(names)),
                       env=env(Method.EXTRACTED, symbol_id.split(":")[-1].split("#")[0]))

        all_env = sorted({name for names in env_vars.values() for name in names})
        if all_env:
            yield Fact(layer=Layer.L2, subject=REPO, predicate="environment_variables",
                       value=all_env, env=env(Method.EXTRACTED, "."))
        if db_users:
            yield Fact(layer=Layer.L2, subject=REPO, predicate="data_stores",
                       value=sorted(db_users), env=env(Method.DERIVED, ".",
                                                       "inferred from imports"))
        if net_users:
            yield Fact(layer=Layer.L2, subject=REPO, predicate="outbound_network",
                       value=sorted(net_users), env=env(Method.DERIVED, ".",
                                                        "inferred from imports"))

        yield Fact(layer=Layer.L2, subject=REPO, predicate="runtime_summary",
                   value={"routes": route_count, "entrypoints": entrypoints,
                          "env_vars": len(all_env)},
                   env=env(Method.EXTRACTED, "."))

        # Frameworks that route through configuration rather than decorators —
        # Django urlpatterns, generated clients, dynamic mounts — are invisible
        # here. Say so, rather than letting a route count of zero read as
        # "this service has no HTTP surface".
        yield Fact(
            layer=Layer.L2, subject=REPO, predicate="behavior_coverage_gap",
            value={"detects": ["decorator routes", "express-style routes",
                               "__main__ entrypoints", "decorator jobs"],
                   "misses": ["Django urlpatterns", "dynamically mounted routes",
                              "config-driven schedulers", "message consumers"],
                   "impact": "a route count of zero is not proof of no HTTP surface"},
            env=env(Method.EXTRACTED, "."),
        )

    # -- python ------------------------------------------------------------

    def _python(self, ctx, path, rel: str, env, env_vars, db_users,
                net_users) -> Iterable[Record]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError, OSError, ValueError):
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in DB_MODULES:
                        db_users.add(root)
                    if root in NET_MODULES:
                        net_users.add(root)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in DB_MODULES:
                    db_users.add(root)
                if root in NET_MODULES:
                    net_users.add(root)

        # `if __name__ == "__main__":` — the process starts here.
        for node in tree.body:
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and any(literal(c) == "__main__" for c in test.comparators)):
                yield Node(layer=Layer.L2, kind="entrypoint", key=rel, name=rel,
                           env=env(Method.EXTRACTED, rel, node.lineno),
                           attrs={"kind": "__main__", "path": rel})

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            symbol_id = f"{Layer.L1}:symbol:{rel}#{node.name}"

            names = self._env_reads(node)
            if names:
                env_vars.setdefault(symbol_id, []).extend(names)

            for decorator in node.decorator_list:
                receiver, attribute, args, keywords = decorator_parts(decorator)
                if attribute is None:
                    continue

                if attribute in ROUTE_ATTRS and args:
                    route_path = literal(args[0])
                    if not route_path:
                        continue
                    verbs = (methods_from_keywords(keywords)
                             or ([attribute.upper()] if attribute in HTTP_METHODS
                                 else ["GET"]))
                    for verb in verbs:
                        key = f"{verb} {route_path}"
                        yield Node(layer=Layer.L2, kind="route", key=key, name=key,
                                   env=env(Method.EXTRACTED, rel, node.lineno),
                                   attrs={"method": verb, "path": route_path,
                                          "handler": node.name, "module": rel,
                                          "framework": receiver or "?"})
                        yield Edge(layer=Layer.L2, kind="handled_by",
                                   src=f"{Layer.L2}:route:{key}", dst=symbol_id,
                                   env=env(Method.EXTRACTED, rel, node.lineno))

                elif attribute in JOB_MARKERS:
                    key = f"{rel}#{node.name}"
                    yield Node(layer=Layer.L2, kind="job", key=key, name=node.name,
                               env=env(Method.EXTRACTED, rel, node.lineno),
                               attrs={"trigger": attribute, "module": rel,
                                      "runner": receiver or "?"})
                    yield Edge(layer=Layer.L2, kind="handled_by",
                               src=f"{Layer.L2}:job:{key}", dst=symbol_id,
                               env=env(Method.EXTRACTED, rel, node.lineno))

    @staticmethod
    def _env_reads(node: ast.AST) -> list[str]:
        """Environment variables a function reads, by name."""
        found: list[str] = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                if (isinstance(func, ast.Attribute) and func.attr in ("getenv", "get")
                        and sub.args):
                    receiver = func.value
                    is_env = (
                        (isinstance(receiver, ast.Name) and receiver.id == "os")
                        or (isinstance(receiver, ast.Attribute)
                            and receiver.attr == "environ")
                    )
                    if is_env:
                        name = literal(sub.args[0])
                        if name:
                            found.append(name)
            elif isinstance(sub, ast.Subscript):
                value = sub.value
                if isinstance(value, ast.Attribute) and value.attr == "environ":
                    name = literal(sub.slice)
                    if name:
                        found.append(name)
        return found

    # -- typescript / javascript -------------------------------------------

    def _typescript(self, ctx, path, rel: str, env) -> Iterable[Record]:
        from .structure_ts import mask

        try:
            source, literals = mask(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            return

        # Route paths are string literals, and masking replaced them with
        # placeholders — resolve back through the table.
        for lineno, line in enumerate(source.splitlines(), 1):
            for match in TS_ROUTE.finditer(line):
                verb, placeholder = match.group(1), match.group(2)
                route_path = placeholder
                index = re.fullmatch(r"\x01(\d+)\x01", placeholder)
                if index:
                    slot = int(index.group(1))
                    if slot >= len(literals):
                        continue
                    route_path = literals[slot]
                if not route_path.startswith("/"):
                    continue
                key = f"{verb.upper()} {route_path}"
                yield Node(layer=Layer.L2, kind="route", key=key, name=key,
                           env=env(Method.DERIVED, rel, lineno,
                                   note="scanned, not parsed", confidence=0.8),
                           attrs={"method": verb.upper(), "path": route_path,
                                  "module": rel, "framework": "express-style"})


register(BehaviorProvider())
