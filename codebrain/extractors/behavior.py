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

#: Next.js App Router: a file literally named route.ts/tsx/js/jsx under an
#: `app` directory IS the route — the URL comes from the directory structure,
#: not from any argument. `export async function GET(...)` is the convention;
#: an arrow-function export is not used anywhere this was checked against, so
#: it is left as a stated gap rather than guessed at.
APP_ROUTE_FILENAMES = frozenset({"route.ts", "route.tsx", "route.js", "route.jsx"})
APP_ROUTE_EXPORT = re.compile(
    r"^export\s+(?:async\s+)?function\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b",
    re.MULTILINE,
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


def app_router_path(rel: str) -> str | None:
    """The URL a Next.js App Router file answers, from its own location.

    `src/app/api/users/[id]/route.ts` -> `/api/users/[id]`. Route-group
    folders (`(admin)`) are Next.js's own convention for organising files
    without affecting the URL, so they are dropped rather than copied in
    verbatim — keeping them would silently emit a path nothing ever answers.
    Dynamic-segment brackets (`[id]`, `[...slug]`) are kept exactly as
    written: that is the literal, unambiguous match token, and inventing a
    different convention (`:id`) would be a translation this module has no
    business making.
    """
    parts = rel.split("/")
    try:
        start = parts.index("app") + 1
    except ValueError:
        return None
    segments = [p for p in parts[start:-1] if not (p.startswith("(") and p.endswith(")"))]
    return "/" + "/".join(segments) if segments else "/"


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
                               "Next.js App Router route.ts handlers",
                               "Django urlpatterns (path/re_path/url)",
                               "__main__ entrypoints", "decorator jobs"],
                   "misses": ["dynamically mounted routes",
                              "config-driven schedulers", "message consumers",
                              "Next.js Pages Router (pages/api/*)",
                              "re-exported or arrow-function route handlers",
                              "Django include() sub-URLconfs are not traversed",
                              "Django route HTTP methods (dispatch is inside the "
                              "view, not the URL config — recorded as ANY)"],
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

        yield from self._django_urls(tree, rel, env)

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

    def _django_urls(self, tree: ast.Module, rel: str, env) -> Iterable[Record]:
        """Django's URL config: a module-level `urlpatterns = [...]` list of
        `path()`/`re_path()`/`url()` calls. Detected by the variable name, not
        the filename — Django itself treats any module with that name as a
        URLconf, wherever it lives, the same principle already used for Next.js
        App Router's filename convention: detect however the framework itself
        determines the mapping.

        Two things this deliberately does not do. It never guesses the HTTP
        method: Django dispatches by the *view's* own method handling, not by
        anything written in urls.py, so the verb is recorded as ANY rather than
        invented. And it never emits a handled_by edge to the view: the view
        function typically lives in a different file (`views.py`) reached
        through an import this pass has not resolved, and a wrong cross-file
        edge is worse than no edge.
        """
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "urlpatterns"
                      for t in node.targets):
                continue
            if not isinstance(node.value, ast.List):
                continue

            for element in node.value.elts:
                if not isinstance(element, ast.Call) or not element.args:
                    continue
                func = element.func
                func_name = (func.id if isinstance(func, ast.Name)
                            else func.attr if isinstance(func, ast.Attribute) else None)
                if func_name not in ("path", "re_path", "url"):
                    continue

                route_path = literal(element.args[0])
                if not route_path:
                    continue
                handler = (self._django_view_name(element.args[1])
                          if len(element.args) > 1 else None)
                if handler is None:
                    continue  # an include() mount or something unnamed — not guessed at

                attrs = {"method": "ANY", "path": route_path, "handler": handler,
                         "module": rel, "framework": "django"}
                for keyword in element.keywords:
                    if keyword.arg == "name":
                        name = literal(keyword.value)
                        if name:
                            attrs["django_name"] = name

                key = f"ANY {route_path}"
                yield Node(
                    layer=Layer.L2, kind="route", key=key, name=key,
                    env=env(Method.EXTRACTED, rel, element.lineno,
                            note="HTTP method is not determinable from urlpatterns "
                                 "alone — Django dispatches by the view's own method "
                                 "handling"),
                    attrs=attrs,
                )

    @staticmethod
    def _django_view_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"  # views.my_view, kept module-qualified
        if isinstance(node, ast.Call):
            # The class name is the meaningful handler, however it was
            # reached: UserListView.as_view() (bare) and
            # views.UserListView.as_view() (module-qualified) both resolve to
            # "UserListView". Anything else callable here — notably
            # include() — is deliberately left unnamed.
            target = node.func
            if isinstance(target, ast.Attribute) and target.attr == "as_view":
                inner = target.value
                if isinstance(inner, ast.Name):
                    return inner.id
                if isinstance(inner, ast.Attribute):
                    return inner.attr
        return None

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

    def _app_router(self, source: str, rel: str, env) -> Iterable[Record]:
        """A Next.js App Router route.ts file. The exported method names are
        real HTTP handlers; the URL comes from where the file lives, not from
        anything in it — see app_router_path."""
        url_path = app_router_path(rel)
        if url_path is None:
            return
        for match in APP_ROUTE_EXPORT.finditer(source):
            verb = match.group(1)
            lineno = source.count("\n", 0, match.start()) + 1
            key = f"{verb} {url_path}"
            yield Node(layer=Layer.L2, kind="route", key=key, name=key,
                       env=env(Method.EXTRACTED, rel, lineno),
                       attrs={"method": verb, "path": url_path, "module": rel,
                              "framework": "next-app-router"})

    def _typescript(self, ctx, path, rel: str, env) -> Iterable[Record]:
        from .structure_ts import mask

        try:
            source, literals = mask(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            return

        if path.name in APP_ROUTE_FILENAMES:
            yield from self._app_router(source, rel, env)

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
