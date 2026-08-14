"""L0 corpus census — what files exist and what this codebase is made of.

Pure filesystem, no git, no network, no language model. The cheapest possible
layer, and the one every later extractor anchors its file ids against.
"""

from __future__ import annotations

from typing import Iterable

from ..envelope import Envelope, Evidence, Method, Status
from ..model import REPO, Fact, Layer, Node, Record
from ..providers import BuildContext, Provider, register

#: Extension → language. Kept explicit rather than guessed: an unknown extension
#: should read as "unknown", not as a confident wrong answer.
LANGUAGES: dict[str, str] = {
    ".py": "Python", ".pyi": "Python",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".mts": "TypeScript", ".cts": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".php": "PHP",
    ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin", ".scala": "Scala", ".groovy": "Groovy",
    ".cs": "C#", ".fs": "F#", ".vb": "VB.NET",
    ".c": "C", ".h": "C", ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".hpp": "C++", ".hh": "C++",
    ".m": "Objective-C", ".mm": "Objective-C++", ".swift": "Swift",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell", ".ps1": "PowerShell",
    ".sql": "SQL", ".r": "R", ".jl": "Julia", ".ex": "Elixir", ".exs": "Elixir",
    ".erl": "Erlang", ".hs": "Haskell", ".lua": "Lua", ".dart": "Dart", ".pl": "Perl",
    ".md": "Markdown", ".rst": "reStructuredText", ".txt": "Text", ".adoc": "AsciiDoc",
    ".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML", ".ini": "INI",
    ".xml": "XML", ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".sass": "Sass", ".less": "Less", ".vue": "Vue", ".svelte": "Svelte",
    ".proto": "Protobuf", ".graphql": "GraphQL", ".gql": "GraphQL", ".tf": "Terraform",
    ".dockerfile": "Dockerfile", ".cmake": "CMake", ".gradle": "Gradle", ".bzl": "Starlark",
}

#: Files whose *name* identifies them regardless of extension.
NAMED: dict[str, str] = {
    "Dockerfile": "Dockerfile", "Makefile": "Make", "Rakefile": "Ruby",
    "Gemfile": "Ruby", "Procfile": "Config", "Jenkinsfile": "Groovy",
    "CMakeLists.txt": "CMake", "BUILD": "Starlark", "BUILD.bazel": "Starlark",
}

#: Languages that count as source when reporting what the repo *is*.
NON_SOURCE = frozenset({
    "Markdown", "reStructuredText", "Text", "AsciiDoc", "JSON", "YAML", "TOML",
    "INI", "XML", "CSS", "SCSS", "Sass", "Less", "Config",
})


def classify(name: str, suffix: str) -> str:
    if name in NAMED:
        return NAMED[name]
    return LANGUAGES.get(suffix.lower(), "unknown")


class CensusProvider(Provider):
    id = "census"
    layers = (Layer.L0,)
    description = "Filesystem census: files, sizes, language mix."
    order = 10  # first — later providers reference the file nodes it creates

    def extract(self, ctx: BuildContext) -> Iterable[Record]:
        env = lambda path, method=Method.EXTRACTED: Envelope.make(  # noqa: E731
            method, source=self.id, as_of=ctx.commit, ts=ctx.ts,
            evidence=(Evidence(path=path, commit=ctx.commit or None),),
            status=Status.FRESH if ctx.commit else Status.UNVERIFIED,
        )

        by_language: dict[str, int] = {}
        source_bytes: dict[str, int] = {}
        total_files = 0
        total_bytes = 0

        for path in ctx.iter_files():
            rel = ctx.rel(path)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            language = classify(path.name, path.suffix)
            total_files += 1
            total_bytes += size
            by_language[language] = by_language.get(language, 0) + 1
            if language not in NON_SOURCE and language != "unknown":
                source_bytes[language] = source_bytes.get(language, 0) + size

            yield Node(
                layer=Layer.L0, kind="file", key=rel, name=path.name, env=env(rel),
                attrs={"bytes": size, "language": language,
                       "text": ctx.readable(path), "ext": path.suffix.lower()},
            )

        repo_env = env(".", Method.EXTRACTED)
        yield Fact(layer=Layer.L0, subject=REPO, predicate="file_count",
                   value=total_files, env=repo_env)
        yield Fact(layer=Layer.L0, subject=REPO, predicate="total_bytes",
                   value=total_bytes, env=repo_env)
        yield Fact(layer=Layer.L0, subject=REPO, predicate="language_mix",
                   value=dict(sorted(by_language.items(), key=lambda kv: (-kv[1], kv[0]))),
                   env=repo_env)

        # The repo's primary language is a judgement call made from evidence, so
        # it is DERIVED, not EXTRACTED — the distinction is the whole point of
        # the envelope, and it must be applied to our own output first.
        if source_bytes:
            primary = max(source_bytes.items(), key=lambda kv: (kv[1], kv[0]))[0]
            yield Fact(
                layer=Layer.L0, subject=REPO, predicate="primary_language", value=primary,
                env=Envelope.make(Method.DERIVED, source=self.id, as_of=ctx.commit, ts=ctx.ts,
                                  evidence=(Evidence(path="."),),
                                  note="largest source language by bytes"),
            )

        # A weak guess at the repo name, from the directory it happens to sit in.
        # gitmeta knows better; the envelope is what lets it win.
        yield Fact(
            layer=Layer.L0, subject=REPO, predicate="repo_name", value=ctx.root.name,
            env=Envelope.make(Method.DERIVED, source=self.id, as_of=ctx.commit, ts=ctx.ts,
                              confidence=0.5, evidence=(Evidence(path="."),),
                              note="inferred from directory name"),
        )


register(CensusProvider())
