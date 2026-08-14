# CodeBrain

**Repositories hold code. They don't hold understanding.**

Every AI coding agent that touches a repo starts amnesiac: it greps, opens a few
files, builds a fragile mental model in its context window, acts on it, and
throws it away. The next agent pays the full cost again and reaches a slightly
different conclusion. Understanding gets re-derived thousands of times,
inconsistently, and never accumulates anywhere.

CodeBrain compiles a repository into a **Brain** — a durable, versioned,
provenance-carrying model of one software system, stored beside the code and
kept true on every commit — so any agent or engineer starting any lifecycle task
begins informed instead of blind.

> Full blueprint — vision, conceptual design, Claude Code integration, roadmap:
> [`docs/vision.html`](docs/vision.html)

---

## Status

**P0 — schema and store.** Complete. This is the foundation everything else
conforms to, not a working product yet. Today CodeBrain can build, read, merge,
diff and validate a Brain, with two extractors running behind the provider
interface. Context packs and the MCP server land in P2.

| Phase | Scope | State |
|-------|-------|-------|
| **P0** | Schema, provenance envelope, store, diff, plugin contract | ✅ done |
| P1 | Deterministic core — structure, operations, git archaeology, Atlas | next |
| P2 | Context packs + MCP server — first agent value | |
| P3 | Verification by execution, drift gate | |
| P4 | Semantics, behavior, constraints | |
| P5 | Memory and agent write-back | |
| P6 | Cortex — federation across repos | |

---

## Install

Requires Python 3.11+. **No third-party dependencies** — the deterministic core
must build a Brain offline, on a locked-down machine, with no API key.

```bash
pip install -e .
```

## Quickstart

```bash
codebrain build && codebrain status
```

```
Brain built at .brain
  33 records · 21 nodes · 0 edges · 12 facts
  providers: census, gitmeta

33 records — 21 nodes, 0 edges, 12 facts

  layer                         records
    L0 corpus       ███████████████████···     28
  · L1 structure    ······················      0
    L4 intent       ███···················      5

  method        DERIVED 2   EXTRACTED 31
  status        fresh 32   unverified 1
```

Other commands:

```bash
codebrain providers                      # what can extract from this repo
codebrain validate                       # structural problems, non-zero on failure
codebrain diff old-brain new-brain --check   # the seed of the CI drift gate
```

---

## The model

A Brain is **eight layers** of **three record types**, each carrying **one
envelope**.

| | Layer | Answers |
|---|---|---|
| L0 | corpus | what is this thing, and what is in it |
| L1 | structure | what exists and how is it wired |
| L2 | behavior | what it does when it runs |
| L3 | semantics | what it means in the business |
| L4 | intent | why it is like this |
| L5 | operations | how to build, test, run, ship it |
| L6 | constraints | what must not break |
| L7 | memory | what we've learned since |

**Nodes** are things that exist. **Edges** are typed relationships. **Facts** are
claims that aren't graph-shaped — and are the unit of verification: *"the test
command is `make test`"* is a hypothesis until CodeBrain runs it.

Every record carries a provenance envelope:

```python
Envelope.make(
    Method.EXTRACTED,          # EXTRACTED | DERIVED | INFERRED | OBSERVED | ASSERTED
    source="census",           # which provider said so
    as_of="d3f91d2",           # the commit it was true at
    evidence=(Evidence(path="src/api.py", start_line=44),),
)
```

This is the honesty model, and it is load-bearing. An agent can demand
`EXTRACTED`-only facts before a risky refactor. Confidence decays with status —
a refuted claim scores 0.0 and can never reach a context pack. **A Brain that
confabulates silently is worse than no Brain.**

Conflicts between providers are settled by the envelope, not by whoever ran
last, and the ordering is total — so two builds of the same inputs produce
byte-identical Brains, and the drift gate never fires on noise.

## On-disk format

```
.brain/
  manifest.json        schema version, as_of commit, providers that ran
  layers/l0.jsonl …    one record per line, sorted by id
  .gitignore           derived indexes only — never the JSONL
```

JSONL sorted by readable id, because **the Brain is committed to the repository
it describes and reviewed on pull requests**. Ids look like
`L1:symbol:payments/api.py#charge_endpoint`, not `a3f9c2e1` — a reviewer has to
be able to read the diff.

## Writing an extractor

```python
from codebrain.providers import Provider, register
from codebrain.model import Fact, Layer, REPO

class MyProvider(Provider):
    id = "mine"
    layers = (Layer.L5,)

    def applies(self, ctx):
        return (ctx.root / "Makefile").is_file()

    def extract(self, ctx):
        yield Fact(layer=Layer.L5, subject=REPO, predicate="test_command",
                   value="make test", env=...)

register(MyProvider())
```

Two rules: declare the layers you write, and return `False` from `applies` when
you can't run. A provider that can't run says so — it does not emit
low-confidence guesses to look busy. A provider that raises is recorded and
skipped, never fatal: a partial Brain beats no Brain.

## Development

```bash
python -m unittest discover -s tests -t .
```

69 tests, no external test runner required.

---

## License

MIT © EnterpriseX Platform
