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

**P4 — semantics, behavior and constraints.** Complete. All eight layers are
populated, every one of them offline and deterministic.

| Phase | Scope | State |
|-------|-------|-------|
| **P0** | Schema, provenance envelope, store, diff, plugin contract | ✅ done |
| **P1** | Deterministic core — L0/L1/L4/L5 extractors, Atlas | ✅ done |
| **P2** | Context packs, MCP server, hooks, thin L6, eval harness | ⚠️ shipped; retrieval gate not met leak-free |
| **P3** | Verification by execution, carry-forward, sync, drift gate | ✅ done |
| **P4** | L2 behavior, L3 semantics, full L6, rigorous eval | ✅ done |
| P5 | Memory and agent write-back | next |
| P6 | Cortex — federation across repos | |

### Measured, not asserted

**Cold build.** 331,000 lines of Python (the CPython standard library) →
234,268 records in **24.5s**, about 7.4s per 100k LOC against a 60s gate.

**Retrieval — and a negative result.** `codebrain eval` generates its own
benchmark from git history: each past commit is a task whose correct answer is
the files it changed. Packs are compared against keyword search over the same
repository, at identical k.

Run leak-free, with a Brain per case built from that commit's *parent* so
nothing in it can know the answer:

| Repository | Files | Pack recall@k | Search recall@k | Delta |
|---|---:|---:|---:|---:|
| django (`--rigorous`, 20 cases) | 2,928 `.py` | 35.0% | 35.0% | **±0.0%** |

**On file retrieval, leak-free, the pack does not beat keyword search.** It
matches it. That is the honest number and it is the one to quote.

The fast mode — one Brain built from HEAD — reports +7.3% on the same
repository, and most of that gap is leakage. The mechanism is specific: L4
co-change coupling is computed over history that *includes the commit being
tested*, so the precedent facet learns "these files change together" from the
very commit it is being asked about. Both arms read the post-change tree, but
only the pack gets that extra hint.

What the benchmark does **not** measure is everything the pack carries beyond
its anchors: the contracts, the constraints, the verified runbook, the blast
radius, the declared unknowns. Those are five of the six facets and they are the
reason the pack exists — but "did it name the right files" cannot see them.
Retrieval parity at ~1,000 tokens *plus* those facets may well be worth it; this
harness cannot demonstrate that, and pretending otherwise would be exactly the
kind of confident unverified claim the whole project is built to avoid.

Measuring the other facets needs agents completing real tasks, which is P5's
problem.

```bash
codebrain eval --cases 60              # fast, leaky, good for iteration
codebrain eval --cases 20 --rigorous   # slow, leak-free, the number to quote
```

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
  1656 records · 457 nodes · 1071 edges · 128 facts
  atlas:     .brain\ATLAS.md
  providers: census, gitmeta, history, structure-py, operations

  layer                         records
    L0 corpus       █·····················     40
    L1 structure    ████████████████████··   1495
    L4 intent       ██····················    118
    L5 operations   █·····················      3

  method        DERIVED 304   EXTRACTED 1352
  status        fresh 1655   unverified 1
```

Other commands:

```bash
codebrain atlas --out -                  # the human-readable Atlas, to stdout
```

```bash
codebrain providers                      # what can extract from this repo
```

```bash
codebrain validate                       # structural problems, non-zero on failure
```

```bash
codebrain diff old-brain new-brain --check   # compare two Brains directly
```

Every command takes the Brain path positionally or as `--brain`.

## What it extracts today

| Provider | Layer | Reads | Produces |
|---|---|---|---|
| `census` | L0 | filesystem | files, sizes, language mix, primary language |
| `gitmeta` | L0 · L4 | git | head, branch, remote, authors, commit count |
| `history` | L4 | `git log --numstat` | churn, hotspots, ownership, co-change coupling |
| `structure-py` | L1 | Python AST | modules, symbols, imports, resolved call graph |
| `structure-ts` | L1 | TS/JS scanner | modules, declarations, import graph |
| `operations` | L5 | manifests, Make, CI, Docker | build/test/run commands, pipelines, CODEOWNERS |
| `behavior` | L2 | AST decorators, imports | routes, entrypoints, jobs, env vars, data stores |
| `semantics` | L3 | the Brain itself | bounded-context candidates, ubiquitous language, entities |
| `constraints` | L6 | the Brain + `.codebrain.toml` | reviewers, danger zones, public contracts, untested churn, policy zones |

Two extractors overlap on purpose. `census` guesses the repo name from the
directory (`DERIVED` 0.5); `gitmeta` reads it from the remote (`EXTRACTED`
0.98). The envelope settles it — a real disagreement resolved by evidence
quality rather than by whichever provider ran last.

## Using it from an agent

```bash
codebrain pack "add rate limiting to the payments API"
```

```
CONTEXT PACK · task: add rate limiting to the payments API · 1653/6000 tokens · brain @f8ed2b1b

ANCHORS       payments/api.py:44  charge_endpoint (function)
BLAST RADIUS  payments/middleware.py:12  chain  (direct) [EXTRACTED]
              + 43 test symbol(s) across 4 test file(s) also depend on this
CONTRACTS     payments/api.py:44  charge_endpoint — 4 external caller(s)
PRECEDENT     payments/settle.py changes with payments/api.py (7 shared commits, 70%)
CONSTRAINTS   payments/api.py needs review from @risk-eng [DERIVED 0.80]
RUNBOOK       test   make test  [DERIVED 0.80]  (never executed)
UNKNOWNS      TS/JS has no call graph — blast radius across TS call sites is incomplete
```

Six facets, because six kinds of ignorance cause six kinds of failure: touching
the wrong code, breaking callers you did not know existed, changing a shape
someone depends on, inventing a pattern the repo already has, violating a
constraint, being unable to check your own work, and guessing confidently.

### Claude Code

`.mcp.json` exposes the Brain as an MCP server with `brain_pack`,
`brain_locate`, `brain_explain`, `brain_impact`, `brain_runbook` and
`brain_constraints`. Five hooks push context in without the model having to ask:

| Hook | Command | Does |
|---|---|---|
| SessionStart | `codebrain brief` | ~400-token orientation |
| UserPromptSubmit | `codebrain pack --stdin` | compiles a pack from the task, before the first action |
| PreToolUse | `codebrain guard` | checks the pending edit against L6 |
| PostToolUse | `codebrain touch` | marks the edited neighbourhood stale |
| Stop | `codebrain learn` | *(P5)* |

**Every hook fails open.** If `.brain/` is missing, stale or corrupt, hooks print
nothing, exit 0, and the session behaves exactly as it would without CodeBrain.
A Brain that can break someone's session gets uninstalled the first time it
does — so this is a tested contract, not an aspiration.

`guard` warns by default and never blocks. Denying an edit on inferred evidence
would get the hook removed; hard gates wait for real compliance zones in P4, and
are opt-in via `--deny-guarded` until then.

## Semantics without a language model

L3 is the layer people expect an LLM to write, and the temptation is to have one
narrate the domain and call the result knowledge. That would be expensive on
every build and would produce confident prose nobody can check.

So the part that *is* derivable is derived. Bounded contexts come from import
cohesion — a directory whose modules import each other far more than they import
outward **is** a boundary, whatever anyone calls it. Ubiquitous language comes
from the vocabulary the code actually uses, weighted by how widely each term is
shared. On this repository that yields `extract, provider, applies, brain, pack,
report, constraints` — which is, in fact, what this codebase is about.

What is left — business rules, entity relationships, *why* a boundary sits where
it does — needs a model, and the provider says so in a `semantics_coverage_gap`
fact rather than leaving the absence to be mistaken for "this repository has no
domain". An LLM-backed provider can register alongside it; its claims will be
`INFERRED`, ranked below these, and overridden by any human who disagrees.

## Declared policy

Everything in L6 is inferred from evidence except one thing. A team can state a
constraint outright in `.codebrain.toml`:

```toml
[[zone]]
name = "pci"
paths = ["payments/", "src/billing/"]
reason = "PCI scope — cardholder data"
requires = ["@risk-eng"]
block_agents = true
```

These are `ASSERTED`, so they outrank anything the machinery derives, and they
are the **only** constraint permitted to stop an agent. Inferred constraints —
churn, bus factor, missing tests — warn and never block, because denying an edit
on a guess gets the hook uninstalled by lunchtime.

## Verification — claims are executable

*"The test command is `make test`"* is a hypothesis until something runs it.

```bash
codebrain verify
```

That is a **dry run**. It prints exactly what it would execute and stops:

```
Dry run — nothing was executed. These commands came from this repository's
own manifests and would run as-is:

  test   python -m unittest discover -s tests -t .   (from pyproject.toml)

Read them, then re-run with --yes to execute and settle the claims.
```

```bash
codebrain verify --yes
```

```
  ok   test   python -m unittest discover -s tests -t .
       passed in 47.4s

  1 claim(s) promoted to OBSERVED · 0 refuted
```

The claim is now `OBSERVED` with the exit code, duration and command recorded as
evidence. A failing command is **refuted**, not deleted — it keeps its place with
the reason attached, scores 0.0 confidence, can never reach a context pack, and
packs say *"EXECUTED AND FAILED — do not rely on this"* rather than sending the
next agent to run something already known to be broken.

**Safety.** `verify` executes commands that came out of a repository, which on
an untrusted repo is arbitrary code execution. So it never runs from a hook or
from `build`; it is a dry run by default and needs an explicit `--yes` with the
command list already on screen; only commands CodeBrain itself extracted are
candidates; servers are excluded by construction (`npm start` never returns, and
a verifier that hangs is worse than one that never ran); and every run is
bounded by a timeout.

## Staying true — sync and the drift gate

```bash
codebrain sync
```

Rebuilds when something moved, and **carries verified and asserted claims
forward**. This is load-bearing: extraction is deterministic, so a rebuild
reproduces every `EXTRACTED` and `DERIVED` claim exactly — but it would also
overwrite the two kinds it cannot regenerate, what execution proved and what a
human stated. Without carry-forward, every build silently erases every
verification and P3 would be pointless by the next commit.

A verification only travels if the claim it proved is unchanged. Verified
`make test`, and the command is now `pytest`? The evidence is about a different
claim and is invalidated rather than carried — that would be the Brain lying
with a real receipt attached.

```bash
codebrain drift --check
```

Rebuilds into memory, compares against the committed Brain, writes nothing, and
exits non-zero when they disagree. That is the CI gate. A stale Brain misleads
every agent downstream, which is worse than having no Brain at all.

```
DRIFT: the committed Brain no longer describes the code.
  +2 -0 ~3 records

  + L1:symbol:codebrain/gitutil.py#drift_probe
  ~ L1:fact:|python_summary  (value)

  Run `codebrain sync` and commit the result.
```

## The Atlas

`codebrain build` also writes `.brain/ATLAS.md`: the onboarding document that is
always true because it is generated, not maintained. What this repo is, how to
run it, the most depended-upon modules, where the risk sits (hotspots,
single-author files, coupled files), who to ask — and a closing section stating
**what the Brain does not know**, because a document that hides its gaps gets
trusted exactly where it is weakest.

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

345 tests, no external test runner required.

---

## License

MIT © EnterpriseX Platform
