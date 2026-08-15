"""Built-in extractors.

Importing this package registers them. Ordering matters only in that later
providers reference nodes earlier ones create — `census` establishes the file
nodes everything else hangs off, so it runs first.

P0 shipped census and gitmeta, deliberately overlapping so that the envelope's
conflict resolution is exercised on a real disagreement. P1 adds the
deterministic core: history (L4), operations (L5), and structure for Python and
TypeScript/JavaScript (L1).
"""

from __future__ import annotations

from . import (
    behavior,
    census,
    constraints,
    gitmeta,
    history,
    operations,
    semantics,
    structure_py,
    structure_rs,
    structure_ts,
)

__all__ = ["behavior", "census", "constraints", "gitmeta", "history", "operations",
           "semantics", "structure_py", "structure_rs", "structure_ts"]
