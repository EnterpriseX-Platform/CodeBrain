"""CodeBrain — compile a repository into a Brain.

A Brain is a durable, versioned, provenance-carrying model of one software
system: eight layers, three record types, one envelope on every claim. Agents
and humans read it through context packs, an MCP server, and a generated Atlas.

P0 is the schema and the store. Nothing here is clever, and everything
downstream conforms to it.
"""

from __future__ import annotations

from .envelope import (
    DECAY,
    DEFAULT_CONFIDENCE,
    SCHEMA_VERSION,
    TRUST,
    Envelope,
    Evidence,
    Method,
    Status,
    utc_now,
)
from .model import (
    LAYER_NAMES,
    REPO,
    Brain,
    Edge,
    Fact,
    Layer,
    Manifest,
    MergeReport,
    Node,
    Record,
    new_brain,
    record_from_json,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "Envelope", "Evidence", "Method", "Status", "utc_now",
    "TRUST", "DECAY", "DEFAULT_CONFIDENCE",
    "Brain", "Node", "Edge", "Fact", "Record", "Layer", "LAYER_NAMES",
    "Manifest", "MergeReport", "REPO", "new_brain", "record_from_json",
]
