"""On-disk format.

The Brain is committed to the repository it describes, so the format is chosen
for git rather than for query speed: one JSONL file per layer, one record per
line, sorted by id, keys sorted within each record. That makes a Brain update a
readable line diff on a pull request, and makes concurrent edits to different
layers merge cleanly.

Query indexes are derived and gitignored — never the source of truth. Losing
`.brain/claims.db` costs a rebuild, not knowledge.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from .model import Brain, Layer, Manifest, Record, record_from_json

BRAIN_DIR = ".brain"
LAYERS_DIR = "layers"
MANIFEST = "manifest.json"

#: Derived artefacts. Present in .brain/ but never committed.
GITIGNORE = """\
# Derived indexes — rebuilt from layers/*.jsonl by `codebrain build`.
# The JSONL files are the source of truth and must stay committed.
claims.db
packs/
*.tmp
"""


class BrainNotFound(FileNotFoundError):
    pass


def layer_path(root: Path, layer: Layer) -> Path:
    return root / LAYERS_DIR / f"{str(layer).lower()}.jsonl"


def _atomic_write(path: Path, text: str) -> None:
    """Write via a sibling temp file so an interrupted build cannot leave a torn Brain."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _dump(rec: Record) -> str:
    # sort_keys + no whitespace padding: two builds of the same inputs must be
    # byte-identical, or the drift gate fires on noise.
    return json.dumps(rec.to_json(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def save(brain: Brain, root: Path | str = BRAIN_DIR) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    buckets: dict[Layer, list[Record]] = {lyr: [] for lyr in Layer}
    for rec in brain.records():
        buckets[rec.layer].append(rec)

    for lyr, recs in buckets.items():
        path = layer_path(root, lyr)
        if not recs:
            path.unlink(missing_ok=True)
            continue
        recs.sort(key=lambda r: r.id)
        _atomic_write(path, "".join(_dump(r) + "\n" for r in recs))

    _atomic_write(root / MANIFEST,
                  json.dumps(brain.manifest.to_json(), indent=2, sort_keys=True) + "\n")
    _atomic_write(root / ".gitignore", GITIGNORE)
    return root


def load(root: Path | str = BRAIN_DIR) -> Brain:
    root = Path(root)
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise BrainNotFound(f"no Brain at {root} — run `codebrain build`")

    brain = Brain(Manifest.from_json(json.loads(manifest_path.read_text(encoding="utf-8"))))
    for lyr in Layer:
        path = layer_path(root, lyr)
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    # Straight into the bucket: the file is already the resolved
                    # state, so re-running merge resolution here would be wrong.
                    rec = record_from_json(json.loads(line))
                except (json.JSONDecodeError, ValueError, KeyError) as exc:
                    raise ValueError(f"{path}:{lineno}: corrupt record — {exc}") from exc
                brain._bucket(rec)[rec.id] = rec
    return brain


def exists(root: Path | str = BRAIN_DIR) -> bool:
    return (Path(root) / MANIFEST).is_file()


def append_memory(root: Path | str, records: Iterable[Record]) -> int:
    """Append-only write to L7.

    Session write-back (the Stop hook) must never rewrite the whole Brain — it
    runs while the user is waiting, and a partial write would be worse than a
    lost lesson.
    """
    root = Path(root)
    path = layer_path(root, Layer.L7)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            if rec.layer is not Layer.L7:
                raise ValueError(f"{rec.id} is not an L7 record")
            fh.write(_dump(rec) + "\n")
            n += 1
    return n
