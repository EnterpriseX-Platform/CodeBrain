"""Built-in extractors.

P0 ships two, deliberately overlapping: they both claim to know the repository's
name, by different means and with different authority. That overlap is the point
— it exercises the envelope's conflict resolution on a real disagreement rather
than a contrived one.

Importing this package registers them.
"""

from __future__ import annotations

from . import census, gitmeta

__all__ = ["census", "gitmeta"]
