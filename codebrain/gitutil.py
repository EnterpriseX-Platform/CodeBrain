"""Talking to git.

Every git read goes through here so that failure means one thing everywhere: a
shallow clone, a missing binary, a corrupt object store and a repo with no
commits all mean "this provider has nothing to say", never a crash.

One rule worth stating: **nothing here may depend on today's date.** A build
must be reproducible from a commit alone, or the drift gate fires whenever the
calendar moves rather than when the code does. That is why windows are measured
in commits (`--max-count`) and never in `--since`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_TIMEOUT = 60

#: Field separators, chosen because git will never emit them inside a value.
#: They come in pairs: the escape that goes *into* a --format argument, and the
#: character that comes back *out* in the output. A literal NUL cannot be passed
#: in argv at all, so the escape form is the only way to ask git for one.
REC_FMT, REC = "%x00", "\x00"
SEP_FMT, SEP = "%x1f", "\x1f"


def git(root: Path, *args: str, timeout: int = DEFAULT_TIMEOUT) -> str | None:
    """Run a git command; return stdout, or None if it could not be answered."""
    try:
        proc = subprocess.run(
            ("git", *args),
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def git_stripped(root: Path, *args: str, timeout: int = DEFAULT_TIMEOUT) -> str | None:
    out = git(root, *args, timeout=timeout)
    return out.strip() if out is not None else None


def is_repo(root: Path) -> bool:
    return git_stripped(root, "rev-parse", "--git-dir") is not None


def head(root: Path) -> str | None:
    return git_stripped(root, "rev-parse", "HEAD")


def normalise_rename(path: str) -> str:
    """Turn git's rename notation into the current path.

    git --numstat writes renames two ways:
        old/name.py => new/name.py
        src/{old => new}/name.py
    Both must resolve to the path as it exists now, or history attaches to files
    that no longer exist.
    """
    if "=>" not in path:
        return path.strip()

    if "{" in path and "}" in path:
        before, rest = path.split("{", 1)
        inner, after = rest.split("}", 1)
        new = inner.split("=>", 1)[1].strip()
        # `a/{b => }/c` collapses a path segment away entirely.
        joined = f"{before}{new}{after}" if new else f"{before.rstrip('/')}{after}"
        return joined.replace("//", "/").strip()

    return path.split("=>", 1)[1].strip()
