"""Helpers for codebases, which are plain directories.

A codebase is a `Path`, not a type. These are the few operations the loop needs
that `pathlib` does not already provide.
"""

import hashlib
import shutil
import stat
from pathlib import Path
from typing import List


def relative_files(root: Path) -> List[str]:
    """Every file under `root`, as sorted relative POSIX paths."""
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


def materialize(source: Path, destination: Path) -> None:
    """Copy `source` to `destination` and make sure the result is writable.

    Stored codebases are read-only (see `make_read_only`) and `copytree`
    preserves mode bits, so the copy has to be made writable again or the agent
    could not edit it.
    """
    shutil.copytree(source, destination, dirs_exist_ok=True)
    make_writable(destination)


def digest(root: Path) -> str:
    """A stable hash of the tree, used to tell whether the agent changed anything.

    Covers relative paths and file bytes; not mode bits or timestamps.
    """
    hasher = hashlib.sha256()

    for relative_path in relative_files(root):
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update((root / relative_path).read_bytes())
        hasher.update(b"\0")

    return hasher.hexdigest()


def _change_mode(root: Path, *, add: int, remove: int) -> None:
    for path in [root, *root.rglob("*")]:
        current = stat.S_IMODE(path.stat().st_mode)
        path.chmod((current | add) & ~remove)


def make_read_only(root: Path) -> None:
    """Drop write permission across the tree.

    Stored solutions are population members that later iterations re-read, so
    making them read-only turns "do not edit a parent" from a convention into a
    property the filesystem enforces.
    """
    _change_mode(root, add=0, remove=stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


def make_writable(root: Path) -> None:
    """Restore write permission across the tree."""
    _change_mode(root, add=stat.S_IWUSR, remove=0)


def remove_tree(root: Path) -> None:
    """Delete a tree even if `make_read_only` was applied to it."""
    if not root.exists():
        return

    make_writable(root)
    shutil.rmtree(root)


def describe(root: Path) -> str:
    """A short listing of the tree, for logs and prompts."""
    files = relative_files(root)

    if not files:
        return "(empty)"

    return "\n".join(
        f"  {name} ({(root / name).stat().st_size} bytes)" for name in files
    )
