"""Print the prompt a run's next director would be shown.

    python3 -m optiverse.search.preview <run directory> [iteration]

The director's prompt is most of what this project is, and looking at one
normally costs two model calls — one to produce the state, one to read the prompt
that state produced. That is a bad loop to iterate prose in. This builds the same
prompt from a run already on disk and prints it, for nothing.

Passing an iteration renders as though the search were about to run that one,
which is how both moving parts are checked: the rotation in *Ideas already tried*
advances and wraps, and the base node and the kick are drawn from the iteration
number, so consecutive numbers show the draws actually varying — and the same
number twice shows them not varying, which is the property the run depends on.

Most iterations are local search and call no director at all. The prompt is
rendered for them anyway, because the point here is to read the prose rather
than to predict the schedule; the header says which it would have been.

It renders. It does not touch what the run has decided.
"""

import sys
from pathlib import Path
from typing import List, Optional

from . import DEFAULT_KICKS, Search
from .policy import Phase, kick_title
from ..director import Director, DirectorContext, DirectorResult
from ..solution import FileSystemStore

PLACEHOLDER = (
    "(the problem statement goes here — it is passed in by the loop and is not "
    "part of the run directory, so a preview cannot show the real one)"
)


class _NoDirector(Director):
    """Stands in for the one that is not going to be called."""

    def decide(self, context: DirectorContext) -> DirectorResult:
        raise NotImplementedError


def render(directory: Path, iteration: Optional[int] = None) -> str:
    store = FileSystemStore(directory)
    search = Search(
        directory=directory,
        director=_NoDirector(),
        kicks=DEFAULT_KICKS,
        store=store,
    )

    at = iteration if iteration is not None else search.completed_iterations() + 1
    move = search.next_move(at)

    note = "" if move.phase is Phase.PERTURB else " — no director runs on this one"

    header = [
        f"<!-- preview: iteration {at} of {directory}{note} -->",
        f"<!-- phase {move.phase.value}"
        f" | base {move.base.id}"
        f" | from {move.parent_solution.id}"
        f" | kick: {_kick(move.kick)} -->",
        "",
    ]

    return "\n".join(header) + search.compose(at, move, PLACEHOLDER)


def _kick(kick: Optional[str]) -> str:
    return "none" if kick is None else kick_title(kick)


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2

    print(render(Path(argv[0]), int(argv[1]) if len(argv) > 1 else None))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
