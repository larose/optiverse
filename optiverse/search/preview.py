"""Print the prompt a run's next director would be shown.

    python3 -m optiverse.search.preview <run directory> [iteration]

The director's prompt is most of what this project is, and looking at one
normally costs two model calls — one to produce the state, one to read the prompt
that state produced. That is a bad loop to iterate prose in. This builds the same
prompt from a run already on disk and prints it, for nothing.

Passing an iteration renders as though the search were about to plan that one,
which is how the rotation in *Ideas already tried* is checked: run it over a few
consecutive numbers and watch the window advance and wrap.

It writes nothing, because `Search.compose` writes nothing. The problem statement
is the one thing not in a run directory, so a placeholder stands in for it.
"""

import sys
from pathlib import Path
from typing import List, Optional

from . import DEFAULT_PLAYBOOK, Search
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
        playbook=DEFAULT_PLAYBOOK,
        store=store,
    )

    at = iteration if iteration is not None else search.completed_iterations() + 1
    prompt, turn = search.compose(at, PLACEHOLDER)

    header = [
        f"<!-- preview: iteration {turn.iteration} of {directory} -->",
        f"<!-- stage {turn.stage.value}"
        f" | review: {turn.reviewing or 'no'}"
        f" | reconciling: {'yes' if turn.reconciling else 'no'} -->",
        "",
    ]

    return "\n".join(header) + prompt


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2

    print(render(Path(argv[0]), int(argv[1]) if len(argv) > 1 else None))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
