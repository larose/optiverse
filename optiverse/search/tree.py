"""Print a run's tree of ideas in full.

    python3 -m optiverse.search.tree <run directory> [node id]

The tree in the prompt folds. It has a line budget, and past a screenful the
shape is the first thing lost — which is right for a glance and wrong for the
question the director actually has to answer: what has this branch tried, and
how did it do. Answering that from `arcs.json` means joining uuid triplets
against every `metadata.json` in the run, by hand, every time.

So this prints all of it, with no cap and nothing folded: every node, the
constraint it commits to in full rather than clipped, every attempt and what
each one scored. Given a node id it prints that node's constraints from the root
down and then its subtree, which is how you follow one line of the search to its
end.

It writes nothing.
"""

import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from .graph import ROOT_NODE_ID, ArcStore, Graph, Node, stats
from ..solution import FileSystemStore, Solution

WIDTH = 88


def render(directory: Path, node_id: Optional[str] = None) -> str:
    store = FileSystemStore(directory)
    solutions = store.get_all_solutions()
    graph = Graph(ArcStore(directory).read(), solutions)

    root = node_id or ROOT_NODE_ID

    if root not in graph:
        return f"{root} is not a node in {directory}.\n"

    lines: List[str] = []
    node = graph.node(root)

    if node.constraints:
        lines += ["Constraints in force here, root first:", ""]

        for position, constraint in enumerate(node.constraints, start=1):
            lines += _wrap(constraint, prefix=f"  {position}. ", hanging="     ")
            lines.append("")

    lines += _walk(graph, node, 0)

    return "\n".join(lines) + "\n"


def _walk(graph: Graph, node: Node, depth: int) -> List[str]:
    indent = "  " * depth

    lines = [f"{indent}{node.id}"]

    if node.constraint is None:
        lines.append(f"{indent}  (no constraints)")
    else:
        lines += _wrap(node.constraint, prefix=f"{indent}  ", hanging=f"{indent}  ")

    lines.append(f"{indent}  {stats(node)}")
    lines += [f"{indent}    {_solution(s)}" for s in node.solutions]

    for child in graph.children(node.id):
        lines.append("")
        lines += _walk(graph, child, depth + 1)

    return lines


def _solution(solution: Solution) -> str:
    score = "did not score" if solution.score is None else f"{solution.score:.6g}"
    when = "" if solution.iteration is None else f"  i{solution.iteration}"

    return f"{solution.id}  {score}{when}"


def _wrap(text: str, *, prefix: str, hanging: str) -> List[str]:
    """A constraint, whole.

    Wrapped rather than clipped: this is the one place a constraint is shown in
    full, and a paragraph the search committed to is not a thing to show the
    first forty characters of.
    """
    return textwrap.wrap(
        " ".join(text.split()),
        width=WIDTH,
        initial_indent=prefix,
        subsequent_indent=hanging,
    ) or [prefix.rstrip()]


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2

    print(render(Path(argv[0]), argv[1] if len(argv) > 1 else None), end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
