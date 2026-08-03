"""The search space: a tree of ideas, stored as its arcs.

A **node** is an idea — the constraints accumulated from the root. A
**constraint** is prose telling the coding agent how to narrow its approach, and
it lives on the **arc**, not in the node. A tree node has exactly one inbound
arc, so an arc *is* a node's whole record, and the node set is implied by the
file: `arcs.json` is a list of `(parent, child, constraint)` triplets and nothing
else.

The root is `ROOT_NODE_ID`, a constant. It is the one node not created by a
decision — it exists because the search does — so it is not data, needs no
storage, and never appears as anyone's child. That is also why every triplet has
three non-null values: an arc runs between two nodes, and a null parent is a
placeholder pretending to be one.

Reading a node's constraints means walking to the root, so no constraint text is
stored twice and there is nowhere to retype one. Which is the point: the old
design identified a branch by its exact set of constraint strings, and continuing
one meant reproducing every paragraph byte-for-byte.

Solutions are not stored here — they carry a `node_id` and this reads it. A
node's statistics are therefore always computed, never maintained.
"""

import difflib
import json
import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, cast

from .store import Solution

ARCS_NAME = "arcs.json"

ROOT_NODE_ID = "n_root"

NODE_ID_PREFIX = "n_"

# Attempts a node may make without improving on its own best before the search
# should stop spending on it. Four attempts to reach — the first sets the best —
# so a single bad sample never kills an idea.
STALENESS_LIMIT = 3


def new_node_id() -> str:
    return NODE_ID_PREFIX + uuid.uuid4().hex


@dataclass(frozen=True)
class Arc:
    """One node's whole record: where it hangs, and what it commits to."""

    parent_node_id: str
    child_node_id: str
    constraint: str


@dataclass(frozen=True)
class Node:
    """A node with everything computed about it, ready to render or judge."""

    id: str
    parent_node_id: Optional[str]
    constraint: Optional[str]
    """The one constraint this node adds. None at the root, which adds nothing."""

    constraints: List[str]
    """Every constraint in force here, root first. This is the idea."""

    solutions: List[Solution]
    """Attempts at this node, in the order they were made."""

    @property
    def scores(self) -> List[float]:
        return [s.score for s in self.solutions if s.score is not None]

    @property
    def best(self) -> Optional[float]:
        scores = self.scores
        return min(scores) if scores else None

    @property
    def stale(self) -> int:
        """Attempts since this node's own best improved.

        Counted over its own solutions rather than its subtree: a node staying
        alive on its children's progress is precisely the case where it is done.
        An attempt that did not score counts as one that did not improve, because
        it did not.
        """
        best: Optional[float] = None
        since = 0

        for solution in self.solutions:
            score = solution.score

            if score is not None and (best is None or score < best):
                best = score
                since = 0
            else:
                since += 1

        return since

    @property
    def dead(self) -> bool:
        return self.stale >= STALENESS_LIMIT


class Graph:
    """The tree, read from `arcs.json` and the solutions that point at it."""

    def __init__(self, arcs: Sequence[Arc], solutions: Sequence[Solution]) -> None:
        by_node: Dict[str, List[Solution]] = {}
        for solution in solutions:
            by_node.setdefault(solution.node_id, []).append(solution)

        parents = {arc.child_node_id: arc.parent_node_id for arc in arcs}
        constraints = {arc.child_node_id: arc.constraint for arc in arcs}

        self._nodes: Dict[str, Node] = {}

        for node_id in [ROOT_NODE_ID, *(arc.child_node_id for arc in arcs)]:
            self._nodes[node_id] = Node(
                id=node_id,
                parent_node_id=parents.get(node_id),
                constraint=constraints.get(node_id),
                constraints=self._accumulate(node_id, parents, constraints),
                solutions=by_node.get(node_id, []),
            )

    @staticmethod
    def _accumulate(
        node_id: str,
        parents: Dict[str, str],
        constraints: Dict[str, str],
    ) -> List[str]:
        """The constraints in force at a node, root first.

        Walks upward and reverses, so the list reads in the order the search
        committed to them — which is the order the coding agent should meet them.
        """
        walked: List[str] = []
        current: Optional[str] = node_id
        seen: set[str] = set()

        while current is not None and current in constraints and current not in seen:
            seen.add(current)
            walked.append(constraints[current])
            current = parents.get(current)

        return list(reversed(walked))

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def node(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def children(self, node_id: str) -> List[Node]:
        return [n for n in self._nodes.values() if n.parent_node_id == node_id]

    def ancestors(self, node_id: str) -> List[str]:
        """Every node from `node_id` up to the root, inclusive."""
        walked: List[str] = []
        current: Optional[str] = node_id

        while current is not None and current in self._nodes:
            walked.append(current)
            current = self._nodes[current].parent_node_id

        return walked

    def render(self, focus: str) -> List[str]:
        """The tree, as the director reads it.

        `focus` is the node the search is sitting on. Its solutions and its
        ancestors' are listed in full, because that is where the next decision is
        made and `parent_solution_id` is chosen from them; elsewhere only the
        best is shown, so a long run does not bury the decision in history.
        """
        detailed = set(self.ancestors(focus))
        lines: List[str] = []

        def walk(node: Node, depth: int) -> None:
            indent = "  " * depth
            label = "(no constraints)" if node.constraint is None else node.constraint

            lines.append(f'{indent}- {node.id}  "{label}"')
            lines.append(f"{indent}  {_stats(node)}")
            lines.extend(
                f"{indent}    {line}"
                for line in _solution_lines(node, full=node.id in detailed)
            )

            for child in sorted(self.children(node.id), key=lambda c: c.id):
                walk(child, depth + 1)

        walk(self._nodes[ROOT_NODE_ID], 0)

        return lines


def _stats(node: Node) -> str:
    attempts = len(node.solutions)
    plural = "" if attempts == 1 else "s"
    scores = sorted(node.scores)

    if not scores:
        state = f"{attempts} attempt{plural}, none scored"
    elif len(scores) == 1:
        # A median over one score is the score again, and the spread it implies
        # is not there. One sample says one thing.
        state = f"{attempts} attempt{plural}, scored {scores[0]:.6g}"
    else:
        state = (
            f"{attempts} attempt{plural}, best {scores[0]:.6g}, "
            f"median {statistics.median(scores):.6g}, worst {scores[-1]:.6g}"
        )

    return f"{state}, stale {node.stale}" + (" — DEAD" if node.dead else "")


def _solution_lines(node: Node, *, full: bool) -> List[str]:
    if not node.solutions:
        return []

    scored = [s for s in node.solutions if s.score is not None]

    if not full:
        if not scored:
            return []
        best = min(scored, key=lambda s: cast(float, s.score))
        return [f"{best.id}  {cast(float, best.score):.6g}  (best here)"]

    return [
        f"{s.id}  " + ("did not score" if s.score is None else f"{s.score:.6g}")
        for s in node.solutions
    ]


class ArcStore:
    """`arcs.json`, read whole and rewritten whole.

    A run has hundreds of arcs at most and each is three short strings, so there
    is nothing here that wants an append-only format or an index.
    """

    def __init__(self, directory: Path) -> None:
        self._path = Path(directory) / ARCS_NAME

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> List[Arc]:
        if not self._path.is_file():
            return []

        raw = cast(List[Dict[str, str]], json.loads(self._path.read_text()))

        return [
            Arc(
                parent_node_id=str(entry["parent_node_id"]),
                child_node_id=str(entry["child_node_id"]),
                constraint=str(entry["constraint"]),
            )
            for entry in raw
        ]

    def add(self, *, parent_node_id: str, constraint: str) -> str:
        """Mint a child of `parent_node_id` and return its id."""
        arcs = self.read()
        child_node_id = new_node_id()

        arcs.append(
            Arc(
                parent_node_id=parent_node_id,
                child_node_id=child_node_id,
                constraint=constraint,
            )
        )

        self._path.write_text(
            json.dumps(
                [
                    {
                        "parent_node_id": arc.parent_node_id,
                        "child_node_id": arc.child_node_id,
                        "constraint": arc.constraint,
                    }
                    for arc in arcs
                ],
                indent=2,
            )
            + "\n"
        )

        return child_node_id


def similar(constraint: str, others: Sequence[str], threshold: float) -> bool:
    """Whether `constraint` reads like one of `others`.

    Only ever a warning. A near-duplicate is usually a retype, but sometimes the
    difference is the whole point and code cannot tell which.
    """
    normalized = _normalize(constraint)

    return any(
        difflib.SequenceMatcher(None, normalized, _normalize(other)).ratio()
        >= threshold
        for other in others
    )


def _normalize(constraint: str) -> str:
    return " ".join(constraint.split()).lower()


def current_node(graph: Graph, solutions: Sequence[Solution]) -> str:
    """Where the search is sitting: the node of the last solution committed.

    The dead check and the playbook trigger both need a node before the director
    has chosen one, and the search sits where it last was. Before there is a
    solution, that is the root — and so is a node the arcs no longer describe,
    which is not worth crashing the prompt over.
    """
    ordered = [s for s in solutions if s.iteration is not None]

    if not ordered:
        return ROOT_NODE_ID

    latest = max(ordered, key=lambda s: cast(int, s.iteration))

    return latest.node_id if latest.node_id in graph else ROOT_NODE_ID
