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

import json
import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, cast

from .store import Solution

ARCS_NAME = "arcs.json"

ROOT_NODE_ID = "n_root"

NODE_ID_PREFIX = "n_"

# Attempts a node may make without improving on its own best before the search
# should stop spending on it. Four attempts to reach — the first sets the best —
# so a single bad sample never kills an idea.
STALENESS_LIMIT = 3

# What the tree may cost the prompt. A tree is worth reading for its shape, and
# past a screenful or two the shape is what is lost first. The constraints that
# get folded away are not lost with it: the prompt puts them back on rotation.
MAX_TREE_LINES = 120

# Attempts listed under a node the search is working. Older ones are counted
# rather than named — a solution nobody is going to build on is a number.
SOLUTIONS_SHOWN = 8


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

    @property
    def last_worked(self) -> Optional[int]:
        """The last iteration that built here.

        A node with four attempts and a good best reads as live until you know
        the last of them was three hundred iterations ago. Without this the tree
        has no time in it at all, and two nodes with identical counts look
        identical however far apart they happened.

        None at a node holding only the seed, which no iteration produced.
        """
        iterations = [s.iteration for s in self.solutions if s.iteration is not None]

        return max(iterations) if iterations else None


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

        # Indexed once rather than scanned per call. `children` is asked for
        # every node on every render, so scanning made rendering quadratic in a
        # structure that is otherwise cheap.
        self._children: Dict[str, List[Node]] = {}

        for node in self._nodes.values():
            if node.parent_node_id is not None:
                self._children.setdefault(node.parent_node_id, []).append(node)

        for siblings in self._children.values():
            siblings.sort(key=lambda node: node.id)

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

    def nodes(self) -> List[Node]:
        return list(self._nodes.values())

    def children(self, node_id: str) -> List[Node]:
        return list(self._children.get(node_id, []))

    def subtree(self, node_id: str) -> List[Node]:
        """`node_id` and everything under it."""
        found: List[Node] = []
        queue = [node_id]

        while queue:
            node = self._nodes[queue.pop()]
            found.append(node)
            queue.extend(child.id for child in self.children(node.id))

        return found

    def ancestors(self, node_id: str) -> List[str]:
        """Every node from `node_id` up to the root, inclusive."""
        walked: List[str] = []
        current: Optional[str] = node_id

        while current is not None and current in self._nodes:
            walked.append(current)
            current = self._nodes[current].parent_node_id

        return walked

    def render(
        self,
        anchors: Sequence[str],
        *,
        best_solution_id: Optional[str] = None,
        max_lines: int = MAX_TREE_LINES,
    ) -> List[str]:
        """The tree, as the director reads it.

        `anchors` are the nodes the search has been working and the one holding
        the run's best. There is deliberately no single focus: a "you are here"
        is a default, and a default is the thing a search loops on. Their
        solutions are listed in full, because `parent_solution_id` is chosen from
        them; elsewhere only the best is named.

        Everything else is folded away — exhausted branches into a count, and
        then, if it is still too long, the least promising subtrees. Folding a
        constraint out of the tree does not hide it: it comes back on rotation in
        the prompt's own section, so what is lost here is depth of detail rather
        than the idea.
        """
        wanted = [node_id for node_id in anchors if node_id in self._nodes]
        anchored = set(wanted or [ROOT_NODE_ID])

        spine: Set[str] = set()
        for node_id in anchored:
            spine.update(self.ancestors(node_id))

        shown = self._reachable(spine, anchored)
        optional = self._optional(shown, spine)

        lines, hidden = self._lay_out(shown, anchored, best_solution_id)

        while len(lines) > max_lines and optional:
            shown.difference_update(node.id for node in self.subtree(optional.pop().id))
            lines, hidden = self._lay_out(shown, anchored, best_solution_id)

        if hidden:
            plural = "" if hidden == 1 else "s"
            lines.append("")
            lines.append(
                f"{hidden} node{plural} not shown. `Ideas already tried` samples "
                "them, and `../../arcs.json` has all of them."
            )

        return lines

    def _reachable(self, spine: Set[str], anchored: Set[str]) -> Set[str]:
        """Every node worth a line of its own.

        A live child of a shown node, because that is where the search can still
        go. Every child of an anchor whether live or not, because the dead
        siblings of where the search has been are exactly the alternatives it is
        choosing between, and a director that cannot see them re-invents them.
        """
        shown = set(spine)
        queue = list(spine)

        while queue:
            node_id = queue.pop()

            for child in self.children(node_id):
                if child.id in shown:
                    continue

                if node_id in anchored or not child.dead:
                    shown.add(child.id)
                    queue.append(child.id)

        return shown

    def _optional(self, shown: Set[str], spine: Set[str]) -> List[Node]:
        """Subtree roots the cap may drop, least promising last.

        Popped from the end, so the worst goes first. A node on the way to an
        anchor is never here: dropping one would hide where the search is.
        """
        roots = [
            self._nodes[node_id]
            for node_id in shown
            if node_id not in spine
            and (self._nodes[node_id].parent_node_id or "") in shown
        ]

        return sorted(
            roots,
            key=lambda node: (
                node.best is None,
                -(node.best if node.best is not None else 0.0),
            ),
        )

    def _lay_out(
        self,
        shown: Set[str],
        anchored: Set[str],
        best_solution_id: Optional[str],
    ) -> Tuple[List[str], int]:
        lines: List[str] = []
        hidden = 0

        def walk(node: Node, depth: int) -> None:
            nonlocal hidden

            indent = "  " * depth
            label = "(no constraints)" if node.constraint is None else node.constraint
            here = "+" if _holds(node, best_solution_id) else "-"

            lines.append(f'{indent}{here} {node.id}  "{label}"')
            lines.append(f"{indent}  {_stats(node)}")
            lines.extend(
                f"{indent}    {line}"
                for line in _solution_lines(
                    node, full=node.id in anchored, best_solution_id=best_solution_id
                )
            )

            folded = [c for c in self.children(node.id) if c.id not in shown]

            if folded:
                buried = [n for root in folded for n in self.subtree(root.id)]
                hidden += len(buried)
                lines.append(f"{indent}  … {_folded(buried)}")

            for child in self.children(node.id):
                if child.id in shown:
                    walk(child, depth + 1)

        walk(self._nodes[ROOT_NODE_ID], 0)

        return lines, hidden


def _folded(buried: Sequence[Node]) -> str:
    """One line standing in for a set of nodes nobody is going to open.

    It names the best of them rather than only counting: a folded branch that
    once scored well is a thing to reopen, and a count alone cannot be reopened.
    """
    plural = "" if len(buried) == 1 else "s"
    scored = [node for node in buried if node.best is not None]

    if not scored:
        return f"{len(buried)} node{plural} below, none scored"

    best = min(scored, key=lambda node: node.best if node.best is not None else 0.0)
    state = "all exhausted" if all(node.dead for node in buried) else "folded away"

    return (
        f"{len(buried)} node{plural} below, {state}, best "
        f'{cast(float, best.best):.6g} at {best.id} "{_clip(best.constraint)}"'
    )


def _clip(constraint: Optional[str], width: int = 48) -> str:
    if constraint is None:
        return "(no constraints)"

    collapsed = " ".join(constraint.split())

    return collapsed if len(collapsed) <= width else collapsed[: width - 1] + "…"


def _holds(node: Node, solution_id: Optional[str]) -> bool:
    return solution_id is not None and any(s.id == solution_id for s in node.solutions)


def _stats(node: Node) -> str:
    attempts = len(node.solutions)
    plural = "" if attempts == 1 else "s"
    scores = sorted(node.scores)

    if not node.solutions:
        # Minted by an attempt that then crashed, since a node is created and
        # worked in one iteration. Named rather than left to look like an idea
        # that failed: nothing was ever built for it.
        return "never attempted"

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

    worked = node.last_worked
    when = "" if worked is None else f", last worked at i{worked}"

    return f"{state}, stale {node.stale}{when}" + (" — DEAD" if node.dead else "")


def _solution_lines(
    node: Node,
    *,
    full: bool,
    best_solution_id: Optional[str] = None,
) -> List[str]:
    if not node.solutions:
        return []

    scored = [s for s in node.solutions if s.score is not None]

    if not full:
        if not scored:
            return []
        best = min(scored, key=lambda s: cast(float, s.score))
        return [f"{_solution(best, best_solution_id)}  (best here)"]

    listed = node.solutions[-SOLUTIONS_SHOWN:]
    earlier = len(node.solutions) - len(listed)
    lines = [f"… {earlier} earlier"] if earlier else []

    return lines + [_solution(s, best_solution_id) for s in listed]


def _solution(solution: Solution, best_solution_id: Optional[str]) -> str:
    score = "did not score" if solution.score is None else f"{solution.score:.6g}"
    marker = "  *" if solution.id == best_solution_id else ""

    return f"{solution.id}  {score}{marker}"


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
