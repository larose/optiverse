"""How the search moves: an iterated local search over the tree of ideas.

The programmer is the local search. Handed the same constraints and the same
code twice it writes something different both times, and a node keeps the best
of what it is handed — so re-running a node *is* a descent. It stops being worth
paying for when the node's own best stops moving, and `PATIENCE` attempts
without an improvement is where that is called.

The perturbation is a constraint, and a constraint is a node. It hangs under a
base drawn uniformly at random, which is the whole of the restart rule: no score
weighting and no exploit probability. A search that always kicks from the
incumbent is the search that spends four hundred iterations polishing one
branch, and that is the failure this module exists to make impossible rather
than to discourage.

Which leaves the director one job — write the constraint — and takes the other
two off it. The code a move starts from is no longer a choice either: it is the
best solution at the base, so the programmer opens on code that already
satisfies the constraints it is about to be given. Choosing the two separately
let a node whose constraints said one thing be worked from code that did
another.

Nothing here reads or writes the disk and nothing here holds state. Every
decision is a function of the tree, the solutions and the iteration number, so
`preview` shows exactly what a run would do without running it, and a retried
iteration decides the same way as the attempt it replaces.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence

from .graph import ROOT_NODE_ID, STALENESS_LIMIT, Graph, Node, current_node
from ..solution import Solution

# Attempts a node may make without improving on its own best before the search
# perturbs instead. It is `STALENESS_LIMIT` under the name that says what it now
# does: the number used to be a label the tree rendered beside a node, and is
# now the thing that ends a descent.
PATIENCE = STALENESS_LIMIT

# How often a perturbation arrives with a kick. Half: often enough that the
# director meets one regularly, rare enough that the tree does not become twenty
# variations on the same handful of suggestions.
KICK_PROBABILITY = 0.5

# What a kick is clipped to when it does not carry a title of its own.
_TITLE_WIDTH = 48


class Phase(Enum):
    """Which half of the loop an iteration is."""

    LOCAL_SEARCH = "local_search"
    """Re-run the node the search is sitting on. No director, no new node."""

    PERTURB = "perturb"
    """Ask the director for one constraint and hang it under a random node."""


@dataclass(frozen=True)
class Move:
    """What the next iteration does, decided before any model is called.

    There is no `node_id` here because on a perturbation there is not one yet:
    the node is minted from the constraint the director has not written. `base`
    is what both phases have — the node a local search re-runs, or the node a
    perturbation hangs its child under.
    """

    phase: Phase
    base: Node
    parent_solution: Solution
    """The code the programmer opens on. Always the base's own best, so it
    satisfies every constraint the programmer is about to be handed."""

    kick: Optional[str]
    """One entry from `kicks.md`, on half of perturbations. None otherwise."""

    drawn_from: Optional[int]
    """How many nodes `base` was drawn out of. None on a local search, which
    draws nothing — the search is already standing where it continues."""


def decide(
    *,
    graph: Graph,
    solutions: Sequence[Solution],
    iteration: int,
    kicks: Sequence[str],
) -> Move:
    """The whole controller.

    Every draw comes from one generator seeded on the iteration number, in a
    fixed order, so an iteration that crashes and is retried draws the same base
    and the same kick. Reproducibility is not incidental here: the run directory
    is the only record, and a decision that cannot be reproduced from it is one
    nobody can explain afterwards.

    The root is never searched locally, only branched from, which is why the
    first iteration of a run perturbs. A local search there would hand the
    programmer a node with no constraints, and the constraints are the entire
    instruction it gets — so the prompt would carry no instruction at all, not a
    weak one. The root is not an idea; it is the absence of one, and the search
    starts by having a first idea rather than by writing code without one.
    """
    here = graph.node(current_node(graph, solutions))

    if here.id != ROOT_NODE_ID and here.stale < PATIENCE:
        return Move(
            phase=Phase.LOCAL_SEARCH,
            base=here,
            parent_solution=code_source(graph, here),
            kick=None,
            drawn_from=None,
        )

    rng = random.Random(iteration)
    candidates = eligible(graph)

    base = rng.choice(candidates)
    kick = (
        rng.choice(list(kicks)) if kicks and rng.random() < KICK_PROBABILITY else None
    )

    return Move(
        phase=Phase.PERTURB,
        base=base,
        parent_solution=code_source(graph, base),
        kick=kick,
        drawn_from=len(candidates),
    )


def eligible(graph: Graph) -> List[Node]:
    """The nodes a perturbation may hang under, in a fixed order.

    A node with nothing scored under it has no code to start from and no
    evidence to build on, so it is not somewhere to return to — which is also
    what keeps the nodes left behind by a crashed perturbation out of the draw.
    The root qualifies as soon as the seed scores, and stands in alone before
    anything has.

    It stays here even though it is never searched locally: hanging a constraint
    under the root creates a child carrying exactly that one constraint, which is
    the restart from the seed and the only way out of a tree that has become a
    single lineage.
    """
    scored = [
        node
        for node in graph.nodes()
        if any(solution.score is not None for solution in node.solutions)
    ]

    return sorted(scored or [graph.node(ROOT_NODE_ID)], key=lambda node: node.id)


def code_source(graph: Graph, node: Node) -> Solution:
    """The code a move at `node` starts from.

    The best solution at the node itself, and failing that the nearest ancestor
    holding one. Walking up rather than sideways is the point: an ancestor's code
    satisfies a subset of this node's constraints, so it is the closest thing to
    a valid starting point that exists. Code from a sibling branch would satisfy
    constraints this node does not have and miss ones it does.
    """
    for node_id in graph.ancestors(node.id):
        best = _best(graph.node(node_id).solutions)

        if best is not None:
            return best

    # Nothing on the path to the root has scored. The seed is still there — the
    # loop commits it before the first iteration — and it is the only code left.
    return graph.node(ROOT_NODE_ID).solutions[0]


def phase_of(solution: Solution, first_seen: Dict[str, int]) -> Phase:
    """Which phase produced a solution, read back off the tree.

    A perturbation mints a fresh node and works it in the same iteration, and
    only nodes with a scored solution can be drawn as a base — so a non-root
    node's earliest solution is always the perturbation that created it, and
    everything after it is local search. Nothing is stored for this, because
    storing it would be a second copy of a fact `arcs.json` and the solutions
    already agree on.

    The root case is a guard rather than a live path: no iteration commits there,
    since the root is never searched locally, and the seed that does sit there
    has no iteration and so is never asked about.
    """
    if solution.node_id == ROOT_NODE_ID:
        return Phase.LOCAL_SEARCH

    return (
        Phase.PERTURB
        if first_seen.get(solution.node_id) == solution.iteration
        else Phase.LOCAL_SEARCH
    )


def kick_title(kick: str) -> str:
    """A kick in a few words, for a log line or a preview header.

    Every entry in `kicks.md` opens `- **Title.** body`, so the title is what
    sits between the first pair of `**`. A file someone has edited into another
    shape falls back to a clipped opening rather than to nothing, because this is
    only ever used to label something the reader can go and look at in full.
    """
    collapsed = " ".join(kick.split())
    _, marker, rest = collapsed.partition("**")
    title, closing, _ = rest.partition("**")

    if marker and closing and title.strip():
        return title.strip().rstrip(".")

    return collapsed[:_TITLE_WIDTH]


def _best(solutions: Sequence[Solution]) -> Optional[Solution]:
    scored = [solution for solution in solutions if solution.score is not None]

    if not scored:
        return None

    return min(scored, key=lambda solution: solution.score or 0.0)


__all__ = [
    "KICK_PROBABILITY",
    "PATIENCE",
    "Move",
    "Phase",
    "code_source",
    "decide",
    "eligible",
    "kick_title",
    "phase_of",
]
