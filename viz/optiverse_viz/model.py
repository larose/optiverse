"""A run directory, read back as something a page can draw.

Nothing here parses the run. `optiverse` already knows how: `FileSystemStore`
reads the population and skips the directories a crashed iteration left half
written, and `Graph` turns `arcs.json` into nodes that already carry their
accumulated constraints, their attempts, their staleness and whether they are
dead. This module joins those two and adds only what a picture needs and a
prompt does not — a depth, a colour band, and the lineage edges as edges.

Two things are deliberately not read. `Journal` is windowed by design (it shows
a director the last twenty moves) and the page wants the whole run, and
`Graph.render` is the director's text tree, capped and folded, where the page
wants every node. Both would have been the wrong shape borrowed for the right
name.

The score bands are quantiles of the node bests rather than equal slices of
their range. A run spans 33,058 down to 2,588 with almost everything inside the
last few hundred, so equal slices paint the whole tree one colour and hide the
part anyone is looking at. The edges are handed to the page as real scores and
printed in the legend, so the non-linearity is stated rather than smuggled in.
"""

import bisect
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict, Union, cast

from optiverse.graph import ArcStore, Graph, Node
from optiverse.journal import iteration_name
from optiverse.store import (
    CODE_DIRECTORY_NAME,
    SOLUTIONS_DIRECTORY_NAME,
    FileSystemStore,
    Solution,
)

# Steps in the score ramp. Five is what the ramp in `app.css` defines and what
# the light and dark ordinal ramps were validated at; a run with fewer distinct
# bests uses fewer bands and the page spreads them over the same five steps.
BANDS = 5


class SolutionView(TypedDict):
    id: str
    score: Optional[float]
    iteration: Optional[int]
    parent_solution_id: Optional[str]
    started_at: str
    ended_at: str
    metrics: Dict[str, Union[float, int]]
    tags: Dict[str, Union[int, str]]
    code_path: str
    iteration_path: Optional[str]


class NodeView(TypedDict):
    id: str
    parent_node_id: Optional[str]
    constraint: Optional[str]
    constraints: List[str]
    depth: int
    created_at_iteration: Optional[int]
    attempts: int
    best: Optional[float]
    median: Optional[float]
    worst: Optional[float]
    stale: int
    dead: bool
    last_worked: Optional[int]
    band: Optional[int]
    solutions: List[SolutionView]


class LineageView(TypedDict):
    """One solution's code parent, as an edge between the nodes it joins."""

    from_node: str
    to_node: str
    from_solution: str
    to_solution: str
    iteration: Optional[int]
    cross_branch: bool


class RunView(TypedDict):
    path: str
    generated_at: str
    iterations: int
    solutions: int
    best_solution_id: Optional[str]
    best_node_id: Optional[str]
    best_score: Optional[float]
    band_edges: List[float]
    bands: int


class View(TypedDict):
    run: RunView
    nodes: List[NodeView]
    lineage: List[LineageView]


def build(directory: Path) -> View:
    """Everything the page draws, from the run at `directory`."""
    directory = Path(directory)

    solutions = FileSystemStore(directory).get_all_solutions()
    graph = Graph(ArcStore(directory).read(), solutions)

    nodes = graph.nodes()
    edges, band_of = _bands([n.best for n in nodes if n.best is not None])

    best = _best(solutions)

    return View(
        run=RunView(
            path=str(directory),
            generated_at=datetime.now().isoformat(timespec="seconds"),
            iterations=sum(1 for s in solutions if s.iteration is not None),
            solutions=len(solutions),
            best_solution_id=best.id if best else None,
            best_node_id=best.node_id if best else None,
            best_score=best.score if best else None,
            band_edges=edges,
            bands=len(edges) + 1,
        ),
        nodes=[_node(graph, node, band_of) for node in nodes],
        lineage=_lineage(graph, solutions),
    )


def _node(graph: Graph, node: Node, band_of: Dict[float, int]) -> NodeView:
    scores = node.scores
    iterations = [s.iteration for s in node.solutions if s.iteration is not None]

    return NodeView(
        id=node.id,
        # The root's parent is None and it is the only node for which that is
        # true, which is what the page stratifies on.
        parent_node_id=node.parent_node_id,
        constraint=node.constraint,
        constraints=node.constraints,
        # `ancestors` includes the node itself, so the root comes back one.
        depth=len(graph.ancestors(node.id)) - 1,
        # A node is minted and worked in one iteration, so its earliest solution
        # dates it. None marks one nothing was ever built for: the iteration that
        # created it crashed before it committed anything.
        created_at_iteration=min(iterations) if iterations else None,
        attempts=len(node.solutions),
        best=node.best,
        median=statistics.median(scores) if scores else None,
        worst=max(scores) if scores else None,
        stale=node.stale,
        dead=node.dead,
        last_worked=node.last_worked,
        band=band_of.get(node.best) if node.best is not None else None,
        solutions=[_solution(s) for s in node.solutions],
    )


def _solution(solution: Solution) -> SolutionView:
    return SolutionView(
        id=solution.id,
        score=solution.score,
        iteration=solution.iteration,
        parent_solution_id=solution.parent_solution_id,
        started_at=solution.started_at,
        ended_at=solution.ended_at,
        metrics=solution.metrics,
        tags=solution.tags,
        # Relative to the run directory rather than absolute: the page is written
        # into the run and is meant to be copied elsewhere, and an absolute path
        # baked into it would be a lie the moment it moved.
        code_path=(f"{SOLUTIONS_DIRECTORY_NAME}/{solution.id}/{CODE_DIRECTORY_NAME}"),
        iteration_path=(
            None
            if solution.iteration is None
            else f"iterations/{iteration_name(solution.iteration)}"
        ),
    )


def _lineage(graph: Graph, solutions: Sequence[Solution]) -> List[LineageView]:
    """Where each solution's *code* came from, as edges between nodes.

    A solution has two independent parents: `node_id` says which idea it
    attempts, `parent_solution_id` says which code it started from. The tree
    draws the first. This is the second, and the only view in which two lines of
    work combining is visible at all — a tree cannot show it by construction.

    Most of these run along an arc the tree already drew, which is the ordinary
    case and not worth a mark. `cross_branch` names the ones that do not, and the
    page shows those by default.
    """
    node_of = {solution.id: solution.node_id for solution in solutions}
    edges: List[LineageView] = []

    for solution in solutions:
        parent_id = solution.parent_solution_id

        if parent_id is None or parent_id not in node_of:
            continue

        from_node = node_of[parent_id]
        to_node = solution.node_id

        if to_node not in graph:
            continue

        # Along the tree when the code came from the node being worked or from
        # its parent — which is what "continue here" and "go one deeper" look
        # like. Anything else reached across the tree for it.
        along_tree = from_node in {to_node, graph.node(to_node).parent_node_id}

        edges.append(
            LineageView(
                from_node=from_node,
                to_node=to_node,
                from_solution=parent_id,
                to_solution=solution.id,
                iteration=solution.iteration,
                cross_branch=not along_tree,
            )
        )

    return edges


def _bands(values: Sequence[float]) -> Tuple[List[float], Dict[float, int]]:
    """Quantile cut points over the node bests, and each best's band.

    `edges[i]` is the lowest score that lands in band `i + 1`, so band 0 is the
    best scores and the page's darkest step. Ties collapse: a population with
    three distinct values gets two edges and three bands rather than five bands
    where two of them can never be reached.
    """
    if not values:
        return [], {}

    ordered = sorted(values)
    cuts: List[float] = []

    for index in range(1, BANDS):
        at = (len(ordered) * index) // BANDS

        if 0 < at < len(ordered):
            cuts.append(ordered[at])

    edges = sorted(set(cuts))

    return edges, {value: bisect.bisect_right(edges, value) for value in ordered}


def _best(solutions: Sequence[Solution]) -> Optional[Solution]:
    scored = [s for s in solutions if s.score is not None]

    if not scored:
        return None

    return min(scored, key=lambda s: cast(float, s.score))
