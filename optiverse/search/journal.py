"""The run as a sequence rather than a shape.

`arcs.json` says what the search space is. It does not say what the search did:
which node was worked when, whether the last eight moves all landed in one
subtree, whether anything has beaten the incumbent this hour. A tree read without
that is read the same way twice in a row, which is what a loop is.

Nothing is stored for this and nothing needs to be. Every solution carries its
node and its iteration, every constraint is an arc, and a perturbation mints a
node and works it in the same iteration — so the phase, the constraint and the
node a perturbation was hung under are all readable off the two files the run
already keeps. There was once a `plan.json` beside each iteration holding the
same facts; it existed only because the director used to *choose* them, and it
went when the choosing did.

That leaves this module a function of the solutions and the tree, with one
directory listing in it: the crashed-attempt count, which lives in the names of
the directories themselves and nowhere else.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, cast

from .graph import ROOT_NODE_ID, Graph
from .policy import Phase, phase_of
from ..solution import Solution

ITERATIONS_DIRECTORY_NAME = "iterations"

# The director's whole output. A constraint is prose, so it is a prose file.
CONSTRAINT_NAME = "constraint.md"

# What each agent was shown and what it did. Named here with the rest of the
# layout rather than beside the code that writes them, because the prompt tells
# the director to go and read them — so the name has two readers, and a literal
# spelled out at the second one is a link that breaks silently.
DIRECTOR_PROMPT_NAME = "director-prompt.md"
DIRECTOR_LOG_NAME = "director.log"
PROGRAMMER_PROMPT_NAME = "programmer-prompt.md"
PROGRAMMER_LOG_NAME = "programmer.log"

# What a set-aside attempt is called: `00079_crashed_1`, beside `00079`.
CRASHED_SUFFIX = "_crashed_"

# Wide enough that names still sort in `ls` past any run length worth having.
# The width has to exceed the ceiling it is chosen for rather than match it: at
# four digits, `10000` would sort ahead of `9999`.
ITERATION_NUMBER_WIDTH = 5

# One line each, newest last. Twenty is about a screen, and it reaches back past
# any single idea's run of attempts — which is the span a loop hides in.
LEDGER_LENGTH = 20

# The whole run, however long, in this many lines. A line count rather than a
# block size is what keeps the section the same height at iteration 40 and at
# iteration 4000.
EPOCH_LINES = 10

# The span the stagnation signals are measured over. Four times the patience a
# node is given: long enough that a node taking its three fair samples does not
# read as a rut, short enough that a rut is caught while there is still budget to
# spend differently.
WINDOW = 12

# Constraint text in a one-line ledger entry. Long enough to recognise an idea,
# short enough that the score stays in the same column.
_CONSTRAINT_WIDTH = 44


def iteration_name(iteration: int) -> str:
    return f"{iteration:0{ITERATION_NUMBER_WIDTH}d}"


def iterations_directory(directory: Path) -> Path:
    return Path(directory) / ITERATIONS_DIRECTORY_NAME


def iteration_directory(directory: Path, iteration: int) -> Path:
    return iterations_directory(directory) / iteration_name(iteration)


@dataclass(frozen=True)
class Move:
    """One iteration: what the search did, and what came back.

    Every field is read off the solution and the tree. `constraint` and
    `base_node_id` are set only on a perturbation, and both come from the arc
    that perturbation created — the constraint is what the arc carries, and the
    base is the arc's parent, since a perturbation hangs a *child* under it.
    """

    iteration: int
    phase: Phase
    solution: Solution

    constraint: Optional[str]
    base_node_id: Optional[str]

    crashes: int
    """Attempts at this iteration that were set aside before one landed. An
    iteration that needed three goes is worth knowing about, and nothing else in
    the run says so."""

    improved_run_best: bool
    run_best_after: Optional[float]

    @property
    def node_id(self) -> str:
        """Where the work happened."""
        return self.solution.node_id

    @property
    def score(self) -> Optional[float]:
        return self.solution.score


@dataclass(frozen=True)
class Progress:
    """The run's own numbers, as distinct from any one node's."""

    iterations: int

    best: Optional[Solution]
    best_iteration: Optional[int]
    """None while the only thing scored is the seed, which no iteration made."""

    drought: int
    """Iterations since the run's best score improved. An iteration that did not
    score counts as one that did not improve, because it did not."""

    window_created: int
    """How many nodes first appeared in the last `WINDOW` iterations. Zero means
    the search spent that window re-sampling ideas it already had."""

    window_worked: int
    """How many distinct nodes those iterations touched."""


class Journal:
    """What the search has been doing, derived from the run directory."""

    def __init__(
        self,
        *,
        directory: Path,
        graph: Graph,
        solutions: Sequence[Solution],
        before: int,
    ) -> None:
        self._directory = Path(directory)
        self._graph = graph

        # `before` is the iteration being planned. Nothing has landed for it yet,
        # so bounding the read is clearer than tolerating a gap.
        self._history = sorted(
            (s for s in solutions if s.iteration is not None and s.iteration < before),
            key=lambda s: cast(int, s.iteration),
        )

        # The seed is the incumbent until something beats it. It has no
        # iteration, so it is not in the history — but leaving it out of the
        # baseline would say "nothing has scored yet" while a score sits at the
        # root, and would let the first candidate that scored anything at all
        # read as an improvement.
        self._seed = _best_solution(s for s in solutions if s.iteration is None)
        self._seed_best = self._seed.score if self._seed else None

        self._first_seen = _first_seen(self._history)
        self._moves = self._read(self._history)

    # --- what happened -------------------------------------------------------

    def moves(self, count: int = LEDGER_LENGTH) -> List[Move]:
        """The most recent moves, oldest first."""
        return self._moves[-count:] if count > 0 else []

    def last(self) -> Optional[Move]:
        return self._moves[-1] if self._moves else None

    def progress(self) -> Progress:
        running = self._seed_best
        best: Optional[Solution] = self._seed
        best_iteration: Optional[int] = None
        drought = 0

        for solution in self._history:
            score = solution.score

            if score is not None and (running is None or score < running):
                running = score
                best = solution
                best_iteration = solution.iteration
                drought = 0
            else:
                drought += 1

        window = self._window()

        return Progress(
            iterations=len(self._history),
            best=best,
            best_iteration=best_iteration,
            drought=drought,
            window_created=sum(
                1
                for node_id, first in self._first_seen.items()
                if first in window and node_id != ROOT_NODE_ID
            ),
            window_worked=len({s.node_id for s in self._history[-WINDOW:]}),
        )

    def node_ids(self, count: int = WINDOW) -> List[str]:
        """The nodes the last `count` iterations landed on, newest first.

        The prompt anchors the tree on these: where the search has been is where
        the alternatives to it are worth seeing.
        """
        seen: List[str] = []

        for solution in reversed(self._history[-count:]):
            if solution.node_id not in seen:
                seen.append(solution.node_id)

        return seen

    # --- rendering -----------------------------------------------------------

    def render_epochs(self) -> List[str]:
        """The whole run in at most `EPOCH_LINES` lines.

        Coarse enough to stay one height forever, and the only place a director
        at iteration 800 can see that the first hundred iterations moved the
        score twice as far as the last four hundred.
        """
        if not self._history:
            return []

        total = cast(int, self._history[-1].iteration)
        size = max(1, math.ceil(total / EPOCH_LINES))

        by_epoch: Dict[int, List[Solution]] = {}
        for solution in self._history:
            by_epoch.setdefault((cast(int, solution.iteration) - 1) // size, []).append(
                solution
            )

        created: Dict[int, int] = {}
        for node_id, first in self._first_seen.items():
            if node_id != ROOT_NODE_ID:
                epoch = (first - 1) // size
                created[epoch] = created.get(epoch, 0) + 1

        lines: List[str] = []
        running = self._seed_best

        for epoch in range((total - 1) // size + 1):
            opening = running
            running = _best_score(by_epoch.get(epoch, []), running)

            start = epoch * size + 1
            end = min(start + size - 1, total)
            made = created.get(epoch, 0)

            span = f"i{start}–{end}" if end > start else f"i{start}"
            plural = "" if made == 1 else "s"
            moved = (
                f"best {_score(opening)} → {_score(running)}"
                if running != opening
                else f"best {_score(running)}, unmoved"
            )

            lines.append(f"  {span:<12}  {made} node{plural} created   {moved}")

        return lines

    def render_ledger(self, count: int = LEDGER_LENGTH) -> List[str]:
        """One line per iteration, oldest first.

        The phase is named on every line rather than left to be inferred from a
        node id that changed. A run of local searches under one node is what
        progress looks like when it is working and what a rut looks like when it
        is not, and either way it should be countable at a glance.
        """
        lines: List[str] = []

        for move in self.moves(count):
            marker = "*" if move.improved_run_best else " "

            if move.phase is Phase.PERTURB:
                action = (
                    f"perturb under {move.base_node_id}: "
                    f'"{_short(move.constraint or "")}"'
                )
            else:
                action = "local search"

            crashed = ""
            if move.crashes:
                plural = "" if move.crashes == 1 else "s"
                crashed = f"   ({move.crashes} crashed attempt{plural} first)"

            lines.append(
                f"  {move.iteration:>5}  {_score(move.score):>10}  {marker}  "
                f"{move.node_id}  {action}{crashed}"
            )

        return lines

    # --- reading -------------------------------------------------------------

    def _read(self, solutions: Sequence[Solution]) -> List[Move]:
        crashes = self._crash_counts()
        running = self._seed_best

        # Replayed over the whole history rather than a window, so the first move
        # shown knows whether it improved on everything before it.
        improved: Dict[int, bool] = {}
        after: Dict[int, Optional[float]] = {}

        for solution in self._history:
            iteration = cast(int, solution.iteration)
            score = solution.score

            improved[iteration] = score is not None and (
                running is None or score < running
            )

            if improved[iteration]:
                running = score

            after[iteration] = running

        moves: List[Move] = []

        for solution in solutions:
            iteration = cast(int, solution.iteration)
            phase = phase_of(solution, self._first_seen)
            node = (
                self._graph.node(solution.node_id)
                if solution.node_id in self._graph
                else None
            )

            moves.append(
                Move(
                    iteration=iteration,
                    phase=phase,
                    solution=solution,
                    constraint=(
                        node.constraint if node and phase is Phase.PERTURB else None
                    ),
                    base_node_id=(
                        node.parent_node_id if node and phase is Phase.PERTURB else None
                    ),
                    crashes=crashes.get(iteration, 0),
                    improved_run_best=improved[iteration],
                    run_best_after=after[iteration],
                )
            )

        return moves

    def _crash_counts(self) -> Dict[int, int]:
        """How many attempts each iteration number needed, from the names.

        One listing rather than a stat per iteration, and the names carry the
        whole answer — which is the point of `898167f` putting the attempt number
        in them rather than in a file.
        """
        counts: Dict[int, int] = {}
        directory = iterations_directory(self._directory)

        if not directory.is_dir():
            return counts

        for entry in directory.iterdir():
            name, _, attempt = entry.name.partition(CRASHED_SUFFIX)

            if attempt and name.isdigit():
                counts[int(name)] = counts.get(int(name), 0) + 1

        return counts

    def _window(self) -> Set[int]:
        return {cast(int, s.iteration) for s in self._history[-WINDOW:]}


def _first_seen(solutions: Sequence[Solution]) -> Dict[str, int]:
    """The iteration each node first held a solution.

    A perturbation mints a node and works it in the same iteration, so a node's
    earliest solution both dates it and marks which attempt was the perturbation.
    Nodes minted by an attempt that then crashed hold none and are absent, which
    is right: an idea nothing was ever built for was not tried.
    """
    first: Dict[str, int] = {}

    for solution in solutions:
        iteration = cast(int, solution.iteration)
        first.setdefault(solution.node_id, iteration)

    return first


def _best_score(
    solutions: Iterable[Solution], running: Optional[float] = None
) -> Optional[float]:
    scores = [s.score for s in solutions if s.score is not None]

    if running is not None:
        scores.append(running)

    return min(scores) if scores else None


def _best_solution(solutions: Iterable[Solution]) -> Optional[Solution]:
    scored = [s for s in solutions if s.score is not None]

    if not scored:
        return None

    return min(scored, key=lambda s: cast(float, s.score))


def _score(score: Optional[float]) -> str:
    return "failed" if score is None else f"{score:.6g}"


def _short(text: str, width: int = _CONSTRAINT_WIDTH) -> str:
    collapsed = " ".join(text.split())

    return collapsed if len(collapsed) <= width else collapsed[: width - 1] + "…"
