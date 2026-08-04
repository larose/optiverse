"""The search: an iterated local search over a tree of ideas.

Where the search goes is decided by `policy` rather than by a model. What this
module does is carry that decision out — prepare the iteration, ask the director
for a constraint when a perturbation wants one, turn the constraint into an arc,
and say where the work happens.

The director's whole vocabulary is one file. It writes `constraint.md`, and that
becomes a node. There is no free-form instruction to the programmer: if the
director wants something done, that is a constraint, and a constraint is a node —
so the graph is the complete record of this search rather than half of one.
Nothing else the director might have said is stored, because nothing else it
might have said is read back.

What the director is *shown* is `brief.py`, which is a much larger job and
changes for entirely different reasons.

Everything is derived and nothing is maintained. The tree comes from `arcs.json`,
the scores from the solutions, and the sequence from the iteration number the
solutions already carry.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from .brief import compose
from ..director import Director, DirectorContext
from ..evaluator import ValidationResult
from .graph import ArcStore, Graph
from .journal import (
    CONSTRAINT_NAME,
    CRASHED_SUFFIX,
    DIRECTOR_LOG_NAME,
    DIRECTOR_PROMPT_NAME,
    PROGRAMMER_LOG_NAME,
    PROGRAMMER_PROMPT_NAME,
    ITERATIONS_DIRECTORY_NAME,
    Journal,
    iteration_directory,
)
from .policy import Move, Phase, decide
from ..solution import Solution, Store

logger = logging.getLogger(__name__)

DEFAULT_KICKS = Path(__file__).parent / "kicks.md"
"""Ways to make the next constraint unlike the last one.

Named here rather than beside the rest of the configuration because it is a file
this package ships and this package reads. A path spelled out somewhere else is
one that goes stale the next time either end moves.
"""

# A file with no floor under it is satisfied by "ok". The floor is crude and it
# works; anything cleverer — hunting for a verb, say — is taste-policing that a
# good constraint can fail.
MINIMUM_CONSTRAINT_CHARACTERS = 30


@dataclass(frozen=True)
class Perturbation:
    """What came back from asking the director for a constraint."""

    node_id: Optional[str]
    """The node that was minted, or None when the iteration has failed. There is
    deliberately nothing to fall back to: see `perturb`."""

    metrics: Dict[str, Union[int, float]]
    tags: Dict[str, Union[int, str]]


class Search:
    """Runs the policy, and the director when the policy asks for it."""

    def __init__(
        self,
        *,
        directory: Path,
        director: Director,
        kicks: Path,
        store: Store,
    ) -> None:
        self._directory = Path(directory)
        self._director = director
        self._kicks_path = kicks
        self._store = store

        self._arcs = ArcStore(self._directory)

        (self._directory / ITERATIONS_DIRECTORY_NAME).mkdir(parents=True, exist_ok=True)

    # --- paths --------------------------------------------------------------

    def iteration_directory(self, iteration: int) -> Path:
        return iteration_directory(self._directory, iteration)

    def constraint_path(self, iteration: int) -> Path:
        return self.iteration_directory(iteration) / CONSTRAINT_NAME

    def programmer_log_path(self, iteration: int) -> Path:
        return self.iteration_directory(iteration) / PROGRAMMER_LOG_NAME

    def write_programmer_prompt(self, iteration: int, prompt: str) -> None:
        (self.iteration_directory(iteration) / PROGRAMMER_PROMPT_NAME).write_text(
            prompt
        )

    # --- the graph -----------------------------------------------------------

    def constraints(self, node_id: str) -> List[str]:
        """Every constraint in force at a node, root first.

        Read by walking `arcs.json` to the root, so the programmer's brief is
        assembled from the graph rather than from a copy kept alongside it.
        """
        return self._graph(self._store.get_all_solutions()).node(node_id).constraints

    # --- the loop's interface -----------------------------------------------

    def completed_iterations(self) -> int:
        """Where a resumed run picks up.

        Every iteration produces exactly one solution and `metadata.json` is
        written atomically and last, so a committed solution *is* the record that
        its iteration finished. There is no checkpoint file that could disagree.
        """
        iterations = [
            solution.iteration
            for solution in self._store.get_all_solutions()
            if solution.iteration is not None
        ]

        return max(iterations, default=0)

    def begin(self, iteration: int) -> Path:
        """The iteration's directory, empty and ready.

        A directory still here belongs to an attempt nothing got to mark — a
        killed process, rather than a failure the loop saw. It is set aside the
        same way and never written over.
        """
        marked = self.mark_crashed(iteration)

        if marked is not None:
            logger.info(
                f"Iteration {iteration} left a directory behind; it is now "
                f"{marked.name}"
            )

        directory = self.iteration_directory(iteration)
        directory.mkdir(parents=True)

        return directory

    def next_move(self, iteration: int) -> Move:
        """What this iteration does, decided before anything is spent on it.

        Touches nothing, so a preview can ask the same question of a run on disk
        without changing it — and so a crashed iteration, retried, decides the
        same way as the attempt it replaces.
        """
        solutions = self._store.get_all_solutions()

        return decide(
            graph=self._graph(solutions),
            solutions=solutions,
            iteration=iteration,
            kicks=self._kicks(),
        )

    def compose(self, iteration: int, move: Move, problem_description: str) -> str:
        """The prompt this iteration's director would be given.

        Separate from `perturb` because it touches nothing, and a preview that
        prepared an iteration directory would change the run it was only reading.
        Which is the point — the prompt is most of what this project is, and
        iterating on it should not cost two model calls a look.
        """
        solutions = self._store.get_all_solutions()
        graph = self._graph(solutions)

        return compose(
            graph=graph,
            iteration=iteration,
            journal=Journal(
                directory=self._directory,
                graph=graph,
                solutions=solutions,
                before=iteration,
            ),
            move=move,
            problem_description=problem_description,
            solutions=solutions,
        )

    def perturb(
        self, iteration: int, move: Move, problem_description: str
    ) -> Perturbation:
        """Ask the director for a constraint, and hang it under the drawn node.

        A `node_id` of None means the iteration has failed, and there is
        deliberately nothing to fall back to. Substituting "another attempt at
        the same node" gave a byte-identical brief every time the director was
        down, so the loop paid for a full write and score to re-derive what it
        already had, and said nothing louder than a warning.
        """
        directory = self.iteration_directory(iteration)
        prompt = self.compose(iteration, move, problem_description)

        (directory / DIRECTOR_PROMPT_NAME).write_text(prompt)

        result = self._director.decide(
            DirectorContext(
                log_path=directory / DIRECTOR_LOG_NAME,
                prompt=prompt,
                validate=lambda: self.validate(iteration),
                workdir=directory,
            )
        )

        if not self.validate(iteration).valid:
            return Perturbation(
                node_id=None, metrics=dict(result.metrics), tags=dict(result.tags)
            )

        constraint = self.constraint_path(iteration).read_text().strip()

        return Perturbation(
            node_id=self._arcs.add(parent_node_id=move.base.id, constraint=constraint),
            metrics=dict(result.metrics),
            tags=dict(result.tags),
        )

    def mark_crashed(self, iteration: int) -> Optional[Path]:
        """Set this attempt aside under a name that says what it is.

        `00079_crashed_1` sits beside `00079` rather than under a `crashed/`
        subdirectory, so it is impossible to miss in `ls` — which is the whole
        job, since what is in it is the reason to look: the prompt that broke a
        model, a half-written log.

        Returns where it went, or None if the attempt left nothing behind.
        """
        directory = self.iteration_directory(iteration)

        if not directory.exists():
            return None

        attempt = 1
        while self._crashed_path(iteration, attempt).exists():
            attempt += 1

        destination = self._crashed_path(iteration, attempt)
        directory.rename(destination)

        return destination

    def _crashed_path(self, iteration: int, attempt: int) -> Path:
        name = self.iteration_directory(iteration).name
        return (
            self._directory
            / ITERATIONS_DIRECTORY_NAME
            / f"{name}{CRASHED_SUFFIX}{attempt}"
        )

    def _graph(self, solutions: List[Solution]) -> Graph:
        return Graph(self._arcs.read(), solutions)

    # --- the director's validate tool ---------------------------------------

    def validate(self, iteration: int) -> ValidationResult:
        """Whether `constraint.md` is usable, and what is wrong with it.

        Everything it reports is something a loop cannot act on, because a valid
        constraint ends the director's turn — there is nowhere for a remark it
        might have taken or left to go.

        Nothing here refuses a constraint for being repetitive. Since there is no
        fallback, a rejection the director cannot satisfy retries the iteration
        forever, and "say something new" is not a thing code can hand back a
        recipe for. That pressure is in the prompt, where it can be specific.
        """
        path = self.constraint_path(iteration)

        if not path.is_file():
            return _invalid(
                f"`{CONSTRAINT_NAME}` does not exist yet. Write it in your "
                "working directory."
            )

        constraint = path.read_text().strip()

        if not constraint:
            return _invalid(f"`{CONSTRAINT_NAME}` is empty.")

        if len(constraint) < MINIMUM_CONSTRAINT_CHARACTERS:
            return _invalid(
                f"`{CONSTRAINT_NAME}` is {len(constraint)} characters. A "
                "constraint that short is not an instruction. Say what you "
                "actually want narrowed — the programmer has none of your "
                "context and this is all it gets."
            )

        return ValidationResult(valid=True, log="")

    # --- the kicks -----------------------------------------------------------

    def _kicks(self) -> List[str]:
        """The entries in `kicks.md`, one per angle.

        Read on every call rather than cached, so editing the file mid-run takes
        effect on the next perturbation. Which one is drawn is `policy`'s job,
        not this one's.
        """
        if not self._kicks_path.is_file():
            return []

        text = self._kicks_path.read_text()
        entries = [block.strip() for block in text.split("\n-") if block.strip()]

        # The first block is the file's own preamble, not a kick.
        return ["- " + entry for entry in entries[1:]]


def _invalid(problem: str) -> ValidationResult:
    return ValidationResult(valid=False, log=f"It cannot be used yet: {problem}")


__all__ = [
    "DEFAULT_KICKS",
    "MINIMUM_CONSTRAINT_CHARACTERS",
    "Move",
    "Perturbation",
    "Phase",
    "Search",
]
