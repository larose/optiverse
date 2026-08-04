"""The search: a director refining a tree of ideas.

The director's whole vocabulary is three fields. It names a node to work under,
names a solution to start the code from, and may add one constraint — which
creates a child node and works there instead. There is no free-form instruction:
if it wants the coding agent to do something, that is a constraint, and a
constraint is a node. So the graph is the complete record of the search rather
than half of it.

Two things the director writes now outlive its turn, and both are prose. In
`plan.json`, beside the decision, it records what it expected — read back to the
next director with the score beside it, which is the difference between a search
that measures and one that only moves. In `memory.md` it keeps its model of the
problem, rewritten rather than appended, which is the only place understanding
accumulates. Neither reaches a coding agent: `memory.md` is the director's alone,
and an agent that could read what things scored is the one thing this design will
not have.

What this module does is run the director and file what it decided. What the
director is *shown* is `brief.py`, which is a much larger job and changes for
entirely different reasons.

Everything else is still derived and nothing is maintained. The tree comes from
`arcs.json`, the scores from the solutions, and the sequence from joining the
two on the iteration number they both already carry.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

from .brief import Stage, Turn, compose
from ..director import Director, DirectorContext
from ..evaluator import ValidationResult
from .graph import ROOT_NODE_ID, ArcStore, Graph
from .journal import (
    CRASHED_SUFFIX,
    DIRECTOR_LOG_NAME,
    DIRECTOR_PROMPT_NAME,
    GENERATOR_LOG_NAME,
    GENERATOR_PROMPT_NAME,
    ITERATIONS_DIRECTORY_NAME,
    MEMORY_NAME,
    PLAN_NAME,
    Journal,
    iteration_directory,
)
from ..solution import Solution, Store

logger = logging.getLogger(__name__)

DEFAULT_PLAYBOOK = Path(__file__).parent / "playbook.md"
"""Angles for inventing a constraint the search has not tried.

Named here rather than beside the rest of the configuration because it is a file
this package ships and this package reads. A path spelled out somewhere else is
one that goes stale the next time either end moves.
"""

# A required string field with no floor under it is satisfied by "ok". The floor
# is crude and it works; anything cleverer — hunting for a number in an
# expectation, say — is taste-policing that a good prediction can fail.
MINIMUM_PROSE_CHARACTERS = 30


@dataclass(frozen=True)
class Plan:
    """What the director decided, once it has been applied.

    `node_id` is where the work happens, which is the child when a constraint was
    added and `parent_node_id` when it was not.
    """

    node_id: str
    parent_solution_id: str


@dataclass(frozen=True)
class SearchResult:
    plan: Optional[Plan]
    """What to build, or None when the iteration has failed. There is no
    fallback plan: see `_apply`."""

    tags: Dict[str, Union[int, str]]


class Search:
    """Runs the director and applies what it decided."""

    def __init__(
        self,
        *,
        directory: Path,
        director: Director,
        playbook: Path,
        store: Store,
    ) -> None:
        self._directory = Path(directory)
        self._director = director
        self._playbook_path = playbook
        self._store = store

        self._arcs = ArcStore(self._directory)
        self._turn: Optional[Turn] = None

        (self._directory / ITERATIONS_DIRECTORY_NAME).mkdir(parents=True, exist_ok=True)

    # --- paths --------------------------------------------------------------

    @property
    def memory_path(self) -> Path:
        return self._directory / MEMORY_NAME

    def iteration_directory(self, iteration: int) -> Path:
        return iteration_directory(self._directory, iteration)

    def plan_path(self, iteration: int) -> Path:
        return self.iteration_directory(iteration) / PLAN_NAME

    def generator_log_path(self, iteration: int) -> Path:
        return self.iteration_directory(iteration) / GENERATOR_LOG_NAME

    def write_generator_prompt(self, iteration: int, prompt: str) -> None:
        (self.iteration_directory(iteration) / GENERATOR_PROMPT_NAME).write_text(prompt)

    # --- the graph -----------------------------------------------------------

    def constraints(self, node_id: str) -> List[str]:
        """Every constraint in force at a node, root first.

        Read by walking `arcs.json` to the root, so the coding agent's brief is
        assembled from the graph rather than from a copy kept alongside it.
        """
        return self._graph(self._store.get_all_solutions()).node(node_id).constraints

    # --- memory --------------------------------------------------------------

    def memory(self) -> Optional[str]:
        """The director's notebook, or None if it has never written one.

        A notebook that exists and says nothing is not the same as no notebook:
        the first is a director that had nothing to add, the second is a run that
        has not started thinking yet, and only the second calls a review.
        """
        if not self.memory_path.is_file():
            return None

        return self.memory_path.read_text()

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

    def decide(self, iteration: int, problem_description: str) -> SearchResult:
        """Ask the director what to try next, and apply it."""
        directory = self._prepare(iteration)

        prompt, turn = self.compose(iteration, problem_description)

        self._turn = turn
        (directory / DIRECTOR_PROMPT_NAME).write_text(prompt)

        result = self._director.decide(
            DirectorContext(
                log_path=directory / DIRECTOR_LOG_NAME,
                prompt=prompt,
                validate=lambda: self.validate(iteration),
                workdir=directory,
            )
        )

        tags: Dict[str, Union[int, str]] = dict(result.tags)
        tags["stage"] = turn.stage.value

        if turn.reviewing is not None:
            tags["review"] = turn.reviewing

        return SearchResult(plan=self._apply(iteration), tags=tags)

    def compose(self, iteration: int, problem_description: str) -> Tuple[str, Turn]:
        """The prompt an iteration would be given, and what it was built from.

        Separate from `decide` because it touches nothing: `decide` prepares the
        iteration's directory first, and a preview that did the same would change
        the run it was only reading. Which is the point — the prompt is most of
        what this project is, and iterating on it should not cost two model calls
        a look.
        """
        solutions = self._store.get_all_solutions()

        return compose(
            graph=self._graph(solutions),
            iteration=iteration,
            journal=Journal(
                directory=self._directory, solutions=solutions, before=iteration
            ),
            memory=self.memory(),
            playbook=self._playbook(),
            problem_description=problem_description,
            solutions=solutions,
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

    def _prepare(self, iteration: int) -> Path:
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

    def _graph(self, solutions: List[Solution]) -> Graph:
        return Graph(self._arcs.read(), solutions)

    # --- the director's validate tool ---------------------------------------

    def validate(self, iteration: int) -> ValidationResult:
        """Whether `plan.json` is usable, and what is wrong with it.

        Everything it reports is something a loop cannot act on, because a valid
        plan ends the director's turn — there is nowhere for a remark it might
        have taken or left to go. The prose fields are here on that footing too:
        an expectation nobody wrote is not a plan *this* iteration cannot use, it
        is one the next iteration cannot, which is the same thing a turn later.

        Nothing here refuses a plan for being repetitive. Since there is no
        fallback, a rejection the director cannot satisfy retries the iteration
        forever, and "go somewhere else" is not a thing code can hand back a
        recipe for. That pressure is in the prompt, where it can be specific.
        """
        path = self.plan_path(iteration)

        if not path.is_file():
            return _invalid([f"{PLAN_NAME} does not exist yet."])

        try:
            raw = cast(object, json.loads(path.read_text()))
        except json.JSONDecodeError as error:
            return _invalid([f"{PLAN_NAME} is not valid JSON: {error}"])

        if not isinstance(raw, dict):
            return _invalid([f"{PLAN_NAME} must hold a JSON object."])

        fields = cast(Dict[str, object], raw)
        solutions = self._store.get_all_solutions()
        graph = self._graph(solutions)
        turn = self._turn if self._turn and self._turn.iteration == iteration else None

        problems: List[str] = []

        parent_node_id = fields.get("parent_node_id")
        if not isinstance(parent_node_id, str) or parent_node_id not in graph:
            problems.append(
                f"`parent_node_id` must name a node that exists. "
                f"{ROOT_NODE_ID} always does."
            )

        parent_solution_id = fields.get("parent_solution_id")
        if not isinstance(parent_solution_id, str) or parent_solution_id not in {
            solution.id for solution in solutions
        }:
            problems.append("`parent_solution_id` must name a solution that exists.")

        constraint = fields.get("constraint")
        if constraint is not None and (
            not isinstance(constraint, str) or not constraint.strip()
        ):
            problems.append(
                "`constraint`, if given, must be a non-empty string. Leave it out "
                "to try the same node again."
            )

        reconciling = turn.reconciling if turn else iteration > 1
        problems.extend(_prose_problems(fields, reconciling=reconciling))

        if turn is not None and turn.reviewing is not None:
            problems.extend(self._memory_problems(iteration))

        if problems:
            return _invalid(problems)

        return ValidationResult(valid=True, log="")

    def _memory_problems(self, iteration: int) -> List[str]:
        if (self.iteration_directory(iteration) / MEMORY_NAME).is_file():
            return []

        return [
            f"This iteration is a review, so it has to leave a `{MEMORY_NAME}` in "
            "your working directory — it is copied to the run and it is what you "
            "will be shown next time. Start from the one quoted above, edit what "
            "is now wrong, and write the whole file back."
        ]

    # --- applying the plan ---------------------------------------------------

    def _apply(self, iteration: int) -> Optional[Plan]:
        """Turn a validated plan into a node to work under and code to start from.

        `None` means the iteration has failed, and there is deliberately nothing
        to fall back to. Substituting "another attempt at the best solution" gave
        a byte-identical brief every time the director was down, so the loop paid
        for a full generation and score to re-derive what it already had, and
        said nothing louder than a warning.
        """
        if not self.validate(iteration).valid:
            return None

        fields = cast(
            Dict[str, object], json.loads(self.plan_path(iteration).read_text())
        )

        self._carry_memory(iteration)

        parent_node_id = cast(str, fields["parent_node_id"])
        constraint = fields.get("constraint")

        node_id = parent_node_id

        if isinstance(constraint, str):
            node_id = self._arcs.add(
                parent_node_id=parent_node_id, constraint=constraint.strip()
            )

        return Plan(
            node_id=node_id,
            parent_solution_id=cast(str, fields["parent_solution_id"]),
        )

    def _carry_memory(self, iteration: int) -> None:
        """Promote this iteration's notebook to the run's, if it wrote one.

        The director writes into its own directory like everything else it
        writes, and the copy happens only once the plan is good — so an attempt
        that crashed cannot leave the run holding half a thought, and
        `iterations/*/memory.md` is a history of the model rather than a single
        file overwritten in place.
        """
        source = self.iteration_directory(iteration) / MEMORY_NAME

        if source.is_file():
            shutil.copyfile(source, self.memory_path)

    # --- the playbook --------------------------------------------------------

    def _playbook(self) -> List[str]:
        """The angles, all of them.

        Shown whole and only when the search is stuck, rather than one drawn at
        random whenever a node went quiet. Sampling cost the run its
        reproducibility and bought nothing: with no record of what had been
        shown, the same angle could come up for the rest of a thousand
        iterations, and twelve short entries are cheaper than the machinery for
        remembering which eleven were skipped.
        """
        if not self._playbook_path.is_file():
            return []

        text = self._playbook_path.read_text()
        entries = [block.strip() for block in text.split("\n-") if block.strip()]

        # The first block is the file's own preamble, not an angle.
        return ["- " + entry for entry in entries[1:]]


# --- validation helpers ------------------------------------------------------


def _prose_problems(fields: Dict[str, object], *, reconciling: bool) -> List[str]:
    """The three fields that are only ever read by the next iteration.

    `verdict` is asked for only when there is something to reconcile. Demanding
    one on the first iteration would teach the model that the field is decorative,
    which is the one thing that would make all three worthless.
    """
    required = ["reasoning", "expectation"]

    if reconciling:
        required.insert(0, "verdict")

    problems: List[str] = []

    for name in required:
        value = fields.get(name)

        if not isinstance(value, str) or not value.strip():
            problems.append(f"`{name}` is required, and it is prose, not a label.")
        elif len(value.strip()) < MINIMUM_PROSE_CHARACTERS:
            problems.append(
                f"`{name}` is {len(value.strip())} characters. A field that short "
                "is not a thought. Say what you actually mean — it is read by "
                "whoever runs next, and they have none of your context."
            )

    return problems


def _invalid(problems: List[str]) -> ValidationResult:
    lines = ["The plan cannot be used yet:", ""]
    lines += [f"- {problem}" for problem in problems]

    return ValidationResult(valid=False, log="\n".join(lines))


__all__ = [
    "DEFAULT_PLAYBOOK",
    "MINIMUM_PROSE_CHARACTERS",
    "Plan",
    "Search",
    "SearchResult",
    "Stage",
    "Turn",
]
