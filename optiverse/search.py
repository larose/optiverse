"""The search: a director refining a tree of ideas.

The director's whole vocabulary is three fields. It names a node to work under,
names a solution to start the code from, and may add one constraint — which
creates a child node and works there instead. There is no free-form instruction:
if it wants the coding agent to do something, that is a constraint, and a
constraint is a node. So the graph is the complete record of the search rather
than half of it.

`plan.json` is transport. The director is a shell agent, so handing anything back
means writing a file; everything in it lands in `arcs.json` or the solution's
metadata. It is left on disk afterwards because nothing in a run is deleted.

Nothing here keeps state between iterations. The tree comes from `arcs.json`, the
scores from the solutions, and the only thing that survives an agent's turn is
what it wrote to `memory.md`.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

from .director import Director, DirectorContext
from .evaluator import ValidationResult
from .graph import ROOT_NODE_ID, ArcStore, Graph, Node, current_node, similar
from .store import Solution, Store

logger = logging.getLogger(__name__)

ITERATIONS_DIRECTORY_NAME = "iterations"
MEMORY_NAME = "memory.md"
PLAN_NAME = "plan.json"

DIRECTOR_PROMPT_NAME = "director-prompt.md"
DIRECTOR_LOG_NAME = "director.log"
GENERATOR_PROMPT_NAME = "generator-prompt.md"
GENERATOR_LOG_NAME = "generator.log"

# What a set-aside attempt is called: `00079_crashed_1`, beside `00079`.
CRASHED_SUFFIX = "_crashed_"

# Wide enough that names still sort in `ls` past any run length worth having.
# The width has to exceed the ceiling it is chosen for rather than match it: at
# four digits, `10000` would sort ahead of `9999`.
ITERATION_NUMBER_WIDTH = 5

# Two constraints this similar are probably the same idea retyped. Warned about,
# never rejected: sometimes the reword is the point.
REWORD_SIMILARITY = 0.9


@dataclass(frozen=True)
class Plan:
    """What the director decided, once it has been applied.

    `node_id` is where the work happens, which is the child when a constraint was
    added and `parent_node_id` when it was not.
    """

    node_id: str
    parent_solution_id: str
    memory: List[str]


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

        # Held between `decide` and the prompt it goes into. Reset per iteration.
        self._playbook_angle: Optional[str] = None

        (self._directory / ITERATIONS_DIRECTORY_NAME).mkdir(parents=True, exist_ok=True)

    # --- paths --------------------------------------------------------------

    @property
    def memory_path(self) -> Path:
        return self._directory / MEMORY_NAME

    def iteration_directory(self, iteration: int) -> Path:
        name = f"{iteration:0{ITERATION_NUMBER_WIDTH}d}"
        return self._directory / ITERATIONS_DIRECTORY_NAME / name

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

    # --- memory -------------------------------------------------------------

    def remember(self, iteration: int, author: str, text: str) -> None:
        """Append one line to `memory.md`.

        Both agents write here and neither reads what the other wrote, so entries
        repeat. Pruning is the director's job — `memory.md` is the one file
        outside its working directory it is allowed to edit.
        """
        entry = " ".join(text.split())

        if not entry:
            return

        with open(self.memory_path, "a") as memory_file:
            memory_file.write(f"- (iteration {iteration}, {author}) {entry}\n")

    def memory(self) -> str:
        if not self.memory_path.is_file():
            return ""

        return self.memory_path.read_text().strip()

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

        solutions = self._store.get_all_solutions()
        graph = self._graph(solutions)
        focus = current_node(graph, solutions)

        self._playbook_angle = self._angle_for(graph.node(focus))

        prompt = self._prompt(graph, focus, problem_description)
        (directory / DIRECTOR_PROMPT_NAME).write_text(prompt)

        result = self._director.decide(
            DirectorContext(
                log_path=directory / DIRECTOR_LOG_NAME,
                prompt=prompt,
                remember=lambda text: self.remember(iteration, "director", text),
                validate=lambda: self.validate(iteration),
                workdir=directory,
            )
        )

        tags: Dict[str, Union[int, str]] = dict(result.tags)

        if self._playbook_angle is not None:
            tags["playbook_angle"] = _angle_title(self._playbook_angle)

        return SearchResult(plan=self._apply(iteration), tags=tags)

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

        Rejections are things the loop cannot act on. The reword check only
        warns: a constraint that reads like a sibling's is usually a retype, but
        sometimes the difference is the whole point, and code cannot tell which.
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

        problems: List[str] = []

        parent_node_id = fields.get("parent_node_id")
        if not isinstance(parent_node_id, str) or parent_node_id not in graph:
            problems.append(
                f"`parent_node_id` must name a node that exists. "
                f"{ROOT_NODE_ID} always does."
            )
            parent_node_id = None

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
            constraint = None

        problems.extend(_memory_problems(fields))

        if problems:
            return _invalid(problems)

        return _valid(self._reword_warning(graph, parent_node_id, constraint))

    def _reword_warning(
        self,
        graph: Graph,
        parent_node_id: Optional[str],
        constraint: object,
    ) -> List[str]:
        if not isinstance(constraint, str) or parent_node_id is None:
            return []

        existing = [
            child.constraint
            for child in graph.children(parent_node_id)
            if child.constraint is not None
        ]

        if not similar(constraint, existing, REWORD_SIMILARITY):
            return []

        return [
            "This constraint reads like one already on a sibling of "
            f"{parent_node_id}, which would start a second node for the same "
            "idea rather than adding to the one that exists. If you meant to "
            "continue that one, name it in `parent_node_id` and leave "
            "`constraint` out. If the rewording is the point, carry on."
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
            memory=[str(line) for line in cast(List[object], fields.get("memory", []))],
        )

    # --- the playbook --------------------------------------------------------

    def _playbook(self) -> List[str]:
        if not self._playbook_path.is_file():
            return []

        text = self._playbook_path.read_text()
        entries = [block.strip() for block in text.split("\n-") if block.strip()]

        # The first block is the file's own preamble, not an angle.
        return ["- " + entry for entry in entries[1:]]

    def _angle_for(self, node: Node) -> Optional[str]:
        """An angle to push with, once the node the search sits on is exhausted.

        Absent entirely while that node is still paying off — a list of ways to
        leave is noise to a director that should be staying. Chosen at random,
        which costs the run its reproducibility and buys not having to record and
        read back which angles have been used.
        """
        entries = self._playbook()

        if not entries or not node.dead:
            return None

        return random.choice(entries)

    # --- the prompt ----------------------------------------------------------

    def _prompt(self, graph: Graph, focus: str, problem_description: str) -> str:
        node = graph.node(focus)

        sections = [
            "# What you are doing",
            "",
            OPENING,
            "",
            "# The problem",
            "",
            problem_description.strip(),
            "",
            "# The search space",
            "",
            SEARCH_SPACE.format(root=ROOT_NODE_ID),
            "",
            "# The search so far",
            "",
            *(graph.render(focus) or ["Nothing has been tried yet."]),
            "",
            "# Where you are",
            "",
            _where(node),
            "",
            "# What to write",
            "",
            PLAN_INSTRUCTIONS.format(memory=MEMORY_NAME, plan=PLAN_NAME),
            "",
            "# What you have learned",
            "",
            self.memory() or f"`{MEMORY_NAME}` is empty. `remember` starts it.",
            "",
            MEMORY_INSTRUCTIONS.format(memory=MEMORY_NAME),
            "",
            "# Where things are",
            "",
            LAYOUT.format(memory=MEMORY_NAME, plan=PLAN_NAME),
        ]

        if self._playbook_angle is not None:
            sections += [
                "",
                "# Take this angle",
                "",
                f"{node.id} has had {node.stale} attempts without improving on "
                "its own best. Either add a constraint below it, or go somewhere "
                "else in the tree. This angle is worth trying:",
                "",
                self._playbook_angle,
            ]

        return "\n".join(sections) + "\n"


def _where(node: Node) -> str:
    best = "nothing scored yet" if node.best is None else f"best {node.best:.6g}"
    attempts = len(node.solutions)
    plural = "" if attempts == 1 else "s"

    line = f"You are on {node.id} — {attempts} attempt{plural}, {best}, stale {node.stale}."

    if node.dead:
        return (
            f"{line} **This node is done.** Add a constraint below it or go elsewhere."
        )

    return (
        f"{line} It is still paying off, so continue it unless you have a reason "
        "not to."
    )


def _memory_problems(fields: Dict[str, object]) -> List[str]:
    raw = fields.get("memory", [])

    if not isinstance(raw, list):
        return ["`memory`, if given, must be a list of strings."]

    lines = cast(List[object], raw)

    if any(not isinstance(line, str) or not line.strip() for line in lines):
        return ["Every entry in `memory` must be a non-empty string."]

    return []


def _valid(warnings: List[str]) -> ValidationResult:
    return ValidationResult(valid=True, log="\n".join(warnings))


def _invalid(problems: List[str]) -> ValidationResult:
    lines = ["The plan cannot be used yet:", ""]
    lines += [f"- {problem}" for problem in problems]

    return ValidationResult(valid=False, log="\n".join(lines))


def _angle_title(angle: str) -> str:
    """The bolded lead of a playbook entry, for the tag."""
    start = angle.find("**")

    if start < 0:
        return angle.strip().splitlines()[0][:60]

    end = angle.find("**", start + 2)

    return angle[start + 2 : end].strip(" .") if end > 0 else angle[:60]


OPENING = """You are steering an automated search for a better solution to the
problem below. Each iteration, a separate coding agent writes one candidate and
it is scored automatically; lower scores are better.

You decide what that agent works on. You see every score; it sees none, because
ranking is your job and an agent that could see its own score would abandon a
novel approach the moment it looked worse than the incumbent.

You have a shell and the whole run directory is readable. Read whatever you
need — a candidate's source, an agent's trajectory, an earlier plan. Going and
looking is the point of you being an agent rather than a formula."""

SEARCH_SPACE = """A **constraint** is prose telling the coding agent how to
narrow its approach — an instruction it will read, not a label. A sentence is
usual; paragraphs are fine.

A constraint sits on an **arc**. A **node** is everything accumulated from the
root down to it, so a node is an idea and going deeper is committing to one more
thing. `{root}` is the root and has no constraints at all.

You act by naming a node to work under, naming a solution whose code the agent
starts from, and optionally adding one constraint — which creates a child of that
node and works there instead.

**That is the whole of what you can say.** There is no free-form instruction to
the coding agent. If you want it to do something, that is a constraint, and a
constraint is a node — which is what keeps the tree above a complete record of
this search rather than half of one.

The node and the solution are chosen separately and need not match. A node you
have just created holds no solutions at all, and even one that does may not hold
the best code to build on.

Trying the same node several times is normal, not a waste. Scores move run to
run, so one attempt tells you very little about whether a constraint helped —
several turn a point into a spread you can compare against its parent."""

PLAN_INSTRUCTIONS = """Write `{plan}` in your working directory:

```json
{{
  "parent_node_id": "n_…",
  "parent_solution_id": "s_…",
  "constraint": "…",
  "memory": ["…"]
}}
```

- `parent_node_id` — the node to work under. Copy an id from the tree above.
- `parent_solution_id` — the solution whose code the agent starts from. Its files
  are copied into the agent's working directory before it begins, so it opens on
  that code rather than an empty directory.
- `constraint` — optional. Leave it out to try `parent_node_id` again. Give one
  to create a child of `parent_node_id` and work there instead. This is the only
  way a node is ever created.
- `memory` — optional. Lines lifted from `{memory}` that bear on this attempt;
  they are copied into the coding agent's prompt. Facts about the environment
  only — a build rule, something that does not work here. Never an instruction
  about what to build, which is what a constraint is for.

Call `validate` to check it, then `done`."""

MEMORY_INSTRUCTIONS = """`{memory}` is what both you and the coding agents have
learned the hard way. Add to it with `remember`. The coding agents write to it
without ever seeing it, so it repeats itself — pruning is yours, and it is the
one file outside your working directory you may edit.

They never read it either, so anything a coding agent needs to know this time has
to go through `memory` in the plan."""

LAYOUT = """You are in this iteration's own directory. You own `{plan}` in it,
and `../../{memory}`. Read anything else in the run; write nothing else.

- `{plan}` — what you write this iteration.
- `../../{memory}` — what has been learned. Shown above.
- `../../arcs.json` — the tree, as (parent, child, constraint) triplets.
- `../../solutions.csv` — every candidate, best score first.
- `../../solutions/<id>/code/` — a candidate's source.
- `../../solutions/<id>/metadata.json` — its score, metrics and lineage.
- `../<earlier>/generator.log` — what a coding agent did and saw.
- `../<earlier>/director.log` — what you did then."""
