"""The search: a strategist refining a tree of constraints.

A **constraint** is prose that narrows the search space, written to be read by the
coding agent that will work under it. A **branch** is the list of constraints in
force, and the list *is* the branch: there are no branch ids, no parent pointers
and no stored edges, because structure is subset inclusion. The root branch is the
empty list — no constraints, the whole space.

Going deeper appends a constraint. Changing your mind means dropping back to a
subset and turning elsewhere. Combining two lines of work needs no machinery at
all: a branch whose constraints contain those of two visited branches *is* that
combination.

Two files hold the search's memory, and they are separate because their write
semantics differ, which is what makes the first one trustworthy:

    journal.jsonl   append-only, written here. What happened.
    knowledge.md    rewritten by the strategist. What it concluded.

The strategist cannot revise the record, only its own reading of it. Two more
files are not memory: `plan.json` is the strategist's return value, emptied every
iteration, and `notebook.md` is the digest this module has to build for the prompt
anyway, saved because it is the only human-legible view of the search.
"""

import difflib
import json
import logging
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

from .evaluator import ValidationResult
from .store import Solution, Store
from .strategist import Strategist, StrategistContext, StrategistResult

logger = logging.getLogger(__name__)

STRATEGIST_DIRECTORY_NAME = "strategist"
JOURNAL_NAME = "journal.jsonl"
KNOWLEDGE_NAME = "knowledge.md"
NOTEBOOK_NAME = "notebook.md"
PLAN_NAME = "plan.json"
LOGS_DIRECTORY_NAME = "logs"

# Wide enough that names still sort in `ls` past any run length worth having.
# The width has to exceed the ceiling it is chosen for rather than match it: at
# four digits, `10000.log` would sort ahead of `9999.log`.
LOG_NUMBER_WIDTH = 5

MAXIMUM_PARENTS = 3
MAXIMUM_KNOWLEDGE_CHARACTERS = 10_000

# How many iterations without a global improvement before the strategist is made
# to justify staying where it is, and handed a playbook angle to take instead.
DEFAULT_STAGNATION_LIMIT = 6

# How much of the journal goes into the digest verbatim. Older iterations are
# still on disk, and the strategist has a shell to read them with.
RECENT_JOURNAL_ENTRIES = 8

# Two constraint sets this similar are probably the same idea retyped. Warned
# about, never rejected: sometimes the reword is the point.
REWORD_SIMILARITY = 0.9

FALLBACK_TASK = "Make a focused improvement to the parent solution."


@dataclass(frozen=True)
class SolutionWithTitle:
    solution: Solution
    title: str


@dataclass(frozen=True)
class SearchResult:
    solutions: List[SolutionWithTitle]
    tags: Dict[str, Union[int, str]]
    task: str


@dataclass(frozen=True)
class Plan:
    constraints: List[str]
    note: Optional[str]
    parent_solution_ids: List[str]
    parent_titles: List[str]
    task: str


def normalize_constraint(constraint: str) -> str:
    """The form two constraints are compared in.

    Trailing whitespace and the number of blank lines between paragraphs carry no
    meaning, and letting them carry identity would fork a branch every time the
    strategist reflowed a paragraph. What gets stored and shown is the text as
    written; only the comparison is normalised.
    """
    lines = [line.rstrip() for line in constraint.strip().splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def branch_key(constraints: Sequence[str]) -> Tuple[str, ...]:
    """A branch's identity: its constraints as a set, normalised and ordered.

    Sorted because the branch is the *set* of constraints in force — the order the
    strategist happened to list them in is not a different branch.
    """
    return tuple(sorted({normalize_constraint(c) for c in constraints}))


@dataclass(frozen=True)
class JournalEntry:
    constraints: List[str]
    ended_at: str
    exit_status: str
    iteration: int
    metrics: Dict[str, Union[int, float]]
    note: Optional[str]
    parent_solution_ids: List[str]
    playbook_entry: Optional[str]
    score: Optional[float]
    solution_id: str
    started_at: str
    task: str

    @property
    def key(self) -> Tuple[str, ...]:
        return branch_key(self.constraints)


@dataclass(frozen=True)
class Branch:
    """One visited constraint set, with everything the journal says about it."""

    constraints: List[str]
    entries: List[JournalEntry]

    @property
    def key(self) -> Tuple[str, ...]:
        return branch_key(self.constraints)

    @property
    def scores(self) -> List[float]:
        return sorted(e.score for e in self.entries if e.score is not None)

    @property
    def last_iteration(self) -> int:
        return max(e.iteration for e in self.entries)


class Search:
    """Runs the strategist and turns its plan into the next candidate's brief."""

    def __init__(
        self,
        *,
        directory: Path,
        playbook: Path,
        stagnation_limit: int = DEFAULT_STAGNATION_LIMIT,
        store: Store,
        strategist: Strategist,
    ) -> None:
        self._directory = Path(directory) / STRATEGIST_DIRECTORY_NAME
        self._playbook_path = playbook
        self._stagnation_limit = stagnation_limit
        self._store = store
        self._strategist = strategist

        # Which angle this iteration was pushed with, if any. Held between
        # `decide` and `record` because the journal line is written once, after
        # the candidate has been scored.
        self._playbook_entry: Optional[str] = None

        self._directory.mkdir(parents=True, exist_ok=True)

    # --- paths -------------------------------------------------------------

    @property
    def journal_path(self) -> Path:
        return self._directory / JOURNAL_NAME

    @property
    def knowledge_path(self) -> Path:
        return self._directory / KNOWLEDGE_NAME

    @property
    def notebook_path(self) -> Path:
        return self._directory / NOTEBOOK_NAME

    @property
    def plan_path(self) -> Path:
        return self._directory / PLAN_NAME

    def log_path(self, iteration: int) -> Path:
        """The strategist's trajectory for one iteration.

        Under `logs/` rather than loose in the working directory, which the
        strategist itself lists every iteration: a hundred of these would bury the
        four files it actually reads and writes.
        """
        name = f"{iteration:0{LOG_NUMBER_WIDTH}d}.log"
        return self._directory / LOGS_DIRECTORY_NAME / name

    # --- the loop's interface ----------------------------------------------

    def completed_iterations(self) -> int:
        """Where a resumed run picks up.

        The journal is the record of what finished, so there is no checkpoint file
        to keep in step with it — and no chance of the two disagreeing about how
        far the run got.
        """
        return len(self._read_journal())

    def decide(self, iteration: int, problem_description: str) -> SearchResult:
        """Ask the strategist what to try next, and shape it into a brief."""
        journal = self._read_journal()
        solutions = self._store.get_all_solutions()

        self._playbook_entry = self._stagnation_angle(journal)

        self.notebook_path.write_text(self._digest(journal))

        # Cleared first: a stale plan from last iteration would otherwise pass
        # validation untouched and be silently re-used.
        self.plan_path.unlink(missing_ok=True)

        result = self._strategist.decide(
            StrategistContext(
                log_path=self.log_path(iteration),
                prompt=self._prompt(problem_description, self._playbook_entry),
                validate=self.validate,
                workdir=self._directory,
            )
        )

        plan = self._read_plan()

        if plan is None:
            return self._fallback(solutions, result)

        return SearchResult(
            solutions=self._parents(plan, solutions),
            tags=dict(result.tags),
            task=compose_task(plan.constraints, plan.task),
        )

    def record(self, iteration: int, solution: Solution) -> None:
        """Append what the iteration actually did, plan and outcome together.

        The plan is written before the candidate exists and the score arrives
        after it, so the line can only be written here, once both are known. That
        is also why `plan.json` is not the record: it is a mailbox, emptied every
        iteration, and this is the thing that keeps.

        A `note` stays on the line of the iteration it was written during, which
        is the one *after* the branch it judges. Moving it back a line would read
        marginally better and would mean rewriting history in an append-only file,
        which is the property that makes this trustworthy at all.
        """
        plan = self._read_plan()

        entry = {
            "constraints": plan.constraints if plan else [],
            "ended_at": solution.ended_at,
            "exit_status": str(solution.tags.get("exit_status", "")),
            "iteration": iteration,
            "metrics": solution.metrics,
            "note": plan.note if plan else None,
            "parent_solution_ids": plan.parent_solution_ids if plan else [],
            "playbook_entry": self._playbook_entry,
            "score": solution.score,
            "solution_id": solution.id,
            "started_at": solution.started_at,
            "task": plan.task if plan else FALLBACK_TASK,
        }

        with open(self.journal_path, "a") as journal_file:
            journal_file.write(json.dumps(entry) + "\n")

        self._playbook_entry = None

    # --- the strategist's validate tool ------------------------------------

    def validate(self) -> ValidationResult:
        """Whether `plan.json` and `knowledge.md` are usable, and what is wrong.

        Rejections are things the loop cannot act on. The reword check only warns:
        a constraint that resembles a visited one is usually a retype, but
        sometimes the difference is the whole point, and code cannot tell which.
        """
        problems: List[str] = []
        warnings: List[str] = []

        problems.extend(self._knowledge_problems())

        if not self.plan_path.is_file():
            problems.append(f"{PLAN_NAME} does not exist yet.")
            return _validation(problems, warnings)

        try:
            raw = cast(object, json.loads(self.plan_path.read_text()))
        except json.JSONDecodeError as error:
            problems.append(f"{PLAN_NAME} is not valid JSON: {error}")
            return _validation(problems, warnings)

        if not isinstance(raw, dict):
            problems.append(f"{PLAN_NAME} must hold a JSON object.")
            return _validation(problems, warnings)

        fields = cast(Dict[str, object], raw)
        solutions = self._store.get_all_solutions()

        constraints = self._constraint_problems(fields, problems)
        self._parent_problems(fields, solutions, problems)

        task = fields.get("task")
        if not isinstance(task, str) or not task.strip():
            problems.append("`task` must be a non-empty string.")

        note = fields.get("note")
        if note is not None and not isinstance(note, str):
            problems.append("`note`, if given, must be a string.")

        warnings.extend(self._reword_warnings(constraints))

        return _validation(problems, warnings)

    # --- validation pieces --------------------------------------------------

    def _knowledge_problems(self) -> List[str]:
        """Keep `knowledge.md` from turning into a second journal.

        It is meant to hold what is true of the problem under every branch, so a
        run's scores and solution ids do not belong in it: they go stale, and they
        are already recorded somewhere that cannot be rewritten.
        """
        if not self.knowledge_path.is_file():
            return []

        knowledge = self.knowledge_path.read_text()
        problems: List[str] = []

        if len(knowledge) > MAXIMUM_KNOWLEDGE_CHARACTERS:
            problems.append(
                f"{KNOWLEDGE_NAME} is {len(knowledge)} characters, over the "
                f"{MAXIMUM_KNOWLEDGE_CHARACTERS} limit. Cut what no longer earns "
                "its place."
            )

        mentioned = sorted(
            solution.id
            for solution in self._store.get_all_solutions()
            if solution.id in knowledge
        )

        if mentioned:
            problems.append(
                f"{KNOWLEDGE_NAME} names solutions ({', '.join(mentioned)}). It is "
                "for what stays true of the problem, not for the state of this "
                "run — that is what the journal is."
            )

        return problems

    def _constraint_problems(
        self, fields: Dict[str, object], problems: List[str]
    ) -> List[str]:
        raw = fields.get("constraints", [])

        if not isinstance(raw, list):
            problems.append("`constraints` must be a list of strings.")
            return []

        constraints = cast(List[object], raw)

        if any(not isinstance(c, str) or not c.strip() for c in constraints):
            problems.append("Every constraint must be a non-empty string.")
            return []

        return [cast(str, c) for c in constraints]

    def _parent_problems(
        self,
        fields: Dict[str, object],
        solutions: List[Solution],
        problems: List[str],
    ) -> None:
        raw = fields.get("parent_solution_ids", [])

        if not isinstance(raw, list):
            problems.append("`parent_solution_ids` must be a list of ids.")
            return

        parent_ids = cast(List[object], raw)

        if len(parent_ids) > MAXIMUM_PARENTS:
            problems.append(
                f"At most {MAXIMUM_PARENTS} parents; {len(parent_ids)} were given."
            )

        known = {solution.id for solution in solutions}
        unknown = [str(i) for i in parent_ids if str(i) not in known]

        if unknown:
            problems.append(f"No such solution: {', '.join(unknown)}.")

    def _reword_warnings(self, constraints: List[str]) -> List[str]:
        if not constraints:
            return []

        key = branch_key(constraints)
        visited = {branch.key for branch in self._branches(self._read_journal())}

        if key in visited:
            return []

        text = "\n".join(key)

        for other in visited:
            ratio = difflib.SequenceMatcher(None, text, "\n".join(other)).ratio()
            if ratio >= REWORD_SIMILARITY:
                return [
                    "This looks like a reworded version of a branch that already "
                    "exists, which would start a new one rather than continue it. "
                    "If you meant to continue it, copy its constraints out of "
                    f"{JOURNAL_NAME} verbatim. If the rewording is the point, "
                    "carry on."
                ]

        return []

    # --- reading the plan ---------------------------------------------------

    def _read_plan(self) -> Optional[Plan]:
        if not self.validate().valid:
            return None

        fields = cast(Dict[str, object], json.loads(self.plan_path.read_text()))

        constraints = [cast(str, c) for c in cast(List[object], fields["constraints"])]
        parent_ids = [
            str(i) for i in cast(List[object], fields.get("parent_solution_ids", []))
        ]
        titles = [str(t) for t in cast(List[object], fields.get("parent_titles", []))]
        note = fields.get("note")

        return Plan(
            constraints=constraints,
            note=cast(Optional[str], note),
            parent_solution_ids=parent_ids,
            parent_titles=titles,
            task=cast(str, fields["task"]),
        )

    def _parents(
        self, plan: Plan, solutions: List[Solution]
    ) -> List[SolutionWithTitle]:
        by_id = {solution.id: solution for solution in solutions}

        parents: List[SolutionWithTitle] = []
        for index, parent_id in enumerate(plan.parent_solution_ids):
            title = (
                plan.parent_titles[index]
                if index < len(plan.parent_titles)
                else f"Solution {index + 1}"
            )
            parents.append(SolutionWithTitle(solution=by_id[parent_id], title=title))

        return parents

    def _fallback(
        self, solutions: List[Solution], result: StrategistResult
    ) -> SearchResult:
        """What to do when no usable plan came back.

        A crashed or confused strategist should cost a weak iteration, not a lost
        one, so the search falls back to improving whatever is currently best.
        """
        logger.warning(
            f"No usable {PLAN_NAME}; falling back to improving the best solution"
        )

        scored = [s for s in solutions if s.score is not None]
        best = (
            min(scored, key=lambda s: cast(float, s.score))
            if scored
            else next((s for s in solutions if s.is_initial), None)
        )

        tags: Dict[str, Union[int, str]] = dict(result.tags)
        tags["fallback"] = "no_plan"

        return SearchResult(
            solutions=(
                [SolutionWithTitle(solution=best, title="Parent")] if best else []
            ),
            tags=tags,
            task=FALLBACK_TASK,
        )

    # --- the journal --------------------------------------------------------

    def _read_journal(self) -> List[JournalEntry]:
        if not self.journal_path.is_file():
            return []

        entries: List[JournalEntry] = []

        for line in self.journal_path.read_text().splitlines():
            if not line.strip():
                continue

            fields = cast(Dict[str, object], json.loads(line))

            entries.append(
                JournalEntry(
                    constraints=[
                        str(c) for c in cast(List[object], fields["constraints"])
                    ],
                    ended_at=str(fields.get("ended_at", "")),
                    exit_status=str(fields.get("exit_status", "")),
                    iteration=cast(int, fields["iteration"]),
                    metrics=cast(
                        Dict[str, Union[int, float]], fields.get("metrics", {})
                    ),
                    note=cast(Optional[str], fields.get("note")),
                    parent_solution_ids=[
                        str(i)
                        for i in cast(
                            List[object], fields.get("parent_solution_ids", [])
                        )
                    ],
                    playbook_entry=cast(Optional[str], fields.get("playbook_entry")),
                    score=cast(Optional[float], fields.get("score")),
                    solution_id=str(fields.get("solution_id", "")),
                    started_at=str(fields.get("started_at", "")),
                    task=str(fields.get("task", "")),
                )
            )

        return entries

    def _branches(self, journal: List[JournalEntry]) -> List[Branch]:
        grouped: Dict[Tuple[str, ...], List[JournalEntry]] = {}
        constraints: Dict[Tuple[str, ...], List[str]] = {}

        for entry in journal:
            grouped.setdefault(entry.key, []).append(entry)
            constraints.setdefault(entry.key, entry.constraints)

        return [
            Branch(constraints=constraints[key], entries=entries)
            for key, entries in grouped.items()
        ]

    # --- the playbook -------------------------------------------------------

    def _playbook(self) -> List[str]:
        if not self._playbook_path.is_file():
            return []

        text = self._playbook_path.read_text()
        entries = [block.strip() for block in text.split("\n-") if block.strip()]

        # The first block is the file's own preamble, not an angle.
        return ["- " + entry for entry in entries[1:]]

    def _stagnation_angle(self, journal: List[JournalEntry]) -> Optional[str]:
        """A playbook entry to push with, once the search has stopped improving.

        Least recently used rather than random: being shoved the same direction
        twice while other angles go untried is the failure this is meant to fix,
        and a deterministic choice also keeps a run reproducible.
        """
        entries = self._playbook()

        if not entries or not self._is_stagnant(journal):
            return None

        used = [e.playbook_entry for e in journal if e.playbook_entry]
        unused = [entry for entry in entries if entry not in used]

        if unused:
            return unused[0]

        # Every angle has been used; take the one used longest ago.
        return min(entries, key=lambda entry: _last_index(used, entry))

    def _is_stagnant(self, journal: List[JournalEntry]) -> bool:
        scored = [e for e in journal if e.score is not None]

        if len(journal) < self._stagnation_limit or not scored:
            return False

        best = min(cast(float, e.score) for e in scored)
        best_iteration = max(
            e.iteration for e in scored if cast(float, e.score) <= best
        )

        return journal[-1].iteration - best_iteration >= self._stagnation_limit

    # --- the prompt and the digest ------------------------------------------

    def _prompt(self, problem_description: str, playbook_entry: Optional[str]) -> str:
        sections = [
            "# What you are doing",
            "",
            OPENING,
            "",
            "# The problem",
            "",
            problem_description.strip(),
            "",
            "# How a branch works",
            "",
            BRANCHES,
            "",
            "# What to write",
            "",
            self._plan_instructions(),
            "",
            "# Where things are",
            "",
            LAYOUT,
            "",
            "# The search so far",
            "",
            self.notebook_path.read_text().strip(),
            "",
            "# Angles worth taking",
            "",
            "\n\n".join(self._playbook()),
        ]

        if playbook_entry is not None:
            sections += [
                "",
                "# Take this one",
                "",
                f"The search has not improved in {self._stagnation_limit} "
                "iterations. Either leave the branch you are on and say why in "
                "`note`, or say why continuing beats dropping a constraint. This "
                "angle has gone untried longest:",
                "",
                playbook_entry,
            ]

        return "\n".join(sections) + "\n"

    def _plan_instructions(self) -> str:
        return PLAN_INSTRUCTIONS.format(
            journal=JOURNAL_NAME,
            knowledge=KNOWLEDGE_NAME,
            maximum_knowledge=MAXIMUM_KNOWLEDGE_CHARACTERS,
            maximum_parents=MAXIMUM_PARENTS,
            plan=PLAN_NAME,
        )

    def _digest(self, journal: List[JournalEntry]) -> str:
        """The history the strategist reads, and the only human view of the run.

        One artifact for both, so a search whose state is unreadable to a person
        is one the strategist could not read either.
        """
        branches = self._branches(journal)

        sections = [
            "# The search so far",
            "",
            self._headline(journal),
            "",
            "## Branches",
            "",
            *(_tree(branches) or ["Nothing has been tried yet."]),
            "",
            "## What the constraints say",
            "",
            *self._live_detail(branches),
            "",
            "## Recent iterations",
            "",
            *self._recent(journal),
            "",
            "## What you have learned",
            "",
            self._knowledge(),
        ]

        return "\n".join(sections) + "\n"

    def _headline(self, journal: List[JournalEntry]) -> str:
        scored = [e for e in journal if e.score is not None]

        if not scored:
            return "No candidate has scored yet."

        best = min(scored, key=lambda e: cast(float, e.score))

        return (
            f"{len(journal)} iterations. Best score {best.score} "
            f"(solution {best.solution_id}, iteration {best.iteration}). "
            "Lower is better."
        )

    def _live_detail(self, branches: List[Branch]) -> List[str]:
        """Full constraint text for branches worth continuing.

        Only the live ones: this is what gets copied to continue a branch, and a
        run with forty branches would otherwise bury the useful ones in the ones
        it walked away from ten iterations ago.
        """
        if not branches:
            return ["Nothing yet."]

        latest = max(branch.last_iteration for branch in branches)
        live = [
            branch
            for branch in sorted(branches, key=lambda b: -b.last_iteration)
            if latest - branch.last_iteration <= RECENT_JOURNAL_ENTRIES
        ]

        lines: List[str] = []
        for branch in live:
            lines.append(f"### {_label(branch.constraints)}")
            lines.append("")

            if not branch.constraints:
                lines.append("(no constraints — the whole space)")
            for constraint in branch.constraints:
                lines.append(f"> {constraint}")
                lines.append("")

            lines.append("")

        return lines

    def _recent(self, journal: List[JournalEntry]) -> List[str]:
        if not journal:
            return ["None yet."]

        lines: List[str] = []
        for entry in journal[-RECENT_JOURNAL_ENTRIES:]:
            score = "did not score" if entry.score is None else f"scored {entry.score}"
            lines.append(
                f"- **{entry.iteration}** — {_label(entry.constraints)}, {score}"
                f" ({entry.exit_status})"
            )
            if entry.note:
                lines.append(f"  - you wrote afterwards: {entry.note}")

        return lines

    def _knowledge(self) -> str:
        if not self.knowledge_path.is_file():
            return f"`{KNOWLEDGE_NAME}` is empty. Start it when you learn something."

        return self.knowledge_path.read_text().strip() or (
            f"`{KNOWLEDGE_NAME}` is empty. Start it when you learn something."
        )


def compose_task(constraints: Sequence[str], task: str) -> str:
    """The brief the coding agent gets: the region, then the instruction.

    The constraints go in verbatim because they were written for this reader. That
    is what makes them load-bearing rather than bookkeeping — they are the reason
    the candidate will differ from its parent, so the agent has to see them.
    """
    if not constraints:
        return task.strip()

    blocks = "\n\n".join(constraint.strip() for constraint in constraints)

    return f"{CONSTRAINTS_HEADING}\n\n{blocks}\n\n{task.strip()}"


def _validation(problems: List[str], warnings: List[str]) -> ValidationResult:
    if problems:
        lines = ["The plan cannot be used yet:", ""]
        lines += [f"- {problem}" for problem in problems]
        return ValidationResult(valid=False, log="\n".join(lines))

    if warnings:
        return ValidationResult(valid=True, log="\n".join(warnings))

    return ValidationResult(valid=True, log="")


def _last_index(used: List[str], entry: str) -> int:
    for index in range(len(used) - 1, -1, -1):
        if used[index] == entry:
            return index
    return -1


def _first_line(constraint: str) -> str:
    stripped = constraint.strip().splitlines()
    first = stripped[0] if stripped else ""
    return first if len(first) <= 70 else first[:69] + "…"


def _label(constraints: Sequence[str]) -> str:
    if not constraints:
        return "(no constraints)"
    return " + ".join(f'"{_first_line(c)}"' for c in constraints)


def _tree(branches: List[Branch]) -> List[str]:
    """Branches as a tree, parents found by subset inclusion rather than stored."""
    if not branches:
        return []

    by_key = {branch.key: branch for branch in branches}
    children: Dict[Tuple[str, ...], List[Tuple[str, ...]]] = {k: [] for k in by_key}
    roots: List[Tuple[str, ...]] = []

    for key in by_key:
        # The closest visited branch this one refines: the largest proper subset.
        supersets = [
            other for other in by_key if other != key and set(other) < set(key)
        ]

        if supersets:
            children[max(supersets, key=len)].append(key)
        else:
            roots.append(key)

    lines: List[str] = []

    def walk(key: Tuple[str, ...], depth: int) -> None:
        branch = by_key[key]
        lines.append(f"{'  ' * depth}- {_label(branch.constraints)} — {_stats(branch)}")

        for note in (e.note for e in branch.entries if e.note):
            lines.append(f"{'  ' * depth}  - note: {note}")

        for child in sorted(children[key]):
            walk(child, depth + 1)

    for root in sorted(roots):
        walk(root, 0)

    return lines


def _stats(branch: Branch) -> str:
    scores = branch.scores
    attempts = f"{len(branch.entries)} attempt{'s' if len(branch.entries) != 1 else ''}"

    if not scores:
        return f"{attempts}, none scored, last at iteration {branch.last_iteration}"

    spread = f", spread {scores[-1] - scores[0]:.4g}" if len(scores) > 1 else ""

    return (
        f"{attempts}, best {scores[0]:.6g}, median "
        f"{statistics.median(scores):.6g}{spread}, "
        f"last at iteration {branch.last_iteration}"
    )


OPENING = """You are steering an automated search for a better solution to the
problem below. Each iteration, a separate coding agent writes one candidate from
scratch and it is scored automatically; lower scores are better.

You decide what that agent works on. You see every score; it sees none, because
ranking is your job and an agent that could see its own score would abandon a
novel approach the moment it looked worse than the incumbent.

You have a shell and the whole run directory is readable. Read whatever you
need — a candidate's source, an agent's trajectory, the raw journal. Going and
looking is the point of you being an agent rather than a formula."""

BRANCHES = """A **constraint** is prose telling the coding agent how to narrow its
approach — an instruction it will read, not a label. A sentence is usual;
paragraphs are fine.

A **branch** is the list of constraints in force, and the list *is* the branch.
There are no branch ids. Two plans with the same constraints are the same branch,
and structure comes from which sets contain which:

- the **root** branch is the empty list: no constraints, build from the problem
  description alone
- **going deeper** adds a constraint to a branch you have already tried
- **changing your mind** means dropping back to a subset and adding something else
- **combining** two lines of work is just a branch listing the constraints of both

Trying the same branch several times is normal, not a waste. The coding agent
rebuilds from an empty directory every time, so one attempt tells you very little
about whether a constraint helped — several attempts turn a point into a spread
you can actually compare against the parent branch."""

LAYOUT = """You are in the strategist directory. You own what is in it; read
anything else, write nothing else.

- `plan.json` — what you write this iteration. Cleared before you start.
- `knowledge.md` — yours to keep. See above.
- `journal.jsonl` — one line per finished iteration. Written for you, not by you.
- `notebook.md` — the digest above, saved.
- `../solutions.csv` — every candidate, best first.
- `../solutions/<id>/code/` — a candidate's source.
- `../solutions/<id>/agent.log` — what the coding agent did and saw.
- `../solutions/<id>/metadata.json` — its score, metrics and timing."""

PLAN_INSTRUCTIONS = """Write `{plan}`:

```json
{{
  "constraints": ["…", "…"],
  "parent_solution_ids": ["…"],
  "parent_titles": ["…"],
  "task": "…",
  "note": "…"
}}
```

- `constraints` — the branch to work in. Empty means the root: build from the
  problem description alone. To continue an existing branch, copy its constraints
  out of `{journal}` **verbatim** — a reworded constraint silently starts a new
  branch rather than continuing the one you meant.
- `parent_solution_ids` — up to {maximum_parents} solutions the agent gets its own
  copies of. Empty is right when you want a fresh start.
- `parent_titles` — what to call each one, in the same order. Say something
  useful: "best under this branch, 506ms" beats "Solution 1".
- `task` — what to try this time, inside those constraints. The agent sees this
  and the constraints, and nothing else about the search. Tell it what you know
  that it cannot measure: what the parent scored, what has already failed, where
  the cost seems to be.
- `note` — optional, about the iteration that just finished. This is where you
  say why you are leaving a branch. It is attached to that branch for you.

Also keep `{knowledge}`: what stays true of this problem however you attack it —
build rules, what the timer does and does not count, how noisy the scores are,
how much a coding agent gets done before it runs out of steps. Under
{maximum_knowledge} characters, and no solution ids or scores: those go stale and
the journal already has them. Every coding agent starts knowing nothing, so what
you put here is what stops each one rediscovering the same things.

Run `validate` when you are done."""

# A plain line rather than a heading: this is spliced into the prompt underneath
# `# Your task`, where a second `#` would read as a competing section.
CONSTRAINTS_HEADING = (
    "**Work within these constraints.** They are not optional, and they are the "
    "point of this attempt."
)
