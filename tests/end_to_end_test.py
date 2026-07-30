"""One real run of the loop, with the model replaced and nothing else.

The store, the search strategy and the evaluator are the real ones — the
evaluator is a real script in a real subprocess, because that contract is a
process contract. Only the agent is stood in for, by a generator that does what
the prompt asks: copy a parent in, improve it, check its own work.

The run happens once for the module. Every test then inspects what it left on
disk, which is what you would look at after a live run.
"""

import json
import shutil
import stat
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, cast

from optiverse import codebase as codebase_helpers
from optiverse.config import OptimizerConfig, Problem
from optiverse.generator import GenerationContext, GenerationResult, Generator
from optiverse.optimizer import Optimizer
from optiverse.search_strategies import IteratedLocalSearch

# Validity is "there is a solver.py"; the score is the number inside it. Small
# enough to read, real enough to exercise both evaluator modes.
EVALUATOR = """import json, sys, pathlib
mode, codebase = sys.argv[1], pathlib.Path(sys.argv[2])
solver = codebase / "solver.py"
if mode == "validate":
    sys.exit(0 if solver.is_file() else 1)
value = float(solver.read_text().split("=")[1]) if solver.is_file() else None
json.dump({"score": value, "metrics": {"length": 3}}, sys.stdout)
"""

INITIAL_VALUE = 100.0
ITERATIONS = 2


class CopyingGenerator(Generator):
    """A stand-in agent that follows the instructions the real one is given."""

    def __init__(self) -> None:
        self.prompts: List[str] = []
        self.codebases_at_entry: List[List[str]] = []

    def generate(self, context: GenerationContext) -> GenerationResult:
        self.prompts.append(context.prompt)
        self.codebases_at_entry.append(
            codebase_helpers.relative_files(context.codebase)
        )

        parents = sorted(context.references_directory.iterdir())
        shutil.copytree(parents[0] / "code", context.codebase, dirs_exist_ok=True)

        solver = context.codebase / "solver.py"
        value = float(solver.read_text().split("=")[1])
        solver.write_text(f"value = {value - 1}\n")

        if not context.validate().valid:
            raise AssertionError("the copied-and-edited solution should validate")

        return GenerationResult(
            metrics={"agent_model_calls": 4}, tags={"exit_status": "Validated"}
        )


@dataclass(frozen=True)
class CompletedRun:
    directory: Path
    generator: CopyingGenerator


_completed_run: Optional[CompletedRun] = None
_temporary_directory: Optional[tempfile.TemporaryDirectory[str]] = None


def setUpModule() -> None:
    global _completed_run, _temporary_directory

    _temporary_directory = tempfile.TemporaryDirectory()
    root = Path(_temporary_directory.name)

    evaluate = root / "evaluate.py"
    evaluate.write_text(EVALUATOR)

    initial = root / "initial"
    initial.mkdir()
    (initial / "solver.py").write_text(f"value = {INITIAL_VALUE}\n")

    generator = CopyingGenerator()
    run_directory = root / "run"

    Optimizer(
        OptimizerConfig(
            directory=run_directory,
            generator=generator,
            max_iterations=ITERATIONS,
            problem=Problem(
                description="Make the number smaller.",
                initial_codebase=initial,
                evaluate_command=[sys.executable, str(evaluate)],
            ),
            search_strategy=IteratedLocalSearch(max_iterations_without_improvements=5),
        )
    ).run()

    _completed_run = CompletedRun(directory=run_directory, generator=generator)


def tearDownModule() -> None:
    assert _temporary_directory is not None
    _temporary_directory.cleanup()


def completed_run() -> CompletedRun:
    assert _completed_run is not None, "setUpModule did not run"
    return _completed_run


class RunTestCase(unittest.TestCase):
    """Read-only access to the run every test in this module shares."""

    def solution_directories(self) -> List[Path]:
        directory = completed_run().directory
        return sorted(path for path in directory.iterdir() if path.is_dir())

    def metadata(self, solution_directory: Path) -> Dict[str, object]:
        return json.loads((solution_directory / "metadata.json").read_text())

    def score(self, solution_directory: Path) -> Optional[float]:
        return cast(Optional[float], self.metadata(solution_directory)["score"])

    def references(self) -> List[Path]:
        return sorted(completed_run().directory.glob("*/references/*"))

    def agent_solutions(self) -> List[Path]:
        """The solutions an agent produced; the initial one is not one of them."""
        return [
            directory
            for directory in self.solution_directories()
            if not self.metadata(directory)["is_initial"]
        ]


class LoopTest(RunTestCase):
    def test_every_iteration_produced_a_scored_solution(self) -> None:
        self.assertEqual(len(self.solution_directories()), ITERATIONS + 1)

        for directory in self.solution_directories():
            self.assertIsNotNone(self.score(directory), directory.name)

    def test_each_iteration_improved_on_the_one_before(self) -> None:
        """The point of the loop: copy a parent in, edit it, beat it."""
        scores = [self.score(directory) for directory in self.solution_directories()]

        self.assertEqual(
            sorted(score for score in scores if score is not None),
            [INITIAL_VALUE - 2, INITIAL_VALUE - 1, INITIAL_VALUE],
        )

    def test_solutions_csv_lists_the_population_best_first(self) -> None:
        rows = (completed_run().directory / "solutions.csv").read_text().splitlines()

        self.assertIn("m_length", rows[0])
        self.assertIn("t_exit_status", rows[0])
        self.assertEqual(rows[1].split(",")[1], str(INITIAL_VALUE - 2))


class WorkingDirectoryTest(RunTestCase):
    def test_the_agent_starts_with_nothing(self) -> None:
        """Not seeded with a parent: what to take from one is the agent's call."""
        self.assertEqual(completed_run().generator.codebases_at_entry, [[], []])

    def test_what_the_agent_left_behind_is_the_solution(self) -> None:
        for directory in self.agent_solutions():
            self.assertEqual(
                codebase_helpers.relative_files(directory / "code"), ["solver.py"]
            )

    def test_nothing_in_the_run_is_read_only(self) -> None:
        """Stored solutions used to be chmod'd; parents are copies now."""
        unwritable = [
            path
            for path in completed_run().directory.rglob("*")
            if not stat.S_IMODE(path.stat().st_mode) & stat.S_IWUSR
        ]

        self.assertEqual(unwritable, [])


class ReferencesTest(RunTestCase):
    def test_a_reference_holds_the_code_and_its_metadata(self) -> None:
        self.assertEqual(len(self.references()), ITERATIONS)

        for reference in self.references():
            self.assertEqual(
                sorted(path.name for path in reference.iterdir()),
                ["code", "metadata.txt"],
            )

    def test_a_reference_is_named_after_the_solution_it_copies(self) -> None:
        """The prompt names ids, so the directory name is what connects them."""
        existing = {directory.name for directory in self.solution_directories()}

        for reference in self.references():
            self.assertIn(reference.name, existing)

    def test_the_copy_holds_what_the_parent_holds(self) -> None:
        for reference in self.references():
            self.assertEqual(
                codebase_helpers.digest(reference / "code"),
                codebase_helpers.digest(
                    completed_run().directory / reference.name / "code"
                ),
            )

    def test_the_parents_agent_log_is_not_copied(self) -> None:
        """It is the largest thing in a solution, and says how rather than what."""
        self.assertEqual(
            list(completed_run().directory.glob("*/references/*/agent.log")), []
        )

    def test_metadata_carries_the_score_and_every_metric(self) -> None:
        reference = self.references()[0]
        original = self.metadata(completed_run().directory / reference.name)

        text = (reference / "metadata.txt").read_text()

        self.assertIn(f"Solution: {reference.name}", text)
        self.assertIn(f"Score: {original['score']}", text)
        self.assertIn("Lower is better.", text)
        self.assertIn("length: 3.0", text)

    def test_editing_a_copy_cannot_reach_what_it_came_from(self) -> None:
        """What replaced the read-only bits: the agent only ever holds a copy."""
        reference = self.references()[0]
        original = completed_run().directory / reference.name / "code"
        before = codebase_helpers.digest(original)

        copy = reference / "code" / "solver.py"
        # The run is shared with every other test in the module, so put back
        # what this one scribbles on.
        self.addCleanup(copy.write_text, copy.read_text())
        copy.write_text("value = 0.0\n")

        self.assertEqual(codebase_helpers.digest(original), before)


class PromptTest(RunTestCase):
    def test_parents_are_named_by_id(self) -> None:
        parent_ids = [reference.name for reference in self.references()]

        for prompt in completed_run().generator.prompts:
            self.assertTrue(any(name in prompt for name in parent_ids), prompt)

    def test_the_prompt_never_states_a_score(self) -> None:
        """Scores live in metadata.txt now, for the agent to read or ignore."""
        for prompt in completed_run().generator.prompts:
            self.assertNotIn("Score", prompt)
            self.assertNotIn(str(INITIAL_VALUE), prompt)

    def test_the_prompt_holds_no_absolute_path(self) -> None:
        """The agent starts in its codebase, so `.` and `../references` do."""
        run_directory = str(completed_run().directory)

        for prompt in completed_run().generator.prompts:
            self.assertNotIn(run_directory, prompt)
            self.assertIn("../references/", prompt)

    def test_the_sections_read_in_order(self) -> None:
        prompt = completed_run().generator.prompts[0]

        headings = [line for line in prompt.splitlines() if line.startswith("# ")]

        self.assertEqual(
            headings,
            [
                "# What you are doing",
                "# The problem",
                "# Your working directory",
                "# The parent solutions",
                "# Your task",
            ],
        )

    def test_the_prompt_carries_the_problem_and_the_task(self) -> None:
        """The task is the strategy's move, and it changes between iterations.

        Nothing is in group 0 when the first iteration starts, so it is told to
        build from scratch; by the second there is something to improve on.
        """
        first, second = completed_run().generator.prompts

        for prompt in (first, second):
            self.assertIn("Make the number smaller.", prompt)

        self.assertIn("Start over.", first)
        self.assertIn("Make a focused improvement", second)


if __name__ == "__main__":
    unittest.main()
