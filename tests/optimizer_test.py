"""End-to-end tests for one turn of the loop.

A stub generator stands in for the agent, so the whole path — allocate,
materialize the parent, generate, score, commit — runs without an LLM. The
evaluator is a real script in a real subprocess, as the contract requires.

The toy problem: a codebase is a single `value.txt` holding a number, and the
score is that number. Lower is better, matching the loop's minimisation.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional

from optiverse import codebase
from optiverse.config import OptimizerConfig, Problem
from optiverse.generator import GenerationContext, GenerationResult, Generator
from optiverse.optimizer import Optimizer
from optiverse.search_strategies import IteratedLocalSearch
from optiverse.store import FileSystemStore

EVALUATOR_SOURCE = """
import json
import sys
from pathlib import Path

mode, codebase = sys.argv[1], Path(sys.argv[2])
value_file = codebase / "value.txt"

if not value_file.is_file():
    print("value.txt is missing", file=sys.stderr)
    sys.exit(1)

try:
    value = float(value_file.read_text().strip())
except ValueError:
    value = None

if mode == "validate":
    # The verdict is the exit code, and nothing is printed on stdout.
    print("value.txt does not hold a number" if value is None else "", file=sys.stderr)
    sys.exit(1 if value is None else 0)

if value is None:
    # Ran fine; this candidate just cannot be scored. Distinct from exiting
    # non-zero, which would mean the evaluator itself broke.
    print("cannot score: value.txt does not hold a number", file=sys.stderr)
    json.dump({"score": None, "metrics": {}}, sys.stdout)
    sys.exit(0)

print("scoring", value, file=sys.stderr)
json.dump({"score": value, "metrics": {"value": value}}, sys.stdout)
"""


class ImprovingGenerator(Generator):
    """Writes a slightly better value each time it is called."""

    def __init__(self, *, description: Optional[str] = None) -> None:
        self.calls = 0
        self.contexts: List[GenerationContext] = []
        self._description = description

    def generate(self, context: GenerationContext) -> GenerationResult:
        self.calls += 1
        self.contexts.append(context)

        current = float((context.codebase / "value.txt").read_text().strip())
        (context.codebase / "value.txt").write_text(f"{current - 1.0}\n")

        if self._description is not None:
            context.description_path.write_text(self._description)

        return GenerationResult(
            metrics={"agent_model_calls": 3}, tags={"exit_status": "Validated"}
        )


class InertGenerator(Generator):
    """Changes nothing, as an agent that gave up would."""

    def generate(self, context: GenerationContext) -> GenerationResult:
        return GenerationResult(metrics={}, tags={"exit_status": "LimitsExceeded"})


class CrashingGenerator(Generator):
    def generate(self, context: GenerationContext) -> GenerationResult:
        raise RuntimeError("the agent exploded")


class BreakingGenerator(Generator):
    """Produces a codebase the evaluator cannot score."""

    def generate(self, context: GenerationContext) -> GenerationResult:
        (context.codebase / "value.txt").write_text("not a number\n")
        return GenerationResult(metrics={}, tags={"exit_status": "Validated"})


class OptimizerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)

        self.initial = self.root / "initial"
        self.initial.mkdir()
        (self.initial / "value.txt").write_text("100.0\n")

        evaluator_script = self.root / "evaluate.py"
        evaluator_script.write_text(EVALUATOR_SOURCE)

        self.run_directory = self.root / "run"
        self.run_directory.mkdir()

        self.problem = Problem(
            description="Make the number smaller.",
            initial_codebase=self.initial,
            evaluate_command=[sys.executable, str(evaluator_script)],
        )

    def tearDown(self) -> None:
        codebase.make_writable(self.run_directory)
        self._temporary_directory.cleanup()

    def optimize(self, generator: Generator, *, max_iterations: int) -> None:
        Optimizer(
            config=OptimizerConfig(
                directory=self.run_directory,
                generator=generator,
                max_iterations=max_iterations,
                problem=self.problem,
                search_strategy=IteratedLocalSearch(
                    max_iterations_without_improvements=10
                ),
            )
        ).run()

    def rows(self) -> List[Dict[str, str]]:
        with open(self.run_directory / "solutions.csv", newline="") as csv_file:
            return list(csv.DictReader(csv_file))


class HappyPathTest(OptimizerTestCase):
    def test_scores_the_initial_solution_then_improves_on_it(self) -> None:
        self.optimize(ImprovingGenerator(), max_iterations=3)

        scores = [float(row["score"]) for row in self.rows()]

        # 100 for the seed, then 99, 98, 97.
        self.assertEqual(sorted(scores), [97.0, 98.0, 99.0, 100.0])

    def test_best_solution_is_first_in_the_csv(self) -> None:
        self.optimize(ImprovingGenerator(), max_iterations=2)

        self.assertEqual(float(self.rows()[0]["score"]), 98.0)

    def test_each_solution_keeps_its_own_codebase(self) -> None:
        self.optimize(ImprovingGenerator(), max_iterations=2)

        for row in self.rows():
            code = self.run_directory / row["id"] / "code"
            self.assertEqual(codebase.relative_files(code), ["value.txt"])
            self.assertEqual(
                float((code / "value.txt").read_text().strip()), float(row["score"])
            )

    def test_generator_metrics_and_tags_reach_the_csv(self) -> None:
        self.optimize(ImprovingGenerator(), max_iterations=1)

        improved = [row for row in self.rows() if row["t_exit_status"] == "Validated"]

        self.assertEqual(len(improved), 1)
        self.assertEqual(improved[0]["m_agent_model_calls"], "3")

    def test_evaluator_metrics_and_generator_metrics_coexist(self) -> None:
        self.optimize(ImprovingGenerator(), max_iterations=1)

        row = self.rows()[0]

        self.assertEqual(row["m_value"], "99.0")
        self.assertEqual(row["m_agent_model_calls"], "3")

    def test_lineage_is_recorded(self) -> None:
        self.optimize(ImprovingGenerator(), max_iterations=1)

        child = [row for row in self.rows() if row["t_parent_id_1"]][0]
        parent_ids = [row["id"] for row in self.rows()]

        self.assertIn(child["t_parent_id_1"], parent_ids)

    def test_seed_codebase_is_never_modified(self) -> None:
        self.optimize(ImprovingGenerator(), max_iterations=3)

        self.assertEqual((self.initial / "value.txt").read_text(), "100.0\n")

    def test_parents_are_read_only_when_handed_to_the_generator(self) -> None:
        """The generator must not be able to corrupt a population member."""
        generator = ImprovingGenerator()
        self.optimize(generator, max_iterations=2)

        for context in generator.contexts:
            for reference in context.references:
                self.assertFalse(
                    reference.path.stat().st_mode & 0o222,
                    f"{reference.path} is writable",
                )


class DescriptionTest(OptimizerTestCase):
    def test_description_written_to_disk_lands_in_metadata(self) -> None:
        self.optimize(
            ImprovingGenerator(description="halved the value"), max_iterations=1
        )

        descriptions = [
            solution.description
            for solution in FileSystemStore(self.run_directory).get_all_solutions()
            if not solution.is_initial
        ]

        self.assertEqual(descriptions, ["halved the value"])

    def test_description_file_is_consumed(self) -> None:
        """It lives in metadata.json afterwards, so leaving a copy would create a
        second source of truth."""
        self.optimize(
            ImprovingGenerator(description="halved the value"), max_iterations=1
        )

        leftovers = list(self.run_directory.rglob("description.txt"))

        self.assertEqual(leftovers, [])


class FailureTest(OptimizerTestCase):
    def test_unchanged_codebase_is_still_scored(self) -> None:
        self.optimize(InertGenerator(), max_iterations=1)

        self.assertEqual([float(row["score"]) for row in self.rows()], [100.0, 100.0])

    def test_unscoreable_candidate_is_recorded_as_failed(self) -> None:
        self.optimize(BreakingGenerator(), max_iterations=1)

        rows = self.rows()

        self.assertEqual(rows[-1]["score"], "FAILED")
        self.assertEqual(len(rows), 2)

    def test_unscoreable_candidate_reports_how_to_reproduce_it(self) -> None:
        """No score log is stored, so the console has to say how to get one."""
        with self.assertLogs("optiverse.optimizer", level="INFO") as captured:
            self.optimize(BreakingGenerator(), max_iterations=1)

        reproduce = [line for line in captured.output if "Reproduce with" in line]

        self.assertEqual(len(reproduce), 1)
        self.assertIn("score", reproduce[0])
        self.assertIn("cannot score", reproduce[0])

    def test_broken_evaluator_does_not_stop_the_run(self) -> None:
        """A missing or crashing evaluator is a different problem from a bad
        candidate, and must not be silently absorbed as 10,000 failures."""
        self.problem = Problem(
            description=self.problem.description,
            initial_codebase=self.initial,
            evaluate_command=[sys.executable, str(self.root / "absent.py")],
        )

        with self.assertLogs("optiverse.optimizer", level="ERROR") as captured:
            self.optimize(ImprovingGenerator(), max_iterations=1)

        self.assertTrue(any("Evaluator failed" in line for line in captured.output))
        self.assertTrue(all(row["score"] == "FAILED" for row in self.rows()))

    def test_crashing_generator_does_not_stop_the_run(self) -> None:
        self.optimize(CrashingGenerator(), max_iterations=2)

        # Only the seed is committed; the loop survived both failures.
        self.assertEqual([float(row["score"]) for row in self.rows()], [100.0])

    def test_crashed_iteration_leaves_its_directory_for_inspection(self) -> None:
        self.optimize(CrashingGenerator(), max_iterations=1)

        committed = {row["id"] for row in self.rows()}
        directories = {
            path.name for path in self.run_directory.iterdir() if path.is_dir()
        }

        self.assertEqual(len(directories - committed), 1)


class CheckpointTest(OptimizerTestCase):
    def test_resuming_continues_rather_than_restarting(self) -> None:
        self.optimize(ImprovingGenerator(), max_iterations=2)
        first_pass = len(self.rows())

        self.optimize(ImprovingGenerator(), max_iterations=4)

        self.assertGreater(len(self.rows()), first_pass)
        # The seed is scored once, not once per invocation.
        seeds = [row for row in self.rows() if float(row["score"]) == 100.0]
        self.assertEqual(len(seeds), 1)


if __name__ == "__main__":
    unittest.main()
