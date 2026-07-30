import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

from optiverse import codebase
from optiverse.store import FileSystemStore


class StoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.store = FileSystemStore(directory=self.root)

    def tearDown(self) -> None:
        codebase.make_writable(self.root)
        self._temporary_directory.cleanup()

    def add(self, *, score: float | None, **kwargs: object) -> str:
        solution_id = self.store.allocate()
        (self.store.codebase_path(solution_id) / "solver.py").write_text("pass\n")
        self.store.commit(
            solution_id,
            description=(
                str(kwargs.get("description")) if "description" in kwargs else None
            ),
            is_initial=bool(kwargs.get("is_initial", False)),
            metrics={},
            score=score,
            tags={},
        )
        return solution_id

    def read_csv(self) -> List[Dict[str, str]]:
        with open(self.root / "solutions.csv", newline="") as csv_file:
            return list(csv.DictReader(csv_file))


class AllocateTest(StoreTestCase):
    def test_creates_an_empty_code_directory(self) -> None:
        solution_id = self.store.allocate()

        self.assertTrue(self.store.codebase_path(solution_id).is_dir())
        self.assertEqual(
            codebase.relative_files(self.store.codebase_path(solution_id)), []
        )

    def test_allocated_but_uncommitted_solutions_are_invisible(self) -> None:
        """An iteration that dies mid-flight must not enter the population."""
        self.store.allocate()

        self.assertEqual(self.store.get_all_solutions(), [])

    def test_ids_are_distinct(self) -> None:
        self.assertNotEqual(self.store.allocate(), self.store.allocate())

    def test_paths_are_absolute_even_from_a_relative_directory(self) -> None:
        """They are handed to an agent running in its own working directory, so a
        path relative to ours names nothing there."""
        store = FileSystemStore(directory=Path("tmp") / "relative")
        solution_id = "0" * 32

        for path in (
            store.codebase_path(solution_id),
            store.description_path(solution_id),
            store.agent_log_path(solution_id),
        ):
            self.assertTrue(path.is_absolute(), path)

    def test_description_and_log_paths_sit_outside_the_codebase(self) -> None:
        """Neither may end up inside code/, or it would become part of the solution."""
        solution_id = self.store.allocate()
        code = self.store.codebase_path(solution_id)

        for path in (
            self.store.description_path(solution_id),
            self.store.agent_log_path(solution_id),
        ):
            self.assertNotIn(code, path.parents)


class CommitTest(StoreTestCase):
    def test_round_trips_metadata(self) -> None:
        solution_id = self.store.allocate()
        (self.store.codebase_path(solution_id) / "solver.py").write_text("pass\n")

        self.store.commit(
            solution_id,
            description="tried a greedy heuristic",
            is_initial=False,
            metrics={"lines": 12},
            score=3.5,
            tags={"move": "local_search", "group": 2},
        )

        solutions = self.store.get_all_solutions()
        self.assertEqual(len(solutions), 1)

        solution = solutions[0]
        self.assertEqual(solution.id, solution_id)
        self.assertEqual(solution.description, "tried a greedy heuristic")
        self.assertEqual(solution.score, 3.5)
        self.assertEqual(solution.metrics, {"lines": 12})
        self.assertEqual(solution.tags, {"move": "local_search", "group": 2})
        self.assertFalse(solution.is_initial)

    def test_solution_codebase_points_at_stored_files(self) -> None:
        self.add(score=1.0)

        solution = self.store.get_all_solutions()[0]

        self.assertEqual(codebase.relative_files(solution.codebase), ["solver.py"])
        self.assertEqual((solution.codebase / "solver.py").read_text(), "pass\n")

    def test_committed_codebase_is_read_only(self) -> None:
        """Parents are re-read by later iterations, so they must not be editable."""
        solution_id = self.add(score=1.0)

        with self.assertRaises(PermissionError):
            (self.store.codebase_path(solution_id) / "solver.py").write_text("edited")

    def test_committing_an_unallocated_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.commit(
                "nonexistent",
                description=None,
                is_initial=False,
                metrics={},
                score=None,
                tags={},
            )

    def test_no_temporary_metadata_file_is_left_behind(self) -> None:
        solution_id = self.add(score=1.0)

        leftovers = [
            name
            for name in codebase.relative_files(self.root / solution_id)
            if name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])


class CrashSafetyTest(StoreTestCase):
    def test_directory_without_metadata_is_skipped(self) -> None:
        good = self.add(score=1.0)
        abandoned = self.store.allocate()
        (self.store.codebase_path(abandoned) / "half-written.py").write_text("x")

        ids = [s.id for s in self.store.get_all_solutions()]

        self.assertEqual(ids, [good])

    def test_abandoned_directory_stays_on_disk_for_inspection(self) -> None:
        abandoned = self.store.allocate()

        self.store.get_all_solutions()

        self.assertTrue(self.store.codebase_path(abandoned).is_dir())

    def test_stray_file_in_the_run_directory_is_ignored(self) -> None:
        self.add(score=1.0)
        (self.root / "notes.txt").write_text("scratch")

        self.assertEqual(len(self.store.get_all_solutions()), 1)

    def test_missing_run_directory_yields_no_solutions(self) -> None:
        store = FileSystemStore(directory=self.root / "absent")

        self.assertEqual(store.get_all_solutions(), [])


class SolutionsCsvTest(StoreTestCase):
    def test_orders_by_score_with_failures_last(self) -> None:
        worst = self.add(score=30.0)
        best = self.add(score=10.0)
        failed = self.add(score=None)
        middle = self.add(score=20.0)

        rows = self.read_csv()

        self.assertEqual([row["id"] for row in rows], [best, middle, worst, failed])
        self.assertEqual(rows[-1]["score"], "FAILED")

    def test_metrics_and_tags_become_prefixed_columns(self) -> None:
        solution_id = self.store.allocate()
        self.store.commit(
            solution_id,
            description=None,
            is_initial=False,
            metrics={"agent_cost_usd": 0.02},
            score=1.0,
            tags={"exit_status": "Validated"},
        )

        rows = self.read_csv()

        self.assertEqual(rows[0]["m_agent_cost_usd"], "0.02")
        self.assertEqual(rows[0]["t_exit_status"], "Validated")

    def test_metadata_is_valid_json_on_disk(self) -> None:
        solution_id = self.add(score=1.0)

        metadata = json.loads((self.root / solution_id / "metadata.json").read_text())

        self.assertEqual(metadata["id"], solution_id)


if __name__ == "__main__":
    unittest.main()
