"""One real report over a run directory built by hand.

The shape of a run is easy to state and awkward to fake convincingly, so the
fixture states it exactly: a root, a chain that wins, a branch that does not, a
recombination with two parents, and one solution that failed to score. Mtimes
are set explicitly, because the ordering they carry is the thing most likely to
break and the thing a copy silently destroys.

The report is generated once for the module and every test then reads what
landed on disk, which is what you would look at after running the command.
"""

import contextlib
import csv
import io
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from optiverse.store import FileSystemStore

from optiverse_analysis.__main__ import (
    LINEAGE_NAME,
    TIMELINE_NAME,
    TREE_NAME,
    main,
)
from optiverse_analysis.run import load

# id, score, parents, minutes after the start. Scores descend down the winning
# chain and the losing branch stays worse, so "best" is never ambiguous.
Plan = Tuple[str, Optional[float], Tuple[str, ...], int]

PLAN: Tuple[Plan, ...] = (
    ("a" * 32, 900.0, (), 0),
    ("b" * 32, 400.0, ("a" * 32,), 10),
    ("c" * 32, 380.0, ("b" * 32,), 20),
    ("d" * 32, 500.0, ("b" * 32,), 30),
    ("e" * 32, None, ("c" * 32,), 40),
    ("f" * 32, 350.0, ("c" * 32, "d" * 32), 50),
)

WINNER = "f" * 32
START = 1_800_000_000


def _build(root: Path) -> None:
    store = FileSystemStore(root)

    for solution_id, score, parents, minute in PLAN:
        codebase = root / solution_id / "code"
        codebase.mkdir(parents=True)

        # Each solution differs from its parent by one line, so every diff in
        # the report has something real in it.
        (codebase / "solver.py").write_text(
            f"value = {score}\nshared = 1\n" if score is not None else "broken = 1\n"
        )

        tags: Dict[str, Union[int, str]] = {}
        if parents:
            tags["move"] = "local_search"
            tags["group"] = 0
            for index, parent in enumerate(parents, 1):
                tags[f"parent_id_{index}"] = parent
                tags[f"parent_title_{index}"] = "Parent"

        store.commit(
            solution_id,
            is_initial=not parents,
            metrics={"line_count": 2},
            score=score,
            tags=tags,
        )

        stamp = START + minute * 60
        os.utime(root / solution_id / "metadata.json", (stamp, stamp))


_directory: Optional[tempfile.TemporaryDirectory[str]] = None
_run: Optional[Path] = None
_output: Optional[Path] = None


def setUpModule() -> None:
    global _directory, _run, _output

    _directory = tempfile.TemporaryDirectory()
    _run = Path(_directory.name) / "20260730_141954"
    _run.mkdir()
    _build(_run)

    _output = _run / "analysis"

    # The command is meant to be chatty; a test run is not.
    with contextlib.redirect_stdout(io.StringIO()):
        exit_code = main([str(_run)])

    if exit_code != 0:
        raise AssertionError("the report should have been generated")


def tearDownModule() -> None:
    if _directory is not None:
        _directory.cleanup()


class ReportTestCase(unittest.TestCase):
    @property
    def run_directory(self) -> Path:
        assert _run is not None
        return _run

    @property
    def output(self) -> Path:
        assert _output is not None
        return _output

    def rows(self) -> List[Dict[str, str]]:
        with open(self.output / TIMELINE_NAME, newline="") as csv_file:
            return list(csv.DictReader(csv_file))


class TimelineTest(ReportTestCase):
    def test_rows_are_ordered_by_when_the_solution_was_committed(self) -> None:
        ids = [row["id"] for row in self.rows()]
        self.assertEqual(ids, [plan[0] for plan in PLAN])

    def test_the_running_best_only_ever_improves(self) -> None:
        bests = [float(row["best_so_far"]) for row in self.rows()]
        self.assertEqual(bests, sorted(bests, reverse=True))

    def test_a_solution_without_a_score_is_marked_failed(self) -> None:
        failed = [row for row in self.rows() if row["score"] == "FAILED"]
        self.assertEqual([row["id"] for row in failed], ["e" * 32])

    def test_only_solutions_that_beat_everything_before_are_new_bests(self) -> None:
        winners = [row["id"] for row in self.rows() if row["is_new_best"]]
        self.assertEqual(winners, ["a" * 32, "b" * 32, "c" * 32, "f" * 32])

    def test_depth_counts_generations_from_the_initial_solution(self) -> None:
        depths = {row["id"]: int(row["depth"]) for row in self.rows()}
        self.assertEqual(depths["a" * 32], 0)
        self.assertEqual(depths["b" * 32], 1)
        self.assertEqual(depths["f" * 32], 3)


class LineageTest(ReportTestCase):
    def report(self) -> str:
        return (self.output / LINEAGE_NAME).read_text()

    def test_the_chain_runs_from_the_initial_solution_to_the_best(self) -> None:
        history = load(self.run_directory)
        from optiverse_analysis import lineage

        steps, complete = lineage.trace(history, history.best)

        self.assertTrue(complete)
        self.assertEqual(
            [step.record.id for step in steps],
            ["a" * 32, "b" * 32, "c" * 32, WINNER],
        )

    def test_the_second_parent_of_a_recombination_is_named(self) -> None:
        history = load(self.run_directory)
        from optiverse_analysis import lineage

        steps, _ = lineage.trace(history, history.best)

        self.assertEqual([other.id for other in steps[-1].merge_ins], ["d" * 32])

    def test_the_report_shows_the_diff_between_a_parent_and_its_child(self) -> None:
        self.assertIn("-value = 400.0", self.report())
        self.assertIn("+value = 380.0", self.report())


class OutputTest(ReportTestCase):
    def test_every_file_is_written(self) -> None:
        for name in (TIMELINE_NAME, TREE_NAME, LINEAGE_NAME):
            self.assertTrue((self.output / name).is_file(), name)

    def test_the_chart_is_a_png(self) -> None:
        self.assertEqual(
            (self.output / TREE_NAME).read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
        )

    def test_the_report_does_not_disturb_the_run_it_describes(self) -> None:
        # The store skips a directory with no metadata.json, so writing the
        # report into the run must not change what the run thinks it holds.
        solutions = FileSystemStore(self.run_directory).get_all_solutions()
        self.assertEqual(len(solutions), len(PLAN))


class FlattenedTimestampsTest(unittest.TestCase):
    def test_a_copy_that_lost_its_mtimes_is_reported_as_suspect(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            run = Path(name) / "20260730_141954"
            run.mkdir()
            _build(run)

            for solution_id, _, _, _ in PLAN:
                os.utime(run / solution_id / "metadata.json", (START, START))

            self.assertTrue(load(run).timestamps_are_suspect)

    def test_a_run_with_real_timestamps_is_not(self) -> None:
        assert _run is not None
        self.assertFalse(load(_run).timestamps_are_suspect)


if __name__ == "__main__":
    unittest.main()
