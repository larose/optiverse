"""Persistence for the population.

A solution is a directory, and it holds the product and nothing else:

    <run>/solutions/<id>/code/          the codebase
    <run>/solutions/<id>/metadata.json  lineage, score, metrics, tags, timing

Everything about how it was made — the prompts, the trajectories, the plan — is
under `<run>/iterations/<n>/`, because that is the process rather than the
product.

A solution has two parents and they are independent. `node_id` says which *idea*
it attempts; `parent_solution_id` says which *code* it started from. Keeping them
apart is what lets the idea tree stay a tree while the code cross-pollinates.

Ids are allocated *before* generation so the agent can work directly in
`<id>/code/`, and they carry an `s_` prefix so a solution id can never be passed
where a node id belongs.

`metadata.json` is written atomically and last, which makes its presence the
marker for a complete solution — and, since every iteration produces exactly one
solution, the marker for a complete iteration. A directory left behind by a
crashed iteration is skipped by `get_all_solutions` and stays on disk to be
inspected.
"""

import csv
import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, cast

CODE_DIRECTORY_NAME = "code"
METADATA_NAME = "metadata.json"
SOLUTIONS_DIRECTORY_NAME = "solutions"
SOLUTIONS_CSV_NAME = "solutions.csv"

SOLUTION_ID_PREFIX = "s_"


@dataclass(frozen=True)
class Solution:
    codebase: Path
    id: str
    metrics: Dict[str, Union[float, int]]
    score: Optional[float]
    tags: Dict[str, Union[int, str]]

    node_id: str
    """The idea this attempts: a node in the search graph."""

    parent_solution_id: Optional[str]
    """The code this started from. None only for the seed, which started from
    the problem's initial codebase rather than from a solution."""

    iteration: Optional[int]
    """Which iteration produced it. None for the seed, which no iteration did —
    so this doubles as the marker of which solution is the seed, and there is no
    separate `is_initial` flag to keep in step with it."""

    started_at: str
    """When work on this solution began, ISO 8601 local time.

    A field rather than a metric because it describes the solution's production
    rather than its quality, and metrics are numeric-only — a timestamp would
    land in solutions.csv as an unreadable epoch float."""

    ended_at: str
    """When it was committed. Elapsed is the difference; storing it too would be
    a third column carrying no information the first two do not."""

    @property
    def is_initial(self) -> bool:
        return self.iteration is None


class Store(ABC):
    @abstractmethod
    def allocate(self) -> str:
        """Create a new solution directory and return its id."""

    @abstractmethod
    def codebase_path(self, solution_id: str) -> Path: ...

    @abstractmethod
    def commit(
        self,
        solution_id: str,
        *,
        iteration: Optional[int],
        metrics: Dict[str, Union[int, float]],
        node_id: str,
        parent_solution_id: Optional[str],
        score: Optional[float],
        started_at: str,
        tags: Dict[str, Union[str, int]],
    ) -> Solution:
        """Finalise an allocated solution, making it visible to the loop.

        Returns what it wrote, so a caller wanting to record the iteration does
        not have to reassemble it or read it back off disk.
        """

    @abstractmethod
    def get_all_solutions(self) -> List[Solution]: ...


class FileSystemStore(Store):
    def __init__(self, directory: Path) -> None:
        # Absolute, because these paths become an agent's working directory and
        # the argument to an evaluator subprocess, neither of which can be
        # trusted to run from here. The prompt is relative; this is not. Not
        # resolved, either: symlinks stay as the caller wrote them.
        self._directory = Path(os.path.abspath(directory))

    def _solutions_directory(self) -> Path:
        return self._directory / SOLUTIONS_DIRECTORY_NAME

    def _solution_directory(self, solution_id: str) -> Path:
        return self._solutions_directory() / solution_id

    def allocate(self) -> str:
        solution_id = SOLUTION_ID_PREFIX + uuid.uuid4().hex
        self.codebase_path(solution_id).mkdir(parents=True)
        return solution_id

    def codebase_path(self, solution_id: str) -> Path:
        return self._solution_directory(solution_id) / CODE_DIRECTORY_NAME

    def commit(
        self,
        solution_id: str,
        *,
        iteration: Optional[int],
        metrics: Dict[str, Union[int, float]],
        node_id: str,
        parent_solution_id: Optional[str],
        score: Optional[float],
        started_at: str,
        tags: Dict[str, Union[str, int]],
    ) -> Solution:
        solution_directory = self._solution_directory(solution_id)

        if not solution_directory.is_dir():
            raise ValueError(f"Solution {solution_id} was never allocated")

        solution = Solution(
            codebase=self.codebase_path(solution_id),
            ended_at=datetime.now().isoformat(timespec="seconds"),
            id=solution_id,
            iteration=iteration,
            metrics=metrics,
            node_id=node_id,
            parent_solution_id=parent_solution_id,
            score=score,
            started_at=started_at,
            tags=tags,
        )

        metadata = {
            "ended_at": solution.ended_at,
            "id": solution.id,
            "iteration": solution.iteration,
            "metrics": solution.metrics,
            "node_id": solution.node_id,
            "parent_solution_id": solution.parent_solution_id,
            "score": solution.score,
            "started_at": solution.started_at,
            "tags": solution.tags,
        }

        # Written atomically and last: its presence means the solution is whole.
        metadata_path = solution_directory / METADATA_NAME
        temporary_path = metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(metadata, indent=2))
        os.replace(temporary_path, metadata_path)

        self._write_solutions_csv()

        return solution

    def get_all_solutions(self) -> List[Solution]:
        solutions: List[Solution] = []

        solutions_directory = self._solutions_directory()

        if not solutions_directory.exists():
            return solutions

        for solution_directory in sorted(solutions_directory.iterdir()):
            if not solution_directory.is_dir():
                continue

            metadata_path = solution_directory / METADATA_NAME

            # A directory without metadata is an iteration that died partway
            # through. Skip it; it stays on disk for debugging.
            if not metadata_path.is_file():
                continue

            metadata = cast(Dict[str, object], json.loads(metadata_path.read_text()))

            solutions.append(
                Solution(
                    codebase=solution_directory / CODE_DIRECTORY_NAME,
                    ended_at=cast(str, metadata["ended_at"]),
                    id=cast(str, metadata["id"]),
                    iteration=cast(Optional[int], metadata["iteration"]),
                    metrics=cast(Dict[str, Union[float, int]], metadata["metrics"]),
                    node_id=cast(str, metadata["node_id"]),
                    parent_solution_id=cast(
                        Optional[str], metadata["parent_solution_id"]
                    ),
                    score=cast(Optional[float], metadata["score"]),
                    started_at=cast(str, metadata["started_at"]),
                    tags=cast(Dict[str, Union[int, str]], metadata["tags"]),
                )
            )

        # By iteration, so the last of these is the last thing the search did.
        # The seed has none and sorts first, which is where it belongs.
        return sorted(
            solutions, key=lambda s: -1 if s.iteration is None else s.iteration
        )

    def _write_solutions_csv(self) -> None:
        """Rewrite solutions.csv, best score first, failures last.

        Lineage gets columns here where the old branch did not: two ids join
        cleanly, where a list of constraint paragraphs never would.
        """
        solutions = self.get_all_solutions()

        valid_solutions = [s for s in solutions if s.score is not None]
        failed_solutions = [s for s in solutions if s.score is None]

        sorted_valid = sorted(valid_solutions, key=lambda s: cast(float, s.score))
        all_sorted = sorted_valid + failed_solutions

        tag_names: Set[str] = set()
        metric_names: Set[str] = set()
        for solution in all_sorted:
            tag_names.update(solution.tags.keys())
            metric_names.update(solution.metrics.keys())

        sorted_tag_names = sorted(tag_names)
        sorted_metric_names = sorted(metric_names)

        csv_path = self._directory / SOLUTIONS_CSV_NAME
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "id",
                    "score",
                    "iteration",
                    "node_id",
                    "parent_solution_id",
                    "started_at",
                    "ended_at",
                ]
                + [f"t_{name}" for name in sorted_tag_names]
                + [f"m_{name}" for name in sorted_metric_names]
            )

            for solution in all_sorted:
                writer.writerow(
                    [
                        solution.id,
                        "failed" if solution.score is None else solution.score,
                        solution.iteration,
                        solution.node_id,
                        solution.parent_solution_id,
                        solution.started_at,
                        solution.ended_at,
                    ]
                    + [solution.tags.get(name) for name in sorted_tag_names]
                    + [solution.metrics.get(name) for name in sorted_metric_names]
                )
