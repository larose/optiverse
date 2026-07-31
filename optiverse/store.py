"""Persistence for the population.

A solution is a directory:

    <run>/solutions/<id>/code/          the codebase
    <run>/solutions/<id>/references/    copies of the parents it was built from
    <run>/solutions/<id>/agent.log      the agent's trajectory
    <run>/solutions/<id>/metadata.json  score, metrics, tags, timing

Solutions live under `solutions/` rather than at the run root so that the rest of
a run — `solutions.csv`, the strategist's own directory — can sit beside them
without `get_all_solutions` having to walk it and reject it.

Ids are allocated *before* generation so the agent can work directly in
`<id>/code/` rather than in a scratch directory that then has to be copied in.

Nothing here is made read-only. A parent is never handed to an agent in place —
it gets its own copy under `references/` — so there is nothing to protect.

`metadata.json` is written atomically and last, which makes its presence the
marker for a complete solution: a directory left behind by a crashed iteration
is skipped by `get_all_solutions` and stays on disk to be inspected.
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
REFERENCES_DIRECTORY_NAME = "references"
AGENT_LOG_NAME = "agent.log"
METADATA_NAME = "metadata.json"
SOLUTIONS_DIRECTORY_NAME = "solutions"
SOLUTIONS_CSV_NAME = "solutions.csv"


@dataclass(frozen=True)
class Solution:
    codebase: Path
    id: str
    is_initial: bool
    metrics: Dict[str, Union[float, int]]
    score: Optional[float]
    tags: Dict[str, Union[int, str]]

    started_at: str
    """When work on this solution began, ISO 8601 local time.

    A field rather than a metric because it describes the solution's production
    rather than its quality, and metrics are numeric-only — a timestamp would
    land in solutions.csv as an unreadable epoch float."""

    ended_at: str
    """When it was committed. Elapsed is the difference; storing it too would be
    a third column carrying no information the first two do not."""


class Store(ABC):
    @abstractmethod
    def allocate(self) -> str:
        """Create a new solution directory and return its id."""

    @abstractmethod
    def codebase_path(self, solution_id: str) -> Path: ...

    @abstractmethod
    def references_path(self, solution_id: str) -> Path:
        """Where the parents are copied for the agent to read.

        Outside `code/`, so a parent never becomes part of the solution built
        from it and is not inherited by children.
        """

    @abstractmethod
    def agent_log_path(self, solution_id: str) -> Path: ...

    @abstractmethod
    def commit(
        self,
        solution_id: str,
        *,
        is_initial: bool,
        metrics: Dict[str, Union[int, float]],
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
        solution_id = uuid.uuid4().hex
        self.codebase_path(solution_id).mkdir(parents=True)
        return solution_id

    def codebase_path(self, solution_id: str) -> Path:
        return self._solution_directory(solution_id) / CODE_DIRECTORY_NAME

    def references_path(self, solution_id: str) -> Path:
        return self._solution_directory(solution_id) / REFERENCES_DIRECTORY_NAME

    def agent_log_path(self, solution_id: str) -> Path:
        return self._solution_directory(solution_id) / AGENT_LOG_NAME

    def commit(
        self,
        solution_id: str,
        *,
        is_initial: bool,
        metrics: Dict[str, Union[int, float]],
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
            is_initial=is_initial,
            metrics=metrics,
            score=score,
            started_at=started_at,
            tags=tags,
        )

        metadata = {
            "ended_at": solution.ended_at,
            "id": solution.id,
            "is_initial": solution.is_initial,
            "metrics": solution.metrics,
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
                    is_initial=cast(bool, metadata["is_initial"]),
                    metrics=cast(Dict[str, Union[float, int]], metadata["metrics"]),
                    score=cast(Optional[float], metadata["score"]),
                    started_at=cast(str, metadata["started_at"]),
                    tags=cast(Dict[str, Union[int, str]], metadata["tags"]),
                )
            )

        return solutions

    def _write_solutions_csv(self) -> None:
        """Rewrite solutions.csv, best score first, failures last."""
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
                ["id", "score", "started_at", "ended_at"]
                + [f"t_{name}" for name in sorted_tag_names]
                + [f"m_{name}" for name in sorted_metric_names]
            )

            for solution in all_sorted:
                writer.writerow(
                    [
                        solution.id,
                        "failed" if solution.score is None else solution.score,
                        solution.started_at,
                        solution.ended_at,
                    ]
                    + [solution.tags.get(name) for name in sorted_tag_names]
                    + [solution.metrics.get(name) for name in sorted_metric_names]
                )
