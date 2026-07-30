"""Persistence for the population.

A solution is a directory:

    <run>/<id>/code/          the codebase
    <run>/<id>/references/    copies of the parents this solution was built from
    <run>/<id>/agent.log      the agent's trajectory
    <run>/<id>/metadata.json  score, metrics, tags

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
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, cast

CODE_DIRECTORY_NAME = "code"
REFERENCES_DIRECTORY_NAME = "references"
AGENT_LOG_NAME = "agent.log"
METADATA_NAME = "metadata.json"


@dataclass(frozen=True)
class Solution:
    codebase: Path
    id: str
    is_initial: bool
    metrics: Dict[str, Union[float, int]]
    score: Optional[float]
    tags: Dict[str, Union[int, str]]


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
        tags: Dict[str, Union[str, int]],
    ) -> None:
        """Finalise an allocated solution, making it visible to the loop."""

    @abstractmethod
    def get_all_solutions(self) -> List[Solution]: ...


class FileSystemStore(Store):
    def __init__(self, directory: Path) -> None:
        # Absolute, because these paths are handed to an agent that runs with its
        # own working directory, where a relative path names nothing. Not
        # resolved: symlinks stay as the caller wrote them.
        self._directory = Path(os.path.abspath(directory))

    def _solution_directory(self, solution_id: str) -> Path:
        return self._directory / solution_id

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
        tags: Dict[str, Union[str, int]],
    ) -> None:
        solution_directory = self._solution_directory(solution_id)

        if not solution_directory.is_dir():
            raise ValueError(f"Solution {solution_id} was never allocated")

        metadata = {
            "id": solution_id,
            "is_initial": is_initial,
            "metrics": metrics,
            "score": score,
            "tags": tags,
        }

        # Written atomically and last: its presence means the solution is whole.
        metadata_path = solution_directory / METADATA_NAME
        temporary_path = metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(metadata, indent=2))
        os.replace(temporary_path, metadata_path)

        self._write_solutions_csv()

    def get_all_solutions(self) -> List[Solution]:
        solutions: List[Solution] = []

        if not self._directory.exists():
            return solutions

        for solution_directory in sorted(self._directory.iterdir()):
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
                    id=cast(str, metadata["id"]),
                    is_initial=cast(bool, metadata["is_initial"]),
                    metrics=cast(Dict[str, Union[float, int]], metadata["metrics"]),
                    score=cast(Optional[float], metadata["score"]),
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

        csv_path = self._directory / "solutions.csv"
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["id", "score"]
                + [f"t_{name}" for name in sorted_tag_names]
                + [f"m_{name}" for name in sorted_metric_names]
            )

            for solution in all_sorted:
                writer.writerow(
                    [
                        solution.id,
                        "FAILED" if solution.score is None else solution.score,
                    ]
                    + [solution.tags.get(name) for name in sorted_tag_names]
                    + [solution.metrics.get(name) for name in sorted_metric_names]
                )
