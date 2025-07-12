from abc import ABC, abstractmethod
from pathlib import Path
import uuid
import json
import shutil
import csv
from typing import List, Dict, Optional, Set, Union, cast
from dataclasses import dataclass


@dataclass
class Solution:
    artifacts: Dict[str, str]
    code: str
    description: Optional[str]
    id: str
    is_initial: bool
    metrics: Dict[str, Union[float, int]]
    score: Optional[float]
    tags: Dict[str, Union[int, str]]


class Store(ABC):
    @abstractmethod
    def add_solution(
        self,
        artifacts: Dict[str, str],
        code: str,
        description: Optional[str],
        is_initial: bool,
        metrics: Dict[str, Union[int, float]],
        score: Optional[float],
        tags: Dict[str, Union[str, int]],
    ) -> str:
        pass

    @abstractmethod
    def remove_solution(self, solution_id: str) -> bool:
        pass

    @abstractmethod
    def get_all_solutions(self) -> List[Solution]:
        pass


class FileSystemStore(Store):
    def __init__(self, directory: Path):
        self._directory = directory
        self._directory.mkdir(exist_ok=True, parents=True)

    def _write_solutions_csv(self) -> None:
        """Write all solutions to solutions.csv file sorted by score (best first)."""
        solutions = self.get_all_solutions()

        # Separate valid solutions from failed solutions
        valid_solutions = [s for s in solutions if s.score is not None]
        failed_solutions = [s for s in solutions if s.score is None]

        # Sort valid solutions by score (best first)
        sorted_valid = sorted(valid_solutions, key=lambda x: cast(float, x.score))

        # Combine: valid solutions first, then failed solutions
        all_sorted = sorted_valid + failed_solutions

        # Collect all unique tag and metric names and sort them alphabetically
        all_tag_names: Set[str] = set()
        all_metric_names: Set[str] = set()
        for solution in all_sorted:
            all_tag_names.update(solution.tags.keys())
            all_metric_names.update(solution.metrics.keys())
        sorted_tag_names = sorted(all_tag_names)
        sorted_metric_names = sorted(all_metric_names)

        csv_path = self._directory / "solutions.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Create dynamic headers with t_{tag_name} and m_{metric_name} format
            tag_headers = [f"t_{tag_name}" for tag_name in sorted_tag_names]
            metric_headers = [f"m_{metric_name}" for metric_name in sorted_metric_names]
            writer.writerow(["id", "score"] + tag_headers + metric_headers)  # Header

            for solution in all_sorted:
                score_display = "FAILED" if solution.score is None else solution.score
                # Create row with tag values in the appropriate columns
                tag_values = [
                    solution.tags.get(tag_name) for tag_name in sorted_tag_names
                ]
                # Create row with metric values in the appropriate columns
                metric_values = [
                    solution.metrics.get(metric_name)
                    for metric_name in sorted_metric_names
                ]
                writer.writerow(
                    [solution.id, score_display] + tag_values + metric_values
                )

    def add_solution(
        self,
        artifacts: Dict[str, str],
        code: str,
        description: Optional[str],
        is_initial: bool,
        metrics: Dict[str, Union[int, float]],
        score: Optional[float],
        tags: Dict[str, Union[str, int]],
    ) -> str:
        id = uuid.uuid4().hex
        solution_dir = self._directory / id
        solution_dir.mkdir(parents=True)

        # Save the solution code
        solution_path = solution_dir / "solution.txt"
        with open(solution_path, "w") as f:
            f.write(code)

        # Save description if provided
        if description is not None:
            description_path = solution_dir / "description.txt"
            with open(description_path, "w") as f:
                f.write(description)

        # Save artifact files
        for artifact_name, artifact_content in artifacts.items():
            artifact_path = solution_dir / artifact_name
            with open(artifact_path, "w") as f:
                f.write(artifact_content)

        # Save metadata
        meta = {
            "id": id,
            "is_initial": is_initial,
            "metrics": metrics,
            "score": score,
            "tags": tags,
        }
        meta_file = solution_dir / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        self._write_solutions_csv()

        return id

    def remove_solution(self, solution_id: str) -> bool:
        solution_dir = self._directory / solution_id
        if not solution_dir.exists():
            return False

        shutil.rmtree(solution_dir)
        self._write_solutions_csv()

        return True

    def get_all_solutions(self) -> List[Solution]:
        solutions: List[Solution] = []

        if not self._directory.exists():
            return solutions

        # Load all solutions from disk
        for solution_dir in self._directory.iterdir():
            if solution_dir.is_dir():
                meta_file = solution_dir / "metadata.json"
                solution_file = solution_dir / "solution.txt"

                # Load metadata
                with open(meta_file, "r") as f:
                    meta = json.load(f)

                # Load solution code
                with open(solution_file, "r") as f:
                    file_content = f.read()

                # Load description if exists
                description_path = solution_dir / "description.txt"
                description = None
                if description_path.exists():
                    with open(description_path, "r") as f:
                        description = f.read()

                # Load artifact files
                artifacts: Dict[str, str] = {}
                known_files = {"metadata.json", "solution.txt", "description.txt"}
                for artifact_file in solution_dir.iterdir():
                    if (
                        artifact_file.is_file()
                        and artifact_file.name not in known_files
                    ):
                        with open(artifact_file, "r") as f:
                            artifacts[artifact_file.name] = f.read()

                solution = Solution(
                    artifacts=artifacts,
                    code=file_content,
                    description=description,
                    id=meta["id"],
                    is_initial=meta["is_initial"],
                    metrics=meta["metrics"],
                    score=meta["score"],
                    tags=meta["tags"],
                )
                solutions.append(solution)

        return solutions
