from abc import ABC, abstractmethod
from pathlib import Path
import uuid
import json
import shutil
import csv
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Solution:
    file: str
    score: float
    id: str
    description: Optional[str]


class Store(ABC):
    @abstractmethod
    def add_solution(self, file: str, score: float, description: Optional[str]) -> str:
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

        # Sort solutions by score for CSV (best first)
        sorted_solutions = sorted(solutions, key=lambda x: x.score)

        csv_path = self._directory / "solutions.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "score"])  # Header
            for solution in sorted_solutions:
                writer.writerow([solution.id, solution.score])

    def add_solution(self, file: str, score: float, description: Optional[str]) -> str:
        """Add a solution and return its ID."""
        solution_id = uuid.uuid4().hex
        solution_dir = self._directory / solution_id
        solution_dir.mkdir(parents=True, exist_ok=True)

        # Save the solution code
        solution_path = solution_dir / "solution.txt"
        with open(solution_path, "w") as f:
            f.write(file)

        # Save description if provided
        if description is not None:
            description_path = solution_dir / "description.txt"
            with open(description_path, "w") as f:
                f.write(description)

        # Save metadata
        meta = {"id": solution_id, "score": score}
        meta_file = solution_dir / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        # Update CSV file
        self._write_solutions_csv()

        return solution_id

    def remove_solution(self, solution_id: str) -> bool:
        """Remove a solution by ID. Returns True if found and removed."""
        solution_dir = self._directory / solution_id
        if not solution_dir.exists():
            return False

        shutil.rmtree(solution_dir)

        # Update CSV file
        self._write_solutions_csv()

        return True

    def get_all_solutions(self) -> List[Solution]:
        """Get all solutions."""
        solutions = []

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

                solution = Solution(
                    file_content, meta["score"], meta["id"], description
                )
                solutions.append(solution)

        return solutions

    def get_all_solutions_with_scores(self) -> List[Dict[str, any]]:
        """Get all solutions with their scores as dictionaries."""
        solutions = self.get_all_solutions()
        return [{"id": sol.id, "score": sol.score} for sol in solutions]
