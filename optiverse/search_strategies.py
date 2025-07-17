from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Union, cast
from .store import Solution, Store


@dataclass
class SearchContext:
    iteration: int
    store: Store


@dataclass
class SolutionWithTitle:
    solution: Solution

    title: str


@dataclass
class SearchResult:
    solutions: List[SolutionWithTitle]
    tags: Dict[str, Union[int, str]]
    task: str


class SearchStrategy(ABC):
    @abstractmethod
    def apply(self, context: SearchContext) -> SearchResult:
        pass

    @abstractmethod
    def result(self, iteration: int, score: Optional[float]) -> None:
        pass


def get_initial_solution(solutions: List[Solution]) -> Solution:
    for solution in solutions:
        if solution.is_initial:
            return solution

    raise ValueError("No initial solution found.")


class IteratedLocalSearch(SearchStrategy):
    def __init__(self, max_iterations_without_improvements: int):
        self._group = 0
        self._best_score = float("inf")
        self._num_iterations_without_improvements = 0
        self._max_iterations_without_improvements = max_iterations_without_improvements

    def _improve(self, solutions: List[Solution]) -> SearchResult:
        solutions_in_current_group = [
            s
            for s in solutions
            if s.score is not None and s.tags.get("group") == self._group
        ]

        if not solutions_in_current_group:
            return self._initial_solution(solutions)

        sorted_solutions = sorted(
            solutions_in_current_group, key=lambda x: cast(float, x.score)
        )

        return SearchResult(
            solutions=[SolutionWithTitle(solution=sorted_solutions[0], title="Parent")],
            tags={"move": "improve", "group": self._group},
            task="Improve the parent solution",
        )

    def _initial_solution(self, solutions: List[Solution]) -> SearchResult:
        initial_solution = get_initial_solution(solutions)
        return SearchResult(
            solutions=[SolutionWithTitle(solution=initial_solution, title="Parent")],
            tags={"move": "initial", "group": self._group},
            task="Improve the parent solution",
        )

    def _perturb(self, solutions: List[Solution]) -> SearchResult:
        self._best_score = float("inf")
        self._group += 1
        self._num_iterations_without_improvements = 0

        if random.random() < 0.1:
            return self._initial_solution(solutions)

        valid_solutions = [s for s in solutions if s.score is not None]
        if not valid_solutions:
            return self._initial_solution(solutions)

        # Group solutions by their group tag
        groups: Dict[Union[str, int], List[Solution]] = {}
        for solution in valid_solutions:
            group_id = solution.tags.get("group")
            if group_id is None:
                continue

            if group_id not in groups:
                groups[group_id] = []

            groups[group_id].append(solution)

        # Pick the best solution from each group (lowest score)
        best_from_each_group: List[Solution] = []
        for group_solutions in groups.values():
            best_solution = min(group_solutions, key=lambda s: cast(float, s.score))
            best_from_each_group.append(best_solution)

        # Randomly pick 3 solutions from these best solutions
        num_to_pick = min(3, len(best_from_each_group))
        if num_to_pick == 0:
            raise Exception("No valid solutions to pick from.")

        parents = random.sample(best_from_each_group, num_to_pick)

        solution_with_titles = [
            SolutionWithTitle(solution=s, title=f"Solution {i + 1}")
            for i, s in enumerate(parents)
        ]

        return SearchResult(
            solutions=solution_with_titles,
            tags={
                "move": "perturb",
                "group": self._group,
            },
            task=f"Create a new solution based on these {len(solution_with_titles)} solutions",
        )

    def apply(self, context: SearchContext) -> SearchResult:
        solutions = context.store.get_all_solutions()

        if (
            self._num_iterations_without_improvements
            < self._max_iterations_without_improvements
        ):
            return self._improve(solutions)

        return self._perturb(solutions)

    def result(self, iteration: int, score: Optional[float]) -> None:
        if score is not None and score < self._best_score:
            self._best_score = score
            self._num_iterations_without_improvements = 0
        else:
            self._num_iterations_without_improvements += 1
