from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Union, cast
from .store import Solution, Store


@dataclass
class StrategyContext:
    iteration: int
    store: Store


@dataclass
class SolutionWithTitle:
    solution: Solution

    title: str


@dataclass
class StrategyResult:
    solutions: List[SolutionWithTitle]
    tags: Dict[str, Union[int, str]]
    task: str


class Strategy(ABC):
    @abstractmethod
    def apply(self, context: StrategyContext) -> StrategyResult:
        pass

    @abstractmethod
    def result(self, iteration: int, score: Optional[float]) -> None:
        pass


def get_initial_solution(solutions: List[Solution]) -> Solution:
    for solution in solutions:
        if solution.is_initial:
            return solution

    raise ValueError("No initial solution found.")


class IteratedLocalSearch(Strategy):
    def __init__(self, max_iterations_without_improvements: int):
        self._phase = 0
        self._best_score = float("inf")
        self._num_iterations_without_improvements = 0
        self._max_iterations_without_improvements = max_iterations_without_improvements

    def _improve(self, solutions: List[Solution]) -> StrategyResult:
        solutions_in_current_phase = [
            s
            for s in solutions
            if s.score is not None and s.tags.get("phase") == self._phase
        ]

        if not solutions_in_current_phase:
            return self._initial_solution(solutions)

        sorted_solutions = sorted(
            solutions_in_current_phase, key=lambda x: cast(float, x.score)
        )

        return StrategyResult(
            solutions=[SolutionWithTitle(solution=sorted_solutions[0], title="Parent")],
            tags={"move": "exploitation", "phase": self._phase},
            task="Improve the parent solution",
        )

    def _initial_solution(self, solutions: List[Solution]) -> StrategyResult:
        initial_solution = get_initial_solution(solutions)
        return StrategyResult(
            solutions=[SolutionWithTitle(solution=initial_solution, title="Parent")],
            tags={"move": "exploitation", "phase": self._phase},
            task="Improve the parent solution",
        )

    def _perturb(self, solutions: List[Solution]) -> StrategyResult:
        self._best_score = float("inf")
        self._phase += 1
        self._num_iterations_without_improvements = 0

        valid_solutions = [s for s in solutions if s.score is not None]
        if not valid_solutions:
            return self._initial_solution(solutions)

        parents = random.choices(
            valid_solutions,
            weights=[1 / cast(float, s.score) for s in valid_solutions],
            k=3,
        )

        solution_with_titles = [
            SolutionWithTitle(solution=s, title=f"Solution {i + 1}")
            for i, s in enumerate(parents)
        ]

        return StrategyResult(
            solutions=solution_with_titles,
            tags={
                "move": "exploration",
                "phase": self._phase,
            },
            task=f"Find a new solution based on these {len(solution_with_titles)} solutions",
        )

    def apply(self, context: StrategyContext) -> StrategyResult:
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
