from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Tuple, cast, List
from .store import Solution, Store
from .config import Problem


@dataclass
class Context:
    iteration: int
    max_iterations: int
    problem: Problem
    store: Store


@dataclass
class PromptResult:
    group: int
    text: str


class PromptGenerator(ABC):
    @abstractmethod
    def generate(self, context: Context) -> PromptResult:
        pass


NUMBER_OF_GROUPS = 5


def get_initial_solution(solutions: List[Solution]) -> Solution:
    for solution in solutions:
        if solution.is_initial:
            return solution

    raise ValueError("No initial solution found.")


class EvolutionaryPromptGenerator(PromptGenerator):

    def _select_group(self, context: Context) -> int:
        iterations_per_group = context.max_iterations // NUMBER_OF_GROUPS
        group = context.iteration // iterations_per_group

        if group >= NUMBER_OF_GROUPS:
            raise Exception("Exceeded maximum number of groups")

        return group

    def _select_solutions_explore_groups(
        self, valid_solutions: List[Solution], group: int
    ) -> Tuple[Solution, List[Solution]]:
        solutions_in_group = [s for s in valid_solutions if s.group == group]
        if not solutions_in_group:
            return get_initial_solution(valid_solutions), []

        sorted_solutions = sorted(
            solutions_in_group, key=lambda s: cast(float, s.score)
        )

        parent_solution = sorted_solutions[0]
        other_solutions = sorted_solutions[1:]
        random.shuffle(other_solutions)
        other_solutions = other_solutions[:2]

        return parent_solution, other_solutions

    def _select_solutions(
        self,
        group: int,
        valid_solutions: List[Solution],
    ) -> Tuple[Solution, List[Solution]]:
        if group < NUMBER_OF_GROUPS - 1:
            return self._select_solutions_explore_groups(
                valid_solutions=valid_solutions,
                group=group,
            )

        sorted_solutions = sorted(valid_solutions, key=lambda s: cast(float, s.score))
        parent_solution = sorted_solutions[0]
        other_solutions = sorted_solutions[1:]
        other_solutions = other_solutions[:2]
        return parent_solution, other_solutions

    def generate(self, context: Context) -> PromptResult:
        group = self._select_group(context)
        all_solutions = context.store.get_all_solutions()
        valid_solutions = [s for s in all_solutions if s.score is not None]

        parent_solution, other_solutions = self._select_solutions(
            group=group,
            valid_solutions=valid_solutions,
        )

        # Build context
        solutions_context = (
            f"## Parent solution\n\nScore: {parent_solution.score:.6f}\n\n"
        )

        if parent_solution.description:
            solutions_context += f"### Description\n\n{parent_solution.description}\n\n"

        solutions_context += f"### Code\n\n```\n{parent_solution.code}\n```\n\n"

        if other_solutions:
            for i, solution in enumerate(other_solutions):
                solutions_context += f"## Solution {i+1} (for inspiration)\n\nScore: {solution.score:.6f}\n\n"

                if solution.description:
                    solutions_context += (
                        f"### Description\n\n{solution.description}\n\n"
                    )

                solutions_context += f"### Code\n\n```\n{solution.code}\n```\n\n"

        text = f"""
# Problem description

{context.problem.description}

# Solutions

{solutions_context}

# Task

Improve the parent solution.
"""

        return PromptResult(group=group, text=text)
