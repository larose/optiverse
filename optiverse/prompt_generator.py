from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import List

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


class EvolutionaryPromptGenerator(PromptGenerator):
    def generate(self, context: Context) -> PromptResult:
        all_solutions = context.store.get_all_solutions()

        # Filter out solutions with None scores (failed solutions)
        valid_solutions = [s for s in all_solutions if s.score is not None]
        sorted_solutions: List[Solution] = list(
            sorted(valid_solutions, key=lambda s: s.score)
        )

        if not sorted_solutions:
            raise Exception("No valid solutions found in store")

        parent_solution = sorted_solutions[0]
        other_solutions = sorted_solutions[1:]
        random.shuffle(other_solutions)
        other_solutions = other_solutions[:3]

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

        return PromptResult(group=parent_solution.group, text=text)
