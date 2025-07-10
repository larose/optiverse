from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

from .store import Store
from .config import Problem


@dataclass
class Context:
    iteration: int
    max_iterations: int
    problem: Problem
    store: Store


class PromptGenerator(ABC):
    @abstractmethod
    def generate(self, context: Context) -> str:
        pass


class EvolutionaryPromptGenerator(PromptGenerator):
    def generate(self, context: Context) -> str:
        all_solutions = context.store.get_all_solutions()
        sorted_solutions = list(sorted(all_solutions, key=lambda s: s.score))

        if not sorted_solutions:
            raise Exception("No solutions found in store")

        parent_solution = sorted_solutions[0]
        other_solutions = sorted_solutions[1:]
        random.shuffle(other_solutions)
        other_solutions = other_solutions[:3]

        # Build context
        solutions_context = f"## Parent solution\n\nScore: {parent_solution.score:.6f}\n\n"

        if parent_solution.description:
            solutions_context += f"### Description\n\n{parent_solution.description}\n\n"

        solutions_context += f"### Code\n\n```\n{parent_solution.code}\n```\n\n"

        if other_solutions:
            for i, solution in enumerate(other_solutions):
                solutions_context += f"## Solution {i+1} (for inspiration)\n\nScore: {solution.score:.6f}\n\n"

                if solution.description:
                    solutions_context += f"### Description\n\n{solution.description}\n\n"

                solutions_context += f"### Code\n\n```\n{solution.code}\n```\n\n"

        prompt = f"""
# Problem description

{context.problem.description}

# Solutions

{solutions_context}

# Task

Improve the parent solution.
"""

        return prompt
