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
        solutions_context = (
            f"Parent solution to improve (Score: {parent_solution.score:.6f}):\n"
        )
        if parent_solution.description:
            solutions_context += f"{parent_solution.description}\n"
        solutions_context += f"```\n{parent_solution.file}\n```\n"

        if other_solutions:
            solutions_context += "\nOther solutions for reference:\n"
            for i, solution in enumerate(other_solutions):
                solutions_context += f"Solution {i+1} (Score: {solution.score:.6f}):\n"
                if solution.description:
                    solutions_context += f"{solution.description}\n"
                solutions_context += f"```\n{solution.file}\n```\n"

        prompt = f"""
# Problem description

{context.problem.description}

# Solutions

{solutions_context}

# Task

Improve the parent solution.
"""

        return prompt
