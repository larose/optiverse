"""Prompt construction.

The prompt describes *what to do*: the problem, what the parent solutions
achieved, and the task. It deliberately does not inline source code — the parents
are directories on disk, and the generator tells the agent where they are. That
keeps a three-parent prompt from carrying three whole codebases, and lets the
agent read only what it decides to read.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .config import Problem
from .search_strategies import SearchResult


@dataclass(frozen=True)
class PromptGeneratorContext:
    problem: Problem
    strategy_result: SearchResult


class PromptGenerator(ABC):
    @abstractmethod
    def generate(self, context: PromptGeneratorContext) -> str: ...


class DefaultPromptGenerator(PromptGenerator):
    def generate(self, context: PromptGeneratorContext) -> str:
        sections: List[str] = [
            "# Problem description",
            "",
            context.problem.description,
            "",
            "# Parent solutions",
            "",
        ]

        if not context.strategy_result.solutions:
            sections.append("None. Build a solution from the problem description.")
            sections.append("")

        for solution_with_title in context.strategy_result.solutions:
            solution = solution_with_title.solution

            sections.append(f"## {solution_with_title.title}")
            sections.append("")
            sections.append(f"Score: {solution.score}")
            sections.append("")

            if solution.metrics:
                sections.append("Metrics:")
                for name, value in solution.metrics.items():
                    sections.append(f"  - {name}: {value}")
                sections.append("")

        sections.append("# Task")
        sections.append("")
        sections.append(context.strategy_result.task)

        return "\n".join(sections) + "\n"
