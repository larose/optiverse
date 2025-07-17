from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from .search_strategies import SearchResult
from .config import Problem


@dataclass
class PromptGeneratorContext:
    problem: Problem
    strategy_result: SearchResult


class PromptGenerator(ABC):
    @abstractmethod
    def generate(self, context: PromptGeneratorContext) -> str:
        pass


class DefaultPromptGenerator(PromptGenerator):

    def generate(self, context: PromptGeneratorContext) -> str:
        solutions_section_buffer: List[str] = []

        for solution_with_title in context.strategy_result.solutions:
            title = solution_with_title.title
            solution = solution_with_title.solution

            solutions_section_buffer.append(f"## {title}")
            solutions_section_buffer.append("")
            solutions_section_buffer.append(f"Score: {solution.score}")
            solutions_section_buffer.append("")

            if solution.metrics:
                solutions_section_buffer.append("Metrics:")

                for metric_name, metric_value in solution.metrics.items():
                    solutions_section_buffer.append(
                        f"  - {metric_name}: {metric_value}"
                    )

                solutions_section_buffer.append("")

            if solution.description is not None:
                solutions_section_buffer.append(f"### Description")
                solutions_section_buffer.append(solution.description)
                solutions_section_buffer.append("")

            solutions_section_buffer.append(f"### Code")
            solutions_section_buffer.append(f"```\n{solution.code}\n```")
            solutions_section_buffer.append("")

        solutions_section = "\n".join(solutions_section_buffer)

        text = f"""# Problem description

{context.problem.description}

# Solutions

{solutions_section}

# Task

{context.strategy_result.task}
"""

        return text
