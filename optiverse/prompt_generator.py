"""Prompt construction.

Everything about *what to do* is assembled here: what the loop is, the problem,
where the agent works, which parents it has, and the task. What is left to the
generator is *how to work* — the tools it has, the format of a reply — because
that varies with what is driving the codebase.

The prompt carries neither source code nor scores. The parents are directories on
disk, each with its own `metadata.txt`, so a three-parent prompt does not carry
three whole codebases and the agent reads only what it decides to read.

Paths are relative. The generator starts the agent in its own codebase, so `.` is
the solution and `../references` is beside it — a run directory pasted in full
three times was noise the agent had to read past.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .config import Problem
from .search import SearchResult

OPENING = """You are one step of an automated search for a better solution to \
the problem below.

Earlier steps produced the parent solutions. You produce exactly one new
candidate. It is scored automatically once you finish, and lower scores are
better; you will not be told what yours was."""


@dataclass(frozen=True)
class PromptGeneratorContext:
    problem: Problem
    search_result: SearchResult

    references_directory: str
    """Where the parent copies are, relative to the codebase the agent works in.

    Relative because the agent reads it, and given rather than assumed because
    where a run puts its files is the optimizer's business, not the prompt's."""


class PromptGenerator(ABC):
    @abstractmethod
    def generate(self, context: PromptGeneratorContext) -> str: ...


class DefaultPromptGenerator(PromptGenerator):
    def generate(self, context: PromptGeneratorContext) -> str:
        sections: List[str] = [
            "# What you are doing",
            "",
            OPENING,
            "",
            "# The problem",
            "",
            context.problem.description.strip(),
            "",
            "# Your working directory",
            "",
            "You are in it, and it is **empty**. Whatever you leave here is your",
            "solution, and nothing outside it is.",
            "",
            *self._parents(context),
            "# Your task",
            "",
            context.search_result.task.strip(),
        ]

        return "\n".join(sections) + "\n"

    def _parents(self, context: PromptGeneratorContext) -> List[str]:
        solutions = context.search_result.solutions

        if not solutions:
            return []

        directory = context.references_directory

        lines = ["# The parent solutions", "", "You have your own copy of each:", ""]

        for solution_with_title in solutions:
            identifier = solution_with_title.solution.id
            lines.append(
                f"- `{directory}/{identifier}/` — {solution_with_title.title}. "
                f"`code/` is the solution, `metadata.txt` its score and metrics."
            )

        lines += [
            "",
            "To start from one of them:",
            "",
            "```",
            f"cp -r {directory}/<id>/code/. .",
            "```",
            "",
            "They are copies. Edit or delete them freely.",
            "",
        ]

        return lines
