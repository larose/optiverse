"""Prompt construction.

Everything about *what to do* is assembled here: what the loop is, the problem,
what is already in the working directory, and the constraints in force. What is
left to the generator is *how to work* — the tools it has — because that varies
with what is driving the codebase.

The prompt carries no scores at all, not the candidate's own and not its
parent's. It carries no source either: the parent's code is already in the
working directory, so a prompt does not have to hold a copy of it.

There is no task. The constraints are the whole instruction, which is what makes
them load-bearing rather than bookkeeping — they are the reason this candidate
will differ from its parent, and the search graph is a complete record precisely
because nothing else is said.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

OPENING = """You are one step of an automated search for a better solution to \
the problem below.

An earlier step produced the code in your working directory. You produce exactly
one new candidate by changing it. It is scored automatically once you finish,
and lower scores are better; you will not be told what yours was, or what
anything else scored."""

WORKING_DIRECTORY = """You are in it, and it already holds a copy of the solution
you are improving. Whatever you leave here is your candidate, and nothing outside
it is. Edit it, rewrite parts of it, or clear it out and start again — but it has
to end up different from what you found, or you cannot finish."""

CONSTRAINTS_OPENING = (
    "**Work within these constraints.** They are not optional, and they are the "
    "point of this attempt."
)


@dataclass(frozen=True)
class PromptGeneratorContext:
    problem_description: str
    """The problem statement, not the whole `Problem`.

    Nothing here ever wanted the seed codebase or the evaluator command, and
    taking the `Problem` would have this module import the configuration that
    imports this module's package."""

    constraints: Sequence[str]
    """Every constraint in force at the node this candidate belongs to, root
    first. Empty at the root, which is the whole space."""


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
            context.problem_description.strip(),
            "",
            "# Your working directory",
            "",
            WORKING_DIRECTORY,
            *self._constraints(context),
        ]

        return "\n".join(sections) + "\n"

    def _constraints(self, context: PromptGeneratorContext) -> List[str]:
        if not context.constraints:
            return []

        lines = ["", "# Your constraints", "", CONSTRAINTS_OPENING, ""]

        for constraint in context.constraints:
            lines.append(constraint.strip())
            lines.append("")

        return lines
