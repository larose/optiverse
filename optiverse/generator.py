"""The generation seam.

A generator is handed a codebase directory holding a copy of the solution it is
improving, and changes it. Nothing is returned but metadata: the codebase *is*
the output, and it is already where it belongs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Union

from .evaluator import ValidationResult


@dataclass(frozen=True)
class GenerationContext:
    codebase: Path
    """The working directory, holding a copy of the parent solution. Whatever
    ends up here is the new solution."""

    log_path: Path
    prompt: str

    remember: Callable[[str], None]
    """Record something worth carrying to a later iteration — a build rule, a
    constraint of the environment, a mistake that cost time.

    Every coding agent starts knowing nothing, and this is the only way anything
    it learns outlives its turn."""

    validate: Callable[[], ValidationResult]
    """Whether `codebase` is currently valid, and what the evaluator said.

    The only way a generator gets to run the evaluator. A generator driving an
    agent is expected to expose this as a tool rather than as a command the agent
    types: the evaluator's path is then never disclosed, so `score` is not one
    word away from `validate`."""


@dataclass(frozen=True)
class GenerationResult:
    metrics: Dict[str, Union[int, float]]
    """Cost and call counts. Surface as m_* columns in solutions.csv."""

    tags: Dict[str, Union[int, str]]
    """Categorical outcomes, such as the agent's exit status."""


class Generator(ABC):
    @abstractmethod
    def generate(self, context: GenerationContext) -> GenerationResult: ...
