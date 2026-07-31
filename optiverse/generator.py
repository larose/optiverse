"""The generation seam.

A generator is handed an empty codebase directory and the parents it may build
from, and fills the directory in. Nothing is returned but metadata: the codebase
*is* the output, and it is already where it belongs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Union

from .evaluator import ValidationResult


@dataclass(frozen=True)
class GenerationContext:
    codebase: Path
    """The working directory. Empty; whatever ends up here is the solution."""

    log_path: Path
    prompt: str

    references_directory: Path
    """Copies of the parents, one directory per solution id, each holding `code/`
    and a `metadata.txt`. They are the generator's own copies, so it may do as it
    likes with them. Empty when the search had no parents to offer."""

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
