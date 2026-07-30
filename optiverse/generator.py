"""The generation seam.

A generator is handed a codebase directory that already contains the chosen
parent, and edits it in place. Nothing is returned but metadata: the codebase
*is* the output, and it is already where it belongs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union


@dataclass(frozen=True)
class ReferenceCodebase:
    """A parent the agent may read but must not edit."""

    path: Path
    title: str
    score: Optional[float]


@dataclass(frozen=True)
class GenerationContext:
    codebase: Path
    """The working directory. Seeded with the primary parent; edited in place."""

    log_path: Path
    prompt: str
    references: List[ReferenceCodebase]

    validate: Callable[[], bool]
    """Whether `codebase` is currently valid.

    The authoritative answer, for a generator that ends a turn on validity. The
    agent's own invocation cannot be trusted for that: it may name a different
    directory, and its exit code says nothing about which one it checked."""

    validate_shell_command: str
    """Exactly what the agent should run to check its work."""


@dataclass(frozen=True)
class GenerationResult:
    metrics: Dict[str, Union[int, float]]
    """Cost and call counts. Surface as m_* columns in solutions.csv."""

    tags: Dict[str, Union[int, str]]
    """Categorical outcomes, such as the agent's exit status."""


class Generator(ABC):
    @abstractmethod
    def generate(self, context: GenerationContext) -> GenerationResult: ...
