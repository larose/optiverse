"""The generation seam.

A generator is handed a codebase directory that already contains the chosen
parent, and edits it in place. Nothing is returned but metadata: the codebase
*is* the output, and it is already where it belongs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


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

    description_path: Path
    """Where the generator may leave a summary. Outside `codebase`, so a
    description never becomes part of the solution it describes."""

    log_path: Path
    prompt: str
    references: List[ReferenceCodebase]
    validate_shell_command: str
    """Exactly what the agent should run to check its work."""


@dataclass(frozen=True)
class GenerationResult:
    metrics: Dict[str, Union[int, float]]
    """Cost and call counts. Surface as m_* columns in solutions.csv."""

    tags: Dict[str, Union[int, str]]
    """Categorical outcomes, such as the agent's exit status."""

    description: Optional[str] = None
    """Set only if the generator did not write to `GenerationContext.description_path`."""


class Generator(ABC):
    @abstractmethod
    def generate(self, context: GenerationContext) -> GenerationResult: ...
