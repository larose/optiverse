"""The generation seam.

A generator is handed a codebase directory holding a copy of the solution it is
improving, and changes it. Nothing is returned but metadata: the codebase *is*
the output, and it is already where it belongs.

`AgentProgrammer` is deliberately not re-exported here: importing it pulls
mini-swe-agent and its dependency tree, and the core is meant to import cleanly
without it. Import it directly:

    from optiverse.programmer.agent import AgentProgrammer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Union

from ..evaluator import ValidationResult


@dataclass(frozen=True)
class ProgrammerContext:
    codebase: Path
    """The working directory, holding a copy of the parent solution. Whatever
    ends up here is the new solution."""

    log_path: Path
    prompt: str

    validate: Callable[[], ValidationResult]
    """Whether `codebase` is currently valid, and what the evaluator said.

    The only way a generator gets to run the evaluator. A generator driving an
    agent is expected to expose this as a tool rather than as a command the agent
    types: the evaluator's path is then never disclosed, so `score` is not one
    word away from `validate`."""


@dataclass(frozen=True)
class ProgrammerResult:
    metrics: Dict[str, Union[int, float]]
    """Cost and call counts. Surface as m_* columns in solutions.csv."""

    tags: Dict[str, Union[int, str]]
    """Categorical outcomes, such as the agent's exit status."""


class Programmer(ABC):
    @abstractmethod
    def write(self, context: ProgrammerContext) -> ProgrammerResult: ...
