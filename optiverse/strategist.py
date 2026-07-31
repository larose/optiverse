"""The planning seam.

A strategist is handed the state of the search and decides what to try next. It
writes `plan.json` in its working directory and nothing is returned but metadata:
the file *is* the output, the same way a generator's output is a codebase.

That indirection is not ceremony. A strategist driving a shell agent has no other
way to hand anything back, and routing the decision through a file it validates
means a malformed plan is something the agent is told to fix rather than
something the loop discovers afterwards.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Union

from .evaluator import ValidationResult


@dataclass(frozen=True)
class StrategistContext:
    log_path: Path
    prompt: str

    validate: Callable[[], ValidationResult]
    """Whether the plan currently on disk is usable, and what was wrong if not.

    The same contract the generator's `validate` has, for the same reason: a
    strategist driving an agent is expected to expose this as a tool, so a bad
    plan costs one correction rather than the iteration."""

    workdir: Path
    """Where `plan.json` and the strategist's own memory live. It may read the
    rest of the run directory, but this is the only part it owns."""


@dataclass(frozen=True)
class StrategistResult:
    metrics: Dict[str, Union[int, float]]
    """Cost and call counts. Surface as m_* columns in solutions.csv."""

    tags: Dict[str, Union[int, str]]
    """Categorical outcomes, such as the agent's exit status."""


class Strategist(ABC):
    @abstractmethod
    def decide(self, context: StrategistContext) -> StrategistResult: ...
