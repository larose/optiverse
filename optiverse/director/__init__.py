"""The planning seam.

A director is handed the state of the search and decides what to try next. It
writes `plan.json` in its working directory and nothing is returned but metadata:
the file *is* the output, the same way a generator's output is a codebase.

That indirection is not ceremony. A director driving a shell agent has no other
way to hand anything back, and routing the decision through a file it validates
means a malformed plan is something the agent is told to fix rather than
something the loop discovers afterwards.

`AgentDirector` is deliberately not re-exported here, for the same reason
`AgentProgrammer` is not: importing it pulls mini-swe-agent and its dependency
tree, and the core is meant to import cleanly without it. Import it directly:

    from optiverse.director.agent import AgentDirector
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Union

from ..evaluator import ValidationResult


@dataclass(frozen=True)
class DirectorContext:
    log_path: Path
    prompt: str

    validate: Callable[[], ValidationResult]
    """Whether the plan currently on disk is usable, and what was wrong if not.

    The same contract the generator's `validate` has, for the same reason: a
    director driving an agent is expected to expose this as a tool, so a bad
    plan costs one correction rather than the iteration."""

    workdir: Path
    """Where `plan.json` goes: the iteration's own directory. The director keeps
    no state of its own between iterations, so this is all it owns. It may read
    the rest of the run."""


@dataclass(frozen=True)
class DirectorResult:
    metrics: Dict[str, Union[int, float]]
    """Call counts and the like. Surface as m_* columns in solutions.csv."""

    tags: Dict[str, Union[int, str]]
    """Categorical outcomes, such as the agent's exit status."""


class Director(ABC):
    @abstractmethod
    def decide(self, context: DirectorContext) -> DirectorResult: ...
