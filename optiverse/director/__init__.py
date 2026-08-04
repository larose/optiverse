"""The idea seam.

A director is handed the state of the search and writes the next constraint. It
writes `constraint.md` in its working directory and nothing is returned but
metadata: the file *is* the output, the same way a programmer's output is a
codebase.

That indirection is not ceremony. A director driving a shell agent has no other
way to hand anything back, and routing the constraint through a file it validates
means an unusable one is something the agent is told to fix rather than something
the loop discovers afterwards.

It is not asked where to work. The policy has already drawn the node the
constraint hangs under and the code the programmer will open on, so the one thing
left to decide is the idea itself.

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
    """Whether the constraint currently on disk is usable, and what was wrong if
    not.

    The same contract the programmer's `validate` has, for the same reason: a
    director driving an agent is expected to expose this as a tool, so a bad
    constraint costs one correction rather than the iteration."""

    workdir: Path
    """Where `constraint.md` goes: the iteration's own directory. The director
    keeps no state of its own between iterations, so this is all it owns. It may
    read the rest of the run."""


@dataclass(frozen=True)
class DirectorResult:
    metrics: Dict[str, Union[int, float]]
    """Call counts and the like. Surface as m_* columns in solutions.csv."""

    tags: Dict[str, Union[int, str]]
    """Categorical outcomes, such as the agent's exit status."""


class Director(ABC):
    @abstractmethod
    def decide(self, context: DirectorContext) -> DirectorResult: ...
