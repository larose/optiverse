"""Run configuration.

There is nothing about models here. Optiverse knows about generators; a
generator knows about whatever produces code.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .evaluator import EvaluatorCommand
from .generator import Generator
from .director import Director

DEFAULT_PLAYBOOK = Path(__file__).parent / "playbook.md"


@dataclass(frozen=True)
class Problem:
    description: str
    initial_codebase: Path
    """Seed codebase. Copied for the first solution; never modified."""

    evaluate_command: Sequence[str]
    """Invoked as `<command> validate|score <codebase_dir>`. Any language."""

    score_timeout_seconds: float = 600.0
    validate_timeout_seconds: float = 120.0

    def evaluator(self) -> EvaluatorCommand:
        return EvaluatorCommand(
            self.evaluate_command,
            score_timeout_seconds=self.score_timeout_seconds,
            validate_timeout_seconds=self.validate_timeout_seconds,
        )


@dataclass(frozen=True)
class OptimizerConfig:
    directory: Path
    generator: Generator
    max_iterations: int
    problem: Problem
    director: Director

    playbook: Path = field(default=DEFAULT_PLAYBOOK)
    """Angles for inventing a constraint the search has not tried.

    Ships with the package because it is about how to search rather than about
    any one problem — the only part of a run that transfers unchanged to the
    next. Point it elsewhere to use your own."""
