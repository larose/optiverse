"""Run configuration.

There is nothing about models here. Optiverse knows about programmers; a
programmer knows about whatever produces code.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .director import Director
from .evaluator import EvaluatorCommand
from .programmer import Programmer
from .search import DEFAULT_KICKS


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
    programmer: Programmer
    max_iterations: int
    problem: Problem
    director: Director

    kicks: Path = field(default=DEFAULT_KICKS)
    """Ways to make the next constraint unlike the last one. One is drawn at
    random on half of all perturbations.

    Ships with the package because it is about how to search rather than about
    any one problem — the only part of a run that transfers unchanged to the
    next. Point it elsewhere to use your own."""
