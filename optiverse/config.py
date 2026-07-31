"""Run configuration.

There is nothing about models here. Optiverse knows about generators; a
generator knows about whatever produces code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .evaluator import EvaluatorCommand
from .generator import Generator
from .search_strategies import SearchStrategy


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
    search_strategy: SearchStrategy
