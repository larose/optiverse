"""The evaluator contract.

An evaluator is a command, not a Python class, so it can be written in any
language:

    <command> validate <codebase_dir>   exit 0 = valid, non-zero = invalid.
                                        Prints diagnostics on any stream.
    <command> score    <codebase_dir>   stdout: {"score": <float|null>,
                                                 "metrics": {...}}
                                        stderr: log.

`validate` answers with its exit code alone. There is no payload, so there is no
score for the programmer to read: withholding it is structural, not a rule.

`score` is the asymmetric mode because it has a machine consumer. A `null` score
means the run completed but the candidate is unscoreable; a non-zero exit means
the evaluator itself broke, which is a different problem and must not be
mistaken for a bad candidate.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, cast

logger = logging.getLogger(__name__)

VALIDATE = "validate"
SCORE = "score"


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of a `validate` run.

    `log` carries both streams. The programmer never runs the evaluator itself — it
    calls a tool that runs it here — so this is the only way the diagnostics
    reach it, and a compiler error it cannot read is an iteration wasted.
    """

    valid: bool
    log: str


@dataclass(frozen=True)
class ScoreResult:
    """The outcome of a `score` run.

    `log` is the captured stderr. It is surfaced when a candidate fails to score
    and then dropped: it is reproducible from the stored codebase, so persisting
    it would only add stale copies.
    """

    score: Optional[float]
    metrics: Dict[str, Union[int, float]]
    log: str


class EvaluatorError(Exception):
    """The evaluator itself failed, as distinct from the candidate being bad."""


class EvaluatorCommand:
    def __init__(
        self,
        command: Sequence[str],
        *,
        score_timeout_seconds: float = 600.0,
        validate_timeout_seconds: float = 120.0,
    ) -> None:
        self._command = _resolve_program(command)
        self._score_timeout_seconds = score_timeout_seconds
        self._validate_timeout_seconds = validate_timeout_seconds

    def argv(self, mode: str, codebase: Path) -> List[str]:
        return [*self._command, mode, os.path.abspath(codebase)]

    def shell_command(self, mode: str, codebase: Path) -> str:
        """The command as a shell string, for the programmer's instructions."""
        return subprocess.list2cmdline(self.argv(mode, codebase))

    def _run(
        self, mode: str, codebase: Path, timeout_seconds: float
    ) -> "subprocess.CompletedProcess[str]":
        argv = self.argv(mode, codebase)

        try:
            return subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise EvaluatorError(
                f"Evaluator timed out after {timeout_seconds}s: {argv}"
            ) from error
        except OSError as error:
            raise EvaluatorError(f"Could not run evaluator: {argv}") from error

    def validate(self, codebase: Path) -> ValidationResult:
        """Whether the candidate is valid, and what the evaluator said about it.

        A non-zero exit means invalid. It cannot be distinguished from "the
        evaluator broke", which is acceptable here: validate only gates the
        agent's turn, and `score` remains authoritative.

        Both streams are joined, in the order a terminal would have shown them.
        An evaluator is free to diagnose on either, and whoever reads this
        has no way to ask for the other one.
        """
        result = self._run(VALIDATE, codebase, self._validate_timeout_seconds)

        return ValidationResult(
            valid=result.returncode == 0,
            log=(result.stdout + result.stderr).strip(),
        )

    def score(self, codebase: Path) -> ScoreResult:
        result = self._run(SCORE, codebase, self._score_timeout_seconds)

        if result.returncode != 0:
            raise EvaluatorError(
                f"Evaluator exited {result.returncode} for {codebase}:\n"
                f"{result.stderr.strip()}"
            )

        return _parse_score(result.stdout, log=result.stderr)


def _resolve_program(command: Sequence[str]) -> List[str]:
    """Make a command that names files runnable from any directory.

    The programmer runs in its own codebase, so a relative `./evaluate` — or the
    script in `[interpreter, script]`, which is the shape the examples use —
    would find nothing from there.

    An argument is rewritten only when it is a path that exists: a bare name is a
    `PATH` lookup and must stay one, and a flag or an argument that merely
    contains a slash is not ours to touch.

    Made absolute rather than resolved, because symlinks carry meaning here: a
    venv's `bin/python` points at the system interpreter, and following it would
    run the evaluator without the venv's packages.
    """
    argv = list(command)

    if not argv:
        raise EvaluatorError("Evaluator command is empty")

    separators = [os.sep] + ([os.altsep] if os.altsep else [])

    for index, argument in enumerate(argv):
        if argument.startswith("-"):
            continue

        if not any(separator in argument for separator in separators):
            continue

        if Path(argument).exists():
            argv[index] = os.path.abspath(argument)

    return argv


def _parse_score(stdout: str, *, log: str) -> ScoreResult:
    try:
        parsed = cast(object, json.loads(stdout))
    except json.JSONDecodeError as error:
        raise EvaluatorError(
            f"Evaluator did not print valid JSON on stdout: {stdout!r}"
        ) from error

    if not isinstance(parsed, dict):
        raise EvaluatorError(f"Evaluator JSON must be an object, got {stdout!r}")

    payload = cast(Dict[str, object], parsed)

    if "score" not in payload:
        raise EvaluatorError(f"Evaluator JSON has no 'score' key: {stdout!r}")

    return ScoreResult(
        score=_parse_number(payload["score"], "score", allow_none=True),
        metrics=_parse_metrics(payload),
        log=log,
    )


def _parse_number(
    value: object, label: str, *, allow_none: bool = False
) -> Optional[float]:
    if value is None and allow_none:
        return None

    # bool is an int subclass, so it has to be rejected explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluatorError(f"Evaluator {label!r} must be a number, got {value!r}")

    return float(value)


def _parse_metrics(payload: Dict[str, object]) -> Dict[str, Union[int, float]]:
    raw_metrics = payload.get("metrics", {})

    if not isinstance(raw_metrics, dict):
        raise EvaluatorError(f"Evaluator 'metrics' must be an object: {raw_metrics!r}")

    metrics: Dict[str, Union[int, float]] = {}

    for key, value in cast(Dict[str, object], raw_metrics).items():
        number = _parse_number(value, f"metrics.{key}")
        assert number is not None
        metrics[key] = number

    return metrics
