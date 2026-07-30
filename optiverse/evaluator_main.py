"""Boilerplate for evaluators that happen to be written in Python.

Optional. The contract is a process contract; nothing in the core depends on
this. It exists so a Python evaluator does not have to re-implement argv parsing
and JSON framing:

    if __name__ == "__main__":
        evaluator_main(score=score, validate=validate)
"""

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

Metrics = Dict[str, Union[int, float]]
ScoreFunction = Callable[[Path], Tuple[Optional[float], Metrics]]
ValidateFunction = Callable[[Path], bool]

USAGE = "usage: <program> {score|validate} <codebase_dir>"


def evaluator_main(
    *,
    score: ScoreFunction,
    validate: ValidateFunction,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Dispatch one evaluator invocation. Returns the process exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)

    if len(arguments) != 2:
        print(USAGE, file=sys.stderr)
        return 2

    mode, raw_codebase = arguments
    codebase = Path(raw_codebase)

    if not codebase.is_dir():
        print(f"not a directory: {codebase}", file=sys.stderr)
        return 2

    if mode == "validate":
        # The verdict is the exit code. Nothing is printed on stdout, so there is
        # no score for the agent to read.
        return 0 if validate(codebase) else 1

    if mode == "score":
        score_value, metrics = score(codebase)
        json.dump({"score": score_value, "metrics": metrics}, sys.stdout)
        sys.stdout.write("\n")
        return 0

    print(f"{USAGE}\nunknown mode: {mode}", file=sys.stderr)
    return 2


def run(score: ScoreFunction, validate: ValidateFunction) -> None:
    """`evaluator_main` plus `sys.exit`, for use under `if __name__`."""
    sys.exit(evaluator_main(score=score, validate=validate))
