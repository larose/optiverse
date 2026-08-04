"""The shell the agent acts in, and the two tools that are not a shell command.

`validate` and `give_up` are answered here rather than by programs on the path,
so the evaluator's command is never anywhere the agent can read it, and so which
action is which is the tool's name rather than a guess about what a command line
meant.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from pathlib import Path
from typing import Any, Callable, Dict

from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted

from . import (
    GAVE_UP_EXIT_STATUS,
    GIVE_UP_TOOL_NAME,
    VALIDATE_TOOL_NAME,
    VALIDATED_EXIT_STATUS,
)
from ..evaluator import EvaluatorError, ValidationResult
from ..solution import codebase as codebase_helpers

UNCHANGED_NOTICE = (
    "\n\n[optiverse] This is valid, but nothing here differs from what you "
    "started with, so there is nothing to finish. Make a real change first."
)

EVALUATOR_FAILED_NOTICE = (
    "[optiverse] The evaluator could not be run, which is a problem with the "
    "setup rather than with your solution: {error}"
)


class ToolEnvironment(LocalEnvironment):
    """Answers the two tools that are not `bash`.

    They are answered here rather than by programs on the path, so the
    evaluator's command is never anywhere the agent can read it — `score` is not
    one word away from `validate` — and so which action is which is the tool's
    name rather than a guess about what a command line meant.

    `validate` is the ordinary way a turn ends. Once the work is correct, more
    work belongs to the outer loop: continuing would let the agent hill-climb,
    which costs tokens and quietly undoes diversification.

    Ending requires the tree to have *changed* as well as validated. The codebase
    arrives holding a copy of the parent, which is already valid, so without that
    guard an agent could finish by validating someone else's work.
    """

    def __init__(
        self,
        *,
        baseline_digest: str,
        codebase: Path,
        validate: Callable[[], ValidationResult],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._baseline_digest = baseline_digest
        self._codebase = codebase
        self._validate = validate
        self.validate_runs = 0

    def execute(
        self, action: Dict[str, Any], cwd: str = "", *, timeout: int | None = None
    ) -> Dict[str, Any]:
        tool = action.get("tool")

        if tool == VALIDATE_TOOL_NAME:
            return self._check()

        if tool == GIVE_UP_TOOL_NAME:
            raise Submitted(_exit(GAVE_UP_EXIT_STATUS, str(action.get("reason", ""))))

        return super().execute(action, cwd, timeout=timeout)

    def _check(self) -> Dict[str, Any]:
        """What the evaluator says — and the end of the turn, if it says enough.

        A refusal is an ordinary observation, so an agent that ran this too early
        or on something broken spends a step rather than the iteration.
        """
        self.validate_runs += 1

        try:
            result = self._validate()
        except EvaluatorError as error:
            # Not the candidate's fault, so it is reported rather than counted as
            # invalid, and the agent gets to keep working. It is not trapped
            # either: `give_up` is still there if the evaluator stays broken.
            return _result(EVALUATOR_FAILED_NOTICE.format(error=error), returncode=1)

        if not result.valid:
            return _result(result.log, returncode=1)

        if codebase_helpers.digest(self._codebase) == self._baseline_digest:
            return _result(result.log + UNCHANGED_NOTICE, returncode=1)

        raise Submitted(_exit(VALIDATED_EXIT_STATUS, ""))


def _result(output: str, *, returncode: int) -> Dict[str, Any]:
    """An observation shaped the way `LocalEnvironment.execute` shapes one."""
    return {"output": output.strip(), "returncode": returncode, "exception_info": ""}


def _exit(status: str, submission: str) -> Dict[str, Any]:
    """The message `Submitted` carries out of the agent loop."""
    return {
        "role": "exit",
        "content": status,
        "extra": {"exit_status": status, "submission": submission},
    }
