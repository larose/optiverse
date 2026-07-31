"""mini-swe-agent glue. Imported only when the `agent` extra is installed.

mini-swe-agent annotates several signatures with bare `dict`, which strict mode
reports as partially unknown. The looseness is in the dependency, not here, so it
is suppressed for this file only.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

import litellm
import yaml
from minisweagent import package_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

from .. import codebase as codebase_helpers
from ..evaluator import EvaluatorError, ValidationResult

logger = logging.getLogger(__name__)

VALIDATED_EXIT_STATUS = "Validated"

# The one word the agent types instead of a command. Never reaches a shell.
VALIDATE_COMMAND = "validate"

UNCHANGED_NOTICE = (
    "\n[optiverse] This solution is valid, but it is byte-for-byte identical to "
    "the one you started from. Make a real change before checking again."
)

ARGUMENTS_NOTICE = (
    f"[optiverse] `{VALIDATE_COMMAND}` is a tool, not a program. Write it on its "
    "own, with no arguments, no path and no redirection."
)

EVALUATOR_FAILED_NOTICE = (
    "[optiverse] The evaluator could not be run, which is a problem with the "
    "setup rather than with your solution: {error}"
)

# How long a provider may ask us to wait before we treat it as an exhausted quota
# rather than a burst, and how many times we are willing to wait at all.
MAXIMUM_RATE_LIMIT_WAIT_SECONDS = 300.0
MAXIMUM_RATE_LIMIT_WAITS = 4

# A clock offset of a second or two turns "retry in 47s" into another rejection.
RATE_LIMIT_MARGIN_SECONDS = 1.0

# Gemini states the delay in the body; most providers use the retry-after header.
_RETRY_DELAY_PATTERNS = (
    re.compile(r'"retryDelay"\s*:\s*"([0-9.]+)s"'),
    re.compile(r"retry in ([0-9.]+)\s*s", re.IGNORECASE),
)


def default_agent_config() -> Dict[str, Any]:
    """The `agent` section of mini-swe-agent's shipped default config.

    Reused for `system_template`, which encodes the action format that the model
    layer parses. Rewriting it would risk silent format errors.
    """
    raw = yaml.safe_load((package_dir / "config" / "default.yaml").read_text())
    return cast(Dict[str, Any], cast(Dict[str, Any], raw)["agent"])


def suggested_delay(error: Exception) -> Optional[float]:
    """How long the provider asked us to wait, if it said.

    Guessing is what makes a retry storm: an exponential ladder starting at four
    seconds spends three doomed requests before it reaches a stated 47.
    """
    header_delay = _header_delay(error)
    if header_delay is not None:
        return header_delay

    message = str(error)

    for pattern in _RETRY_DELAY_PATTERNS:
        match = pattern.search(message)
        if match:
            return float(match.group(1))

    return None


def _header_delay(error: Exception) -> Optional[float]:
    """`Retry-After`, which OpenAI and Anthropic send and Gemini does not."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)

    if headers is None:
        return None

    raw_value = headers.get("retry-after")

    if raw_value is None:
        return None

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        # The HTTP-date form. Rare, and the body patterns usually cover it.
        return None


class RateLimitAwareModel(LitellmTextbasedModel):
    """Waits as long as the provider says, rather than guessing.

    mini-swe-agent retries on a fixed exponential ladder that never reads the
    response, so against a stated 47-second delay it spends its first three
    attempts on requests that cannot succeed, each one charged against the very
    quota it is waiting for.

    Rate limits are handled here and only here: `RateLimitError` is added to the
    abort list so the outer retry does not re-try what this loop already gave up
    on, which would otherwise multiply into dozens of requests.
    """

    abort_exceptions = [
        *LitellmModel.abort_exceptions,
        litellm.exceptions.RateLimitError,
    ]

    def __init__(self, *, sleep: Callable[[float], None] = time.sleep, **kwargs: Any):
        super().__init__(**kwargs)
        self._sleep = sleep

    def _query(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        for waits in range(MAXIMUM_RATE_LIMIT_WAITS + 1):
            try:
                return super()._query(messages, **kwargs)
            except litellm.exceptions.RateLimitError as error:
                if waits == MAXIMUM_RATE_LIMIT_WAITS:
                    raise

                delay = self._wait_for(error)

                if delay is None:
                    raise

                self._sleep(delay)

        raise AssertionError("unreachable")

    def _wait_for(self, error: Exception) -> Optional[float]:
        """The wait to take before retrying, or None to give up.

        A provider asking for minutes has run out of quota rather than hit a
        burst, and waiting it out would eat the agent's whole turn.
        """
        stated = suggested_delay(error)

        if stated is None:
            logger.info("Rate limited with no stated delay; not retrying")
            return None

        if stated > MAXIMUM_RATE_LIMIT_WAIT_SECONDS:
            logger.info(
                f"Rate limited for {stated:.0f}s, beyond the "
                f"{MAXIMUM_RATE_LIMIT_WAIT_SECONDS:.0f}s this is willing to wait"
            )
            return None

        delay = stated + RATE_LIMIT_MARGIN_SECONDS
        # Said out loud: a silent 47-second sleep is indistinguishable from a hang.
        logger.info(f"Rate limited; waiting the {stated:.0f}s the provider asked for")

        return delay


class ValidateTerminatesEnvironment(LocalEnvironment):
    """Adds a `validate` tool, and ends the agent's turn once it passes.

    `validate` is a word this class intercepts, not a program: it never reaches a
    shell. So the evaluator's path is never in the prompt and `score` is not one
    word away from `validate`, and recognising the check is string equality
    rather than a guess about what a command line meant.

    Once the code is correct, further work belongs to the outer loop: continuing
    would let the agent hill-climb, which costs tokens and quietly undoes
    diversification.

    Termination requires the tree to have *changed* as well as validated. A
    parent the agent copied in is already valid, so without that guard it could
    finish by validating someone else's work.

    This sits on top of mini-swe-agent's own submission protocol rather than
    replacing it: an agent that gives up can still exit via
    `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, and the resulting `exit_status`
    distinguishes the two paths.
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
        command = str(action.get("command", "")).strip()

        if command == VALIDATE_COMMAND:
            return self._run_validate()

        if _names_validate(command):
            # `validate .`, `./validate`, `validate | tail`. Saying so costs one
            # line; letting bash answer `command not found` costs a whole step.
            return _result(ARGUMENTS_NOTICE, returncode=1)

        return super().execute(action, cwd, timeout=timeout)

    def _run_validate(self) -> Dict[str, Any]:
        self.validate_runs += 1

        try:
            result = self._validate()
        except EvaluatorError as error:
            # Not the candidate's fault, so it is reported rather than counted
            # as invalid, and the agent gets to keep working.
            return _result(EVALUATOR_FAILED_NOTICE.format(error=error), returncode=1)

        if not result.valid:
            return _result(result.log, returncode=1)

        if codebase_helpers.digest(self._codebase) == self._baseline_digest:
            return _result(result.log + UNCHANGED_NOTICE, returncode=0)

        raise Submitted(
            {
                "role": "exit",
                "content": VALIDATED_EXIT_STATUS,
                "extra": {"exit_status": VALIDATED_EXIT_STATUS, "submission": ""},
            }
        )


def _names_validate(command: str) -> bool:
    """Whether the agent was reaching for the tool but wrote something else."""
    first_word = command.split(maxsplit=1)[0] if command.split() else ""
    return first_word.lstrip("./") == VALIDATE_COMMAND


def _result(output: str, *, returncode: int) -> Dict[str, Any]:
    """An observation shaped the way `LocalEnvironment.execute` shapes one."""
    return {"output": output.strip(), "returncode": returncode, "exception_info": ""}
