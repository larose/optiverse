"""mini-swe-agent glue. Imported only when the `agent` extra is installed.

mini-swe-agent annotates several signatures with bare `dict`, which strict mode
reports as partially unknown. The looseness is in the dependency, not here, so it
is suppressed for this file only.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import logging
import re
import shlex
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

logger = logging.getLogger(__name__)

VALIDATED_EXIT_STATUS = "Validated"

UNCHANGED_NOTICE = (
    "\n[optiverse] This solution is valid, but it is byte-for-byte identical to "
    "the one you started from. Make a real change before checking again."
)

ELSEWHERE_NOTICE = (
    "\n[optiverse] That check passed, but your own solution does not validate. "
    "Check yours with exactly:\n{command}"
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
    """Ends the agent's turn as soon as its solution validates.

    Once the code is correct, further work belongs to the outer loop: continuing
    would let the agent hill-climb, which costs tokens and quietly undoes
    diversification.

    Termination requires the tree to have *changed* as well as validated. The
    starting solution is already valid by construction, so without that guard an
    agent could finish by running the check before doing any work.

    Whether a command was the check is decided loosely — the evaluator program
    and the `validate` mode, in any phrasing — but whether the solution is valid
    is decided by running the check ourselves. An agent that rewrites the path,
    adds a redirection or validates some other directory then neither escapes
    termination nor triggers it wrongly.

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
        validate: Callable[[], bool],
        validate_command: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._baseline_digest = baseline_digest
        self._codebase = codebase
        self._validate = validate
        self._validate_command = validate_command
        self._program = _program_name(validate_command)
        self.validate_runs = 0

    def execute(
        self, action: Dict[str, Any], cwd: str = "", *, timeout: int | None = None
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = super().execute(action, cwd, timeout=timeout)

        command = str(action.get("command", ""))
        if not self._is_validate(command):
            return output

        self.validate_runs += 1

        if output.get("returncode") != 0:
            return output

        if codebase_helpers.digest(self._codebase) == self._baseline_digest:
            return _with_notice(output, UNCHANGED_NOTICE)

        if not self._validate():
            return _with_notice(
                output, ELSEWHERE_NOTICE.format(command=self._validate_command)
            )

        raise Submitted(
            {
                "role": "exit",
                "content": VALIDATED_EXIT_STATUS,
                "extra": {"exit_status": VALIDATED_EXIT_STATUS, "submission": ""},
            }
        )

    def _is_validate(self, command: str) -> bool:
        """Whether this command was an attempt to run the check.

        Deliberately not an equality test on the string we handed over: agents
        rewrite paths and add redirections, and a missed match costs the whole
        step budget.
        """
        return self._program in command and "validate" in command


def _program_name(validate_command: str) -> str:
    """The evaluator's file name, which survives any rewriting of its path.

    The command is `<program...> validate <codebase>`, so dropping the last two
    tokens leaves the invocation. Its last non-flag word is the evaluator itself:
    `evaluate.py` rather than the interpreter that happens to run it, which would
    match every other Python command the agent runs.
    """
    tokens = shlex.split(validate_command)
    program_tokens = tokens[:-2] or tokens[:1]

    for token in reversed(program_tokens):
        if not token.startswith("-"):
            return Path(token).name

    return Path(program_tokens[0]).name


def _with_notice(output: Dict[str, Any], notice: str) -> Dict[str, Any]:
    output["output"] = str(output.get("output", "")) + notice
    return output
