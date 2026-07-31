"""mini-swe-agent glue, shared by both agents.

Two things in this project are driven by a coding agent: the generator, which
writes a candidate, and the strategist, which decides what to try next. They
differ in their templates and their working directory, not in their plumbing, so
the model layer, the limits and the validate-ends-your-turn environment live here.

mini-swe-agent annotates several signatures with bare `dict`, which strict mode
reports as partially unknown. The looseness is in the dependency, not here, so it
is suppressed for this file only.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

import litellm
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from . import codebase as codebase_helpers
from .evaluator import EvaluatorError, ValidationResult

logger = logging.getLogger(__name__)

VALIDATED_EXIT_STATUS = "validated"

# The two tools the agent is given. `bash` is mini-swe-agent's own. `validate` is
# ours, and it is a real tool rather than a word smuggled into a bash command
# because a model given nowhere to put it puts it in the shell, where there is no
# such program.
BASH_TOOL_NAME = "bash"
VALIDATE_TOOL_NAME = "validate"

VALIDATE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": VALIDATE_TOOL_NAME,
        "description": (
            "Check your work. Answers valid or invalid and prints diagnostics; "
            "it says nothing about how good the result is. Your turn ends the "
            "moment it reports valid on something you changed."
        ),
        # It takes none, and the schema is the place to say so.
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# mini-swe-agent's own system template describes the ```mswea_bash_command fence
# its text-based model layer parses out of prose. There is no fence here: the
# form of an action is in the tool schema rather than in the prompt.
SYSTEM_TEMPLATE = """You are a helpful assistant that can interact with a computer.

Act by calling a tool. Explain your reasoning before each call.
"""

# Worded here rather than through `format_error_template`, whose default relays
# mini-swe-agent's own bash-only phrasing to a model that has two tools.
FORMAT_ERROR_NOTICE = (
    "Every response must call a tool: `bash` to run a command, or `validate` to "
    "check your work."
)

UNCHANGED_NOTICE = (
    "\n[optiverse] This solution is valid, but it is byte-for-byte identical to "
    "the one you started from. Make a real change before checking again."
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


DEFAULT_STEP_LIMIT = 40
DEFAULT_COST_LIMIT = 0.0
DEFAULT_WALL_TIME_LIMIT_SECONDS = 900
DEFAULT_COMMAND_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class AgentLimits:
    """Bounds on one agent run. All are enforced by mini-swe-agent itself.

    `cost_limit` is off by default: mini-swe-agent reads 0 as "no limit". Spend is
    still recorded per run as a metric, so it is observable without being
    throttled. The bounds that always hold are the step and wall-time limits.
    """

    step_limit: int = DEFAULT_STEP_LIMIT
    cost_limit: float = DEFAULT_COST_LIMIT
    wall_time_limit_seconds: int = DEFAULT_WALL_TIME_LIMIT_SECONDS
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS


def build_model(model_name: str) -> Any:
    """The model layer, which gives the agent its two tools.

    Tool-calling rather than text-based, so `validate` is a tool the model calls
    rather than a word it is asked to type into a shell that has no such program.
    The cost of that is a model that cannot call tools, which this cannot use.

    `cost_tracking="ignore_errors"` because mini-swe-agent otherwise raises when
    litellm cannot price a model — outside its own retry loop, losing the whole
    iteration. A missing price is not a reason to discard a candidate.
    """
    return ToolCallingModel(
        model_name=model_name,
        model_kwargs={"drop_params": True},
        cost_tracking="ignore_errors",
    )


# CamelCase boundaries, including the tail of an acronym: `HTTPError` splits
# before `Error` rather than collapsing to `httperror`.
_CAMEL_BOUNDARIES = (
    re.compile(r"(?<=[a-z0-9])(?=[A-Z])"),
    re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])"),
)


def normalize_exit_status(status: str) -> str:
    """Lower-case, underscore-separated, whatever spelling it arrived in.

    Exit statuses reach us from three places: this module (`validated`),
    mini-swe-agent's own protocol (`LimitsExceeded`,
    `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`) and exception class names
    (`RateLimitError`). Only the first is ours to spell, so normalising at the
    boundary is what actually makes the stored values consistent.

    The `error:` prefix keeps its colon, so a crash stays distinguishable from a
    normal outcome at a glance.
    """
    return ":".join(_snake_case(part) for part in status.split(":"))


def _snake_case(text: str) -> str:
    for pattern in _CAMEL_BOUNDARIES:
        text = pattern.sub("_", text)
    return text.lower()


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


class ToolCallingModel(LitellmModel):
    """Offers both tools, and waits as long as the provider says.

    mini-swe-agent sends exactly one tool and its parser rejects every other
    name, so a second tool means owning the request and the parse. Both are
    overridden here; nothing else about the model layer changes.

    On rate limits, mini-swe-agent retries on a fixed exponential ladder that
    never reads the response, so against a stated 47-second delay it spends its
    first three attempts on requests that cannot succeed, each one charged
    against the very quota it is waiting for. `RateLimitError` is added to the
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
        """The request, with both tools on it.

        A full override rather than a `super()` call: the parent names `tools`
        when it calls `litellm.completion`, so passing ours through `**kwargs`
        would be two values for one argument.
        """
        for waits in range(MAXIMUM_RATE_LIMIT_WAITS + 1):
            try:
                return litellm.completion(
                    model=self.config.model_name,
                    messages=messages,
                    tools=[BASH_TOOL, VALIDATE_TOOL],
                    **(self.config.model_kwargs | kwargs),
                )
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

    def _parse_actions(self, response: Any) -> List[Dict[str, Any]]:
        """Tool calls, keeping the name so the environment can dispatch on it.

        mini-swe-agent's own parser drops the name and rejects anything but
        `bash`, which is exactly the two things a second tool needs from it.

        `command` is set on every action, including `validate`'s, because
        `LocalEnvironment.execute` reads that key whatever the action turns out
        to be.
        """
        tool_calls = cast(List[Any], response.choices[0].message.tool_calls or [])
        actions: List[Dict[str, Any]] = []

        for tool_call in tool_calls:
            name = str(tool_call.function.name)

            if name == VALIDATE_TOOL_NAME:
                command = ""
            elif name == BASH_TOOL_NAME:
                command = _bash_command(tool_call)
            else:
                command = None

            if command is None:
                raise FormatError(_format_error())

            actions.append(
                {"tool": name, "command": command, "tool_call_id": tool_call.id}
            )

        if not actions:
            raise FormatError(_format_error())

        return actions


def _bash_command(tool_call: Any) -> Optional[str]:
    """The command out of a `bash` call, or None if it did not carry one."""
    try:
        arguments = json.loads(tool_call.function.arguments)
    except ValueError:
        return None

    if not isinstance(arguments, dict):
        return None

    command = cast(Dict[str, Any], arguments).get("command")

    return command if isinstance(command, str) else None


def _format_error() -> Dict[str, Any]:
    """A rejection shaped the way mini-swe-agent shapes one.

    `extra` is not optional: the model layer records the call's cost and response
    on it before re-raising, so that a reply the parser threw away is still
    billed and still in the trajectory.
    """
    return {
        "role": "user",
        "content": FORMAT_ERROR_NOTICE,
        "extra": {"interrupt_type": "FormatError"},
    }


class ValidateTerminatesEnvironment(LocalEnvironment):
    """Runs the `validate` tool, and ends the agent's turn once it passes.

    `validate` is answered here rather than by a program on the path, so the
    evaluator's command is never anywhere the agent can read it and `score` is
    not one word away from `validate`. Which action is the check is the tool's
    name, not a guess about what a command line meant.

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
        if action.get("tool") == VALIDATE_TOOL_NAME:
            return self._run_validate()

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


def _result(output: str, *, returncode: int) -> Dict[str, Any]:
    """An observation shaped the way `LocalEnvironment.execute` shapes one."""
    return {"output": output.strip(), "returncode": returncode, "exception_info": ""}
