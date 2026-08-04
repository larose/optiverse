"""The model layer: three tools on every request, and rate limits taken as read.

mini-swe-agent sends exactly one tool and parses the reply expecting one name,
so offering three means owning both the request and the parse. That is all this
module is, plus the one behaviour worth having that the dependency does not: when
a provider says how long to wait, waiting that long.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, cast

import litellm
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel

from . import BASH_TOOL_NAME, GIVE_UP_TOOL_NAME, TOOLS, VALIDATE_TOOL_NAME

logger = logging.getLogger(__name__)

# Worded here rather than through `format_error_template`, whose default relays
# mini-swe-agent's own bash-only phrasing to a model that has three tools.
FORMAT_ERROR_NOTICE = (
    "Every response must call a tool: `bash`, `validate` or `give_up`."
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


def build_model(model_name: str) -> Any:
    """The model layer, which gives the agent its three tools.

    Tool-calling rather than text-based, so every action the agent can take is a
    tool the model calls rather than a word it is asked to type into a shell. The
    cost of that is a model that cannot call tools, which this cannot use.

    `cost_tracking="ignore_errors"` because mini-swe-agent otherwise raises when
    litellm cannot price a model — outside its own retry loop, losing the whole
    iteration. A missing price is not a reason to discard a candidate.
    """
    return ToolCallingModel(
        model_name=model_name,
        model_kwargs={"drop_params": True},
        cost_tracking="ignore_errors",
    )


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
    """Offers all three tools, and waits as long as the provider says.

    mini-swe-agent sends exactly one tool and its parser rejects every other
    name, so more than one tool means owning the request and the parse. Both are
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
        """The request, with every tool on it.

        A full override rather than a `super()` call: the parent names `tools`
        when it calls `litellm.completion`, so passing ours through `**kwargs`
        would be two values for one argument.
        """
        for waits in range(MAXIMUM_RATE_LIMIT_WAITS + 1):
            try:
                return litellm.completion(
                    model=self.config.model_name,
                    messages=messages,
                    tools=TOOLS,
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
        `bash`, which is exactly the two things more than one tool needs from it.
        """
        tool_calls = cast(List[Any], response.choices[0].message.tool_calls or [])
        actions = [_action(tool_call) for tool_call in tool_calls]

        if not actions or any(action is None for action in actions):
            raise FormatError(_format_error())

        return cast(List[Dict[str, Any]], actions)


# The one argument each tool takes, for the two that take one. `bash` names its
# `command` because that is the key `LocalEnvironment.execute` reads.
_TOOL_ARGUMENTS = {
    BASH_TOOL_NAME: "command",
    GIVE_UP_TOOL_NAME: "reason",
}


def _action(tool_call: Any) -> Optional[Dict[str, Any]]:
    """One tool call as an action, or None if it did not carry what it needs.

    `command` is set whatever the tool, because `LocalEnvironment.execute` reads
    that key on the way past and only our own dispatch knows the difference.
    """
    name = str(tool_call.function.name)
    action: Dict[str, Any] = {
        "tool": name,
        "command": "",
        "tool_call_id": tool_call.id,
    }

    if name == VALIDATE_TOOL_NAME:
        return action

    argument = _TOOL_ARGUMENTS.get(name)

    if argument is None:
        return None

    value = _argument(tool_call, argument)

    if value is None:
        return None

    action[argument] = value

    return action


def _argument(tool_call: Any, name: str) -> Optional[str]:
    """One named string out of a call's arguments, or None if it was not there."""
    try:
        arguments = json.loads(tool_call.function.arguments)
    except ValueError:
        return None

    if not isinstance(arguments, dict):
        return None

    value = cast(Dict[str, Any], arguments).get(name)

    return value if isinstance(value, str) else None


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
