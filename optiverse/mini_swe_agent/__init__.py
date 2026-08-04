"""mini-swe-agent glue, shared by both agents.

Two things in this project are driven by a coding agent: the programmer, which
writes a candidate, and the director, which decides what to try next. They differ
in their templates and their working directory, not in their plumbing, so the
model layer, the limits and the three tools live here — under neither of them,
because neither is a special case of the other.

This is also the only corner of the project that imports `litellm` or
`minisweagent`. `model` is the request and the parse; `environment` is the shell
and the two tools that are not a shell command; and what is here is what both of
those and both agents agree on.

Nothing re-exports the other two. Importing `model` costs litellm and importing
this does not, which is a distinction worth keeping visible at the call site.

Every one of those tools is a tool. Nothing is a word the agent is asked to type
into a shell that has no such program — not the check, not the give-up. A model
given nowhere to put a word puts it in the shell.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from minisweagent.models.utils.actions_toolcall import BASH_TOOL

VALIDATED_EXIT_STATUS = "validated"
GAVE_UP_EXIT_STATUS = "gave_up"

BASH_TOOL_NAME = "bash"
VALIDATE_TOOL_NAME = "validate"
GIVE_UP_TOOL_NAME = "give_up"

# `bash` is mini-swe-agent's own; the other two are ours. `validate` is both the
# check and the ordinary way a turn ends: an agent that has produced something
# valid and different from what it was handed is finished, and continuing would
# only let it hill-climb on a score it cannot see.
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

GIVE_UP_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": GIVE_UP_TOOL_NAME,
        "description": (
            "Stop without a working result. Use this rather than burning steps "
            "on something you cannot get to work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "What defeated you.",
                }
            },
            "required": ["reason"],
        },
    },
}

TOOLS: List[Dict[str, Any]] = [BASH_TOOL, VALIDATE_TOOL, GIVE_UP_TOOL]

# mini-swe-agent's own system template describes the ```mswea_bash_command fence
# its text-based model layer parses out of prose. There is no fence here: the
# form of an action is in the tool schema rather than in the prompt.
SYSTEM_TEMPLATE = """You are a helpful assistant that can interact with a computer.

Act by calling a tool. Explain your reasoning before each call.
"""

DEFAULT_STEP_LIMIT = 40
DEFAULT_COST_LIMIT = 0.0
DEFAULT_WALL_TIME_LIMIT_SECONDS = 900
DEFAULT_COMMAND_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class AgentLimits:
    """Bounds on one agent run. All are enforced by mini-swe-agent itself.

    `cost_limit` is off by default: mini-swe-agent reads 0 as "no limit". Spend is
    not recorded anywhere — litellm prices only the models it has heard of, so
    the number was zero for exactly the models a run is most likely to use. The
    bounds that always hold are the step and wall-time limits.
    """

    step_limit: int = DEFAULT_STEP_LIMIT
    cost_limit: float = DEFAULT_COST_LIMIT
    wall_time_limit_seconds: int = DEFAULT_WALL_TIME_LIMIT_SECONDS
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS


# CamelCase boundaries, including the tail of an acronym: `HTTPError` splits
# before `Error` rather than collapsing to `httperror`.
_CAMEL_BOUNDARIES = (
    re.compile(r"(?<=[a-z0-9])(?=[A-Z])"),
    re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])"),
)


def normalize_exit_status(status: str) -> str:
    """Lower-case, underscore-separated, whatever spelling it arrived in.

    Exit statuses reach us from three places: this package (`validated`,
    `gave_up`), mini-swe-agent's own protocol (`LimitsExceeded`) and exception
    class names (`RateLimitError`). Only the first is ours to spell, so
    normalising at the boundary is what actually makes the stored values
    consistent.

    The `error:` prefix keeps its colon, so a crash stays distinguishable from a
    normal outcome at a glance.
    """
    return ":".join(_snake_case(part) for part in status.split(":"))


def _snake_case(text: str) -> str:
    for pattern in _CAMEL_BOUNDARIES:
        text = pattern.sub("_", text)
    return text.lower()
