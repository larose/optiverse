"""Planning by a coding agent, over mini-swe-agent.

The strategist gets a shell and the run directory. That is the whole idea: rather
than being handed a summary somebody else decided was sufficient, it goes and
reads whatever it needs — a candidate's source, an agent's trajectory, the raw
journal — and forms its own view.

Unlike the generator it *is* shown the scores, because ranking is its job. The
generator is kept ignorant of them so it cannot abandon a novel approach for
looking worse than the incumbent; the strategist exists to make exactly that
call.

It is read-only outside its own directory by convention rather than by
enforcement, which is the same footing the generator is on with the parent copies
it is told not to reach past.

mini-swe-agent annotates several signatures with bare `dict`, which strict mode
reports as partially unknown. That looseness is in the dependency, not here, so
it is suppressed for this file only.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import logging
import os
from typing import Any, Dict, Optional, cast

from .._mini_swe_agent import AgentLimits, normalize_exit_status
from ..strategist import Strategist, StrategistContext, StrategistResult

logger = logging.getLogger(__name__)

MODEL_VARIABLE = "OPTIVERSE_STRATEGIST_MODEL"
FALLBACK_MODEL_VARIABLE = "OPTIVERSE_MODEL"

# Deciding what to try next is cheaper than building it, so the strategist is
# held to a tighter budget than the generator's 40 steps and 900 seconds. If it
# needs forty steps to pick a direction, the digest it was given is the problem.
DEFAULT_LIMITS = AgentLimits(step_limit=25, wall_time_limit_seconds=600)

# `ValidateTerminatesEnvironment` refuses to end a turn on an unchanged tree,
# which stops a generator submitting a parent it merely copied in. The strategist
# has no equivalent hazard — the plan it validates is the plan it just wrote — so
# the guard is switched off with a digest nothing can produce, `digest` always
# returning a full hex hash.
NO_BASELINE_DIGEST = ""

INSTANCE_TEMPLATE = """{{task}}

# Checking your work

Run `validate` to check the plan you have written:

```mswea_bash_command
validate
```

It is a tool, not a program: write it on its own, with no arguments and no path.
It reports what is wrong with `plan.json` and `knowledge.md`, or nothing if they
are fine. **When it reports valid after you have changed something, your task
ends immediately** — you do not need to submit anything.

# Rules

- Write only inside your working directory. Everything else in the run directory
  is there for you to read, and reading it is the point.
- Copy constraint text; do not retype it. A branch is identified by its exact
  constraint strings, so a reworded one silently starts a new branch. To continue
  an existing branch, lift its constraints out of `journal.jsonl` with a script.

# Response format

You can execute bash commands. Every response must contain exactly one action.

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is
   executed in a new subshell, starting in your working directory. However, you
   can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/dir && ...`
4. If you get stuck and cannot produce a valid plan, issue
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` on its own to give up.

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

## Formatting your response

<example_response>
THOUGHT: I should see what the best candidate actually does before deciding.

```mswea_bash_command
cat ../solutions/851621dd*/code/*.go
```
</example_response>

## Useful command examples

### Write the plan:

```mswea_bash_command
cat <<'EOF' > plan.json
{"constraints": [], "parent_solution_ids": [], "task": "..."}
EOF
```

### Read the last few iterations:

```mswea_bash_command
tail -n 3 journal.jsonl | python3 -m json.tool --json-lines
```
"""


class AgentStrategist(Strategist):
    def __init__(
        self,
        *,
        model_name: str,
        limits: Optional[AgentLimits] = None,
    ) -> None:
        self._model_name = model_name
        self._limits = limits or DEFAULT_LIMITS

    @classmethod
    def from_env(cls, *, limits: Optional[AgentLimits] = None) -> "AgentStrategist":
        """Build from `OPTIVERSE_STRATEGIST_MODEL`, falling back to the generator's.

        Separate because the two jobs do not want the same model: planning reads a
        lot and writes a little, and a run may well want to spend differently on
        it than on writing code. Falling back means it stays one variable until
        you care.
        """
        model_name = os.getenv(MODEL_VARIABLE) or os.getenv(FALLBACK_MODEL_VARIABLE)

        if not model_name:
            raise ValueError(
                f"{MODEL_VARIABLE} or {FALLBACK_MODEL_VARIABLE} "
                "environment variable is required"
            )

        return cls(model_name=model_name, limits=limits)

    def decide(self, context: StrategistContext) -> StrategistResult:
        # Imported here so the core stays importable without mini-swe-agent.
        from minisweagent.agents.default import DefaultAgent

        from .._mini_swe_agent import (
            ValidateTerminatesEnvironment,
            build_model,
            default_agent_config,
        )

        environment = ValidateTerminatesEnvironment(
            baseline_digest=NO_BASELINE_DIGEST,
            codebase=context.workdir,
            cwd=str(context.workdir),
            timeout=self._limits.command_timeout_seconds,
            validate=context.validate,
        )

        agent_config = default_agent_config()

        agent = DefaultAgent(
            build_model(self._model_name),
            environment,
            cost_limit=self._limits.cost_limit,
            instance_template=INSTANCE_TEMPLATE,
            output_path=context.log_path,
            step_limit=self._limits.step_limit,
            system_template=cast(str, agent_config["system_template"]),
            wall_time_limit_seconds=self._limits.wall_time_limit_seconds,
        )

        exit_status = self._run(agent, context)

        return StrategistResult(
            metrics={
                "strategist_cost_usd": float(agent.cost),
                "strategist_model_calls": int(agent.n_calls),
            },
            tags={"strategist_exit_status": exit_status},
        )

    def _run(self, agent: Any, context: StrategistContext) -> str:
        """Run the agent, treating any failure as a normal outcome.

        A strategist that crashes leaves no plan, and the search falls back to
        improving the best solution — a weak iteration rather than a lost one.
        """
        try:
            outcome = cast(Dict[str, Any], agent.run(task=context.prompt))
        except Exception as error:
            logger.warning(f"Strategist failed: {error}", exc_info=True)
            return normalize_exit_status(f"error:{type(error).__name__}")

        return normalize_exit_status(str(outcome.get("exit_status", "unknown")))
