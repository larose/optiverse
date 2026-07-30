"""Generation by a coding agent, over mini-swe-agent.

The agent is given a codebase directory and a way to check its own work. It is
never given the score: ranking candidates is the search loop's job, and an agent
that could see the score would abandon a novel approach the moment it looked
worse than the incumbent — which is exactly the move the loop relies on to
escape local optima.

Requires the `agent` extra: `pip install optiverse[agent]`. mini-swe-agent is
imported inside the methods that use it, so `import optiverse` stays
dependency-free.

mini-swe-agent annotates several signatures with bare `dict`, which strict mode
reports as partially unknown. That looseness is in the dependency, not here, so
it is suppressed for this file only.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, cast

from .. import codebase as codebase_helpers
from ..generator import (
    GenerationContext,
    GenerationResult,
    Generator,
    ReferenceCodebase,
)

logger = logging.getLogger(__name__)

# Models already reported as unpriced, so the notice is given once per process.
_UNPRICED_MODELS: Set[str] = set()

MODEL_VARIABLE = "OPTIVERSE_MODEL"

DEFAULT_STEP_LIMIT = 40
DEFAULT_COST_LIMIT = 0.0
DEFAULT_WALL_TIME_LIMIT_SECONDS = 900
DEFAULT_COMMAND_TIMEOUT_SECONDS = 180

VALIDATED_EXIT_STATUS = "Validated"

INSTANCE_TEMPLATE = """{{task}}

# Your working directory

You are working in `{{optiverse_codebase}}`, which already contains the starting
solution. Edit it in place. You may add, modify and delete files there.

{{optiverse_references}}

# Checking your work

Run this to check that your solution is valid:

```
{{optiverse_validate_command}}
```

It exits 0 when the solution builds and behaves correctly, and prints
diagnostics otherwise. **When it exits 0 after you have made a change, your task
ends immediately** — you do not need to submit anything.

It reports validity only. It says nothing about how good the solution is; that is
judged after you finish. Do not try to measure or optimise runtime.

# Before you finish

Write two or three plain sentences describing the approach you took to
`{{optiverse_description_path}}`. A later agent reads this to understand what was
already tried, so describe the idea, not the edits.

# Rules

- Leave no build artifacts, binaries or caches in the working directory. Build in
  a temporary directory if you need to.
- Do not edit anything outside your working directory. Other solutions are
  read-only.

## Response format

You can execute bash commands. Every response must contain exactly one action.

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is
   executed in a new subshell. However, you can prefix any action with
   `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...`
4. If you get stuck and cannot produce a valid solution, issue
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` on its own to give up.

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

## Formatting your response

<example_response>
THOUGHT: I need to understand the current solution first.

```mswea_bash_command
ls -la
```
</example_response>

## Useful command examples

### Create a new file:

```mswea_bash_command
cat <<'EOF' > newfile.py
hello = "world"
EOF
```

### View file content:

```mswea_bash_command
nl -ba filename.py | sed -n '10,20p'
```
"""


@dataclass(frozen=True)
class AgentLimits:
    """Bounds on one generation. All are enforced by mini-swe-agent itself.

    `cost_limit` is off by default: mini-swe-agent reads 0 as "no limit". Spend is
    still recorded per candidate as `m_agent_cost_usd`, so it is observable
    without being throttled. The bounds that always hold are the step and
    wall-time limits.
    """

    step_limit: int = DEFAULT_STEP_LIMIT
    cost_limit: float = DEFAULT_COST_LIMIT
    wall_time_limit_seconds: int = DEFAULT_WALL_TIME_LIMIT_SECONDS
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS


class AgentGenerator(Generator):
    def __init__(
        self,
        *,
        model_name: str,
        limits: Optional[AgentLimits] = None,
    ) -> None:
        self._model_name = model_name
        self._limits = limits or AgentLimits()

    @classmethod
    def from_env(cls, *, limits: Optional[AgentLimits] = None) -> "AgentGenerator":
        """Build from `OPTIVERSE_MODEL`, a litellm model name.

        Credentials are the provider's own environment variables, set the way that
        provider's documentation says — `GEMINI_API_KEY` for `gemini/...`,
        `ANTHROPIC_API_KEY` for `anthropic/...`, `OLLAMA_API_BASE` for a local
        server. litellm reads them itself, so there is nothing to pass through.
        """
        model_name = os.getenv(MODEL_VARIABLE)
        if not model_name:
            raise ValueError(f"{MODEL_VARIABLE} environment variable is required")

        return cls(model_name=model_name, limits=limits)

    def build_model(self) -> Any:
        """The model layer, matched to the templates the agent is given.

        Text-based rather than tool-calling: both templates `generate` renders
        describe the ```mswea_bash_command fence, which is what this class's
        `action_regex` parses. The tool-calling class would reject those replies
        as format errors, and would also rule out every model without tool
        support.

        `cost_tracking="ignore_errors"` because mini-swe-agent otherwise raises
        when litellm cannot price a model — outside its own retry loop, losing the
        whole iteration. A missing price is not a reason to discard a candidate.
        """
        # Imported here so the core stays importable without the agent extra.
        from ._mini_swe_agent import RateLimitAwareModel

        return RateLimitAwareModel(
            model_name=self._model_name,
            model_kwargs={"drop_params": True},
            cost_tracking="ignore_errors",
        )

    def generate(self, context: GenerationContext) -> GenerationResult:
        from minisweagent.agents.default import DefaultAgent

        from ._mini_swe_agent import (
            ValidateTerminatesEnvironment,
            default_agent_config,
        )

        baseline_digest = codebase_helpers.digest(context.codebase)

        environment = ValidateTerminatesEnvironment(
            baseline_digest=baseline_digest,
            codebase=context.codebase,
            validate=context.validate,
            validate_command=context.validate_shell_command,
            cwd=str(context.codebase),
            timeout=self._limits.command_timeout_seconds,
        )

        agent_config = default_agent_config()

        agent = DefaultAgent(
            self.build_model(),
            environment,
            system_template=cast(str, agent_config["system_template"]),
            instance_template=INSTANCE_TEMPLATE,
            step_limit=self._limits.step_limit,
            cost_limit=self._limits.cost_limit,
            wall_time_limit_seconds=self._limits.wall_time_limit_seconds,
            output_path=context.log_path,
        )

        agent.extra_template_vars |= {
            "optiverse_codebase": str(context.codebase),
            "optiverse_description_path": str(context.description_path),
            "optiverse_references": _render_references(context.references),
            "optiverse_validate_command": context.validate_shell_command,
        }

        exit_status = self._run(agent, context)
        self._report_unpriced(agent)

        return GenerationResult(
            metrics={
                "agent_cost_usd": float(agent.cost),
                "agent_model_calls": int(agent.n_calls),
                "agent_validate_runs": environment.validate_runs,
            },
            tags={"exit_status": exit_status},
        )

    def _report_unpriced(self, agent: Any) -> None:
        """Say so when a model turns out to be unpriced, rather than looking free.

        Cost tracking is set to ignore errors, so an unknown model reports zero
        instead of failing. Said once per process: repeating it every iteration
        would bury the run's own output.
        """
        if agent.n_calls <= 0 or agent.cost > 0.0:
            return

        if self._model_name in _UNPRICED_MODELS:
            return

        _UNPRICED_MODELS.add(self._model_name)
        logger.info(
            f"litellm has no pricing for {self._model_name}, so agent_cost_usd "
            "will read 0. Step and wall-time limits still bound each iteration."
        )

    def _run(self, agent: Any, context: GenerationContext) -> str:
        """Run the agent, treating any failure as a normal outcome.

        Whatever is on disk is scored regardless, so a crashed or exhausted agent
        is a weak candidate rather than a lost iteration.
        """
        try:
            outcome = cast(Dict[str, Any], agent.run(task=context.prompt))
        except Exception as error:
            logger.warning(
                f"Agent failed on {context.codebase}: {error}", exc_info=True
            )
            return f"Error:{type(error).__name__}"

        return str(outcome.get("exit_status", "Unknown"))


def _render_references(references: List[ReferenceCodebase]) -> str:
    """Point at the other parents rather than pasting them into the prompt."""
    if not references:
        return ""

    lines = [
        "# Other solutions you may read",
        "",
        "These are read-only. Read them if useful; you do not have to.",
        "",
    ]

    for reference in references:
        lines.append(
            f"- {reference.title} (score {reference.score}): `{reference.path}`"
        )

    return "\n".join(lines) + "\n"
