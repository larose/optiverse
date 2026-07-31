"""Generation by a coding agent, over mini-swe-agent.

The agent is given a codebase directory and two tools, `bash` and `validate`. It
is never given the score: ranking candidates is the search loop's job, and an
agent that could see the score would abandon a novel approach the moment it
looked worse than the incumbent — which is exactly the move the loop relies on to
escape local optima. Answering `validate` ourselves rather than putting a program
on the path is part of that: the evaluator's command never appears anywhere the
agent can read it, so `score` is not one word away from `validate`.

mini-swe-agent is imported inside the methods that use it, so `import optiverse`
stays dependency-free.

mini-swe-agent annotates several signatures with bare `dict`, which strict mode
reports as partially unknown. That looseness is in the dependency, not here, so
it is suppressed for this file only.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import logging
import os
from typing import Any, Dict, Optional, Set, cast

from .. import codebase as codebase_helpers
from .._mini_swe_agent import AgentLimits, normalize_exit_status
from ..generator import GenerationContext, GenerationResult, Generator

logger = logging.getLogger(__name__)

# Models already reported as unpriced, so the notice is given once per process.
_UNPRICED_MODELS: Set[str] = set()

MODEL_VARIABLE = "OPTIVERSE_MODEL"

INSTANCE_TEMPLATE = """{{task}}

# Checking your work

Call the `validate` tool to check your solution. **When it reports valid after
you have changed something, your task ends immediately** — you do not need to
submit anything.

It reports validity only. It says nothing about how good the solution is; that is
judged after you finish. It is also the only way to run anything belonging to
this problem — there is no way to time or measure your own solution.

# Rules

- Leave no build artifacts, binaries or caches in your working directory. Build
  in a temporary directory if you need to.
- Do not edit anything outside your working directory and the parent copies.
- Directory and environment variable changes are not persistent. Every `bash`
  call runs in a new subshell, starting in your working directory. Prefix a call
  with `MY_ENV_VAR=MY_VALUE cd /path/to/dir && ...` if you need either to stick.
- If you get stuck and cannot produce a valid solution, run
  `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` on its own to give up.

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

# Useful commands

Create a file:

    cat <<'EOF' > newfile.py
    hello = "world"
    EOF

View part of one:

    nl -ba filename.py | sed -n '10,20p'
"""


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

    def generate(self, context: GenerationContext) -> GenerationResult:
        # Imported here so the core stays importable without mini-swe-agent.
        from minisweagent.agents.default import DefaultAgent

        from .._mini_swe_agent import (
            SYSTEM_TEMPLATE,
            ValidateTerminatesEnvironment,
            build_model,
        )

        baseline_digest = codebase_helpers.digest(context.codebase)

        environment = ValidateTerminatesEnvironment(
            baseline_digest=baseline_digest,
            codebase=context.codebase,
            validate=context.validate,
            cwd=str(context.codebase),
            timeout=self._limits.command_timeout_seconds,
        )

        agent = DefaultAgent(
            build_model(self._model_name),
            environment,
            system_template=SYSTEM_TEMPLATE,
            instance_template=INSTANCE_TEMPLATE,
            step_limit=self._limits.step_limit,
            cost_limit=self._limits.cost_limit,
            wall_time_limit_seconds=self._limits.wall_time_limit_seconds,
            output_path=context.log_path,
        )

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
            return normalize_exit_status(f"error:{type(error).__name__}")

        return normalize_exit_status(str(outcome.get("exit_status", "unknown")))
