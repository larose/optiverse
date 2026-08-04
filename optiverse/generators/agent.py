"""Generation by a coding agent, over mini-swe-agent.

The agent is given a working directory holding a copy of the solution it is
improving, and three tools. It is never given a score — not its own, not its
parent's. Ranking is the search loop's job, and an agent that could see the
score would abandon a novel approach the moment it looked worse than the
incumbent, which is exactly the move the loop relies on to escape local optima.
Answering `validate` ourselves rather than putting a program on the path is part
of that: the evaluator's command never appears anywhere the agent can read it,
so `score` is not one word away from `validate`.

mini-swe-agent is imported inside the methods that use it, so `import optiverse`
stays dependency-free.

mini-swe-agent annotates several signatures with bare `dict`, which strict mode
reports as partially unknown. That looseness is in the dependency, not here, so
it is suppressed for this file only.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import logging
import os
from typing import Any, Dict, Optional, cast

from .. import codebase as codebase_helpers
from .._mini_swe_agent import AgentLimits, normalize_exit_status
from ..generator import GenerationContext, GenerationResult, Generator

logger = logging.getLogger(__name__)

MODEL_VARIABLE = "OPTIVERSE_MODEL"

INSTANCE_TEMPLATE = """{{task}}

# Rules

- Leave no build artifacts, binaries or caches in your working directory. Build
  in a temporary directory if you need to.
- Do not edit anything outside your working directory.
- Directory and environment variable changes are not persistent. Every `bash`
  call runs in a new subshell, starting in your working directory. Prefix a call
  with `MY_ENV_VAR=MY_VALUE cd /path/to/dir && ...` if you need either to stick.
- Call `validate` whenever you want to know whether your work holds up. It
  answers valid or invalid and prints diagnostics. It is also the only way to run
  anything belonging to this problem — there is no way to time or measure your
  own solution.
- Your turn ends the moment `validate` reports valid on something you changed,
  so run it when you are finished, not to check a half-written edit.
- Call `give_up` rather than burning steps on something you cannot get to work.

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

        from .._mini_swe_agent import SYSTEM_TEMPLATE, ToolEnvironment, build_model

        # Taken after the parent has been copied in, so `validate` refusing to
        # end the turn on an unchanged tree means unchanged *relative to the
        # parent*.
        baseline_digest = codebase_helpers.digest(context.codebase)

        environment = ToolEnvironment(
            baseline_digest=baseline_digest,
            codebase=context.codebase,
            cwd=str(context.codebase),
            timeout=self._limits.command_timeout_seconds,
            validate=context.validate,
        )

        agent = DefaultAgent(
            build_model(self._model_name),
            environment,
            cost_limit=self._limits.cost_limit,
            instance_template=INSTANCE_TEMPLATE,
            output_path=context.log_path,
            step_limit=self._limits.step_limit,
            system_template=SYSTEM_TEMPLATE,
            wall_time_limit_seconds=self._limits.wall_time_limit_seconds,
        )

        exit_status = self._run(agent, context)

        return GenerationResult(
            metrics={
                "agent_model_calls": int(agent.n_calls),
                "agent_validate_runs": environment.validate_runs,
            },
            tags={"exit_status": exit_status},
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
