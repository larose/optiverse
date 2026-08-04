"""Planning by a coding agent, over mini-swe-agent.

The director gets a shell and the run directory. That is the whole idea: rather
than being handed a summary somebody else decided was sufficient, it goes and
reads whatever it needs — a candidate's source, a programmer's trajectory, the raw
arcs — and forms its own view.

Unlike the programmer it *is* shown the scores, because ranking is its job. The
programmer is kept ignorant of them so it cannot abandon a novel approach for
looking worse than the incumbent; the director exists to make exactly that call.

It is read-only outside its own directory by convention rather than by
enforcement.

mini-swe-agent annotates several signatures with bare `dict`, which strict mode
reports as partially unknown. That looseness is in the dependency, not here, so
it is suppressed for this file only.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import logging
import os
from typing import Any, Dict, Optional, cast

from ..mini_swe_agent import AgentLimits, normalize_exit_status
from ..director import Director, DirectorContext, DirectorResult

logger = logging.getLogger(__name__)

MODEL_VARIABLE = "OPTIVERSE_DIRECTOR_MODEL"
FALLBACK_MODEL_VARIABLE = "OPTIVERSE_MODEL"

# Matched to the programmer's, and for the same reason: the job is no longer only
# to pick a direction. On a review the director reads a candidate it has not seen,
# reconciles the last prediction, and rewrites its notebook before it decides —
# and a budget sized for choosing from a list would cut it off mid-thought.
DEFAULT_LIMITS = AgentLimits(step_limit=40, wall_time_limit_seconds=900)

# `validate` refuses to end a turn on an unchanged tree, which stops a programmer
# submitting the parent it was handed. The director has no equivalent hazard —
# the plan it validates is the plan it just wrote — so the guard is switched off
# with a digest nothing can produce, `digest` always returning a full hex hash.
NO_BASELINE_DIGEST = ""

INSTANCE_TEMPLATE = """{{task}}

# Rules

- Write only inside your working directory. Everything else in the run directory
  is there for you to read, and reading it is the point.
- Directory and environment variable changes are not persistent. Every `bash`
  call runs in a new subshell, starting in your working directory.
- Call `validate` once you have written `plan.json`. It reports what is wrong
  with it, and ends your turn if there is nothing wrong.
- **`validate` ends your turn, so write `memory.md` before you write
  `plan.json`.** A notebook you meant to update afterwards is one you did not.
- Call `give_up` if you cannot write a plan.

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

# Useful commands

Read what a candidate actually does:

    cat ../../solutions/s_851621dd*/code/*.go

Read what an earlier iteration decided and expected:

    cat ../00042/plan.json

Rewrite your notebook — it is a model, not a log, so replace it rather than
appending to it:

    cat <<'EOF' > memory.md
    ## What this problem rewards
    ...
    EOF

Write the plan:

    cat <<'EOF' > plan.json
    {"verdict": "...", "reasoning": "...", "expectation": "...",
     "parent_node_id": "n_root", "parent_solution_id": "s_...", "constraint": "..."}
    EOF

Read the tree as data:

    python3 -m json.tool ../../arcs.json
"""


class AgentDirector(Director):
    def __init__(
        self,
        *,
        model_name: str,
        limits: Optional[AgentLimits] = None,
    ) -> None:
        self._model_name = model_name
        self._limits = limits or DEFAULT_LIMITS

    @classmethod
    def from_env(cls, *, limits: Optional[AgentLimits] = None) -> "AgentDirector":
        """Build from `OPTIVERSE_DIRECTOR_MODEL`, falling back to the shared one.

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

    def decide(self, context: DirectorContext) -> DirectorResult:
        # Imported here so the core stays importable without mini-swe-agent.
        from minisweagent.agents.default import DefaultAgent

        from ..mini_swe_agent import SYSTEM_TEMPLATE
        from ..mini_swe_agent.environment import ToolEnvironment
        from ..mini_swe_agent.model import build_model

        environment = ToolEnvironment(
            baseline_digest=NO_BASELINE_DIGEST,
            codebase=context.workdir,
            cwd=str(context.workdir),
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

        return DirectorResult(
            metrics={"director_model_calls": int(agent.n_calls)},
            tags={"director_exit_status": exit_status},
        )

    def _run(self, agent: Any, context: DirectorContext) -> str:
        """Run the agent, reporting a failure rather than raising through.

        A director that crashes leaves no plan, which the loop treats as a
        crashed iteration: there is deliberately nothing to fall back to. The
        exit status is what says which kind of failure it was.
        """
        try:
            outcome = cast(Dict[str, Any], agent.run(task=context.prompt))
        except Exception as error:
            logger.warning(f"Director failed: {error}", exc_info=True)
            return normalize_exit_status(f"error:{type(error).__name__}")

        return normalize_exit_status(str(outcome.get("exit_status", "unknown")))
