"""Planning by a coding agent, over mini-swe-agent.

The director gets a shell and the run directory. That is the whole idea: rather
than being handed a summary somebody else decided was sufficient, it goes and
reads whatever it needs — a candidate's source, an agent's trajectory, the raw
arcs — and forms its own view.

Unlike the generator it *is* shown the scores, because ranking is its job. The
generator is kept ignorant of them so it cannot abandon a novel approach for
looking worse than the incumbent; the director exists to make exactly that call.

It is read-only outside its own directory and `memory.md` by convention rather
than by enforcement.

mini-swe-agent annotates several signatures with bare `dict`, which strict mode
reports as partially unknown. That looseness is in the dependency, not here, so
it is suppressed for this file only.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import logging
import os
from typing import Any, Dict, Optional, cast

from .._mini_swe_agent import AgentLimits, normalize_exit_status
from ..director import Director, DirectorContext, DirectorResult

logger = logging.getLogger(__name__)

MODEL_VARIABLE = "OPTIVERSE_DIRECTOR_MODEL"
FALLBACK_MODEL_VARIABLE = "OPTIVERSE_MODEL"

# Deciding what to try next is cheaper than building it, so the director is held
# to a tighter budget than the generator's 40 steps and 900 seconds. If it needs
# forty steps to pick a direction, the tree it was given is the problem.
DEFAULT_LIMITS = AgentLimits(step_limit=25, wall_time_limit_seconds=600)

# `done` refuses to end a turn on an unchanged tree, which stops a generator
# submitting the parent it was handed. The director has no equivalent hazard —
# the plan it validates is the plan it just wrote — so the guard is switched off
# with a digest nothing can produce, `digest` always returning a full hex hash.
NO_BASELINE_DIGEST = ""

INSTANCE_TEMPLATE = """{{task}}

# Rules

- Write only inside your working directory, plus `../../memory.md`. Everything
  else in the run directory is there for you to read, and reading it is the
  point.
- Directory and environment variable changes are not persistent. Every `bash`
  call runs in a new subshell, starting in your working directory.
- Call `validate` to check the plan you have written. It reports what is wrong
  with `plan.json`, or nothing if it is fine. It does not end your turn.
- Call `done` once the plan is valid, or `give_up` if you cannot write one.

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

# Useful commands

Read what a candidate actually does:

    cat ../../solutions/s_851621dd*/code/*.go

Write the plan:

    cat <<'EOF' > plan.json
    {"parent_node_id": "n_root", "parent_solution_id": "s_...", "constraint": "..."}
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
        """Build from `OPTIVERSE_DIRECTOR_MODEL`, falling back to the generator's.

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

        from .._mini_swe_agent import SYSTEM_TEMPLATE, ToolEnvironment, build_model

        environment = ToolEnvironment(
            baseline_digest=NO_BASELINE_DIGEST,
            codebase=context.workdir,
            cwd=str(context.workdir),
            remember=context.remember,
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
        """Run the agent, treating any failure as a normal outcome.

        A director that crashes leaves no plan, and the search falls back to
        another attempt at the best solution — a weak iteration rather than a
        lost one.
        """
        try:
            outcome = cast(Dict[str, Any], agent.run(task=context.prompt))
        except Exception as error:
            logger.warning(f"Director failed: {error}", exc_info=True)
            return normalize_exit_status(f"error:{type(error).__name__}")

        return normalize_exit_status(str(outcome.get("exit_status", "unknown")))
