"""mini-swe-agent glue. Imported only when the `agent` extra is installed.

mini-swe-agent annotates several signatures with bare `dict`, which strict mode
reports as partially unknown. The looseness is in the dependency, not here, so it
is suppressed for this file only.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from pathlib import Path
from typing import Any, Dict, cast

import yaml
from minisweagent import package_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted

from .. import codebase as codebase_helpers

VALIDATED_EXIT_STATUS = "Validated"

UNCHANGED_NOTICE = (
    "\n[optiverse] This solution is valid, but it is byte-for-byte identical to "
    "the one you started from. Make a real change before checking again."
)


def default_agent_config() -> Dict[str, Any]:
    """The `agent` section of mini-swe-agent's shipped default config.

    Reused for `system_template`, which encodes the action format that the model
    layer parses. Rewriting it would risk silent format errors.
    """
    raw = yaml.safe_load((package_dir / "config" / "default.yaml").read_text())
    return cast(Dict[str, Any], cast(Dict[str, Any], raw)["agent"])


class ValidateTerminatesEnvironment(LocalEnvironment):
    """Ends the agent's turn as soon as its solution validates.

    Once the code is correct, further work belongs to the outer loop: continuing
    would let the agent hill-climb, which costs tokens and quietly undoes
    diversification.

    Termination requires the tree to have *changed* as well as validated. The
    starting solution is already valid by construction, so without that guard an
    agent could finish by running the check before doing any work.

    This sits on top of mini-swe-agent's own submission protocol rather than
    replacing it: an agent that gives up can still exit via
    `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, and the resulting `exit_status`
    distinguishes the two paths. Recognising the check is a substring match on the
    command we handed the agent, so an agent that hand-rolls an equivalent
    invocation simply is not auto-terminated — it falls back to submitting or to
    the limits.
    """

    def __init__(
        self,
        *,
        baseline_digest: str,
        codebase: Path,
        validate_command: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._baseline_digest = baseline_digest
        self._codebase = codebase
        self._validate_command = validate_command
        self.validate_runs = 0

    def execute(
        self, action: Dict[str, Any], cwd: str = "", *, timeout: int | None = None
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = super().execute(action, cwd, timeout=timeout)

        command = str(action.get("command", ""))
        if self._validate_command not in command:
            return output

        self.validate_runs += 1

        if output.get("returncode") != 0:
            return output

        if codebase_helpers.digest(self._codebase) == self._baseline_digest:
            output["output"] = str(output.get("output", "")) + UNCHANGED_NOTICE
            return output

        raise Submitted(
            {
                "role": "exit",
                "content": VALIDATED_EXIT_STATUS,
                "extra": {"exit_status": VALIDATED_EXIT_STATUS, "submission": ""},
            }
        )
