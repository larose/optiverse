"""Tests for ending the agent's turn when its solution validates.

Requires the `agent` extra, which `make init` installs.

`Submitted.messages` is a bare `dict` upstream, hence the suppression.
"""

# pyright: reportUnknownMemberType=false

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, cast

from minisweagent.exceptions import Submitted

from optiverse import codebase
from optiverse.generators._mini_swe_agent import (
    VALIDATED_EXIT_STATUS,
    ValidateTerminatesEnvironment,
)


def exit_message(error: Submitted) -> Dict[str, Any]:
    return cast(Dict[str, Any], error.messages[0])


class ValidateTerminatesEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)

        self.codebase = self.root / "code"
        self.codebase.mkdir()
        (self.codebase / "solver.py").write_text("original\n")
        self.baseline_digest = codebase.digest(self.codebase)

        # A real script, because the environment runs real shell commands. It
        # fails only when told to, so both verdicts can be exercised.
        self.fail_marker = self.root / "SHOULD_FAIL"
        script = self.root / "evaluate"
        script.write_text(f'#!/bin/sh\n[ -f "{self.fail_marker}" ] && exit 1\nexit 0\n')
        script.chmod(0o755)

        # Separate from the script's exit code, so a command that succeeds while
        # the solution is invalid — the agent checked somewhere else — can be
        # exercised.
        self.invalid_marker = self.root / "INVALID"

        self.validate_command = f"{script} validate {self.codebase}"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def make_validation_fail(self) -> None:
        self.fail_marker.write_text("")
        self.invalid_marker.write_text("")

    def validate(self) -> bool:
        """Stands in for the evaluator the optimizer would pass in, answering
        about *our* codebase however the agent phrased its own command."""
        return not self.invalid_marker.is_file()

    def environment(self) -> "ValidateTerminatesEnvironment":
        return ValidateTerminatesEnvironment(
            baseline_digest=self.baseline_digest,
            codebase=self.codebase,
            validate=self.validate,
            validate_command=self.validate_command,
            cwd=str(self.codebase),
            timeout=30,
        )

    def change_the_tree(self) -> None:
        (self.codebase / "solver.py").write_text("improved\n")

    def run_command(self, command: str) -> Dict[str, Any]:
        return self.environment().execute({"command": command})

    def test_validating_after_a_change_ends_the_turn(self) -> None:
        self.change_the_tree()

        with self.assertRaises(Submitted) as caught:
            self.run_command(self.validate_command)

        message = exit_message(caught.exception)
        self.assertEqual(message["role"], "exit")
        self.assertEqual(message["extra"]["exit_status"], VALIDATED_EXIT_STATUS)

    def test_decorated_invocation_still_ends_the_turn(self) -> None:
        """Agents rarely paste a command bare, so the match has to survive
        surrounding shell."""
        self.change_the_tree()

        with self.assertRaises(Submitted):
            self.run_command(
                f"cd {self.codebase} && {self.validate_command} && echo done"
            )

    def test_rewritten_path_still_ends_the_turn(self) -> None:
        """What actually happened on the first live run: the agent reworded the
        path, the old exact-string match missed it, and the turn ran to the step
        limit hill-climbing."""
        self.change_the_tree()

        with self.assertRaises(Submitted):
            self.run_command(f"{self.root / 'evaluate'} validate .")

    def test_validating_a_different_directory_does_not_end_the_turn(self) -> None:
        """A passing exit code says nothing about which directory was checked."""
        self.change_the_tree()
        self.invalid_marker.write_text("")

        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()

        output = self.run_command(f"{self.root / 'evaluate'} validate {elsewhere}")

        self.assertEqual(output["returncode"], 0)
        self.assertIn("your own solution does not validate", output["output"])

    def test_validating_without_a_change_does_not_end_the_turn(self) -> None:
        """The starting solution already validates, so this would otherwise let
        an agent finish having done nothing."""
        output = self.run_command(self.validate_command)

        self.assertEqual(output["returncode"], 0)

    def test_unchanged_tree_is_told_why_it_did_not_finish(self) -> None:
        output = self.run_command(self.validate_command)

        self.assertIn("identical to", output["output"])

    def test_failing_validation_does_not_end_the_turn(self) -> None:
        self.change_the_tree()
        self.make_validation_fail()

        output = self.run_command(self.validate_command)

        self.assertEqual(output["returncode"], 1)

    def test_other_commands_never_end_the_turn(self) -> None:
        self.change_the_tree()

        output = self.run_command("echo just looking around")

        self.assertEqual(output["returncode"], 0)
        self.assertIn("just looking around", output["output"])

    def test_validate_runs_are_counted(self) -> None:
        environment = self.environment()

        environment.execute({"command": self.validate_command})
        environment.execute({"command": "ls"})
        environment.execute({"command": self.validate_command})

        self.assertEqual(environment.validate_runs, 2)

    def test_giving_up_still_ends_the_turn(self) -> None:
        """The library's own submission path must keep working, so an agent that
        cannot produce a valid solution is not trapped until its limits expire."""
        with self.assertRaises(Submitted) as caught:
            self.run_command("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")

        self.assertEqual(
            exit_message(caught.exception)["extra"]["exit_status"], "Submitted"
        )

    def test_commands_run_inside_the_codebase(self) -> None:
        output = self.run_command("pwd")

        self.assertIn(str(self.codebase), output["output"])


if __name__ == "__main__":
    unittest.main()
