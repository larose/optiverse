"""Tests for the evaluator process contract.

Each fake evaluator is a real script run in a real subprocess: the contract is a
process contract, so testing it any other way would test something else.
"""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

from optiverse.evaluator import EvaluatorCommand, EvaluatorError


class EvaluatorContractTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.codebase = self.root / "code"
        self.codebase.mkdir()
        (self.codebase / "solver.py").write_text("pass\n")

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def evaluator(self, body: str, **kwargs: float) -> EvaluatorCommand:
        """Build an evaluator from a Python snippet run as a script."""
        script = self.root / "evaluate.py"
        script.write_text("import sys, json\n" + body)
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return EvaluatorCommand([sys.executable, str(script)], **kwargs)


class ValidateTest(EvaluatorContractTestCase):
    def test_exit_zero_is_valid(self) -> None:
        evaluator = self.evaluator("sys.exit(0)")
        self.assertTrue(evaluator.validate(self.codebase))

    def test_non_zero_exit_is_invalid(self) -> None:
        evaluator = self.evaluator("sys.exit(1)")
        self.assertFalse(evaluator.validate(self.codebase))

    def test_verdict_ignores_stdout(self) -> None:
        """The exit code is the whole answer, so noisy output cannot flip it."""
        evaluator = self.evaluator("print('score: 1.23'); sys.exit(1)")
        self.assertFalse(evaluator.validate(self.codebase))

    def test_verdict_ignores_stderr(self) -> None:
        evaluator = self.evaluator(
            "print('compile error', file=sys.stderr); sys.exit(0)"
        )
        self.assertTrue(evaluator.validate(self.codebase))

    def test_receives_mode_and_codebase_as_arguments(self) -> None:
        evaluator = self.evaluator(
            "sys.exit(0 if sys.argv[1] == 'validate' and sys.argv[2].endswith('code') else 3)"
        )
        self.assertTrue(evaluator.validate(self.codebase))

    def test_timeout_raises_rather_than_reporting_invalid(self) -> None:
        evaluator = self.evaluator(
            "import time; time.sleep(30)", validate_timeout_seconds=0.5
        )

        with self.assertRaises(EvaluatorError) as caught:
            evaluator.validate(self.codebase)

        self.assertIn("timed out", str(caught.exception))

    def test_missing_program_raises(self) -> None:
        evaluator = EvaluatorCommand([str(self.root / "does-not-exist")])

        with self.assertRaises(EvaluatorError):
            evaluator.validate(self.codebase)


class ScoreTest(EvaluatorContractTestCase):
    def test_parses_score_and_metrics(self) -> None:
        evaluator = self.evaluator(
            "json.dump({'score': 12.5, 'metrics': {'lines': 3}}, sys.stdout)"
        )

        result = evaluator.score(self.codebase)

        self.assertEqual(result.score, 12.5)
        self.assertEqual(result.metrics, {"lines": 3})

    def test_null_score_means_unscoreable_not_broken(self) -> None:
        evaluator = self.evaluator(
            "json.dump({'score': None, 'metrics': {}}, sys.stdout)"
        )

        result = evaluator.score(self.codebase)

        self.assertIsNone(result.score)

    def test_metrics_are_optional(self) -> None:
        evaluator = self.evaluator("json.dump({'score': 1.0}, sys.stdout)")

        self.assertEqual(evaluator.score(self.codebase).metrics, {})

    def test_captures_stderr_as_the_log(self) -> None:
        evaluator = self.evaluator(
            "print('=== run 1 ===', file=sys.stderr);"
            "json.dump({'score': 1.0}, sys.stdout)"
        )

        self.assertIn("=== run 1 ===", evaluator.score(self.codebase).log)

    def test_noisy_stderr_does_not_break_json_parsing(self) -> None:
        """stdout is the machine channel; stderr may contain anything."""
        evaluator = self.evaluator(
            "print('{not json at all', file=sys.stderr);"
            "json.dump({'score': 2.0}, sys.stdout)"
        )

        self.assertEqual(evaluator.score(self.codebase).score, 2.0)

    def test_non_zero_exit_raises(self) -> None:
        evaluator = self.evaluator(
            "print('harness crashed', file=sys.stderr); sys.exit(1)"
        )

        with self.assertRaises(EvaluatorError) as caught:
            evaluator.score(self.codebase)

        self.assertIn("harness crashed", str(caught.exception))

    def test_malformed_json_raises(self) -> None:
        evaluator = self.evaluator("print('not json')")

        with self.assertRaises(EvaluatorError):
            evaluator.score(self.codebase)

    def test_missing_score_key_raises(self) -> None:
        evaluator = self.evaluator("json.dump({'metrics': {}}, sys.stdout)")

        with self.assertRaises(EvaluatorError):
            evaluator.score(self.codebase)

    def test_non_numeric_score_raises(self) -> None:
        evaluator = self.evaluator("json.dump({'score': 'fast'}, sys.stdout)")

        with self.assertRaises(EvaluatorError):
            evaluator.score(self.codebase)

    def test_non_numeric_metric_raises(self) -> None:
        evaluator = self.evaluator(
            "json.dump({'score': 1.0, 'metrics': {'a': 'b'}}, sys.stdout)"
        )

        with self.assertRaises(EvaluatorError):
            evaluator.score(self.codebase)

    def test_timeout_raises(self) -> None:
        evaluator = self.evaluator(
            "import time; time.sleep(30)", score_timeout_seconds=0.5
        )

        with self.assertRaises(EvaluatorError) as caught:
            evaluator.score(self.codebase)

        self.assertIn("timed out", str(caught.exception))


class ShellCommandTest(EvaluatorContractTestCase):
    def test_quotes_paths_containing_spaces(self) -> None:
        evaluator = EvaluatorCommand(["/usr/bin/evaluate"])
        spaced = Path("/tmp/a directory/code")

        rendered = evaluator.shell_command("validate", spaced)

        self.assertIn("validate", rendered)
        self.assertIn('"/tmp/a directory/code"', rendered)

    def test_argv_places_mode_before_codebase(self) -> None:
        evaluator = EvaluatorCommand(["run", "--flag"])

        argv: List[str] = evaluator.argv("score", Path("/tmp/code"))

        self.assertEqual(argv, ["run", "--flag", "score", "/tmp/code"])


class AbsolutePathTest(EvaluatorContractTestCase):
    """The agent runs in its own codebase, so a path relative to ours is a path
    to nothing. This is what made the first live run fail."""

    def test_codebase_is_named_absolutely(self) -> None:
        evaluator = EvaluatorCommand(["/usr/bin/evaluate"])

        argv = evaluator.argv("validate", Path("tmp/run/id/code"))

        self.assertEqual(argv[-1], os.path.abspath("tmp/run/id/code"))

    def relative_script(self) -> str:
        """A real script named relative to wherever the tests are being run."""
        script = self.root / "evaluate"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        return os.path.relpath(script, Path.cwd())

    def test_a_program_given_as_a_path_is_resolved(self) -> None:
        evaluator = EvaluatorCommand([self.relative_script()])

        self.assertEqual(
            evaluator.argv("score", self.codebase)[0], str(self.root / "evaluate")
        )

    def test_the_script_after_an_interpreter_is_resolved(self) -> None:
        """`[python, evaluate.py]` is the shape both examples use, so the path
        that has to survive is the second argument, not the first."""
        evaluator = EvaluatorCommand([sys.executable, self.relative_script()])

        self.assertEqual(
            evaluator.argv("score", self.codebase)[1], str(self.root / "evaluate")
        )

    def test_a_bare_program_stays_a_path_lookup(self) -> None:
        """`evaluate` means "whatever PATH finds", not a file in the cwd."""
        evaluator = EvaluatorCommand(["evaluate"])

        self.assertEqual(evaluator.argv("score", self.codebase)[0], "evaluate")

    def test_arguments_that_are_not_paths_are_left_alone(self) -> None:
        evaluator = EvaluatorCommand(["evaluate", "--mode=a/b", "no/such/file"])

        self.assertEqual(
            evaluator.argv("score", self.codebase)[:3],
            ["evaluate", "--mode=a/b", "no/such/file"],
        )

    def test_an_empty_command_is_rejected(self) -> None:
        with self.assertRaises(EvaluatorError):
            EvaluatorCommand([])


if __name__ == "__main__":
    unittest.main()
