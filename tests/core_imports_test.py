"""Guards the claim that the core has no third-party dependencies.

`pyproject.toml` declares `dependencies = []`, so `import optiverse` must work
with no extras installed. A single stray top-level import in the wrong module
would break that silently — nothing else in the suite would notice, because the
test environment installs `[dev,agent]`.

Run in a subprocess because this process has already imported mini-swe-agent.
"""

import subprocess
import sys
import unittest

# Everything the `agent` extra drags in, directly or transitively.
FORBIDDEN_PREFIXES = (
    "minisweagent",
    "litellm",
    "openai",
    "pydantic",
    "yaml",
    "jinja2",
    "datasets",
    "textual",
    "typer",
    "rich",
    "numpy",
    "pandas",
)

PROBE = """
import sys

import optiverse
from optiverse.config import OptimizerConfig, Problem
from optiverse.evaluator import EvaluatorCommand
from optiverse.generator import Generator
from optiverse.optimizer import Optimizer
from optiverse.store import FileSystemStore

forbidden = sorted(
    name
    for name in sys.modules
    if name.split(".")[0] in %r
)
print(",".join(forbidden))
"""


class CoreImportTest(unittest.TestCase):
    def test_importing_the_core_pulls_no_third_party_modules(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", PROBE % (FORBIDDEN_PREFIXES,)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "",
            "the core imported a third-party module; keep agent imports lazy",
        )

    def test_generators_package_is_importable_without_the_agent(self) -> None:
        """Importing the subpackage must not eagerly pull mini-swe-agent."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import optiverse.generators, sys; "
                "print('minisweagent' in sys.modules)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
