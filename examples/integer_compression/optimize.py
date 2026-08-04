import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import optiverse
from optiverse.programmer.agent import AgentProgrammer
from optiverse.director.agent import AgentDirector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

EXAMPLE_DIRECTORY = Path(__file__).parent


def main() -> None:
    raw_directory = os.getenv("DIRECTORY")

    if raw_directory is None:
        directory = Path("tmp") / datetime.now().strftime("%Y%m%d_%H%M%S")
        directory.mkdir(exist_ok=True, parents=True)
    else:
        directory = Path(raw_directory)

    problem = optiverse.config.Problem(
        description=(EXAMPLE_DIRECTORY / "problem.md").read_text(),
        initial_codebase=EXAMPLE_DIRECTORY / "initial",
        # sys.executable rather than a shebang: the evaluator imports optiverse,
        # so it has to run under the interpreter optiverse is installed in.
        evaluate_command=[
            sys.executable,
            str(EXAMPLE_DIRECTORY / "harness" / "evaluate.py"),
        ],
        # Scoring compiles the candidate, then streams a 577 MB dataset through
        # Compress/Decompress once per run, in a process per run. Validation
        # compiles it too, which is most of what its budget is for.
        #
        # Both are larger than the harness's own worst case, so a slow candidate
        # is killed by the timeout that knows what it was doing rather than by
        # this one, which would report it as a broken evaluator.
        score_timeout_seconds=600.0,
        validate_timeout_seconds=180.0,
    )

    config = optiverse.config.OptimizerConfig(
        directory=directory,
        programmer=AgentProgrammer.from_env(),
        max_iterations=1000,
        problem=problem,
        director=AgentDirector.from_env(),
    )

    optiverse.optimizer.Optimizer(config=config).run()


if __name__ == "__main__":
    main()
