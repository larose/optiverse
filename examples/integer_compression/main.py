import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import optiverse
from optiverse.generators.agent import AgentGenerator

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
        evaluate_command=[sys.executable, str(EXAMPLE_DIRECTORY / "evaluate.py")],
        # Scoring streams a 1.6 GB dataset through Compress/Decompress 3 times.
        score_timeout_seconds=2400.0,
        validate_timeout_seconds=300.0,
    )

    config = optiverse.config.OptimizerConfig(
        directory=directory,
        generator=AgentGenerator.from_env(),
        max_iterations=1000,
        problem=problem,
        search_strategy=optiverse.search_strategies.IteratedLocalSearch(
            max_iterations_without_improvements=10
        ),
    )

    optiverse.optimizer.Optimizer(config=config).run()


if __name__ == "__main__":
    main()
