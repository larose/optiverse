from datetime import datetime
import logging
from pathlib import Path
from evaluator import IntegerCompressionEvaluator
import optiverse
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    raw_directory = os.getenv("DIRECTORY")

    if raw_directory is None:
        directory = Path("tmp") / datetime.now().strftime("%Y%m%d_%H%M%S")
        directory.mkdir(exist_ok=True, parents=True)
    else:
        directory = Path(raw_directory)

    problem = optiverse.config.Problem(
        description=open(Path(__file__).parent / "problem.md").read(),
        initial_solution=open(
            Path(__file__).parent / "solution" / "compressor.go"
        ).read(),
        evaluator=IntegerCompressionEvaluator(),
    )

    config = optiverse.config.OptimizerConfig(
        llm=optiverse.config.create_llm_config_from_env(),
        max_iterations=10_000,
        problem=problem,
        search_strategy=optiverse.search_strategies.IteratedLocalSearch(
            max_iterations_without_improvements=10
        ),
        directory=directory,
    )

    optimizer = optiverse.optimizer.Optimizer(config=config)
    optimizer.run()


if __name__ == "__main__":
    main()
