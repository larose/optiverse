from datetime import datetime
import logging
from pathlib import Path
from .evaluator import TSPEvaluator
import optiverse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    directory = Path("tmp") / datetime.now().strftime("%Y%m%d_%H%M%S")
    directory.mkdir(exist_ok=True, parents=True)

    evaluator = TSPEvaluator()

    problem = optiverse.config.Problem(
        description=open(Path(__file__).parent / "problem.md").read(),
        initial_solution=open(Path(__file__).parent / "solution" / "solver.py").read(),
        evaluator=evaluator,
    )

    config = optiverse.config.OptimizerConfig(
        llm=optiverse.config.create_llm_config_from_env(),
        max_iterations=100,
        problem=problem,
        search_strategy=optiverse.strategies.IteratedLocalSearch(
            max_iterations_without_improvements=10
        ),
        directory=directory,
    )

    optimizer = optiverse.optimizer.Optimizer(config=config)
    optimizer.run()


if __name__ == "__main__":
    main()
