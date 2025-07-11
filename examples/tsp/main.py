from datetime import datetime
import logging
from pathlib import Path
from openai import OpenAI
import optiverse
import shutil
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple, Optional


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TSPEvaluator(optiverse.evaluator.Evaluator):
    def __init__(self):
        pass

    def _evaluate_in_temp_dir(
        self, code: str, temp_dir: Path
    ) -> optiverse.evaluator.EvaluatorResult:
        """
        Evaluate a TSP solution by running it in a temporary directory.

        Args:
            code: The solution code as a string
            temp_dir: Path to temporary directory for evaluation

        Returns:
            EvaluatorResult with artifacts and average tour distance
        """
        # Write the solution file
        Path(temp_dir / "solver.py").write_text(code)

        # Copy necessary files
        shutil.copy2(
            Path(__file__).parent / "solution" / "a280.tsp", temp_dir / "a280.tsp"
        )
        shutil.copy2(
            Path(__file__).parent / "solution" / "context.py", temp_dir / "context.py"
        )
        shutil.copy2(
            Path(__file__).parent / "solution" / "main.py", temp_dir / "main.py"
        )

        scores: List[float] = []
        artifacts: Dict[str, str] = {}

        # Run 3 times and collect scores and artifacts
        for i in range(3):
            score, stdout, stderr = self._run(temp_dir)
            logger.info(f"Score {i + 1}: {score}")
            # Store artifacts for this run
            artifacts[f"{i + 1}_stdout.txt"] = stdout
            artifacts[f"{i + 1}_stderr.txt"] = stderr

            if score is None:
                return optiverse.evaluator.EvaluatorResult(
                    artifacts=artifacts, score=None
                )

            scores.append(score)

        # All runs succeeded, calculate average
        average_score = sum(scores) / len(scores)
        return optiverse.evaluator.EvaluatorResult(
            artifacts=artifacts, score=average_score
        )

    def _execute_subprocess(self, temp_dir: Path) -> subprocess.CompletedProcess[str]:
        """Execute the subprocess and return the result."""
        return subprocess.run(
            ["python", "main.py"],
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=40,
        )

    def _run(self, temp_dir: Path) -> Tuple[Optional[float], str, str]:
        """
        Execute the runner and extract the tour distance.

        Args:
            temp_dir: Path to temporary directory containing solution files

        Returns:
            Tuple of (tour distance or None if failed, stdout, stderr)
        """
        try:
            result = self._execute_subprocess(temp_dir)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            logger.error(f"Error running solution in {temp_dir}", exc_info=True)
            return None, "", ""

        # Now we know we have a valid result
        stdout = result.stdout
        stderr = result.stderr

        # Extract the distance from stdout
        for line in stdout.split("\n"):
            if line.startswith(">>>"):
                distance_str = line.replace(">>>", "").strip()
                return float(distance_str), stdout, stderr

        logger.error(f"No valid output found in {temp_dir}:\n{stdout}")
        # If no output found, return None
        return None, stdout, stderr

    def evaluate(self, code: str) -> optiverse.evaluator.EvaluatorResult:
        with tempfile.TemporaryDirectory() as temp_dir:
            return self._evaluate_in_temp_dir(code=code, temp_dir=Path(temp_dir))


def main():
    config = optiverse.config.Config(
        llm=optiverse.config.LLM(
            model="gemini-2.0-flash",
            client=OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
        ),
        max_iterations=25,
        problem=optiverse.config.Problem(
            description=open(Path(__file__).parent / "problem.md").read(),
            initial_solution=open(
                Path(__file__).parent / "solution" / "solver.py"
            ).read(),
        ),
    )

    store_directory = Path("tmp") / datetime.now().strftime("%Y%m%d_%H%M%S")
    store_directory.mkdir(exist_ok=True, parents=True)

    evaluator = TSPEvaluator()

    optimizer = optiverse.optimizer.Optimizer(
        config=config,
        evaluator=evaluator,
        prompt_generator=optiverse.prompt_generator.EvolutionaryPromptGenerator(),
        store=optiverse.store.FileSystemStore(directory=store_directory),
    )
    optimizer.run()


if __name__ == "__main__":
    main()
