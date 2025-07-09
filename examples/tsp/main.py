from datetime import datetime
import logging
from pathlib import Path
from openai import OpenAI
import optiverse
import shutil
import subprocess
import tempfile
import os


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TSPEvaluator(optiverse.evaluator.Evaluator):
    def __init__(self):
        pass

    def _evaluate_in_temp_dir(self, file: str, temp_dir: Path) -> float:
        """
        Evaluate a TSP solution by running it in a temporary directory.

        Args:
            file: The solution code as a string
            temp_dir: Path to temporary directory for evaluation

        Returns:
            The average tour distance across 3 runs
        """
        # Write the solution file
        Path(temp_dir / "solver.py").write_text(file)

        # Copy necessary files
        shutil.copy2(
            Path(__file__).parent / "solution" / "main.py", temp_dir / "main.py"
        )
        shutil.copy2(
            Path(__file__).parent / "solution" / "a280.tsp", temp_dir / "a280.tsp"
        )

        scores = []

        # Run 3 times and collect scores
        for i in range(3):
            distance = self._run(temp_dir)
            scores.append(distance)

        # Calculate average, including infinite values
        return sum(scores) / len(scores)

    def _run(self, temp_dir: Path) -> float:
        """
        Execute the runner and extract the tour distance.

        Args:
            temp_dir: Path to temporary directory containing solution files

        Returns:
            The tour distance from this run
        """
        try:
            # Execute the runner script
            result = subprocess.run(
                ["python", "main.py"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=40,
            )

            # Extract the distance from stdout
            for line in result.stdout.split("\n"):
                if line.startswith(">>>"):
                    distance_str = line.replace(">>>", "").strip()
                    return float(distance_str)

            # If no output found, return a large penalty
            return float("inf")

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            # Return a large penalty for failed runs
            return float("inf")

    def evaluate(self, file: str) -> float:
        with tempfile.TemporaryDirectory() as temp_dir:
            return self._evaluate_in_temp_dir(file=file, temp_dir=Path(temp_dir))


PROBLEM_DESCRIPTION = """
Write a Python program that implements a heuristic for the Traveling Salesman Problem (TSP).

Your script must define a `solve` function with the following signature:

```
def solve(context: Context) -> None:
    pass
```

- The `Context` object provides the TSP instance data and methods to report solutions.
- You may only modify the `solve` function. You are allowed to define and call additional helper functions within your script, but you cannot modify the `Context` class itself.

Your implementation should:

- Access the TSP data through the Context object.
- Call `context.report_new_best_solution(solution)` only when you have found a better solution than previously reported.
- The most recently reported solution will be used as the final answer when time runs out.

Important: Any solution reported after time runs out will be ignored. Ensure your implementation checks the remaining time and only reports solutions while time is available.
"""


def main():
    config = optiverse.config.Config(
        llm=optiverse.config.LLM(
            model="gemini-1.5-flash",
            client=OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
        ),
        max_iterations=10,
        problem=optiverse.config.Problem(
            description=PROBLEM_DESCRIPTION,
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
