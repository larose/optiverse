from pathlib import Path
import optiverse
import shutil
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TSPEvaluator(optiverse.evaluator.Evaluator):
    def _calculate_code_metrics(self, code: str) -> Dict[str, Union[int, float]]:
        """Calculate simple metrics from the solution code"""
        metrics: Dict[str, Union[int, float]] = {}

        # Simple line count: just count \n characters
        metrics["line_count"] = code.count("\n")

        return metrics

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

        # Calculate basic metrics from the code
        metrics = self._calculate_code_metrics(code)

        # Run 3 times and collect scores and artifacts
        for i in range(3):
            score, stdout, stderr = self._run(temp_dir)
            logger.info(f"Score {i + 1}: {score}")
            # Store artifacts for this run
            artifacts[f"{i + 1}_stdout.txt"] = stdout
            artifacts[f"{i + 1}_stderr.txt"] = stderr

            if score is None:
                return optiverse.evaluator.EvaluatorResult(
                    artifacts=artifacts, metrics=metrics, score=None
                )

            scores.append(score)

        # All runs succeeded, calculate average and score statistics
        average_score = sum(scores) / len(scores)
        score_variance = sum((s - average_score) ** 2 for s in scores) / len(scores)

        # Add score statistics to metrics
        metrics["score_variance"] = score_variance
        metrics["best_score"] = min(scores)
        metrics["worst_score"] = max(scores)

        return optiverse.evaluator.EvaluatorResult(
            artifacts=artifacts, metrics=metrics, score=average_score
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
