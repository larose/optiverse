import os
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging

import optiverse
from .solution.data_loader import find_challenge_by_id, load_arc_data

logger = logging.getLogger(__name__)


class ARCEvaluator(optiverse.evaluator.Evaluator):
    def __init__(self, challenge_id: str):
        self.challenge_id = challenge_id

        # Load ARC data to validate challenge exists
        data_dir = Path(__file__).parent / "data"
        arc_data = load_arc_data(data_dir)
        self.task, self.solutions = find_challenge_by_id(arc_data, challenge_id)

    def _calculate_code_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate simple metrics from the solution code"""
        metrics: Dict[str, Any] = {}

        # Simple line count
        metrics["line_count"] = code.count("\n")

        return metrics

    def _evaluate_in_temp_dir(
        self, code: str, temp_dir: Path
    ) -> optiverse.evaluator.EvaluatorResult:
        """
        Evaluate an ARC solution by running it in a temporary directory.

        Args:
            code: The solution code as a string
            temp_dir: Path to temporary directory for evaluation

        Returns:
            EvaluatorResult with artifacts and score
        """
        # Write the solution file
        (temp_dir / "solver.py").write_text(code)

        # Copy necessary template files
        solution_dir = Path(__file__).parent / "solution"
        shutil.copy2(solution_dir / "data_loader.py", temp_dir / "data_loader.py")
        shutil.copy2(solution_dir / "main.py", temp_dir / "main.py")

        # Copy data files
        data_dir = Path(__file__).parent / "data"
        temp_data_dir = temp_dir / "data"
        temp_data_dir.mkdir()
        for data_file in data_dir.glob("*.json"):
            shutil.copy2(data_file, temp_data_dir / data_file.name)

        # Calculate basic metrics from the code
        metrics = self._calculate_code_metrics(code)

        # Run the evaluation
        score, stdout, stderr = self._run(temp_dir)

        artifacts = {
            "stdout.txt": stdout,
            "stderr.txt": stderr,
        }

        if score is not None:
            metrics["accuracy"] = score
            metrics["challenge_id"] = self.challenge_id
            metrics["test_count"] = len(self.task.test)
            if self.solutions:
                metrics["has_solutions"] = True
                metrics["solution_count"] = len(self.solutions)
            else:
                metrics["has_solutions"] = False

        return optiverse.evaluator.EvaluatorResult(
            artifacts=artifacts,
            metrics=metrics,
            score=score,
        )

    def _execute_subprocess(self, temp_dir: Path) -> subprocess.CompletedProcess[str]:
        """Execute the subprocess and return the result."""
        env = os.environ.copy()
        env["CHALLENGE_ID"] = self.challenge_id

        return subprocess.run(
            ["python", "main.py"],
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=40,
            env=env,
        )

    def _run(self, temp_dir: Path) -> Tuple[Optional[float], str, str]:
        """
        Execute the runner and extract the accuracy score.

        Args:
            temp_dir: Path to temporary directory containing solution files

        Returns:
            Tuple of (accuracy score or None if failed, stdout, stderr)
        """
        try:
            result = self._execute_subprocess(temp_dir)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            logger.error(f"Error running solution in {temp_dir}", exc_info=True)
            return None, "", ""

        stdout = result.stdout
        stderr = result.stderr

        # Extract the score from stdout
        for line in stdout.split("\n"):
            if line.startswith(">>>ACCURACY:"):
                score_str = line.replace(">>>ACCURACY:", "").strip()
                try:
                    return float(score_str), stdout, stderr
                except ValueError:
                    logger.error(f"Invalid accuracy score: {score_str}")
            elif line.startswith(">>>COMPLETION:"):
                score_str = line.replace(">>>COMPLETION:", "").strip()
                try:
                    return float(score_str), stdout, stderr
                except ValueError:
                    logger.error(f"Invalid completion score: {score_str}")

        logger.error(f"No valid output found in {temp_dir}:\n{stdout}")
        return None, stdout, stderr

    def evaluate(self, code: str) -> optiverse.evaluator.EvaluatorResult:
        """
        Evaluate an ARC solver implementation.

        Args:
            code: The solver code as a string

        Returns:
            EvaluatorResult with accuracy score and artifacts
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            return self._evaluate_in_temp_dir(code=code, temp_dir=Path(temp_dir))
