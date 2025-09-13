import os
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging

import optiverse

logger = logging.getLogger(__name__)


def format_grid(grid):
    """Format a grid as ASCII for better LLM readability"""
    return '\n'.join(' '.join(str(cell) for cell in row) for row in grid)


class ARCEvaluator(optiverse.evaluator.Evaluator):
    def __init__(self, challenge_id: str):
        self.challenge_id = challenge_id

        # Load challenge data for prompt generation
        from .solution.data_loader import find_challenge_by_id, load_arc_data
        data_dir = Path(__file__).parent / "data"
        arc_data = load_arc_data(data_dir)
        self.task, self.solutions = find_challenge_by_id(arc_data, challenge_id)

    def get_training_examples_prompt(self) -> str:
        """Generate training examples section for the prompt"""
        if not self.task.train:
            return ""

        prompt = f"\n\n## Training Examples for Challenge ID: {self.challenge_id}\n\n"

        for i, pair in enumerate(self.task.train, 1):
            prompt += f"Example {i}:\n"
            prompt += "Input:\n"
            prompt += format_grid(pair.input)
            prompt += "\n\nOutput:\n"
            prompt += format_grid(pair.output)
            prompt += "\n\n"

        prompt += "Your task: Implement the transform() function to apply this pattern to test cases."
        return prompt

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
        # Create solution subdirectory to maintain proper directory structure
        solution_temp_dir = temp_dir / "solution"
        solution_temp_dir.mkdir()

        # Write the solution file in solution subdirectory
        (solution_temp_dir / "solver.py").write_text(code)

        # Copy necessary template files to solution subdirectory
        solution_dir = Path(__file__).parent / "solution"
        shutil.copy2(
            solution_dir / "data_loader.py", solution_temp_dir / "data_loader.py"
        )
        shutil.copy2(solution_dir / "main.py", solution_temp_dir / "main.py")

        # Create symlink to data directory at temp root level
        data_dir = Path(__file__).parent / "data"
        (temp_dir / "data").symlink_to(data_dir)

        # Calculate basic metrics from the code
        metrics = self._calculate_code_metrics(code)

        # Run the evaluation from solution subdirectory
        score, stdout, stderr = self._run(solution_temp_dir)

        artifacts = {
            "stdout.txt": stdout,
            "stderr.txt": stderr,
        }

        if score is not None:
            metrics["accuracy"] = score
            metrics["challenge_id"] = self.challenge_id

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
