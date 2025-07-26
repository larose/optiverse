import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import optiverse
from data_generator import generate_test_files


@dataclass
class ProgramRunResult:
    decompression_time: Optional[float]
    compression_ratio: Optional[float]
    compression_time: Optional[float]
    stdout: str
    stderr: str


@dataclass
class DatasetStats:
    decompression_time: float
    compression_ratio: float
    compression_time: float
    dataset_name: str


logger = logging.getLogger(__name__)


class IntegerCompressionEvaluator(optiverse.evaluator.Evaluator):

    def _setup_temp_directory(self, code: str, temp_dir: Path) -> None:
        """Setup temporary directory with necessary files"""
        # Generate test data files
        solution_dir = Path(__file__).parent / "solution"
        generate_test_files(solution_dir)

        # Write the compressor.go file
        Path(temp_dir / "compressor.go").write_text(code)

        # Copy necessary files
        shutil.copy2(solution_dir / "main.go", temp_dir / "main.go")
        shutil.copy2(solution_dir / "go.mod", temp_dir / "go.mod")

        # Create symlink to test data file instead of copying
        (temp_dir / "ts.txt").symlink_to(solution_dir / "ts.txt")

    def _run_single_dataset_test(
        self,
        temp_dir: Path,
        test_file: str,
        size_name: str,
        artifacts: Dict[str, str],
    ) -> Optional[DatasetStats]:
        """Run tests on a single dataset and return calculated stats"""
        result = self._run_go_program(temp_dir, test_file)

        # Store artifacts
        artifacts[f"{test_file}_stdout.txt"] = result.stdout
        artifacts[f"{test_file}_stderr.txt"] = result.stderr

        if result.decompression_time is None:
            # Early exit - mark this dataset as failed
            return None

        if result.compression_ratio is None:
            raise ValueError("compression_ratio not found in program output")
        if result.compression_time is None:
            raise ValueError("compression_time not found in program output")

        # Create DatasetStats directly from the averaged results returned by Go program
        dataset_stats = DatasetStats(
            decompression_time=result.decompression_time,
            compression_ratio=result.compression_ratio,
            compression_time=result.compression_time,
            dataset_name=size_name,
        )

        logger.info(
            f"Results for {size_name}: decompression={result.decompression_time:.0f}ms, ratio={result.compression_ratio:.3f}"
        )

        return dataset_stats

    def _run_dataset_tests(
        self,
        temp_dir: Path,
        test_configs: List[Tuple[str, str]],
        artifacts: Dict[str, str],
    ) -> Dict[str, Optional[DatasetStats]]:
        """Run tests on all datasets and return calculated stats for each data file"""
        dataset_stats: Dict[str, Optional[DatasetStats]] = {}

        for test_file, size_name in test_configs:
            stats = self._run_single_dataset_test(
                temp_dir, test_file, size_name, artifacts
            )
            dataset_stats[size_name] = stats

            if stats is None:
                # Early exit if dataset failed
                return dataset_stats

        return dataset_stats

    def _calculate_metrics_from_results(
        self, dataset_stats: Dict[str, Optional[DatasetStats]]
    ) -> Tuple[Dict[str, Union[int, float]], Optional[float]]:
        """Calculate metrics from dataset stats"""
        metrics: Dict[str, Union[int, float]] = {}
        overall_decompression_times: List[float] = []

        for size_name, stats in dataset_stats.items():
            if stats is None:
                # No stats for this dataset - return None score for early failure
                return metrics, None

            # Store metrics with size-specific names
            metrics[f"{size_name}_decompression_time"] = stats.decompression_time
            metrics[f"{size_name}_compression_ratio"] = stats.compression_ratio
            metrics[f"{size_name}_compression_time"] = stats.compression_time

            # Add the average decompression time for this dataset
            overall_decompression_times.append(stats.decompression_time)

        # Calculate overall score
        score = (
            sum(overall_decompression_times) / len(overall_decompression_times)
            if overall_decompression_times
            else None
        )

        return metrics, score

    def _evaluate_in_temp_dir(
        self, code: str, temp_dir: Path
    ) -> optiverse.evaluator.EvaluatorResult:
        """Evaluate Go compression solution in temporary directory"""
        artifacts: Dict[str, str] = {}

        # Setup phase
        self._setup_temp_directory(code, temp_dir)

        # Test execution phase
        test_configs = [
            ("ts.txt", "ts"),
        ]
        dataset_results = self._run_dataset_tests(temp_dir, test_configs, artifacts)

        # Results processing phase
        metrics, score = self._calculate_metrics_from_results(dataset_results)

        return optiverse.evaluator.EvaluatorResult(
            artifacts=artifacts, metrics=metrics, score=score
        )

    def _parse_program_output(
        self, stdout: str
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Parse metrics from program output"""
        decompression_time = None
        compression_ratio = None
        compression_time = None

        for line in stdout.split("\n"):
            if line.startswith(">>> decompression_time:"):
                decompression_time = float(line.split(":")[1].strip())
            elif line.startswith(">>> compression_ratio:"):
                compression_ratio = float(line.split(":")[1].strip())
            elif line.startswith(">>> compression_time:"):
                compression_time = float(line.split(":")[1].strip())

        return decompression_time, compression_ratio, compression_time

    def _run_go_program(self, temp_dir: Path, test_file: str) -> ProgramRunResult:
        """Run the Go program directly with go run"""

        run_result = subprocess.run(
            ["go", "run", ".", test_file],
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )

        if run_result.returncode != 0:
            logger.error(f"Go program failed: {run_result.stderr}")
            return ProgramRunResult(
                decompression_time=None,
                compression_ratio=None,
                compression_time=None,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
            )

        decompression_time, compression_ratio, compression_time = (
            self._parse_program_output(run_result.stdout)
        )

        if decompression_time is None:
            logger.error("No decompression time found in output")
            return ProgramRunResult(
                decompression_time=None,
                compression_ratio=None,
                compression_time=None,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
            )

        return ProgramRunResult(
            decompression_time=decompression_time,
            compression_ratio=compression_ratio,
            compression_time=compression_time,
            stdout=run_result.stdout,
            stderr=run_result.stderr,
        )

    def evaluate(self, code: str) -> optiverse.evaluator.EvaluatorResult:
        with tempfile.TemporaryDirectory() as temp_dir:
            return self._evaluate_in_temp_dir(code=code, temp_dir=Path(temp_dir))
