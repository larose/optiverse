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


logger = logging.getLogger(__name__)


class IntegerCompressionEvaluator(optiverse.evaluator.Evaluator):
    def __init__(self, force_regen: bool) -> None:
        self._force_regen = force_regen

    def _evaluate_in_temp_dir(
        self, code: str, temp_dir: Path
    ) -> optiverse.evaluator.EvaluatorResult:
        # Generate test data files
        solution_dir = Path(__file__).parent / "solution"
        generate_test_files(solution_dir, self._force_regen)
        # Write the compressor.go file
        Path(temp_dir / "compressor.go").write_text(code)

        # Copy necessary files
        shutil.copy2(solution_dir / "main.go", temp_dir / "main.go")
        shutil.copy2(solution_dir / "go.mod", temp_dir / "go.mod")

        # Copy test data files to temp directory
        for filename in ["data_a.bin", "data_b.bin", "data_c.bin"]:
            shutil.copy2(solution_dir / filename, temp_dir / filename)

        metrics: Dict[str, Union[int, float]] = {}
        artifacts: Dict[str, str] = {}

        # Test on different datasets
        test_configs = [
            ("data_a.bin", "a"),
            ("data_b.bin", "b"),
            ("data_c.bin", "c"),
        ]

        overall_decompression_times: List[float] = []

        for test_file, size_name in test_configs:
            decompression_times: List[float] = []
            compression_ratios: List[float] = []
            compression_times: List[float] = []

            for run in range(3):  # 3 runs per dataset
                result = self._run_go_program(temp_dir, test_file)

                artifacts[f"{test_file}_{run+1}_stdout.txt"] = result.stdout
                artifacts[f"{test_file}_{run+1}_stderr.txt"] = result.stderr

                if result.decompression_time is None:
                    return optiverse.evaluator.EvaluatorResult(
                        artifacts=artifacts, metrics=metrics, score=None
                    )

                if result.compression_ratio is None:
                    raise ValueError("compression_ratio not found in program output")
                if result.compression_time is None:
                    raise ValueError("compression_time not found in program output")

                decompression_times.append(result.decompression_time)
                compression_ratios.append(result.compression_ratio)
                compression_times.append(result.compression_time)
                overall_decompression_times.append(result.decompression_time)

            # Calculate averages for this dataset size
            avg_decompression_time = sum(decompression_times) / len(decompression_times)
            avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
            avg_compression_time = sum(compression_times) / len(compression_times)

            # Store metrics with size-specific names
            metrics[f"{size_name}_decompression_time"] = avg_decompression_time
            metrics[f"{size_name}_compression_ratio"] = avg_compression_ratio
            metrics[f"{size_name}_compression_time"] = avg_compression_time

            logger.info(
                f"Results for {test_file}: decompression={avg_decompression_time:.0f}ns, ratio={avg_compression_ratio:.3f}"
            )

        # Overall score is the average decompression time across all datasets
        overall_score = sum(overall_decompression_times) / len(
            overall_decompression_times
        )

        return optiverse.evaluator.EvaluatorResult(
            artifacts=artifacts, metrics=metrics, score=overall_score
        )

    def _build_go_program(self, temp_dir: Path) -> subprocess.CompletedProcess[str]:
        """Build the Go program"""
        return subprocess.run(
            ["go", "build", "-o", "compressor"],
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )

    def _run_compiled_program(
        self, temp_dir: Path, test_file: str
    ) -> subprocess.CompletedProcess[str]:
        """Run the compiled Go program"""
        return subprocess.run(
            ["./compressor", test_file],
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
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
        """Build and run the Go program"""
        try:
            build_result = self._build_go_program(temp_dir)
            if build_result.returncode != 0:
                logger.error(f"Go build failed: {build_result.stderr}")
                return ProgramRunResult(
                    decompression_time=None,
                    compression_ratio=None,
                    compression_time=None,
                    stdout=build_result.stdout,
                    stderr=build_result.stderr,
                )

            run_result = self._run_compiled_program(temp_dir, test_file)
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

        except Exception as e:
            logger.error(f"Error running Go program: {e}")
            return ProgramRunResult(
                decompression_time=None,
                compression_ratio=None,
                compression_time=None,
                stdout="",
                stderr=str(e),
            )

    def evaluate(self, code: str) -> optiverse.evaluator.EvaluatorResult:
        with tempfile.TemporaryDirectory() as temp_dir:
            return self._evaluate_in_temp_dir(code=code, temp_dir=Path(temp_dir))
