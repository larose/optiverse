#!/usr/bin/env python3
"""Integer compression evaluator.

    evaluate.py validate <codebase_dir>   exit 0 if it builds and round-trips
    evaluate.py score    <codebase_dir>   average decompression time in ms

The evolvable unit is the whole Go package, so the candidate may contain any
number of `.go` files. Both modes copy the candidate's `.go` files into a
temporary directory and add this example's own harness on top. The harness wins,
so a candidate cannot change how it is measured.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from optiverse.evaluator_main import run

EXAMPLE_DIRECTORY = Path(__file__).parent
HARNESS_DIRECTORY = EXAMPLE_DIRECTORY / "harness"
DATA_FILE = HARNESS_DIRECTORY / "ts.txt"

# Names the harness owns. A candidate file with one of these names is dropped
# rather than allowed to shadow the measurement code.
RESERVED_NAMES = frozenset({"main.go", "go.mod", "go.sum"})

BUILD_TIMEOUT_SECONDS = 300
VALIDATE_TIMEOUT_SECONDS = 120
SCORE_TIMEOUT_SECONDS = 1800

Metrics = Dict[str, Union[int, float]]


def _candidate_sources(codebase: Path) -> List[Path]:
    return sorted(
        path
        for path in codebase.rglob("*.go")
        if path.is_file() and path.name not in RESERVED_NAMES
    )


def _prepare(codebase: Path, workspace: Path, harness_main: Path) -> Optional[str]:
    """Assemble a buildable package. Returns an error message, or None."""
    sources = _candidate_sources(codebase)

    if not sources:
        return "the codebase contains no .go files"

    for source in sources:
        shutil.copy2(source, workspace / source.name)

    shutil.copy2(HARNESS_DIRECTORY / "go.mod", workspace / "go.mod")
    shutil.copy2(harness_main, workspace / "main.go")

    return None


def _go(
    arguments: List[str], workspace: Path, timeout_seconds: int
) -> Optional["subprocess.CompletedProcess[str]"]:
    try:
        return subprocess.run(
            ["go", *arguments],
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(f"go {' '.join(arguments)} exceeded {timeout_seconds}s", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("the go toolchain is not installed", file=sys.stderr)
        return None


def validate(codebase: Path) -> bool:
    with tempfile.TemporaryDirectory() as raw_workspace:
        workspace = Path(raw_workspace)

        error = _prepare(
            codebase, workspace, HARNESS_DIRECTORY / "validate" / "main.go"
        )
        if error is not None:
            print(error, file=sys.stderr)
            return False

        result = _go(["run", "."], workspace, VALIDATE_TIMEOUT_SECONDS)

        if result is None:
            return False

        # Compiler errors and round-trip failures both land here, and the agent
        # reads this to fix its code.
        sys.stderr.write(result.stderr)
        sys.stdout.write("")

        if result.returncode != 0:
            return False

    return True


def _parse_markers(stdout: str) -> Dict[str, float]:
    values: Dict[str, float] = {}

    for line in stdout.splitlines():
        if not line.startswith(">>>"):
            continue

        name, _, raw_value = line.removeprefix(">>>").partition(":")
        values[name.strip()] = float(raw_value.strip())

    return values


def score(codebase: Path) -> Tuple[Optional[float], Metrics]:
    metrics: Metrics = {"go_file_count": len(_candidate_sources(codebase))}

    if not DATA_FILE.is_file():
        print(
            f"benchmark data is missing: {DATA_FILE}\n"
            f"run: python {EXAMPLE_DIRECTORY / 'data_generator.py'}",
            file=sys.stderr,
        )
        return None, metrics

    with tempfile.TemporaryDirectory() as raw_workspace:
        workspace = Path(raw_workspace)

        error = _prepare(codebase, workspace, HARNESS_DIRECTORY / "score" / "main.go")
        if error is not None:
            print(error, file=sys.stderr)
            return None, metrics

        result = _go(["run", ".", str(DATA_FILE)], workspace, SCORE_TIMEOUT_SECONDS)

        if result is None:
            return None, metrics

        sys.stderr.write(result.stderr)

        if result.returncode != 0:
            print(f"go run exited {result.returncode}", file=sys.stderr)
            return None, metrics

        values = _parse_markers(result.stdout)

    if "decompression_time" not in values:
        print("harness printed no decompression_time", file=sys.stderr)
        return None, metrics

    metrics.update(
        {
            "compression_ratio": values.get("compression_ratio", 0.0),
            "compression_time": values.get("compression_time", 0.0),
            "decompression_time": values["decompression_time"],
        }
    )

    # Lower is better, matching the loop's minimisation.
    return values["decompression_time"], metrics


if __name__ == "__main__":
    run(score=score, validate=validate)
