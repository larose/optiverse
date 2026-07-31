#!/usr/bin/env python3
"""Integer compression evaluator.

    evaluate.py validate <codebase_dir>   exit 0 if it builds and round-trips
    evaluate.py score    <codebase_dir>   average decompression time in ms

Both modes assemble a Go module, build it, run it, and delete it:

    workspace/
      go.mod  main.go        <- from here
      candidate/             <- the codebase, verbatim
      bench                  <- the built binary

The candidate is its own package, imported by `main.go` as `harness/candidate`.
It cannot stand in for anything the harness owns, whatever it names its files, so
there is no reserved-name list and nothing to overwrite. Only `go.mod` and the
one program are copied, never this directory wholesale: an `evaluate.py` inside
the workspace would let a candidate score itself.

Scoring builds once and then runs the binary several times, each in a directory
of its own. Separate processes, so nothing one run leaves in package state is
there for the next.

What the harness reports and what this file measures are deliberately different.
The binary writes its timings to a file — a candidate printing debug output
cannot corrupt its own score — and writes out the compressed bytes, which this
file sizes itself rather than trusting a ratio the harness was asked to print.
The timings are still taken inside the candidate's own process, which is what a
determined candidate could falsify; see README.md.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import dataset
from optiverse.evaluator_main import run

HARNESS_DIRECTORY = Path(__file__).parent

CANDIDATE_DIRECTORY_NAME = "candidate"
BINARY_NAME = "bench"
MEASUREMENTS_NAME = "result.json"
COMPRESSED_NAME = "result.bin"

# The module belongs to the harness. A candidate that ships one of these turns
# its own directory into a separate module, which drops it out of the build; say
# so rather than letting the agent read a confusing Go error.
MODULE_FILE_NAMES = frozenset({"go.mod", "go.sum"})

SCORE_RUNS = 3

# Sized to fit inside the timeouts optimize.py gives optiverse, worst case
# included: a build plus every run has to finish before the outer one fires.
# Otherwise a merely slow candidate is reported as a broken evaluator, and on the
# validate path the agent is told the setup is at fault rather than its code.
BUILD_TIMEOUT_SECONDS = 120
VALIDATE_TIMEOUT_SECONDS = 30
SCORE_TIMEOUT_SECONDS = 120

Metrics = Dict[str, Union[int, float]]


def _candidate_sources(codebase: Path) -> List[Path]:
    """The Go files that make up the package. A solution may be more than one.

    The root only, because that is what `package candidate` is: Go allows one
    package per directory, so anything in a subdirectory is a package of its own
    and is not compiled unless something imports it. Counting those would report
    lines that never reach the build.
    """
    return sorted(path for path in codebase.glob("*.go") if path.is_file())


def _rejection(codebase: Path) -> Optional[str]:
    """Why this codebase cannot be built at all, or None."""
    if not _candidate_sources(codebase):
        return "the codebase contains no .go files at its root"

    for path in sorted(codebase.rglob("*")):
        if path.is_file() and path.name in MODULE_FILE_NAMES:
            return (
                f"the codebase contains {path.relative_to(codebase)}; the Go "
                "module belongs to the harness, so do not add one"
            )

    return None


def _prepare(codebase: Path, program: str, workspace: Path) -> None:
    """Lay out a buildable module: harness at the root, candidate one level down."""
    shutil.copy2(HARNESS_DIRECTORY / "go.mod", workspace / "go.mod")
    shutil.copy2(HARNESS_DIRECTORY / program / "main.go", workspace / "main.go")
    shutil.copytree(codebase, workspace / CANDIDATE_DIRECTORY_NAME)


def _build(workspace: Path) -> bool:
    """Compile the workspace. Compiler errors go to stderr for the agent to read.

    The build keeps the real environment: it needs the toolchain, and it needs a
    warm GOCACHE, which lives under HOME. `GOPROXY=off` is what enforces
    problem.md's standard-library-only rule — an import of anything else fails
    here instead of being fetched.
    """
    try:
        result = subprocess.run(
            ["go", "build", "-o", BINARY_NAME, "."],
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=BUILD_TIMEOUT_SECONDS,
            env={**os.environ, "GOPROXY": "off"},
        )
    except subprocess.TimeoutExpired:
        print(f"go build exceeded {BUILD_TIMEOUT_SECONDS}s", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("the go toolchain is not installed", file=sys.stderr)
        return False

    sys.stderr.write(result.stdout)
    sys.stderr.write(result.stderr)

    return result.returncode == 0


def _execute(binary: Path, arguments: List[str], timeout_seconds: float) -> bool:
    """Run a built binary in its own directory, with the environment scrubbed.

    Scrubbed the way tsp scrubs it: a candidate cannot leave itself notes between
    runs in the obvious places. A literal /tmp path still could.
    """
    directory = binary.parent

    try:
        result = subprocess.run(
            [str(binary), *arguments],
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(directory),
                "TMPDIR": str(directory),
            },
        )
    except subprocess.TimeoutExpired:
        print(f"the run exceeded {timeout_seconds:g}s and was killed", file=sys.stderr)
        return False

    sys.stderr.write(result.stderr)

    if result.returncode != 0:
        print(f"the run exited {result.returncode}", file=sys.stderr)
        return False

    return True


def _read_number(measurements: Dict[str, object], name: str) -> Optional[float]:
    value = measurements.get(name)

    # bool is an int subclass, so it has to be rejected explicitly.
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        print(f"the harness reported no {name}", file=sys.stderr)
        return None

    return float(value)


def _read_measurements(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        print("the run produced no measurements", file=sys.stderr)
        return None

    try:
        parsed = cast(object, json.loads(path.read_text()))
    except json.JSONDecodeError:
        print("the measurements are not valid JSON", file=sys.stderr)
        return None

    if not isinstance(parsed, dict):
        print("the measurements are not an object", file=sys.stderr)
        return None

    return cast(Dict[str, object], parsed)


def validate(codebase: Path) -> bool:
    reason = _rejection(codebase)

    if reason is not None:
        print(reason, file=sys.stderr)
        return False

    with tempfile.TemporaryDirectory() as raw_workspace:
        workspace = Path(raw_workspace)
        _prepare(codebase, "validate", workspace)

        if not _build(workspace):
            return False

        return _execute(workspace / BINARY_NAME, [], VALIDATE_TIMEOUT_SECONDS)


def _run_once(
    binary: Path, run_directory: Path, *, write_compressed: bool
) -> Optional[Dict[str, object]]:
    """One measurement, in a directory nothing else has touched.

    The compressed bytes are wanted once, to size them. Writing them on every run
    would put half a gigabyte through TMPDIR for each one, which for a candidate
    that barely compresses is more I/O than the thing being measured.
    """
    executable = run_directory / BINARY_NAME
    shutil.copy2(binary, executable)

    arguments = [
        "-instance",
        str(dataset.BINARY_FILE),
        "-output",
        MEASUREMENTS_NAME,
    ]

    if write_compressed:
        arguments += ["-compressed", COMPRESSED_NAME]

    if not _execute(executable, arguments, SCORE_TIMEOUT_SECONDS):
        return None

    return _read_measurements(run_directory / MEASUREMENTS_NAME)


def score(codebase: Path) -> Tuple[Optional[float], Metrics]:
    sources = _candidate_sources(codebase)
    metrics: Metrics = {
        "file_count": len(sources),
        "line_count": sum(source.read_text().count("\n") for source in sources),
    }

    reason = _rejection(codebase)

    if reason is not None:
        print(reason, file=sys.stderr)
        return None, metrics

    if not dataset.BINARY_FILE.is_file():
        print(
            f"the benchmark dataset is missing: {dataset.BINARY_FILE}\n"
            f"run: python {Path(dataset.__file__)}",
            file=sys.stderr,
        )
        return None, metrics

    with tempfile.TemporaryDirectory() as raw_workspace:
        workspace = Path(raw_workspace)
        _prepare(codebase, "score", workspace)

        if not _build(workspace):
            return None, metrics

        binary = workspace / BINARY_NAME
        decompression_times: List[float] = []
        compression_times: List[float] = []
        compressed_size: Optional[int] = None

        for attempt in range(SCORE_RUNS):
            print(f"=== run {attempt + 1}/{SCORE_RUNS} ===", file=sys.stderr)

            # A directory of its own per run, so that what one run writes -- to
            # the working directory, to TMPDIR, to HOME -- is not there for the
            # next one to find.
            with tempfile.TemporaryDirectory() as raw_run_directory:
                run_directory = Path(raw_run_directory)
                first_run = attempt == 0
                measurements = _run_once(
                    binary, run_directory, write_compressed=first_run
                )

                if measurements is None:
                    return None, metrics

                decompression = _read_number(measurements, "decompression_time_ms")
                compression = _read_number(measurements, "compression_time_ms")

                if decompression is None or compression is None:
                    return None, metrics

                if first_run:
                    compressed_path = run_directory / COMPRESSED_NAME

                    if not compressed_path.is_file():
                        print("the run produced no compressed output", file=sys.stderr)
                        return None, metrics

                    compressed_size = compressed_path.stat().st_size
                    print(f"compressed: {compressed_size} bytes", file=sys.stderr)

                print(f"decompression: {decompression:.1f} ms", file=sys.stderr)

                decompression_times.append(decompression)
                compression_times.append(compression)

    if compressed_size is None or compressed_size == 0:
        print("the run produced no compressed output", file=sys.stderr)
        return None, metrics

    # Sized here rather than reported by the harness, so the ratio is a
    # measurement and not a claim. Recorded, not scored.
    metrics["compression_ratio"] = dataset.BINARY_FILE.stat().st_size / compressed_size
    metrics["compression_time_ms"] = sum(compression_times) / len(compression_times)

    average = sum(decompression_times) / len(decompression_times)

    metrics["best_decompression_time_ms"] = min(decompression_times)
    metrics["worst_decompression_time_ms"] = max(decompression_times)

    return average, metrics


if __name__ == "__main__":
    run(score=score, validate=validate)
