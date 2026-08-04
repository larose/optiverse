#!/usr/bin/env python3
"""TSP evaluator.

    evaluate.py validate <codebase_dir>   exit 0 if the solver runs and is legal
    evaluate.py score    <codebase_dir>   average tour length over 3 runs

Both modes assemble a workspace, run the candidate in it, and delete it:

    workspace/
      run.py  context.py  instance.py  a280.tsp    <- from here
      candidate/                                   <- the codebase, verbatim

The candidate sits one level down and `run.py` appends it to `sys.path`, so
nothing it contains can shadow the harness or the standard library, and there is
nothing to overwrite or forbid. Only the four files are copied, never this
directory wholesale: an `evaluate.py` inside the workspace would let a candidate
run `score` on itself.

`run.py` hands back a tour and nothing else. The length is measured here, from
this directory's own copy of the instance, so a candidate chooses which tour it
submits but not what that tour is worth. Every scoring run gets its own
workspace, so nothing one run leaves behind is there for the next.
"""

import ast
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import instance
from optiverse.evaluator.main import run

HARNESS_DIRECTORY = Path(__file__).parent
HARNESS_FILES = ("context.py", "instance.py", "run.py")
INSTANCE_FILE = HARNESS_DIRECTORY / "a280.tsp"

CANDIDATE_DIRECTORY_NAME = "candidate"
SOLVER_NAME = "solver.py"
TOUR_NAME = "tour.json"

SCORE_RUNS = 3
SCORE_SECONDS = 30.0

# Validation only asks "does this run and produce a legal tour", so it gets a
# fraction of the time budget. A good solver reports its first tour early.
VALIDATE_SECONDS = 3.0

# A candidate can reach into the `Context` it was handed and move its own
# deadline; no in-process guard survives that. This is the bound that does — the
# run is killed, and so fails, once it overruns by this much.
OVERRUN_SLACK_SECONDS = 5.0

Metrics = Dict[str, Union[int, float]]


class NestedFunctionDetector(ast.NodeVisitor):
    """problem.md requires helpers to be defined at the top level."""

    def __init__(self) -> None:
        self.function_depth = 0
        self.has_nested = False

    def _visit_function(self, node: ast.AST) -> None:
        if self.function_depth > 0:
            self.has_nested = True
            return

        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def has_nested_functions(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Let the run surface the real error instead of reporting it as nesting.
        return False

    detector = NestedFunctionDetector()
    detector.visit(tree)
    return detector.has_nested


def _candidate_sources(codebase: Path) -> List[Path]:
    """Every Python file in the codebase. A solution may be more than one."""
    return sorted(path for path in codebase.rglob("*.py") if path.is_file())


def _prepare(codebase: Path, workspace: Path) -> None:
    for name in HARNESS_FILES:
        shutil.copy2(HARNESS_DIRECTORY / name, workspace / name)

    shutil.copy2(INSTANCE_FILE, workspace / INSTANCE_FILE.name)
    shutil.copytree(codebase, workspace / CANDIDATE_DIRECTORY_NAME)


def _run_once(codebase: Path, seconds: float) -> Optional[List[int]]:
    """Run the solver once, in a workspace of its own, and return its tour."""
    with tempfile.TemporaryDirectory() as raw_workspace:
        workspace = Path(raw_workspace)
        _prepare(codebase, workspace)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "run.py",
                    "--candidate",
                    CANDIDATE_DIRECTORY_NAME,
                    "--instance",
                    INSTANCE_FILE.name,
                    "--seconds",
                    str(seconds),
                    "--tour",
                    TOUR_NAME,
                ],
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=seconds + OVERRUN_SLACK_SECONDS,
                # Scrubbed, so a candidate cannot leave itself notes between runs
                # in the obvious places. A literal /tmp path still could.
                env={
                    "PATH": "/usr/bin:/bin",
                    "HOME": str(workspace),
                    "TMPDIR": str(workspace),
                },
            )
        except subprocess.TimeoutExpired:
            print(
                f"the solver ran past its {seconds:g}s budget and was killed",
                file=sys.stderr,
            )
            return None

        sys.stderr.write(result.stderr)

        if result.returncode != 0:
            print(f"the solver exited {result.returncode}", file=sys.stderr)
            return None

        return _read_tour(workspace / TOUR_NAME)


def _read_tour(tour_path: Path) -> Optional[List[int]]:
    if not tour_path.is_file():
        print("the solver produced no tour", file=sys.stderr)
        return None

    try:
        parsed = cast(object, json.loads(tour_path.read_text()))
    except json.JSONDecodeError:
        print("the tour is not valid JSON", file=sys.stderr)
        return None

    if not isinstance(parsed, list):
        print("the tour is not a list of city ids", file=sys.stderr)
        return None

    cities = cast(List[object], parsed)

    # bool is an int subclass, so it has to be rejected explicitly.
    if any(not isinstance(city, int) or isinstance(city, bool) for city in cities):
        print("the tour is not a list of city ids", file=sys.stderr)
        return None

    return [city for city in cities if isinstance(city, int)]


def _measure(tour: Optional[List[int]]) -> Optional[float]:
    """What a tour is worth, judged here rather than by whatever produced it."""
    if tour is None:
        return None

    coordinates = instance.parse(INSTANCE_FILE)
    reason = instance.illegal_reason(tour, len(coordinates))

    if reason:
        print(f"illegal tour: {reason}", file=sys.stderr)
        return None

    return instance.tour_length(tour, coordinates)


def validate(codebase: Path) -> bool:
    if not (codebase / SOLVER_NAME).is_file():
        print(
            f"{SOLVER_NAME} is missing from the root of the codebase", file=sys.stderr
        )
        return False

    for source in _candidate_sources(codebase):
        if has_nested_functions(source.read_text()):
            print(
                f"{source.relative_to(codebase)} defines nested functions, "
                "which problem.md forbids",
                file=sys.stderr,
            )
            return False

    return _measure(_run_once(codebase, VALIDATE_SECONDS)) is not None


def score(codebase: Path) -> Tuple[Optional[float], Metrics]:
    sources = _candidate_sources(codebase)
    metrics: Metrics = {
        "line_count": sum(source.read_text().count("\n") for source in sources),
        "file_count": len(sources),
    }

    lengths: List[float] = []

    for attempt in range(SCORE_RUNS):
        print(f"=== run {attempt + 1}/{SCORE_RUNS} ===", file=sys.stderr)
        length = _measure(_run_once(codebase, SCORE_SECONDS))

        if length is None:
            return None, metrics

        print(f"tour length: {length}", file=sys.stderr)
        lengths.append(length)

    average = sum(lengths) / len(lengths)

    metrics["best_length"] = min(lengths)
    metrics["worst_length"] = max(lengths)
    metrics["length_variance"] = sum((x - average) ** 2 for x in lengths) / len(lengths)

    return average, metrics


if __name__ == "__main__":
    run(score=score, validate=validate)
