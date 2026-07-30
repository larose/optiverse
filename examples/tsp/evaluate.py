#!/usr/bin/env python3
"""TSP evaluator.

    evaluate.py validate <codebase_dir>   exit 0 if the solver runs and is legal
    evaluate.py score    <codebase_dir>   average tour length over 3 runs

Both modes copy the candidate into a temporary directory and lay this example's
own harness over the top. The harness always wins, so a candidate cannot alter
how it is measured, and the stored codebase is never touched.
"""

import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from optiverse.evaluator_main import run

HARNESS_DIRECTORY = Path(__file__).parent / "harness"
HARNESS_FILES = ("a280.tsp", "context.py", "main.py")

SCORE_RUNS = 3
SCORE_TIMEOUT_SECONDS = 30
SCORE_KILL_AFTER_SECONDS = 40

# Validation only asks "does this run and produce a legal tour", so it gets a
# fraction of the time budget. A good solver reports its first tour early.
VALIDATE_TIMEOUT_SECONDS = 3
VALIDATE_KILL_AFTER_SECONDS = 20

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


def _prepare(codebase: Path, workspace: Path) -> None:
    shutil.copytree(codebase, workspace, dirs_exist_ok=True)

    for name in HARNESS_FILES:
        shutil.copy2(HARNESS_DIRECTORY / name, workspace / name)


def _run_once(
    workspace: Path, timeout_seconds: int, kill_after_seconds: int
) -> Optional[float]:
    """Run the solver once and return the tour length, or None on failure."""
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=kill_after_seconds,
            env={"TIMEOUT_SECONDS": str(timeout_seconds), "PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired:
        print(f"solver exceeded {kill_after_seconds}s", file=sys.stderr)
        return None

    sys.stderr.write(result.stderr)

    if result.returncode != 0:
        print(f"solver exited {result.returncode}", file=sys.stderr)
        return None

    for line in result.stdout.splitlines():
        if line.startswith(">>>"):
            return float(line.removeprefix(">>>").strip())

    print("solver printed no tour length", file=sys.stderr)
    return None


def _solver_source(codebase: Path) -> str:
    solver = codebase / "solver.py"

    if not solver.is_file():
        return ""

    return solver.read_text()


def validate(codebase: Path) -> bool:
    source = _solver_source(codebase)

    if not source:
        print("solver.py is missing", file=sys.stderr)
        return False

    if has_nested_functions(source):
        print(
            "solver.py defines nested functions, which problem.md forbids",
            file=sys.stderr,
        )
        return False

    with tempfile.TemporaryDirectory() as raw_workspace:
        workspace = Path(raw_workspace)
        _prepare(codebase, workspace)

        return (
            _run_once(workspace, VALIDATE_TIMEOUT_SECONDS, VALIDATE_KILL_AFTER_SECONDS)
            is not None
        )


def score(codebase: Path) -> Tuple[Optional[float], Metrics]:
    metrics: Metrics = {"line_count": _solver_source(codebase).count("\n")}

    with tempfile.TemporaryDirectory() as raw_workspace:
        workspace = Path(raw_workspace)
        _prepare(codebase, workspace)

        lengths: List[float] = []

        for attempt in range(SCORE_RUNS):
            print(f"=== run {attempt + 1}/{SCORE_RUNS} ===", file=sys.stderr)
            length = _run_once(
                workspace, SCORE_TIMEOUT_SECONDS, SCORE_KILL_AFTER_SECONDS
            )

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
