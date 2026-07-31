"""`python -m optiverse_analysis <run directory>`.

Writes three files and prints the short version. Reports land in
`<run>/analysis/` by default rather than under the caller's `tmp/`, so the
report travels with the run it describes and the command means the same thing
from any working directory. The store ignores a subdirectory without a
`metadata.json`, so this is invisible to a run that is still going.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import lineage, timeline, tree
from .run import load

REPORT_DIRECTORY_NAME = "analysis"

TIMELINE_NAME = "solutions_timeline.csv"
TREE_NAME = "lineage_tree.png"
LINEAGE_NAME = "lineage.md"


def _parse(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="optiverse_analysis",
        description="Chart the descent of a run and trace how its best solution was reached.",
    )
    parser.add_argument("directory", type=Path, help="a run directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"where to write the report (default: <directory>/{REPORT_DIRECTORY_NAME})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parse(argv)
    directory: Path = arguments.directory

    if not directory.is_dir():
        print(f"No such run directory: {directory}", file=sys.stderr)
        return 2

    history = load(directory)
    output: Path = arguments.output or directory / REPORT_DIRECTORY_NAME

    warnings: List[str] = list(timeline.compare_with_run_csv(history))
    if history.timestamps_are_suspect:
        warnings.append(
            "solutions share too few distinct mtimes — this directory was probably "
            "copied without -a, and the ordering below is not the real one"
        )

    timeline.write(history, output / TIMELINE_NAME)
    tree.write(history, output / TREE_NAME)

    best = history.best
    steps, complete = lineage.trace(history, best)
    (output / LINEAGE_NAME).write_text(lineage.render(history, steps, complete))

    print(timeline.describe(history))
    print()
    print("Improvements to the running best")
    print(timeline.improvements_table(history))
    print()
    print(f"How {best.short_id} was reached")
    print(lineage.summary_table(steps))

    if not complete:
        warnings.append(
            "the ancestry walk never reached the initial solution — a parent is "
            "missing from this directory"
        )

    print()
    for name in (TIMELINE_NAME, TREE_NAME, LINEAGE_NAME):
        print(f"wrote {output / name}")

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
