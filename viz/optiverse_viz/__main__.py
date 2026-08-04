"""`python -m optiverse_viz <run directory> [-o PATH]`.

Writes the page and prints where it went, so the path can be pasted into a
browser or piped into one. It writes into the run by default rather than into
the caller's working directory, so the command means the same thing from
anywhere and the report stays with the run it describes.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .render import render, write


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="optiverse_viz", description=__doc__)
    parser.add_argument("directory", type=Path, help="a run directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write here instead of <run>/viz/index.html; - for stdout",
    )

    arguments = parser.parse_args(argv)
    directory = Path(arguments.directory)

    if not directory.is_dir():
        print(f"{directory} is not a directory", file=sys.stderr)
        return 2

    if str(arguments.output) == "-":
        sys.stdout.write(render(directory))
        return 0

    print(write(directory, arguments.output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
