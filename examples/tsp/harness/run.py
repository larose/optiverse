"""Runs one candidate solver against one instance, once.

Called only by `evaluate.py`, never by an agent. It transports a tour and
computes nothing: the length is measured by `evaluate.py`, in a process the
candidate has no reach into. So a candidate can decide which tour it submits —
which is its job — but not what that tour is worth.

The candidate lives in its own directory, **appended** to `sys.path` rather than
inserted. Appended, it loses every name collision: `import datetime` finds the
standard library, `import context` finds the copy next to this file, and only
`import solver` resolves in the candidate, because nothing else is named that.
That is what makes it safe to let a candidate be a directory of arbitrary files.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import instance
from context import Context


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one solver on one instance.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--tour", type=Path, required=True)
    arguments = parser.parse_args()

    coordinates = instance.parse(arguments.instance)

    sys.path.append(str(arguments.candidate.resolve()))
    from solver import solve

    end_time = datetime.now(tz=timezone.utc) + timedelta(seconds=arguments.seconds)
    context = Context(instance=coordinates, end_time=end_time)

    solve(context)

    if context.best_solution is None:
        print("the solver reported no tour", file=sys.stderr)
        raise SystemExit(1)

    arguments.tour.write_text(json.dumps(context.best_solution))


if __name__ == "__main__":
    main()
