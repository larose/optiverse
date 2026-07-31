"""The population as a time series.

The run's own `solutions.csv` answers "which solution won"; it is sorted by
score and carries no clock. This writes the other view — the same rows in the
order they were produced, with the running best beside each one — which is what
a question about how the search behaved needs.

Column names follow the store's convention so the two files line up: `t_` for a
tag, `m_` for a metric, alphabetical within each group, `FAILED` for a solution
that could not be scored.
"""

import csv
from pathlib import Path
from typing import List, Optional, Sequence

from .run import History, Record

HOUR = 3600.0

FAILED = "FAILED"

LEADING_COLUMNS = (
    "order",
    "committed_at",
    "elapsed_seconds",
    "id",
    "score",
    "best_so_far",
    "is_new_best",
    "depth",
)


def _number(value: Optional[float]) -> str:
    return "" if value is None else repr(value)


def _row(record: Record, history: History) -> List[str]:
    solution = record.solution

    row = [
        str(record.order),
        record.committed_at.isoformat(sep=" ", timespec="seconds"),
        f"{record.elapsed:.3f}",
        record.id,
        FAILED if record.failed else _number(solution.score),
        _number(record.best_so_far),
        "yes" if record.is_new_best else "",
        str(record.depth),
    ]

    row.extend(record.tag(name) for name in history.tag_names)
    row.extend(_number(solution.metrics.get(name)) for name in history.metric_names)

    return row


def write(history: History, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            list(LEADING_COLUMNS)
            + [f"t_{name}" for name in history.tag_names]
            + [f"m_{name}" for name in history.metric_names]
        )
        for record in history.records:
            writer.writerow(_row(record, history))

    return path


def describe(history: History) -> str:
    """The run in four lines, for a terminal that has no pixels."""
    best = history.best
    initial = history.initial

    if best.score is None:
        raise ValueError("the best solution has no score")

    lines = [
        f"{len(history.records)} solutions over {history.duration / HOUR:.1f}h "
        f"({len(history.failures)} failed to score)",
        f"best  {best.score:,.2f}  {best.short_id}  "
        f"at {best.elapsed / HOUR:.1f}h, group {best.tag('group') or '-'}, "
        f"depth {best.depth}",
    ]

    if initial is not None and initial.solution.score is not None:
        lines.insert(1, f"start {initial.solution.score:,.2f}  {initial.short_id}")

    improvements = [record for record in history.scored if record.is_new_best]
    lines.append(f"{len(improvements)} improvements to the running best")

    return "\n".join(lines)


def improvements_table(history: History) -> str:
    """Every time the running best moved — the descent, as numbers."""
    header = f"{'hours':>7}  {'solution':8}  {'score':>12}  {'gain':>11}  {'move':16}"
    out = [header, "-" * len(header)]

    previous: Optional[float] = None
    for record in history.scored:
        if not record.is_new_best or record.score is None:
            continue
        gain = "" if previous is None else f"{record.score - previous:+,.2f}"
        out.append(
            f"{record.elapsed / HOUR:>7.2f}  {record.short_id:8}  "
            f"{record.score:>12,.2f}  {gain:>11}  {record.tag('move') or 'initial':16}"
        )
        previous = record.score

    return "\n".join(out)


def compare_with_run_csv(history: History) -> Sequence[str]:
    """Check this view against the store's own, and report any drift.

    Both are built from the same `metadata.json` files, so they can only
    disagree if the copy is incomplete or if the run wrote more solutions after
    it was copied. Either is worth knowing before reading the charts, and
    neither is worth failing over.
    """
    path = history.directory / "solutions.csv"
    if not path.is_file():
        return [f"{path.name} is missing from the run directory"]

    with open(path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    in_csv = {row["id"] for row in rows if row.get("id")}
    here = {record.id for record in history.records}

    problems: List[str] = []
    missing = in_csv - here
    extra = here - in_csv

    if missing:
        problems.append(
            f"{len(missing)} solution(s) in solutions.csv have no directory here"
        )
    if extra:
        problems.append(
            f"{len(extra)} solution(s) on disk are absent from solutions.csv"
        )

    return problems
