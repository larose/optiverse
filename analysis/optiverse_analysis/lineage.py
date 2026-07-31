"""How one solution came to exist.

Every solution names the parents it was built from, and every parent's codebase
is still on disk, so the edit that produced a solution can be recovered exactly:
diff the two directories. Walking that back to the initial solution turns the
best result of a run into a readable sequence of changes.

Two things the report has to be honest about, because both make the sequence
look stranger than a plain descent:

Recombination gives a solution up to three parents. The chain here follows the
first — the one the strategy calls "Parent" — and names the others as merge-ins
at the step where they appear, rather than pretending the history is linear.

A `perturb_restart` edge is not an edit at all. The strategy hands the agent the
initial solution and tells it to ignore the parent and start over, so the diff
is a rewrite. Those steps are labelled, and the score can move backwards across
one: `_improve` parents on the best solution *in the current group*, so a new
group starts over from wherever its perturbation landed.
"""

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from optiverse.codebase import relative_files

from .run import History, Record, agent_summary

RESTART_MOVE = "perturb_restart"

# Long enough for a real edit, short enough that a restart's whole-file rewrite
# does not bury the steps after it.
MAX_DIFF_LINES = 120

SUMMARY_LIMIT = 300


@dataclass(frozen=True)
class Step:
    """One edge of the ancestry: a parent, a child, and what changed."""

    record: Record
    parent: Optional[Record]
    merge_ins: Tuple[Record, ...]

    @property
    def is_restart(self) -> bool:
        return self.record.tag("move") == RESTART_MOVE

    @property
    def delta(self) -> Optional[float]:
        if self.parent is None:
            return None
        score, parent_score = self.record.score, self.parent.score
        if score is None or parent_score is None:
            return None
        return score - parent_score


def trace(history: History, target: Record) -> Tuple[List[Step], bool]:
    """The ancestry of `target`, oldest first.

    Returns the steps and whether the walk reached the initial solution. It can
    fail to: a parent may be missing from a partial copy, and a run interrupted
    before its initial solution was committed has no root at all.
    """
    chain: List[Record] = []
    seen: set[str] = set()
    complete = False

    current: Optional[Record] = target
    while current is not None and current.id not in seen:
        seen.add(current.id)
        chain.append(current)

        if current.solution.is_initial:
            complete = True
            break

        current = history.by_id(current.parents[0]) if current.parents else None

    chain.reverse()

    steps: List[Step] = []
    for index, record in enumerate(chain):
        parent = chain[index - 1] if index > 0 else None
        merge_ins = tuple(
            found
            for parent_id in record.parents[1:]
            for found in [history.by_id(parent_id)]
            if found is not None
        )
        steps.append(Step(record=record, parent=parent, merge_ins=merge_ins))

    return steps, complete


def diff(parent: Path, child: Path, max_lines: int = MAX_DIFF_LINES) -> str:
    """A unified diff between two codebases, over the union of their files."""
    names = sorted(set(relative_files(parent)) | set(relative_files(child)))

    lines: List[str] = []
    for name in names:
        before = _read(parent / name)
        after = _read(child / name)

        if before == after:
            continue

        lines.extend(
            line.rstrip("\n")
            for line in difflib.unified_diff(
                before, after, fromfile=f"a/{name}", tofile=f"b/{name}", n=3
            )
        )

    if not lines:
        return "(no change to any file)"

    if len(lines) > max_lines:
        hidden = len(lines) - max_lines
        return "\n".join(lines[:max_lines] + [f"... {hidden} more diff lines"])

    return "\n".join(lines)


def _read(path: Path) -> List[str]:
    if not path.is_file():
        return []
    try:
        return path.read_text().splitlines(keepends=True)
    except UnicodeDecodeError:
        return ["(binary file)\n"]


def _score(record: Record) -> str:
    return "FAILED" if record.score is None else f"{record.score:,.2f}"


def _delta(step: Step) -> str:
    delta = step.delta
    if delta is None or step.parent is None or step.parent.score is None:
        return ""
    percent = 100.0 * delta / step.parent.score
    return f"{delta:+,.2f} ({percent:+.2f}%)"


def summary_table(steps: Sequence[Step]) -> str:
    """A fixed-width version of the chain, for the terminal."""
    header = f"{'#':>3}  {'solution':8}  {'move':16}  {'grp':>3}  {'score':>12}  {'change vs parent':>22}  {'lines':>5}"
    rows = [header, "-" * len(header)]

    for index, step in enumerate(steps):
        record = step.record
        rows.append(
            f"{index:>3}  {record.short_id:8}  {record.tag('move') or 'initial':16}  "
            f"{record.tag('group') or '-':>3}  {_score(record):>12}  {_delta(step):>22}  "
            f"{_metric(record, 'line_count'):>5}"
        )

    return "\n".join(rows)


def _metric(record: Record, name: str) -> str:
    value = record.solution.metrics.get(name)
    return "" if value is None else f"{value:,.0f}"


def render(history: History, steps: Sequence[Step], complete: bool) -> str:
    """The full ancestry as Markdown, one section per step."""
    if not steps:
        return "# Ancestry\n\nNothing to trace.\n"

    target = steps[-1].record
    lines: List[str] = [
        f"# How `{target.short_id}` was reached",
        "",
        f"Run `{history.directory.name}` — {len(history.records)} solutions, "
        f"{len(history.scored)} scored, {len(history.failures)} failed.",
        "",
        f"The best solution scores **{_score(target)}** and sits {len(steps) - 1} "
        f"step(s) from the start of its chain.",
        "",
    ]

    if not complete:
        lines.extend(
            [
                "> The walk did not reach the initial solution — a parent is missing "
                "from this directory, so the chain below is partial.",
                "",
            ]
        )

    first = steps[0].record
    if first.score is not None and target.score is not None and first.score > 0:
        improvement = 100.0 * (first.score - target.score) / first.score
        lines.extend(
            [
                f"From `{first.short_id}` at {_score(first)} to `{target.short_id}` "
                f"at {_score(target)}: **{improvement:.1f}% shorter**.",
                "",
            ]
        )

    lines.extend(["```", summary_table(steps), "```", ""])

    restarts = [index for index, step in enumerate(steps) if step.is_restart]
    if restarts:
        lines.extend(
            [
                f"Steps {', '.join(str(index) for index in restarts)} are "
                "`perturb_restart`: the agent was handed the initial solution and "
                "told to ignore it, so those diffs are rewrites rather than edits.",
                "",
            ]
        )

    for index, step in enumerate(steps):
        lines.extend(_section(index, step, history))

    return "\n".join(lines) + "\n"


def _restart_context(history: History, record: Record) -> str:
    """Why the strategy restarted inside a group it had already opened.

    A group normally opens with exactly one perturbation. A second one means
    `_improve` found no scored solution to build on — the group's opening
    attempt failed — and fell back to starting over. That is worth saying,
    because it is how a dead end turns into the branch that wins.
    """
    group = record.tag("group")
    if not group:
        return ""

    earlier = [
        other
        for other in history.records
        if other.tag("group") == group and other.order < record.order
    ]

    if not earlier or not all(other.failed for other in earlier):
        return ""

    named = ", ".join(
        f"`{other.short_id}` ({other.tag('move')}, "
        f"{other.tag('exit_status') or 'no score'})"
        for other in earlier
    )

    return (
        f"Group {group} had no scored solution to improve on when this ran — its "
        f"opening attempt {named} — so the strategy fell back to starting over "
        "from the initial solution. This branch exists because that attempt failed."
    )


def _section(index: int, step: Step, history: History) -> List[str]:
    record = step.record
    move = record.tag("move") or "initial solution"

    lines = [f"## Step {index} — {move} — `{record.short_id}`", ""]

    facts = [
        f"- Score: **{_score(record)}**",
        f"- Group: {record.tag('group') or '-'}",
        f"- Committed: {record.committed_at.isoformat(sep=' ', timespec='seconds')} "
        f"({record.elapsed / 3600:.2f}h into the run)",
    ]

    if step.parent is not None:
        change = _delta(step)
        facts.insert(1, f"- Change vs `{step.parent.short_id}`: {change or 'n/a'}")

    exit_status = record.tag("exit_status")
    if exit_status:
        facts.append(f"- Exit status: {exit_status}")

    for name in ("line_count", "agent_model_calls", "agent_validate_runs"):
        value = _metric(record, name)
        if value:
            facts.append(f"- {name.replace('_', ' ').capitalize()}: {value}")

    if step.merge_ins:
        merged = ", ".join(
            f"`{other.short_id}` ({_score(other)})" for other in step.merge_ins
        )
        facts.append(f"- Also given: {merged}")

    lines.extend(facts)
    lines.append("")

    said = agent_summary(record, SUMMARY_LIMIT)
    if said:
        lines.extend([f"> {said}", ""])

    if step.parent is not None:
        if step.is_restart:
            lines.append(
                "Rewrite from the initial solution — the diff below is against the "
                "parent the strategy passed, which the agent was told to ignore."
            )
            context = _restart_context(history, record)
            if context:
                lines.append("")
                lines.append(context)
            lines.append("")

        lines.extend(
            [
                "```diff",
                diff(step.parent.solution.codebase, record.solution.codebase),
                "```",
                "",
            ]
        )

    return lines
