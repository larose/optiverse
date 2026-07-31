"""A run directory, read back as an ordered history.

The store records what each solution *is* — score, metrics, tags, parents — but
not *when* it happened. There is no timestamp in `metadata.json` and no
iteration index anywhere; `solutions.csv` is sorted by score. The one thing on
disk that remembers the order is the mtime of `metadata.json`, which the store
writes atomically and last, so it is the instant the iteration finished.

That makes the mtime load-bearing here. A run copied with `cp -r` arrives with
every solution stamped at the moment of the copy, and the history collapses to a
single point in time; `cp -a`, `rsync -a` and `tar` all keep it. `load` refuses
to guess, so a flattened copy shows up as a warning rather than as a chart with
a meaningless axis.

Lineage is a DAG, not a chain: the recombining perturbations hand the agent up
to three parents, so a solution can have three. `depth` is therefore the
shortest way back to the initial solution rather than a generation number.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast

from optiverse.store import METADATA_NAME, FileSystemStore, Solution

# The store enumerates parents from 1 with no documented ceiling. Eight is far
# past the three the shipped strategy can produce and keeps the scan bounded.
MAX_PARENTS = 8

AGENT_LOG_NAME = "agent.log"


@dataclass(frozen=True)
class Record:
    """One solution, plus everything the run directory implies about it."""

    solution: Solution
    committed_at: datetime
    elapsed: float
    order: int
    depth: int
    parents: Tuple[str, ...]
    best_so_far: Optional[float]
    is_new_best: bool

    @property
    def id(self) -> str:
        return self.solution.id

    @property
    def short_id(self) -> str:
        return self.solution.id[:8]

    @property
    def score(self) -> Optional[float]:
        return self.solution.score

    @property
    def failed(self) -> bool:
        return self.solution.score is None

    @property
    def directory(self) -> Path:
        return self.solution.codebase.parent

    @property
    def agent_log(self) -> Optional[Path]:
        path = self.directory / AGENT_LOG_NAME
        return path if path.is_file() else None

    def tag(self, name: str) -> str:
        value = self.solution.tags.get(name)
        return "" if value is None else str(value)


@dataclass(frozen=True)
class History:
    directory: Path
    records: Tuple[Record, ...]
    tag_names: Tuple[str, ...]
    metric_names: Tuple[str, ...]
    # Set when every solution shares a handful of mtimes, which means the copy
    # lost them and `order`/`elapsed` are not to be trusted.
    timestamps_are_suspect: bool

    def by_id(self, solution_id: str) -> Optional[Record]:
        for record in self.records:
            if record.id == solution_id:
                return record
        return None

    @property
    def scored(self) -> Tuple[Record, ...]:
        return tuple(record for record in self.records if not record.failed)

    @property
    def failures(self) -> Tuple[Record, ...]:
        return tuple(record for record in self.records if record.failed)

    @property
    def best(self) -> Record:
        scored = self.scored
        if not scored:
            raise ValueError(f"No scored solution in {self.directory}")
        return min(scored, key=lambda record: _score_of(record))

    @property
    def initial(self) -> Optional[Record]:
        for record in self.records:
            if record.solution.is_initial:
                return record
        return None

    @property
    def duration(self) -> float:
        return self.records[-1].elapsed if self.records else 0.0


def _score_of(record: Record) -> float:
    score = record.solution.score
    if score is None:
        raise ValueError(f"{record.id} has no score")
    return score


def parent_ids(tags: Mapping[str, Union[int, str]]) -> Tuple[str, ...]:
    """The parents a solution was built from, oldest naming scheme included.

    Current runs tag `parent_id_1..3`; one early run in `tmp/` uses
    `parent_1_id`. Both appear in directories people still have on disk, so read
    either. Indexes are scanned rather than walked so a gap cannot truncate the
    list.
    """
    ids: List[str] = []

    for index in range(1, MAX_PARENTS + 1):
        value = tags.get(f"parent_id_{index}")
        if value is None:
            value = tags.get(f"parent_{index}_id")
        if value is not None:
            ids.append(str(value))

    return tuple(ids)


def _depths(
    solutions: Sequence[Solution], parents_by_id: Mapping[str, Tuple[str, ...]]
) -> Dict[str, int]:
    """Shortest distance from each solution back to the initial one.

    Depth is derived rather than stored, and it is a minimum over parents
    because recombination gives a solution several. A solution whose parent is
    missing from the directory — a partial copy, a hand-pruned run — gets -1
    instead of being dropped.
    """
    known: Dict[str, Solution] = {solution.id: solution for solution in solutions}
    depths: Dict[str, int] = {}

    def resolve(solution_id: str, seen: Set[str]) -> int:
        if solution_id in depths:
            return depths[solution_id]

        solution = known.get(solution_id)
        if solution is None or solution_id in seen:
            return -1

        if solution.is_initial:
            depths[solution_id] = 0
            return 0

        candidates = [
            resolve(parent, seen | {solution_id})
            for parent in parents_by_id.get(solution_id, ())
        ]
        reachable = [depth for depth in candidates if depth >= 0]

        depth = min(reachable) + 1 if reachable else -1
        depths[solution_id] = depth
        return depth

    for solution in solutions:
        resolve(solution.id, set())

    return depths


def load(directory: Path) -> History:
    """Read a run directory into records ordered by when they were committed."""
    solutions = FileSystemStore(directory).get_all_solutions()

    if not solutions:
        raise ValueError(f"No complete solution in {directory}")

    parents_by_id = {solution.id: parent_ids(solution.tags) for solution in solutions}
    depths = _depths(solutions, parents_by_id)

    mtimes = {
        solution.id: (directory / solution.id / METADATA_NAME).stat().st_mtime
        for solution in solutions
    }

    # Ties are broken by id so that two runs of this tool over the same
    # directory always produce the same order.
    ordered = sorted(solutions, key=lambda solution: (mtimes[solution.id], solution.id))
    first = mtimes[ordered[0].id]

    records: List[Record] = []
    best_so_far: Optional[float] = None

    for order, solution in enumerate(ordered):
        mtime = mtimes[solution.id]
        is_new_best = solution.score is not None and (
            best_so_far is None or solution.score < best_so_far
        )
        if is_new_best and solution.score is not None:
            best_so_far = solution.score

        records.append(
            Record(
                solution=solution,
                committed_at=datetime.fromtimestamp(mtime),
                elapsed=mtime - first,
                order=order,
                depth=depths.get(solution.id, -1),
                parents=parents_by_id[solution.id],
                best_so_far=best_so_far,
                is_new_best=is_new_best,
            )
        )

    tag_names: Set[str] = set()
    metric_names: Set[str] = set()
    for solution in solutions:
        tag_names.update(solution.tags.keys())
        metric_names.update(solution.metrics.keys())

    distinct = {record.committed_at for record in records}

    return History(
        directory=directory,
        records=tuple(records),
        tag_names=tuple(sorted(tag_names)),
        metric_names=tuple(sorted(metric_names)),
        # One timestamp per solution is what a real run leaves. Far fewer means
        # a copy rewrote them all at once.
        timestamps_are_suspect=len(records) > 2 and len(distinct) < len(records) / 2,
    )


def agent_summary(record: Record, limit: int) -> str:
    """What the agent said it did, from its trajectory.

    `info.submission` is the agent's own closing statement and is the best
    single sentence about a step. Not every trajectory has one — a run that hit
    the timeout has no submission — so fall back to its last message.
    """
    path = record.agent_log
    if path is None:
        return ""

    try:
        parsed: object = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""

    if not isinstance(parsed, dict):
        return ""

    trajectory = cast(Dict[str, object], parsed)

    text = ""
    info = trajectory.get("info")
    if isinstance(info, dict):
        submission = cast(Dict[str, object], info).get("submission")
        if isinstance(submission, str):
            text = submission

    if not text:
        messages = trajectory.get("messages")
        if isinstance(messages, list):
            for message in reversed(cast(List[object], messages)):
                if not isinstance(message, dict):
                    continue
                content = cast(Dict[str, object], message).get("content")
                if isinstance(content, str) and content.strip():
                    text = content
                    break

    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
