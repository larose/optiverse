"""Which of the evaluator's numbers the score actually follows.

An evaluator returns metrics beside the score, they land in `metadata.json` and
in the `m_*` columns of `solutions.csv`, and until now nothing put them in front
of the one agent whose job is to notice that the thing being scored moves with
something else. The score is a single number with no inside; these are the only
view into it the search has.

Rank correlation rather than a linear one, because nothing here is expected to be
a straight line — only monotone. Ranking also makes an outlier a rank rather than
a leverage point, and the population always has a few: a candidate that barely
worked scores wildly and would otherwise decide the answer on its own.

It says nothing about cause, and the prompt says so. Two metrics that both follow
code size will both correlate with a score that follows code size, and no
statistic in a population this small can tell which one to constrain toward. It
is a place to look.
"""

import statistics
from typing import Dict, List, Optional, Sequence, Tuple

from ..solution import Solution

# Below this there is no correlation worth printing — three points can be
# perfectly monotone by chance, and a director shown ±1.0 will believe it.
MINIMUM_SAMPLES = 5

# Under this, the ordering is noise dressed as a finding. Printed as "says
# nothing" rather than dropped, because knowing the score ignores a metric is
# worth as much as knowing it follows one.
WEAK = 0.2


def correlations(solutions: Sequence[Solution]) -> List[Tuple[str, float]]:
    """Every metric, by how strongly the score follows it. Strongest first.

    Metrics with too few samples or no variation are dropped rather than shown at
    zero: a metric the same on every candidate has no correlation, not a weak
    one.
    """
    scored = [s for s in solutions if s.score is not None]

    if len(scored) < MINIMUM_SAMPLES:
        return []

    scores = [float(s.score) for s in scored if s.score is not None]
    series: Dict[str, List[Optional[float]]] = {}

    for name in {name for solution in scored for name in solution.metrics}:
        series[name] = [
            float(s.metrics[name]) if name in s.metrics else None for s in scored
        ]

    found: List[Tuple[str, float]] = []

    for name, values in series.items():
        paired = [(v, scores[i]) for i, v in enumerate(values) if v is not None]

        if len(paired) < MINIMUM_SAMPLES:
            continue

        left = [v for v, _ in paired]
        right = [s for _, s in paired]

        if len(set(left)) < 2 or len(set(right)) < 2:
            continue

        found.append((name, statistics.correlation(_ranks(left), _ranks(right))))

    return sorted(found, key=lambda pair: -abs(pair[1]))


def render(solutions: Sequence[Solution]) -> List[str]:
    found = correlations(solutions)

    if not found:
        return [
            f"Fewer than {MINIMUM_SAMPLES} candidates have scored, or the "
            "evaluator records no metrics. There is nothing to correlate yet."
        ]

    width = max(len(name) for name, _ in found)

    return [
        f"  {name:<{width}}  {rho:+.2f}"
        + ("   says nothing" if abs(rho) < WEAK else "")
        for name, rho in found
    ]


def _ranks(values: Sequence[float]) -> List[float]:
    """Ranks, ties averaged.

    Averaging matters more here than it usually does: metrics like a file count
    are the same across most of a population, and giving those an arbitrary order
    would invent a correlation out of the sort.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks: List[float] = [0.0] * len(values)

    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1

        shared = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = shared

        start = end + 1

    return ranks
