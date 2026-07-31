"""The genealogy: every solution under the one it was built from.

The initial solution is the root and each child hangs below its parent, so the
shape of the search is the shape of the picture — where it went deep, where it
fanned out and got nowhere, and which thread reached the best result.

Depth is the y axis. Horizontal position carries no meaning beyond keeping
siblings apart and in the order they were produced, so the x axis is not drawn;
reading a value off it would be reading noise.

Colour is the score, which is magnitude, so it is one hue light→dark with a
scale legend — darkest is best, because on this problem lower is better. The
bands are quantiles of the population rather than equal slices of the range: a
run spans 34,232 down to 2,588 with almost everything in the last few hundred,
and equal slices would paint all of it one colour. Every band edge is a real
score printed on the colourbar, so the non-linearity is stated rather than
hidden. The five steps pass the ordinal checks on the light surface.

Structure the tree cannot show, it says out loud instead: the two recombination
solutions have parents besides the one they hang from, drawn as a separate
lighter edge, and solutions that failed to score have no score to colour, so
they take the status colour and their own marker shape.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .run import History, Record  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
EDGE = "#d8d7d0"
WINNER = "#2a78d6"
FAILED = "#d03b3b"

# One hue, light→dark, five steps: passes the ordinal checks on this surface
# (monotone lightness, every adjacent gap >= 0.06, light end clear of the
# surface at 2.06:1).
SCORE_BANDS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def _score(record: Record) -> float:
    score = record.solution.score
    if score is None:
        raise ValueError(f"{record.id} has no score")
    return score


def _band_edges(scores: Sequence[float]) -> List[float]:
    """Quantile cuts, one per colour step, strictly increasing.

    Ties collapse — a run where most solutions share a score has fewer usable
    cuts than steps — so the result is deduplicated and the caller uses however
    many bands survive.
    """
    ordered = sorted(scores)
    count = len(SCORE_BANDS)

    edges = [ordered[0]]
    for index in range(1, count):
        edges.append(ordered[min(len(ordered) - 1, index * len(ordered) // count)])
    edges.append(ordered[-1])

    unique: List[float] = []
    for edge in edges:
        if not unique or edge > unique[-1]:
            unique.append(edge)

    if len(unique) < 2:
        unique = [ordered[0], ordered[0] + 1.0]

    return unique


def _tree(history: History) -> Tuple[Dict[str, List[str]], List[str]]:
    """Children by primary parent, plus the roots, in commit order.

    The primary parent — the one the strategy titles "Parent" — is what the tree
    hangs on. A solution whose parent is not in this directory becomes a root of
    its own rather than disappearing.
    """
    children: Dict[str, List[str]] = {}
    roots: List[str] = []

    for record in history.records:
        parent = record.parents[0] if record.parents else None
        if parent is not None and history.by_id(parent) is not None:
            children.setdefault(parent, []).append(record.id)
        else:
            roots.append(record.id)

    return children, roots


def _layout(
    children: Dict[str, List[str]], roots: Sequence[str]
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Leaves get consecutive slots; a parent sits over the middle of its own.

    Iterative rather than recursive: a long run is a deep tree, and the depth
    here is bounded by the number of solutions rather than by anything smaller.
    """
    x: Dict[str, float] = {}
    depth: Dict[str, int] = {}
    next_slot = 0.0

    for root in roots:
        stack: List[Tuple[str, int, bool]] = [(root, 0, False)]
        while stack:
            node, level, expanded = stack.pop()
            depth[node] = level

            kids = children.get(node, [])
            if not kids:
                x[node] = next_slot
                next_slot += 1.0
                continue

            if expanded:
                x[node] = (x[kids[0]] + x[kids[-1]]) / 2.0
                continue

            stack.append((node, level, True))
            for kid in reversed(kids):
                stack.append((kid, level + 1, False))

    return x, depth


def _winning_path(history: History, best: Record) -> List[str]:
    path: List[str] = []
    seen: set[str] = set()
    current: Optional[Record] = best

    while current is not None and current.id not in seen:
        seen.add(current.id)
        path.append(current.id)
        parent = current.parents[0] if current.parents else None
        current = history.by_id(parent) if parent else None

    return path


def _elbow(
    axes: Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str,
    width: float,
    zorder: int,
) -> None:
    """A right-angled connector, which reads as descent far better than a slant."""
    middle = (start[1] + end[1]) / 2.0
    axes.plot(
        [start[0], start[0], end[0], end[0]],
        [start[1], middle, middle, end[1]],
        color=color,
        linewidth=width,
        solid_joinstyle="miter",
        zorder=zorder,
    )


def write(history: History, path: Path) -> Path:
    scored = history.scored
    if not scored:
        raise ValueError(f"No scored solution to plot in {history.directory}")

    children, roots = _tree(history)
    x, depth = _layout(children, roots)
    best = history.best
    winners = set(_winning_path(history, best))

    edges = _band_edges([_score(record) for record in scored])
    # Reversed: lower is better here, so the darkest step has to land on the
    # best score rather than on the largest number. Light→dark still runs one
    # way across the ramp; it is the data that is inverted, not the palette.
    colormap = ListedColormap(list(reversed(SCORE_BANDS[: len(edges) - 1])))
    norm = BoundaryNorm(edges, colormap.N)

    width = max(11.0, min(30.0, 0.20 * (max(x.values()) + 1)))
    height = max(6.0, 0.62 * (max(depth.values()) + 2))

    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=(width, height), dpi=150)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    for parent_id, kids in children.items():
        for kid in kids:
            on_path = parent_id in winners and kid in winners
            _elbow(
                axes,
                (x[parent_id], depth[parent_id]),
                (x[kid], depth[kid]),
                WINNER if on_path else EDGE,
                1.8 if on_path else 0.8,
                3 if on_path else 1,
            )

    # The recombining moves are handed more than one parent. Only the first
    # holds the tree together; the rest would be invisible without this.
    extra_drawn = 0
    for record in history.records:
        for other_id in record.parents[1:]:
            if other_id not in x:
                continue
            axes.annotate(
                "",
                xy=(x[record.id], depth[record.id]),
                xytext=(x[other_id], depth[other_id]),
                arrowprops={
                    "arrowstyle": "-",
                    "color": MUTED,
                    "linewidth": 0.8,
                    "alpha": 0.7,
                    "connectionstyle": "arc3,rad=0.18",
                },
                zorder=2,
            )
            extra_drawn += 1

    axes.scatter(
        [x[record.id] for record in scored],
        [depth[record.id] for record in scored],
        c=[_score(record) for record in scored],
        cmap=colormap,
        norm=norm,
        s=90,
        linewidths=0.8,
        edgecolors=SURFACE,
        zorder=4,
    )

    failures = history.failures
    if failures:
        axes.scatter(
            [x[record.id] for record in failures],
            [depth[record.id] for record in failures],
            s=70,
            marker="X",
            color=FAILED,
            linewidths=0.8,
            edgecolors=SURFACE,
            zorder=5,
            label=f"Failed to score ({len(failures)})",
        )

    axes.scatter(
        [x[best.id]],
        [depth[best.id]],
        s=320,
        marker="*",
        color=WINNER,
        linewidths=1.2,
        edgecolors=SURFACE,
        zorder=6,
        label=f"Best — {_score(best):,.1f}",
    )
    axes.annotate(
        f"{best.short_id}\n{_score(best):,.1f}",
        xy=(x[best.id], depth[best.id]),
        xytext=(10, -4),
        textcoords="offset points",
        fontsize=9,
        color=TEXT_SECONDARY,
        linespacing=1.35,
        zorder=6,
    )

    initial = history.initial
    if initial is not None and initial.id in x:
        axes.annotate(
            (
                f"initial · {initial.solution.score:,.0f}"
                if initial.solution.score is not None
                else "initial"
            ),
            xy=(x[initial.id], depth[initial.id]),
            xytext=(0, 14),
            textcoords="offset points",
            fontsize=9,
            color=TEXT_SECONDARY,
            ha="center",
            zorder=6,
        )

    _style(axes, max(depth.values()))

    axes.set_title(
        f"{history.directory.name} — the family tree of the search",
        color=TEXT_PRIMARY,
        fontsize=13,
        loc="left",
        pad=30,
    )
    axes.annotate(
        f"{len(history.records)} solutions · {len(history.failures)} failed · "
        f"best {_score(best):,.1f} at depth {depth[best.id]} · "
        f"the highlighted thread is its ancestry",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=9.5,
        color=TEXT_SECONDARY,
    )

    handles, labels = axes.get_legend_handles_labels()
    if extra_drawn:
        line = Line2D(
            [], [], color=MUTED, linewidth=0.8, label="Extra parent (recombination)"
        )
        handles.append(line)
        labels.append(str(line.get_label()))
    axes.legend(
        handles,
        labels,
        loc="lower right",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
    )

    bar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=colormap),
        ax=axes,
        boundaries=edges,
        ticks=edges,
        spacing="uniform",
        pad=0.012,
        fraction=0.022,
    )
    bar.set_label("Tour length — darker is better", color=TEXT_SECONDARY, fontsize=9.5)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.ax.set_yticklabels([f"{edge:,.0f}" for edge in edges])
    bar.outline.set_visible(False)

    if history.timestamps_are_suspect:
        figure.text(
            0.5,
            0.005,
            "Timestamps look flattened by a copy — sibling order is not the real one.",
            ha="center",
            fontsize=9,
            color=FAILED,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)

    return path


def _style(axes: Axes, deepest: int) -> None:
    axes.set_ylim(deepest + 0.6, -0.9)
    axes.set_ylabel(
        "Generations from the initial solution", color=TEXT_SECONDARY, fontsize=10
    )
    axes.set_yticks(range(deepest + 1))
    axes.grid(True, axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)

    axes.get_xaxis().set_visible(False)
    for side in ("top", "right", "bottom"):
        axes.spines[side].set_visible(False)
    axes.spines["left"].set_color(BASELINE)
    axes.spines["left"].set_linewidth(0.8)
    axes.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
