"""What the director is shown.

The prompt is most of what this project is. It is a separate module from the
mechanics that run the director because the two change for different reasons:
`search.py` changes when the shape of a decision changes, and this changes
whenever a run comes back reading like a search that did not know something it
was told.

Nothing here touches the disk or holds state. `compose` is a function of the run
as it stands — the tree, the sequence, the population, the move already
decided — so the same run always renders the same prompt, and rendering one
costs nothing.
"""

from typing import List, Sequence

from .graph import ROOT_NODE_ID, Graph, Node, stats
from .journal import (
    CONSTRAINT_NAME,
    PROGRAMMER_LOG_NAME,
    WINDOW,
    Journal,
    Progress,
)
from .metrics import render as render_metrics
from .policy import PATIENCE, Move, eligible
from ..solution import Solution

# Ideas shown every iteration because they scored, and ideas shown because it is
# their turn. Eight of each: enough that the rotation sweeps a large tree inside a
# few dozen iterations, few enough that the section stays a glance.
HITS = 8
ROTATION = 8


def compose(
    *,
    graph: Graph,
    iteration: int,
    journal: Journal,
    move: Move,
    problem_description: str,
    solutions: Sequence[Solution],
) -> str:
    """The prompt a perturbation would be given.

    `move` is already decided when this is called. The director is not choosing
    where to work — it is being told, and the prompt's job is to make the place
    it has landed legible enough that the next idea is a good one.
    """
    progress = journal.progress()

    best = progress.best
    anchors = journal.node_ids(WINDOW)

    if best is not None and best.node_id not in anchors:
        anchors = [*anchors, best.node_id]

    if move.base.id not in anchors:
        anchors = [*anchors, move.base.id]

    sections = [
        "# What you are doing",
        "",
        OPENING,
        "",
        "# The problem",
        "",
        problem_description.strip(),
        "",
        "# The search space",
        "",
        SEARCH_SPACE.format(root=ROOT_NODE_ID, patience=PATIENCE),
        "",
        "# The tree",
        "",
        *(
            graph.render(anchors, best_solution_id=best.id if best else None)
            or ["Nothing has been tried yet."]
        ),
        "",
        "# Where the search stands",
        "",
        _state(graph, progress),
        "",
        "# What you have been doing",
        "",
        *_history(journal),
        "",
        "# Ideas already tried",
        "",
        *_ideas_section(graph, iteration),
        "",
        "# What the metrics say",
        "",
        METRICS_OPENING,
        "",
        *render_metrics(solutions),
        "",
        "# Your move",
        "",
        *_move(graph, move),
    ]

    if move.kick is not None:
        sections += ["", "# Your kick", "", KICK_OPENING, "", move.kick]

    sections += [
        "",
        "# Where things are",
        "",
        LAYOUT.format(constraint=CONSTRAINT_NAME, programmer_log=PROGRAMMER_LOG_NAME),
        "",
        "# What to do",
        "",
        TASK.format(constraint=CONSTRAINT_NAME),
    ]

    return "\n".join(sections) + "\n"


# --- the state of the run ---------------------------------------------------


def _state(graph: Graph, progress: Progress) -> str:
    nodes = graph.nodes()
    live = [n for n in nodes if n.solutions and not n.dead]
    empty = [n for n in nodes if not n.solutions]

    lines: List[str] = []
    best = progress.best

    if best is None or best.score is None:
        lines.append("Nothing has scored yet.")
    else:
        if progress.best_iteration is None:
            ago = ", the seed nothing has beaten yet"
        elif progress.drought == 0:
            ago = f", from iteration {progress.best_iteration} — the last one"
        else:
            plural = "" if progress.drought == 1 else "s"
            ago = (
                f", from iteration {progress.best_iteration}, "
                f"{progress.drought} iteration{plural} ago"
            )
        lines.append(
            f"Best so far: {best.score:.6g} — {best.id} at {best.node_id}{ago}."
        )

    unattempted = f", {len(empty)} never attempted" if empty else ""
    lines.append(
        f"Tree: {len(nodes)} node{'' if len(nodes) == 1 else 's'} — "
        f"{len(live)} live{unattempted}."
    )

    span = min(WINDOW, progress.iterations)

    if span:
        plural = "" if progress.window_created == 1 else "s"
        lines.append(
            f"The last {span} iterations worked {progress.window_worked} distinct "
            f"node{'' if progress.window_worked == 1 else 's'} and created "
            f"{progress.window_created} new one{plural}."
        )

    return "\n".join(lines)


# --- the sections -----------------------------------------------------------


def _move(graph: Graph, move: Move) -> List[str]:
    """Where the search has put you, and what it will build from.

    The draw is described as a draw. A director told only *which* node it is
    standing on will reach for a reason it is standing there, and the reason does
    not exist — inventing one is how a random restart quietly turns back into the
    exploitation it was meant to replace.
    """
    base = move.base
    source = move.parent_solution
    score = "did not score" if source.score is None else f"{source.score:.6g}"

    label = "(no constraints)" if base.constraint is None else base.constraint

    lines = [
        f"The search drew {base.id} out of {len(eligible(graph))} nodes with a "
        "score under them. Your constraint becomes a child of it.",
        "",
        f'  {base.id}  "{label}"',
        f"  {stats(base)}",
    ]

    if base.constraints:
        lines += ["", "Constraints already in force there, root first:", ""]
        lines += [
            f"  {position}. {constraint}"
            for position, constraint in enumerate(base.constraints, start=1)
        ]
    else:
        lines += ["", "Nothing is in force there. It is the root: an open field."]

    lines += [
        "",
        f"The programmer opens on {source.id}, which scored {score} — the best "
        "code at that node. Your constraint is what tells it to change that code "
        "rather than admire it.",
        "",
        DRAW_NOTE,
    ]

    return lines


def _history(journal: Journal) -> List[str]:
    epochs = journal.render_epochs()
    ledger = journal.render_ledger()

    if not ledger:
        return ["Nothing has been tried yet."]

    lines: List[str] = []

    if len(epochs) > 1:
        lines += ["The whole run, coarsely:", "", *epochs, ""]

    lines += [f"The last {len(ledger)} iterations — `*` marks a new run best:", ""]
    lines += ledger

    return lines


def _ideas_section(graph: Graph, iteration: int) -> List[str]:
    ideas = _ideas(graph, iteration)

    if not ideas:
        return ["No constraint has been written yet, so there is nothing to sample."]

    return [IDEAS_OPENING, "", *ideas]


def _ideas(graph: Graph, iteration: int) -> List[str]:
    """Every constraint ever written, sampled: the best, and then the rest in turn.

    The ledger reaches back twenty iterations and the tree folds cold branches
    away, so without this an idea from iteration thirty is gone. The best are
    always here because a good idea keeps mattering however old it is; everything
    else is a window that advances by iteration, so the whole tree passes the
    director's desk on a fixed cycle and does so reproducibly.

    It is a sample and it says so. The whole list is one command away, and the
    task asks for it.
    """
    ideas = [n for n in graph.nodes() if n.constraint is not None]

    if not ideas:
        return []

    scored = sorted(
        (n for n in ideas if n.best is not None),
        key=lambda n: n.best if n.best is not None else 0.0,
    )
    hits = scored[:HITS]
    chosen = {node.id for node in hits}

    rest = sorted((n for n in ideas if n.id not in chosen), key=lambda n: n.id)
    turn: List[Node] = []

    if rest:
        start = (iteration * ROTATION) % len(rest)
        turn = [
            rest[(start + offset) % len(rest)]
            for offset in range(min(ROTATION, len(rest)))
        ]

    lines: List[str] = []

    for node, rotated in [(n, False) for n in hits] + [(n, True) for n in turn]:
        mark = "  ↺" if rotated else ""
        lines.append(f'  {node.id}  "{node.constraint}"{mark}')
        lines.append(f"          {_idea_stats(node)}")

    return lines


def _idea_stats(node: Node) -> str:
    attempts = len(node.solutions)
    plural = "" if attempts == 1 else "s"

    if not attempts:
        return "never attempted"

    best = "none scored" if node.best is None else f"best {node.best:.6g}"
    worked = node.last_worked
    when = "" if worked is None else f", last worked at i{worked}"

    return f"{attempts} attempt{plural}, {best}{when}"


OPENING = """You are steering an automated search for a better solution to the
problem below. Each iteration a separate coding agent — the **programmer** —
writes one candidate, and it is scored automatically; lower scores are better.

The search moves on its own. It re-runs one idea until that idea stops improving,
then it picks somewhere at random and starts a new one — and starting a new one
is when you are called. It has already chosen the node your idea hangs under and
the code the programmer will open on. Both are named below, and neither is yours
to change.

What is yours is the idea. One constraint, which is the entire instruction the
programmer gets. You see every score; it sees none, because ranking is your job
and a programmer that could see its own score would abandon a novel approach the
moment it looked worse than the incumbent.

Work like a researcher, not like a chooser. The tree below is every idea this
search has had and what each one scored, and the move worth making is the one
that tests something the tree does not already answer. A constraint that varies a
branch by a word buys a sample you already have.

You are a fresh instance every iteration and nothing you write is read back to
you except the constraint itself, which becomes a node in that tree. Write it for
a stranger, because that is who reads it.

You have a shell and the whole run directory is readable. Read whatever you
need — a candidate's source, a programmer's trajectory, the branch you are about
to write off. Going and looking is the point of you being an agent rather than a
formula, and it is the only way anything above gets checked."""

SEARCH_SPACE = """A **constraint** is prose telling the programmer how to narrow
its approach — an instruction it will read, not a label. A sentence is usual;
paragraphs are fine.

A constraint sits on an **arc**. A **node** is everything accumulated from the
root down to it, so a node is an idea and going deeper is committing to one more
thing. `{root}` is the root and has no constraints at all.

**A constraint is the whole of what you can say.** There is no free-form
instruction to the programmer. If you want it to do something, that is a
constraint, and a constraint is a node — which is what keeps the tree a complete
record of this search rather than half of one.

How the search spends its iterations, so the ledger below reads as what it is:

- It works one node until that node has made {patience} attempts without beating
  its own best. Each attempt is the programmer writing a fresh candidate from the
  same constraints and the same code. Scoring is noisy and one sample settles
  very little, so those repeats are how a node earns a spread instead of a point.
- Then it perturbs. A node is drawn **uniformly at random** from every node with
  a score under it, you are asked for one constraint, and that constraint becomes
  a new child of the drawn node. The draw does not favour the incumbent, because
  a search that always builds on its best is how four hundred iterations go into
  polishing one branch.

The evaluator records **metrics** beside the score — the `m_*` columns in
`solutions.csv`, and in every solution's `metadata.json`. Nothing scores them and
no constraint has to mention them, but they are the only view you have inside the
score. A metric that moves with it is a lever you can constrain toward. A metric
that moves without it tells you what the score does not care about, which is
worth as much."""

DRAW_NOTE = """The draw was random. It is not a judgement that this node is
promising and nothing asks you to defend standing here. The question is the best
next idea *given* that this is where you are standing — which is a different
question from the best idea in the tree, and usually has a different answer."""

KICK_OPENING = """One of these is drawn at random on half of all perturbations,
and this is the one that came up. It is a way in rather than an order, and it
says nothing about this problem — only about how to look at one."""

IDEAS_OPENING = """The eight that scored best, and eight more on rotation, marked
`↺`. The rotation advances every iteration, so every idea in the tree comes round
eventually — and nothing else does, since the ledger reaches back twenty
iterations and the tree folds cold branches into a count.

This is a sample. `python3 -m optiverse.search.tree ../..` is the whole list."""

METRICS_OPENING = """Every metric the evaluator returns, ranked by how strongly
the score follows it across the candidates that scored. Rank correlation, so a
straight line is not required — only that one rises when the other does.

Nothing here says which causes which, and two metrics that both follow code size
will both follow a score that follows code size. It is a place to look, not a
conclusion."""

TASK = """1. **Walk the tree.** Everything above is a summary, and a search that
   only ever reads summaries repeats them. Print the whole thing — every node,
   every constraint in full, every attempt and what it scored:

       python3 -m optiverse.search.tree ../..
       python3 -m optiverse.search.tree ../.. n_1a2b3c

   Then read something raw: the source of a candidate whose score surprised you,
   the trajectory of a programmer whose attempt failed, the constraint on a
   branch you are about to write off.

2. **Write `{constraint}` in your working directory.** One constraint, prose,
   addressed to the programmer. It is the entire instruction it gets, on top of
   the constraints already in force at the node you were given.

3. **Call `validate`.** It says what is wrong, or ends your turn if there is
   nothing wrong."""

LAYOUT = """You are in this iteration's own directory, and you write one file in
it: `{constraint}`. Read anything else in the run.

- `{constraint}` — your constraint. It becomes a node in the tree.
- `python3 -m optiverse.search.tree ../..` — the whole tree, nothing folded.
- `../../arcs.json` — the same tree, as (parent, child, constraint) triplets.
- `../../solutions.csv` — every candidate, best score first.
- `../../solutions/<id>/code/` — a candidate's source.
- `../../solutions/<id>/metadata.json` — its score, metrics and lineage.
- `../<earlier>/{programmer_log}` — what a programmer did and saw.
- `../<earlier>/{constraint}`, `../<earlier>/director.log` — what you did then.
- `../<earlier>_crashed_<n>/` — an attempt that was set aside."""


__all__ = ["HITS", "ROTATION", "compose"]
