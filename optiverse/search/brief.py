"""What the director is shown.

The prompt is most of what this project is. It is a separate module from the
mechanics that run the director because the two change for different reasons:
`search.py` changes when the shape of a plan changes, and this changes whenever a
run comes back reading like a search that did not know something it was told.

Nothing here touches the disk or holds state. `compose` is a function of the run
as it stands — the tree, the sequence, the population, the notebook — so the same
run always renders the same prompt, and rendering one costs nothing.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from .graph import ROOT_NODE_ID, Graph, Node, current_node
from .journal import (
    PROGRAMMER_LOG_NAME,
    MEMORY_NAME,
    PLAN_NAME,
    WINDOW,
    Journal,
    Progress,
)
from .metrics import render as render_metrics
from ..solution import Solution

# About eighty lines. Stated to the director rather than enforced: the ceiling is
# there to make it rewrite the file instead of appending to it, and a rejection
# it cannot satisfy without deleting something it believes is a worse trade than
# a notebook that ran long.
MEMORY_LIMIT_CHARACTERS = 6000

# Iterations between reviews when nothing else has called one. Ten is short
# enough that a wrong model cannot steer more than ten attempts, long enough that
# most iterations are still spent deciding rather than writing.
REVIEW_INTERVAL = 10

# Iterations without a new run best before the prompt says so, and before it
# stops taking the director's word that the current line is working.
DRIFT_DROUGHT = 5
STUCK_DROUGHT = 10

# What share of a window has to land in one subtree before that is worth saying
# out loud. Two thirds: half is where a search that is exploiting normally sits.
CONCENTRATION_SHARE = 2 / 3

# Ideas shown every iteration because they scored, and ideas shown because it is
# their turn. Eight of each: enough that the rotation sweeps a large tree inside a
# few dozen iterations, few enough that the section stays a glance.
HITS = 8
ROTATION = 8


class Stage(Enum):
    """How much the prompt leans on the director to go somewhere else."""

    WORKING = "working"
    DRIFTING = "drifting"
    STUCK = "stuck"
    RESET = "reset"


@dataclass(frozen=True)
class Turn:
    """What one iteration's prompt was built from.

    `validate` needs `reviewing` and `reconciling` and cannot be given them: it
    reaches the director as a callable of no arguments, called whenever it likes.
    So it is held on the instance, keyed by iteration — a stale one is ignored
    rather than believed.
    """

    iteration: int
    stage: Stage
    reviewing: Optional[str]
    """Why this iteration is a review, or None if it is an ordinary one."""

    reconciling: bool
    """Whether there is a previous prediction to answer for."""


def compose(
    *,
    graph: Graph,
    iteration: int,
    journal: Journal,
    memory: Optional[str],
    playbook: Sequence[str],
    problem_description: str,
    solutions: Sequence[Solution],
) -> Tuple[str, Turn]:
    """The prompt an iteration would be given, and what it was built from.

    `memory` is optional rather than empty-when-absent because a notebook that
    exists and says nothing is a different thing from one that was never written:
    the first is a director that had nothing to say, the second is a run that has
    not started thinking yet, and only the second calls a review.
    """
    progress = journal.progress()
    stage = _stage(progress, graph.node(current_node(graph, solutions)))

    turn = Turn(
        iteration=iteration,
        stage=stage,
        reviewing=_reviewing(stage, journal, iteration, memory),
        reconciling=journal.last() is not None,
    )

    best = progress.best
    worked = [move.node_id for move in journal.moves(WINDOW)]
    anchors = journal.node_ids(WINDOW)

    if best is not None and best.node_id not in anchors:
        anchors = [*anchors, best.node_id]

    sections = [
        "# What you are doing",
        "",
        OPENING.format(memory=MEMORY_NAME, plan=PLAN_NAME),
        "",
        "# The problem",
        "",
        problem_description.strip(),
        "",
        "# The search space",
        "",
        SEARCH_SPACE.format(root=ROOT_NODE_ID),
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
        _state(graph, progress, _concentration(graph, worked)),
        "",
        _rung(stage, progress),
        "",
        "# What you have been doing",
        "",
        *_history(journal),
        "",
        "# Your last prediction",
        "",
        *journal.render_prediction(),
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
        "# What you believe",
        "",
        (memory or "").strip()
        or f"`{MEMORY_NAME}` is empty. This iteration starts it.",
    ]

    if stage in (Stage.STUCK, Stage.RESET) and playbook:
        sections += [
            "",
            "# Angles for finding a new line",
            "",
            PLAYBOOK_OPENING,
            "",
            "\n\n".join(playbook),
        ]

    sections += [
        "",
        "# Where things are",
        "",
        LAYOUT.format(
            programmer_log=PROGRAMMER_LOG_NAME, memory=MEMORY_NAME, plan=PLAN_NAME
        ),
        "",
        "# What to do",
        "",
        *_task(turn.reviewing),
    ]

    return "\n".join(sections) + "\n", turn


# --- the state of the run ---------------------------------------------------


def _stage(progress: Progress, latest: Node) -> Stage:
    """How hard the prompt should push, from the run's numbers rather than a node's.

    `RESET` keys on a window that created nothing rather than on the drought
    alone. Late in a long run against a noisy evaluator a drought of fifty is
    ordinary — the incumbent is genuinely hard to beat — so a rung on the drought
    would be permanently on from the middle of the run, would demand a new node
    every iteration, and would fill the tree with ideas nobody sampled twice. A
    window that created nothing while losing is the actual pathology, and it
    clears itself the moment one node is made.
    """
    if progress.window_created == 0 and progress.drought >= WINDOW:
        return Stage.RESET

    if progress.drought >= STUCK_DROUGHT or latest.dead:
        return Stage.STUCK

    if progress.drought >= DRIFT_DROUGHT:
        return Stage.DRIFTING

    return Stage.WORKING


def _reviewing(
    stage: Stage, journal: Journal, iteration: int, memory: Optional[str]
) -> Optional[str]:
    """Why this iteration is a review, or None if it is an ordinary one.

    A review is the same director with two more things it has to do: look at
    something the prompt did not show it, and write down what it now believes.
    Both are cheap and neither is worth doing every iteration — what makes them
    worth doing is that they happen on a schedule the director does not set, and
    on the occasions it would least choose to.
    """
    if memory is None:
        return "there is no notebook yet"

    if stage is Stage.RESET:
        return "the search has spent a whole window re-running ideas it already had"

    if stage is Stage.STUCK:
        return "nothing has improved for a long time"

    last = journal.last()

    if last is not None and last.score is None:
        return "the last attempt did not score"

    if last is not None and last.crashes:
        return "the last iteration took more than one attempt"

    if iteration % REVIEW_INTERVAL == 0:
        return f"every {REVIEW_INTERVAL}th iteration is one"

    return None


def _concentration(graph: Graph, worked: Sequence[str]) -> Optional[Tuple[str, int]]:
    """The deepest subtree most of the recent work landed in, and how much of it.

    Counted over ancestors rather than over the nodes themselves, because a
    director minting a fresh child of one exhausted node every iteration touches
    a different node each time and looks perfectly diverse. What it is actually
    doing is visible only one level up.
    """
    if not worked:
        return None

    counts: Dict[str, int] = {}

    for node_id in worked:
        for ancestor in graph.ancestors(node_id):
            counts[ancestor] = counts.get(ancestor, 0) + 1

    threshold = math.ceil(len(worked) * CONCENTRATION_SHARE)
    found = [
        (node_id, n)
        for node_id, n in counts.items()
        # The root is an ancestor of everything, so it always qualifies and never
        # means anything: "the search stayed inside the search" is not a rut.
        if n >= threshold and node_id != ROOT_NODE_ID
    ]

    if not found:
        return None

    return max(found, key=lambda pair: (len(graph.ancestors(pair[0])), pair[1]))


def _state(
    graph: Graph, progress: Progress, concentration: Optional[Tuple[str, int]]
) -> str:
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

    if concentration is not None and span:
        node_id, count = concentration
        improved = "" if best is None or best.score is None else f" {best.score:.6g}"
        lines.append(
            f"{count} of those {span} landed on {node_id} or below it, and none of "
            f"them improved on{improved or ' anything'}."
        )

    return "\n".join(lines)


def _rung(stage: Stage, progress: Progress) -> str:
    drought = progress.drought

    if stage is Stage.WORKING:
        return (
            "The search is still finding things. Nothing here is asking you to "
            "change what you are doing."
        )

    if stage is Stage.DRIFTING:
        return (
            f"Nothing has beaten the incumbent for {drought} iterations. That is "
            "short enough to be noise. It is long enough that another attempt at "
            "the same thing needs a reason, so put one in `reasoning`: what makes "
            "this move different from the ones that did not work?"
        )

    if stage is Stage.STUCK:
        return (
            f"Nothing has beaten the incumbent for {drought} iterations, or the "
            "node the last attempt landed on has run out. Take something off "
            "`## Ideas worth returning to`, or one of the angles below. A plan "
            "with no constraint is allowed, but `reasoning` has to say what one "
            "more sample settles that the ones already there do not."
        )

    return (
        f"Nothing has beaten the incumbent for {drought} iterations and the last "
        f"{WINDOW} created no node at all: the search has spent that window "
        "re-running ideas it already had. Nothing here refuses a plan — but a "
        "window of evidence says the tree you have does not contain the answer, "
        "and the move that follows from that is a constraint nothing in the tree "
        "already says."
    )


# --- the sections -----------------------------------------------------------


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

    detail = journal.render_detail()

    if detail:
        lines += ["", "The moves before your last, in full:", "", *detail]

    return lines


def _ideas_section(graph: Graph, iteration: int) -> List[str]:
    ideas = _ideas(graph, iteration)

    if not ideas:
        return ["No constraint has been written yet, so there is nothing to sample."]

    return [IDEAS_OPENING.format(memory=MEMORY_NAME), "", *ideas]


def _ideas(graph: Graph, iteration: int) -> List[str]:
    """Every constraint ever written, sampled: the best, and then the rest in turn.

    The ledger reaches back twenty iterations and the tree folds cold branches
    away, so without this an idea from iteration thirty is gone. The best are
    always here because a good idea keeps mattering however old it is; everything
    else is a window that advances by iteration, so the whole tree passes the
    director's desk on a fixed cycle and does so reproducibly — which is exactly
    what drawing at random could not do.

    What survives the rotation is not this section's job. It is `memory.md`'s,
    and the task says so.
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


def _task(reviewing: Optional[str]) -> List[str]:
    lines = [TASK_ORDER]

    if reviewing is not None:
        lines += ["", REVIEW_OPENING.format(reason=reviewing, memory=MEMORY_NAME)]

    lines += [
        "",
        (
            TASK_STEPS.format(memory=MEMORY_NAME, limit=MEMORY_LIMIT_CHARACTERS)
            if reviewing is not None
            else TASK_STEPS_ORDINARY.format(memory=MEMORY_NAME)
        ),
        "",
        PLAN_INSTRUCTIONS.format(plan=PLAN_NAME),
    ]

    return lines


OPENING = """You are steering an automated search for a better solution to the
problem below. Each iteration a separate coding agent — the **programmer** —
writes one candidate, and it is scored automatically; lower scores are better.

You decide what the programmer works on. You see every score; it sees none,
because ranking is your job and a programmer that could see its own score would
abandon a novel approach the moment it looked worse than the incumbent.

Work like a researcher, not like a chooser. A chooser reads the tree and takes
the best-looking branch, which is how a search spends four hundred iterations
polishing one idea and calls it progress. A researcher holds a model of what this
problem rewards, makes the move that would test it, writes down what it expects
to see, and changes the model when the result disagrees. Scoring is noisy, so one
result rarely settles anything by itself — which is exactly why the prediction
has to be written before the number arrives, and why you are shown the ones you
wrote earlier with the numbers beside them.

You are a fresh instance every iteration and this prompt is all you get. Two
files carry anything forward. `{memory}` is what you believe; `{plan}` is what
you decided, why, and what you expected. Both are read back to whoever runs next,
and that is you. Write them for a stranger, because that is who reads them.

You have a shell and the whole run directory is readable. Read whatever you
need — a candidate's source, a programmer's trajectory, an earlier plan. Going
and looking is the point of you being an agent rather than a formula."""

SEARCH_SPACE = """A **constraint** is prose telling the programmer how to narrow
its approach — an instruction it will read, not a label. A sentence is usual;
paragraphs are fine.

A constraint sits on an **arc**. A **node** is everything accumulated from the
root down to it, so a node is an idea and going deeper is committing to one more
thing. `{root}` is the root and has no constraints at all.

You act by naming a node to work under, naming a solution whose code the
programmer starts from, and optionally adding one constraint — which creates a
child of that node and works there instead.

**That is the whole of what you can say.** There is no free-form instruction to
the programmer. If you want it to do something, that is a constraint, and a
constraint is a node — which is what keeps the tree above a complete record of
this search rather than half of one.

The node and the solution are chosen separately and need not match. A node you
have just created holds no solutions at all, and even one that does may not hold
the best code to build on.

Scoring is noisy: the same code scores differently run to run. So a single
attempt tells you very little about whether a constraint helped, and repeating a
node is a real move — it turns a point into a spread you can compare against its
parent. It is also the move that quietly spends a run. Repeat a node when you can
say what the extra sample would settle. Otherwise you are buying a number you
already have.

The evaluator records **metrics** beside the score — the `m_*` columns in
`solutions.csv`, and in every solution's `metadata.json`. Nothing scores them and
no constraint has to mention them, but they are the only view you have inside the
score. A metric that moves with it is a lever you can constrain toward. A metric
that moves without it tells you what the score does not care about, which is
worth as much."""

IDEAS_OPENING = """The eight that scored best, and eight more on rotation, marked
`↺`. The rotation advances every iteration, so every idea in the tree comes round
eventually — and nothing else does, since the ledger reaches back twenty
iterations and the tree folds cold branches into a count.

Anything here you would not want to rediscover in two hundred iterations belongs
in `{memory}`, which does not rotate."""

METRICS_OPENING = """Every metric the evaluator returns, ranked by how strongly
the score follows it across the candidates that scored. Rank correlation, so a
straight line is not required — only that one rises when the other does.

Nothing here says which causes which, and two metrics that both follow code size
will both follow a score that follows code size. It is a place to look, not a
conclusion."""

PLAYBOOK_OPENING = """The search has stopped paying off, so here is the whole
list. These are about how to search, not about this problem — none of them names
a technique, because that is your job given what this problem has turned out to
reward."""

TASK_ORDER = """Do these in order. The order is the point twice over: a decision
reached before the last result has been read is a decision made from the tree
alone, and the tree looks the same today as it did yesterday — and `validate`
ends your turn the moment the plan is good, so anything you meant to write down
afterwards never gets written."""

REVIEW_OPENING = """**This iteration is a review**, because {reason}. That means
two extra steps below, and they come before the plan. A review is not a different
job; it is the same job done with the notebook open."""

TASK_STEPS_ORDINARY = """1. **Read the last result against what you expected.**
   Say whether the expectation held, failed, or was not settled by one noisy
   sample — and if it failed, whether the *idea* failed or only this attempt at
   it. Those are different, and confusing them is how a good idea gets abandoned
   and a bad one gets four more attempts.

2. **Decide, and write the plan.**

3. **Call `validate`.** It says what is wrong, or ends your turn if there is
   nothing wrong.

You may write `{memory}` in your working directory on any iteration, not only a
review, and it is worth doing the moment you learn something. It is copied to the
run when your plan is applied. Write it *before* the plan: `validate` ends your
turn."""

TASK_STEPS = """1. **Read the last result against what you expected.** Say
   whether the expectation held, failed, or was not settled by one noisy sample —
   and if it failed, whether the *idea* failed or only this attempt at it. Those
   are different, and confusing them is how a good idea gets abandoned and a bad
   one gets four more attempts.

2. **Go and look at something** this prompt has not already shown you: the source
   of a candidate whose score surprised you, the trajectory of a programmer whose
   attempt failed, the constraint on a branch you are about to write off.
   Everything above is a summary, and a search that only ever reads summaries
   repeats them.

3. **Write `{memory}` in your working directory.** It is copied to the run when
   your plan is applied, and it is what you will be shown next time. It is a
   model rather than a log: start from the one quoted above and edit the lines
   that are now wrong, rather than appending a correction underneath them. Keep
   it under {limit} characters — the ceiling is what keeps it a model. These
   headings are a suggestion and nothing parses them:

   ```
   ## What this problem rewards
   ## What is settled
   ## Dead ends — and why, so they are not rediscovered
   ## Ideas worth returning to
   ## Open questions
   ```

   Read *Ideas already tried* while you are here. That section rotates, so
   anything in it you would not want to meet again as a fresh thought has to be
   written down under one of the last two headings.

4. **Decide, and write the plan.**

5. **Call `validate`.** It says what is wrong, or ends your turn if there is
   nothing wrong."""

PLAN_INSTRUCTIONS = """Write `{plan}` in your working directory:

```json
{{
  "verdict": "…",
  "reasoning": "…",
  "expectation": "…",
  "parent_node_id": "n_…",
  "parent_solution_id": "s_…",
  "constraint": "…"
}}
```

- `verdict` — step 1, in a sentence or two. Leave it out only on the first
  iteration of a run, when there is nothing to reconcile.
- `reasoning` — why this move and not the obvious one, and what it is testing.
- `expectation` — what you expect the score to do, and what result would tell you
  the idea is wrong. A prediction no result could contradict is not a prediction.
  You will be shown this one next iteration with the number beside it.
- `parent_node_id` — the node to work under. Copy an id from the tree above.
- `parent_solution_id` — the solution whose code the programmer starts from. Its
  files are copied into the programmer's working directory before it begins, so
  it opens on that code rather than an empty directory.
- `constraint` — optional. Leave it out to try `parent_node_id` again. Give one
  to create a child of `parent_node_id` and work there instead. This is the only
  way a node is ever created.

The first three are prose and they are required, because all three are read by
whoever runs next. That is you, with none of your context."""

LAYOUT = """You are in this iteration's own directory, and you write two files in
it: `{plan}`, and `{memory}` when you have something to say. Read anything else
in the run.

- `{plan}` — what you decided this iteration, and why.
- `{memory}` — your notebook. Copied to `../../{memory}` once the plan is good.
- `../../{memory}` — what you believed going in. Quoted above.
- `../../arcs.json` — the tree, as (parent, child, constraint) triplets.
- `../../solutions.csv` — every candidate, best score first.
- `../../solutions/<id>/code/` — a candidate's source.
- `../../solutions/<id>/metadata.json` — its score, metrics and lineage.
- `../<earlier>/{programmer_log}` — what a programmer did and saw.
- `../<earlier>/director.log`, `../<earlier>/{plan}` — what you did then.
- `../<earlier>_crashed_<n>/` — an attempt that was set aside."""
