# Optiverse

Optiverse is an optimizer, and its search space is code. You write an evaluator
that scores a codebase (objective function), give it something that works to
start from (initial solution), and it searches for a codebase that scores better
(search strategy).

For example, the [traveling salesman problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem)
is to find the shortest tour that visits every city once. Your evaluator scores
a solver by the average length of the tours it produces, and you start from a
solver that returns a random tour.

A candidate is a directory of source files, in any language. Your evaluator is
any executable.

## Where this comes from

DeepMind's [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
established the premise: an LLM inside an evolutionary loop reaches algorithms
that neither the model alone nor the search alone finds. In Optiverse, a coding
agent with a shell writes each candidate, rather than a model editing regions
marked inside a file. That agent is the **programmer**; the one that decides what
it works on is the **director**.

## How it works

**A solution is a directory.** Any file in it can be added, rewritten or
deleted.

**The evaluator is the objective.** Nothing else measures a candidate, not the
programmer that wrote it and not the model behind it.

**The search is driven by an agent too.** A **director** with a shell and the run
directory reads whatever it needs — a candidate's source, a programmer's
trajectory, the raw graph — and decides what to try next.

**The search space is a tree of ideas.** A *constraint* is prose telling the
programmer how to narrow its approach, and it sits on an **arc**. A **node** is
everything accumulated from the root down to it, so a node is an idea and going
deeper is committing to one more thing. The root has no constraints at all.

Each iteration:

1. The director names a node to work under, names a solution whose code the
   programmer starts from, and may add one constraint — which creates a child of
   that node and works there instead. **That is the whole of what it can say to
   the programmer.** There is no free-form brief: if it wants the programmer to
   do something, that is a constraint, and a constraint is a node. Beside that it
   writes three pieces of prose for itself — see below.
2. The chosen solution's code is copied into the programmer's working directory,
   and it changes it. It has three tools — `bash`, `validate` and `give_up` — and
   its turn ends when `validate` reports valid on something it changed.
3. Optiverse scores the result and files it under the node it belongs to.

The node and the solution are chosen separately and need not match. A node the
director has just created holds no solutions at all, and even one that does may
not hold the best code to build on — which is how a tree of ideas still lets two
lines of work combine.

The director sees every score; the programmer sees none — not its own, not its
parent's. Ranking is the director's job, and a programmer that could see a score
would abandon a novel approach the moment it looked worse than the incumbent.

**The director predicts before it measures.** Every plan carries a `reasoning`
and an `expectation` alongside the decision, and the next iteration opens with
that expectation and the score it got, and a `verdict` field to answer it in. A
director shown only a tree reads the same tree the same way every time, which is
what a loop is; one shown what it said last time and what happened has something
to disagree with. Scoring is noisy, so the prediction has to be written before
the number arrives for it to be worth anything.

**The director keeps a notebook.** `memory.md` is its model of the problem — what
this problem rewards, what is settled, what is a dead end and why. It is
rewritten rather than appended, it is the director's alone, and it never reaches
the programmer, because beliefs about what scores well cannot go to something
deliberately kept ignorant of scores. Every so often — on a schedule the director
does not set, and whenever the run stalls — an iteration becomes a **review**: it
has to go and read something the prompt did not show it, and write the notebook
back, before it may plan.

**The deep past is put back in front of it.** A run of eight hundred iterations
has a tree too large to print, so the prompt shows the eight ideas that scored
best and eight more on a rotation that advances every iteration. Every constraint
ever written comes round on a fixed cycle. That is the raw exposure; deciding
which of it matters is what the notebook is for.

Nothing the programmer works out survives its turn. What carries between
iterations is the tree, the scores, the code, and the two things the director
wrote for itself.

## Quick start

Requires Python 3.10 or newer.

```bash
pip install optiverse
```

To run the bundled examples, work from a checkout:

```bash
git clone https://github.com/larose/optiverse
cd optiverse
make init
source venv/bin/activate
```

### Choosing a model

Model access goes through [LiteLLM](https://github.com/BerriAI/litellm), so set
`OPTIVERSE_MODEL` to a LiteLLM model name such as `gemini/gemini-3.6-flash`,
`anthropic/claude-sonnet-5` or `ollama/qwen3`.

The model must support tool calling. Every action either agent can take is a
tool, so a model without it cannot drive either one.

`OPTIVERSE_PROGRAMMER_MODEL` and `OPTIVERSE_DIRECTOR_MODEL` override it for one
agent each, and both fall back to `OPTIVERSE_MODEL`, so it stays one variable
until you want to spend differently on writing code than on deciding what to
write.

Credentials are your provider's own environment variables, set as that provider
documents them (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_API_BASE`).

### Running the TSP example

```bash
GEMINI_API_KEY="your-gemini-api-key" OPTIVERSE_MODEL="gemini/gemini-3.6-flash" make run.tsp
```

```
INFO - Initial solution saved with ID: d174dcc51ad6417185c641bb155d774f, score: 33974.57856961731
INFO - Starting iteration 1/100
INFO - Saved solution cacfb3eb073849a5a6cc8fba90766a68, score: 2840.0775730311016
INFO - Starting iteration 2/100
INFO - Saved solution a037326a1b3740098ea928789675a80d, score: 2678.5583501149986
```

## What comes out of a run

Each run writes to `tmp/YYYYMMDD_HHMMSS`, named for when it started:

```
tmp/20260730_133833/
  arcs.json                         the tree: (parent, child, constraint) triplets
  memory.md                         what the director currently believes
  solutions.csv                     the population, best score first
  solutions/s_<id>/
    code/                           the solution itself
    metadata.json                   score, metrics, tags, timing, and its lineage
  iterations/00001/
    director-prompt.md              what the director was shown
    director.log                    what it did
    plan.json                       what it decided, why, and what it expected
    memory.md                       its notebook as of this iteration, if it wrote one
    programmer-prompt.md            what the programmer was shown
    programmer.log                  what it did
  iterations/00002_crashed_1/       an attempt that died, set aside intact
```

`iterations/` holds the process, `solutions/` holds the product. Every file is
either a primary fact or a verbatim artifact — nothing is derived and saved, so
there is nothing that can drift out of step with what actually happened. That
includes the run's history: what the search *did*, as opposed to what its tree
looks like, is the join of each `plan.json` against the solution its iteration
committed, on the iteration number both already carry.

Start with the newest `director-prompt.md`. It carries the tree with attempts and
scores, and it is exactly what the director read — so a search that is
unreadable to you was unreadable to it. To read one without spending a run:

```bash
python3 -m optiverse.search.preview tmp/20260730_133833 [iteration]
```

`git diff` across `iterations/*/memory.md` is the other thing worth reading. It
is the director's model of the problem changing, and a notebook that only ever
grows is a director that is not actually thinking.

In `solutions.csv`, every metric an evaluator returns becomes an `m_*` column and
every tag a programmer or director sets becomes a `t_*` column, so problem-specific
measurements plot without extra tooling. A candidate the evaluator could not score
reads `failed` and sorts to the bottom. Lineage is three columns — `iteration`,
`node_id` and `parent_solution_id` — which join back to `arcs.json` and to the
other solutions; the constraint paragraphs stay in `arcs.json`, where they do not
have to fit in a CSV cell.

To resume, point a run at a directory it already wrote. There is no checkpoint
file: every iteration produces exactly one solution and `metadata.json` is
written last, so the highest `iteration` among committed solutions is where the
run picks up. An iteration that produced no solution did not happen: whatever it
left behind is renamed `iterations/NNNNN_crashed_1` — beside the original, so it
is hard to miss — and the same number is attempted again. There is deliberately
no fallback plan. Substituting one gave a byte-identical brief every time the
director was down, so a broken run kept paying for candidates that re-derived
what it already had; now it retries in plain sight instead.

```bash
DIRECTORY=tmp/20260730_133833 make run.tsp
```

No evaluator log is kept, since re-running one beats a stale copy:
`python examples/tsp/harness/evaluate.py score tmp/<run>/solutions/<id>/code`. A
solution directory with no `metadata.json` belongs to an attempt that died after
the codebase was allocated; it is ignored and left for you to inspect.

## Defining your own problem

A problem is a seed codebase, a description, and an evaluator command:

```python
import optiverse
from optiverse.programmer.agent import AgentProgrammer
from optiverse.director.agent import AgentDirector

optiverse.optimizer.Optimizer(
    optiverse.config.OptimizerConfig(
        directory=Path("tmp/run"),
        director=AgentDirector.from_env(),
        programmer=AgentProgrammer.from_env(),
        max_iterations=100,
        problem=optiverse.config.Problem(
            description=Path("problem.md").read_text(),
            initial_codebase=Path("initial"),
            evaluate_command=["./evaluate"],
        ),
    )
).run()
```

`Programmer` and `Director` are both interfaces, so either agent can be replaced.
Each reads its own model variable — `OPTIVERSE_PROGRAMMER_MODEL` and
`OPTIVERSE_DIRECTOR_MODEL` — falling back to `OPTIVERSE_MODEL`, so planning and
coding can use different models without it being two variables until you care.

The director is also given a **playbook**: angles for inventing a constraint the
search has not tried, such as borrowing from another domain or inverting an
assumption every node shares. It appears only once the run has stopped paying
off, since that is the moment a new constraint is the way out and any earlier it
is noise — and then it appears whole. Naming one angle at random cost the run its
reproducibility and, with nothing recording what had been shown, could name the
same one for the rest of a thousand iterations. It ships as
`optiverse/search/playbook.md`, and `OptimizerConfig.playbook` points somewhere
else if you want your own.

### The evaluator contract

An evaluator is any executable that accepts two subcommands:

```
<command> validate <codebase_dir>   exit 0 = valid, non-zero = invalid
<command> score    <codebase_dir>   stdout: {"score": <float|null>, "metrics": {...}}
```

- **`validate`** answers with its exit code alone. Print whatever diagnostics
  help on either stream; the programmer reads all of it. Because there is no
  payload, there is no score to leak. Keep it cheap: it is run repeatedly, and it
  only has to answer "does this work".
- **`score`** prints JSON on stdout and may log freely on stderr. **Lower scores
  are better.** `"score": null` means the candidate cannot be scored. A non-zero
  exit means the *evaluator itself* broke, which Optiverse reports loudly rather
  than counting as another bad candidate.

If your evaluator is Python, `optiverse.evaluator.main` handles the plumbing:

```python
from optiverse.evaluator.main import run

if __name__ == "__main__":
    run(score=score, validate=validate)
```

## Examples

**[Traveling Salesman Problem](examples/tsp/README.md).** About 300 iterations
produced an Iterated Local Search heuristic with 2-opt and four perturbation
operators, averaging a tour length of 2593 on a 280-city instance, within about
0.5% of the known optimum.

**[Integer compression](examples/integer_compression/README.md).** About 1000
iterations produced a Go implementation of block-based delta encoding with
binary packing, reaching a compression ratio of 230 at decompression speeds in
the range of established C implementations.

Both were run in 2025 with Qwen3-235B-A22B, on one machine. For the design as it
stood at the start of the project, see the 2025 announcement post,
[Optiverse: Evolving Code with LLMs](https://mathieularose.com/optiverse-evolving-code-with-llms).

## Development

```bash
make init      # virtualenv and dependencies
make test      # formatting and types
make format    # black over the Python, gofmt over the Go
```

One concept per folder, and each folder's `__init__.py` holds the thing it is
named after:

```
optiverse/
  optimizer.py       the loop
  config.py          what you hand it
  search/            the tree of ideas, the run's history, and the director's prompt
  director/          the planning seam, and the agent that fills it
  programmer/        the writing seam, the agent that fills it, and its prompt
  solution/          what a candidate is and where it is kept
  evaluator/         the process contract, and a helper for Python evaluators
  mini_swe_agent/    the shell, the tools and the model layer both agents share
```

The loop, the store and the two contracts import nothing outside the standard
library. The one dependency,
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), is reached only
through `optiverse/mini_swe_agent/`, which is what the two agent implementations
are built on.

## License

Optiverse is free software under the GNU General Public License v3.0. See
[LICENSE](LICENSE).
