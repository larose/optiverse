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
marked inside a file.

## How it works

**A solution is a directory.** Any file in it can be added, rewritten or
deleted.

**The evaluator is the objective.** Nothing else measures a candidate, not the
agent that wrote it and not the model behind it.

**The search is driven by an agent too.** A strategist with a shell and the run
directory reads whatever it needs — a candidate's source, an agent's trajectory,
the raw journal — and decides what to try next.

Each iteration:

1. The strategist picks a **branch** to work in and writes the brief. A branch is
   a list of *constraints*: prose telling the coding agent how to narrow its
   approach. The list *is* the branch, so going deeper means adding a constraint
   and changing your mind means dropping back to a subset.
2. A coding agent writes a new candidate under those constraints, with the chosen
   parents and their scores available to read. It checks itself with `validate`,
   and its turn ends the moment that passes on code it changed.
3. Optiverse scores the result and appends the iteration to the journal — the
   plan and its outcome on one line.

The strategist sees every score; the coding agent sees none. Ranking is the
strategist's job, and an agent that could see its own score would abandon a novel
approach the moment it looked worse than the incumbent.

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
  solutions.csv                     the population, best score first
  solutions/<solution id>/
    code/                           the solution itself
    references/<parent id>/         the copy of each parent the agent was given
    agent.log                       the agent's full trajectory
    metadata.json                   id, score, metrics, tags, timing
  strategist/
    notebook.md                     the state of the search, readable
    journal.jsonl                   one line per iteration, plan and outcome
    knowledge.md                    what the strategist has learned holds
    plan.json                       this iteration's decision
    logs/00001.log                  the strategist's full trajectory
```

Start with `strategist/notebook.md`. It is the branch tree with attempts and
scores, and it is exactly what the strategist reads — one artifact for both, so a
search that is unreadable to you was unreadable to it.

In `solutions.csv`, every metric an evaluator returns becomes an `m_*` column and
every tag a generator or strategist sets becomes a `t_*` column, so cost and
problem-specific measurements plot without extra tooling. A candidate the
evaluator could not score reads `failed` and sorts to the bottom. Which branch a
candidate belongs to and which parents it had are in `journal.jsonl` instead —
constraints are paragraphs, which do not belong in a CSV cell — joined back by
`solution_id`.

To resume, point a run at a directory it already wrote. There is no checkpoint
file: the journal has one line per finished iteration, so its length is where the
run picks up, and an iteration that died partway through is simply re-run.

```bash
DIRECTORY=tmp/20260730_133833 make run.tsp
```

No evaluator log is kept, since re-running one beats a stale copy:
`python examples/tsp/harness/evaluate.py score tmp/<run>/solutions/<id>/code`. A
directory with no `metadata.json` is an iteration that died partway through; it is
ignored and left for you to inspect.

## Defining your own problem

A problem is a seed codebase, a description, and an evaluator command:

```python
import optiverse
from optiverse.generators.agent import AgentGenerator
from optiverse.strategists.agent import AgentStrategist

optiverse.optimizer.Optimizer(
    optiverse.config.OptimizerConfig(
        directory=Path("tmp/run"),
        generator=AgentGenerator.from_env(),
        max_iterations=100,
        problem=optiverse.config.Problem(
            description=Path("problem.md").read_text(),
            initial_codebase=Path("initial"),
            evaluate_command=["./evaluate"],
        ),
        strategist=AgentStrategist.from_env(),
    )
).run()
```

`Generator` and `Strategist` are both interfaces, so either agent can be
replaced. The strategist reads `OPTIVERSE_STRATEGIST_MODEL`, falling back to
`OPTIVERSE_MODEL`, so planning and coding can use different models without it
being two variables until you care.

The strategist is also given a **playbook**: angles for inventing a constraint
the search has not tried, such as borrowing from another domain or inverting an
assumption every branch shares. It is always in the prompt, and when the search
stops improving one entry is named as a directive — least recently used, so the
search is not shoved the same direction twice. It ships as `optiverse/playbook.md`
and `OptimizerConfig.playbook` points somewhere else if you want your own.

### The evaluator contract

An evaluator is any executable that accepts two subcommands:

```
<command> validate <codebase_dir>   exit 0 = valid, non-zero = invalid
<command> score    <codebase_dir>   stdout: {"score": <float|null>, "metrics": {...}}
```

- **`validate`** answers with its exit code alone. Print whatever diagnostics
  help on either stream; the agent reads all of it. Because there is no payload,
  there is no score to leak. Keep it cheap: the agent runs it repeatedly, and it
  only has to answer "does this work".
- **`score`** prints JSON on stdout and may log freely on stderr. **Lower scores
  are better.** `"score": null` means the candidate cannot be scored. A non-zero
  exit means the *evaluator itself* broke, which Optiverse reports loudly rather
  than counting as another bad candidate.

If your evaluator is Python, `optiverse.evaluator_main` handles the plumbing:

```python
from optiverse.evaluator_main import run

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
make test      # formatting, types and the end-to-end test
make format    # black over the Python, gofmt over the Go
```

The loop, the store and the two contracts import nothing outside the standard
library. The one dependency,
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), belongs to the
agent generator.

## License

Optiverse is free software under the GNU General Public License v3.0. See
[LICENSE](LICENSE).
