# Optiverse

Optiverse searches for better code. You give it a seed codebase, a description
of the problem, and a command that measures a candidate. It evolves a population
of solutions, each one written by a coding agent and ranked by your evaluator.

The unit of evolution is a **directory**, so a candidate can restructure a whole
package rather than fill in a marked region. The measurement is a **process**,
so the evolved code can be in any language, and what counts as better is decided
by a program you wrote.

## Where this comes from

DeepMind's [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
established the premise: an LLM inside an evolutionary loop reaches algorithms
that neither the model alone nor the search alone finds. Optiverse takes that
premise and gives the writing of each candidate to a coding agent with a shell,
and the judging of it to an ordinary program of your own.

## What the design commits to

**A solution is a directory.** The thing being improved is a package, so nothing
has to be marked as evolvable and any file in it can be added, rewritten or
deleted.

**The evaluator is the spec.** It is a command, in any language, and the only
thing that ever measures anything. This is where the real difficulty of the
method lives.

**Everything is a plain file.** Solutions, lineage, agent trajectories and the
checkpoint all live under one run directory, so a run is resumable and
inspectable with the tools you already have.

## How an iteration works

1. The search strategy picks parent solutions and decides whether to exploit or
   diversify.
2. An agent writes a new candidate, with the parents and their scores available
   to read. It checks itself with `validate`, and its turn ends the moment that
   passes on code it changed.
3. Optiverse scores the result and records it with its metrics and lineage.

The agent is told whether its code is **valid**, never how it **scores**. It can
read its parents' scores but has no way to measure its own work, because an
agent that could rank itself would abandon a novel approach as soon as it looked
worse than the incumbent, which is the very move that escapes a local optimum.

## Quick start

Requirements: Python 3.10 or newer. The integer compression example also needs a
Go toolchain and downloads about 1.6 GB of benchmark data; TSP needs neither.

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
`OPTIVERSE_MODEL` to a [LiteLLM model name](https://docs.litellm.ai/docs/providers)
such as `gemini/gemini-3.6-flash`, `anthropic/claude-sonnet-5` or `ollama/qwen3`.

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

Iterations are bounded by a step limit and a wall-clock limit, but **not** by
cost, so check what the first few cost before leaving a long run unattended.

## What comes out of a run

Each run writes to `tmp/YYYYMMDD_HHMMSS`, named for when it started:

```
tmp/20260730_133833/
  solutions.csv                     the population, best score first
  checkpoint.json                   where to resume from
  <solution id>/
    code/                           the solution itself
    references/<parent id>/         the copy of each parent the agent was given
    agent.log                       the agent's full trajectory
    metadata.json                   id, score, metrics, tags
```

In `solutions.csv`, every metric an evaluator returns becomes an `m_*` column
and every tag a strategy or generator sets becomes a `t_*` column, so cost,
lineage and problem-specific measurements plot without extra tooling. A
candidate the evaluator could not score reads `FAILED` and sorts to the bottom.

To resume, point a run at a directory it already wrote. It continues from the
iteration after the last one that finished, with the population it had found:

```bash
DIRECTORY=tmp/20260730_133833 make run.tsp
```

No evaluator log is kept, since re-running one beats a stale copy:
`python examples/tsp/harness/evaluate.py score tmp/<run>/<id>/code`. A directory
with no `metadata.json` is an iteration that died partway through; it is ignored
and left for you to inspect.

## Defining your own problem

A problem is a seed codebase, a description, and an evaluator command:

```python
import optiverse
from optiverse.generators.agent import AgentGenerator

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
        search_strategy=optiverse.search_strategies.IteratedLocalSearch(
            max_iterations_without_improvements=10
        ),
    )
).run()
```

That strategy improves the best solution it has until ten iterations pass
without progress, then perturbs. Strategy and generator are interfaces, so
either can be replaced.

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

### Making an evaluator hard to cheat

Put the candidate in a **subdirectory** of a temporary workspace, with your
harness at the root, so it cannot stand in for part of your harness whatever it
names its files. Blocklisting names only covers the ones you thought of.

Then be deliberate about what the candidate's own process is trusted to report.
If the objective is something you can recompute, such as a tour length or a
compressed size, take the artifact and measure it yourself. The
[integer compression example](examples/integer_compression/README.md) works
through a case where that is not possible and documents which holes it left
open.

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
