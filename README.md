# Optiverse

Optiverse is a Python library for evolving code and algorithms using coding agents. Inspired by Deepmind's [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), it provides a flexible framework to iteratively improve whole codebases, in any programming language.

With Optiverse, you define a problem and provide an evaluator. The system then generates and evolves candidate solutions over multiple iterations, learning which approaches yield better results.

Each candidate is produced by an **agent working in a real directory**, not by a single model call. The agent can read files, compile, run tests and fix its own mistakes before handing the solution back — so an iteration rarely ends in code that does not even build.

📖 **Read the announcement post:** [Optiverse: Evolving Code with LLMs](https://mathieularose.com/optiverse-evolving-code-with-llms)

## Table of Contents


- [Why Optiverse?](#why-optiverse)
- [Use Cases](#use-cases)
- [Quick Start](#quick-start)
- [License](#license)

## Why Optiverse?

Optiverse helps developers and researchers automate code improvement by generating, refining, and optimizing entire programs. Its design enables broad experimentation and fast iteration across diverse problem domains. Key capabilities include:

- **Whole-codebase optimization**: Unlike other implementations that operate on isolated functions or code blocks, a solution in Optiverse is a directory. An agent may add, rename and delete files, so it can restructure a package rather than rewriting one file.
- **Agents, not one-shot answers**: candidates are produced by [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) working in a real directory, with a command it can run to check its own work.
- **Modular architecture**: Swap or customize search strategies and generators to experiment with different approaches.
- **Multi-language support**: An evaluator is a command, not a Python class, so it can be a shell script, a Go binary or a Makefile — and the code under test can be in any language.
- **Flexible LLM integration**: model access goes through [LiteLLM](https://github.com/BerriAI/litellm), which talks to ~150 providers in their own dialect — hosted, self-hosted or local — so switching models is one environment variable.
- **A dependency-free core**: the search loop, the store and the two process contracts are pure standard library. Only the agent generator needs an extra.

### How an iteration works

1. The search strategy picks parent solutions and decides whether to exploit or diversify.
2. The best parent's directory is copied into a fresh solution directory.
3. An agent edits that directory. It can run `<evaluate> validate <dir>` as often as it likes; the moment that passes on changed code, its turn ends.
4. Optiverse runs `<evaluate> score <dir>` and records the score, metrics and lineage.

The agent is told whether its own code is **valid**, never how it **scores**. It sees the parent solutions' scores as context, but it has no way to measure its own work — and that is the point. Ranking candidates is the search loop's job; an agent that could score itself would abandon a novel approach as soon as it looked worse than the incumbent, which is the very move that escapes local optima.

## Use Cases

These examples showcase Optiverse's ability to generate and refine code for a wide range of programming tasks, regardless of domain or language.

### Traveling Salesman Problem

The TSP example evolved an advanced Iterated Local Search algorithm with 2-opt improvements, achieving near-optimal results on benchmark instances. The evolved solution includes sophisticated perturbation operators and performance optimizations.

[View detailed TSP results →](examples/tsp/README.md)

### Integer Compression

The integer compression example evolved a Go implementation with performance comparable to established C implementations. The evolved algorithm uses block-based delta encoding with binary packing, achieving competitive decompression speeds.

[View detailed integer compression results →](examples/integer_compression/README.md)

## Quick Start

Follow these steps to set up Optiverse and run an example.

### 1. Set Up Environment

First, create a virtual environment and install dependencies:

```bash
make init
```

Then, activate the virtual environment:

```bash
source venv/bin/activate
```

### 2. Run the Traveling Salesman Problem example:

This example uses Optiverse to solve the [Traveling Salesman Problem (TSP)](https://en.wikipedia.org/wiki/Travelling_salesman_problem). The code is in the [examples/tsp](examples/tsp) directory.

Model selection goes through LiteLLM, which speaks each provider's own API — Anthropic, Bedrock and Vertex are as native as OpenAI, and local servers work too. Set one variable:

- `OPTIVERSE_MODEL`: a [LiteLLM model name](https://docs.litellm.ai/docs/providers), such as `gemini/gemini-3.6-flash`, `anthropic/claude-sonnet-5` or `ollama/qwen3`

Credentials are your provider's own environment variables, set exactly as that provider's documentation says — `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_API_BASE` for a local server. LiteLLM reads them directly; Optiverse never handles your key.

```bash
GEMINI_API_KEY="your-gemini-api-key" OPTIVERSE_MODEL="gemini/gemini-3.6-flash" make run.tsp
```

Each iteration is bounded by a step limit and a wall-clock limit, but **not** by cost. Spend is recorded per candidate as `m_agent_cost_usd` in `solutions.csv`; check the first few rows before leaving a long run unattended.

### Sample Output:

When you run it, you'll see output like this:

```bash
2026-07-30 10:47:22 - optiverse.optimizer - INFO - Starting fresh optimization...
2026-07-30 10:47:22 - optiverse.optimizer - INFO - Evaluating and saving initial solution...
2026-07-30 10:47:22 - optiverse.optimizer - INFO - Initial solution saved with ID: 2787c4d511664076952e530a8e9a20fc, score: 34271.8174318594
2026-07-30 10:47:24 - optiverse.optimizer - INFO - Starting iteration 1/100
2026-07-30 10:48:41 - optiverse.optimizer - INFO - Saved solution 9f1c0a3d5b784e2a8c6f0d21b4e37a58, score: 3146.3732212611126
2026-07-30 10:48:41 - optiverse.optimizer - INFO - Starting iteration 2/100

...

2026-07-30 12:31:09 - optiverse.optimizer - INFO - ==================================================
2026-07-30 12:31:09 - optiverse.optimizer - INFO - BEST SOLUTION:
2026-07-30 12:31:09 - optiverse.optimizer - INFO - ==================================================
2026-07-30 12:31:09 - optiverse.optimizer - INFO - ID: 2787c4d511664076952e530a8e9a20fc
2026-07-30 12:31:09 - optiverse.optimizer - INFO - Score: 2593.108
2026-07-30 12:31:09 - optiverse.optimizer - INFO - Codebase: tmp/20260730_104722/2787c4d511664076952e530a8e9a20fc/code
2026-07-30 12:31:09 - optiverse.optimizer - INFO - Files:
  solver.py (6134 bytes)
```

### Understanding the Results

During optimization, Optiverse saves results in directories named `tmp/YYYYMMDD_HHMM`, indicating the date and time of each run.

#### `solutions.csv`


Inside each run directory, `solutions.csv` provides a high-level overview of all solutions explored:


| id                               | score           | t_group | t_move          | t_exit_status | t_parent_id_1                    | m_agent_cost_usd | ... |
|----------------------------------|-----------------|---------|-----------------|---------------|----------------------------------|------------------|-----|
| 0d8c5789dde24a94901871c18d6d9854 | 2817.0555492637 | 1       | local_search    | Validated     | 034afd2da8894ab6838efb17bc28f201 | 0.031            | ... |
| a42574709f27484fade7adc76683b2cf | 2828.6215589938 | 6       | perturb_explore | Validated     | 96c7140054154a20a4b67fd986658dd4 | 0.048            | ... |
| 883f6de275c14b32822feb5cdaaba55d | 2837.3500311898 | 6       | local_search    | LimitsExceeded| a42574709f27484fade7adc76683b2cf | 0.100            | ... |

Any metric an evaluator returns becomes an `m_*` column, and any tag a strategy
or generator sets becomes a `t_*` column, so `t_exit_status` and
`m_agent_cost_usd` need no extra tooling to plot.

#### Individual Solution Directories

Each solution has a dedicated directory named after its ID, containing:

- `code/`: The solution itself — a directory, with however many files the agent chose to write.
- `agent.log`: The agent's full trajectory, including every command it ran.
- `metadata.json`: ID, score, metrics and tags.

There is deliberately no stored evaluator log. It is reproducible from the files
that are kept, and re-running gives fuller output than a stale copy:

```bash
python examples/tsp/evaluate.py score tmp/<run>/<id>/code
```

A directory with no `metadata.json` is an iteration that died partway through. It
is ignored by later iterations and left in place for you to inspect.

## Defining Your Own Problem

A problem is a seed codebase, a description, and an evaluator command:

```python
problem = optiverse.config.Problem(
    description=Path("problem.md").read_text(),
    initial_codebase=Path("initial"),
    evaluate_command=["./evaluate"],
)
```

### The evaluator contract

An evaluator is any executable that accepts two subcommands:

```
<command> validate <codebase_dir>   exit 0 = valid, non-zero = invalid
<command> score    <codebase_dir>   stdout: {"score": <float|null>, "metrics": {...}}
```

- **`validate`** answers with its exit code alone. Print whatever diagnostics
  help — compiler errors, failing assertions — on either stream; the agent reads
  all of it. Because there is no payload, there is no score to leak.
- **`score`** prints JSON on stdout and may log freely on stderr. **Lower scores
  are better.** `"score": null` means the run completed but the candidate cannot
  be scored; a non-zero exit means the *evaluator itself* broke, which Optiverse
  reports loudly rather than counting as another bad candidate.

Keep `validate` cheap. The agent runs it repeatedly, and it only has to answer
"does this work", not "how good is it" — the integer compression example checks
`Decompress(Compress(data)) == data` on a few thousand synthetic integers in
milliseconds, while scoring streams a 1.6 GB dataset.

Your evaluator owns its own test harness. Copy the candidate's files into a
temporary directory and lay your harness over the top: that way a candidate
cannot alter how it is measured, and the stored solution is never touched.

If your evaluator is Python, `optiverse.evaluator_main` handles the argv and JSON
plumbing:

```python
from optiverse.evaluator_main import run

def validate(codebase: Path) -> bool: ...
def score(codebase: Path) -> tuple[float | None, dict[str, float]]: ...

if __name__ == "__main__":
    run(score=score, validate=validate)
```

## License

Optiverse is open source and licensed under the GNU General Public License v3.0 (GPLv3).
