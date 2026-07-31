Implement a heuristic solver for the Traveling Salesman Problem (TSP) in Python. Your goal is to find the shortest possible tour within the given time limit.

## Your codebase

Your codebase is a directory. It must contain `solver.py` at its root, defining:

```python
def solve(context: Context) -> None:
    ...
```

You may add as many further Python modules alongside it as you like and import them from `solver.py`. Define every function at the top level; do not nest functions.

Nothing else is in your codebase. `context.py` and the code that runs your solver belong to the evaluator and are not yours to see or change. The `Context` interface below is the whole of what you get.

## How your solver is run

Something imports `solve` from your `solver.py`, builds a `Context` around one problem instance and a time limit, and calls `solve` exactly once. Your answer is the last tour you reported before the limit expired; tours reported after it are ignored. Start `solver.py` with:

```python
from context import Context
```

You cannot run your solver yourself. The `validate` tool is the only way to exercise it: it answers valid or invalid and prints diagnostics. It never tells you a tour length, a running time, or a score.

## Requirements

Submit only valid tours that visit each city exactly once and form a closed loop.

Call `context.report_new_best_solution(solution)` only when a better solution is found, to minimise overhead.

Ensure that solutions are reported before the time limit expires. A solver that keeps working past the limit is killed and scores nothing.

Use only the Python Standard Library; external packages are not allowed.

You are scored on instances you have never seen, so do not tune for one particular instance, and do not try to carry state from one run to the next.

## `Context` Interface

```python
class Context:
    @property
    def instance(self) -> List[Tuple[float, float]]:
        """
        Returns a list of (x, y) coordinates for each city.
        The index of a city in this list is its unique integer ID.
        """
        ...

    def remaining_time(self) -> timedelta:
        """
        Returns the time remaining for computation.
        """
        ...

    def report_new_best_solution(self, solution: List[int]) -> None:
        """
        Reports a new best solution found. The 'solution' is a list of city IDs
        representing the order of cities in the tour. The first and last city
        in the tour are implicitly connected to form a loop.
        """
        ...
```

## Evaluation Criteria

Your score is the average length of the tours you produce, over several runs. Lower is better.
